use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;

use crate::brain::{Brain, ForwardResult, INPUTS, MEMORY_SIZE};
use crate::evolution::Agent;
use crate::signal::{self, Signal, SignalGrid, NUM_SYMBOLS};

// Input layout offsets derived from constants
const SIGNAL_INPUT_START: usize = 9; // after pred(3) + food(3) + ally(3)
const MEMORY_INPUT_START: usize = SIGNAL_INPUT_START + NUM_SYMBOLS * 3;
const ENERGY_INPUT_IDX: usize = MEMORY_INPUT_START + MEMORY_SIZE;
const _: () = assert!(ENERGY_INPUT_IDX + 1 == INPUTS, "input layout size mismatch");

pub const INPUT_NAMES: [&str; INPUTS] = [
    "zone_damage",
    "energy_delta",
    "dead_spare",
    "food_dx",
    "food_dy",
    "food_dist",
    "ally_dx",
    "ally_dy",
    "ally_dist",
    "sig0_str",
    "sig0_dx",
    "sig0_dy",
    "sig1_str",
    "sig1_dx",
    "sig1_dy",
    "sig2_str",
    "sig2_dx",
    "sig2_dy",
    "sig3_str",
    "sig3_dx",
    "sig3_dy",
    "sig4_str",
    "sig4_dx",
    "sig4_dy",
    "sig5_str",
    "sig5_dx",
    "sig5_dy",
    "mem0",
    "mem1",
    "mem2",
    "mem3",
    "mem4",
    "mem5",
    "mem6",
    "mem7",
    "energy",
];

/// Wrap-aware signed delta: shortest path on a toroidal grid.
/// Returns value in (-size/2, size/2].
pub(crate) fn wrap_delta(a: i32, b: i32, size: i32) -> i32 {
    let d = b - a;
    if d > size / 2 {
        d - size
    } else if d < -(size / 2) {
        d + size
    } else {
        d
    }
}

pub(crate) fn wrap_dist_sq(ax: i32, ay: i32, bx: i32, by: i32, grid_size: i32) -> f32 {
    let dx = wrap_delta(ax, bx, grid_size) as f32;
    let dy = wrap_delta(ay, by, grid_size) as f32;
    dx * dx + dy * dy
}

/// Wrap-aware signed delta for f32 coordinates on a toroidal grid.
fn wrap_delta_f32(a: f32, b: f32, size: f32) -> f32 {
    let d = b - a;
    if d > size / 2.0 {
        d - size
    } else if d < -(size / 2.0) {
        d + size
    } else {
        d
    }
}

/// Spatial index: uniform grid for O(1) neighbor lookups on a toroidal grid.
struct CellGrid {
    cells: Vec<Vec<u16>>,
    grid_size: i32,
}

impl CellGrid {
    fn new(grid_size: i32) -> Self {
        let n = (grid_size * grid_size) as usize;
        Self {
            cells: (0..n).map(|_| Vec::new()).collect(),
            grid_size,
        }
    }

    fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }

    fn cell_idx(&self, x: i32, y: i32) -> usize {
        (y * self.grid_size + x) as usize
    }

    fn insert(&mut self, x: i32, y: i32, idx: u16) {
        let ci = (y * self.grid_size + x) as usize;
        self.cells[ci].push(idx);
    }

    fn remove(&mut self, x: i32, y: i32, idx: u16) {
        let ci = (y * self.grid_size + x) as usize;
        let cell = &mut self.cells[ci];
        if let Some(pos) = cell.iter().position(|&v| v == idx) {
            cell.swap_remove(pos);
        }
    }

    /// Expanding Chebyshev ring search with toroidal wrapping.
    /// Finds nearest entity within `max_radius` cells. Returns index and squared distance.
    #[allow(clippy::similar_names)]
    fn nearest(&self, x: i32, y: i32, max_radius: i32, skip_idx: u16) -> Option<(u16, f32)> {
        let gs = self.grid_size;
        let mut best: Option<(u16, f32)> = None;

        for r in 0..=max_radius {
            // Once we find something, no ring further out can beat it
            // (Chebyshev distance r means all cells are at least r away)
            if best.is_some() && r > 0 {
                // Check: could any entity in ring r be closer than current best?
                // Min possible squared dist at Chebyshev r is r*r (on axis)
                let min_possible = (r * r) as f32;
                if let Some((_, bd)) = best {
                    if min_possible >= bd {
                        break;
                    }
                }
            }

            if r == 0 {
                let ci = self.cell_idx(x, y);
                for &idx in &self.cells[ci] {
                    if idx == skip_idx {
                        continue;
                    }
                    return Some((idx, 0.0));
                }
            } else {
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx.abs().max(dy.abs()) != r {
                            continue;
                        }
                        let cx = (x + dx).rem_euclid(gs);
                        let cy = (y + dy).rem_euclid(gs);
                        let ci = self.cell_idx(cx, cy);
                        for &idx in &self.cells[ci] {
                            if idx == skip_idx {
                                continue;
                            }
                            let edx = wrap_delta(x, cx, gs) as f32;
                            let edy = wrap_delta(y, cy, gs) as f32;
                            let d_sq = edx * edx + edy * edy;
                            match best {
                                None => best = Some((idx, d_sq)),
                                Some((_, bd)) if d_sq < bd => best = Some((idx, d_sq)),
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        best
    }
}

#[derive(Clone, Debug)]
pub struct Prey {
    pub x: i32,
    pub y: i32,
    pub energy: f32,
    pub alive: bool,
    pub brain: Brain,
    pub ticks_alive: u32,
    pub food_eaten: u32,
    pub memory: [f32; MEMORY_SIZE],
    /// Per-prey action counts when receiving any signal, by context.
    pub actions_with_signal: [[u32; 5]; 2],
    /// Per-prey action counts when not receiving any signal, by context.
    pub actions_without_signal: [[u32; 5]; 2],
    /// Actions at the tick a signal disappears (onset of silence), by context.
    pub silence_onset_actions: [[u32; 5]; 2],
    /// Whether this prey received a signal on the previous tick (for onset detection).
    pub had_signal_prev_tick: bool,
    /// Accumulated zone damage. Separate from energy so food cannot offset it.
    /// Death occurs at `zone_damage` >= 1.0. Ticks to die depends on `zone_drain_rate`
    /// and distance from zone center (gradient damage).
    pub zone_damage: f32,
    /// Energy at the start of the previous tick. Used to compute `energy_delta` input.
    pub prev_energy: f32,
}

#[derive(Clone, Debug)]
pub struct KillZone {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub speed: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Food {
    pub x: i32,
    pub y: i32,
    pub is_patch: bool,
}

pub struct SignalEvent {
    pub symbol: u8,
    pub zone_dist: f32,
    pub inputs: [f32; INPUTS],
    pub emitter_idx: usize,
}

pub struct World {
    pub prey: Vec<Prey>,
    pub zones: Vec<KillZone>,
    pub food: Vec<Food>,
    pub signals: Vec<Signal>,
    pub tick: u32,
    pub signals_emitted: u32,
    pub signal_events: Vec<SignalEvent>,
    pub ticks_in_zone: u32,
    pub total_prey_ticks: u32,
    /// Receiver response spectrum: `[signal_state][context][action]` counts.
    /// `signal_state`: 0=none, `1..=NUM_SYMBOLS` for each symbol (strongest received).
    /// `context`: 0=not in zone, 1=in zone.
    /// `action`: 0-4 (up/down/right/left/eat).
    pub receiver_counts: [[[u32; 5]; 2]; 1 + NUM_SYMBOLS],
    /// Signal count per tick (for silence correlation).
    pub signals_per_tick: Vec<u32>,
    /// Alive prey count per tick (for normalizing silence correlation).
    pub alive_per_tick: Vec<u32>,
    /// Minimum zone-edge distance to alive prey per tick (observer metric).
    pub min_zone_dist_per_tick: Vec<f32>,
    /// When true, signal emission is suppressed (counterfactual mode).
    pub no_signals: bool,
    /// When false, skip metrics bookkeeping (signal events, receiver tracking, observer distances).
    pub collect_metrics: bool,
    /// Count of prey that died from zone damage this evaluation.
    pub zone_deaths: u32,
    // Spatial indices (rebuilt each tick for prey/signals, maintained incrementally for food)
    prey_grid: CellGrid,
    food_grid: CellGrid,
    signal_grid: SignalGrid,
    // Pre-allocated per-tick buffers (reused across ticks to avoid allocation)
    shuffled_indices: Vec<usize>,
    order_scratch: Vec<usize>,
    alive_scratch: Vec<usize>,
    #[allow(clippy::type_complexity)]
    computed_scratch: Vec<(usize, [f32; INPUTS], ForwardResult, f32, usize, Option<u8>)>,
    // Simulation parameters
    pub grid_size: i32,
    pub food_count: usize,
    pub signal_range: f32,
    pub base_drain: f32,
    pub neuron_cost: f32,
    pub signal_cost: f32,
    pub patch_ratio: f32,
    pub zone_drain_rate: f32,
    pub signal_ticks: u32,
}

impl World {
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_positions(
        agents: &[Agent],
        num_zones: usize,
        rng: &mut impl Rng,
        no_signals: bool,
        collect_metrics: bool,
        grid_size: i32,
        food_count: usize,
        signal_range: f32,
        zone_radius: f32,
        zone_speed: f32,
        base_drain: f32,
        neuron_cost: f32,
        signal_cost: f32,
        patch_ratio: f32,
        zone_drain_rate: f32,
        signal_ticks: u32,
    ) -> Self {
        let prey: Vec<Prey> = agents
            .iter()
            .map(|agent| Prey {
                x: agent.x,
                y: agent.y,
                energy: 1.0,
                alive: true,
                brain: agent.brain.clone(),
                ticks_alive: 0,
                food_eaten: 0,
                memory: std::array::from_fn(|_| rng.gen_range(-0.1..0.1)),
                actions_with_signal: [[0; 5]; 2],
                actions_without_signal: [[0; 5]; 2],
                silence_onset_actions: [[0; 5]; 2],
                had_signal_prev_tick: false,
                zone_damage: 0.0,
                prev_energy: 1.0,
            })
            .collect();

        let gs = grid_size as f32;
        let zones = (0..num_zones)
            .map(|_| KillZone {
                x: rng.gen_range(0.0..gs),
                y: rng.gen_range(0.0..gs),
                radius: zone_radius,
                speed: zone_speed,
            })
            .collect();

        let food: Vec<Food> = (0..food_count)
            .map(|_| Food {
                x: rng.gen_range(0..grid_size),
                y: rng.gen_range(0..grid_size),
                is_patch: rng.gen::<f32>() < patch_ratio,
            })
            .collect();

        let mut food_grid = CellGrid::new(grid_size);
        for (i, f) in food.iter().enumerate() {
            food_grid.insert(f.x, f.y, i as u16);
        }

        let prey_count = prey.len();
        Self {
            prey,
            zones,
            food,
            signals: Vec::new(),
            tick: 0,
            signals_emitted: 0,
            signal_events: if collect_metrics {
                Vec::with_capacity(50_000)
            } else {
                Vec::new()
            },
            ticks_in_zone: 0,
            total_prey_ticks: 0,
            receiver_counts: [[[0u32; 5]; 2]; 1 + NUM_SYMBOLS],
            signals_per_tick: Vec::new(),
            alive_per_tick: Vec::new(),
            min_zone_dist_per_tick: Vec::new(),
            no_signals,
            collect_metrics,
            zone_deaths: 0,
            prey_grid: CellGrid::new(grid_size),
            food_grid,
            signal_grid: SignalGrid::new(grid_size, signal_range),
            shuffled_indices: (0..prey_count).collect(),
            order_scratch: Vec::with_capacity(prey_count),
            alive_scratch: Vec::with_capacity(prey_count),
            computed_scratch: Vec::with_capacity(prey_count),
            grid_size,
            food_count,
            signal_range,
            base_drain,
            neuron_cost,
            signal_cost,
            patch_ratio,
            zone_drain_rate,
            signal_ticks,
        }
    }

    /// Compute distance from (x,y) to nearest zone edge. Negative = inside zone.
    fn nearest_zone_edge_dist(&self, x: i32, y: i32) -> f32 {
        let mut best = f32::MAX;
        let gs = self.grid_size as f32;
        for zone in &self.zones {
            let dx = wrap_delta_f32(x as f32, zone.x, gs);
            let dy = wrap_delta_f32(y as f32, zone.y, gs);
            let center_dist = (dx * dx + dy * dy).sqrt();
            let edge_dist = center_dist - zone.radius;
            if edge_dist < best {
                best = edge_dist;
            }
        }
        best
    }

    pub fn any_alive(&self) -> bool {
        self.prey.iter().any(|p| p.alive)
    }

    #[allow(clippy::too_many_lines)]
    pub fn step(&mut self, rng: &mut impl Rng) {
        self.tick += 1;

        let signals_before = self.signals_emitted;

        let sig_ticks = self.signal_ticks;
        self.signals
            .retain(|s| self.tick.saturating_sub(s.tick_emitted) <= sig_ticks);

        // Rebuild spatial grids
        self.signal_grid.rebuild(&self.signals, self.tick);
        self.rebuild_prey_grid();

        // Track minimum zone-edge distance to alive prey this tick (observer metric)
        if self.collect_metrics {
            let mut min_zone_dist = f32::MAX;
            for p in &self.prey {
                if !p.alive {
                    continue;
                }
                let d = self.nearest_zone_edge_dist(p.x, p.y);
                if d < min_zone_dist {
                    min_zone_dist = d;
                }
            }
            self.min_zone_dist_per_tick.push(min_zone_dist);
        }

        // Shuffle prey processing order to prevent index bias
        self.shuffled_indices.clear();
        self.shuffled_indices.extend(0..self.prey.len());
        self.shuffled_indices.shuffle(rng);

        // Copy shuffled order into scratch buffer, then take it out for borrow splitting
        self.order_scratch.clear();
        self.order_scratch.extend_from_slice(&self.shuffled_indices);
        let order = std::mem::take(&mut self.order_scratch);

        // Snapshot energy for energy_delta input (computed next tick)
        for &i in &order {
            if self.prey[i].alive {
                self.prey[i].prev_energy = self.prey[i].energy;
            }
        }

        // Metabolism + alive filter in one pass (saves an O(n) traversal)
        self.alive_scratch.clear();
        for &i in &order {
            if !self.prey[i].alive {
                continue;
            }
            let total_hidden =
                self.prey[i].brain.base_hidden_size + self.prey[i].brain.signal_hidden_size;
            let drain = self.base_drain + total_hidden as f32 * self.neuron_cost;
            self.prey[i].energy -= drain;
            if self.prey[i].energy <= 0.0 {
                self.prey[i].alive = false;
                continue;
            }
            self.alive_scratch.push(i);
        }
        self.order_scratch = order;

        // Parallel compute: build inputs + run brain forward for all alive prey
        let alive = std::mem::take(&mut self.alive_scratch);
        let mut computed = std::mem::take(&mut self.computed_scratch);
        alive
            .par_iter()
            .map(|&i| {
                let zone_dist = self.nearest_zone_edge_dist(self.prey[i].x, self.prey[i].y);
                let inputs = self.build_inputs_fast(i);
                let result = self.prey[i].brain.forward(&inputs);
                // Action argmax (moved from sequential phase)
                let mut action = 0;
                let mut best_val = result.actions[0];
                for (j, &val) in result.actions[1..].iter().enumerate() {
                    if val >= best_val {
                        best_val = val;
                        action = j + 1;
                    }
                }
                // Emit decision: pure softmax + threshold (moved from sequential phase)
                let emit = signal::maybe_emit(&result.signals);
                (i, inputs, result, zone_dist, action, emit)
            })
            .collect_into_vec(&mut computed);
        self.alive_scratch = alive;

        // Sequential apply: mutations to world state
        for &(i, ref inputs, ref result, zone_dist, action, emit) in &computed {
            let in_zone = zone_dist <= 0.0;

            if self.collect_metrics {
                self.total_prey_ticks += 1;
                if in_zone {
                    self.ticks_in_zone += 1;
                }

                // Receiver response spectrum: classify signal state, context, and chosen action
                let mut max_str = 0.0_f32;
                let mut best_sym = 0;
                for s in 0..NUM_SYMBOLS {
                    let str_val = inputs[SIGNAL_INPUT_START + s * 3];
                    if str_val > max_str {
                        max_str = str_val;
                        best_sym = s;
                    }
                }
                let signal_state: usize = if max_str > 0.0 { 1 + best_sym } else { 0 };
                let context = usize::from(in_zone); // 0=not in zone, 1=in zone
                self.receiver_counts[signal_state][context][action] += 1;

                // Per-prey receiver tracking for three-way coupling + silence onset
                let has_signal = max_str > 0.0;
                if has_signal {
                    self.prey[i].actions_with_signal[context][action] += 1;
                } else {
                    self.prey[i].actions_without_signal[context][action] += 1;
                    if self.prey[i].had_signal_prev_tick {
                        self.prey[i].silence_onset_actions[context][action] += 1;
                    }
                }
                self.prey[i].had_signal_prev_tick = has_signal;
            }

            self.apply_outputs(i, action, emit, inputs, zone_dist);

            // Memory EMA update: new_mem = 0.9 * old + 0.1 * output
            for m in 0..MEMORY_SIZE {
                self.prey[i].memory[m] =
                    0.9 * self.prey[i].memory[m] + 0.1 * result.memory_write[m];
            }

            self.prey[i].ticks_alive += 1;
        }
        self.computed_scratch = computed;

        self.move_zones(rng);
        self.zone_drain();

        if self.food.len() < self.food_count / 2 {
            while self.food.len() < self.food_count {
                let f = Food {
                    x: rng.gen_range(0..self.grid_size),
                    y: rng.gen_range(0..self.grid_size),
                    is_patch: rng.gen::<f32>() < self.patch_ratio,
                };
                let idx = self.food.len() as u16;
                self.food_grid.insert(f.x, f.y, idx);
                self.food.push(f);
            }
        }

        if self.collect_metrics {
            self.signals_per_tick
                .push(self.signals_emitted - signals_before);
            self.alive_per_tick
                .push(self.prey.iter().filter(|p| p.alive).count() as u32);
        }
    }

    /// Build input vector using spatial grid for ally/food lookup.
    /// Layout: [danger(2)+spare(1), food(3), ally(3), signals(18), memory(8), energy(1)] = 36
    /// Inputs 0-1: `zone_damage`, `energy_delta`. Input 2: spare (zero).
    #[allow(clippy::similar_names)]
    fn build_inputs_fast(&self, prey_idx: usize) -> [f32; INPUTS] {
        let p = &self.prey[prey_idx];
        let mut inp = [0.0_f32; INPUTS];
        let gs = self.grid_size as f32;

        // 0: accumulated zone damage (body state - prey's own pain, not zone perception)
        inp[0] = p.zone_damage;
        // 1: energy delta since last tick (disambiguates zone damage from metabolism)
        inp[1] = p.energy - p.prev_energy;
        // 2: spare slot (always zero)

        // 3-5: Nearest food (dx, dy, distance)
        if let Some((fi, food_dist_sq)) =
            self.food_grid
                .nearest(p.x, p.y, self.grid_size / 2, u16::MAX)
        {
            let f = &self.food[fi as usize];
            inp[3] = wrap_delta(p.x, f.x, self.grid_size) as f32 / gs;
            inp[4] = wrap_delta(p.y, f.y, self.grid_size) as f32 / gs;
            inp[5] = (food_dist_sq.sqrt() / gs).min(1.0);
        }

        // 6-8: Nearest ally (dx, dy, distance)
        let vision_cells = (gs / 2.0).ceil() as i32;
        if let Some((ally_idx, ally_dist_sq)) =
            self.prey_grid
                .nearest(p.x, p.y, vision_cells, prey_idx as u16)
        {
            let ally = &self.prey[ally_idx as usize];
            inp[6] = wrap_delta(p.x, ally.x, self.grid_size) as f32 / gs;
            inp[7] = wrap_delta(p.y, ally.y, self.grid_size) as f32 / gs;
            inp[8] = (ally_dist_sq.sqrt() / gs).min(1.0);
        } else {
            inp[8] = 1.0; // no ally visible -> max distance
        }

        // Incoming signals (strength + direction per symbol)
        let sig = signal::receive_detailed_grid(
            &self.signals,
            &self.signal_grid,
            p.x,
            p.y,
            gs,
            self.signal_range,
        );
        for (s, rs) in sig.iter().enumerate() {
            let base = SIGNAL_INPUT_START + s * 3;
            inp[base] = rs.strength;
            inp[base + 1] = rs.dx;
            inp[base + 2] = rs.dy;
        }

        // Memory cells
        inp[MEMORY_INPUT_START..MEMORY_INPUT_START + MEMORY_SIZE].copy_from_slice(&p.memory);

        // Own energy
        inp[ENERGY_INPUT_IDX] = p.energy.clamp(0.0, 1.0);

        inp
    }

    /// Rebuild prey spatial grid from current positions. Used by tests that
    /// call `move_zones()` directly without going through `step()`.
    fn rebuild_prey_grid(&mut self) {
        self.prey_grid.clear();
        for (i, p) in self.prey.iter().enumerate() {
            if p.alive {
                self.prey_grid.insert(p.x, p.y, i as u16);
            }
        }
    }

    #[cfg(test)]
    fn build_inputs(&self, prey_idx: usize) -> [f32; INPUTS] {
        self.build_inputs_fast(prey_idx)
    }

    fn apply_outputs(
        &mut self,
        prey_idx: usize,
        action: usize,
        emit: Option<u8>,
        inputs: &[f32; INPUTS],
        zone_dist: f32,
    ) {
        match action {
            0 => self.prey[prey_idx].y = (self.prey[prey_idx].y - 1).rem_euclid(self.grid_size),
            1 => self.prey[prey_idx].y = (self.prey[prey_idx].y + 1).rem_euclid(self.grid_size),
            2 => self.prey[prey_idx].x = (self.prey[prey_idx].x + 1).rem_euclid(self.grid_size),
            3 => self.prey[prey_idx].x = (self.prey[prey_idx].x - 1).rem_euclid(self.grid_size),
            4 => {
                let px = self.prey[prey_idx].x;
                let py = self.prey[prey_idx].y;
                if let Some(fi) = self.nearest_food_grid(px, py, 1) {
                    // Cooperative harvesting: patch food requires 2+ nearby prey
                    if self.food[fi].is_patch {
                        let mut has_partner = false;
                        'search: for dy in -2..=2_i32 {
                            for dx in -2..=2_i32 {
                                let cx = (px + dx).rem_euclid(self.grid_size);
                                let cy = (py + dy).rem_euclid(self.grid_size);
                                let ci = self.prey_grid.cell_idx(cx, cy);
                                for &idx in &self.prey_grid.cells[ci] {
                                    if idx as usize != prey_idx {
                                        has_partner = true;
                                        break 'search;
                                    }
                                }
                            }
                        }
                        if has_partner {
                            self.consume_food(prey_idx, fi);
                        }
                    } else {
                        self.consume_food(prey_idx, fi);
                    }
                }
            }
            _ => {}
        }

        // Signal emission - emit decision pre-computed in parallel phase
        // Suppressed in counterfactual mode (--no-signals)
        let px = self.prey[prey_idx].x;
        let py = self.prey[prey_idx].y;
        if !self.no_signals && self.prey[prey_idx].energy > self.signal_cost {
            if let Some(symbol) = emit {
                self.prey[prey_idx].energy -= self.signal_cost;
                if self.collect_metrics {
                    self.signal_events.push(SignalEvent {
                        symbol,
                        zone_dist,
                        inputs: *inputs,
                        emitter_idx: prey_idx,
                    });
                }
                self.signals.push(Signal {
                    x: px,
                    y: py,
                    symbol,
                    tick_emitted: self.tick,
                });
                self.signals_emitted += 1;
            }
        }
    }

    fn consume_food(&mut self, prey_idx: usize, food_idx: usize) {
        let old_food = self.food[food_idx];
        self.food_grid
            .remove(old_food.x, old_food.y, food_idx as u16);
        self.food.swap_remove(food_idx);
        // swap_remove moved the last element to food_idx - update its grid entry
        if food_idx < self.food.len() {
            let moved = &self.food[food_idx];
            self.food_grid
                .remove(moved.x, moved.y, self.food.len() as u16);
            self.food_grid.insert(moved.x, moved.y, food_idx as u16);
        }
        self.prey[prey_idx].energy = (self.prey[prey_idx].energy + 0.3).min(1.0);
        self.prey[prey_idx].food_eaten += 1;
    }

    /// Move zones via probabilistic random walk. Each zone moves 1 cell in a random
    /// direction with probability = zone.speed each tick.
    fn move_zones(&mut self, rng: &mut impl Rng) {
        let gs = self.grid_size as f32;
        for zone in &mut self.zones {
            if rng.gen::<f32>() < zone.speed {
                let dir = rng.gen_range(0..4);
                match dir {
                    0 => zone.y -= 1.0,
                    1 => zone.y += 1.0,
                    2 => zone.x += 1.0,
                    _ => zone.x -= 1.0,
                }
                // Toroidal wrapping
                zone.x = zone.x.rem_euclid(gs);
                zone.y = zone.y.rem_euclid(gs);
            }
        }
    }

    /// Accumulate zone damage on prey inside kill zones. Stacks across overlapping zones.
    /// Zone damage is separate from energy - food cannot offset it. Death at `zone_damage` >= 1.0.
    /// Dying prey emit only their brain's last chosen signal (via `apply_outputs` before this runs).
    fn zone_drain(&mut self) {
        let gs = self.grid_size as f32;

        for p in &mut self.prey {
            if !p.alive {
                continue;
            }
            for zone in &self.zones {
                let dx = wrap_delta_f32(p.x as f32, zone.x, gs);
                let dy = wrap_delta_f32(p.y as f32, zone.y, gs);
                let dist = (dx * dx + dy * dy).sqrt();
                if dist <= zone.radius {
                    let gradient = 1.0 - dist / zone.radius;
                    p.zone_damage += self.zone_drain_rate * gradient;
                }
            }
            if p.zone_damage >= 1.0 {
                p.alive = false;
                self.zone_deaths += 1;
            }
        }
    }

    /// Grid-based nearest food within `max_dist` Manhattan distance.
    fn nearest_food_grid(&self, x: i32, y: i32, max_dist: i32) -> Option<usize> {
        // Search expanding rings up to max_dist
        for r in 0..=max_dist {
            let mut best: Option<(usize, i32)> = None;
            if r == 0 {
                let ci = self.food_grid.cell_idx(x, y);
                if let Some(&idx) = self.food_grid.cells[ci].first() {
                    return Some(idx as usize);
                }
            } else {
                // Check all cells at Chebyshev distance <= r (Manhattan can differ)
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx.abs().max(dy.abs()) != r {
                            continue; // Only the new ring
                        }
                        let cx = (x + dx).rem_euclid(self.grid_size);
                        let cy = (y + dy).rem_euclid(self.grid_size);
                        let ci = self.food_grid.cell_idx(cx, cy);
                        for &idx in &self.food_grid.cells[ci] {
                            let f = &self.food[idx as usize];
                            let md = wrap_delta(x, f.x, self.grid_size).abs()
                                + wrap_delta(y, f.y, self.grid_size).abs();
                            if md <= max_dist {
                                match best {
                                    None => best = Some((idx as usize, md)),
                                    Some((_, bd)) if md < bd => best = Some((idx as usize, md)),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                if best.is_some() {
                    return best.map(|(i, _)| i);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{Brain, DEFAULT_BASE_HIDDEN, INPUTS, MAX_GENOME_LEN};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // Test defaults
    const TEST_GRID: i32 = 20;
    const TEST_FOOD: usize = 25;
    const TEST_SIGNAL_RANGE: f32 = 8.0;
    const TEST_ZONE_RADIUS: f32 = 8.0;
    const TEST_ZONE_SPEED: f32 = 0.5;
    const TEST_BASE_DRAIN: f32 = 0.0008;
    const TEST_NEURON_COST: f32 = 0.00002;
    const TEST_SIGNAL_COST: f32 = 0.0;
    const TEST_PATCH_RATIO: f32 = 0.0;
    const TEST_ZONE_DRAIN_RATE: f32 = 0.10;

    fn test_prey(x: i32, y: i32) -> Prey {
        Prey {
            x,
            y,
            energy: 1.0,
            alive: true,
            brain: Brain::zero(),
            ticks_alive: 0,
            food_eaten: 0,
            memory: [0.0; MEMORY_SIZE],
            actions_with_signal: [[0; 5]; 2],
            actions_without_signal: [[0; 5]; 2],
            silence_onset_actions: [[0; 5]; 2],
            had_signal_prev_tick: false,
            zone_damage: 0.0,
            prev_energy: 1.0,
        }
    }

    fn minimal_world(prey_positions: &[(i32, i32)], zone_center: (f32, f32)) -> World {
        let prey: Vec<Prey> = prey_positions
            .iter()
            .map(|&(x, y)| test_prey(x, y))
            .collect();
        let prey_count = prey.len();
        World {
            prey,
            zones: vec![KillZone {
                x: zone_center.0,
                y: zone_center.1,
                radius: TEST_ZONE_RADIUS,
                speed: TEST_ZONE_SPEED,
            }],
            food: Vec::new(),
            signals: Vec::new(),
            tick: 0,
            signals_emitted: 0,
            signal_events: Vec::new(),
            ticks_in_zone: 0,
            total_prey_ticks: 0,
            receiver_counts: [[[0u32; 5]; 2]; 1 + NUM_SYMBOLS],
            signals_per_tick: Vec::new(),
            alive_per_tick: Vec::new(),
            min_zone_dist_per_tick: Vec::new(),
            no_signals: true,
            collect_metrics: true,
            prey_grid: CellGrid::new(TEST_GRID),
            food_grid: CellGrid::new(TEST_GRID),
            signal_grid: SignalGrid::new(TEST_GRID, TEST_SIGNAL_RANGE),
            shuffled_indices: (0..prey_count).collect(),
            order_scratch: Vec::with_capacity(prey_count),
            alive_scratch: Vec::with_capacity(prey_count),
            computed_scratch: Vec::with_capacity(prey_count),
            grid_size: TEST_GRID,
            food_count: TEST_FOOD,
            signal_range: TEST_SIGNAL_RANGE,
            base_drain: TEST_BASE_DRAIN,
            neuron_cost: TEST_NEURON_COST,
            signal_cost: TEST_SIGNAL_COST,
            patch_ratio: TEST_PATCH_RATIO,
            zone_drain_rate: TEST_ZONE_DRAIN_RATE,
            signal_ticks: 4,
            zone_deaths: 0,
        }
    }

    // --- Toroidal wrapping ---

    #[test]
    fn wrap_delta_no_wrap() {
        assert_eq!(wrap_delta(3, 7, TEST_GRID), 4);
        assert_eq!(wrap_delta(7, 3, TEST_GRID), -4);
    }

    #[test]
    fn wrap_delta_across_boundary() {
        assert_eq!(wrap_delta(18, 1, TEST_GRID), 3);
        assert_eq!(wrap_delta(1, 18, TEST_GRID), -3);
    }

    #[test]
    fn wrap_delta_half_grid() {
        assert_eq!(wrap_delta(0, 10, TEST_GRID), 10);
        assert_eq!(wrap_delta(10, 0, TEST_GRID), -10);
    }

    #[test]
    fn wrap_dist_sq_same_cell_is_zero() {
        assert!((wrap_dist_sq(5, 5, 5, 5, TEST_GRID) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn wrap_dist_sq_across_boundary() {
        let d = wrap_dist_sq(19, 0, 1, 0, TEST_GRID);
        assert!((d - 4.0).abs() < 1e-6);
    }

    // --- Zone movement ---

    #[test]
    fn zone_moves_with_toroidal_wrapping() {
        let mut world = minimal_world(&[(0, 0)], (19.0, 19.0));
        // Force zone to move by setting speed to 1.0 (always moves)
        world.zones[0].speed = 1.0;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let start_x = world.zones[0].x;
        let start_y = world.zones[0].y;
        world.move_zones(&mut rng);

        // Zone should have moved exactly 1 cell in some direction
        let dx = wrap_delta_f32(start_x, world.zones[0].x, TEST_GRID as f32);
        let dy = wrap_delta_f32(start_y, world.zones[0].y, TEST_GRID as f32);
        let moved_dist = (dx * dx + dy * dy).sqrt();
        assert!(
            (moved_dist - 1.0).abs() < 1e-6,
            "Zone should move exactly 1 cell"
        );
        // Position should be valid (within grid)
        assert!(world.zones[0].x >= 0.0 && world.zones[0].x < TEST_GRID as f32);
        assert!(world.zones[0].y >= 0.0 && world.zones[0].y < TEST_GRID as f32);
    }

    #[test]
    fn zone_stationary_when_speed_zero() {
        let mut world = minimal_world(&[(0, 0)], (10.0, 10.0));
        world.zones[0].speed = 0.0;
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let start_x = world.zones[0].x;
        let start_y = world.zones[0].y;
        for _ in 0..100 {
            world.move_zones(&mut rng);
        }

        assert!((world.zones[0].x - start_x).abs() < 1e-6);
        assert!((world.zones[0].y - start_y).abs() < 1e-6);
    }

    // --- Zone drain ---

    #[test]
    fn zone_drains_zone_damage_when_inside() {
        // Prey at (5,5), zone centered at (5.0,5.0) with radius 8 - prey is inside
        let mut world = minimal_world(&[(5, 5)], (5.0, 5.0));

        let energy_before = world.prey[0].energy;
        world.zone_drain();

        // zone_damage accumulates; energy is unchanged (food cannot offset zone damage)
        assert!((world.prey[0].zone_damage - TEST_ZONE_DRAIN_RATE).abs() < 1e-6);
        assert!(
            (world.prey[0].energy - energy_before).abs() < 1e-6,
            "Energy unchanged by zone"
        );
    }

    #[test]
    fn zone_no_drain_when_outside() {
        // Prey at (0,0), zone centered at (15.0,15.0) with radius 8 - prey is outside
        let mut world = minimal_world(&[(0, 0)], (15.0, 15.0));
        // Distance from (0,0) to (15,15) on 20-grid = sqrt(25+25) = 7.07 via wrapping
        // Actually wrap_delta(0,15,20) = -5, so dist = sqrt(25+25)=7.07 < 8 = inside!
        // Use a farther zone instead.
        world.zones[0].x = 10.0;
        world.zones[0].y = 10.0;
        world.zones[0].radius = 2.0;
        // dist from (0,0) to (10,10) = sqrt(100+100) = 14.14, but wrapping: min(-10,10) = 10
        // wrap_delta(0,10,20) = 10, so dist = sqrt(200) = 14.14 > 2.0

        let before = world.prey[0].energy;
        world.zone_drain();
        let after = world.prey[0].energy;

        assert!((before - after).abs() < 1e-6, "No drain when outside zone");
    }

    #[test]
    fn zone_drain_kills_at_threshold() {
        let mut world = minimal_world(&[(5, 5)], (5.0, 5.0));
        // Pre-load zone_damage just below threshold - one more drain should kill
        world.prey[0].zone_damage = 1.0 - TEST_ZONE_DRAIN_RATE * 0.5;

        world.zone_drain();

        assert!(
            !world.prey[0].alive,
            "Prey should die when zone_damage >= 1.0"
        );
        assert_eq!(world.zone_deaths, 1);
    }

    #[test]
    fn zone_drain_stacks_across_overlapping_zones() {
        let mut world = minimal_world(&[(5, 5)], (5.0, 5.0));
        // Add a second zone also centered at (5,5) so both have dist=0, gradient=1.0
        world.zones.push(KillZone {
            x: 5.0,
            y: 5.0,
            radius: TEST_ZONE_RADIUS,
            speed: TEST_ZONE_SPEED,
        });

        world.zone_drain();

        assert!(
            (world.prey[0].zone_damage - 2.0 * TEST_ZONE_DRAIN_RATE).abs() < 1e-6,
            "Damage should stack from two overlapping zones"
        );
    }

    #[test]
    fn zone_drain_gradient_scales_with_distance() {
        // Prey halfway between center and edge gets ~half the damage
        let mut world = minimal_world(&[(5, 5)], (5.0, 5.0));
        // Place prey at distance = radius/2 from zone center
        let half_r = (TEST_ZONE_RADIUS / 2.0).round() as i32;
        world.prey[0].x = 5 + half_r;

        world.zone_drain();

        let dist = half_r as f32; // exact integer distance
        let expected_gradient = 1.0 - dist / TEST_ZONE_RADIUS;
        let expected_damage = TEST_ZONE_DRAIN_RATE * expected_gradient;
        assert!(
            (world.prey[0].zone_damage - expected_damage).abs() < 1e-4,
            "Gradient damage: expected {expected_damage}, got {}",
            world.prey[0].zone_damage
        );
    }

    #[test]
    fn zone_drain_zero_at_edge() {
        // Prey right at zone edge gets near-zero damage
        let mut world = minimal_world(&[(5, 5)], (5.0, 5.0));
        // Place prey at distance ~= radius (just inside)
        let edge_dist = (TEST_ZONE_RADIUS - 0.1).round() as i32;
        world.prey[0].x = 5 + edge_dist;

        world.zone_drain();

        // Should be very small but non-zero
        assert!(
            world.prey[0].zone_damage < TEST_ZONE_DRAIN_RATE * 0.3,
            "Edge prey should get much less damage than center prey, got {}",
            world.prey[0].zone_damage
        );
    }

    #[test]
    fn zone_damage_does_not_deplete_energy() {
        // Energy should remain unchanged regardless of zone damage accumulation
        let mut world = minimal_world(&[(5, 5)], (5.0, 5.0));
        let energy_before = world.prey[0].energy;

        for _ in 0..5 {
            world.zone_drain();
            // Keep prey alive by resetting zone_damage
            world.prey[0].zone_damage = 0.0;
        }

        assert!(
            (world.prey[0].energy - energy_before).abs() < 1e-6,
            "Zone drain must not deplete energy"
        );
    }

    #[test]
    fn zone_death_emits_no_extra_signals() {
        // zone_drain kills prey but does not inject artificial signals into the channel.
        // Dying prey signal only via apply_outputs (their brain's last chosen signal).
        let mut world = minimal_world(&[(5, 5)], (5.0, 5.0));
        world.prey[0].zone_damage = 1.0 - TEST_ZONE_DRAIN_RATE * 0.5;
        assert!(world.signals.is_empty());

        world.zone_drain();

        assert!(!world.prey[0].alive);
        assert!(
            world.signals.is_empty(),
            "zone_drain must not inject signals - dying prey signal only via apply_outputs"
        );
    }

    #[test]
    fn zone_no_drain_energy_when_outside() {
        // Prey at (0,0), zone centered far away - neither energy nor zone_damage changes
        let mut world = minimal_world(&[(0, 0)], (15.0, 15.0));
        world.zones[0].x = 10.0;
        world.zones[0].y = 10.0;
        world.zones[0].radius = 2.0;

        let energy_before = world.prey[0].energy;
        world.zone_drain();

        assert!(
            (world.prey[0].energy - energy_before).abs() < 1e-6,
            "No energy change when outside zone"
        );
        assert!(
            world.prey[0].zone_damage < 1e-6,
            "No zone_damage when outside zone"
        );
    }

    // --- new_with_positions ---

    #[test]
    fn new_with_positions_places_correctly() {
        let agents = vec![
            Agent {
                brain: Brain {
                    weights: [0.1; MAX_GENOME_LEN],
                    base_hidden_size: DEFAULT_BASE_HIDDEN,
                    signal_hidden_size: crate::brain::DEFAULT_SIGNAL_HIDDEN,
                },
                x: 3,
                y: 7,
                parent_indices: [None, None],
                grandparent_indices: [None; 4],
            },
            Agent {
                brain: Brain {
                    weights: [0.2; MAX_GENOME_LEN],
                    base_hidden_size: DEFAULT_BASE_HIDDEN,
                    signal_hidden_size: crate::brain::DEFAULT_SIGNAL_HIDDEN,
                },
                x: 15,
                y: 2,
                parent_indices: [None, None],
                grandparent_indices: [None; 4],
            },
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let world = World::new_with_positions(
            &agents,
            1,
            &mut rng,
            false,
            true,
            TEST_GRID,
            TEST_FOOD,
            TEST_SIGNAL_RANGE,
            TEST_ZONE_RADIUS,
            TEST_ZONE_SPEED,
            TEST_BASE_DRAIN,
            TEST_NEURON_COST,
            TEST_SIGNAL_COST,
            TEST_PATCH_RATIO,
            TEST_ZONE_DRAIN_RATE,
            4,
        );

        assert_eq!(world.prey[0].x, 3);
        assert_eq!(world.prey[0].y, 7);
        assert_eq!(world.prey[1].x, 15);
        assert_eq!(world.prey[1].y, 2);
        assert_eq!(world.zones.len(), 1);
    }

    // --- Energy mechanics ---

    #[test]
    fn energy_drains_per_tick() {
        // Place zone far from prey with small radius to avoid zone drain
        let mut world = minimal_world(&[(0, 0)], (10.0, 10.0));
        world.zones[0].radius = 1.0; // small zone far from (0,0)
        world.food.push(Food {
            x: 10,
            y: 10,
            is_patch: false,
        });
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let before = world.prey[0].energy;
        world.step(&mut rng);
        let after = world.prey[0].energy;

        let total_hidden = DEFAULT_BASE_HIDDEN + crate::brain::DEFAULT_SIGNAL_HIDDEN;
        let expected_drain = TEST_BASE_DRAIN + total_hidden as f32 * TEST_NEURON_COST;
        assert!((before - after - expected_drain).abs() < 1e-6);
    }

    #[test]
    fn food_consumption_restores_energy() {
        let mut world = minimal_world(&[(5, 5)], (15.0, 15.0));
        world.prey[0].energy = 0.5;
        world.food.push(Food {
            x: 5,
            y: 5,
            is_patch: false,
        });
        world.food_grid.insert(5, 5, 0);

        let _result = ForwardResult {
            actions: [0.0, 0.0, 0.0, 0.0, 1.0],
            signals: [0.0; crate::brain::SIGNAL_OUTPUTS],
            memory_write: [0.0; MEMORY_SIZE],
        };
        let inputs = [0.0_f32; INPUTS];
        world.apply_outputs(0, 4, None, &inputs, f32::MAX);

        assert!((world.prey[0].energy - 0.8).abs() < 1e-6);
    }

    #[test]
    fn energy_caps_at_one() {
        let mut world = minimal_world(&[(5, 5)], (15.0, 15.0));
        world.prey[0].energy = 0.9;
        world.food.push(Food {
            x: 5,
            y: 5,
            is_patch: false,
        });
        world.food_grid.insert(5, 5, 0);

        let _result = ForwardResult {
            actions: [0.0, 0.0, 0.0, 0.0, 1.0],
            signals: [0.0; crate::brain::SIGNAL_OUTPUTS],
            memory_write: [0.0; MEMORY_SIZE],
        };
        let inputs = [0.0_f32; INPUTS];
        world.apply_outputs(0, 4, None, &inputs, f32::MAX);

        assert!((world.prey[0].energy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn energy_death_at_zero() {
        let mut world = minimal_world(&[(0, 0)], (10.0, 10.0));
        world.zones[0].radius = 1.0; // small zone far from prey
        let total_hidden = DEFAULT_BASE_HIDDEN + crate::brain::DEFAULT_SIGNAL_HIDDEN;
        let drain = TEST_BASE_DRAIN + total_hidden as f32 * TEST_NEURON_COST;
        world.prey[0].energy = drain * 0.5;
        world.food.push(Food {
            x: 10,
            y: 10,
            is_patch: false,
        });
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.step(&mut rng);

        assert!(!world.prey[0].alive);
    }

    // --- Food respawn ---

    #[test]
    fn food_respawns_when_below_half() {
        let mut world = minimal_world(&[(0, 0)], (10.0, 10.0));
        world.zones[0].radius = 1.0; // small zone far from prey
        for x in 5..16 {
            world.food.push(Food {
                x,
                y: 5,
                is_patch: false,
            });
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.step(&mut rng);

        assert_eq!(world.food.len(), TEST_FOOD);
    }

    #[test]
    fn food_does_not_respawn_above_half() {
        let mut world = minimal_world(&[(0, 0)], (10.0, 10.0));
        world.zones[0].radius = 1.0; // small zone far from prey
        for x in 5..18 {
            world.food.push(Food {
                x,
                y: 5,
                is_patch: false,
            });
        }
        let initial_count = world.food.len();
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.step(&mut rng);

        assert!(world.food.len() <= initial_count);
    }

    // --- Danger sense inputs ---

    #[test]
    fn input_zone_damage_reflects_accumulation() {
        let mut world = minimal_world(&[(5, 5)], (5.0, 5.0)); // prey at zone center
        world.prey[0].zone_damage = 0.5;

        let inputs = world.build_inputs(0);

        assert!(
            (inputs[0] - 0.5).abs() < 1e-6,
            "Input 0 should be zone_damage"
        );
    }

    #[test]
    fn input_energy_delta_negative_after_drain() {
        let mut world = minimal_world(&[(5, 5)], (15.0, 15.0)); // prey far from zone
        world.zones[0].radius = 1.0;
        // Simulate: prev_energy was 1.0, energy dropped to 0.9 from metabolism
        world.prey[0].prev_energy = 1.0;
        world.prey[0].energy = 0.9;

        let inputs = world.build_inputs(0);

        assert!(
            (inputs[1] - (-0.1)).abs() < 1e-4,
            "Input 1 should be energy_delta = energy - prev_energy"
        );
    }

    #[test]
    fn input_energy_delta_positive_after_food() {
        let mut world = minimal_world(&[(5, 5)], (15.0, 15.0));
        world.zones[0].radius = 1.0;
        // Simulate: prey ate food, energy went up
        world.prey[0].prev_energy = 0.5;
        world.prey[0].energy = 0.8;

        let inputs = world.build_inputs(0);

        assert!(
            (inputs[1] - 0.3).abs() < 1e-4,
            "Input 1 should be positive after food"
        );
    }

    #[test]
    fn input_spare_slot_always_zero() {
        let mut world = minimal_world(&[(5, 5)], (5.0, 5.0));
        world.prey[0].zone_damage = 0.5;
        world.prey[0].energy = 0.7;

        let inputs = world.build_inputs(0);

        assert!((inputs[2]).abs() < 1e-6, "Input 2 (spare) should be zero");
    }

    // --- Per-prey receiver tracking ---

    #[test]
    fn per_prey_tracking_accumulates_with_and_without_signal() {
        let mut world = minimal_world(&[(5, 5)], (15.0, 15.0));
        world.zones[0].radius = 1.0; // small zone far from prey
        world.food.push(Food {
            x: 10,
            y: 10,
            is_patch: false,
        });
        world.signals.push(crate::signal::Signal {
            x: 5,
            y: 5,
            symbol: 0,
            tick_emitted: 0,
        });
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.step(&mut rng);
        let total_with: u32 = world.prey[0]
            .actions_with_signal
            .iter()
            .flat_map(|c| c.iter())
            .sum();
        let total_without: u32 = world.prey[0]
            .actions_without_signal
            .iter()
            .flat_map(|c| c.iter())
            .sum();
        assert_eq!(total_with, 1, "Should have 1 action with signal");
        assert_eq!(total_without, 0, "Should have 0 actions without signal");
        assert!(world.prey[0].had_signal_prev_tick);

        world.signals.clear();
        world.step(&mut rng);
        let total_onset: u32 = world.prey[0]
            .silence_onset_actions
            .iter()
            .flat_map(|c| c.iter())
            .sum();
        let total_without_after: u32 = world.prey[0]
            .actions_without_signal
            .iter()
            .flat_map(|c| c.iter())
            .sum();
        assert_eq!(total_onset, 1, "Should detect silence onset");
        assert_eq!(
            total_without_after, 1,
            "Should have 1 action without signal"
        );
        assert!(!world.prey[0].had_signal_prev_tick);

        world.step(&mut rng);
        let total_onset_after: u32 = world.prey[0]
            .silence_onset_actions
            .iter()
            .flat_map(|c| c.iter())
            .sum();
        assert_eq!(total_onset_after, 1, "Onset should not increment again");
    }

    #[test]
    fn build_inputs_returns_correct_size() {
        let mut world = minimal_world(&[(0, 0)], (15.0, 15.0));
        world.food.push(Food {
            x: 5,
            y: 5,
            is_patch: false,
        });

        let inputs = world.build_inputs(0);

        assert_eq!(inputs.len(), INPUTS);
    }

    #[test]
    fn larger_brain_drains_more() {
        let mut world = minimal_world(&[(0, 0), (5, 5)], (15.0, 15.0));
        world.zones[0].radius = 1.0;
        world.food.push(Food {
            x: 10,
            y: 10,
            is_patch: false,
        });
        world.prey[0].brain.base_hidden_size = DEFAULT_BASE_HIDDEN;
        world.prey[0].brain.signal_hidden_size = crate::brain::DEFAULT_SIGNAL_HIDDEN;
        world.prey[1].brain.base_hidden_size = crate::brain::MAX_BASE_HIDDEN;
        world.prey[1].brain.signal_hidden_size = crate::brain::MAX_SIGNAL_HIDDEN;

        let mut rng = ChaCha8Rng::seed_from_u64(0);
        world.step(&mut rng);

        // Larger brain should have less energy remaining
        assert!(
            world.prey[1].energy < world.prey[0].energy,
            "Larger brain ({}) should drain more than default ({})",
            world.prey[1].energy,
            world.prey[0].energy
        );
    }

    #[test]
    fn min_brain_drains_less() {
        let mut world = minimal_world(&[(0, 0), (5, 5)], (15.0, 15.0));
        world.zones[0].radius = 1.0;
        world.food.push(Food {
            x: 10,
            y: 10,
            is_patch: false,
        });
        world.prey[0].brain.base_hidden_size = DEFAULT_BASE_HIDDEN;
        world.prey[0].brain.signal_hidden_size = crate::brain::DEFAULT_SIGNAL_HIDDEN;
        world.prey[1].brain.base_hidden_size = crate::brain::MIN_BASE_HIDDEN;
        world.prey[1].brain.signal_hidden_size = crate::brain::MIN_SIGNAL_HIDDEN;

        let mut rng = ChaCha8Rng::seed_from_u64(0);
        world.step(&mut rng);

        // Smaller brain should have more energy remaining
        assert!(
            world.prey[1].energy > world.prey[0].energy,
            "Min brain ({}) should drain less than default ({})",
            world.prey[1].energy,
            world.prey[0].energy
        );
    }

    #[test]
    fn default_hidden_drain_with_cheap_neurons() {
        let total_hidden = DEFAULT_BASE_HIDDEN + crate::brain::DEFAULT_SIGNAL_HIDDEN;
        let drain = TEST_BASE_DRAIN + total_hidden as f32 * TEST_NEURON_COST;
        // base_hidden=12, signal_hidden=6 -> total=18 -> 0.0008 + 18*0.00002 = 0.00116
        assert!(
            (drain - 0.00116).abs() < 1e-6,
            "Default drain should be 0.00116 with cheap neurons, got {drain}"
        );
    }

    // --- Cooperative food patches ---

    #[test]
    fn patch_food_requires_partner() {
        let mut world = minimal_world(&[(5, 5)], (15.0, 15.0));
        world.food.push(Food {
            x: 5,
            y: 5,
            is_patch: true,
        });
        world.food_grid.insert(5, 5, 0);
        world.rebuild_prey_grid();

        let _result = ForwardResult {
            actions: [0.0, 0.0, 0.0, 0.0, 1.0],
            signals: [0.0; crate::brain::SIGNAL_OUTPUTS],
            memory_write: [0.0; MEMORY_SIZE],
        };
        let inputs = [0.0_f32; INPUTS];
        world.apply_outputs(0, 4, None, &inputs, f32::MAX);

        // Solo prey should NOT consume patch food
        assert_eq!(
            world.food.len(),
            1,
            "Patch food should not be consumed solo"
        );
        assert_eq!(world.prey[0].food_eaten, 0);
    }

    #[test]
    fn patch_food_consumed_with_partner() {
        let mut world = minimal_world(&[(5, 5), (6, 5)], (15.0, 15.0));
        world.food.push(Food {
            x: 5,
            y: 5,
            is_patch: true,
        });
        world.food_grid.insert(5, 5, 0);
        world.rebuild_prey_grid();

        let _result = ForwardResult {
            actions: [0.0, 0.0, 0.0, 0.0, 1.0],
            signals: [0.0; crate::brain::SIGNAL_OUTPUTS],
            memory_write: [0.0; MEMORY_SIZE],
        };
        let inputs = [0.0_f32; INPUTS];
        world.apply_outputs(0, 4, None, &inputs, f32::MAX);

        assert_eq!(
            world.food.len(),
            0,
            "Patch food should be consumed with partner nearby"
        );
        assert_eq!(world.prey[0].food_eaten, 1);
    }

    #[test]
    fn non_patch_food_consumed_solo() {
        let mut world = minimal_world(&[(5, 5)], (15.0, 15.0));
        world.food.push(Food {
            x: 5,
            y: 5,
            is_patch: false,
        });
        world.food_grid.insert(5, 5, 0);
        world.rebuild_prey_grid();

        let _result = ForwardResult {
            actions: [0.0, 0.0, 0.0, 0.0, 1.0],
            signals: [0.0; crate::brain::SIGNAL_OUTPUTS],
            memory_write: [0.0; MEMORY_SIZE],
        };
        let inputs = [0.0_f32; INPUTS];
        world.apply_outputs(0, 4, None, &inputs, f32::MAX);

        assert_eq!(
            world.food.len(),
            0,
            "Non-patch food should be consumed solo"
        );
        assert_eq!(world.prey[0].food_eaten, 1);
    }

    // --- Memory ---

    #[test]
    fn memory_initialized_small_random() {
        let agents = vec![Agent {
            brain: Brain::zero(),
            x: 0,
            y: 0,
            parent_indices: [None, None],
            grandparent_indices: [None; 4],
        }];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let world = World::new_with_positions(
            &agents,
            1,
            &mut rng,
            false,
            true,
            TEST_GRID,
            TEST_FOOD,
            TEST_SIGNAL_RANGE,
            TEST_ZONE_RADIUS,
            TEST_ZONE_SPEED,
            TEST_BASE_DRAIN,
            TEST_NEURON_COST,
            TEST_SIGNAL_COST,
            TEST_PATCH_RATIO,
            TEST_ZONE_DRAIN_RATE,
            4,
        );

        for &m in &world.prey[0].memory {
            assert!(
                m.abs() <= 0.1,
                "Memory init should be in [-0.1, 0.1], got {m}"
            );
        }
    }
}
