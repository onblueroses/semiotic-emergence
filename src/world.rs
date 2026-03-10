use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;

use crate::brain::{Brain, INPUTS, OUTPUTS};
use crate::evolution::Agent;
use crate::signal::{self, Signal, SIGNAL_THRESHOLD};

pub const INPUT_NAMES: [&str; INPUTS] = [
    "pred_dx",
    "pred_dy",
    "pred_dist",
    "food_dx",
    "food_dy",
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
                    // Same cell = distance 0
                    match best {
                        None => best = Some((idx, 0.0)),
                        Some(_) => return best, // Can't beat 0
                    }
                }
            } else {
                // Walk the 4 edges of the Chebyshev ring at distance r
                for edge in 0..4 {
                    let (start_dx, start_dy, step_dx, step_dy) = match edge {
                        0 => (-r, -r, 1, 0), // top edge: left to right
                        1 => (r, -r, 0, 1),  // right edge: top to bottom
                        2 => (r, r, -1, 0),  // bottom edge: right to left
                        _ => (-r, r, 0, -1), // left edge: bottom to top
                    };
                    for step in 0..(2 * r) {
                        let dx = start_dx + step * step_dx;
                        let dy = start_dy + step * step_dy;
                        let cx = (x + dx).rem_euclid(gs);
                        let cy = (y + dy).rem_euclid(gs);
                        let ci = self.cell_idx(cx, cy);
                        for &idx in &self.cells[ci] {
                            if idx == skip_idx {
                                continue;
                            }
                            let d = wrap_dist_sq(x, y, cx, cy, gs);
                            match best {
                                None => best = Some((idx, d)),
                                Some((_, bd)) if d < bd => best = Some((idx, d)),
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
    /// Per-prey action counts when receiving any signal, by context.
    pub actions_with_signal: [[u32; 5]; 2],
    /// Per-prey action counts when not receiving any signal, by context.
    pub actions_without_signal: [[u32; 5]; 2],
    /// Actions at the tick a signal disappears (onset of silence), by context.
    pub silence_onset_actions: [[u32; 5]; 2],
    /// Whether this prey received a signal on the previous tick (for onset detection).
    pub had_signal_prev_tick: bool,
}

#[derive(Clone, Debug)]
pub struct Predator {
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct Food {
    pub x: i32,
    pub y: i32,
}

pub struct SignalEvent {
    pub symbol: u8,
    pub predator_dist: f32,
    pub inputs: [f32; INPUTS],
    pub emitter_idx: usize,
}

pub struct World {
    pub prey: Vec<Prey>,
    pub predators: Vec<Predator>,
    pub food: Vec<Food>,
    pub signals: Vec<Signal>,
    pub tick: u32,
    pub signals_emitted: u32,
    pub signal_events: Vec<SignalEvent>,
    pub ticks_near_predator: u32,
    pub total_prey_ticks: u32,
    /// Receiver response spectrum: `[signal_state][context][action]` counts.
    /// `signal_state`: 0=none, 1=sym0, 2=sym1, 3=sym2 (strongest received).
    /// `context`: 0=no predator, 1=predator visible.
    /// `action`: 0-4 (up/down/right/left/eat).
    pub receiver_counts: [[[u32; 5]; 2]; 4],
    /// Signal count per tick (for silence correlation).
    pub signals_per_tick: Vec<u32>,
    /// Minimum predator-to-alive-prey distance per tick.
    pub min_pred_dist_per_tick: Vec<f32>,
    /// When true, signal emission is suppressed (counterfactual mode).
    pub no_signals: bool,
    // Spatial indices (rebuilt each tick for prey, maintained incrementally for food)
    prey_grid: CellGrid,
    food_grid: CellGrid,
    // Pre-allocated per-tick buffers
    shuffled_indices: Vec<usize>,
    prey_positions: Vec<(i32, i32)>,
    // Simulation parameters
    pub grid_size: i32,
    pub food_count: usize,
    pub prey_vision_range: f32,
    pub signal_range: f32,
    pub predator_speed: u32,
    pub base_drain: f32,
    pub neuron_cost: f32,
    pub signal_cost: f32,
}

impl World {
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_positions(
        agents: &[Agent],
        num_predators: usize,
        rng: &mut impl Rng,
        no_signals: bool,
        grid_size: i32,
        food_count: usize,
        prey_vision_range: f32,
        signal_range: f32,
        predator_speed: u32,
        base_drain: f32,
        neuron_cost: f32,
        signal_cost: f32,
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
                actions_with_signal: [[0; 5]; 2],
                actions_without_signal: [[0; 5]; 2],
                silence_onset_actions: [[0; 5]; 2],
                had_signal_prev_tick: false,
            })
            .collect();

        let predators = (0..num_predators)
            .map(|_| Predator {
                x: rng.gen_range(0..grid_size),
                y: rng.gen_range(0..grid_size),
            })
            .collect();

        let food: Vec<Food> = (0..food_count)
            .map(|_| Food {
                x: rng.gen_range(0..grid_size),
                y: rng.gen_range(0..grid_size),
            })
            .collect();

        let mut food_grid = CellGrid::new(grid_size);
        for (i, f) in food.iter().enumerate() {
            food_grid.insert(f.x, f.y, i as u16);
        }

        let prey_count = prey.len();
        Self {
            prey,
            predators,
            food,
            signals: Vec::new(),
            tick: 0,
            signals_emitted: 0,
            signal_events: Vec::new(),
            ticks_near_predator: 0,
            total_prey_ticks: 0,
            receiver_counts: [[[0u32; 5]; 2]; 4],
            signals_per_tick: Vec::new(),
            min_pred_dist_per_tick: Vec::new(),
            no_signals,
            prey_grid: CellGrid::new(grid_size),
            food_grid,
            shuffled_indices: (0..prey_count).collect(),
            prey_positions: Vec::with_capacity(prey_count),
            grid_size,
            food_count,
            prey_vision_range,
            signal_range,
            predator_speed,
            base_drain,
            neuron_cost,
            signal_cost,
        }
    }

    #[cfg(test)]
    fn nearest_predator(&self, x: i32, y: i32) -> &Predator {
        let mut best = &self.predators[0];
        let mut best_d = wrap_dist_sq(x, y, best.x, best.y, self.grid_size);
        for pred in &self.predators[1..] {
            let d = wrap_dist_sq(x, y, pred.x, pred.y, self.grid_size);
            if d < best_d {
                best = pred;
                best_d = d;
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

        self.signals
            .retain(|s| self.tick.saturating_sub(s.tick_emitted) <= 4);

        // Build prey spatial grid and position snapshot
        self.rebuild_prey_grid();
        self.prey_positions.clear();
        for p in &self.prey {
            self.prey_positions.push((p.x, p.y));
        }

        // Track minimum predator-to-alive-prey distance this tick
        let mut min_pred_dist = f32::MAX;
        for p in &self.prey {
            if !p.alive {
                continue;
            }
            for pred in &self.predators {
                let d = wrap_dist_sq(p.x, p.y, pred.x, pred.y, self.grid_size).sqrt();
                if d < min_pred_dist {
                    min_pred_dist = d;
                }
            }
        }
        self.min_pred_dist_per_tick.push(min_pred_dist);

        // Shuffle prey processing order to prevent index bias
        self.shuffled_indices.clear();
        self.shuffled_indices.extend(0..self.prey.len());
        self.shuffled_indices.shuffle(rng);

        // Cache nearest predator per prey (avoids 2-3 redundant scans per prey)
        let mut cached_pred: Vec<(usize, f32)> = Vec::with_capacity(self.prey.len());
        for p in &self.prey {
            if !p.alive {
                cached_pred.push((0, f32::MAX));
                continue;
            }
            let mut best_idx = 0;
            let mut best_d = f32::MAX;
            for (pi, pred) in self.predators.iter().enumerate() {
                let d = wrap_dist_sq(p.x, p.y, pred.x, pred.y, self.grid_size);
                if d < best_d {
                    best_d = d;
                    best_idx = pi;
                }
            }
            cached_pred.push((best_idx, best_d));
        }

        // Need a stable copy of the order for the borrow-splitting loop
        let order: Vec<usize> = self.shuffled_indices.clone();

        // Apply metabolism sequentially (mutates prey energy/alive, cheap)
        for &i in &order {
            if !self.prey[i].alive {
                continue;
            }
            let drain = self.base_drain + self.prey[i].brain.hidden_size as f32 * self.neuron_cost;
            self.prey[i].energy -= drain;
            if self.prey[i].energy <= 0.0 {
                self.prey[i].alive = false;
            }
        }

        // Parallel compute: build inputs + run brain forward for all alive prey
        // All reads are from immutable &self state (grids, predators, signals, food)
        let computed: Vec<(usize, [f32; INPUTS], [f32; OUTPUTS], f32)> = order
            .par_iter()
            .filter_map(|&i| {
                if !self.prey[i].alive {
                    return None;
                }
                let (pred_idx, pred_dist_sq) = cached_pred[i];
                let pdist = pred_dist_sq.sqrt();
                let inputs = self.build_inputs_fast(i, pred_idx, pdist, &self.prey_positions);
                let outputs = self.prey[i].brain.forward(&inputs);
                Some((i, inputs, outputs, pdist))
            })
            .collect();

        // Sequential apply: mutations to world state
        for &(i, ref inputs, ref outputs, pdist) in &computed {
            self.total_prey_ticks += 1;
            if pdist <= self.prey_vision_range {
                self.ticks_near_predator += 1;
            }

            // Receiver response spectrum: classify signal state, context, and chosen action
            let strengths = [inputs[6], inputs[9], inputs[12]];
            let max_str = strengths[0].max(strengths[1]).max(strengths[2]);
            let signal_state: usize = if max_str > 0.0 {
                1 + strengths
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(i, _)| i)
            } else {
                0
            };
            let context = usize::from(inputs[2] > 0.0);
            let action = outputs[..5]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);
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

            self.apply_outputs(i, outputs, inputs, pdist);

            // Evasion boost: +1 movement when signaled AND predator nearby
            let received_signal = inputs[6] > 0.0 || inputs[9] > 0.0 || inputs[12] > 0.0;
            if received_signal && pdist <= self.signal_range && action < 4 {
                match action {
                    0 => self.prey[i].y = (self.prey[i].y - 1).rem_euclid(self.grid_size),
                    1 => self.prey[i].y = (self.prey[i].y + 1).rem_euclid(self.grid_size),
                    2 => self.prey[i].x = (self.prey[i].x + 1).rem_euclid(self.grid_size),
                    3 => self.prey[i].x = (self.prey[i].x - 1).rem_euclid(self.grid_size),
                    _ => {}
                }
            }

            self.prey[i].ticks_alive += 1;
            self.prey[i].had_signal_prev_tick = has_signal;
        }

        self.move_predators();
        // Kill check runs once after all predator movement, not per sub-step.
        self.predator_kill();

        if self.food.len() < self.food_count / 2 {
            while self.food.len() < self.food_count {
                let f = Food {
                    x: rng.gen_range(0..self.grid_size),
                    y: rng.gen_range(0..self.grid_size),
                };
                let idx = self.food.len() as u16;
                self.food_grid.insert(f.x, f.y, idx);
                self.food.push(f);
            }
        }

        self.signals_per_tick
            .push(self.signals_emitted - signals_before);
    }

    /// Build input vector using cached predator info and spatial grid for ally lookup.
    #[allow(clippy::similar_names)]
    fn build_inputs_fast(
        &self,
        prey_idx: usize,
        pred_idx: usize,
        pdist: f32,
        positions: &[(i32, i32)],
    ) -> [f32; INPUTS] {
        let p = &self.prey[prey_idx];
        let mut inp = [0.0_f32; INPUTS];
        let gs = self.grid_size as f32;

        // 0-2: Nearest predator (from cache, gated by vision range)
        if pdist <= self.prey_vision_range {
            let pred = &self.predators[pred_idx];
            let pdx = wrap_delta(p.x, pred.x, self.grid_size) as f32;
            let pdy = wrap_delta(p.y, pred.y, self.grid_size) as f32;
            inp[0] = pdx / gs;
            inp[1] = pdy / gs;
            inp[2] = (pdist / self.prey_vision_range).min(1.0);
        }

        // 3-4: Nearest food via spatial grid
        if let Some((fi, _)) = self
            .food_grid
            .nearest(p.x, p.y, self.grid_size / 2, u16::MAX)
        {
            let f = &self.food[fi as usize];
            inp[3] = wrap_delta(p.x, f.x, self.grid_size) as f32 / gs;
            inp[4] = wrap_delta(p.y, f.y, self.grid_size) as f32 / gs;
        }

        // 5: Nearest ally via prey grid (uses start-of-tick positions)
        let _ = positions; // positions captured for borrow-checker; grid uses cell coords
        let vision_cells = (gs / 2.0).ceil() as i32;
        let ally = self
            .prey_grid
            .nearest(p.x, p.y, vision_cells, prey_idx as u16);
        inp[5] = if let Some((_, d_sq)) = ally {
            (d_sq.sqrt() / gs).min(1.0)
        } else {
            1.0
        };

        // 6-14: Incoming signals (strength + direction per symbol)
        let sig =
            signal::receive_detailed(&self.signals, p.x, p.y, self.tick, gs, self.signal_range);
        inp[6] = sig[0].strength;
        inp[7] = sig[0].dx;
        inp[8] = sig[0].dy;
        inp[9] = sig[1].strength;
        inp[10] = sig[1].dx;
        inp[11] = sig[1].dy;
        inp[12] = sig[2].strength;
        inp[13] = sig[2].dx;
        inp[14] = sig[2].dy;

        // 15: Own energy
        inp[15] = p.energy.clamp(0.0, 1.0);

        inp
    }

    /// Rebuild prey spatial grid from current positions. Used by tests that
    /// call `move_predators()` directly without going through `step()`.
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
        let p = &self.prey[prey_idx];
        let nearest_pred = self.nearest_predator(p.x, p.y);
        let pred_idx = self
            .predators
            .iter()
            .position(|pr| std::ptr::eq(pr, nearest_pred))
            .unwrap_or(0);
        let pdx = wrap_delta(p.x, nearest_pred.x, self.grid_size) as f32;
        let pdy = wrap_delta(p.y, nearest_pred.y, self.grid_size) as f32;
        let pdist = (pdx * pdx + pdy * pdy).sqrt();

        // Need food grid populated for tests using build_inputs directly
        self.build_inputs_fast(prey_idx, pred_idx, pdist, &[])
    }

    fn apply_outputs(
        &mut self,
        prey_idx: usize,
        outputs: &[f32; OUTPUTS],
        inputs: &[f32; INPUTS],
        predator_dist: f32,
    ) {
        let action = outputs[..5]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);

        match action {
            0 => self.prey[prey_idx].y = (self.prey[prey_idx].y - 1).rem_euclid(self.grid_size),
            1 => self.prey[prey_idx].y = (self.prey[prey_idx].y + 1).rem_euclid(self.grid_size),
            2 => self.prey[prey_idx].x = (self.prey[prey_idx].x + 1).rem_euclid(self.grid_size),
            3 => self.prey[prey_idx].x = (self.prey[prey_idx].x - 1).rem_euclid(self.grid_size),
            4 => {
                let px = self.prey[prey_idx].x;
                let py = self.prey[prey_idx].y;
                if let Some(fi) = self.nearest_food_grid(px, py, 1) {
                    let old_food = self.food[fi];
                    self.food_grid.remove(old_food.x, old_food.y, fi as u16);
                    self.food.swap_remove(fi);
                    // swap_remove moved the last element to fi - update its grid entry
                    if fi < self.food.len() {
                        let moved = &self.food[fi];
                        self.food_grid
                            .remove(moved.x, moved.y, self.food.len() as u16);
                        self.food_grid.insert(moved.x, moved.y, fi as u16);
                    }
                    self.prey[prey_idx].energy = (self.prey[prey_idx].energy + 0.3).min(1.0);
                    self.prey[prey_idx].food_eaten += 1;
                }
            }
            _ => {}
        }

        // Signal emission (outputs 5-7) - costs energy
        // Suppressed in counterfactual mode (--no-signals)
        let px = self.prey[prey_idx].x;
        let py = self.prey[prey_idx].y;
        if !self.no_signals && self.prey[prey_idx].energy > self.signal_cost {
            if let Some(symbol) = signal::maybe_emit(outputs.as_slice(), SIGNAL_THRESHOLD) {
                self.prey[prey_idx].energy -= self.signal_cost;
                self.signal_events.push(SignalEvent {
                    symbol,
                    predator_dist,
                    inputs: *inputs,
                    emitter_idx: prey_idx,
                });
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

    fn move_predators(&mut self) {
        // prey_grid is stale (from tick start) but acceptable for predator chasing
        for pred_idx in 0..self.predators.len() {
            for _ in 0..self.predator_speed {
                let px = self.predators[pred_idx].x;
                let py = self.predators[pred_idx].y;
                let nearest = self.prey_grid.nearest(px, py, self.grid_size / 2, u16::MAX);

                if let Some((prey_idx, _)) = nearest {
                    let tp = &self.prey[prey_idx as usize];
                    let dx = wrap_delta(px, tp.x, self.grid_size);
                    let dy = wrap_delta(py, tp.y, self.grid_size);
                    if dx.abs() >= dy.abs() {
                        self.predators[pred_idx].x += dx.signum();
                    } else {
                        self.predators[pred_idx].y += dy.signum();
                    }
                    self.predators[pred_idx].x =
                        self.predators[pred_idx].x.rem_euclid(self.grid_size);
                    self.predators[pred_idx].y =
                        self.predators[pred_idx].y.rem_euclid(self.grid_size);
                }
            }
        }
    }

    fn predator_kill(&mut self) {
        for p in &mut self.prey {
            if !p.alive {
                continue;
            }
            for pred in &self.predators {
                let dx = wrap_delta(pred.x, p.x, self.grid_size).abs();
                let dy = wrap_delta(pred.y, p.y, self.grid_size).abs();
                if dx == 0 && dy == 0 {
                    p.alive = false;
                    break;
                }
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

    #[cfg(test)]
    fn nearest_food(&self, x: i32, y: i32) -> Option<&Food> {
        let gs = self.grid_size;
        self.food
            .iter()
            .min_by_key(|f| wrap_delta(x, f.x, gs).abs() + wrap_delta(y, f.y, gs).abs())
    }

    #[cfg(test)]
    fn nearest_food_idx(&self, x: i32, y: i32, max_dist: i32) -> Option<usize> {
        let gs = self.grid_size;
        self.food
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                wrap_delta(x, f.x, gs).abs() + wrap_delta(y, f.y, gs).abs() <= max_dist
            })
            .min_by_key(|(_, f)| wrap_delta(x, f.x, gs).abs() + wrap_delta(y, f.y, gs).abs())
            .map(|(i, _)| i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{Brain, DEFAULT_HIDDEN, INPUTS, MAX_GENOME_LEN};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // Test defaults matching old constants
    const TEST_GRID: i32 = 20;
    const TEST_FOOD: usize = 25;
    const TEST_VISION: f32 = 4.0;
    const TEST_SIGNAL_RANGE: f32 = 8.0;
    const TEST_PRED_SPEED: u32 = 2;
    const TEST_BASE_DRAIN: f32 = 0.0008;
    const TEST_NEURON_COST: f32 = 0.00002;
    const TEST_SIGNAL_COST: f32 = 0.0;

    fn minimal_world(prey_positions: &[(i32, i32)], predator: (i32, i32)) -> World {
        let prey: Vec<Prey> = prey_positions
            .iter()
            .map(|&(x, y)| Prey {
                x,
                y,
                energy: 1.0,
                alive: true,
                brain: Brain::zero(),
                ticks_alive: 0,
                food_eaten: 0,
                actions_with_signal: [[0; 5]; 2],
                actions_without_signal: [[0; 5]; 2],
                silence_onset_actions: [[0; 5]; 2],
                had_signal_prev_tick: false,
            })
            .collect();
        let prey_count = prey.len();
        World {
            prey,
            predators: vec![Predator {
                x: predator.0,
                y: predator.1,
            }],
            food: Vec::new(),
            signals: Vec::new(),
            tick: 0,
            signals_emitted: 0,
            signal_events: Vec::new(),
            ticks_near_predator: 0,
            total_prey_ticks: 0,
            receiver_counts: [[[0u32; 5]; 2]; 4],
            signals_per_tick: Vec::new(),
            min_pred_dist_per_tick: Vec::new(),
            no_signals: true,
            prey_grid: CellGrid::new(TEST_GRID),
            food_grid: CellGrid::new(TEST_GRID),
            shuffled_indices: (0..prey_count).collect(),
            prey_positions: Vec::with_capacity(prey_count),
            grid_size: TEST_GRID,
            food_count: TEST_FOOD,
            prey_vision_range: TEST_VISION,
            signal_range: TEST_SIGNAL_RANGE,
            predator_speed: TEST_PRED_SPEED,
            base_drain: TEST_BASE_DRAIN,
            neuron_cost: TEST_NEURON_COST,
            signal_cost: TEST_SIGNAL_COST,
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

    // --- Predator movement ---

    #[test]
    fn predator_moves_toward_nearest_prey() {
        let mut world = minimal_world(&[(10, 5)], (5, 5));
        world.rebuild_prey_grid();

        world.move_predators();

        // Predator at (5,5), prey at (10,5): dx=5, dy=0. Should move +x 2 times.
        assert_eq!(world.predators[0].x, 7);
        assert_eq!(world.predators[0].y, 5);
    }

    #[test]
    fn predator_chases_through_wrap_boundary() {
        let mut world = minimal_world(&[(18, 10)], (1, 10));
        world.rebuild_prey_grid();

        world.move_predators();

        assert_eq!(world.predators[0].x, 19);
        assert_eq!(world.predators[0].y, 10);
    }

    #[test]
    fn predator_always_chases_nearest() {
        // 3+ prey nearby - predator should still chase, never move randomly
        let px = 10;
        let py = 10;
        let mut world = minimal_world(
            &[(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)],
            (px, py),
        );
        world.rebuild_prey_grid();

        world.move_predators();

        // Predator should have moved deterministically toward nearest prey (dist=1),
        // not randomly. Since all 4 are equidistant, it picks whichever is first in
        // iteration order, but it definitely moves purposefully.
        let pred = &world.predators[0];
        let moved_dist = wrap_dist_sq(px, py, pred.x, pred.y, TEST_GRID).sqrt();
        assert!(moved_dist > 0.0, "Predator should have moved");
        assert!(
            moved_dist <= TEST_PRED_SPEED as f32,
            "Predator moved too far"
        );
    }

    // --- Predator kill ---

    #[test]
    fn predator_kills_prey_on_same_cell() {
        let mut world = minimal_world(&[(5, 5)], (5, 5));

        world.predator_kill();

        assert!(!world.prey[0].alive);
    }

    #[test]
    fn predator_does_not_kill_adjacent_prey() {
        let mut world = minimal_world(&[(5, 6)], (5, 5));

        world.predator_kill();

        assert!(world.prey[0].alive);
    }

    #[test]
    fn predator_kill_marks_dead() {
        let mut world = minimal_world(&[(3, 3), (3, 3), (7, 7)], (3, 3));

        world.predator_kill();

        assert!(!world.prey[0].alive);
        assert!(!world.prey[1].alive);
        assert!(world.prey[2].alive);
    }

    // --- Multiple predators ---

    #[test]
    fn multiple_predators_chase_independently() {
        let prey = vec![
            Prey {
                x: 0,
                y: 0,
                energy: 1.0,
                alive: true,
                brain: Brain::zero(),
                ticks_alive: 0,
                food_eaten: 0,
                actions_with_signal: [[0; 5]; 2],
                actions_without_signal: [[0; 5]; 2],
                silence_onset_actions: [[0; 5]; 2],
                had_signal_prev_tick: false,
            },
            Prey {
                x: 19,
                y: 19,
                energy: 1.0,
                alive: true,
                brain: Brain::zero(),
                ticks_alive: 0,
                food_eaten: 0,
                actions_with_signal: [[0; 5]; 2],
                actions_without_signal: [[0; 5]; 2],
                silence_onset_actions: [[0; 5]; 2],
                had_signal_prev_tick: false,
            },
        ];
        let prey_count = prey.len();
        let mut world = World {
            prey,
            predators: vec![
                Predator { x: 3, y: 0 },   // near prey 0
                Predator { x: 16, y: 19 }, // near prey 1
            ],
            food: Vec::new(),
            signals: Vec::new(),
            tick: 0,
            signals_emitted: 0,
            signal_events: Vec::new(),
            ticks_near_predator: 0,
            total_prey_ticks: 0,
            receiver_counts: [[[0u32; 5]; 2]; 4],
            signals_per_tick: Vec::new(),
            min_pred_dist_per_tick: Vec::new(),
            no_signals: true,
            prey_grid: CellGrid::new(TEST_GRID),
            food_grid: CellGrid::new(TEST_GRID),
            shuffled_indices: (0..prey_count).collect(),
            prey_positions: Vec::with_capacity(prey_count),
            grid_size: TEST_GRID,
            food_count: TEST_FOOD,
            prey_vision_range: TEST_VISION,
            signal_range: TEST_SIGNAL_RANGE,
            predator_speed: TEST_PRED_SPEED,
            base_drain: TEST_BASE_DRAIN,
            neuron_cost: TEST_NEURON_COST,
            signal_cost: TEST_SIGNAL_COST,
        };
        world.rebuild_prey_grid();

        world.move_predators();

        // Predator 0 should move toward prey at (0,0): wrap_delta(3,0,20) = -3
        assert!(world.predators[0].x < 3, "Pred 0 should move toward (0,0)");
        // Predator 1 should move toward prey at (19,19): wrap_delta(16,19,20) = 3
        assert!(
            world.predators[1].x > 16,
            "Pred 1 should move toward (19,19)"
        );
    }

    // --- new_with_positions ---

    #[test]
    fn new_with_positions_places_correctly() {
        let agents = vec![
            Agent {
                brain: Brain {
                    weights: [0.1; MAX_GENOME_LEN],
                    hidden_size: DEFAULT_HIDDEN,
                },
                x: 3,
                y: 7,
            },
            Agent {
                brain: Brain {
                    weights: [0.2; MAX_GENOME_LEN],
                    hidden_size: DEFAULT_HIDDEN,
                },
                x: 15,
                y: 2,
            },
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let world = World::new_with_positions(
            &agents,
            1,
            &mut rng,
            false,
            TEST_GRID,
            TEST_FOOD,
            TEST_VISION,
            TEST_SIGNAL_RANGE,
            TEST_PRED_SPEED,
            TEST_BASE_DRAIN,
            TEST_NEURON_COST,
            TEST_SIGNAL_COST,
        );

        assert_eq!(world.prey[0].x, 3);
        assert_eq!(world.prey[0].y, 7);
        assert_eq!(world.prey[1].x, 15);
        assert_eq!(world.prey[1].y, 2);
        assert_eq!(world.predators.len(), 1);
    }

    // --- Energy mechanics ---

    #[test]
    fn energy_drains_per_tick() {
        let mut world = minimal_world(&[(0, 0)], (15, 15));
        world.food.push(Food { x: 10, y: 10 });
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let before = world.prey[0].energy;
        world.step(&mut rng);
        let after = world.prey[0].energy;

        let expected_drain = TEST_BASE_DRAIN + DEFAULT_HIDDEN as f32 * TEST_NEURON_COST;
        assert!((before - after - expected_drain).abs() < 1e-6);
    }

    #[test]
    fn food_consumption_restores_energy() {
        let mut world = minimal_world(&[(5, 5)], (15, 15));
        world.prey[0].energy = 0.5;
        world.food.push(Food { x: 5, y: 5 });
        world.food_grid.insert(5, 5, 0);

        let mut outputs = [0.0_f32; crate::brain::OUTPUTS];
        outputs[4] = 1.0;
        let inputs = [0.0_f32; INPUTS];
        world.apply_outputs(0, &outputs, &inputs, f32::MAX);

        assert!((world.prey[0].energy - 0.8).abs() < 1e-6);
    }

    #[test]
    fn energy_caps_at_one() {
        let mut world = minimal_world(&[(5, 5)], (15, 15));
        world.prey[0].energy = 0.9;
        world.food.push(Food { x: 5, y: 5 });
        world.food_grid.insert(5, 5, 0);

        let mut outputs = [0.0_f32; crate::brain::OUTPUTS];
        outputs[4] = 1.0;
        let inputs = [0.0_f32; INPUTS];
        world.apply_outputs(0, &outputs, &inputs, f32::MAX);

        assert!((world.prey[0].energy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn energy_death_at_zero() {
        let mut world = minimal_world(&[(0, 0)], (15, 15));
        let drain = TEST_BASE_DRAIN + DEFAULT_HIDDEN as f32 * TEST_NEURON_COST;
        world.prey[0].energy = drain * 0.5;
        world.food.push(Food { x: 10, y: 10 });
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.step(&mut rng);

        assert!(!world.prey[0].alive);
    }

    // --- Food respawn ---

    #[test]
    fn food_respawns_when_below_half() {
        let mut world = minimal_world(&[(0, 0)], (15, 15));
        for x in 5..16 {
            world.food.push(Food { x, y: 5 });
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.step(&mut rng);

        assert_eq!(world.food.len(), TEST_FOOD);
    }

    #[test]
    fn food_does_not_respawn_above_half() {
        let mut world = minimal_world(&[(0, 0)], (15, 15));
        for x in 5..18 {
            world.food.push(Food { x, y: 5 });
        }
        let initial_count = world.food.len();
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.step(&mut rng);

        assert!(world.food.len() <= initial_count);
    }

    // --- Input building / vision gating ---

    #[test]
    fn predator_inputs_zeroed_when_out_of_range() {
        let mut world = minimal_world(&[(0, 0)], (15, 15));
        world.food.push(Food { x: 10, y: 10 });

        let inputs = world.build_inputs(0);

        assert!((inputs[0]).abs() < 1e-6);
        assert!((inputs[1]).abs() < 1e-6);
        assert!((inputs[2]).abs() < 1e-6);
    }

    #[test]
    fn predator_inputs_populated_when_in_range() {
        let mut world = minimal_world(&[(10, 10)], (12, 10));
        world.food.push(Food { x: 5, y: 5 });

        let inputs = world.build_inputs(0);

        assert!((inputs[0] - 0.1).abs() < 1e-6);
        assert!((inputs[1]).abs() < 1e-6);
        assert!((inputs[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn predator_inputs_at_vision_boundary() {
        let mut world = minimal_world(&[(0, 0)], (4, 0));
        world.food.push(Food { x: 10, y: 10 });

        let inputs = world.build_inputs(0);

        assert!(inputs[0] > 0.0 || inputs[2] > 0.0);
    }

    // --- Per-prey receiver tracking ---

    #[test]
    fn per_prey_tracking_accumulates_with_and_without_signal() {
        let mut world = minimal_world(&[(5, 5)], (15, 15));
        world.food.push(Food { x: 10, y: 10 });
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
        let mut world = minimal_world(&[(0, 0)], (15, 15));
        world.food.push(Food { x: 5, y: 5 });

        let inputs = world.build_inputs(0);

        assert_eq!(inputs.len(), INPUTS);
    }

    #[test]
    fn larger_brain_drains_more() {
        use crate::brain::MAX_HIDDEN;
        let mut world = minimal_world(&[(0, 0), (5, 5)], (15, 15));
        world.food.push(Food { x: 10, y: 10 });
        world.prey[0].brain.hidden_size = DEFAULT_HIDDEN;
        world.prey[1].brain.hidden_size = MAX_HIDDEN;

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
        use crate::brain::MIN_HIDDEN;
        let mut world = minimal_world(&[(0, 0), (5, 5)], (15, 15));
        world.food.push(Food { x: 10, y: 10 });
        world.prey[0].brain.hidden_size = DEFAULT_HIDDEN;
        world.prey[1].brain.hidden_size = MIN_HIDDEN;

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
        // At hidden_size=6: BASE_DRAIN(0.0008) + 6 * NEURON_COST(0.00002) = 0.00092
        let drain = TEST_BASE_DRAIN + DEFAULT_HIDDEN as f32 * TEST_NEURON_COST;
        assert!(
            (drain - 0.00092).abs() < 1e-6,
            "Default drain should be 0.00092 with cheap neurons"
        );
    }
}
