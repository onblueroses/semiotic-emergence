use rand::seq::SliceRandom;
use rand::Rng;

use crate::brain::{Brain, INPUTS, OUTPUTS};
use crate::signal::{self, Signal, SIGNAL_THRESHOLD};

pub const GRID_SIZE: i32 = 20;
pub const FOOD_COUNT: usize = 25;
pub const PREY_VISION_RANGE: f32 = 4.0;
pub const CONFUSION_THRESHOLD: usize = 3;
pub const CONFUSION_RADIUS: f32 = 4.0;
pub const PREDATOR_SPEED: u32 = 3;
pub const ENERGY_DRAIN: f32 = 0.002;
pub const SIGNAL_COST: f32 = 0.01;

/// Wrap-aware signed delta: shortest path on a toroidal grid.
/// Returns value in (-size/2, size/2].
fn wrap_delta(a: i32, b: i32, size: i32) -> i32 {
    let d = b - a;
    if d > size / 2 {
        d - size
    } else if d < -(size / 2) {
        d + size
    } else {
        d
    }
}

fn wrap_dist_sq(ax: i32, ay: i32, bx: i32, by: i32) -> f32 {
    let dx = wrap_delta(ax, bx, GRID_SIZE) as f32;
    let dy = wrap_delta(ay, by, GRID_SIZE) as f32;
    dx * dx + dy * dy
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
}

#[derive(Clone, Debug)]
pub struct Predator {
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, Debug)]
pub struct Food {
    pub x: i32,
    pub y: i32,
}

pub struct SignalEvent {
    pub symbol: u8,
    pub predator_dist: f32,
}

pub struct World {
    pub prey: Vec<Prey>,
    pub predator: Predator,
    pub food: Vec<Food>,
    pub signals: Vec<Signal>,
    pub tick: u32,
    pub signals_emitted: u32,
    pub signal_events: Vec<SignalEvent>,
    pub ticks_near_predator: u32,
    pub total_prey_ticks: u32,
    pub confusion_ticks: u32,
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
}

impl World {
    pub fn new(brains: Vec<Brain>, rng: &mut impl Rng, no_signals: bool) -> Self {
        let prey = brains
            .into_iter()
            .map(|brain| Prey {
                x: rng.gen_range(0..GRID_SIZE),
                y: rng.gen_range(0..GRID_SIZE),
                energy: 1.0,
                alive: true,
                brain,
                ticks_alive: 0,
                food_eaten: 0,
            })
            .collect();

        let predator = Predator {
            x: rng.gen_range(0..GRID_SIZE),
            y: rng.gen_range(0..GRID_SIZE),
        };

        let food = (0..FOOD_COUNT)
            .map(|_| Food {
                x: rng.gen_range(0..GRID_SIZE),
                y: rng.gen_range(0..GRID_SIZE),
            })
            .collect();

        Self {
            prey,
            predator,
            food,
            signals: Vec::new(),
            tick: 0,
            signals_emitted: 0,
            signal_events: Vec::new(),
            ticks_near_predator: 0,
            total_prey_ticks: 0,
            confusion_ticks: 0,
            receiver_counts: [[[0u32; 5]; 2]; 4],
            signals_per_tick: Vec::new(),
            min_pred_dist_per_tick: Vec::new(),
            no_signals,
        }
    }

    pub fn any_alive(&self) -> bool {
        self.prey.iter().any(|p| p.alive)
    }

    pub fn step(&mut self, rng: &mut impl Rng) {
        self.tick += 1;

        let signals_before = self.signals_emitted;

        self.signals
            .retain(|s| self.tick.saturating_sub(s.tick_emitted) <= 4);

        // Track minimum predator-to-alive-prey distance this tick
        let min_pred_dist = self
            .prey
            .iter()
            .filter(|p| p.alive)
            .map(|p| wrap_dist_sq(p.x, p.y, self.predator.x, self.predator.y).sqrt())
            .fold(f32::MAX, f32::min);
        self.min_pred_dist_per_tick.push(min_pred_dist);

        // Shuffle prey processing order to prevent index bias
        let mut order: Vec<usize> = (0..self.prey.len()).collect();
        order.shuffle(rng);

        for &i in &order {
            if !self.prey[i].alive {
                continue;
            }

            // Metabolism: drain energy each tick
            self.prey[i].energy -= ENERGY_DRAIN;
            if self.prey[i].energy <= 0.0 {
                self.prey[i].alive = false;
                continue;
            }

            // Track proximity stats for iconicity baseline
            let pdist = wrap_dist_sq(
                self.prey[i].x,
                self.prey[i].y,
                self.predator.x,
                self.predator.y,
            )
            .sqrt();
            self.total_prey_ticks += 1;
            if pdist <= PREY_VISION_RANGE {
                self.ticks_near_predator += 1;
            }

            let inputs = self.build_inputs(i);
            let outputs = self.prey[i].brain.forward(&inputs);

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

            self.apply_outputs(i, &outputs);

            self.prey[i].ticks_alive += 1;
        }

        self.move_predator(rng);
        // Kill check runs once after all 3 movement sub-steps, not after each.
        // The predator can pass through a prey's cell without killing it. This is
        // intentional: the overshoot makes the confusion effect more impactful and
        // creates stronger evolutionary pressure for grouping behavior.
        self.predator_kill();

        if self.food.len() < FOOD_COUNT / 2 {
            while self.food.len() < FOOD_COUNT {
                self.food.push(Food {
                    x: rng.gen_range(0..GRID_SIZE),
                    y: rng.gen_range(0..GRID_SIZE),
                });
            }
        }

        self.signals_per_tick
            .push(self.signals_emitted - signals_before);
    }

    fn build_inputs(&self, prey_idx: usize) -> [f32; INPUTS] {
        let p = &self.prey[prey_idx];
        let mut inp = [0.0_f32; INPUTS];
        let gs = GRID_SIZE as f32;

        // 0-2: Predator relative dx, dy, distance (gated by vision range)
        let pdx = wrap_delta(p.x, self.predator.x, GRID_SIZE) as f32;
        let pdy = wrap_delta(p.y, self.predator.y, GRID_SIZE) as f32;
        let pdist = (pdx * pdx + pdy * pdy).sqrt();
        if pdist <= PREY_VISION_RANGE {
            inp[0] = pdx / gs;
            inp[1] = pdy / gs;
            inp[2] = (pdist / PREY_VISION_RANGE).min(1.0);
        }

        // 3-4: Nearest food dx, dy
        if let Some(f) = self.nearest_food(p.x, p.y) {
            inp[3] = wrap_delta(p.x, f.x, GRID_SIZE) as f32 / gs;
            inp[4] = wrap_delta(p.y, f.y, GRID_SIZE) as f32 / gs;
        }

        // 5: Nearest ally distance
        let mut min_ally = f32::MAX;
        for (j, other) in self.prey.iter().enumerate() {
            if j == prey_idx || !other.alive {
                continue;
            }
            let d = wrap_dist_sq(p.x, p.y, other.x, other.y).sqrt();
            if d < min_ally {
                min_ally = d;
            }
        }
        inp[5] = if min_ally < f32::MAX {
            (min_ally / gs).min(1.0)
        } else {
            1.0
        };

        // 6-14: Incoming signals (strength + direction per symbol)
        let sig = signal::receive_detailed(&self.signals, p.x, p.y, self.tick, gs);
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

    fn apply_outputs(&mut self, prey_idx: usize, outputs: &[f32; OUTPUTS]) {
        let action = outputs[..5]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);

        match action {
            0 => self.prey[prey_idx].y = (self.prey[prey_idx].y - 1).rem_euclid(GRID_SIZE),
            1 => self.prey[prey_idx].y = (self.prey[prey_idx].y + 1).rem_euclid(GRID_SIZE),
            2 => self.prey[prey_idx].x = (self.prey[prey_idx].x + 1).rem_euclid(GRID_SIZE),
            3 => self.prey[prey_idx].x = (self.prey[prey_idx].x - 1).rem_euclid(GRID_SIZE),
            4 => {
                let px = self.prey[prey_idx].x;
                let py = self.prey[prey_idx].y;
                if let Some(fi) = self.nearest_food_idx(px, py, 1) {
                    self.food.remove(fi);
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
        if !self.no_signals && self.prey[prey_idx].energy > SIGNAL_COST {
            if let Some(symbol) = signal::maybe_emit(outputs.as_slice(), SIGNAL_THRESHOLD) {
                self.prey[prey_idx].energy -= SIGNAL_COST;
                let predator_dist = wrap_dist_sq(px, py, self.predator.x, self.predator.y).sqrt();
                self.signal_events.push(SignalEvent {
                    symbol,
                    predator_dist,
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

    fn move_predator(&mut self, rng: &mut impl Rng) {
        for _ in 0..PREDATOR_SPEED {
            // Confusion effect: 3+ alive prey within radius -> predator moves randomly
            let nearby = self
                .prey
                .iter()
                .filter(|p| {
                    p.alive
                        && wrap_dist_sq(p.x, p.y, self.predator.x, self.predator.y).sqrt()
                            <= CONFUSION_RADIUS
                })
                .count();

            if nearby >= CONFUSION_THRESHOLD {
                self.confusion_ticks += 1;
                match rng.gen_range(0..4) {
                    0 => self.predator.y = (self.predator.y - 1).rem_euclid(GRID_SIZE),
                    1 => self.predator.y = (self.predator.y + 1).rem_euclid(GRID_SIZE),
                    2 => self.predator.x = (self.predator.x + 1).rem_euclid(GRID_SIZE),
                    _ => self.predator.x = (self.predator.x - 1).rem_euclid(GRID_SIZE),
                }
                continue;
            }

            let mut nearest: Option<(i32, i32, f32)> = None;
            for p in &self.prey {
                if !p.alive {
                    continue;
                }
                let d = wrap_dist_sq(self.predator.x, self.predator.y, p.x, p.y);
                if nearest.is_none() || d < nearest.unwrap_or((0, 0, f32::MAX)).2 {
                    nearest = Some((p.x, p.y, d));
                }
            }

            if let Some((tx, ty, _)) = nearest {
                let dx = wrap_delta(self.predator.x, tx, GRID_SIZE);
                let dy = wrap_delta(self.predator.y, ty, GRID_SIZE);
                if dx.abs() >= dy.abs() {
                    self.predator.x += dx.signum();
                } else {
                    self.predator.y += dy.signum();
                }
                self.predator.x = self.predator.x.rem_euclid(GRID_SIZE);
                self.predator.y = self.predator.y.rem_euclid(GRID_SIZE);
            }
        }
    }

    fn predator_kill(&mut self) {
        for p in &mut self.prey {
            if !p.alive {
                continue;
            }
            let dx = wrap_delta(self.predator.x, p.x, GRID_SIZE).abs();
            let dy = wrap_delta(self.predator.y, p.y, GRID_SIZE).abs();
            if dx == 0 && dy == 0 {
                p.alive = false;
            }
        }
    }

    fn nearest_food(&self, x: i32, y: i32) -> Option<&Food> {
        self.food.iter().min_by_key(|f| {
            wrap_delta(x, f.x, GRID_SIZE).abs() + wrap_delta(y, f.y, GRID_SIZE).abs()
        })
    }

    fn nearest_food_idx(&self, x: i32, y: i32, max_dist: i32) -> Option<usize> {
        self.food
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                wrap_delta(x, f.x, GRID_SIZE).abs() + wrap_delta(y, f.y, GRID_SIZE).abs()
                    <= max_dist
            })
            .min_by_key(|(_, f)| {
                wrap_delta(x, f.x, GRID_SIZE).abs() + wrap_delta(y, f.y, GRID_SIZE).abs()
            })
            .map(|(i, _)| i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{Brain, GENOME_LEN, INPUTS};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn zero_brain() -> Brain {
        Brain {
            weights: [0.0; GENOME_LEN],
        }
    }

    fn minimal_world(prey_positions: &[(i32, i32)], predator: (i32, i32)) -> World {
        let prey = prey_positions
            .iter()
            .map(|&(x, y)| Prey {
                x,
                y,
                energy: 1.0,
                alive: true,
                brain: zero_brain(),
                ticks_alive: 0,
                food_eaten: 0,
            })
            .collect();
        World {
            prey,
            predator: Predator {
                x: predator.0,
                y: predator.1,
            },
            food: Vec::new(),
            signals: Vec::new(),
            tick: 0,
            signals_emitted: 0,
            signal_events: Vec::new(),
            ticks_near_predator: 0,
            total_prey_ticks: 0,
            confusion_ticks: 0,
            receiver_counts: [[[0u32; 5]; 2]; 4],
            signals_per_tick: Vec::new(),
            min_pred_dist_per_tick: Vec::new(),
            no_signals: true,
        }
    }

    // --- Toroidal wrapping ---

    #[test]
    fn wrap_delta_no_wrap() {
        assert_eq!(wrap_delta(3, 7, GRID_SIZE), 4);
        assert_eq!(wrap_delta(7, 3, GRID_SIZE), -4);
    }

    #[test]
    fn wrap_delta_across_boundary() {
        // 18 -> 1: naive delta = -17, but wrapping = +3 (shorter path)
        assert_eq!(wrap_delta(18, 1, GRID_SIZE), 3);
        // 1 -> 18: naive delta = 17, but wrapping = -3
        assert_eq!(wrap_delta(1, 18, GRID_SIZE), -3);
    }

    #[test]
    fn wrap_delta_half_grid() {
        // Exactly half the grid: delta = 10, which is size/2, lands in the else branch
        assert_eq!(wrap_delta(0, 10, GRID_SIZE), 10);
        // -10 is exactly -(size/2), also the else branch
        assert_eq!(wrap_delta(10, 0, GRID_SIZE), -10);
    }

    #[test]
    fn wrap_dist_sq_same_cell_is_zero() {
        assert!((wrap_dist_sq(5, 5, 5, 5) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn wrap_dist_sq_across_boundary() {
        // (19, 0) to (1, 0): wrap delta x = 2
        let d = wrap_dist_sq(19, 0, 1, 0);
        assert!((d - 4.0).abs() < 1e-6); // 2^2 = 4
    }

    // --- Predator movement ---

    #[test]
    fn predator_moves_toward_nearest_prey() {
        let mut world = minimal_world(&[(10, 5)], (5, 5));
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.move_predator(&mut rng);

        // Predator at (5,5), prey at (10,5): dx=5, dy=0. Should move +x 3 times.
        assert_eq!(world.predator.x, 8);
        assert_eq!(world.predator.y, 5);
    }

    #[test]
    fn predator_chases_through_wrap_boundary() {
        // Predator at (1,10), prey at (18,10): wrap dx = -3, so predator moves left
        let mut world = minimal_world(&[(18, 10)], (1, 10));
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.move_predator(&mut rng);

        // 3 steps of x-1 wrapping: 1 -> 0 -> 19 -> 18
        assert_eq!(world.predator.x, 18);
        assert_eq!(world.predator.y, 10);
    }

    #[test]
    fn predator_confused_by_three_nearby_prey() {
        // Place 3 prey within CONFUSION_RADIUS of predator
        let px = 10;
        let py = 10;
        let mut world = minimal_world(&[(px + 1, py), (px - 1, py), (px, py + 1)], (px, py));
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        world.move_predator(&mut rng);

        // All sub-steps should be confused (3 prey within radius 4.0 of predator)
        assert_eq!(world.confusion_ticks, PREDATOR_SPEED);
    }

    #[test]
    fn predator_not_confused_by_two_nearby_prey() {
        let px = 10;
        let py = 10;
        let mut world = minimal_world(&[(px + 1, py), (px - 1, py)], (px, py));
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.move_predator(&mut rng);

        assert_eq!(world.confusion_ticks, 0);
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

    // --- Energy mechanics ---

    #[test]
    fn energy_drains_per_tick() {
        let mut world = minimal_world(&[(0, 0)], (15, 15));
        // Place food far away so prey can't eat
        world.food.push(Food { x: 10, y: 10 });
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let before = world.prey[0].energy;
        world.step(&mut rng);
        let after = world.prey[0].energy;

        assert!((before - after - ENERGY_DRAIN).abs() < 1e-6);
    }

    #[test]
    fn food_consumption_restores_energy() {
        let mut world = minimal_world(&[(5, 5)], (15, 15));
        world.prey[0].energy = 0.5;
        // Place food on the prey's cell
        world.food.push(Food { x: 5, y: 5 });

        // Directly test apply_outputs with eat action (output 4 is highest)
        let mut outputs = [0.0_f32; crate::brain::OUTPUTS];
        outputs[4] = 1.0; // eat action
        world.apply_outputs(0, &outputs);

        // Energy should increase by 0.3 (minus nothing - apply_outputs doesn't drain)
        assert!((world.prey[0].energy - 0.8).abs() < 1e-6);
    }

    #[test]
    fn energy_caps_at_one() {
        let mut world = minimal_world(&[(5, 5)], (15, 15));
        world.prey[0].energy = 0.9;
        world.food.push(Food { x: 5, y: 5 });

        let mut outputs = [0.0_f32; crate::brain::OUTPUTS];
        outputs[4] = 1.0;
        world.apply_outputs(0, &outputs);

        assert!((world.prey[0].energy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn energy_death_at_zero() {
        let mut world = minimal_world(&[(0, 0)], (15, 15));
        world.prey[0].energy = ENERGY_DRAIN * 0.5; // less than one tick's drain
        world.food.push(Food { x: 10, y: 10 });
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.step(&mut rng);

        assert!(!world.prey[0].alive);
    }

    // --- Food respawn ---

    #[test]
    fn food_respawns_when_below_half() {
        let mut world = minimal_world(&[(0, 0)], (15, 15));
        // Start with fewer than FOOD_COUNT/2 food items (FOOD_COUNT=25, so half=12, add 11)
        for x in 5..16 {
            world.food.push(Food { x, y: 5 });
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.step(&mut rng);

        assert_eq!(world.food.len(), FOOD_COUNT);
    }

    #[test]
    fn food_does_not_respawn_above_half() {
        let mut world = minimal_world(&[(0, 0)], (15, 15));
        // Start with exactly FOOD_COUNT/2 + 1 items (above threshold: 13 items)
        for x in 5..18 {
            world.food.push(Food { x, y: 5 });
        }
        let initial_count = world.food.len();
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        world.step(&mut rng);

        // Prey might eat one, but food won't respawn because count >= FOOD_COUNT/2
        assert!(world.food.len() <= initial_count);
    }

    // --- Input building / vision gating ---

    #[test]
    fn predator_inputs_zeroed_when_out_of_range() {
        let mut world = minimal_world(&[(0, 0)], (15, 15));
        // Distance: wrap_dist from (0,0) to (15,15) = sqrt(5^2 + 5^2) = ~7.07 > PREY_VISION_RANGE (4.0)
        world.food.push(Food { x: 10, y: 10 });

        let inputs = world.build_inputs(0);

        // Predator inputs (0-2) should all be zero
        assert!((inputs[0]).abs() < 1e-6);
        assert!((inputs[1]).abs() < 1e-6);
        assert!((inputs[2]).abs() < 1e-6);
    }

    #[test]
    fn predator_inputs_populated_when_in_range() {
        // Prey at (10,10), predator at (12,10): distance = 2.0 < 4.0
        let mut world = minimal_world(&[(10, 10)], (12, 10));
        world.food.push(Food { x: 5, y: 5 });

        let inputs = world.build_inputs(0);

        // dx = wrap_delta(10, 12, 20) = 2, normalized by grid: 2/20 = 0.1
        assert!((inputs[0] - 0.1).abs() < 1e-6);
        // dy = 0
        assert!((inputs[1]).abs() < 1e-6);
        // distance = 2.0, normalized: 2.0/4.0 = 0.5
        assert!((inputs[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn predator_inputs_at_vision_boundary() {
        // Place predator exactly at vision range: distance = 4.0
        // Prey at (0,0), predator at (4,0): distance = 4.0 <= PREY_VISION_RANGE
        let mut world = minimal_world(&[(0, 0)], (4, 0));
        world.food.push(Food { x: 10, y: 10 });

        let inputs = world.build_inputs(0);

        // Should be populated (distance <= range, not strictly <)
        assert!(inputs[0] > 0.0 || inputs[2] > 0.0);
    }

    #[test]
    fn build_inputs_returns_correct_size() {
        let mut world = minimal_world(&[(0, 0)], (15, 15));
        world.food.push(Food { x: 5, y: 5 });

        let inputs = world.build_inputs(0);

        assert_eq!(inputs.len(), INPUTS);
    }
}
