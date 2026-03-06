use rand::seq::SliceRandom;
use rand::Rng;

use crate::brain::{Brain, INPUTS, OUTPUTS};
use crate::signal::{self, Signal, SIGNAL_THRESHOLD};

pub const GRID_SIZE: i32 = 20;
pub const FOOD_COUNT: usize = 25;
pub const PREY_VISION_RANGE: f32 = 4.0;
pub const CONFUSION_THRESHOLD: usize = 3;
pub const CONFUSION_RADIUS: f32 = 4.0;
pub const PREDATOR_SPEED: u32 = 2;
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
}

impl World {
    pub fn new(brains: Vec<Brain>, rng: &mut impl Rng) -> Self {
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
        }
    }

    pub fn any_alive(&self) -> bool {
        self.prey.iter().any(|p| p.alive)
    }

    pub fn step(&mut self, rng: &mut impl Rng) {
        self.tick += 1;

        self.signals
            .retain(|s| self.tick.saturating_sub(s.tick_emitted) <= 4);

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
            self.apply_outputs(i, &outputs);

            self.prey[i].ticks_alive += 1;
        }

        self.move_predator(rng);
        self.predator_kill();

        if self.food.len() < FOOD_COUNT / 2 {
            while self.food.len() < FOOD_COUNT {
                self.food.push(Food {
                    x: rng.gen_range(0..GRID_SIZE),
                    y: rng.gen_range(0..GRID_SIZE),
                });
            }
        }
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
        let px = self.prey[prey_idx].x;
        let py = self.prey[prey_idx].y;
        if self.prey[prey_idx].energy > SIGNAL_COST {
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
            if dx <= 1 && dy <= 1 {
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
