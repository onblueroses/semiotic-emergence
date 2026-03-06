use rand::Rng;

use crate::brain::{Brain, INPUTS, OUTPUTS};
use crate::signal::{self, Signal, SIGNAL_THRESHOLD};

pub const GRID_SIZE: i32 = 20;
pub const FOOD_COUNT: usize = 15;

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

pub struct World {
    pub prey: Vec<Prey>,
    pub predator: Predator,
    pub food: Vec<Food>,
    pub signals: Vec<Signal>,
    pub tick: u32,
    pub signals_emitted: u32,
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
        }
    }

    pub fn any_alive(&self) -> bool {
        self.prey.iter().any(|p| p.alive)
    }

    pub fn step(&mut self, rng: &mut impl Rng) {
        self.tick += 1;

        // Prune old signals (older than range in ticks, say 4 ticks)
        self.signals
            .retain(|s| self.tick.saturating_sub(s.tick_emitted) <= 4);

        // Process each prey
        for i in 0..self.prey.len() {
            if !self.prey[i].alive {
                continue;
            }

            let inputs = self.build_inputs(i);
            let outputs = self.prey[i].brain.forward(&inputs);
            self.apply_outputs(i, &outputs);

            self.prey[i].ticks_alive += 1;
        }

        // Move predator toward nearest alive prey
        self.move_predator();

        // Kill adjacent prey
        self.predator_kill();

        // Respawn food if running low
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

        // 0-2: Predator relative dx, dy, distance
        let pdx = (self.predator.x - p.x) as f32;
        let pdy = (self.predator.y - p.y) as f32;
        let pdist = (pdx * pdx + pdy * pdy).sqrt();
        inp[0] = pdx / gs;
        inp[1] = pdy / gs;
        inp[2] = (pdist / gs).min(1.0);

        // 3-4: Nearest food dx, dy
        if let Some(f) = self.nearest_food(p.x, p.y) {
            inp[3] = (f.x - p.x) as f32 / gs;
            inp[4] = (f.y - p.y) as f32 / gs;
        }

        // 5: Nearest ally distance
        let mut min_ally = f32::MAX;
        for (j, other) in self.prey.iter().enumerate() {
            if j == prey_idx || !other.alive {
                continue;
            }
            let dx = (other.x - p.x) as f32;
            let dy = (other.y - p.y) as f32;
            let d = (dx * dx + dy * dy).sqrt();
            if d < min_ally {
                min_ally = d;
            }
        }
        inp[5] = if min_ally < f32::MAX {
            (min_ally / gs).min(1.0)
        } else {
            1.0
        };

        // 6-8: Incoming signal strengths
        let sig_strengths = signal::receive(&self.signals, p.x, p.y, self.tick);
        inp[6] = sig_strengths[0];
        inp[7] = sig_strengths[1];
        inp[8] = sig_strengths[2];

        // 9: Own energy
        inp[9] = p.energy.clamp(0.0, 1.0);

        inp
    }

    fn apply_outputs(&mut self, prey_idx: usize, outputs: &[f32; OUTPUTS]) {
        // Outputs 0-4: movement (N/S/E/W) + eat
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

        // Signal emission (outputs 5-7)
        let px = self.prey[prey_idx].x;
        let py = self.prey[prey_idx].y;
        if let Some(symbol) = signal::maybe_emit(outputs.as_slice(), SIGNAL_THRESHOLD) {
            self.signals.push(Signal {
                x: px,
                y: py,
                symbol,
                tick_emitted: self.tick,
            });
            self.signals_emitted += 1;
        }
    }

    fn move_predator(&mut self) {
        let mut nearest: Option<(i32, i32, f32)> = None;
        for p in &self.prey {
            if !p.alive {
                continue;
            }
            let dx = (p.x - self.predator.x) as f32;
            let dy = (p.y - self.predator.y) as f32;
            let d = dx * dx + dy * dy;
            if nearest.is_none() || d < nearest.unwrap_or((0, 0, f32::MAX)).2 {
                nearest = Some((p.x, p.y, d));
            }
        }

        if let Some((tx, ty, _)) = nearest {
            let dx = tx - self.predator.x;
            let dy = ty - self.predator.y;
            // Move 1 step toward target (prefer axis with larger delta)
            if dx.abs() >= dy.abs() {
                self.predator.x += dx.signum();
            } else {
                self.predator.y += dy.signum();
            }
            self.predator.x = self.predator.x.rem_euclid(GRID_SIZE);
            self.predator.y = self.predator.y.rem_euclid(GRID_SIZE);
        }
    }

    fn predator_kill(&mut self) {
        for p in &mut self.prey {
            if !p.alive {
                continue;
            }
            let dx = (p.x - self.predator.x).abs();
            let dy = (p.y - self.predator.y).abs();
            if dx <= 1 && dy <= 1 {
                p.alive = false;
            }
        }
    }

    fn nearest_food(&self, x: i32, y: i32) -> Option<&Food> {
        self.food
            .iter()
            .min_by_key(|f| (f.x - x).abs() + (f.y - y).abs())
    }

    fn nearest_food_idx(&self, x: i32, y: i32, max_dist: i32) -> Option<usize> {
        self.food
            .iter()
            .enumerate()
            .filter(|(_, f)| (f.x - x).abs() + (f.y - y).abs() <= max_dist)
            .min_by_key(|(_, f)| (f.x - x).abs() + (f.y - y).abs())
            .map(|(i, _)| i)
    }
}
