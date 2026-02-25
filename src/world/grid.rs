use crate::agent::predator::Predator;
use crate::agent::prey::Prey;
use crate::rng::SeededRng;
use crate::signal::message::ActiveSignal;
use crate::world::food::Food;
use crate::world::terrain::Terrain;

pub struct World {
    pub width: u32,
    pub height: u32,
    pub terrain: Vec<Terrain>,
    pub food: Vec<Option<Food>>,
    pub prey: Vec<Prey>,
    pub predators: Vec<Predator>,
    pub signals: Vec<ActiveSignal>,
    pub tick: u64,
    pub generation: u32,
    pub rng: SeededRng,
}

impl World {
    pub fn new(width: u32, height: u32, seed: u64) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            terrain: vec![Terrain::Open; size],
            food: vec![None; size],
            prey: Vec::new(),
            predators: Vec::new(),
            signals: Vec::new(),
            tick: 0,
            generation: 0,
            rng: SeededRng::new(seed),
        }
    }

    pub fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    pub fn terrain_at(&self, x: u32, y: u32) -> Terrain {
        self.terrain[self.idx(x, y)]
    }

    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height
    }
}
