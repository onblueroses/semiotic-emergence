use crate::agent::predator::Predator;
use crate::agent::prey::Prey;
use crate::rng::SeededRng;
use crate::signal::message::ActiveSignal;
use crate::world::food::Food;
use crate::world::terrain::Terrain;

#[expect(
    dead_code,
    reason = "constructed in simulation main loop; remove when sim loop is implemented"
)]
pub(crate) struct World {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) terrain: Vec<Terrain>,
    pub(crate) food: Vec<Option<Food>>,
    pub(crate) prey: Vec<Prey>,
    pub(crate) predators: Vec<Predator>,
    pub(crate) signals: Vec<ActiveSignal>,
    pub(crate) tick: u64,
    pub(crate) generation: u32,
    pub(crate) rng: SeededRng,
}

#[expect(
    dead_code,
    reason = "constructed in simulation main loop; remove when sim loop is implemented"
)]
impl World {
    pub(crate) fn new(width: u32, height: u32, seed: u64) -> Self {
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

    pub(crate) fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    pub(crate) fn terrain_at(&self, x: u32, y: u32) -> Terrain {
        self.terrain[self.idx(x, y)]
    }

    pub(crate) fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height
    }
}
