#[derive(Clone, Debug)]
pub(crate) struct Food {
    pub(crate) energy: f32,
    pub(crate) regrow_timer: u32,
}

impl Food {
    pub(crate) fn new(energy: f32) -> Self {
        Self {
            energy,
            regrow_timer: 0,
        }
    }
}

use crate::rng::SeededRng;
use crate::world::terrain::Terrain;

/// Place food items on the terrain grid.
///
/// Food spawns preferentially on Open/Grass (80% weight), Bush (15%), Tree/Rock (5%).
/// Never on Water. Returns `Vec<Option<Food>>` matching terrain length.
pub(crate) fn place_food(
    terrain: &[Terrain],
    density: f32,
    energy: f32,
    rng: &mut SeededRng,
) -> Vec<Option<Food>> {
    let target_count = (terrain.len() as f32 * density) as usize;
    let mut food: Vec<Option<Food>> = vec![None; terrain.len()];

    // Build list of candidate cells with weights
    let mut candidates: Vec<usize> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();

    for (i, t) in terrain.iter().enumerate() {
        let w = match t {
            Terrain::Open | Terrain::Grass => 0.80,
            Terrain::Bush => 0.15,
            Terrain::Tree | Terrain::Rock => 0.05,
            Terrain::Water => 0.0,
        };
        if w > 0.0 {
            candidates.push(i);
            weights.push(w);
        }
    }

    // Weighted random placement
    let mut placed = 0_usize;
    let mut attempts = 0_usize;
    let max_attempts = target_count * 10;

    while placed < target_count && attempts < max_attempts && !candidates.is_empty() {
        // Pick a random candidate weighted by terrain preference
        let idx = rng.gen_range(0..candidates.len());
        let cell = candidates[idx];

        if food[cell].is_none() && rng.gen_f32() < weights[idx] {
            food[cell] = Some(Food::new(energy));
            placed += 1;
        }
        attempts += 1;
    }

    // If weighted placement fell short, fill remaining randomly
    if placed < target_count {
        let mut remaining: Vec<usize> = candidates
            .into_iter()
            .filter(|&i| food[i].is_none())
            .collect();
        rng.shuffle(&mut remaining);
        for &cell in remaining.iter().take(target_count - placed) {
            food[cell] = Some(Food::new(energy));
        }
    }

    food
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SeededRng;

    #[test]
    fn place_food_correct_count() {
        let terrain = vec![Terrain::Open; 100];
        let mut rng = SeededRng::new(42);
        let food = place_food(&terrain, 0.15, 20.0, &mut rng);
        let count = food.iter().filter(|f| f.is_some()).count();
        assert_eq!(count, 15);
    }

    #[test]
    fn place_food_no_water() {
        let mut terrain = vec![Terrain::Open; 100];
        // Make half the grid water
        for t in terrain.iter_mut().take(50) {
            *t = Terrain::Water;
        }
        let mut rng = SeededRng::new(42);
        let food = place_food(&terrain, 0.15, 20.0, &mut rng);
        // No food on water cells (indices 0..50)
        for f in food.iter().take(50) {
            assert!(f.is_none());
        }
    }

    #[test]
    fn place_food_deterministic() {
        let terrain = vec![Terrain::Open; 200];
        let mut rng1 = SeededRng::new(42);
        let mut rng2 = SeededRng::new(42);
        let food1 = place_food(&terrain, 0.15, 20.0, &mut rng1);
        let food2 = place_food(&terrain, 0.15, 20.0, &mut rng2);
        for (a, b) in food1.iter().zip(food2.iter()) {
            assert_eq!(a.is_some(), b.is_some());
        }
    }
}
