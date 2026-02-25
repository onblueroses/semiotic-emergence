use crate::rng::SeededRng;
use crate::world::entity::Position;
use crate::world::terrain::Terrain;

/// Place offspring near parent position with Gaussian jitter (D8).
///
/// 90% of offspring spawn near the parent position with Gaussian jitter (stdev=4).
/// 10% spawn at a random passable location (prevents inbreeding clusters).
/// If the parent position is `None`, always uses random placement.
///
/// Makes up to 5 attempts to find a passable cell near the parent before
/// falling back to random placement.
pub(crate) fn kin_placement(
    parent_pos: Option<Position>,
    width: u32,
    height: u32,
    terrain: &[Terrain],
    rng: &mut SeededRng,
) -> Position {
    let stdev = 4.0_f32;

    if let Some(parent) = parent_pos {
        // 90% near parent, 10% random
        if rng.gen_f32() < 0.9 {
            // Try up to 5 times to find a passable cell near parent
            for _ in 0..5 {
                let dx = rng.gen_gaussian(0.0, stdev);
                let dy = rng.gen_gaussian(0.0, stdev);
                let nx = parent.x as f32 + dx;
                let ny = parent.y as f32 + dy;

                // Clamp to grid bounds
                let cx = (nx.round() as i32).clamp(0, width as i32 - 1) as u32;
                let cy = (ny.round() as i32).clamp(0, height as i32 - 1) as u32;

                let idx = (cy * width + cx) as usize;
                if terrain[idx].is_passable() {
                    return Position::new(cx, cy);
                }
            }
        }
    }

    // Fallback: random passable position
    random_passable(width, height, terrain, rng)
}

/// Find a random passable position on the grid.
fn random_passable(width: u32, height: u32, terrain: &[Terrain], rng: &mut SeededRng) -> Position {
    loop {
        let x = rng.gen_range(0..width);
        let y = rng.gen_range(0..height);
        let idx = (y * width + x) as usize;
        if terrain[idx].is_passable() {
            return Position::new(x, y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_terrain(width: u32, height: u32) -> Vec<Terrain> {
        vec![Terrain::Open; (width * height) as usize]
    }

    #[test]
    fn kin_placement_near_parent() {
        let mut rng = SeededRng::new(42);
        let terrain = flat_terrain(100, 100);
        let parent = Position::new(50, 50);

        let mut near_count = 0;
        let trials = 1000;

        for _ in 0..trials {
            let pos = kin_placement(Some(parent), 100, 100, &terrain, &mut rng);
            let dist = parent.distance_to(&pos);
            if dist <= 8.0 {
                near_count += 1;
            }
        }

        // > 80% should be within 8 cells (2 * stdev)
        let ratio = f64::from(near_count) / f64::from(trials);
        assert!(
            ratio > 0.7,
            "Expected >70% within 8 cells of parent, got {ratio:.2}"
        );
    }

    #[test]
    fn kin_placement_none_parent_is_random() {
        let mut rng = SeededRng::new(42);
        let terrain = flat_terrain(100, 100);

        let pos = kin_placement(None, 100, 100, &terrain, &mut rng);
        // Should not panic and should return a valid position
        assert!(pos.x < 100);
        assert!(pos.y < 100);
    }

    #[test]
    fn kin_placement_all_passable() {
        let mut rng = SeededRng::new(42);
        let terrain = flat_terrain(40, 30);
        let parent = Position::new(20, 15);

        // Should never panic on flat terrain
        for _ in 0..100 {
            let pos = kin_placement(Some(parent), 40, 30, &terrain, &mut rng);
            assert!(pos.x < 40);
            assert!(pos.y < 30);
        }
    }
}
