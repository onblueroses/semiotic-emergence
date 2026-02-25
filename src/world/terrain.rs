use serde::{Deserialize, Serialize};

use crate::config::WorldConfig;
use crate::rng::SeededRng;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Terrain {
    Open,
    Grass,
    Tree,
    Rock,
    Water,
    Bush,
}

impl Terrain {
    pub fn is_passable(&self) -> bool {
        !matches!(self, Terrain::Water)
    }

    pub fn ascii_char(&self) -> char {
        match self {
            Terrain::Open => '.',
            Terrain::Grass => ',',
            Terrain::Tree => 'T',
            Terrain::Rock => 'R',
            Terrain::Water => '~',
            Terrain::Bush => '#',
        }
    }
}

/// Generate terrain for the world grid.
///
/// Places tree clusters, rock outcrops, water rivers, and scattered bushes
/// according to config percentages. Validates that >90% of passable cells
/// are connected via flood fill. Retries up to 10 times on validation failure.
pub(crate) fn generate_terrain(
    width: u32,
    height: u32,
    config: &WorldConfig,
    rng: &mut SeededRng,
) -> Result<Vec<Terrain>, crate::config::SimError> {
    let size = (width * height) as usize;

    for _attempt in 0..10 {
        let mut grid = vec![Terrain::Open; size];

        let idx = |x: u32, y: u32| -> usize { (y * width + x) as usize };

        // Phase 1: Tree clusters (5-15 cells each)
        let target_trees = (size as f32 * config.terrain_tree_pct) as usize;
        let mut tree_count = 0;
        while tree_count < target_trees {
            let cluster_size = rng.gen_range(5..=15);
            let sx = rng.gen_range(0..width);
            let sy = rng.gen_range(0..height);
            let mut cx = sx;
            let mut cy = sy;
            for _ in 0..cluster_size {
                if tree_count >= target_trees {
                    break;
                }
                let i = idx(cx, cy);
                if grid[i] == Terrain::Open {
                    grid[i] = Terrain::Tree;
                    tree_count += 1;
                }
                // Random walk for cluster shape
                match rng.gen_range(0..4) {
                    0 if cy > 0 => cy -= 1,
                    1 if cy < height - 1 => cy += 1,
                    2 if cx > 0 => cx -= 1,
                    3 if cx < width - 1 => cx += 1,
                    _ => {}
                }
            }
        }

        // Phase 2: Rock outcrops (3-8 cells each)
        let target_rocks = (size as f32 * config.terrain_rock_pct) as usize;
        let mut rock_count = 0;
        while rock_count < target_rocks {
            let cluster_size = rng.gen_range(3..=8);
            let sx = rng.gen_range(0..width);
            let sy = rng.gen_range(0..height);
            let mut cx = sx;
            let mut cy = sy;
            for _ in 0..cluster_size {
                if rock_count >= target_rocks {
                    break;
                }
                let i = idx(cx, cy);
                if grid[i] == Terrain::Open {
                    grid[i] = Terrain::Rock;
                    rock_count += 1;
                }
                match rng.gen_range(0..4) {
                    0 if cy > 0 => cy -= 1,
                    1 if cy < height - 1 => cy += 1,
                    2 if cx > 0 => cx -= 1,
                    3 if cx < width - 1 => cx += 1,
                    _ => {}
                }
            }
        }

        // Phase 3: Water rivers (random walk with 70% momentum)
        let target_water = (size as f32 * config.terrain_water_pct) as usize;
        let mut water_count = 0;
        while water_count < target_water {
            let mut wx = rng.gen_range(0..width);
            let mut wy = rng.gen_range(0..height);
            // Pick initial direction: 0=N, 1=S, 2=E, 3=W
            let mut direction = rng.gen_range(0..4_u8);
            let river_len = rng.gen_range(10..30).min(target_water - water_count);
            for _ in 0..river_len {
                let i = idx(wx, wy);
                if grid[i] == Terrain::Open {
                    grid[i] = Terrain::Water;
                    water_count += 1;
                }
                // 70% chance to keep direction, 30% chance to turn
                if rng.gen_f32() > 0.7 {
                    direction = rng.gen_range(0..4);
                }
                match direction {
                    0 if wy > 0 => wy -= 1,
                    1 if wy < height - 1 => wy += 1,
                    2 if wx < width - 1 => wx += 1,
                    3 if wx > 0 => wx -= 1,
                    _ => {
                        // Hit edge, reverse direction
                        direction = match direction {
                            0 => 1,
                            1 => 0,
                            2 => 3,
                            _ => 2,
                        };
                    }
                }
            }
        }

        // Phase 4: Scattered bushes
        let target_bushes = (size as f32 * config.terrain_bush_pct) as usize;
        let mut bush_count = 0;
        while bush_count < target_bushes {
            let bx = rng.gen_range(0..width);
            let by = rng.gen_range(0..height);
            let i = idx(bx, by);
            if grid[i] == Terrain::Open {
                grid[i] = Terrain::Bush;
                bush_count += 1;
            }
        }

        // Phase 5: Convert remaining Open cells to roughly 50/50 Open/Grass
        for cell in &mut grid {
            if *cell == Terrain::Open && rng.gen_bool(0.5) {
                *cell = Terrain::Grass;
            }
        }

        // Validate connectivity: >90% of passable cells reachable from first passable cell
        if flood_fill_validate(&grid, width, height) {
            return Ok(grid);
        }
    }

    Err(crate::config::SimError::Simulation(
        "Terrain generation failed: could not achieve >90% connectivity after 10 attempts".into(),
    ))
}

/// Check that >90% of passable cells are reachable from the first passable cell.
fn flood_fill_validate(grid: &[Terrain], width: u32, height: u32) -> bool {
    let size = grid.len();
    let passable_count = grid.iter().filter(|t| t.is_passable()).count();
    if passable_count == 0 {
        return true;
    }

    // Find first passable cell
    let Some(start) = grid.iter().position(Terrain::is_passable) else {
        return true;
    };

    let mut visited = vec![false; size];
    let mut stack = vec![start];
    visited[start] = true;
    let mut reachable = 0_usize;

    while let Some(pos) = stack.pop() {
        reachable += 1;
        let x = (pos % width as usize) as u32;
        let y = (pos / width as usize) as u32;

        // Check 4 neighbors
        let neighbors: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for (dx, dy) in neighbors {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && ny >= 0 && (nx as u32) < width && (ny as u32) < height {
                let ni = (ny as u32 * width + nx as u32) as usize;
                if !visited[ni] && grid[ni].is_passable() {
                    visited[ni] = true;
                    stack.push(ni);
                }
            }
        }
    }

    let ratio = reachable as f32 / passable_count as f32;
    ratio > 0.9
}

#[cfg(test)]
#[expect(
    clippy::panic,
    reason = "tests use panic via unwrap_or_else and assert macros"
)]
mod tests {
    use super::*;
    use crate::config::WorldConfig;
    use crate::rng::SeededRng;

    fn test_config() -> WorldConfig {
        WorldConfig {
            width: 40,
            height: 30,
            food_density: 0.15,
            food_energy: 20.0,
            food_regrow_ticks: 50,
            terrain_tree_pct: 0.10,
            terrain_rock_pct: 0.05,
            terrain_water_pct: 0.05,
            terrain_bush_pct: 0.08,
        }
    }

    #[test]
    fn terrain_generation_deterministic() {
        let config = test_config();
        let mut rng1 = SeededRng::new(42);
        let mut rng2 = SeededRng::new(42);
        let grid1 = generate_terrain(40, 30, &config, &mut rng1).unwrap_or_else(|e| panic!("{e}"));
        let grid2 = generate_terrain(40, 30, &config, &mut rng2).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(grid1, grid2);
    }

    #[test]
    fn terrain_generation_correct_size() {
        let config = test_config();
        let mut rng = SeededRng::new(42);
        let grid = generate_terrain(40, 30, &config, &mut rng).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(grid.len(), 40 * 30);
    }

    #[test]
    fn terrain_generation_approximate_percentages() {
        let config = test_config();
        let mut rng = SeededRng::new(42);
        let grid = generate_terrain(40, 30, &config, &mut rng).unwrap_or_else(|e| panic!("{e}"));
        let size = grid.len() as f32;
        let trees = grid.iter().filter(|t| **t == Terrain::Tree).count() as f32;
        let rocks = grid.iter().filter(|t| **t == Terrain::Rock).count() as f32;
        let water = grid.iter().filter(|t| **t == Terrain::Water).count() as f32;
        let bushes = grid.iter().filter(|t| **t == Terrain::Bush).count() as f32;

        // Allow +/- 3% tolerance (absolute)
        assert!(
            (trees / size - config.terrain_tree_pct).abs() < 0.03,
            "trees: {}",
            trees / size
        );
        assert!(
            (rocks / size - config.terrain_rock_pct).abs() < 0.03,
            "rocks: {}",
            rocks / size
        );
        assert!(
            (water / size - config.terrain_water_pct).abs() < 0.03,
            "water: {}",
            water / size
        );
        assert!(
            (bushes / size - config.terrain_bush_pct).abs() < 0.03,
            "bushes: {}",
            bushes / size
        );
    }

    #[test]
    fn terrain_passes_flood_fill() {
        let config = test_config();
        let mut rng = SeededRng::new(42);
        let grid = generate_terrain(40, 30, &config, &mut rng).unwrap_or_else(|e| panic!("{e}"));
        assert!(flood_fill_validate(&grid, 40, 30));
    }
}
