use rand::Rng;

use crate::brain::{Brain, INPUTS, MAX_GENOME_LEN, MAX_HIDDEN, MIN_HIDDEN, OUTPUTS};
use crate::world::wrap_dist_sq;

#[derive(Clone, Debug)]
pub struct Agent {
    pub brain: Brain,
    pub x: i32,
    pub y: i32,
}

const OFFSPRING_JITTER: i32 = 1;
const HIDDEN_SIZE_MUTATION_RATE: f32 = 0.05;
/// Mutate weights up to `hidden_size` + `MUTATION_HEADROOM` neurons.
/// Keeps dormant weights pre-seeded for when `hidden_size` grows.
const MUTATION_HEADROOM: usize = 4;

fn tournament_select<'a>(
    candidates: &'a [(Agent, f32)],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> &'a Agent {
    let mut best_idx = rng.gen_range(0..candidates.len());
    let mut best_fit = candidates[best_idx].1;
    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..candidates.len());
        if candidates[idx].1 > best_fit {
            best_idx = idx;
            best_fit = candidates[idx].1;
        }
    }
    &candidates[best_idx].0
}

fn local_tournament_select<'a>(
    candidates: &'a [(Agent, f32)],
    center_x: i32,
    center_y: i32,
    radius: f32,
    tournament_size: usize,
    grid_size: i32,
    rng: &mut impl Rng,
) -> Option<&'a Agent> {
    let radius_sq = radius * radius;
    let nearby: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, (agent, _))| {
            wrap_dist_sq(agent.x, agent.y, center_x, center_y, grid_size) <= radius_sq
        })
        .map(|(i, _)| i)
        .collect();

    if nearby.len() < 2 {
        return None;
    }

    let mut best_idx = nearby[rng.gen_range(0..nearby.len())];
    let mut best_fit = candidates[best_idx].1;
    for _ in 1..tournament_size {
        let idx = nearby[rng.gen_range(0..nearby.len())];
        if candidates[idx].1 > best_fit {
            best_idx = idx;
            best_fit = candidates[idx].1;
        }
    }
    Some(&candidates[best_idx].0)
}

pub fn crossover(a: &Brain, b: &Brain, rng: &mut impl Rng) -> Brain {
    let point = rng.gen_range(1..MAX_GENOME_LEN);
    let mut weights = a.weights;
    weights[point..].copy_from_slice(&b.weights[point..]);
    // Inherit hidden_size from a random parent (50/50)
    let hidden_size = if rng.gen_bool(0.5) {
        a.hidden_size
    } else {
        b.hidden_size
    };
    Brain {
        weights,
        hidden_size,
    }
}

fn gaussian_noise(sigma: f32, rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(f32::MIN_POSITIVE);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * sigma
}

pub fn mutate(brain: &mut Brain, sigma: f32, rng: &mut impl Rng) {
    let scope = (brain.hidden_size + MUTATION_HEADROOM).min(MAX_HIDDEN);
    let w = &mut brain.weights;

    // Input->hidden: for each input row, mutate only columns 0..scope
    for i in 0..INPUTS {
        let row_start = i * MAX_HIDDEN;
        for h in 0..scope {
            w[row_start + h] += gaussian_noise(sigma, rng);
        }
    }

    // Hidden biases: 0..scope
    let bias_start = INPUTS * MAX_HIDDEN;
    for h in 0..scope {
        w[bias_start + h] += gaussian_noise(sigma, rng);
    }

    // Hidden->output: neurons 0..scope, all 8 outputs each
    let ho_start = bias_start + MAX_HIDDEN;
    for h in 0..scope {
        for o in 0..OUTPUTS {
            w[ho_start + h * OUTPUTS + o] += gaussian_noise(sigma, rng);
        }
    }

    // Output biases: always mutate
    let ob_start = ho_start + MAX_HIDDEN * OUTPUTS;
    for o in 0..OUTPUTS {
        w[ob_start + o] += gaussian_noise(sigma, rng);
    }
}

#[allow(clippy::cast_possible_wrap)]
pub fn mutate_hidden_size(brain: &mut Brain, rng: &mut impl Rng) {
    if rng.gen::<f32>() < HIDDEN_SIZE_MUTATION_RATE {
        let delta: i32 = if rng.gen_bool(0.5) { 1 } else { -1 };
        let new_size =
            (brain.hidden_size as i32 + delta).clamp(MIN_HIDDEN as i32, MAX_HIDDEN as i32);
        brain.hidden_size = new_size as usize;
    }
}

#[allow(clippy::too_many_arguments)]
fn select_parent<'a>(
    top_pool: &'a [(Agent, f32)],
    sx: i32,
    sy: i32,
    tournament_size: usize,
    grid_size: i32,
    reproduction_radius: f32,
    fallback_radius: f32,
    rng: &mut impl Rng,
) -> &'a Agent {
    if let Some(a) = local_tournament_select(
        top_pool,
        sx,
        sy,
        reproduction_radius,
        tournament_size,
        grid_size,
        rng,
    ) {
        return a;
    }
    if let Some(a) = local_tournament_select(
        top_pool,
        sx,
        sy,
        fallback_radius,
        tournament_size,
        grid_size,
        rng,
    ) {
        return a;
    }
    tournament_select(top_pool, tournament_size, rng)
}

/// Spatial evolution:
/// - Sort by fitness descending
/// - Top `elite_count` agents keep brain AND position unchanged
/// - Bottom agents are replaced: their positions become offspring slots
/// - Offspring selected from nearby parents via local tournament
#[allow(clippy::too_many_arguments)]
pub fn evolve_spatial(
    scored: &mut [(Agent, f32)],
    elite_count: usize,
    tournament_size: usize,
    sigma: f32,
    grid_size: i32,
    reproduction_radius: f32,
    fallback_radius: f32,
    rng: &mut impl Rng,
) -> Vec<Agent> {
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let scored = &*scored; // reborrow as immutable - no more mutation needed

    let pop_size = scored.len();
    let mut next_gen: Vec<Agent> = Vec::with_capacity(pop_size);

    // Elites keep brain AND position
    for (agent, _) in scored.iter().take(elite_count) {
        next_gen.push(agent.clone());
    }

    // Fill remaining slots at dead agents' positions
    for (agent, _) in scored.iter().skip(elite_count) {
        let sx = agent.x;
        let sy = agent.y;

        let parent_a = select_parent(
            scored,
            sx,
            sy,
            tournament_size,
            grid_size,
            reproduction_radius,
            fallback_radius,
            rng,
        );
        let parent_b = select_parent(
            scored,
            sx,
            sy,
            tournament_size,
            grid_size,
            reproduction_radius,
            fallback_radius,
            rng,
        );
        let mut child_brain = crossover(&parent_a.brain, &parent_b.brain, rng);
        mutate(&mut child_brain, sigma, rng);
        mutate_hidden_size(&mut child_brain, rng);

        let jx = rng.gen_range(-OFFSPRING_JITTER..=OFFSPRING_JITTER);
        let jy = rng.gen_range(-OFFSPRING_JITTER..=OFFSPRING_JITTER);

        next_gen.push(Agent {
            brain: child_brain,
            x: (sx + jx).rem_euclid(grid_size),
            y: (sy + jy).rem_euclid(grid_size),
        });
    }

    next_gen
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_GRID: i32 = 20;
    const TEST_REPRO_RADIUS: f32 = 6.0;
    const TEST_FALLBACK_RADIUS: f32 = 10.0;

    #[test]
    fn crossover_preserves_length() {
        let mut rng = rand::thread_rng();
        let a = Brain::random(&mut rng);
        let b = Brain::random(&mut rng);
        let child = crossover(&a, &b, &mut rng);
        assert_eq!(child.weights.len(), MAX_GENOME_LEN);
    }

    #[test]
    fn crossover_inherits_hidden_size() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut a = Brain::random(&mut rng);
        let mut b = Brain::random(&mut rng);
        a.hidden_size = 8;
        b.hidden_size = 12;

        let mut got_a = false;
        let mut got_b = false;
        for _ in 0..100 {
            let child = crossover(&a, &b, &mut rng);
            if child.hidden_size == 8 {
                got_a = true;
            }
            if child.hidden_size == 12 {
                got_b = true;
            }
        }
        assert!(got_a, "Should sometimes inherit parent a's hidden_size");
        assert!(got_b, "Should sometimes inherit parent b's hidden_size");
    }

    #[test]
    fn mutate_hidden_size_stays_in_bounds() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(99);

        let mut brain = Brain::zero();
        brain.hidden_size = MIN_HIDDEN;
        for _ in 0..1000 {
            mutate_hidden_size(&mut brain, &mut rng);
            assert!(brain.hidden_size >= MIN_HIDDEN);
            assert!(brain.hidden_size <= MAX_HIDDEN);
        }

        brain.hidden_size = MAX_HIDDEN;
        for _ in 0..1000 {
            mutate_hidden_size(&mut brain, &mut rng);
            assert!(brain.hidden_size >= MIN_HIDDEN);
            assert!(brain.hidden_size <= MAX_HIDDEN);
        }
    }

    #[test]
    fn evolve_spatial_preserves_population_size() {
        let mut rng = rand::thread_rng();
        let mut scored: Vec<(Agent, f32)> = (0..20)
            .map(|i| {
                (
                    Agent {
                        brain: Brain::random(&mut rng),
                        x: rng.gen_range(0..TEST_GRID),
                        y: rng.gen_range(0..TEST_GRID),
                    },
                    i as f32,
                )
            })
            .collect();
        let next = evolve_spatial(
            &mut scored,
            4,
            3,
            0.1,
            TEST_GRID,
            TEST_REPRO_RADIUS,
            TEST_FALLBACK_RADIUS,
            &mut rng,
        );
        assert_eq!(next.len(), 20);
    }

    #[test]
    fn evolve_spatial_elites_keep_positions() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let mut scored: Vec<(Agent, f32)> = (0..20)
            .map(|i| {
                (
                    Agent {
                        brain: Brain::random(&mut rng),
                        x: i as i32,
                        y: i as i32 + 1,
                    },
                    (20 - i) as f32, // highest fitness first
                )
            })
            .collect();
        let elite_count = 4;
        // Save elite positions before evolve
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let elite_positions: Vec<(i32, i32)> = scored
            .iter()
            .take(elite_count)
            .map(|(a, _)| (a.x, a.y))
            .collect();
        // Re-scramble so evolve_spatial does its own sort
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let next = evolve_spatial(
            &mut scored,
            elite_count,
            3,
            0.1,
            TEST_GRID,
            TEST_REPRO_RADIUS,
            TEST_FALLBACK_RADIUS,
            &mut rng,
        );

        for (i, (ex, ey)) in elite_positions.iter().enumerate() {
            assert_eq!(next[i].x, *ex, "Elite {i} x mismatch");
            assert_eq!(next[i].y, *ey, "Elite {i} y mismatch");
        }
    }

    #[test]
    fn evolve_spatial_propagates_hidden_size() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut scored: Vec<(Agent, f32)> = (0..20)
            .map(|i| {
                let mut brain = Brain::random(&mut rng);
                brain.hidden_size = 10;
                (
                    Agent {
                        brain,
                        x: rng.gen_range(0..TEST_GRID),
                        y: rng.gen_range(0..TEST_GRID),
                    },
                    i as f32,
                )
            })
            .collect();

        let next = evolve_spatial(
            &mut scored,
            4,
            3,
            0.1,
            TEST_GRID,
            TEST_REPRO_RADIUS,
            TEST_FALLBACK_RADIUS,
            &mut rng,
        );

        // Elites should keep hidden_size=10
        for agent in next.iter().take(4) {
            assert_eq!(
                agent.brain.hidden_size, 10,
                "Elites should preserve hidden_size"
            );
        }
        // Non-elites: inherited from parents (all 10) +/- mutation
        for agent in &next {
            assert!(agent.brain.hidden_size >= MIN_HIDDEN);
            assert!(agent.brain.hidden_size <= MAX_HIDDEN);
        }
    }

    #[test]
    fn local_tournament_selects_nearby() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let candidates: Vec<(Agent, f32)> = vec![
            (
                Agent {
                    brain: Brain::zero(),
                    x: 0,
                    y: 0,
                },
                10.0,
            ),
            (
                Agent {
                    brain: Brain::zero(),
                    x: 1,
                    y: 0,
                },
                5.0,
            ),
            (
                Agent {
                    brain: Brain::zero(),
                    x: 15,
                    y: 15,
                },
                100.0,
            ),
        ];

        // Center at (0,0), radius 3.0 - should only see agents at (0,0) and (1,0)
        let result = local_tournament_select(&candidates, 0, 0, 3.0, 5, TEST_GRID, &mut rng);
        assert!(result.is_some());
        let selected = result.unwrap();
        // Should be one of the two nearby agents, never the far one at (15,15)
        assert!(
            (selected.x == 0 && selected.y == 0) || (selected.x == 1 && selected.y == 0),
            "Selected agent at ({}, {}) is not nearby",
            selected.x,
            selected.y
        );
    }
}
