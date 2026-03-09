use rand::Rng;

use crate::brain::{Brain, GENOME_LEN};
use crate::world::{wrap_dist_sq, GRID_SIZE};

#[derive(Clone, Debug)]
pub struct Agent {
    pub brain: Brain,
    pub x: i32,
    pub y: i32,
}

const REPRODUCTION_RADIUS: f32 = 6.0;
const FALLBACK_RADIUS: f32 = 10.0;
const OFFSPRING_JITTER: i32 = 1;

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
    rng: &mut impl Rng,
) -> Option<&'a Agent> {
    let radius_sq = radius * radius;
    let nearby: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, (agent, _))| wrap_dist_sq(agent.x, agent.y, center_x, center_y) <= radius_sq)
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
    let point = rng.gen_range(1..GENOME_LEN);
    let mut weights = a.weights;
    weights[point..].copy_from_slice(&b.weights[point..]);
    Brain { weights }
}

pub fn mutate(brain: &mut Brain, sigma: f32, rng: &mut impl Rng) {
    for w in &mut brain.weights {
        // Box-Muller transform: Gaussian with mean 0, std dev sigma
        let u1: f32 = rng.gen::<f32>().max(f32::MIN_POSITIVE);
        let u2: f32 = rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        *w += z * sigma;
    }
}

fn select_parent<'a>(
    top_pool: &'a [(Agent, f32)],
    sx: i32,
    sy: i32,
    tournament_size: usize,
    rng: &mut impl Rng,
) -> &'a Agent {
    if let Some(a) =
        local_tournament_select(top_pool, sx, sy, REPRODUCTION_RADIUS, tournament_size, rng)
    {
        return a;
    }
    if let Some(a) =
        local_tournament_select(top_pool, sx, sy, FALLBACK_RADIUS, tournament_size, rng)
    {
        return a;
    }
    tournament_select(top_pool, tournament_size, rng)
}

/// Spatial evolution:
/// - Sort by fitness descending
/// - Top `elite_count` agents keep brain AND position unchanged
/// - Bottom agents are replaced: their positions become offspring slots
/// - Offspring selected from nearby parents via local tournament
pub fn evolve_spatial(
    scored: &mut [(Agent, f32)],
    elite_count: usize,
    tournament_size: usize,
    sigma: f32,
    rng: &mut impl Rng,
) -> Vec<Agent> {
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let pop_size = scored.len();
    let mut next_gen: Vec<Agent> = Vec::with_capacity(pop_size);

    // Elites keep brain AND position
    for (agent, _) in scored.iter().take(elite_count) {
        next_gen.push(agent.clone());
    }

    // Top pool for parent selection (the elites + other survivors)
    let top_pool: Vec<(Agent, f32)> = scored
        .iter()
        .take(pop_size.max(elite_count))
        .cloned()
        .collect();

    // Fill remaining slots at dead agents' positions
    for (agent, _) in scored.iter().skip(elite_count) {
        let sx = agent.x;
        let sy = agent.y;

        let parent_a = select_parent(&top_pool, sx, sy, tournament_size, rng);
        let parent_b = select_parent(&top_pool, sx, sy, tournament_size, rng);
        let mut child_brain = crossover(&parent_a.brain, &parent_b.brain, rng);
        mutate(&mut child_brain, sigma, rng);

        let jx = rng.gen_range(-OFFSPRING_JITTER..=OFFSPRING_JITTER);
        let jy = rng.gen_range(-OFFSPRING_JITTER..=OFFSPRING_JITTER);

        next_gen.push(Agent {
            brain: child_brain,
            x: (sx + jx).rem_euclid(GRID_SIZE),
            y: (sy + jy).rem_euclid(GRID_SIZE),
        });
    }

    next_gen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossover_preserves_length() {
        let mut rng = rand::thread_rng();
        let a = Brain::random(&mut rng);
        let b = Brain::random(&mut rng);
        let child = crossover(&a, &b, &mut rng);
        assert_eq!(child.weights.len(), GENOME_LEN);
    }

    #[test]
    fn evolve_spatial_preserves_population_size() {
        let mut rng = rand::thread_rng();
        let mut scored: Vec<(Agent, f32)> = (0..20)
            .map(|i| {
                (
                    Agent {
                        brain: Brain::random(&mut rng),
                        x: rng.gen_range(0..GRID_SIZE),
                        y: rng.gen_range(0..GRID_SIZE),
                    },
                    i as f32,
                )
            })
            .collect();
        let next = evolve_spatial(&mut scored, 4, 3, 0.1, &mut rng);
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

        let next = evolve_spatial(&mut scored, elite_count, 3, 0.1, &mut rng);

        for (i, (ex, ey)) in elite_positions.iter().enumerate() {
            assert_eq!(next[i].x, *ex, "Elite {i} x mismatch");
            assert_eq!(next[i].y, *ey, "Elite {i} y mismatch");
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
                    brain: Brain {
                        weights: [0.0; GENOME_LEN],
                    },
                    x: 0,
                    y: 0,
                },
                10.0,
            ),
            (
                Agent {
                    brain: Brain {
                        weights: [0.0; GENOME_LEN],
                    },
                    x: 1,
                    y: 0,
                },
                5.0,
            ),
            (
                Agent {
                    brain: Brain {
                        weights: [0.0; GENOME_LEN],
                    },
                    x: 15,
                    y: 15,
                },
                100.0,
            ),
        ];

        // Center at (0,0), radius 3.0 - should only see agents at (0,0) and (1,0)
        let result = local_tournament_select(&candidates, 0, 0, 3.0, 5, &mut rng);
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
