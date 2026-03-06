use rand::Rng;

use crate::brain::Brain;

pub fn tournament_select<'a>(
    population: &'a [(Brain, f32)],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> &'a Brain {
    let mut best_idx = rng.gen_range(0..population.len());
    let mut best_fit = population[best_idx].1;
    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..population.len());
        if population[idx].1 > best_fit {
            best_idx = idx;
            best_fit = population[idx].1;
        }
    }
    &population[best_idx].0
}

pub fn crossover(a: &Brain, b: &Brain, rng: &mut impl Rng) -> Brain {
    let point = rng.gen_range(1..a.weights.len());
    let mut weights = Vec::with_capacity(a.weights.len());
    weights.extend_from_slice(&a.weights[..point]);
    weights.extend_from_slice(&b.weights[point..]);
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

/// Run one generation of evolution:
/// - Sort by fitness descending
/// - Keep top `elite_count` unchanged
/// - Fill remaining via tournament selection + crossover + mutation
pub fn evolve(
    scored: &mut [(Brain, f32)],
    elite_count: usize,
    tournament_size: usize,
    sigma: f32,
    rng: &mut impl Rng,
) -> Vec<Brain> {
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let pop_size = scored.len();
    let mut next_gen: Vec<Brain> = Vec::with_capacity(pop_size);

    // Elites pass through
    for (brain, _) in scored.iter().take(elite_count) {
        next_gen.push(brain.clone());
    }

    // Fill remaining slots
    while next_gen.len() < pop_size {
        let parent_a = tournament_select(scored, tournament_size, rng);
        let parent_b = tournament_select(scored, tournament_size, rng);
        let mut child = crossover(parent_a, parent_b, rng);
        mutate(&mut child, sigma, rng);
        next_gen.push(child);
    }

    next_gen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::GENOME_LEN;

    #[test]
    fn crossover_preserves_length() {
        let mut rng = rand::thread_rng();
        let a = Brain::random(&mut rng);
        let b = Brain::random(&mut rng);
        let child = crossover(&a, &b, &mut rng);
        assert_eq!(child.weights.len(), GENOME_LEN);
    }

    #[test]
    fn evolve_preserves_population_size() {
        let mut rng = rand::thread_rng();
        let mut scored: Vec<(Brain, f32)> = (0..20)
            .map(|i| (Brain::random(&mut rng), i as f32))
            .collect();
        let next = evolve(&mut scored, 4, 3, 0.1, &mut rng);
        assert_eq!(next.len(), 20);
    }
}
