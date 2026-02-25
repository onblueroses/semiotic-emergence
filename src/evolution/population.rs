use crate::brain::genome::{GenomeId, NeatGenome};
use crate::brain::innovation::InnovationCounter;
use crate::config::SimConfig;
use crate::evolution::selection;
use crate::evolution::species::{
    SpeciesState, adjust_threshold, assign_species_and_update, find_global_best_species,
    remove_stagnant_species,
};
use crate::rng::SeededRng;
use crate::world::entity::Position;

/// Member data for a species during offspring allocation.
type SpeciesMemberData = Vec<(GenomeId, f32, Position)>;

/// Top-level NEAT evolution function.
///
/// Takes evaluated genomes (genome + fitness + position), performs speciation,
/// stagnation culling, fitness sharing, offspring allocation, crossover, mutation.
/// Returns next generation's genomes with optional parent positions for kin placement.
pub(crate) fn evolve_population(
    evaluated: &[(NeatGenome, f32, Position)],
    species_state: &mut SpeciesState,
    config: &SimConfig,
    rng: &mut SeededRng,
    innovations: &mut InnovationCounter,
) -> Vec<(NeatGenome, Option<Position>)> {
    let pop_size = config.neat.population_size as usize;

    if evaluated.is_empty() {
        return Vec::new();
    }

    // Step 1: Reset innovation counter for this generation
    innovations.reset_generation();

    // Step 2: Build flat genome+fitness list for speciation
    let genomes_with_fitness: Vec<(NeatGenome, f32)> =
        evaluated.iter().map(|(g, f, _)| (g.clone(), *f)).collect();

    // Step 3: Assign species
    assign_species_and_update(&genomes_with_fitness, species_state, config, rng);

    // Step 4: Find global best species (protected from stagnation culling)
    let fitness_map: std::collections::HashMap<GenomeId, f32> =
        evaluated.iter().map(|(g, f, _)| (g.id, *f)).collect();
    let global_best_species = find_global_best_species(species_state, &fitness_map);

    // Step 5: Remove stagnant species (D16: limit=20 from config)
    remove_stagnant_species(
        species_state,
        config.neat.stagnation_limit,
        global_best_species,
    );

    // Step 6: Adjust threshold toward target species count (D5: 5-15, delta=0.3)
    adjust_threshold(species_state, 5, 15, 0.3);

    // Step 7: Compute adjusted fitness and allocate offspring
    let genome_map: std::collections::HashMap<GenomeId, (&NeatGenome, f32, Position)> = evaluated
        .iter()
        .map(|(g, f, p)| (g.id, (g, *f, *p)))
        .collect();

    let mut allocations = allocate_offspring(species_state, &genome_map, pop_size);

    // Step 8: Produce offspring per species
    let mut next_gen = produce_all_offspring(
        &mut allocations,
        &genome_map,
        config,
        rng,
        innovations,
        evaluated,
    );

    // Pad to exactly pop_size if needed
    let mut next_genome_id =
        evaluated.iter().map(|(g, _, _)| g.id.0).max().unwrap_or(0) + 1 + next_gen.len() as u64;
    while next_gen.len() < pop_size {
        if let Some((last_genome, pos)) = next_gen.last().cloned() {
            let mut padded = last_genome;
            padded.id = GenomeId(next_genome_id);
            next_genome_id += 1;
            selection::mutate_weights(&mut padded, config, rng);
            next_gen.push((padded, pos));
        } else {
            break;
        }
    }

    next_gen.truncate(pop_size);
    next_gen
}

/// Compute adjusted fitness per species and allocate offspring counts.
fn allocate_offspring(
    species_state: &SpeciesState,
    genome_map: &std::collections::HashMap<GenomeId, (&NeatGenome, f32, Position)>,
    pop_size: usize,
) -> Vec<(usize, SpeciesMemberData)> {
    let mut allocations: Vec<(usize, SpeciesMemberData)> = Vec::new();
    let mut total_adjusted = 0.0_f64;

    for species in &species_state.species {
        let size = species.members.len() as f64;
        let mut members: SpeciesMemberData = Vec::new();
        let mut species_sum = 0.0_f64;

        for &member_id in &species.members {
            if let Some(&(_, fitness, pos)) = genome_map.get(&member_id) {
                species_sum += f64::from(fitness) / size;
                members.push((member_id, fitness, pos));
            }
        }

        total_adjusted += species_sum;
        allocations.push((0, members));
    }

    // Proportional allocation, minimum 1 per non-empty species
    let reserved = allocations.iter().filter(|(_, m)| !m.is_empty()).count();
    let remaining = pop_size.saturating_sub(reserved);

    for (alloc, members) in &mut allocations {
        if members.is_empty() {
            *alloc = 0;
            continue;
        }

        *alloc = 1;

        if total_adjusted > 0.0 {
            let species_adj: f64 = members
                .iter()
                .map(|(_, f, _)| f64::from(*f) / members.len() as f64)
                .sum();
            let extra = (species_adj / total_adjusted * remaining as f64).round() as usize;
            *alloc += extra;
        }
    }

    normalize_allocations(&mut allocations, pop_size);
    allocations
}

/// Adjust allocations to sum to exactly `pop_size`.
fn normalize_allocations(allocations: &mut [(usize, SpeciesMemberData)], pop_size: usize) {
    let total: usize = allocations.iter().map(|(a, _)| *a).sum();

    if total > pop_size {
        let mut sorted_indices: Vec<usize> = (0..allocations.len()).collect();
        sorted_indices.sort_by(|&a, &b| allocations[b].0.cmp(&allocations[a].0));
        let mut excess = total - pop_size;
        for &idx in &sorted_indices {
            if excess == 0 {
                break;
            }
            if allocations[idx].0 > 1 {
                let trim = excess.min(allocations[idx].0 - 1);
                allocations[idx].0 -= trim;
                excess -= trim;
            }
        }
    } else if total < pop_size
        && let Some(largest) = allocations
            .iter_mut()
            .max_by_key(|(_, members)| members.len())
    {
        largest.0 += pop_size - total;
    }
}

/// Produce offspring for all species according to their allocations.
fn produce_all_offspring(
    allocations: &mut [(usize, SpeciesMemberData)],
    genome_map: &std::collections::HashMap<GenomeId, (&NeatGenome, f32, Position)>,
    config: &SimConfig,
    rng: &mut SeededRng,
    innovations: &mut InnovationCounter,
    evaluated: &[(NeatGenome, f32, Position)],
) -> Vec<(NeatGenome, Option<Position>)> {
    let pop_size = config.neat.population_size as usize;
    let mut next_gen: Vec<(NeatGenome, Option<Position>)> = Vec::with_capacity(pop_size);
    let mut next_id = evaluated.iter().map(|(g, _, _)| g.id.0).max().unwrap_or(0) + 1;

    for (alloc, members) in allocations {
        if *alloc == 0 || members.is_empty() {
            continue;
        }

        // Sort by fitness descending for elite selection
        members.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Elite: best genome passes through unchanged (if species gets >= 2 offspring)
        let elite_count = if *alloc >= 2 {
            config.neat.elitism_count.min(*alloc as u32) as usize
        } else {
            0
        };

        for (elite_id, _fitness, pos) in members.iter().take(elite_count) {
            if let Some(&(genome, _, _)) = genome_map.get(elite_id) {
                next_gen.push((genome.clone(), Some(*pos)));
            }
        }

        // Produce remaining offspring via crossover + mutation
        let offspring_needed = *alloc - elite_count;
        let member_fitness: Vec<(GenomeId, f32)> =
            members.iter().map(|(id, f, _)| (*id, *f)).collect();

        for _ in 0..offspring_needed {
            let id_a = selection::tournament_select(&member_fitness, 3, rng);
            let id_b = selection::tournament_select(&member_fitness, 3, rng);

            let data_a = genome_map.get(&id_a);
            let data_b = genome_map.get(&id_b);

            if let (Some(&(pa, fa, pos_a)), Some(&(pb, fb, _))) = (data_a, data_b) {
                let child_id = GenomeId(next_id);
                next_id += 1;

                let mut child = selection::crossover(pa, fa, pb, fb, child_id, rng);
                selection::mutate(&mut child, config, rng, innovations);

                next_gen.push((child, Some(pos_a)));
            }
        }
    }

    next_gen
}
