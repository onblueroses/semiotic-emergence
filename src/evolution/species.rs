use crate::brain::genome::{GenomeId, NeatGenome, SpeciesId};
use crate::config::SimConfig;

// ---------------------------------------------------------------------------
// Data structures (Step 3.1)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct Species {
    pub(crate) id: SpeciesId,
    pub(crate) representative: NeatGenome,
    pub(crate) members: Vec<GenomeId>,
    pub(crate) best_fitness: f32,
    pub(crate) stagnation_counter: u32,
}

#[derive(Debug)]
pub(crate) struct SpeciesState {
    pub(crate) species: Vec<Species>,
    next_species_id: u32,
    pub(crate) threshold: f32,
}

impl SpeciesState {
    pub(crate) fn new(initial_threshold: f32) -> Self {
        Self {
            species: Vec::new(),
            next_species_id: 0,
            threshold: initial_threshold,
        }
    }

    fn allocate_species_id(&mut self) -> SpeciesId {
        let id = SpeciesId(self.next_species_id);
        self.next_species_id += 1;
        id
    }
}

// ---------------------------------------------------------------------------
// Compatibility distance (Step 3.2)
// ---------------------------------------------------------------------------

/// Compute compatibility distance between two genomes using NEAT formula.
///
/// `delta = c1*E/N + c2*D/N + c3*w_avg` where E = excess genes, D = disjoint genes,
/// `w_avg` = average weight diff of matching genes, N = max(genome lengths, 1) (D4).
pub(crate) fn compatibility_distance(
    genome_a: &NeatGenome,
    genome_b: &NeatGenome,
    config: &SimConfig,
) -> f32 {
    let neat = &config.neat;

    let mut idx_a = 0;
    let mut idx_b = 0;
    let mut matching = 0;
    let mut disjoint = 0;
    let mut weight_diff_sum = 0.0_f32;

    while idx_a < genome_a.connections.len() && idx_b < genome_b.connections.len() {
        let inn_a = genome_a.connections[idx_a].innovation.0;
        let inn_b = genome_b.connections[idx_b].innovation.0;

        match inn_a.cmp(&inn_b) {
            std::cmp::Ordering::Equal => {
                matching += 1;
                weight_diff_sum +=
                    (genome_a.connections[idx_a].weight - genome_b.connections[idx_b].weight).abs();
                idx_a += 1;
                idx_b += 1;
            }
            std::cmp::Ordering::Less => {
                disjoint += 1;
                idx_a += 1;
            }
            std::cmp::Ordering::Greater => {
                disjoint += 1;
                idx_b += 1;
            }
        }
    }

    // Remaining genes in either parent are excess
    let excess = (genome_a.connections.len() - idx_a) + (genome_b.connections.len() - idx_b);

    let norm = genome_a
        .connections
        .len()
        .max(genome_b.connections.len())
        .max(1) as f32;
    let w_avg = if matching > 0 {
        weight_diff_sum / matching as f32
    } else {
        0.0
    };

    neat.c1_excess * excess as f32 / norm
        + neat.c2_disjoint * disjoint as f32 / norm
        + neat.c3_weight * w_avg
}

// ---------------------------------------------------------------------------
// Species assignment (Step 3.3)
// ---------------------------------------------------------------------------

/// Assign genomes to species, track stagnation, and pick new representatives.
pub(crate) fn assign_species_and_update(
    genomes_with_fitness: &[(NeatGenome, f32)],
    species_state: &mut SpeciesState,
    config: &SimConfig,
    rng: &mut crate::rng::SeededRng,
) {
    // Save old best fitnesses for stagnation tracking
    let old_bests: std::collections::HashMap<SpeciesId, f32> = species_state
        .species
        .iter()
        .map(|s| (s.id, s.best_fitness))
        .collect();

    // Clear member lists
    for species in &mut species_state.species {
        species.members.clear();
    }

    // Assign each genome to a species
    for (genome, _fitness) in genomes_with_fitness {
        let mut assigned = false;

        for species in &mut species_state.species {
            let dist = compatibility_distance(genome, &species.representative, config);
            if dist < species_state.threshold {
                species.members.push(genome.id);
                assigned = true;
                break;
            }
        }

        if !assigned {
            let id = species_state.allocate_species_id();
            species_state.species.push(Species {
                id,
                representative: genome.clone(),
                members: vec![genome.id],
                best_fitness: 0.0,
                stagnation_counter: 0,
            });
        }
    }

    // Remove empty species
    species_state.species.retain(|s| !s.members.is_empty());

    // Build genome lookup for fitness and representative selection
    let genome_map: std::collections::HashMap<GenomeId, &NeatGenome> = genomes_with_fitness
        .iter()
        .map(|(g, _)| (g.id, g))
        .collect();
    let fitness_map: std::collections::HashMap<GenomeId, f32> = genomes_with_fitness
        .iter()
        .map(|(g, f)| (g.id, *f))
        .collect();

    // Update each species: best fitness, stagnation, and pick new representative
    for species in &mut species_state.species {
        // Find best fitness among current members
        let current_best = species
            .members
            .iter()
            .filter_map(|id| fitness_map.get(id))
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let old_best = old_bests.get(&species.id).copied().unwrap_or(0.0);

        if current_best > old_best {
            species.best_fitness = current_best;
            species.stagnation_counter = 0;
        } else {
            species.best_fitness = old_best;
            species.stagnation_counter += 1;
        }

        // Pick random representative from members for next generation's comparison
        if !species.members.is_empty() {
            let idx = rng.gen_range(0..species.members.len());
            if let Some(rep) = genome_map.get(&species.members[idx]) {
                species.representative = (*rep).clone();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stagnation removal (Step 3.4)
// ---------------------------------------------------------------------------

/// Remove species that have stagnated for too many generations.
///
/// EXCEPTION: never remove the species containing the global best genome (D6).
pub(crate) fn remove_stagnant_species(
    species_state: &mut SpeciesState,
    limit: u32,
    global_best_species_id: Option<SpeciesId>,
) {
    species_state.species.retain(|s| {
        // Always keep the global best's species
        if Some(s.id) == global_best_species_id {
            return true;
        }
        s.stagnation_counter < limit
    });
}

// ---------------------------------------------------------------------------
// Dynamic threshold adjustment (Step 3.5)
// ---------------------------------------------------------------------------

/// Adjust compatibility threshold to maintain target species count (D5).
///
/// If too many species: increase threshold (makes it easier to group together).
/// If too few: decrease threshold (makes it harder to be in same species).
/// Clamped to [0.5, 10.0].
pub(crate) fn adjust_threshold(
    species_state: &mut SpeciesState,
    target_min: u32,
    target_max: u32,
    delta: f32,
) {
    let count = species_state.species.len() as u32;
    if count > target_max {
        species_state.threshold += delta;
    } else if count < target_min {
        species_state.threshold -= delta;
    }
    species_state.threshold = species_state.threshold.clamp(0.5, 10.0);
}

/// Find the species ID containing the genome with the highest fitness.
pub(crate) fn find_global_best_species(
    species_state: &SpeciesState,
    fitness_map: &std::collections::HashMap<GenomeId, f32>,
) -> Option<SpeciesId> {
    let mut best_fitness = f32::NEG_INFINITY;
    let mut best_species = None;

    for species in &species_state.species {
        for member_id in &species.members {
            let fitness = fitness_map.get(member_id).copied().unwrap_or(0.0);
            if fitness > best_fitness {
                best_fitness = fitness;
                best_species = Some(species.id);
            }
        }
    }

    best_species
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::genome::{GenomeId, NeatGenome};
    use crate::brain::innovation::InnovationCounter;
    use crate::rng::SeededRng;

    #[expect(clippy::panic, reason = "test helper")]
    fn make_test_config() -> SimConfig {
        let toml_str = include_str!("../../config/default.toml");
        toml::from_str(toml_str).unwrap_or_else(|e| panic!("Failed to parse default config: {e}"))
    }

    #[test]
    fn identical_genomes_zero_distance() {
        let mut rng = SeededRng::new(42);
        let mut innovations = InnovationCounter::new(0, 8);
        let config = make_test_config();
        let genome = NeatGenome::create_minimal(GenomeId(1), 4, 3, &mut rng, &mut innovations);

        let dist = compatibility_distance(&genome, &genome, &config);
        assert!(
            dist.abs() < f32::EPSILON,
            "Distance between identical genomes should be 0, got {dist}"
        );
    }

    #[test]
    fn compatibility_distance_symmetric() {
        let mut rng = SeededRng::new(42);
        let mut innovations = InnovationCounter::new(0, 8);
        let config = make_test_config();
        let a = NeatGenome::create_minimal(GenomeId(1), 4, 3, &mut rng, &mut innovations);
        let b = NeatGenome::create_minimal(GenomeId(2), 4, 3, &mut rng, &mut innovations);

        let d_ab = compatibility_distance(&a, &b, &config);
        let d_ba = compatibility_distance(&b, &a, &config);
        assert!(
            (d_ab - d_ba).abs() < f32::EPSILON,
            "Distance should be symmetric: {d_ab} vs {d_ba}"
        );
    }

    #[test]
    fn stagnant_species_removed_except_best() {
        let mut state = SpeciesState::new(3.0);
        let mut rng = SeededRng::new(42);
        let mut innovations = InnovationCounter::new(0, 8);

        let g1 = NeatGenome::create_minimal(GenomeId(1), 4, 3, &mut rng, &mut innovations);
        let g2 = NeatGenome::create_minimal(GenomeId(2), 4, 3, &mut rng, &mut innovations);

        // Create two species manually
        state.species.push(Species {
            id: SpeciesId(0),
            representative: g1,
            members: vec![GenomeId(1)],
            best_fitness: 1.0,
            stagnation_counter: 25, // Over limit
        });
        state.species.push(Species {
            id: SpeciesId(1),
            representative: g2,
            members: vec![GenomeId(2)],
            best_fitness: 2.0,
            stagnation_counter: 25, // Over limit
        });
        state.next_species_id = 2;

        // Species 1 is global best - should survive
        remove_stagnant_species(&mut state, 20, Some(SpeciesId(1)));
        assert_eq!(state.species.len(), 1, "Only global best should survive");
        assert_eq!(state.species[0].id, SpeciesId(1));
    }

    #[test]
    fn threshold_adjusts_toward_target() {
        let mut state = SpeciesState::new(3.0);

        // Simulate too many species
        for i in 0..20 {
            state.species.push(Species {
                id: SpeciesId(i),
                representative: NeatGenome::new(GenomeId(u64::from(i)), vec![], vec![]),
                members: vec![GenomeId(u64::from(i))],
                best_fitness: 0.0,
                stagnation_counter: 0,
            });
        }
        state.next_species_id = 20;

        let old_threshold = state.threshold;
        adjust_threshold(&mut state, 5, 15, 0.3);
        assert!(
            state.threshold > old_threshold,
            "Threshold should increase when too many species"
        );
    }
}
