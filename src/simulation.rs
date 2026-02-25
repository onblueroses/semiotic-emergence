use crate::agent::sensor;
use crate::brain::genome::{GenomeId, NeatGenome};
use crate::brain::innovation::InnovationCounter;
use crate::config::{SimConfig, SimError};
use crate::world::grid::World;

/// CLI options that override config file values.
#[derive(Debug)]
pub struct SimOptions {
    pub seed: Option<u64>,
    pub max_generations: Option<u32>,
}

/// Run the full simulation loop.
#[expect(clippy::print_stderr, reason = "simulation progress output")]
pub fn run_simulation(config: &mut SimConfig, options: &SimOptions) -> Result<(), SimError> {
    if let Some(seed) = options.seed {
        config.seed = seed;
    }
    let max_generations = options
        .max_generations
        .unwrap_or(config.evolution.max_generations);

    // Initialize world (terrain + food + predators)
    let mut world = World::initialize(config)?;

    // Create innovation counter. Initial node count = input_count + output_count.
    let input_count = sensor::input_count(config.signal.vocab_size);
    let output_count = sensor::output_count(config.signal.vocab_size);
    let initial_node_count = (input_count + output_count) as u32;
    let mut innovations = InnovationCounter::new(0, initial_node_count);

    // Generate initial population
    let mut genomes = create_initial_population(
        config.neat.population_size,
        input_count,
        output_count,
        &mut world,
        &mut innovations,
    );

    eprintln!(
        "Initialized: {}x{} world, {} prey, {} predators, {} food",
        config.world.width,
        config.world.height,
        genomes.len(),
        world.predators.len(),
        world.food.iter().filter(|f| f.is_some()).count(),
    );

    // Main generation loop
    for generation in 0..max_generations {
        world.generation = generation;
        world.spawn_prey(&genomes, config);

        eprintln!(
            "Generation {generation}: {} prey, {} predators",
            world.prey.len(),
            world.predators.len(),
        );

        // Run the generation (tick loop)
        let result = world.run_generation(config);

        eprintln!(
            "  -> {} ticks, {} alive, best fitness: {:.1}",
            result.ticks_elapsed,
            result.prey_alive_end,
            result
                .genomes_with_fitness
                .iter()
                .map(|(_, f)| *f)
                .fold(0.0_f32, f32::max),
        );

        // Placeholder evolution: keep elites, randomize rest
        genomes = next_generation_placeholder(
            &result.genomes_with_fitness,
            config,
            &mut world,
            &mut innovations,
            input_count,
            output_count,
        );

        // Reset world for next generation
        world.reset_for_generation(config);
    }

    eprintln!("Simulation complete: {max_generations} generations.");
    Ok(())
}

fn create_initial_population(
    population_size: u32,
    input_count: usize,
    output_count: usize,
    world: &mut World,
    innovations: &mut InnovationCounter,
) -> Vec<NeatGenome> {
    let mut genomes = Vec::with_capacity(population_size as usize);
    for i in 0..population_size {
        let genome = NeatGenome::create_minimal(
            GenomeId(u64::from(i)),
            input_count,
            output_count,
            &mut world.rng,
            innovations,
        );
        genomes.push(genome);
    }
    innovations.reset_generation();
    genomes
}

/// Placeholder for real NEAT evolution. Keeps elites, randomizes rest.
fn next_generation_placeholder(
    genomes_with_fitness: &[(NeatGenome, f32)],
    config: &SimConfig,
    world: &mut World,
    innovations: &mut InnovationCounter,
    input_count: usize,
    output_count: usize,
) -> Vec<NeatGenome> {
    let pop_size = config.neat.population_size as usize;
    let elite_count = config.neat.elitism_count as usize;

    // Sort by fitness descending
    let mut sorted: Vec<_> = genomes_with_fitness.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut next_gen = Vec::with_capacity(pop_size);

    // Keep elites
    for (genome, _) in sorted.iter().take(elite_count.min(pop_size)) {
        next_gen.push(genome.clone());
    }

    // Fill rest with new random genomes
    innovations.reset_generation();
    let mut genome_id_counter = genomes_with_fitness.len() as u64;
    while next_gen.len() < pop_size {
        let genome = NeatGenome::create_minimal(
            GenomeId(genome_id_counter),
            input_count,
            output_count,
            &mut world.rng,
            innovations,
        );
        genome_id_counter += 1;
        next_gen.push(genome);
    }
    innovations.reset_generation();

    next_gen
}
