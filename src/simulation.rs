use crate::agent::sensor;
use crate::brain::genome::{GenomeId, NeatGenome};
use crate::brain::innovation::InnovationCounter;
use crate::config::{SimConfig, SimError};
use crate::evolution::population::evolve_population;
use crate::evolution::species::SpeciesState;
use crate::stats::collector::StatsCollector;
use crate::stats::export::export_csv;
use crate::world::entity::Position;
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

    // Create innovation counter. Initial node count = input + bias + output.
    let input_count = sensor::input_count(config.signal.vocab_size);
    let output_count = sensor::output_count(config.signal.vocab_size);
    let initial_node_count = (input_count + 1 + output_count) as u32; // +1 for bias node
    let mut innovations = InnovationCounter::new(0, initial_node_count);

    // Generate initial population (no parent positions for gen 0)
    let initial_genomes = create_initial_population(
        config.neat.population_size,
        input_count,
        output_count,
        &mut world,
        &mut innovations,
    );
    let mut next_gen_with_pos: Vec<(NeatGenome, Option<Position>)> =
        initial_genomes.into_iter().map(|g| (g, None)).collect();

    // Persistent species state across generations (D3: lives in simulation, not World)
    let mut species_state = SpeciesState::new(config.neat.compatibility_threshold);

    // Stats collection for semiotic metrics (MI, TopSim, iconicity)
    let mut stats_collector = StatsCollector::new();
    let stats_interval = config.stats.stats_interval;
    let vocab_size = config.signal.vocab_size;

    eprintln!(
        "Initialized: {}x{} world, {} prey, {} predators, {} food",
        config.world.width,
        config.world.height,
        next_gen_with_pos.len(),
        world.predators.len(),
        world.food.iter().filter(|f| f.is_some()).count(),
    );

    // Main generation loop
    for generation in 0..max_generations {
        world.generation = generation;
        world.spawn_prey_with_positions(&next_gen_with_pos, config);

        eprintln!(
            "Generation {generation}: {} prey, {} predators",
            world.prey.len(),
            world.predators.len(),
        );

        // Run the generation (tick loop)
        let result = world.run_generation(config);

        let best_fitness = result
            .genomes_with_fitness
            .iter()
            .map(|(_, f, _)| *f)
            .fold(0.0_f32, f32::max);
        let avg_fitness = if result.genomes_with_fitness.is_empty() {
            0.0
        } else {
            result
                .genomes_with_fitness
                .iter()
                .map(|(_, f, _)| *f)
                .sum::<f32>()
                / result.genomes_with_fitness.len() as f32
        };

        eprintln!(
            "  -> {} ticks, {} alive, best fitness: {:.1}, {} species",
            result.ticks_elapsed,
            result.prey_alive_end,
            best_fitness,
            species_state.species.len(),
        );

        // Feed signal events to stats collector and finalize at stats_interval
        if stats_interval > 0 && (generation + 1) % stats_interval == 0 {
            for event in &result.signal_events {
                stats_collector.record_signal(event.clone());
            }
            stats_collector.finalize_generation(
                generation,
                avg_fitness,
                best_fitness,
                species_state.species.len() as u32,
                result.prey_alive_end,
                u32::from(vocab_size),
            );
        }

        // NEAT evolution: speciation, selection, crossover, mutation
        // Returns (genome, Option<parent_position>) for kin-aware placement
        next_gen_with_pos = evolve_population(
            &result.genomes_with_fitness,
            &mut species_state,
            config,
            &mut world.rng,
            &mut innovations,
        );

        // Reset world for next generation
        world.reset_for_generation(config);
    }

    // Export stats CSV if we have data
    if !stats_collector.generations.is_empty() {
        if let Err(err) = export_csv(&stats_collector, &config.stats.export_path) {
            eprintln!("Warning: failed to export stats CSV: {err}");
        } else {
            eprintln!("Stats exported to {}", config.stats.export_path);
        }
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
