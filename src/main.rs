mod brain;
mod evolution;
mod signal;
mod world;

use std::fs::File;
use std::io::Write;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use brain::Brain;
use world::World;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let generations: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let ticks_per_eval: u32 = 500;

    let pop_size = 20;
    let group_size = 4;
    let elite_count = 4;
    let tournament_size = 3;
    let mutation_sigma = 0.1;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut population: Vec<Brain> = (0..pop_size).map(|_| Brain::random(&mut rng)).collect();

    let mut csv = File::create("output.csv")?;
    writeln!(csv, "generation,avg_fitness,max_fitness,signals_emitted")?;

    for gen in 0..generations {
        let mut scored: Vec<(Brain, f32)> = Vec::with_capacity(pop_size);
        let mut total_signals: u32 = 0;

        for chunk in population.chunks(group_size) {
            let brains: Vec<Brain> = chunk.to_vec();
            let mut world = World::new(brains, &mut rng);

            for _ in 0..ticks_per_eval {
                if !world.any_alive() {
                    break;
                }
                world.step(&mut rng);
            }

            total_signals += world.signals_emitted;

            for (i, prey) in world.prey.iter().enumerate() {
                let fitness = prey.ticks_alive as f32 + prey.food_eaten as f32 * 10.0;
                scored.push((chunk[i].clone(), fitness));
            }
        }

        let avg_fitness: f32 = scored.iter().map(|(_, f)| f).sum::<f32>() / scored.len() as f32;
        let max_fitness: f32 = scored
            .iter()
            .map(|(_, f)| *f)
            .fold(f32::NEG_INFINITY, f32::max);

        if gen % 10 == 0 || gen == generations - 1 {
            println!(
                "gen {gen:>4} | avg {avg_fitness:>7.1} | max {max_fitness:>7.1} | signals {total_signals}"
            );
        }

        writeln!(
            csv,
            "{gen},{avg_fitness:.1},{max_fitness:.1},{total_signals}"
        )?;

        population = evolution::evolve(
            &mut scored,
            elite_count,
            tournament_size,
            mutation_sigma,
            &mut rng,
        );
    }

    println!("Done. Results in output.csv");
    Ok(())
}
