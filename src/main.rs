mod brain;
mod evolution;
mod signal;
mod world;

use std::fs::File;
use std::io::Write;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use brain::Brain;
use world::{SignalEvent, World};

const EVAL_ROUNDS: usize = 5;
const KIN_ROUNDS: usize = 2;

fn compute_iconicity(signal_events: &[SignalEvent], ticks_near: u32, total_ticks: u32) -> f32 {
    if signal_events.is_empty() || total_ticks == 0 {
        return 0.0;
    }
    let signals_near = signal_events
        .iter()
        .filter(|e| e.predator_dist < world::PREY_VISION_RANGE)
        .count() as f32;
    let signal_near_rate = signals_near / signal_events.len() as f32;
    let baseline_near_rate = ticks_near as f32 / total_ticks as f32;
    signal_near_rate - baseline_near_rate
}

fn compute_mutual_info(signal_events: &[SignalEvent]) -> f32 {
    if signal_events.len() < 20 {
        return 0.0;
    }
    // 3 symbols x 4 distance bins aligned with game mechanics:
    // [0-4) = within prey vision, [4-8) = signal range only, [8-11) = far, [11+) = max
    let mut counts = [[0u32; 4]; 3];
    for e in signal_events {
        let sym = (e.symbol as usize).min(2);
        let bin = if e.predator_dist < 4.0 {
            0
        } else if e.predator_dist < 8.0 {
            1
        } else if e.predator_dist < 11.0 {
            2
        } else {
            3
        };
        counts[sym][bin] += 1;
    }

    let n = signal_events.len() as f32;
    let mut mi = 0.0_f32;
    for s in 0..3 {
        let prob_s = counts[s].iter().sum::<u32>() as f32 / n;
        if prob_s == 0.0 {
            continue;
        }
        for (bin, &count) in counts[s].iter().enumerate() {
            let prob_d: f32 = (0..3).map(|ss| counts[ss][bin]).sum::<u32>() as f32 / n;
            if prob_d == 0.0 {
                continue;
            }
            let prob_joint = count as f32 / n;
            if prob_joint > 0.0 {
                mi += prob_joint * (prob_joint / (prob_s * prob_d)).ln();
            }
        }
    }
    mi
}

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

    let pop_size = 48;
    let group_size = 8;
    let elite_count = 8;
    let tournament_size = 3;
    let mutation_sigma = 0.1;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut population: Vec<Brain> = (0..pop_size).map(|_| Brain::random(&mut rng)).collect();

    let mut csv = File::create("output.csv")?;
    writeln!(
        csv,
        "generation,avg_fitness,max_fitness,signals_emitted,iconicity,mutual_info,confusion_ticks"
    )?;

    for gen in 0..generations {
        let mut fitness_accum: Vec<f32> = vec![0.0; pop_size];
        let mut total_signals: u32 = 0;
        let mut all_signal_events: Vec<SignalEvent> = Vec::new();
        let mut total_ticks_near: u32 = 0;
        let mut total_prey_ticks: u32 = 0;
        let mut total_confusion_ticks: u32 = 0;

        for round in 0..EVAL_ROUNDS {
            let mut indices: Vec<usize> = (0..pop_size).collect();
            if round < KIN_ROUNDS {
                // Kin grouping: sort by genome weight sum so similar genomes evaluate together.
                // Signalers co-located with other signalers get confusion benefit.
                indices.sort_by(|&a, &b| {
                    let sum_a: f32 = population[a].weights.iter().sum();
                    let sum_b: f32 = population[b].weights.iter().sum();
                    sum_a
                        .partial_cmp(&sum_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            } else {
                indices.shuffle(&mut rng);
            }

            for group_indices in indices.chunks(group_size) {
                let brains: Vec<Brain> = group_indices
                    .iter()
                    .map(|&idx| population[idx].clone())
                    .collect();
                let mut world = World::new(brains, &mut rng);

                for _ in 0..ticks_per_eval {
                    if !world.any_alive() {
                        break;
                    }
                    world.step(&mut rng);
                }

                total_signals += world.signals_emitted;
                total_ticks_near += world.ticks_near_predator;
                total_prey_ticks += world.total_prey_ticks;
                total_confusion_ticks += world.confusion_ticks;
                all_signal_events.append(&mut world.signal_events);

                for (i, prey) in world.prey.iter().enumerate() {
                    let fitness = prey.ticks_alive as f32 + prey.food_eaten as f32 * 10.0;
                    fitness_accum[group_indices[i]] += fitness;
                }
            }
        }

        let mut scored: Vec<(Brain, f32)> = population
            .iter()
            .enumerate()
            .map(|(i, brain)| (brain.clone(), fitness_accum[i] / EVAL_ROUNDS as f32))
            .collect();

        let avg_fitness: f32 = scored.iter().map(|(_, f)| f).sum::<f32>() / scored.len() as f32;
        let max_fitness: f32 = scored
            .iter()
            .map(|(_, f)| *f)
            .fold(f32::NEG_INFINITY, f32::max);

        let iconicity = compute_iconicity(&all_signal_events, total_ticks_near, total_prey_ticks);
        let mutual_info = compute_mutual_info(&all_signal_events);

        if gen % 10 == 0 || gen == generations - 1 {
            println!(
                "gen {gen:>4} | avg {avg_fitness:>7.1} | max {max_fitness:>7.1} | signals {total_signals} | icon {iconicity:.3} | MI {mutual_info:.3} | conf {total_confusion_ticks}"
            );
        }

        writeln!(
            csv,
            "{gen},{avg_fitness:.1},{max_fitness:.1},{total_signals},{iconicity:.4},{mutual_info:.4},{total_confusion_ticks}"
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
