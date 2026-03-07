mod brain;
mod evolution;
mod metrics;
mod signal;
mod world;

use std::fs::File;
use std::io::Write;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use brain::{Brain, INPUTS};
use world::{World, INPUT_NAMES};

const EVAL_ROUNDS: usize = 5;
const KIN_ROUNDS: usize = 2;
const POP_SIZE: usize = 48;
const GROUP_SIZE: usize = 8;
const TICKS_PER_EVAL: u32 = 500;
const FLUCT_WINDOW: usize = 10;

struct RunResult {
    final_matrix: [[u32; 4]; 3],
    avg_fitness: f32,
    max_fitness: f32,
    mutual_info: f32,
}

struct GenMetrics {
    avg_fitness: f32,
    max_fitness: f32,
    total_signals: u32,
    confusion_ticks: u32,
    iconicity: f32,
    mutual_info: f32,
    jsd_no_pred: f32,
    jsd_pred: f32,
    per_sym_jsd: [f32; 3],
    silence_corr: f32,
    gen_matrix: [[u32; 4]; 3],
    traj_jsd: f32,
    input_mi: [f32; INPUTS],
    mi_kin: f32,
    mi_rnd: f32,
    jsd_no_pred_kin: f32,
    jsd_no_pred_rnd: f32,
    jsd_pred_kin: f32,
    jsd_pred_rnd: f32,
    contrast: [f32; 3],
    sender_fit_corr: f32,
    traj_fluct_ratio: f32,
}

impl GenMetrics {
    fn write_csv(&self, f: &mut File, gen: usize) -> Result<(), Box<dyn std::error::Error>> {
        writeln!(
            f,
            "{gen},{:.1},{:.1},{},{:.4},{:.4},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            self.avg_fitness,
            self.max_fitness,
            self.total_signals,
            self.iconicity,
            self.mutual_info,
            self.confusion_ticks,
            self.jsd_no_pred,
            self.jsd_pred,
            self.silence_corr,
            self.mi_kin,
            self.mi_rnd,
            self.jsd_no_pred_kin,
            self.jsd_no_pred_rnd,
            self.jsd_pred_kin,
            self.jsd_pred_rnd,
            self.sender_fit_corr,
            self.traj_fluct_ratio
        )?;
        Ok(())
    }

    fn write_input_mi(&self, f: &mut File, gen: usize) -> Result<(), Box<dyn std::error::Error>> {
        write!(f, "{gen}")?;
        for &v in &self.input_mi {
            write!(f, ",{v:.4}")?;
        }
        writeln!(f)?;
        Ok(())
    }

    fn write_trajectory(&self, f: &mut File, gen: usize) -> Result<(), Box<dyn std::error::Error>> {
        let m = &self.gen_matrix;
        writeln!(
            f,
            "{gen},{},{},{},{},{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            m[0][0],
            m[0][1],
            m[0][2],
            m[0][3],
            m[1][0],
            m[1][1],
            m[1][2],
            m[1][3],
            m[2][0],
            m[2][1],
            m[2][2],
            m[2][3],
            self.per_sym_jsd[0],
            self.per_sym_jsd[1],
            self.per_sym_jsd[2],
            self.traj_jsd,
            self.contrast[0],
            self.contrast[1],
            self.contrast[2]
        )?;
        Ok(())
    }

    fn print_log(&self, gen: usize) {
        println!(
            "gen {gen:>4} | avg {:>7.1} | max {:>7.1} | signals {} | icon {:.3} | MI {:.3} | conf {} | jsd {:.3}/{:.3} | sym [{:.3},{:.3},{:.3}] | sil {:.3}",
            self.avg_fitness, self.max_fitness, self.total_signals,
            self.iconicity, self.mutual_info, self.confusion_ticks,
            self.jsd_no_pred, self.jsd_pred,
            self.per_sym_jsd[0], self.per_sym_jsd[1], self.per_sym_jsd[2],
            self.silence_corr
        );
    }
}

struct EvalResult {
    fitness: Vec<f32>,
    signal_events: Vec<world::SignalEvent>,
    total_signals: u32,
    ticks_near: u32,
    prey_ticks: u32,
    confusion_ticks: u32,
    receiver_counts: [[[u32; 5]; 2]; 4],
    receiver_counts_kin: [[[u32; 5]; 2]; 4],
    receiver_counts_rnd: [[[u32; 5]; 2]; 4],
    signals_per_tick: Vec<f32>,
    min_pred_dist: Vec<f32>,
    /// Per-prey signal rate (signals per tick alive) across all rounds.
    signal_rate_per_prey: Vec<f32>,
}

fn evaluate_generation(population: &[Brain], rng: &mut ChaCha8Rng, no_signals: bool) -> EvalResult {
    let mut fitness = vec![0.0_f32; POP_SIZE];
    let mut signal_events: Vec<world::SignalEvent> = Vec::new();
    let mut total_signals: u32 = 0;
    let mut ticks_near: u32 = 0;
    let mut prey_ticks: u32 = 0;
    let mut confusion_ticks: u32 = 0;
    let mut receiver_counts = [[[0u32; 5]; 2]; 4];
    let mut receiver_counts_kin = [[[0u32; 5]; 2]; 4];
    let mut receiver_counts_rnd = [[[0u32; 5]; 2]; 4];
    let mut signals_per_tick: Vec<f32> = Vec::new();
    let mut min_pred_dist: Vec<f32> = Vec::new();
    let mut signal_counts_per_prey = [0.0_f32; POP_SIZE];
    let mut ticks_alive_per_prey = [0.0_f32; POP_SIZE];

    for round in 0..EVAL_ROUNDS {
        let kin_round = round < KIN_ROUNDS;
        let mut indices: Vec<usize> = (0..POP_SIZE).collect();
        if kin_round {
            indices.sort_by(|&a, &b| {
                let sum_a: f32 = population[a].weights.iter().sum();
                let sum_b: f32 = population[b].weights.iter().sum();
                sum_a
                    .partial_cmp(&sum_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            indices.shuffle(rng);
        }

        for group_indices in indices.chunks(GROUP_SIZE) {
            let brains: Vec<Brain> = group_indices
                .iter()
                .map(|&idx| population[idx].clone())
                .collect();
            let mut world = World::new(brains, rng, no_signals, kin_round);

            for _ in 0..TICKS_PER_EVAL {
                if !world.any_alive() {
                    break;
                }
                world.step(rng);
            }

            total_signals += world.signals_emitted;
            ticks_near += world.ticks_near_predator;
            prey_ticks += world.total_prey_ticks;
            confusion_ticks += world.confusion_ticks;
            for event in &world.signal_events {
                let pop_idx = group_indices[event.emitter_idx];
                signal_counts_per_prey[pop_idx] += 1.0;
            }
            signal_events.append(&mut world.signal_events);

            for (tc, wc) in receiver_counts.iter_mut().zip(&world.receiver_counts) {
                for (tc_ctx, wc_ctx) in tc.iter_mut().zip(wc) {
                    for (t, w) in tc_ctx.iter_mut().zip(wc_ctx) {
                        *t += w;
                    }
                }
            }
            let target_counts = if kin_round {
                &mut receiver_counts_kin
            } else {
                &mut receiver_counts_rnd
            };
            let source_counts = if kin_round {
                &world.receiver_counts_kin
            } else {
                &world.receiver_counts_rnd
            };
            for (tc, wc) in target_counts.iter_mut().zip(source_counts) {
                for (tc_ctx, wc_ctx) in tc.iter_mut().zip(wc) {
                    for (t, w) in tc_ctx.iter_mut().zip(wc_ctx) {
                        *t += w;
                    }
                }
            }

            signals_per_tick.extend(world.signals_per_tick.iter().map(|&s| s as f32));
            min_pred_dist.extend_from_slice(&world.min_pred_dist_per_tick);

            for (i, prey) in world.prey.iter().enumerate() {
                fitness[group_indices[i]] +=
                    prey.ticks_alive as f32 + prey.food_eaten as f32 * 10.0;
                ticks_alive_per_prey[group_indices[i]] += prey.ticks_alive as f32;
            }
        }
    }

    let signal_rate_per_prey: Vec<f32> = signal_counts_per_prey
        .iter()
        .zip(&ticks_alive_per_prey)
        .map(|(&signals, &ticks)| if ticks > 0.0 { signals / ticks } else { 0.0 })
        .collect();

    EvalResult {
        fitness,
        signal_events,
        total_signals,
        ticks_near,
        prey_ticks,
        confusion_ticks,
        receiver_counts,
        receiver_counts_kin,
        receiver_counts_rnd,
        signals_per_tick,
        min_pred_dist,
        signal_rate_per_prey,
    }
}

fn compute_gen_metrics(
    ev: &EvalResult,
    scored: &[(Brain, f32)],
    prev_norm_matrix: &mut Option<[[f32; 4]; 3]>,
    traj_jsd_history: &mut Vec<f32>,
) -> GenMetrics {
    let avg_fitness = scored.iter().map(|(_, f)| f).sum::<f32>() / scored.len() as f32;
    let max_fitness = scored
        .iter()
        .map(|(_, f)| *f)
        .fold(f32::NEG_INFINITY, f32::max);
    let iconicity = metrics::compute_iconicity(&ev.signal_events, ev.ticks_near, ev.prey_ticks);
    let mutual_info = metrics::compute_mutual_info(&ev.signal_events);
    let (jsd_no_pred, jsd_pred) = metrics::compute_receiver_jsd(&ev.receiver_counts);
    let per_sym_jsd = metrics::compute_per_symbol_jsd(&ev.receiver_counts);
    let silence_corr = metrics::pearson(&ev.signals_per_tick, &ev.min_pred_dist);
    let input_mi = metrics::compute_input_mi(&ev.signal_events);
    let kin_events: Vec<&world::SignalEvent> =
        ev.signal_events.iter().filter(|e| e.kin_round).collect();
    let rnd_events: Vec<&world::SignalEvent> =
        ev.signal_events.iter().filter(|e| !e.kin_round).collect();
    let mi_kin = metrics::compute_mutual_info_refs(&kin_events);
    let mi_rnd = metrics::compute_mutual_info_refs(&rnd_events);
    let (jsd_no_pred_kin, jsd_pred_kin) = metrics::compute_receiver_jsd(&ev.receiver_counts_kin);
    let (jsd_no_pred_rnd, jsd_pred_rnd) = metrics::compute_receiver_jsd(&ev.receiver_counts_rnd);
    let gen_matrix = metrics::signal_context_matrix(&ev.signal_events);
    let curr_norm = metrics::normalize_matrix(&gen_matrix);
    let traj_jsd = match (&*prev_norm_matrix, &curr_norm) {
        (Some(prev), Some(curr)) => metrics::trajectory_jsd(prev, curr),
        _ => 0.0,
    };
    let contrast = curr_norm
        .as_ref()
        .map_or([0.0; 3], metrics::inter_symbol_jsd);
    if let Some(norm) = curr_norm {
        *prev_norm_matrix = Some(norm);
    }

    let fitness_normalized: Vec<f32> = ev.fitness.iter().map(|f| f / EVAL_ROUNDS as f32).collect();
    let sender_fit_corr = metrics::pearson(&ev.signal_rate_per_prey, &fitness_normalized);

    traj_jsd_history.push(traj_jsd);
    let traj_fluct_ratio = metrics::rolling_fluctuation_ratio(traj_jsd_history, FLUCT_WINDOW);

    GenMetrics {
        avg_fitness,
        max_fitness,
        total_signals: ev.total_signals,
        confusion_ticks: ev.confusion_ticks,
        iconicity,
        mutual_info,
        jsd_no_pred,
        jsd_pred,
        per_sym_jsd,
        silence_corr,
        gen_matrix,
        traj_jsd,
        input_mi,
        mi_kin,
        mi_rnd,
        jsd_no_pred_kin,
        jsd_no_pred_rnd,
        jsd_pred_kin,
        jsd_pred_rnd,
        contrast,
        sender_fit_corr,
        traj_fluct_ratio,
    }
}

fn run_seed(
    seed: u64,
    generations: usize,
    no_signals: bool,
    write_csv: bool,
) -> Result<RunResult, Box<dyn std::error::Error>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut population: Vec<Brain> = (0..POP_SIZE).map(|_| Brain::random(&mut rng)).collect();

    let mut csv = write_csv.then(|| File::create("output.csv")).transpose()?;
    let mut traj_csv = write_csv
        .then(|| File::create("trajectory.csv"))
        .transpose()?;
    let mut input_mi_csv = write_csv
        .then(|| File::create("input_mi.csv"))
        .transpose()?;

    if let Some(ref mut f) = csv {
        writeln!(f, "generation,avg_fitness,max_fitness,signals_emitted,iconicity,mutual_info,confusion_ticks,jsd_no_pred,jsd_pred,silence_corr,mi_kin,mi_rnd,jsd_no_pred_kin,jsd_no_pred_rnd,jsd_pred_kin,jsd_pred_rnd,sender_fit_corr,traj_fluct_ratio")?;
    }
    if let Some(ref mut f) = traj_csv {
        writeln!(f, "generation,s0d0,s0d1,s0d2,s0d3,s1d0,s1d1,s1d2,s1d3,s2d0,s2d1,s2d2,s2d3,jsd_sym0,jsd_sym1,jsd_sym2,trajectory_jsd,contrast_01,contrast_02,contrast_12")?;
    }
    if let Some(ref mut f) = input_mi_csv {
        write!(f, "generation")?;
        for name in &INPUT_NAMES {
            write!(f, ",mi_{name}")?;
        }
        writeln!(f)?;
    }

    let mut last_result = RunResult {
        final_matrix: [[0; 4]; 3],
        avg_fitness: 0.0,
        max_fitness: 0.0,
        mutual_info: 0.0,
    };
    let mut prev_norm_matrix: Option<[[f32; 4]; 3]> = None;
    let mut traj_jsd_history: Vec<f32> = Vec::new();

    for gen in 0..generations {
        let ev = evaluate_generation(&population, &mut rng, no_signals);

        let mut scored: Vec<(Brain, f32)> = population
            .iter()
            .enumerate()
            .map(|(i, brain)| (brain.clone(), ev.fitness[i] / EVAL_ROUNDS as f32))
            .collect();

        let gm = compute_gen_metrics(&ev, &scored, &mut prev_norm_matrix, &mut traj_jsd_history);

        if let Some(ref mut f) = csv {
            gm.write_csv(f, gen)?;
        }
        if let Some(ref mut f) = traj_csv {
            gm.write_trajectory(f, gen)?;
        }
        if let Some(ref mut f) = input_mi_csv {
            gm.write_input_mi(f, gen)?;
        }
        if write_csv && (gen.is_multiple_of(10) || gen == generations - 1) {
            gm.print_log(gen);
        }

        last_result = RunResult {
            final_matrix: gm.gen_matrix,
            avg_fitness: gm.avg_fitness,
            max_fitness: gm.max_fitness,
            mutual_info: gm.mutual_info,
        };

        population = evolution::evolve(&mut scored, 8, 3, 0.1, &mut rng);
    }

    if write_csv {
        println!("Done. Results in output.csv, trajectory.csv, input_mi.csv");
    }
    Ok(last_result)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let no_signals = args.iter().any(|a| a == "--no-signals");
    let batch_mode = args.iter().position(|a| a == "--batch");

    if let Some(pos) = batch_mode {
        let n: usize = args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or(5);
        let generations: usize = args
            .get(pos + 2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);

        println!("Batch mode: {n} seeds x {generations} generations");
        let mut results: Vec<RunResult> = Vec::new();
        for seed in 0..n as u64 {
            println!("--- seed {seed} ---");
            results.push(run_seed(seed, generations, no_signals, false)?);
        }

        let norm_matrices: Vec<Option<[[f32; 4]; 3]>> = results
            .iter()
            .map(|r| metrics::normalize_matrix(&r.final_matrix))
            .collect();

        println!("\nDivergence matrix (permutation-aware JSD):");
        print!("     ");
        for j in 0..n {
            print!("  s{j:<4}");
        }
        println!();

        let mut div_csv = File::create("divergence.csv")?;
        write!(div_csv, "seed")?;
        for j in 0..n {
            write!(div_csv, ",s{j}")?;
        }
        writeln!(div_csv)?;

        for i in 0..n {
            print!("s{i:<4}");
            write!(div_csv, "{i}")?;
            for j in 0..n {
                let div = match (&norm_matrices[i], &norm_matrices[j]) {
                    (Some(a), Some(b)) => metrics::cross_population_divergence(a, b),
                    _ => f32::NAN,
                };
                print!("  {div:.4}");
                write!(div_csv, ",{div:.4}")?;
            }
            println!();
            writeln!(div_csv)?;
        }

        println!("\nPer-seed summary:");
        for (i, r) in results.iter().enumerate() {
            println!(
                "  seed {i}: avg={:.1} max={:.1} MI={:.3}",
                r.avg_fitness, r.max_fitness, r.mutual_info
            );
        }
        println!("Divergence matrix saved to divergence.csv");
    } else {
        let positional: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();
        let seed: u64 = positional
            .first()
            .and_then(|s| s.parse().ok())
            .unwrap_or(42);
        let generations: usize = positional
            .get(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);

        if no_signals {
            println!("Counterfactual mode: signals disabled");
        }
        run_seed(seed, generations, no_signals, true)?;
    }

    Ok(())
}
