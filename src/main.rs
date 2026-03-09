mod brain;
mod evolution;
mod metrics;
mod signal;
mod world;

use std::fs::File;
use std::io::Write;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use brain::{Brain, INPUTS};
use evolution::Agent;
use world::{World, GRID_SIZE, INPUT_NAMES};

const POP_SIZE: usize = 48;
const TICKS_PER_EVAL: u32 = 500;
const NUM_PREDATORS: usize = 2;
const FLUCT_WINDOW: usize = 10;
const MIN_RECEIVER_SAMPLES: u32 = 30;

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
    iconicity: f32,
    mutual_info: f32,
    jsd_no_pred: f32,
    jsd_pred: f32,
    per_sym_jsd: [f32; 3],
    silence_corr: f32,
    gen_matrix: [[u32; 4]; 3],
    traj_jsd: f32,
    input_mi: [f32; INPUTS],
    contrast: [f32; 3],
    sender_fit_corr: f32,
    traj_fluct_ratio: f32,
    receiver_fit_corr: f32,
    response_fit_corr: f32,
    silence_onset_jsd: f32,
    silence_move_delta: f32,
}

impl GenMetrics {
    fn write_csv(&self, f: &mut File, gen: usize) -> Result<(), Box<dyn std::error::Error>> {
        writeln!(
            f,
            "{gen},{:.1},{:.1},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            self.avg_fitness,
            self.max_fitness,
            self.total_signals,
            self.iconicity,
            self.mutual_info,
            self.jsd_no_pred,
            self.jsd_pred,
            self.silence_corr,
            self.sender_fit_corr,
            self.traj_fluct_ratio,
            self.receiver_fit_corr,
            self.response_fit_corr,
            self.silence_onset_jsd,
            self.silence_move_delta
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
            "gen {gen:>4} | avg {:>7.1} | max {:>7.1} | signals {} | icon {:.3} | MI {:.3} | jsd {:.3}/{:.3} | sym [{:.3},{:.3},{:.3}] | sil {:.3}",
            self.avg_fitness, self.max_fitness, self.total_signals,
            self.iconicity, self.mutual_info,
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
    receiver_counts: [[[u32; 5]; 2]; 4],
    signals_per_tick: Vec<f32>,
    min_pred_dist: Vec<f32>,
    signal_rate_per_prey: Vec<f32>,
    actions_with_signal: Vec<[[u32; 5]; 2]>,
    actions_without_signal: Vec<[[u32; 5]; 2]>,
    silence_onset_actions: Vec<[[u32; 5]; 2]>,
}

fn evaluate_generation(population: &[Agent], rng: &mut ChaCha8Rng, no_signals: bool) -> EvalResult {
    let mut world = World::new_with_positions(population, NUM_PREDATORS, rng, no_signals);

    for _ in 0..TICKS_PER_EVAL {
        if !world.any_alive() {
            break;
        }
        world.step(rng);
    }

    let fitness: Vec<f32> = world
        .prey
        .iter()
        .map(|p| p.ticks_alive as f32 + p.food_eaten as f32 * 10.0)
        .collect();

    let signal_rate_per_prey: Vec<f32> = world
        .prey
        .iter()
        .enumerate()
        .map(|(idx, prey)| {
            if prey.ticks_alive == 0 {
                return 0.0;
            }
            let count = world
                .signal_events
                .iter()
                .filter(|e| e.emitter_idx == idx)
                .count() as f32;
            count / prey.ticks_alive as f32
        })
        .collect();

    let actions_with_signal: Vec<[[u32; 5]; 2]> =
        world.prey.iter().map(|p| p.actions_with_signal).collect();
    let actions_without_signal: Vec<[[u32; 5]; 2]> = world
        .prey
        .iter()
        .map(|p| p.actions_without_signal)
        .collect();
    let silence_onset_actions: Vec<[[u32; 5]; 2]> =
        world.prey.iter().map(|p| p.silence_onset_actions).collect();

    EvalResult {
        fitness,
        signal_events: world.signal_events,
        total_signals: world.signals_emitted,
        ticks_near: world.ticks_near_predator,
        prey_ticks: world.total_prey_ticks,
        receiver_counts: world.receiver_counts,
        signals_per_tick: world.signals_per_tick.iter().map(|&s| s as f32).collect(),
        min_pred_dist: world.min_pred_dist_per_tick,
        signal_rate_per_prey,
        actions_with_signal,
        actions_without_signal,
        silence_onset_actions,
    }
}

fn compute_gen_metrics(
    ev: &EvalResult,
    scored: &[(Agent, f32)],
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

    let fitness_vec: Vec<f32> = ev.fitness.clone();
    let sender_fit_corr = metrics::pearson(&ev.signal_rate_per_prey, &fitness_vec);

    traj_jsd_history.push(traj_jsd);
    let traj_fluct_ratio = metrics::rolling_fluctuation_ratio(traj_jsd_history, FLUCT_WINDOW);

    // Three-way coupling: receiver_fit_corr and response_fit_corr
    let reception_rates: Vec<f32> = ev
        .actions_with_signal
        .iter()
        .zip(&ev.actions_without_signal)
        .map(|(w, wo)| {
            let total_w: u32 = w.iter().flat_map(|c| c.iter()).sum();
            let total_wo: u32 = wo.iter().flat_map(|c| c.iter()).sum();
            let total = total_w + total_wo;
            if total > 0 {
                total_w as f32 / total as f32
            } else {
                0.0
            }
        })
        .collect();
    let receiver_fit_corr = metrics::pearson(&reception_rates, &fitness_vec);

    let per_prey_jsd_vec: Vec<f32> = ev
        .actions_with_signal
        .iter()
        .zip(&ev.actions_without_signal)
        .map(|(w, wo)| metrics::per_prey_receiver_jsd(w, wo, MIN_RECEIVER_SAMPLES))
        .collect();
    let response_fit_corr = metrics::pearson(&per_prey_jsd_vec, &fitness_vec);

    let (silence_onset_jsd, silence_move_delta) =
        metrics::compute_silence_onset_metrics(&ev.silence_onset_actions, &ev.actions_with_signal);

    GenMetrics {
        avg_fitness,
        max_fitness,
        total_signals: ev.total_signals,
        iconicity,
        mutual_info,
        jsd_no_pred,
        jsd_pred,
        per_sym_jsd,
        silence_corr,
        gen_matrix,
        traj_jsd,
        input_mi,
        contrast,
        sender_fit_corr,
        traj_fluct_ratio,
        receiver_fit_corr,
        response_fit_corr,
        silence_onset_jsd,
        silence_move_delta,
    }
}

fn run_seed(
    seed: u64,
    generations: usize,
    no_signals: bool,
    write_csv: bool,
) -> Result<RunResult, Box<dyn std::error::Error>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut population: Vec<Agent> = (0..POP_SIZE)
        .map(|_| Agent {
            brain: Brain::random(&mut rng),
            x: rng.gen_range(0..GRID_SIZE),
            y: rng.gen_range(0..GRID_SIZE),
        })
        .collect();

    let mut csv = write_csv.then(|| File::create("output.csv")).transpose()?;
    let mut traj_csv = write_csv
        .then(|| File::create("trajectory.csv"))
        .transpose()?;
    let mut input_mi_csv = write_csv
        .then(|| File::create("input_mi.csv"))
        .transpose()?;

    if let Some(ref mut f) = csv {
        writeln!(f, "generation,avg_fitness,max_fitness,signals_emitted,iconicity,mutual_info,jsd_no_pred,jsd_pred,silence_corr,sender_fit_corr,traj_fluct_ratio,receiver_fit_corr,response_fit_corr,silence_onset_jsd,silence_move_delta")?;
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

        let mut scored: Vec<(Agent, f32)> = population
            .iter()
            .enumerate()
            .map(|(i, agent)| (agent.clone(), ev.fitness[i]))
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

        population = evolution::evolve_spatial(&mut scored, 8, 3, 0.1, &mut rng);
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
