mod brain;
mod evolution;
mod metrics;
mod signal;
mod world;

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use brain::{Brain, INPUTS};
use evolution::Agent;
use signal::NUM_SYMBOLS;
use world::{World, INPUT_NAMES};

const FLUCT_WINDOW: usize = 10;
const MIN_RECEIVER_SAMPLES: u32 = 10;

struct SimParams {
    pop_size: usize,
    grid_size: i32,
    num_zones: usize,
    food_count: usize,
    ticks_per_eval: u32,
    signal_range: f32,
    zone_radius: f32,
    zone_speed: f32,
    reproduction_radius: f32,
    fallback_radius: f32,
    mi_bins: [f32; 3],
    elite_count: usize,
    tournament_size: usize,
    mutation_sigma: f32,
    base_drain: f32,
    neuron_cost: f32,
    signal_cost: f32,
    zone_drain_rate: f32,
    no_signals: bool,
    patch_ratio: f32,
    kin_bonus: f32,
    metrics_interval: usize,
    fast_fail_tick: u32,
}

impl SimParams {
    fn from_cli(args: &[String]) -> Self {
        let pop_size = parse_flag(args, "--pop").unwrap_or(384);
        let grid_size = parse_flag::<i32>(args, "--grid").unwrap_or(56);
        let num_zones = parse_flag(args, "--pred").unwrap_or(3);
        let food_count = parse_flag(args, "--food").unwrap_or(100);
        let ticks_per_eval = parse_flag(args, "--ticks").unwrap_or(500);
        let no_signals = args.iter().any(|a| a == "--no-signals");
        let patch_ratio = parse_flag(args, "--patch-ratio").unwrap_or(0.5);
        let kin_bonus = parse_flag(args, "--kin-bonus").unwrap_or(0.1);
        let metrics_interval = parse_flag(args, "--metrics-interval").unwrap_or(1);
        let zone_radius: f32 = parse_flag(args, "--zone-radius").unwrap_or(8.0);
        let zone_speed = parse_flag(args, "--zone-speed").unwrap_or(0.5);
        let zone_drain_rate = parse_flag(args, "--zone-drain").unwrap_or(0.02);
        let signal_cost = parse_flag(args, "--signal-cost").unwrap_or(0.002);
        let fast_fail_tick: u32 = parse_flag(args, "--fast-fail").unwrap_or(0);

        let zone_coverage: Option<f32> = parse_flag(args, "--zone-coverage");
        let num_zones = if let Some(coverage) = zone_coverage {
            if parse_flag::<usize>(args, "--pred").is_some() {
                eprintln!("Warning: --zone-coverage overrides --pred");
            }
            let grid_area = (grid_size as f32).powi(2);
            let zone_area = std::f32::consts::PI * zone_radius.powi(2);
            (coverage * grid_area / zone_area).ceil() as usize
        } else {
            num_zones
        };

        let scale = grid_size as f32 / 20.0;
        let signal_range = 8.0 * scale;
        let reproduction_radius = 6.0 * scale;
        let fallback_radius = 10.0 * scale;
        let mi_bins = [zone_radius, signal_range, signal_range * 1.375];
        let elite_count = (pop_size / 6).max(2);

        SimParams {
            pop_size,
            grid_size,
            num_zones,
            food_count,
            ticks_per_eval,
            signal_range,
            zone_radius,
            zone_speed,
            reproduction_radius,
            fallback_radius,
            mi_bins,
            elite_count,
            tournament_size: 3,
            mutation_sigma: 0.1,
            base_drain: 0.0008,
            neuron_cost: 0.0,
            signal_cost,
            zone_drain_rate,
            no_signals,
            patch_ratio,
            kin_bonus,
            metrics_interval: metrics_interval.max(1),
            fast_fail_tick,
        }
    }
}

fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

struct RunResult {
    final_matrix: [[u32; 4]; NUM_SYMBOLS],
    avg_fitness: f32,
    max_fitness: f32,
    mutual_info: f32,
    sender_fit_corr: f32,
    receiver_fit_corr: f32,
    response_fit_corr: f32,
    zone_deaths: u32,
    generations_completed: usize,
}

struct GenMetrics {
    avg_fitness: f32,
    max_fitness: f32,
    total_signals: u32,
    iconicity: f32,
    mutual_info: f32,
    jsd_no_pred: f32,
    jsd_pred: f32,
    per_sym_jsd: [f32; NUM_SYMBOLS],
    silence_corr: f32,
    gen_matrix: [[u32; 4]; NUM_SYMBOLS],
    traj_jsd: f32,
    input_mi: [f32; INPUTS],
    contrast: Vec<f32>,
    sender_fit_corr: f32,
    traj_fluct_ratio: f32,
    receiver_fit_corr: f32,
    response_fit_corr: f32,
    silence_onset_jsd: f32,
    silence_move_delta: f32,
    avg_base_hidden: f32,
    min_base_hidden: usize,
    max_base_hidden: usize,
    avg_signal_hidden: f32,
    min_signal_hidden: usize,
    max_signal_hidden: usize,
    zone_deaths: u32,
    signal_entropy: f32,
}

impl GenMetrics {
    fn write_csv(&self, f: &mut impl Write, gen: usize) -> Result<(), Box<dyn std::error::Error>> {
        writeln!(
            f,
            "{gen},{:.1},{:.1},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1},{},{},{:.1},{},{},{},{:.4}",
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
            self.silence_move_delta,
            self.avg_base_hidden,
            self.min_base_hidden,
            self.max_base_hidden,
            self.avg_signal_hidden,
            self.min_signal_hidden,
            self.max_signal_hidden,
            self.zone_deaths,
            self.signal_entropy
        )?;
        Ok(())
    }

    fn write_input_mi(
        &self,
        f: &mut impl Write,
        gen: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        write!(f, "{gen}")?;
        for &v in &self.input_mi {
            write!(f, ",{v:.4}")?;
        }
        writeln!(f)?;
        Ok(())
    }

    fn write_trajectory(
        &self,
        f: &mut impl Write,
        gen: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        write!(f, "{gen}")?;
        for row in &self.gen_matrix {
            for &val in row {
                write!(f, ",{val}")?;
            }
        }
        for &v in &self.per_sym_jsd {
            write!(f, ",{v:.4}")?;
        }
        write!(f, ",{:.4}", self.traj_jsd)?;
        for v in &self.contrast {
            write!(f, ",{v:.4}")?;
        }
        writeln!(f)?;
        Ok(())
    }

    fn print_log(&self, gen: usize) {
        let sym_str: String = self
            .per_sym_jsd
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "gen {gen:>4} | avg {:>7.1} | max {:>7.1} | signals {} | icon {:.3} | MI {:.3} | ent {:.3} | jsd {:.3}/{:.3} | sym [{sym_str}] | sil {:.3} | zd {} | base {:.1} [{}-{}] | sig {:.1} [{}-{}]",
            self.avg_fitness, self.max_fitness, self.total_signals,
            self.iconicity, self.mutual_info, self.signal_entropy,
            self.jsd_no_pred, self.jsd_pred,
            self.silence_corr,
            self.zone_deaths,
            self.avg_base_hidden, self.min_base_hidden, self.max_base_hidden,
            self.avg_signal_hidden, self.min_signal_hidden, self.max_signal_hidden
        );
    }
}

struct EvalResult {
    fitness: Vec<f32>,
    signal_events: Vec<world::SignalEvent>,
    total_signals: u32,
    ticks_in_zone: u32,
    prey_ticks: u32,
    receiver_counts: [[[u32; 5]; 2]; 1 + NUM_SYMBOLS],
    signals_per_tick: Vec<f32>,
    alive_per_tick: Vec<f32>,
    min_zone_dist: Vec<f32>,
    signal_rate_per_prey: Vec<f32>,
    actions_with_signal: Vec<[[u32; 5]; 2]>,
    actions_without_signal: Vec<[[u32; 5]; 2]>,
    silence_onset_actions: Vec<[[u32; 5]; 2]>,
    zone_deaths: u32,
}

fn evaluate_generation(
    population: &[Agent],
    rng: &mut ChaCha8Rng,
    params: &SimParams,
    collect_metrics: bool,
) -> EvalResult {
    let mut world = World::new_with_positions(
        population,
        params.num_zones,
        rng,
        params.no_signals,
        collect_metrics,
        params.grid_size,
        params.food_count,
        params.signal_range,
        params.zone_radius,
        params.zone_speed,
        params.base_drain,
        params.neuron_cost,
        params.signal_cost,
        params.patch_ratio,
        params.zone_drain_rate,
    );

    for _ in 0..params.ticks_per_eval {
        if !world.any_alive() {
            break;
        }
        world.step(rng);
        if params.fast_fail_tick > 0 && world.tick == params.fast_fail_tick {
            for p in &mut world.prey {
                if p.alive && p.energy <= 0.3 && p.food_eaten == 0 {
                    p.alive = false;
                }
            }
        }
    }

    let fitness: Vec<f32> = world
        .prey
        .iter()
        .map(|p| p.ticks_alive as f32 + p.food_eaten as f32 * 10.0)
        .collect();

    let signal_rate_per_prey: Vec<f32> = {
        let mut emit_counts = vec![0u32; world.prey.len()];
        for e in &world.signal_events {
            emit_counts[e.emitter_idx] += 1;
        }
        world
            .prey
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if p.ticks_alive == 0 {
                    0.0
                } else {
                    emit_counts[i] as f32 / p.ticks_alive as f32
                }
            })
            .collect()
    };

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
        ticks_in_zone: world.ticks_in_zone,
        prey_ticks: world.total_prey_ticks,
        receiver_counts: world.receiver_counts,
        signals_per_tick: world.signals_per_tick.iter().map(|&s| s as f32).collect(),
        alive_per_tick: world.alive_per_tick.iter().map(|&a| a as f32).collect(),
        min_zone_dist: world.min_zone_dist_per_tick,
        signal_rate_per_prey,
        actions_with_signal,
        actions_without_signal,
        silence_onset_actions,
        zone_deaths: world.zone_deaths,
    }
}

#[allow(clippy::too_many_lines)]
fn compute_gen_metrics(
    ev: &EvalResult,
    population: &[Agent],
    prev_norm_matrix: &mut Option<[[f32; 4]; NUM_SYMBOLS]>,
    traj_jsd_history: &mut Vec<f32>,
    params: &SimParams,
) -> GenMetrics {
    let avg_fitness = ev.fitness.iter().sum::<f32>() / ev.fitness.len() as f32;
    let max_fitness = ev.fitness.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let iconicity = metrics::compute_iconicity(
        &ev.signal_events,
        ev.ticks_in_zone,
        ev.prey_ticks,
        params.zone_radius,
    );
    let mutual_info = metrics::compute_mutual_info(&ev.signal_events, &params.mi_bins);
    let signal_entropy = metrics::compute_signal_entropy(&ev.signal_events);
    let (jsd_no_pred, jsd_pred) = metrics::compute_receiver_jsd(&ev.receiver_counts);
    let per_sym_jsd = metrics::compute_per_symbol_jsd(&ev.receiver_counts);
    let normalized_signal_rate: Vec<f32> = ev
        .signals_per_tick
        .iter()
        .zip(&ev.alive_per_tick)
        .map(|(&s, &a)| if a > 0.0 { s / a } else { 0.0 })
        .collect();
    let silence_corr = metrics::pearson(&normalized_signal_rate, &ev.min_zone_dist);
    let input_mi = metrics::compute_input_mi(&ev.signal_events);
    let gen_matrix = metrics::signal_context_matrix(&ev.signal_events, &params.mi_bins);
    let curr_norm = metrics::normalize_matrix(&gen_matrix);
    let traj_jsd = match (&*prev_norm_matrix, &curr_norm) {
        (Some(prev), Some(curr)) => metrics::trajectory_jsd(prev, curr),
        _ => 0.0,
    };
    let n_pairs = NUM_SYMBOLS * (NUM_SYMBOLS - 1) / 2;
    let contrast = curr_norm
        .as_ref()
        .map_or(vec![0.0; n_pairs], metrics::inter_symbol_jsd);
    if let Some(norm) = curr_norm {
        *prev_norm_matrix = Some(norm);
    }

    let sender_fit_corr = metrics::pearson(&ev.signal_rate_per_prey, &ev.fitness);

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
    let receiver_fit_corr = metrics::pearson(&reception_rates, &ev.fitness);

    let per_prey_jsd_vec: Vec<f32> = ev
        .actions_with_signal
        .iter()
        .zip(&ev.actions_without_signal)
        .map(|(w, wo)| metrics::per_prey_receiver_jsd(w, wo, MIN_RECEIVER_SAMPLES))
        .collect();
    let response_fit_corr = metrics::pearson(&per_prey_jsd_vec, &ev.fitness);

    let (silence_onset_jsd, silence_move_delta) =
        metrics::compute_silence_onset_metrics(&ev.silence_onset_actions, &ev.actions_with_signal);

    // Brain size stats - split into base and signal
    let base_sizes: Vec<usize> = population
        .iter()
        .map(|a| a.brain.base_hidden_size)
        .collect();
    let avg_base_hidden = base_sizes.iter().sum::<usize>() as f32 / base_sizes.len() as f32;
    let min_base_hidden = base_sizes.iter().copied().min().unwrap_or(0);
    let max_base_hidden = base_sizes.iter().copied().max().unwrap_or(0);

    let sig_sizes: Vec<usize> = population
        .iter()
        .map(|a| a.brain.signal_hidden_size)
        .collect();
    let avg_signal_hidden = sig_sizes.iter().sum::<usize>() as f32 / sig_sizes.len() as f32;
    let min_signal_hidden = sig_sizes.iter().copied().min().unwrap_or(0);
    let max_signal_hidden = sig_sizes.iter().copied().max().unwrap_or(0);

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
        avg_base_hidden,
        min_base_hidden,
        max_base_hidden,
        avg_signal_hidden,
        min_signal_hidden,
        max_signal_hidden,
        zone_deaths: ev.zone_deaths,
        signal_entropy,
    }
}

#[allow(clippy::too_many_lines)]
fn run_seed(
    seed: u64,
    generations: usize,
    params: &SimParams,
    write_csv: bool,
) -> Result<RunResult, Box<dyn std::error::Error>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut population: Vec<Agent> = (0..params.pop_size)
        .map(|_| Agent {
            brain: Brain::random(&mut rng),
            x: rng.gen_range(0..params.grid_size),
            y: rng.gen_range(0..params.grid_size),
            parent_indices: [None, None],
            grandparent_indices: [None; 4],
        })
        .collect();

    let mut csv = write_csv
        .then(|| File::create("output.csv").map(BufWriter::new))
        .transpose()?;
    let mut traj_csv = write_csv
        .then(|| File::create("trajectory.csv").map(BufWriter::new))
        .transpose()?;
    let mut input_mi_csv = write_csv
        .then(|| File::create("input_mi.csv").map(BufWriter::new))
        .transpose()?;

    if let Some(ref mut f) = csv {
        writeln!(f, "generation,avg_fitness,max_fitness,signals_emitted,iconicity,mutual_info,jsd_no_pred,jsd_pred,silence_corr,sender_fit_corr,traj_fluct_ratio,receiver_fit_corr,response_fit_corr,silence_onset_jsd,silence_move_delta,avg_base_hidden,min_base_hidden,max_base_hidden,avg_signal_hidden,min_signal_hidden,max_signal_hidden,zone_deaths,signal_entropy")?;
    }
    if let Some(ref mut f) = traj_csv {
        write!(f, "generation")?;
        for s in 0..NUM_SYMBOLS {
            for d in 0..4 {
                write!(f, ",s{s}d{d}")?;
            }
        }
        for s in 0..NUM_SYMBOLS {
            write!(f, ",jsd_sym{s}")?;
        }
        write!(f, ",trajectory_jsd")?;
        for i in 0..NUM_SYMBOLS {
            for j in (i + 1)..NUM_SYMBOLS {
                write!(f, ",contrast_{i}{j}")?;
            }
        }
        writeln!(f)?;
    }
    if let Some(ref mut f) = input_mi_csv {
        write!(f, "generation")?;
        for name in &INPUT_NAMES {
            write!(f, ",mi_{name}")?;
        }
        writeln!(f)?;
    }

    let mut last_result = RunResult {
        final_matrix: [[0; 4]; NUM_SYMBOLS],
        avg_fitness: 0.0,
        max_fitness: 0.0,
        mutual_info: 0.0,
        sender_fit_corr: 0.0,
        receiver_fit_corr: 0.0,
        response_fit_corr: 0.0,
        zone_deaths: 0,
        generations_completed: 0,
    };
    let mut prev_norm_matrix: Option<[[f32; 4]; NUM_SYMBOLS]> = None;
    let mut traj_jsd_history: Vec<f32> = Vec::new();

    for gen in 0..generations {
        let is_metrics_gen = gen % params.metrics_interval == 0 || gen == generations - 1;
        let ev = evaluate_generation(&population, &mut rng, params, is_metrics_gen);

        // Apply kin fitness bonus
        let mut fitness = ev.fitness.clone();
        if params.kin_bonus > 0.0 {
            // Sparse kin fitness: build index of parent/grandparent -> agent indices
            let mut parent_to_agents: HashMap<usize, Vec<usize>> = HashMap::new();
            let mut grandparent_to_agents: HashMap<usize, Vec<usize>> = HashMap::new();
            for (i, agent) in population.iter().enumerate() {
                for &p in &agent.parent_indices {
                    if let Some(p) = p {
                        parent_to_agents.entry(p).or_default().push(i);
                    }
                }
                for &g in &agent.grandparent_indices {
                    if let Some(g) = g {
                        grandparent_to_agents.entry(g).or_default().push(i);
                    }
                }
            }
            for i in 0..population.len() {
                let mut kin_sum = 0.0_f32;
                // Track siblings to avoid double-counting as cousins
                let mut seen_sibling = vec![false; population.len()];
                // Siblings: agents sharing a parent (r=0.5)
                for &p in &population[i].parent_indices {
                    if let Some(p) = p {
                        if let Some(siblings) = parent_to_agents.get(&p) {
                            for &j in siblings {
                                if j != i && !seen_sibling[j] && fitness[j] > 0.0 {
                                    kin_sum += 0.5;
                                    seen_sibling[j] = true;
                                }
                            }
                        }
                    }
                }
                // Cousins: agents sharing a grandparent but not already siblings (r=0.25)
                let mut seen_cousin = vec![false; population.len()];
                for &g in &population[i].grandparent_indices {
                    if let Some(g) = g {
                        if let Some(cousins) = grandparent_to_agents.get(&g) {
                            for &j in cousins {
                                if j != i && !seen_sibling[j] && !seen_cousin[j] && fitness[j] > 0.0
                                {
                                    kin_sum += 0.25;
                                    seen_cousin[j] = true;
                                }
                            }
                        }
                    }
                }
                fitness[i] += params.kin_bonus * kin_sum;
            }
        }

        let mut scored: Vec<(usize, f32)> =
            fitness.iter().enumerate().map(|(i, &f)| (i, f)).collect();

        if is_metrics_gen {
            // Use original fitness (without kin bonus) for metrics to avoid confounding
            let ev_for_metrics = EvalResult {
                fitness: ev.fitness,
                signal_events: ev.signal_events,
                total_signals: ev.total_signals,
                ticks_in_zone: ev.ticks_in_zone,
                prey_ticks: ev.prey_ticks,
                receiver_counts: ev.receiver_counts,
                signals_per_tick: ev.signals_per_tick,
                alive_per_tick: ev.alive_per_tick,
                min_zone_dist: ev.min_zone_dist,
                signal_rate_per_prey: ev.signal_rate_per_prey,
                actions_with_signal: ev.actions_with_signal,
                actions_without_signal: ev.actions_without_signal,
                silence_onset_actions: ev.silence_onset_actions,
                zone_deaths: ev.zone_deaths,
            };

            let gm = compute_gen_metrics(
                &ev_for_metrics,
                &population,
                &mut prev_norm_matrix,
                &mut traj_jsd_history,
                params,
            );

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
                sender_fit_corr: gm.sender_fit_corr,
                receiver_fit_corr: gm.receiver_fit_corr,
                response_fit_corr: gm.response_fit_corr,
                zone_deaths: gm.zone_deaths,
                generations_completed: gen + 1,
            };
        } else if write_csv && gen.is_multiple_of(10) {
            // Lightweight progress log on non-metrics gens
            let avg = ev.fitness.iter().sum::<f32>() / ev.fitness.len() as f32;
            let max = ev.fitness.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let base_sizes: Vec<usize> = population
                .iter()
                .map(|a| a.brain.base_hidden_size)
                .collect();
            let avg_base = base_sizes.iter().sum::<usize>() as f32 / base_sizes.len() as f32;
            let min_base = base_sizes.iter().copied().min().unwrap_or(0);
            let max_base = base_sizes.iter().copied().max().unwrap_or(0);
            let sig_sizes: Vec<usize> = population
                .iter()
                .map(|a| a.brain.signal_hidden_size)
                .collect();
            let avg_sig = sig_sizes.iter().sum::<usize>() as f32 / sig_sizes.len() as f32;
            let min_sig = sig_sizes.iter().copied().min().unwrap_or(0);
            let max_sig = sig_sizes.iter().copied().max().unwrap_or(0);
            println!(
                "gen {:>4} | avg {:>7.1} | max {:>7.1} | signals {} | base {:.1} [{}-{}] | sig {:.1} [{}-{}]",
                gen, avg, max, ev.total_signals,
                avg_base, min_base, max_base,
                avg_sig, min_sig, max_sig
            );
        }

        population = evolution::evolve_spatial(
            &population,
            &mut scored,
            params.elite_count,
            params.tournament_size,
            params.mutation_sigma,
            params.grid_size,
            params.reproduction_radius,
            params.fallback_radius,
            &mut rng,
        );
    }

    // Flush buffered writers
    if let Some(ref mut f) = csv {
        f.flush()?;
    }
    if let Some(ref mut f) = traj_csv {
        f.flush()?;
    }
    if let Some(ref mut f) = input_mi_csv {
        f.flush()?;
    }

    if write_csv {
        println!("Done. Results in output.csv, trajectory.csv, input_mi.csv");
    }
    Ok(last_result)
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let params = SimParams::from_cli(&args);
    let batch_mode = args.iter().position(|a| a == "--batch");

    if let Some(pos) = batch_mode {
        let n: usize = args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or(5);
        let generations: usize = args
            .get(pos + 2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);

        println!(
            "Config: pop={} grid={} zones={} radius={:.1} speed={:.1} drain={:.3} food={} ticks={} patches={:.0}% kin_bonus={:.2} sig_cost={:.4}",
            params.pop_size,
            params.grid_size,
            params.num_zones,
            params.zone_radius,
            params.zone_speed,
            params.zone_drain_rate,
            params.food_count,
            params.ticks_per_eval,
            params.patch_ratio * 100.0,
            params.kin_bonus,
            params.signal_cost
        );
        println!("Batch mode: {n} seeds x {generations} generations");
        let mut results: Vec<RunResult> = Vec::new();
        for seed in 0..n as u64 {
            println!("--- seed {seed} ---");
            results.push(run_seed(seed, generations, &params, false)?);
        }

        let norm_matrices: Vec<Option<[[f32; 4]; NUM_SYMBOLS]>> = results
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
        let positional: Vec<&String> = args[1..]
            .iter()
            .filter(|a| !a.starts_with("--"))
            .filter(|a| {
                // Skip values that follow -- flags
                let idx = args.iter().position(|x| x == *a).unwrap_or(0);
                idx == 0
                    || !args
                        .get(idx.wrapping_sub(1))
                        .is_some_and(|prev| prev.starts_with("--"))
            })
            .collect();
        let seed: u64 = positional
            .first()
            .and_then(|s| s.parse().ok())
            .unwrap_or(42);
        let generations: usize = positional
            .get(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);

        println!(
            "Config: pop={} grid={} zones={} radius={:.1} speed={:.1} drain={:.3} food={} ticks={} patches={:.0}% kin_bonus={:.2} sig_cost={:.4}",
            params.pop_size,
            params.grid_size,
            params.num_zones,
            params.zone_radius,
            params.zone_speed,
            params.zone_drain_rate,
            params.food_count,
            params.ticks_per_eval,
            params.patch_ratio * 100.0,
            params.kin_bonus,
            params.signal_cost
        );

        let probe_mode = args.iter().any(|a| a == "--probe");
        if probe_mode {
            let zone_coverage =
                params.num_zones as f32 * std::f32::consts::PI * params.zone_radius.powi(2)
                    / (params.grid_size as f32).powi(2);

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let population: Vec<Agent> = (0..params.pop_size)
                .map(|_| Agent {
                    brain: Brain::random(&mut rng),
                    x: rng.gen_range(0..params.grid_size),
                    y: rng.gen_range(0..params.grid_size),
                    parent_indices: [None; 2],
                    grandparent_indices: [None; 4],
                })
                .collect();
            let ev = evaluate_generation(&population, &mut rng, &params, true);
            let avg_fitness: f32 = ev.fitness.iter().sum::<f32>() / ev.fitness.len() as f32;
            let max_fitness = ev.fitness.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let alive_count = ev.fitness.iter().filter(|&&f| f > 0.0).count();
            let mi = metrics::compute_mutual_info(&ev.signal_events, &params.mi_bins);

            println!("--- Probe (seed {seed}, random brains) ---");
            println!("  Zone coverage: {:.1}%", zone_coverage * 100.0);
            println!("  Floor fitness: avg={avg_fitness:.1} max={max_fitness:.1}");
            println!(
                "  Survival: {}/{} alive ({:.0}%)",
                alive_count,
                params.pop_size,
                alive_count as f32 / params.pop_size as f32 * 100.0
            );
            println!("  Zone deaths: {}", ev.zone_deaths);
            println!("  Signals emitted: {}", ev.total_signals);
            println!("  MI(signal;zone): {mi:.4}");
            return Ok(());
        }

        if params.no_signals {
            println!("Counterfactual mode: signals disabled");
        }
        let result = run_seed(seed, generations, &params, true)?;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open("runs.tsv")?;
        writeln!(
            log,
            "{timestamp}\tunknown\t{seed}\t{gens}\t{pop}\t{grid}\t{zones}\t{radius:.1}\t{speed:.1}\t{food}\t{ticks}\t{patches:.2}\t{kin}\t{ff}\t{avg:.1}\t{max:.1}\t{mi:.4}\t{sfc:.4}\t{rfc:.4}\t{rpfc:.4}\t{zd}",
            gens = result.generations_completed,
            pop = params.pop_size,
            grid = params.grid_size,
            zones = params.num_zones,
            radius = params.zone_radius,
            speed = params.zone_speed,
            food = params.food_count,
            ticks = params.ticks_per_eval,
            patches = params.patch_ratio,
            kin = params.kin_bonus,
            ff = params.fast_fail_tick,
            avg = result.avg_fitness,
            max = result.max_fitness,
            mi = result.mutual_info,
            sfc = result.sender_fit_corr,
            rfc = result.receiver_fit_corr,
            rpfc = result.response_fit_corr,
            zd = result.zone_deaths,
        )?;
    }

    Ok(())
}
