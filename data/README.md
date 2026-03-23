# Data

Simulation output from all significant runs. For experiment context, see [EXPERIMENTS.md](../EXPERIMENTS.md).

## Runs

**seed42-10k/** - Era 1 baseline. 48 prey, 20x20 grid, 10k generations with fixed 6-neuron brain and 3 symbols.

**baseline-s100/** - Era 5 counterfactual signal run. 384 prey at drain=0.10 (accidentally high), signals enabled, 96k generations.

**mute-s100/** - Era 5 counterfactual mute run. Same config as baseline-s100 but with `--no-signals`. Mute prey 20-25% fitter.

**v6-signal-s300/** - Era 6 freeze+flee zones with signals. 384 prey, drain=0.02, seed 300. Achieved input MI 0.137 on food_dx - strongest structured encoding in the project.

**v6-mute-s300/** - Era 6 mute counterfactual. Same config as v6-signal-s300 without signals. Mute ~8% fitter despite food encoding.

**v7-signals-42/** - Era 7 full-featured run. Pop=1000, drain=0.05, demes=3, death echoes, signal threshold=0.3. Signals -12.8% vs mute.

**v7-mute-42/** - Era 7 mute counterfactual. Same config as v7-signals-42 without signals.

**v10-2k-42/** - Era 10 population scale test. Pop=2000, 100k generations. Signal hidden layers grew to 26.8/32, all 6 symbols active, but response_fit_corr unmeasurable (pre-fix architecture).

**v11-cap6-42/** - Era 9 volume knob experiment (constrained). Pop=384, max-signal-hidden=6. Symbol differentiation is maladaptive: response_fit_corr -0.13 to -0.28.

**v11-cap32-42/** - Era 9 volume knob control. Pop=384, max-signal-hidden=32. response_fit_corr near zero, occasionally +0.16.

**v12-blind6-42/** - Era 12 blind mode. Pop=384, all spatial perception stripped. MI~0, 2 symbols extinct, fitness halved. Memory replaces perception, not signals.

**v13-2k-42/** - Era 13 population scale redux. Pop=2000, 100k generations with fixed response_fit_corr metric. receiver_fit_corr=0.74 (strongest in project), food_mi=0.14, but response_fit_corr=-0.29 (symbol differentiation still maladaptive). 4 of 6 symbols active. Signal strengths interleaved with food inputs in encoding hierarchy.

## CSV Column Formats

**output.csv** (25 columns): generation, avg_fitness, max_fitness, signals_emitted, iconicity, mutual_info, jsd_no_pred, jsd_pred, silence_corr, sender_fit_corr, traj_fluct_ratio, receiver_fit_corr, response_fit_corr, silence_onset_jsd, silence_move_delta, avg_base_hidden, min_base_hidden, max_base_hidden, avg_signal_hidden, min_signal_hidden, max_signal_hidden, zone_deaths, signal_entropy, freeze_zone_deaths, food_mi

**trajectory.csv** (47 columns): generation, s0d0..s5d3 (24 contingency counts), jsd_sym0..jsd_sym5 (6 per-symbol JSD), trajectory_jsd, contrast_01..contrast_45 (15 pairwise symbol contrasts)

**input_mi.csv** (40 columns): generation, mi_zone_damage, mi_energy_delta, mi_freeze_pressure, mi_food_dx, mi_food_dy, mi_food_dist, mi_ally_dx, mi_ally_dy, mi_ally_dist, mi_sig0_str..mi_sig5_dy (18 signal inputs), mi_mem0..mi_mem7 (8 memory), mi_energy, mi_death_nearby, mi_death_dx, mi_death_dy

Earlier runs may have fewer columns (e.g., seed42-10k predates split-head architecture). The analysis tool (`analyze.py`) auto-detects format from headers.
