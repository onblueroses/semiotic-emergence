# Data

Raw simulation output organized by experimental era. For era descriptions, see [EXPERIMENTS.md](../EXPERIMENTS.md).

## Directory Structure

```
data/
  era1/          Baseline (20x20, 48 prey, 10k gens, seed 42)
```

Additional run data used for figure generation is in `figures/` subdirectories (one per run).

## CSV Column Formats

**output.csv** (25 columns): generation, avg_fitness, max_fitness, signals_emitted, iconicity, mutual_info, jsd_no_pred, jsd_pred, silence_corr, sender_fit_corr, traj_fluct_ratio, receiver_fit_corr, response_fit_corr, silence_onset_jsd, silence_move_delta, avg_base_hidden, min_base_hidden, max_base_hidden, avg_signal_hidden, min_signal_hidden, max_signal_hidden, zone_deaths, signal_entropy, freeze_zone_deaths, food_mi

**trajectory.csv** (47 columns): generation, s0d0..s5d3 (24 contingency counts), jsd_sym0..jsd_sym5 (6 per-symbol JSD), trajectory_jsd, contrast_01..contrast_45 (15 pairwise symbol contrasts)

**input_mi.csv** (40 columns): generation, mi_zone_damage, mi_energy_delta, mi_freeze_pressure, mi_food_dx, mi_food_dy, mi_food_dist, mi_ally_dx, mi_ally_dy, mi_ally_dist, mi_sig0_str..mi_sig5_dy (18 signal inputs), mi_mem0..mi_mem7 (8 memory), mi_energy, mi_death_nearby, mi_death_dx, mi_death_dy

Earlier eras may have fewer columns (e.g., Era 1 predates the split-head architecture and some metrics). The analysis tool (`analyze.py`) auto-detects format from headers.
