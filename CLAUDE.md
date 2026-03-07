# semiotic-emergence

A semiotic observatory for watching meaning emerge in a minimal evolutionary world. See [FRAMEWORK.md](FRAMEWORK.md) for the governing intellectual framework.

**FRAMEWORK.md is the centerpiece of this project.** Read it before making any change. Every modification - new metrics, parameter tuning, architectural decisions, new code - must be evaluated against the five development principles in FRAMEWORK.md. If a change doesn't help us see something new about semiotic emergence, it doesn't belong here.

## Commands

```bash
cargo build
cargo run --release -- [seed] [gens]                # normal
cargo run --release -- [seed] [gens] --no-signals   # counterfactual (signals suppressed)
cargo run --release -- --batch N [gens]             # cross-population divergence
cargo test
cargo clippy --all-targets -- -D warnings
```

## Structure

```
FRAMEWORK.md      - Governing intellectual framework (READ FIRST)
src/brain.rs      - NN forward pass (16->6->8, 158 weights)
src/evolution.rs  - GA: tournament select, crossover, mutation
src/world.rs      - Grid, prey/predator structs, tick loop, receiver instrumentation
src/signal.rs     - 3-symbol broadcast, distance decay, 1-tick delay
src/metrics.rs    - All instruments: MI, iconicity, JSD, silence, divergence, input MI, contrast, coupling, fluctuation
src/main.rs       - Generation loop, CSV output (3 files), batch mode, counterfactual mode
```

## Key numbers

- Grid: 20x20, wrapping edges (toroidal - all distances use shortest path)
- Population: 48 genomes, evaluated in groups of 8
- Elites: top 8 pass through unchanged
- Tournament size: 3
- Mutation: Gaussian (Box-Muller), sigma=0.1
- Eval rounds: 5 (2 kin-grouped, 3 random-shuffled)
- Ticks per evaluation: 500
- Signal range: 8 cells, linear decay
- Signal cost: 0.01 energy per emission
- Prey vision: 4.0 cells
- Predator speed: 3 cells/tick (prey move 1)
- Confusion: radius 4.0, threshold 3 nearby prey
- Food: 25 items, respawn when < 50%, +0.3 energy each
- Energy: start 1.0, drain 0.002/tick, death at 0
- MI bins: [0-4), [4-8), [8-11), [11+) aligned with vision/signal range

## Invariants

- Single RNG (`ChaCha8Rng`) seeded from CLI arg for reproducibility
- Prey processed in shuffled order each tick (no index bias)
- Signals emitted on tick T receivable from tick T+1, persist up to 4 ticks
- Predator moves 3 cells/tick toward nearest prey (confused by 3+ nearby prey), kills on same cell only
- NN outputs 0-4 = movement/eat (argmax), outputs 5-7 = signal (emit if max > 0.5, costs energy)

## Current semiotic findings

- MI > 0: signal symbols correlate with predator distance (sender-side structure exists)
- Negative iconicity: prey suppress signals near the predator (silence = danger pattern)
- These are Level 1 (index) phenomena - see FRAMEWORK.md hierarchy
- JSD > 0.2 by gen 40: receivers change behavior in response to signals (instrument #1)
- Negative silence_corr: signal rate drops near predator (temporal confirmation, instrument #2)
- Counterfactual mode (--no-signals) confirms signal channel affects population dynamics

## Implemented instruments (from FRAMEWORK.md)

Original five instruments in `src/metrics.rs`:

1. **Receiver response spectrum** - JSD between action distributions with/without signal, per context (jsd_no_pred, jsd_pred)
2. **Silence detection** - Pearson correlation between signals-per-tick and min-predator-distance (silence_corr)
3. **Semiotic trajectory** - Per-generation signal-context matrix evolution in trajectory.csv, trajectory_jsd for phase transitions
4. **Cross-population divergence** - Permutation-aware JSD across seeds (--batch mode, divergence.csv)
5. **Counterfactual value** - --no-signals flag suppresses emission for fitness comparison

### Observatory enrichment (5 additional instruments)

6. **Input telescope** - I(Symbol; X_i) for all 16 input dimensions at emission time, quartile-binned (input_mi.csv)
7. **Social telescope** - Kin vs random round split: mi_kin/mi_rnd, jsd_*_kin/jsd_*_rnd in output.csv
8. **Contrast telescope** - Pairwise JSD between symbols' context distributions: contrast_01/02/12 in trajectory.csv
9. **Fitness coupling** - Pearson(signal_count, fitness) per prey: sender_fit_corr in output.csv. receiver_fit_corr is reserved (always 0.0 - requires per-prey receiver tracking).
10. **Phase transition stats** - Rolling fluctuation ratio on trajectory_jsd: traj_fluct_ratio in output.csv

### SignalEvent enrichment

SignalEvent captures full input vector, kin_round flag, and emitter_idx at emission time. This enables all observatory instruments without modifying the simulation physics.

## CSV output

**output.csv** - One row per generation:
`generation,avg_fitness,max_fitness,signals_emitted,iconicity,mutual_info,confusion_ticks,jsd_no_pred,jsd_pred,silence_corr,mi_kin,mi_rnd,jsd_no_pred_kin,jsd_no_pred_rnd,jsd_pred_kin,jsd_pred_rnd,sender_fit_corr,receiver_fit_corr,traj_fluct_ratio`

**trajectory.csv** - One row per generation:
`generation,s0d0..s2d3,jsd_sym0..jsd_sym2,trajectory_jsd,contrast_01,contrast_02,contrast_12`

**input_mi.csv** - One row per generation:
`generation,mi_pred_dx,mi_pred_dy,mi_pred_dist,mi_food_dx,mi_food_dy,mi_ally_dist,mi_sig0_str..mi_sig2_dy,mi_energy`

**divergence.csv** - NxN matrix from `--batch` mode.
