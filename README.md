# semiotic-emergence

Evolutionary simulation exploring whether a communication system can emerge from natural selection alone. Prey with evolvable neural networks share a grid with predators. They can broadcast one of three symbols to neighbors. At generation zero, those symbols mean nothing.

This is my first project in evolutionary simulation and computational semiotics. The goal is to set up conditions where communication could emerge - an information gap between what prey can see and what signals can reach - and then observe what actually happens without steering the outcome.

[FRAMEWORK.md](FRAMEWORK.md) has the semiotic framework behind the project.

## Setup

384 prey on a 56x56 toroidal grid with 3 predators and 100 food sources. Each prey has an evolvable neural network (16 inputs, 4-124 hidden neurons selected by evolution, 8 outputs). Vision range is ~11 cells, signal range ~22 cells. The 2:1 ratio is the core design choice: signals reach further than sight, so prey that see a predator have information that distant prey lack.

Signals cost energy (0.002 per emission). Spatial reproduction creates natural kin clusters - offspring appear near parents - so communicator genes can spread through kinship proximity without artificial grouping.

## Run it

```bash
cargo run --release -- [seed] [generations]       # normal run
cargo run --release -- 42 300                     # seed 42, 300 gens
cargo run --release -- 42 300 --no-signals        # counterfactual (signals suppressed)
cargo run --release -- --batch 10 300             # cross-population divergence
```

Output: `output.csv`, `trajectory.csv`, `input_mi.csv`. Batch mode also writes `divergence.csv`.

## Metrics

Per-generation CSV tracks:

- **MI** (mutual information) - does symbol choice correlate with sender context (predator distance)?
- **JSD** (Jensen-Shannon divergence) - do receivers change behavior depending on which signal they get?
- **Silence correlation** - do prey go quiet near predators?
- **Iconicity** - are signals spatially concentrated near danger?
- **Fitness coupling** - does signaling behavior correlate with survival?
- **Counterfactual value** - fitness delta between signal and no-signal runs (via `--no-signals`)

Full causal chain for genuine communication requires all three simultaneously: sender encodes context (MI), receiver differentiates behavior (JSD), and that differentiation improves fitness.

## The code

~3300 lines of Rust across six files:

```
src/brain.rs      - Neural network (16 inputs, 4-124 evolvable hidden, 8 outputs)
src/evolution.rs  - Spatial evolution (local tournament, crossover, mutation)
src/world.rs      - Grid, physics, predators, food, energy economy
src/signal.rs     - Three symbols, linear decay, one-tick delay
src/metrics.rs    - MI, JSD, silence, trajectory, divergence instruments
src/main.rs       - Generation loop, batch mode, counterfactual mode, CSV output
```

## Findings so far

Two initial runs (84k and 68k generations) with different parameters found that signals evolved as a survival resource, not a communication medium - an evasion boost mechanic rewarded signal presence regardless of content. The full causal chain (encode + respond + fitness benefit) completed in <0.1% of generations. See [FINDINGS.md](FINDINGS.md) for the full analysis and the parameter changes made for subsequent runs.
