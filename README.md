# semiotic-emergence

Evolutionary simulation exploring whether a communication system can emerge from natural selection alone. Prey with evolvable neural networks share a grid with predators. They can broadcast one of six symbols to neighbors. At generation zero, those symbols mean nothing.

[FRAMEWORK.md](FRAMEWORK.md) has the semiotic framework behind the project.

## Setup

384 prey on a 56x56 toroidal grid with 3 predators and 100 food sources. Each prey has a split-head neural network with two independently evolvable hidden layers: a base layer (4-64 neurons) shared across all outputs, and a dedicated signal layer (2-32 neurons) for communication. The brain takes 36 inputs and produces movement decisions, signal emissions, and memory updates.

Vision range is ~5.6 cells, signal range ~22.4 cells. The 4:1 ratio is the core design choice: signals reach far beyond sight, forcing heavy reliance on social information for early predator warning.

Key mechanics:

- **Split-head architecture** - base hidden layer feeds movement and memory; signal hidden layer gives evolution independent control over communication vs. locomotion
- **Recurrent memory** - 8 memory cells updated each tick via EMA, fed back as inputs, enabling temporal reasoning
- **Cooperative food patches** - 50% of food requires 2+ nearby prey to harvest, rewarding spatial coordination
- **Kin fitness** - siblings (0.5) and cousins (0.25) boost each other's selection score, supporting altruistic signaling
- **Softmax emission** - signals emitted when one symbol dominates the softmax distribution (above 1/6 uniform baseline)
- **Signal cost** - 0.002 energy per emission creates selective pressure against noise
- **Lineage tracking** - parent and grandparent indices enable relatedness computation without artificial grouping

## Run it

```bash
cargo run --release -- [seed] [generations]       # normal run
cargo run --release -- 42 1000                    # seed 42, 1000 gens
cargo run --release -- 42 1000 --no-signals       # counterfactual (signals suppressed)
cargo run --release -- --batch 10 300             # cross-population divergence
```

CLI flags: `--pop N`, `--grid N`, `--pred N`, `--food N`, `--ticks N`, `--patch-ratio F`, `--kin-bonus F`.

Output: `output.csv` (21 columns), `trajectory.csv`, `input_mi.csv`. Batch mode also writes `divergence.csv`.

## Metrics

Per-generation CSV tracks 21 columns including:

- **MI** (mutual information) - does symbol choice correlate with sender context (predator distance)?
- **JSD** (Jensen-Shannon divergence) - do receivers change behavior depending on which signal they get?
- **Silence correlation** - do prey go quiet near predators?
- **Iconicity** - are signals spatially concentrated near danger?
- **Fitness coupling** - does signaling behavior correlate with survival?
- **Brain stats** - base and signal hidden layer sizes (avg/min/max) tracked separately
- **Counterfactual value** - fitness delta between signal and no-signal runs (via `--no-signals`)

Full causal chain for genuine communication requires all three simultaneously: sender encodes context (MI), receiver differentiates behavior (JSD), and that differentiation improves fitness.

## The code

~4000 lines of Rust across six files:

```
src/brain.rs      - Split-head NN (36 inputs, base hidden 4-64, signal hidden 2-32, 5491-weight genome)
src/evolution.rs  - Spatial evolution, lineage tracking, scoped mutation per hidden layer
src/world.rs      - Grid, predators, food patches, memory, energy economy
src/signal.rs     - Six symbols, softmax emission, linear decay, one-tick delay
src/metrics.rs    - 10 instruments: MI, JSD, silence, trajectory, divergence, fitness coupling
src/main.rs       - Generation loop, kin fitness, batch mode, counterfactual mode, 21-column CSV
```

## Findings so far

See [FINDINGS.md](FINDINGS.md) for full analysis across runs.

**Runs 1-2** (84k and 68k generations): Signals evolved as a survival resource, not a communication medium - an evasion boost mechanic rewarded signal presence regardless of content. The full causal chain completed in <0.1% of generations. This led to removing the evasion boost, adding signal costs, and reducing predator density.

**Run 3** (100k generations, seed 42): MI peaked at 0.624 during a sustained surge (gen 75k-100k) after brain size naturally expanded to 15 neurons. Signals encoded predator distance (MI 0.472) and direction, with evidence of signal relaying. But the architecture was a single hidden layer - signal and movement outputs were coupled, producing spandrels. Brain expansion also showed the convention collapsed during the growth transition (gen 46k-50k).

**Current architecture** (v2): Split-head brain separates signal processing from movement, 6 symbols instead of 3, recurrent memory, cooperative food patches, and kin fitness. Designed to address the coupling problem and convention fragility observed in run 3.
