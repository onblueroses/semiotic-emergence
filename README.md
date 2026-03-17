# semiotic-emergence

Evolutionary simulation exploring whether a communication system can emerge from natural selection alone. Prey with evolvable neural networks navigate a toroidal grid with invisible kill zones - regions that silently drain energy until death. They can broadcast one of six symbols to neighbors. At generation zero, those symbols mean nothing.

[FRAMEWORK.md](FRAMEWORK.md) has the semiotic framework behind the project.

## Setup

Default: 384 prey on a 56x56 toroidal grid (long runs use 1000 prey on 72x72). 3 flee zones + 2 freeze zones, 100 food sources. Each prey has a split-head neural network with two independently evolvable hidden layers: a base layer (4-64 neurons) shared across all outputs, and a dedicated signal layer (2-32 neurons) for communication. The brain takes 39 inputs and produces movement decisions, signal emissions, and memory updates.

Kill zones are invisible circular regions that drift randomly across the grid. Flee zones drain energy on a gradient from center to edge - prey must move away to survive. Freeze zones penalize movement (3x damage when moving, 0.1x when still) - prey must stay still. The two incompatible optimal responses prevent a single hardcoded strategy from working everywhere.

Prey cannot see zones directly; they sense danger through body-state inputs: accumulated zone damage (pain), energy delta, and freeze pressure. Death witness inputs (intensity + direction to nearest recent zone death within signal range) create a 3-tier information chain: witnesses who saw a death > signal receivers who heard about it > uninformed prey. Signals carry directional information (dx/dy), providing the only way to know which direction to flee.

Vision range is ~5.6 cells, signal range ~22.4 cells. The 4:1 ratio is the core design choice: signals reach far beyond sight, making social information the primary source of spatial awareness.

Key mechanics:

- **Invisible kill zones** - flee zones (drain on gradient) + freeze zones (penalize movement). Prey feel pain but can't see zones
- **Death witness inputs** - prey near zone deaths get directional info about danger (30-tick echo with spatial+temporal decay)
- **Split-head architecture** - base hidden layer feeds movement and memory; signal hidden layer gives evolution independent control over communication vs. locomotion
- **Recurrent memory** - 8 memory cells updated each tick via EMA, fed back as inputs, enabling temporal reasoning
- **Cooperative food patches** - 50% of food requires 2+ nearby prey to harvest, rewarding spatial coordination
- **Kin fitness** - siblings (0.5) and cousins (0.25) boost each other's selection score, supporting altruistic signaling
- **Deme-based group selection** - grid divided into N x N demes with within-deme evolution, inter-deme migration, and periodic group selection (bottom demes lose agents to top demes)
- **Configurable signal threshold** - softmax emission threshold (default 1/6, higher values make silence the default)
- **Signal cost** - configurable energy per emission creates selective pressure against noise
- **Lineage tracking** - parent and grandparent indices enable relatedness computation without artificial grouping

## Run it

```bash
cargo run --release -- [seed] [generations]       # normal run
cargo run --release -- 42 1000                    # seed 42, 1000 gens
cargo run --release -- 42 1000 --no-signals       # counterfactual (signals suppressed)
cargo run --release -- --batch 10 300             # cross-population divergence
cargo run --release -- 42 100000 --demes 3 --signal-threshold 0.3  # with group selection
```

CLI flags: `--pop N`, `--grid N`, `--pred N` (flee zones), `--freeze-zones N`, `--food N`, `--ticks N`, `--zone-radius F`, `--zone-speed F`, `--zone-drain F`, `--zone-coverage F`, `--signal-cost F`, `--signal-range F`, `--signal-threshold F`, `--patch-ratio F`, `--kin-bonus F`, `--demes N`, `--migration-rate F`, `--group-interval N`, `--checkpoint-interval N`, `--resume PATH`, `--metrics-interval N`.

Output: `output.csv` (24 columns), `trajectory.csv`, `input_mi.csv`. Batch mode also writes `divergence.csv`.

## Metrics

Per-generation CSV tracks 24 columns including:

- **MI** (mutual information) - does symbol choice correlate with sender context (zone distance)?
- **JSD** (Jensen-Shannon divergence) - do receivers change behavior depending on which signal they get?
- **Silence correlation** - do prey go quiet inside kill zones?
- **Iconicity** - are signals spatially concentrated near/inside zones?
- **Fitness coupling** - does signaling behavior correlate with survival?
- **Brain stats** - base and signal hidden layer sizes (avg/min/max) tracked separately
- **Counterfactual value** - fitness delta between signal and no-signal runs (via `--no-signals`)

Full causal chain for genuine communication requires all three simultaneously: sender encodes context (MI), receiver differentiates behavior (JSD), and that differentiation improves fitness.

## The code

~6400 lines of Rust across six files:

```
src/brain.rs      - Split-head NN (39 inputs, base hidden 4-64, signal hidden 2-32, 5683-weight genome)
src/evolution.rs  - Spatial evolution, deme-based group selection, migration, lineage tracking
src/world.rs      - Grid, invisible kill zones (flee + freeze), death echoes, food patches, energy economy
src/signal.rs     - Six symbols, configurable softmax threshold, linear decay, one-tick delay
src/metrics.rs    - 10 instruments: MI, JSD, silence, trajectory, divergence, fitness coupling
src/main.rs       - Generation loop, kin fitness, deme dynamics, batch mode, 24-column CSV
```

## Findings so far

See [FINDINGS.md](FINDINGS.md) for full analysis across runs.

**Runs 1-2** (84k and 68k generations): Signals evolved as a survival resource, not a communication medium - an evasion boost mechanic rewarded signal presence regardless of content. The full causal chain completed in <0.1% of generations. This led to removing the evasion boost, adding signal costs, and reducing predator density.

**Run 3** (100k generations, seed 42): MI peaked at 0.624 during a sustained surge (gen 75k-100k) after brain size naturally expanded to 15 neurons. Signals encoded predator distance (MI 0.472) and direction, with evidence of signal relaying. But the architecture was a single hidden layer - signal and movement outputs were coupled, producing spandrels. Brain expansion also showed the convention collapsed during the growth transition (gen 46k-50k).

**Architecture v2**: Split-head brain separates signal processing from movement, 6 symbols instead of 3, recurrent memory, cooperative food patches, and kin fitness. Designed to address the coupling problem and convention fragility observed in run 3.

**Kill zones** (v4): Visible predators replaced with invisible kill zones. Zones are circular regions that drift randomly. Prey cannot see zones; body-state inputs (zone damage, energy delta, freeze pressure) provide indirect sensation. Flee zones drain energy on a gradient; freeze zones penalize movement. The incompatible optimal responses prevent hardcoded strategies.

**v7** (current): Three structural barriers to communication identified from v6 analysis (98k gens, MI~0): no information asymmetry, individual-only selection, trivially easy emission. v7 addresses all three: death witness inputs (prey near zone deaths get directional info about danger), deme-based group selection (multi-level selection rewarding communicating demes), and configurable signal threshold (higher values make silence the default, giving signals dynamic range). 39 brain inputs (up from 36), 5683-weight genome. First runs in progress.
