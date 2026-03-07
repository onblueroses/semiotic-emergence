# semiotic-emergence

A semiotic observatory for watching meaning emerge in a minimal evolutionary world. See [FRAMEWORK.md](FRAMEWORK.md) for the governing intellectual framework.

**FRAMEWORK.md is the centerpiece of this project.** Read it before making any change. Every modification - new metrics, parameter tuning, architectural decisions, new code - must be evaluated against the five development principles in FRAMEWORK.md. If a change doesn't help us see something new about semiotic emergence, it doesn't belong here.

## Commands

```bash
cargo build
cargo run -- [seed] [generations]
cargo test
cargo clippy --all-targets -- -D warnings
```

## Structure

```
FRAMEWORK.md      - Governing intellectual framework (READ FIRST)
src/brain.rs      - NN forward pass (16->6->8, 158 weights)
src/evolution.rs  - GA: tournament select, crossover, mutation
src/world.rs      - Grid, prey/predator structs, tick loop
src/signal.rs     - 3-symbol broadcast, distance decay, 1-tick delay
src/main.rs       - Generation loop, CSV output
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
- Receiver-side effects are unmeasured (the critical blind spot)

## Development priorities (from FRAMEWORK.md)

1. Receiver response spectrum - does signal content change receiver behavior?
2. Silence detection - is the zero sign functional?
3. Semiotic trajectory - how does the signal-meaning mapping evolve over time?
4. Cross-population divergence - convention vs. constraint?
5. Counterfactual value - does the signal channel improve fitness?
