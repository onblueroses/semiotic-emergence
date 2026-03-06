# predator-prey

Minimal evolutionary simulation. Prey with neural network brains evolve on a 20x20 grid under predator pressure. Learning project for genetic algorithms and emergent communication.

## Commands

```bash
cargo build
cargo run -- [seed] [generations]
cargo test
cargo clippy --all-targets -- -D warnings
```

## Structure

```
src/brain.rs      - NN forward pass (16->6->8, 158 weights)
src/evolution.rs  - GA: tournament select, crossover, mutation
src/world.rs      - Grid, prey/predator structs, tick loop
src/signal.rs     - 3-symbol broadcast, distance decay, 1-tick delay
src/main.rs       - Generation loop, CSV output
```

## Key numbers

- Grid: 20x20, wrapping edges
- Population: 40 genomes, evaluated in groups of 8
- Elites: top 8 pass through unchanged
- Tournament size: 3
- Mutation: Gaussian, sigma=0.1
- Ticks per evaluation: 500
- Signal range: 8 cells, linear decay
- Prey vision: 4.0 cells
- Predator speed: 2 cells/tick (prey move 1)
- Confusion: radius 4.0, threshold 3 nearby prey
- Food: 25 items, respawn when < 50%

## Invariants

- Single RNG (`ChaCha8Rng`) seeded from CLI arg for reproducibility
- Signals emitted on tick T receivable on tick T+1 only
- Predator moves 2 cells/tick toward nearest prey (confused by 3+ nearby prey), kills within 1 cell
- NN outputs 0-4 = movement/eat (argmax), outputs 5-7 = signal (emit if max > 0.5)
