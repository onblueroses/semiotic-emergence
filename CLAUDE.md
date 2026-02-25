# predator-prey-evolution-communication

Evolutionary simulation where prey evolve emergent communication via NEAT neuroevolution. Predators apply selection pressure; prey that coordinate survive longer and reproduce.

## Commands

```bash
cargo ca          # check --all-targets (fast compile check)
cargo lint        # clippy --all-targets -- -D warnings
cargo ta          # test --all-targets
cargo fmt         # format all code
cargo doc --no-deps  # build docs, catch missing doc links
```

## Architecture

Full design: `ARCHITECTURE.md` (~96KB). Section index:

1. Design Philosophy - data flows down, events flow up
2. Module Structure - 7 modules + viz behind feature flag
3. Core Data Structures - genome, network, world, agent, signal
4. Simulation Loop - tick lifecycle, generation boundary
5. NEAT Implementation - topology mutation, speciation, crossover
6. Signal System - emit/propagate/receive, symbol vocabulary
7. Fitness and Selection - survival time, kin, communication bonuses
8. World Design - grid, terrain, food regrowth
9. Configuration and Reproducibility - TOML config, seeded RNG
10. Statistics and Analysis - signal entropy, mutual information, TopSim
11. API Boundary - WorldSnapshot/AgentSnapshot (read-only views)
12. Dependency Graph and Crate Layout
13. Error Handling Strategy
14. Beginner Pitfall Guide

## Dependency Flow

```
viz (terminal-viz feature)
 |
snapshot
 |
evolution  <->  stats
 |               |
world/grid ------+
 |    |     \
agent      signal
 |    \       |
brain   world/{entity,terrain,food}
 |         |
 +----+----+
      |
 config, rng
```

`world/grid` is the simulation container - it owns `Vec<Prey>`, `Vec<Predator>`, `Vec<ActiveSignal>`. It imports from agent/ and signal/. `world/{entity,terrain,food}` are shared primitive types at the bottom. This bidirectional relationship between world/grid and agent is by design (container pattern), not a violation.

`brain` never imports `world`, `agent`, or `signal`. `signal` never imports `agent` or `evolution`.

## Invariants

1. **Sorted connections** - `NeatGenome.connections` sorted by innovation number at all times. Call `sort_connections()` after any mutation.
2. **Single RNG** - `World.rng` is the only randomness source. Never construct a second RNG. Pass `&mut world.rng` to anything needing randomness.
3. **Feedforward only** - `NeatNetwork` is topologically sorted. Cycles in the genome graph are dropped during network construction.
4. **No unwrap/expect** - Denied by clippy. Use `?`, `map_or`, or `unwrap_or_else` with meaningful messages.
5. **Signal delay** - Signals emitted on tick T are receivable on tick T+1, never T. `propagation.rs` enforces this.
6. **Grid index** - `y * width + x`. Always use `World::idx()`, never compute manually.
7. **Sensor count** - 36 inputs (with vocab_size=8), 18 outputs. `sensor.rs` constants are the source of truth; genome initialization must match.

## Boundaries

**Always**: run `cargo lint` before committing. Keep modules importing only downward per the dependency flow. Use `pub(crate)` for internal types - external API is `WorldSnapshot`/`AgentSnapshot` only.

**Ask first**: changing sensor/output neuron count (breaks all existing genomes), modifying fitness function weights, adding new dependencies to Cargo.toml, changing grid coordinate system.

**Never**: `#[allow(...)]` (denied by clippy - use `#[expect(...)]` with a reason string). Import world/agent types from brain/. Access `World` fields directly from viz/ (use snapshots). Construct RNG outside `World.rng`.
