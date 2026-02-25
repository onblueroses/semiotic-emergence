---
globs: ["src/agent/**/*.rs"]
---

# Agent Module Rules

## Dependency boundary
agent/ imports from world/{entity,terrain} for shared types (Position, Direction, Terrain) and from brain/ for neural network types. agent/ does NOT import from world/grid.rs (would create a tight circular dependency). The `World` struct in grid.rs holds `Vec<Prey>` and `Vec<Predator>` - that's the container direction (grid owns agents), not the other way around.

## Sensor encoding contract
Prey brains receive exactly `input_count(vocab_size)` inputs (36 with vocab_size=8). See `sensor.rs` constants for the exact layout:
- 0-8: predator distances/directions (aerial, ground, pack)
- 9-10: terrain flags (on_tree, on_rock)
- 11-16: environment (nearest tree/rock/food/prey distances, prey density)
- 17-32: signal inputs (per-symbol strength + direction)
- 33-35: self state (energy, is_protected, ticks_since_signal)

Output count: `output_count(vocab_size)` (18 with vocab_size=8).
- 0-8: movement + actions (N/S/E/W, eat, reproduce, climb, hide, idle)
- 9: signal emit probability
- 10-17: signal symbol selection

If you change sensor or output count, ALL existing genomes become invalid.

## Action resolution order
Each tick, actions resolve in this order: move, eat, climb/hide, signal, reproduce. An agent can only perform ONE primary action per tick (highest-activation output wins). Signal emission is secondary (can co-occur with any primary action if OUT_SIGNAL_EMIT > threshold).

## Prey tick shuffle
Process prey in random order each tick (`world.rng`). Never iterate `world.prey` in index order for action resolution - it creates positional bias.

## Signaling is non-exclusive
A prey can emit a signal AND perform another action in the same tick. Signal emission uses a separate output neuron (OUT_SIGNAL_EMIT) as a gate, independent of the primary action selection.
