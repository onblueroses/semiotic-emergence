---
globs: ["src/world/**/*.rs"]
---

# World Module Rules

## Module structure
world/ has two layers with different dependency roles:
- `entity.rs`, `terrain.rs`, `food.rs` - shared primitive types (Position, Direction, IDs, Terrain, Food). These sit at the bottom of the dependency graph. Imported by agent/ and signal/.
- `grid.rs` - the simulation container (`World` struct). Owns `Vec<Prey>`, `Vec<Predator>`, `Vec<ActiveSignal>`. Imports FROM agent/ and signal/. This is the container pattern - grid.rs is above agent/ in the dependency graph even though it lives in world/.

## Grid index formula
Index into terrain/food arrays: `y * width + x`. Always use `World::idx(x, y)`, never compute the index inline. Getting this wrong causes silent data corruption.

## Bounds checking
`World::in_bounds(x, y)` takes `i32` parameters (agents compute relative positions that can go negative). All position arithmetic must use i32 before bounds-checking, then cast to u32 only after confirming in-bounds.

## Single RNG ownership
`World.rng` is the ONLY source of randomness in the entire simulation. Never create a second RNG (e.g., `thread_rng()`, `ChaCha8Rng::from_entropy()`). Pass `&mut world.rng` to any function needing randomness. This ensures deterministic replay from a seed.

## Terrain passability
- `Open` - all agents can enter
- `Tree` - prey can climb (sets `is_climbing`), ground predators cannot enter, aerial predators ignore
- `Rock` - prey can hide behind (sets `is_hidden`), no predator can enter
- `Water` - impassable to all agents

## Food regrowth
Food cells use `Option<Food>`. `None` = no food. `Some(food)` with `regrow_timer > 0` = consumed, regrowing. Do not check `energy == 0` for empty cells - use the Option.
