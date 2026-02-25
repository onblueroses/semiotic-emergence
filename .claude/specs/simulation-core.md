# Spec: Simulation Core
Created: 2026-02-25 | Status: in-progress
Last updated: 2026-02-25 22:30 | Session: f09a32a0

## Quick Start
> Fresh context or just resumed? Read this section, then go to `>> Current Step`.

**Files touched:**
- `src/world/terrain.rs` - Terrain generation algorithm (clusters, rivers, flood-fill validation)
- `src/world/food.rs` - Food placement function, remove dead_code
- `src/world/grid.rs` - World::initialize, spawn_prey, all 9 tick phases, predator AI, generation lifecycle, sensor encoding, SensorContext, reset_for_generation, snapshot
- `src/brain/genome.rs` - create_minimal_genome for initial population
- `src/brain/network.rs` - Remove file-level dead_code annotation
- `src/brain/innovation.rs` - Remove dead_code from InnovationCounter impl
- `src/signal/propagation.rs` - Full signal receive/decay/create implementation
- `src/agent/sensor.rs` - decode_outputs function, remove file-level dead_code
- `src/agent/prey.rs` - Remove dead_code from Prey struct
- `src/agent/predator.rs` - Remove dead_code from Predator struct
- `src/main.rs` - Wire generation loop, CLI config loading, simulation entry point
- `src/config.rs` - Add Simulation variant to SimError
- `src/snapshot.rs` - Used by World::snapshot (already implemented, just wired)

**-> Next:** Step 6.1 - Add deterministic seed integration test (details in `>> Current Step` below)

**Mode: EXECUTION** - Do NOT re-plan. Do NOT spawn Plan or Explore subagents. Do NOT enter plan mode. The spec is the authoritative plan. Any planning-agent output in this session that differs from this spec is stale - ignore it.

## Goal
Implement the simulation core: world initialization (terrain, food, agents), the 9-phase tick lifecycle, signal propagation, scripted predator AI, generation boundaries with fitness eval, and the main loop - so `cargo run` produces a working simulation that loops through generations.

## Non-Goals
- **Full NEAT evolution** (speciation, crossover, mutation, reproduction) - Phases 7-9. Generation boundary creates a stub hook; genomes re-randomized between generations for now.
- **Kin tracking and relatedness** - Phase 12. Fitness uses simplified formula without kin bonus.
- **Stats collection, metrics, export** - Phases 13-14. Trigger points only, no implementation.
- **Terminal visualization or dashboard** - Phase 15. Snapshot method wired but not consumed.
- **Checkpointing / save-load** - deferred until stats infrastructure exists.
- **In-sim reproduction via NEAT mutation** - Reproduce action spawns offspring with cloned genome + random weight perturbation. Full NEAT offspring come Phase 9.

## Context
Phase 6 of 16-phase build. All primitive types exist with correct fields/methods but carry `#[expect(dead_code)]` because nothing instantiates them. main.rs parses CLI args and prints version only. Architecture doc prescribes strict 9-phase tick order and 7-step generation lifecycle. Edition 2024, strict clippy (pedantic + deny unwrap/expect/panic/todo/print). No `#[allow(...)]` - only `#[expect(...)]` with reason.

Key numeric contracts: 80x60 grid, 150 prey, 9 predators (2+3+4), ~720 food, 2000 ticks/gen, 1000 max gens.

## Decisions
Settled. Do not re-debate after compaction.

| # | Decision | Rationale | Alternatives Rejected | Date |
|---|----------|-----------|----------------------|------|
| D1 | All simulation methods as `impl World` in grid.rs | World owns all mutable state; methods have natural access. Free fns for predator AI take individual field borrows. | Separate tick phase files (would need &mut World everywhere or unsafe borrow splitting) | 2026-02-25 |
| D2 | Signal propagation as free fns in signal/propagation.rs taking (&[ActiveSignal], Position, ...) | Respects signal module dependency boundary (no agent/evolution imports). World calls these. | Methods on World (violates module boundary); trait (over-engineered) | 2026-02-25 |
| D3 | Terrain generation as free fn in world/terrain.rs | Keeps terrain logic cohesive with Terrain enum. Returns Vec<Terrain> to World::initialize. | In grid.rs (bloats already-large file); separate module (one function doesn't justify a module) | 2026-02-25 |
| D4 | create_minimal_genome in brain/genome.rs | Brain module owns genome types. Creates sparse initial topology (20% random connections). No world/agent imports needed. | In evolution/population.rs (evolution stubs not ready yet); in grid.rs (brain boundary violation) | 2026-02-25 |
| D5 | Generation boundary collects Vec<(NeatGenome, f32)>, calls next_generation_placeholder that randomizes new genomes | Clean handoff point. When Phases 7-9 implement NEAT, they replace this one function. | Full NEAT now (scope explosion); no placeholder (can't test generation loop) | 2026-02-25 |
| D6 | Predator AI as three free fns: tick_aerial, tick_ground, tick_pack in grid.rs | Each takes specific field borrows. Readable behavior trees. Avoids single 200-line method. | Trait per predator type (over-engineered); methods on Predator (can't access world state) | 2026-02-25 |
| D7 | SimError gets Simulation(String) variant | Config and IO errors exist; terrain gen failure and runtime issues need a home. | panic! (denied by clippy); Result<(), Box<dyn Error>> (loses type info) | 2026-02-25 |
| D8 | Sensor encoding as free fn in grid.rs (not agent/sensor.rs) | Agent module cannot import world/grid. Encoding needs World data. Fn lives where data is. | SensorContext struct passed to agent (agent still can't import grid types); encode in sensor.rs with raw slices (loses type safety) | 2026-02-25 |
| D9 | decode_outputs as free fn in agent/sensor.rs | Sits next to input encoding constants. Pure function, no world deps. | In grid.rs (muddies boundary); in action.rs (action is just the enum) | 2026-02-25 |
| D10 | World::initialize(config, seed) does terrain+food+predators but NOT prey | Prey spawning needs genomes from generation lifecycle. Separating lets gen loop reuse same terrain. | Init everything at once (genomes don't exist at world creation time) | 2026-02-25 |

## Plan

### Phase 1: Terrain Generation and World Initialization (~30 min)
- [x] **Step 1.1**: Add `generate_terrain` to `world/terrain.rs` - cluster placement, rivers, flood-fill validation -> `cargo lint` passes, deterministic output with seed 42
- [x] **Step 1.2**: Add food placement `place_food` to `world/food.rs` -> ~density*cells items placed, no food on Water, deterministic _(Depends on: 1.1)_
- [x] **Step 1.3**: Expand `World` init in `grid.rs` - `World::initialize(config)` calls terrain gen, food placement, predator spawning. Remove dead_code annotations. -> correct dimensions, ~720 food, 9 predators, `cargo lint` passes _(Depends on: 1.1, 1.2)_
- [x] **Step 1.4**: Add `create_minimal_genome` to `brain/genome.rs` -> genome has correct node counts, ~20% connections, sorted, NeatNetwork::from_genome succeeds
- [x] **Step 1.5**: Add `spawn_prey` helper in `grid.rs` -> `world.prey.len() == genomes.len()`, each prey has valid brain, correct initial energy _(Depends on: 1.3, 1.4)_
- [x] **Step 1.6**: Wire initial population in `main.rs` -> `cargo run` prints init info, `cargo lint` passes, dead_code annotations removed from live types _(Depends on: 1.5)_

### Phase 2: Sensor Encoding and Brain Phase (~30 min)
- [x] **Step 2.1**: Implement sensor encoding fn in `grid.rs` -> produces SensorReading with inputs.len() == input_count(vocab_size)
- [x] **Step 2.2**: Implement `decode_outputs` in `agent/sensor.rs` -> correct action for known outputs, signal only when threshold exceeded _(Depends on: 2.1)_
- [x] **Step 2.3**: Implement signal `receive_signals` in `signal/propagation.rs` -> one-tick delay enforced, strongest-per-symbol, distance attenuation
- [x] **Step 2.4**: Wire sensor + brain + decode as `World::sense_and_decide` -> each prey produces an action, method compiles _(Depends on: 2.1, 2.2, 2.3)_

### Phase 3: Signal System and Action Resolution (~30 min)
- [x] **Step 3.1**: Implement signal emission, decay, and create in `signal/propagation.rs` -> signals appear in world, energy deducted, decay removes old signals _(Depends on: 2.3)_
- [x] **Step 3.2**: Implement action resolution `resolve_actions` on World -> prey positions update, food consumed, energy costs applied, offspring spawned, shuffled order _(Depends on: 2.4)_
- [x] **Step 3.3**: Implement food regrowth `regrow_food` on World -> eaten food starts countdown, respawns after timer with 80% probability
- [x] **Step 3.4**: Implement `emit_signals` phase on World -> signals created, energy deducted, prey state updated _(Depends on: 3.1)_

### Phase 4: Predator AI (~30 min)
- [x] **Step 4.1**: Implement aerial predator (eagle) behavior in grid.rs -> patrols, targets exposed prey, cannot kill tree-climbing prey, rests after kill
- [x] **Step 4.2**: Implement ground predator (snake) behavior in grid.rs -> ambushes from bush, lunges, cannot kill rock-hiding prey, patience timer
- [x] **Step 4.3**: Implement pack predator (wolves) behavior in grid.rs -> coordinate on target, fan out, reduced effectiveness vs scattered prey
- [x] **Step 4.4**: Wire predator phase `tick_predators` + death phase on World -> all predators execute, kill list returned, dead prey removed with fitness computed _(Depends on: 4.1, 4.2, 4.3)_

### Phase 5: Tick Loop and Generation Lifecycle (~30 min)
- [x] **Step 5.1**: Assemble 9-phase tick in `World::tick` -> one call advances simulation one step, all phases in order, tick counter increments _(Depends on: 3.2, 3.3, 3.4, 4.4)_
- [x] **Step 5.2**: Implement generation boundary `run_generation` + `GenerationResult` struct -> runs tick loop, returns genomes with fitness _(Depends on: 5.1)_
- [x] **Step 5.3**: Implement `next_generation_placeholder` + `World::reset_for_generation` -> keeps terrain, resets food/agents/signals, placeholder preserves elite genomes _(Depends on: 5.2)_
- [x] **Step 5.4**: Wire main simulation loop in `main.rs` -> `cargo run` runs full loop, stderr shows generation progress _(Depends on: 5.3)_
- [x] **Step 5.5**: Clean up all dead_code annotations, add `World::snapshot`, run `cargo lint` + `cargo ta` -> zero warnings, all tests pass _(Depends on: 5.4)_

### Phase 6: Integration Testing (~25 min)
- [ ] **Step 6.1**: Add deterministic seed integration test (fast_test config, 1 generation) -> some prey survived, some died, food consumed, deterministic replay confirmed
- [ ] **Step 6.2**: Add tick-level unit tests (sensor encoding, output decoding, signal delay, food cycle, predator kills) -> all pass
- [ ] **Step 6.3**: Verify full loop with `cargo run -- --config config/fast_test.toml` -> completes 100 generations, no panics, sensible output _(Depends on: 6.1, 6.2)_

## Validation

| Level | Check | How to verify |
|-------|-------|---------------|
| L1 - Syntax | Compiles, clippy clean | `cargo lint` (clippy --all-targets -- -D warnings) |
| L2 - Unit | Sensor encoding, output decoding, signal delay, food cycle, predator kills | `cargo ta` (nextest) |
| L3 - Integration | Full generation loop runs, deterministic replay | Integration test with fast_test.toml + `cargo run` smoke test |
| L4 - Determinism | Same seed = identical trace | Run 10 gens twice with seed 42, compare GenerationResult |

## >> Current Step
Working on: Step 6.1 - Add deterministic seed integration test
Status: Phases 1-5 complete. All code compiles, `cargo lint` clean (0 warnings), 14 tests pass, `cargo run -- --config config/fast_test.toml` runs 100 generations successfully. Next: add integration tests for determinism and tick-level behavior.

## Completed
- [x] **Step 1.1**: Terrain generation with clusters, rivers, flood-fill (verified: 4 tests pass, deterministic)
- [x] **Step 1.2**: Food placement with weighted terrain preference (verified: 3 tests pass)
- [x] **Step 1.3**: World::initialize with terrain+food+predators (verified: cargo lint passes)
- [x] **Step 1.4**: create_minimal genome with ~20% sparse connections (verified: NeatNetwork::from_genome succeeds)
- [x] **Step 1.5**: spawn_prey helper (verified: prey count matches genomes, valid brains)
- [x] **Step 1.6**: Simulation loop wired in src/simulation.rs (verified: cargo run prints init info)
- [x] **Step 2.1**: Sensor encoding via SensorContext struct (verified: 36 inputs produced)
- [x] **Step 2.2**: decode_outputs (verified: 4 unit tests)
- [x] **Step 2.3**: receive_signals with one-tick delay (verified: 3 unit tests)
- [x] **Step 2.4**: sense_and_decide wired (verified: compiles, runs)
- [x] **Step 3.1**: Signal emission, decay, create (verified: 3 unit tests)
- [x] **Step 3.2**: resolve_actions with shuffled order (verified: runs without panics)
- [x] **Step 3.3**: Food regrowth with timer + 80% probability (verified: food respawns)
- [x] **Step 3.4**: emit_signals phase (verified: signals appear, energy deducted)
- [x] **Step 4.1**: Aerial predator (eagle) behavior (verified: targets exposed prey, respects tree cover)
- [x] **Step 4.2**: Ground predator (snake) behavior (verified: stalking + lunge, rock hiding works)
- [x] **Step 4.3**: Pack predator (wolves) behavior (verified: coordinate on shared target)
- [x] **Step 4.4**: tick_predators + death_phase (verified: kills collected, dead prey removed with fitness)
- [x] **Step 5.1**: 9-phase tick assembled (verified: runs in correct order)
- [x] **Step 5.2**: run_generation + GenerationResult (verified: returns genomes with fitness)
- [x] **Step 5.3**: next_generation_placeholder + reset_for_generation (verified: elites preserved)
- [x] **Step 5.4**: Main simulation loop (verified: 100 gens complete, stderr progress output)
- [x] **Step 5.5**: Lint cleanup - SensorContext struct, collapsible_if, is_none_or/is_some_and, doc backticks, dead_code expects (verified: cargo lint clean, 14 tests pass)

## Learnings
- [2026-02-25] `gen` is a reserved keyword in Rust edition 2024 - use `generation` instead
- [2026-02-25] `f32::from(u32)` doesn't exist - must use `as f32` cast
- [2026-02-25] Borrow splitting in resolve_actions: can't hold `&mut self.prey[idx]` and call `self.in_bounds()`. Fix: inline bounds checks or use direct field access `self.prey[prey_idx].field`
- [2026-02-25] `ref mut` patterns not allowed in edition 2024 implicit borrow mode
- [2026-02-25] `#[expect(dead_code)]` is unfulfilled when item is used by test target but not lib target
- [2026-02-25] `pub(crate)` items in lib are NOT visible from binary crate main.rs - moved logic to lib-side simulation.rs module
- [2026-02-25] Edition 2024 clippy: `map_or(true, ...)` -> `is_none_or(...)`, `map_or(false, ...)` -> `is_some_and(...)`
- [2026-02-25] Too many function args: bundle into a context struct (SensorContext) to stay under clippy's 7-arg limit
