Now let me produce the full architecture document. This is a substantial design effort - I'll write it directly as a comprehensive response.

---

# predator-prey-evolution-communication: Architecture Document

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Module Structure](#2-module-structure)
3. [Core Data Structures](#3-core-data-structures)
4. [Simulation Loop](#4-simulation-loop)
5. [NEAT Implementation](#5-neat-implementation)
6. [Signal System](#6-signal-system)
7. [Fitness and Selection](#7-fitness-and-selection)
8. [World Design](#8-world-design)
9. [Configuration and Reproducibility](#9-configuration-and-reproducibility)
10. [Statistics and Analysis](#10-statistics-and-analysis)
11. [API Boundary for Future Frontends](#11-api-boundary)
12. [Dependency Graph and Crate Layout](#12-dependency-graph)
13. [Error Handling Strategy](#13-error-handling)
14. [Beginner Pitfall Guide](#14-beginner-pitfalls)

---

## 1. Design Philosophy

**Three core tensions drive every decision:**

1. **Biological plausibility vs. computational tractability.** Real evolution is massively parallel and runs for millions of generations. We need evolutionary pressure strong enough to produce interesting communication in ~500-2000 generations on a single machine. This means the world must be small enough to iterate fast but rich enough to reward communication.

2. **Modularity vs. simplicity.** A Rust beginner will fight the borrow checker on every abstraction boundary. Every trait, every `Arc<Mutex<>>`, every lifetime annotation is a potential wall. The architecture uses plain structs with owned data, passes things by value or `&mut`, and avoids trait objects except at the one boundary that actually needs polymorphism (the viz/frontend interface).

3. **Purity of the simulation core vs. observability.** The engine must compile to WASM with zero rendering dependencies. But we also need rich data collection. Solution: the stats system is part of the core (it's just data), while all rendering lives behind a feature flag in a separate module that depends on the core, never the reverse.

**Guiding principle: data flows down, events flow up.** The simulation produces `WorldSnapshot` structs. Consumers (terminal viz, Bevy renderer, analysis tools) read snapshots. They never reach into the simulation's mutable state.

---

## 2. Module Structure

```
predator-prey-evolution-communication/
  Cargo.toml
  config/
    default.toml          # Ship with sane defaults
    fast_test.toml        # Small world, few agents, quick iteration
  src/
    lib.rs                # Re-exports core modules. This IS the library crate.
    main.rs               # CLI entry point (runs sim, optional viz)

    config.rs             # SimConfig, TOML loading, CLI override
    rng.rs                # Seeded RNG wrapper (ChaCha8Rng)

    world/
      mod.rs
      grid.rs             # Grid struct, coordinate math, terrain
      terrain.rs          # Terrain enum, terrain generation
      entity.rs           # EntityId, EntityKind, position tracking
      food.rs             # Food spawning, energy values, regrowth

    agent/
      mod.rs
      prey.rs             # Prey struct, sensor/actuator definitions
      predator.rs         # Predator struct, AI behaviors
      action.rs           # Action enum (Move, Eat, Signal, Reproduce, Idle)
      sensor.rs           # SensorReading struct, vision cone logic

    brain/
      mod.rs
      genome.rs           # NeatGenome, ConnectionGene, NodeGene
      network.rs          # NeatNetwork (phenotype), activation
      activation.rs       # Activation functions (sigmoid, tanh, relu)
      innovation.rs       # InnovationCounter, InnovationRecord

    signal/
      mod.rs
      message.rs          # Signal struct, symbol vocabulary
      propagation.rs      # Broadcast/receive logic, range, decay

    evolution/
      mod.rs
      population.rs       # Population struct, generation management
      species.rs          # Species struct, compatibility, speciation
      fitness.rs          # Fitness calculation, components
      selection.rs        # Tournament selection, crossover, mutation
      reproduction.rs     # Offspring generation, elitism
      kin.rs              # Lineage tracking, relatedness coefficient

    stats/
      mod.rs
      collector.rs        # Per-tick and per-generation data collection
      metrics.rs          # Derived metrics (signal entropy, mutual info)
      export.rs           # CSV/JSON serialization

    snapshot.rs           # WorldSnapshot, AgentSnapshot - read-only view

    viz/                  # Behind `feature = "terminal-viz"`
      mod.rs
      terminal.rs         # ASCII renderer using crossterm
      dashboard.rs        # Stats panel, generation info
```

### Why this structure

**No ECS.** An ECS (like `hecs` or `specs`) would be the "Rusty" choice for a game, but it's the wrong call here. Reasons: (a) a Rust beginner will drown in the ECS abstraction layer on top of already learning ownership, (b) the entity count is small (50-200 prey, 5-20 predators) so cache-line optimization is irrelevant, (c) the interesting logic is in NEAT and signals, not in entity management. Plain structs in `Vec`s are the right tool.

**Single crate, not a workspace.** A workspace with `core`, `neat`, `viz` crates adds Cargo complexity for no gain at this scale. Feature flags on a single crate achieve the same separation. If the project grows past ~10k lines, split then.

**`brain/` is its own module tree, not a dependency.** NEAT could be a separate crate, but keeping it in-tree means (a) no version management overhead, (b) the genome can directly reference simulation-specific constants (number of sensor inputs, number of action outputs) without an adapter layer, (c) a beginner doesn't have to understand Cargo workspaces.

**`rng.rs` is a top-level module, not buried in `config`.** Every system needs randomness. A single seeded `ChaCha8Rng` (from the `rand_chacha` crate) wrapped in a newtype ensures determinism. It's important enough to be visible at the top level.

---

## 3. Core Data Structures

### 3.1 World

```rust
// world/grid.rs

pub struct World {
    pub width: u32,
    pub height: u32,
    pub terrain: Vec<Terrain>,         // Flat array, indexed by y * width + x
    pub food: Vec<Option<Food>>,       // Same flat layout, None = no food at cell
    pub prey: Vec<Prey>,               // All living prey
    pub predators: Vec<Predator>,      // All living predators
    pub signals: Vec<ActiveSignal>,    // Currently propagating signals
    pub tick: u64,                     // Current simulation tick
    pub generation: u32,               // Current evolutionary generation
    pub rng: SeededRng,                // Deterministic RNG
}
```

**Why a flat `Vec<Terrain>` instead of `Vec<Vec<Terrain>>`?** Single contiguous allocation, cache-friendly iteration, and `index = y * width + x` is trivial. A 2D wrapper is unnecessary complexity.

**Why `Vec<Prey>` instead of a `HashMap<EntityId, Prey>`?** The prey list is iterated every tick. `Vec` iteration is fast. When a prey dies, we swap-remove it (O(1)). We never need random access by ID during the hot loop - only during inter-agent lookups, which use spatial queries anyway.

```rust
// world/terrain.rs

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Terrain {
    Open,           // No cover, fast movement
    Grass,          // Slight concealment, normal speed
    Tree,           // Hides from aerial predators, climbable
    Rock,           // Blocks movement, provides high-ground escape
    Water,          // Impassable for ground predators, slows prey
    Bush,           // Concealment from ground predators
}
```

**Why no "elevation" float?** Discrete terrain types are far easier to reason about in neural network inputs (one-hot encoding) and for a beginner to debug. Continuous terrain would require interpolation, gradient sensing, and more complex predator pathfinding. The discrete set above provides enough richness for the three predator counter-strategies: Trees counter aerial (hide under canopy), Rocks counter ground/ambush (climb up), Water + Open counters pack (scatter across obstacles).

```rust
// world/food.rs

pub struct Food {
    pub energy: f32,        // Energy gained when eaten
    pub regrow_timer: u32,  // Ticks until this cell can regrow food (0 = ready)
}
```

### 3.2 Agents

```rust
// agent/prey.rs

pub struct Prey {
    pub id: PreyId,                    // Unique within a generation
    pub pos: Position,                 // (x, y) on the grid
    pub energy: f32,                   // Dies at 0, reproduces above threshold
    pub age: u32,                      // Ticks alive (contributes to fitness)
    pub genome: NeatGenome,            // The heritable blueprint
    pub brain: NeatNetwork,            // Phenotype built from genome (cached)
    pub facing: Direction,             // N/S/E/W - affects sensor cone
    pub last_action: Action,           // For stat tracking
    pub last_signal: Option<Symbol>,   // What symbol was broadcast last tick
    pub lineage: LineageId,            // For kin selection tracking
    pub parent_genome_id: Option<GenomeId>,  // Direct parent (for relatedness)
    pub generation_born: u32,          // When this individual was created
    pub offspring_count: u32,          // Successful reproductions
    pub fitness_cache: Option<f32>,    // Computed at death or generation end
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreyId(pub u32);

#[derive(Clone, Copy)]
pub struct Position {
    pub x: u32,
    pub y: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
}
```

**Why `brain` is stored alongside `genome`:** Building the network from the genome is not free (topological sort, layer assignment). We do it once at birth and cache it. The genome is the heritable data; the brain is the runtime artifact.

**Why `lineage: LineageId` and not a full family tree:** Tracking every ancestor creates unbounded memory growth. A `LineageId` is a coarse grouping - all descendants of a common ancestor within N generations share a lineage. This is enough to compute approximate relatedness for kin selection (see section 7) without storing a tree.

```rust
// agent/predator.rs

pub struct Predator {
    pub id: PredatorId,
    pub kind: PredatorKind,
    pub pos: Position,
    pub energy: f32,           // Predators have energy too - they need to eat
    pub state: PredatorState,  // Hunting, stalking, resting, etc.
    pub target: Option<PreyId>,
    pub cooldown: u32,         // Ticks until next attack attempt
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PredatorKind {
    Aerial,  // Eagle: fast, long vision, defeated by tree cover
    Ground,  // Snake: slow, ambush, defeated by climbing/rock
    Pack,    // Wolf: medium speed, coordinates with others, defeated by scattering
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PredatorState {
    Roaming,          // No target, patrolling
    Stalking(PreyId), // Approaching target stealthily
    Attacking,        // In attack range, executing strike
    Resting,          // Post-kill cooldown, digesting
}
```

**Why predators are scripted, not evolved:** This is a deliberate asymmetry. The research question is "can prey evolve communication?" not "can predators and prey co-evolve?" Co-evolution creates an arms race that makes it very hard to analyze what's happening - you can never tell if a communication strategy failed because it was bad or because the predators happened to evolve a counter. Scripted predators provide consistent, reproducible selection pressure. Each predator type has a fixed behavior tree (see section 4) that creates clear, distinct threats requiring different responses.

**Why predators have energy:** Without energy, predators are immortal and never stop hunting. Energy forces predators to rest after kills, creating temporal patterns (safe periods after a kill) that prey can learn to exploit. It also prevents the simulation from collapsing to "predators always win."

### 3.3 Actions

```rust
// agent/action.rs

#[derive(Clone, Copy, Debug)]
pub enum Action {
    Move(Direction),           // Step one cell in a direction
    Eat,                       // Consume food at current cell
    Signal(Symbol),            // Broadcast a symbol
    Reproduce,                 // Attempt asexual reproduction (if enough energy)
    Climb,                     // Climb tree/rock at current cell (anti-ground-predator)
    Hide,                      // Take cover at current cell (anti-aerial)
    Idle,                      // Do nothing (saves energy)
}
```

**Why `Climb` and `Hide` are separate from `Move`:** These are distinct survival behaviors that the neural network needs to be able to select independently. A prey that merely moves onto a tree tile isn't protected - it needs to explicitly choose `Hide` (duck under canopy) or `Climb` (get elevation). This forces the network to learn the distinction between "move to safety" and "use the safety."

**Why asexual reproduction:** Sexual reproduction requires mate finding, courtship, and doubles the crossover complexity. NEAT already handles crossover at the population level between generations. In-simulation reproduction is asexual: a prey spends energy to spawn a copy (with slight mutation) nearby. This creates kin clusters naturally - relatives are spatially proximate because offspring spawn near parents - which is exactly what we need for kin selection to work without explicit kin recognition.

### 3.4 NEAT Genome (detailed in section 5, key structures here)

```rust
// brain/genome.rs

#[derive(Clone)]
pub struct NeatGenome {
    pub id: GenomeId,
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
    pub fitness: f32,
    pub species_id: Option<SpeciesId>,
}

#[derive(Clone, Copy)]
pub struct NodeGene {
    pub id: NodeId,
    pub kind: NodeKind,
    pub activation: ActivationFn,
    pub bias: f32,               // Evolved bias term
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Input,
    Hidden,
    Output,
}

#[derive(Clone, Copy)]
pub struct ConnectionGene {
    pub from: NodeId,
    pub to: NodeId,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: InnovationNumber,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InnovationNumber(pub u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenomeId(pub u64);
```

### 3.5 Signals

```rust
// signal/message.rs

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Symbol(pub u8);  // 0..VOCAB_SIZE-1

pub struct ActiveSignal {
    pub origin: Position,        // Where it was broadcast
    pub sender_id: PreyId,       // Who sent it
    pub symbol: Symbol,          // What symbol
    pub tick_emitted: u64,       // When it was sent
    pub strength: f32,           // Decays with time/distance
}
```

**Why `Symbol(u8)` and not an enum:** The vocabulary size is configurable (8-16 symbols). An enum would require code changes to resize. A `u8` newtype is flexible, and the output neurons simply map to symbol indices via argmax.

### 3.6 Configuration

```rust
// config.rs

#[derive(Deserialize, Clone)]
pub struct SimConfig {
    pub world: WorldConfig,
    pub prey: PreyConfig,
    pub predators: PredatorConfig,
    pub neat: NeatConfig,
    pub signal: SignalConfig,
    pub evolution: EvolutionConfig,
    pub stats: StatsConfig,
    pub seed: u64,
}

#[derive(Deserialize, Clone)]
pub struct WorldConfig {
    pub width: u32,                    // Default: 80
    pub height: u32,                   // Default: 60
    pub food_density: f32,             // Fraction of cells with food (0.0-1.0). Default: 0.15
    pub food_energy: f32,              // Energy per food item. Default: 20.0
    pub food_regrow_ticks: u32,        // Ticks until eaten food respawns. Default: 50
    pub terrain_tree_pct: f32,         // Fraction of cells that are trees. Default: 0.10
    pub terrain_rock_pct: f32,         // Default: 0.05
    pub terrain_water_pct: f32,        // Default: 0.05
    pub terrain_bush_pct: f32,         // Default: 0.08
}

#[derive(Deserialize, Clone)]
pub struct PreyConfig {
    pub initial_count: u32,            // Default: 100
    pub initial_energy: f32,           // Default: 100.0
    pub max_energy: f32,               // Default: 200.0
    pub energy_per_tick: f32,          // Basal metabolism cost. Default: 0.5
    pub move_energy_cost: f32,         // Default: 1.0
    pub signal_energy_cost: f32,       // Default: 3.0 (intentionally expensive!)
    pub reproduce_energy_cost: f32,    // Default: 60.0
    pub reproduce_energy_threshold: f32, // Default: 120.0
    pub vision_range: u32,             // Cells visible ahead. Default: 8
    pub vision_angle: f32,             // Cone half-angle in degrees. Default: 60.0
    pub hearing_range: u32,            // Max cells for signal reception. Default: 12
}

#[derive(Deserialize, Clone)]
pub struct PredatorConfig {
    pub aerial_count: u32,             // Default: 2
    pub ground_count: u32,             // Default: 3
    pub pack_count: u32,               // Default: 4 (1 pack of 4)
    pub aerial_speed: u32,             // Cells per tick. Default: 3
    pub ground_speed: u32,             // Default: 1
    pub pack_speed: u32,               // Default: 2
    pub aerial_vision: u32,            // Default: 15
    pub ground_vision: u32,            // Default: 4
    pub pack_vision: u32,              // Default: 10
    pub attack_cooldown: u32,          // Ticks after kill before hunting again. Default: 30
    pub kill_radius: u32,              // Must be this close to kill. Default: 1
}

#[derive(Deserialize, Clone)]
pub struct NeatConfig {
    pub population_size: u32,          // Default: 150
    pub c1_excess: f32,                // Compatibility: excess gene weight. Default: 1.0
    pub c2_disjoint: f32,              // Compatibility: disjoint gene weight. Default: 1.0
    pub c3_weight: f32,                // Compatibility: weight diff. Default: 0.4
    pub compatibility_threshold: f32,  // Default: 3.0
    pub weight_mutate_rate: f32,       // Default: 0.8
    pub weight_perturb_rate: f32,      // Of mutated weights, fraction perturbed vs replaced. Default: 0.9
    pub weight_perturb_strength: f32,  // Default: 0.5
    pub add_node_rate: f32,            // Default: 0.03
    pub add_connection_rate: f32,      // Default: 0.05
    pub disable_gene_rate: f32,        // Default: 0.01
    pub interspecies_mate_rate: f32,   // Default: 0.001
    pub stagnation_limit: u32,         // Generations before species is killed. Default: 15
    pub elitism_count: u32,            // Top N genomes survive unchanged. Default: 2
    pub survival_rate: f32,            // Fraction of species that reproduce. Default: 0.2
}

#[derive(Deserialize, Clone)]
pub struct SignalConfig {
    pub vocab_size: u8,                // Number of distinct symbols. Default: 8
    pub signal_range: u32,             // Max broadcast radius. Default: 12
    pub signal_decay_rate: f32,        // Strength loss per tick. Default: 0.3
    pub signal_lifetime: u32,          // Max ticks a signal persists. Default: 3
}

#[derive(Deserialize, Clone)]
pub struct EvolutionConfig {
    pub generation_ticks: u64,         // Ticks per generation. Default: 2000
    pub max_generations: u32,          // Stop after this many. Default: 1000
    pub min_prey_alive: u32,           // End generation early if prey drops below. Default: 5
    pub fitness_survival_weight: f32,  // Default: 1.0
    pub fitness_energy_weight: f32,    // Default: 0.3
    pub fitness_offspring_weight: f32, // Default: 2.0
    pub fitness_kin_bonus: f32,        // Bonus for kin survival. Default: 0.5
    pub kin_relatedness_generations: u32, // How far back to track relatedness. Default: 3
}

#[derive(Deserialize, Clone)]
pub struct StatsConfig {
    pub export_every_n_generations: u32, // Default: 10
    pub export_format: ExportFormat,     // Default: Csv
    pub export_path: String,             // Default: "output/"
    pub track_signals: bool,             // Default: true
    pub track_lineage: bool,             // Default: true
}
```

**Why signal cost of 3.0 (6x the movement cost):** This is the single most important parameter for honest signaling to evolve. If signaling is free, there's no cost to lying, and signals become noise - the "cheap talk" problem from evolutionary game theory. At 3.0, a prey that signals frivolously dies faster than one that stays quiet. But a prey whose signal saves 3 nearby kin (who share its genes) gets indirect fitness benefit exceeding the cost. This is literally Hamilton's rule: rB > C, where r is relatedness, B is benefit to kin, C is cost to signaler. The 3.0 cost sets the break-even at approximately r=0.3 with 3 nearby kin - achievable for siblings/cousins in our spatial reproduction model.

---

## 4. Simulation Loop

### Tick Order (every tick within a generation)

```
for each tick in 0..generation_ticks:

  1. SENSOR PHASE
     For each prey:
       - Scan vision cone: terrain, food, predators (with type), other prey
       - Scan hearing range: collect active signals (symbol + direction + strength)
       - Read internal state: own energy, own position, current terrain
       -> Produce SensorReading struct

  2. BRAIN PHASE
     For each prey:
       - Encode SensorReading into f32 input vector
       - Feed through NeatNetwork
       - Decode output vector into (Action, Option<Symbol>)
       -> Each prey now has a "desired action"

  3. SIGNAL EMISSION PHASE
     For each prey that chose Signal(symbol):
       - Deduct signal_energy_cost
       - Create ActiveSignal at prey's position
       - Add to world.signals

  4. ACTION PHASE (Movement, Eating, Reproducing)
     For each prey (shuffled order to prevent position bias):
       - Execute Move: update position (check bounds, terrain passability)
       - Execute Eat: if food at cell, consume it
       - Execute Reproduce: if energy > threshold, spawn offspring nearby
       - Execute Climb/Hide: set a "protected" flag for this tick
       - Execute Idle: no-op
       - Deduct energy costs (basal + action-specific)

  5. PREDATOR PHASE
     For each predator:
       - Run behavior tree for its type (see below)
       - Move toward target / patrol
       - If adjacent to unprotected prey: kill (prey removed, predator rests)
       - Deduct predator energy, handle predator death/respawn

  6. SIGNAL DECAY PHASE
     For each active signal:
       - Reduce strength by decay_rate
       - Remove if strength <= 0 or age > signal_lifetime

  7. FOOD REGROW PHASE
     For each empty food cell:
       - Decrement regrow_timer
       - If timer == 0: spawn new food

  8. DEATH PHASE
     Remove prey with energy <= 0
     Record their fitness

  9. STATS PHASE
     Record per-tick metrics (alive count, signals emitted, kills by type)
```

### Why this order matters

**Sensors before brains before actions:** This seems obvious but the alternative - interleaving sense-think-act per agent - creates order-of-update dependency where the first agent to act changes the world for the second. By separating into phases, all agents sense the same world state, then all act simultaneously (with conflict resolution by shuffle). This is the standard approach in artificial life simulations and avoids first-mover advantage.

**Signals emit before next tick's sensors:** A signal emitted in tick T is sensed by other agents in tick T+1. This one-tick delay is realistic (sound travels, reaction takes time) and prevents infinite loops where agent A signals, agent B reacts, agent A reacts to B's reaction, all within one tick.

**Predators after prey actions:** Prey get to run and hide before predators move. This means a prey that correctly identifies a threat and flees has a chance. If predators moved first, the simulation would be "did the prey start the tick in the right place?" which is less interesting.

**Shuffle in action phase:** Without shuffling, prey at index 0 always gets first pick of food. Shuffling each tick removes systematic bias. Use `rng.shuffle(&mut indices)` and iterate in that order.

### Predator Behavior Trees

Predators are NOT evolved. They run fixed behavior trees that create consistent, learnable threats.

**Aerial (Eagle):**
```
if state == Resting && cooldown > 0:
    decrement cooldown, stay put
else if state == Resting && cooldown == 0:
    transition to Roaming
else if state == Roaming:
    scan with aerial_vision radius
    if prey visible && prey not under Tree:
        target nearest exposed prey
        transition to Attacking
    else:
        move in patrol pattern (large circles)
else if state == Attacking:
    move aerial_speed cells toward target
    if adjacent to target:
        if target is Hidden under Tree: lose target, transition to Roaming
        else: kill, transition to Resting with cooldown
    if target out of vision: transition to Roaming
```

**Ground (Snake):**
```
if state == Resting: handle cooldown
else if state == Roaming:
    move slowly, prefer bush/grass terrain
    if prey within ground_vision AND prey not on Rock:
        transition to Stalking
    else: wander toward nearest bush cluster
else if state == Stalking:
    don't move (ambush predator!)
    if target moves adjacent: strike (transition to Attacking)
    if target moves beyond vision: give up, Roaming
    patience timer: after 20 ticks of stalking without success, Roaming
else if state == Attacking:
    lunge (move ground_speed toward target)
    if adjacent && target not Climbing: kill
    else: transition to Roaming (ambush failed)
```

**Pack (Wolves):**
```
// Wolves coordinate: all wolves in the pack share a target
if state == Resting: handle cooldown
else if state == Roaming:
    move as loose group (each wolf stays within 5 cells of pack center)
    if any wolf spots prey within pack_vision:
        all wolves set same target, transition to Attacking
else if state == Attacking:
    each wolf moves pack_speed toward target
    wolves try to surround: move to minimize max distance from target
    if any wolf adjacent to target: kill
    counter: if prey scatter (>3 cells apart from each other), wolves
        pick the most isolated prey (punishes clumping, rewards scattering)
    if target out of all wolves' vision: give up
```

**Why wolves punish clumping and reward scattering:** This creates the core communication pressure. Against eagles, prey should cluster under trees. Against wolves, prey should scatter. These opposing optimal behaviors mean prey must identify which predator is coming and coordinate the correct response. This is the selection pressure that produces predator-specific alarm calls.

### Generation Lifecycle

```
for generation in 0..max_generations:
    1. Build world (terrain is fixed across generations, food resets)
    2. Spawn prey from current population genomes
    3. Spawn predators (fixed count, positions randomized)
    4. Run tick loop for generation_ticks (or until min_prey_alive)
    5. Collect final fitness for all prey (alive and dead)
    6. Run NEAT evolution:
       a. Speciate population
       b. Compute adjusted fitness (fitness sharing)
       c. Remove stagnant species
       d. Allocate offspring per species
       e. Select parents, crossover, mutate
       f. Produce next generation's genomes
    7. Export stats for this generation
    8. Optional: checkpoint save
```

**Why terrain persists across generations:** Prey need to learn terrain layout. If terrain randomizes every generation, there's no spatial knowledge to evolve. Fixed terrain means specific locations become meaningful ("the tree cluster at position 20,30 is a good hiding spot"). This mirrors real ecosystems where animals learn their home range.

**Why food resets:** Food scarcity creates competition, which is good. But leftover food distributions from last generation would create an unfair advantage for prey that happen to spawn near depleted areas. Fresh food every generation ensures equal opportunity.

---

## 5. NEAT Implementation

### 5.1 Genome Representation

The genome has two gene types:

**Node Genes** define neurons:
- `id: NodeId` - unique within the genome
- `kind: NodeKind` - Input, Hidden, or Output (Input/Output are fixed at creation)
- `activation: ActivationFn` - Sigmoid for output neurons, tanh for hidden (configurable)
- `bias: f32` - evolved bias term, initialized to 0.0

**Connection Genes** define synapses:
- `from: NodeId` - source neuron
- `to: NodeId` - destination neuron
- `weight: f32` - connection strength
- `enabled: bool` - can be disabled by mutation (structural gene silencing)
- `innovation: InnovationNumber` - global historical marker

**Initial genome (minimal topology):**
Every genome starts with:
- All input nodes (sensor encoding, ~30-40 nodes depending on config)
- All output nodes (action selection, ~12 nodes)
- Zero hidden nodes
- A small random set of direct input-to-output connections (not fully connected!)

**Why not fully connected initially:** The original NEAT paper starts with full connectivity. This is wrong for our case. With 35 inputs and 12 outputs, that's 420 connections - too many parameters for the initial population to explore meaningfully. Starting sparse (say, 20% random connectivity) means the initial genomes are diverse (different subsets of connections) and NEAT's "add connection" mutation can discover useful connections incrementally. This also makes speciation more effective early on, since sparse genomes have more structural variation.

### Input/Output Neuron Layout

**Inputs (order matters - this IS the interface contract):**

```
// Sensor encoding - all values normalized to [0, 1] or [-1, 1]

// --- Vision (per visible cell in cone, nearest 5 cells only) ---
// For each of 5 nearest visible cells:
//   terrain_one_hot[6]  (Open, Grass, Tree, Rock, Water, Bush)  = 6 * 5 = 30? NO.

// SIMPLIFIED: aggregate vision to keep input count manageable
// --- Predator detection (3 inputs per predator type) ---
[0]  nearest_aerial_distance     // 0.0 = adjacent, 1.0 = at vision limit, -1.0 = none visible
[1]  nearest_aerial_direction_x  // -1.0 to 1.0 relative direction
[2]  nearest_aerial_direction_y
[3]  nearest_ground_distance
[4]  nearest_ground_direction_x
[5]  nearest_ground_direction_y
[6]  nearest_pack_distance
[7]  nearest_pack_direction_x
[8]  nearest_pack_direction_y

// --- Terrain awareness (4 inputs) ---
[9]  current_terrain_is_tree     // 1.0 if on tree, else 0.0
[10] current_terrain_is_rock
[11] nearest_tree_distance       // 0.0-1.0 normalized
[12] nearest_rock_distance

// --- Food detection (2 inputs) ---
[13] nearest_food_distance       // normalized
[14] nearest_food_direction_x    // combined into angle

// --- Nearby prey (2 inputs) ---
[15] prey_density_nearby         // count of prey within 5 cells, normalized
[16] nearest_prey_distance

// --- Signal reception (vocab_size * 2 inputs) ---
// For each symbol s in 0..vocab_size:
[17 + s*2]     signal_s_strength     // 0.0 = no signal, 1.0 = strong nearby
[17 + s*2 + 1] signal_s_direction_x  // Where did it come from
// With vocab_size=8: inputs 17-32 (16 signal inputs)

// --- Internal state (3 inputs) ---
[33] own_energy_normalized       // current / max
[34] is_climbing                 // 1.0 if currently climbing/hidden
[35] ticks_since_last_signal     // normalized, encourages not spamming

// TOTAL: 36 inputs (with vocab_size=8)
```

**Why aggregated vision instead of per-cell:** Per-cell vision with a range of 8 and a 120-degree cone could mean 30+ cells, each needing 6+ values. That's 180+ inputs, far too many for NEAT to evolve meaningfully in reasonable time. Aggregated "nearest threat of each type + direction" compresses the information to what actually matters for survival decisions.

**Outputs:**

```
[0]  move_north     // Highest activated among 0-3 determines move direction
[1]  move_south
[2]  move_east
[3]  move_west
[4]  eat            // If > threshold (0.5), eat
[5]  reproduce      // If > threshold, attempt reproduction
[6]  climb          // If > threshold, climb (if on tree/rock)
[7]  hide           // If > threshold, hide (if on tree/bush)
[8]  idle           // If > threshold, do nothing
[9]  signal_emit    // If > threshold, emit a signal this tick
[10] signal_symbol_0  // Highest among 10..10+vocab_size picks the symbol
...
[10 + vocab_size - 1]

// TOTAL: 10 + vocab_size outputs = 18 (with vocab_size=8)
```

**Action resolution:** The action with the highest activation wins (argmax over move/eat/reproduce/climb/hide/idle). If `signal_emit` exceeds 0.5, a signal is ALSO emitted (signaling is not mutually exclusive with other actions, because in nature you can scream while running).

### 5.2 Network Construction from Genome

```rust
// brain/network.rs

pub struct NeatNetwork {
    nodes: Vec<NetworkNode>,
    // Nodes sorted in activation order (topological sort).
    // Input nodes first, then hidden in dependency order, then output nodes.
    edges: Vec<NetworkEdge>,
    input_count: usize,
    output_count: usize,
}

struct NetworkNode {
    id: NodeId,
    activation: ActivationFn,
    bias: f32,
    value: f32,        // Current activation value
    incoming: Range<usize>,  // Slice into edges array for this node's inputs
}

struct NetworkEdge {
    from_index: usize,  // Index into nodes array (NOT NodeId)
    weight: f32,
}
```

**Construction algorithm:**

1. Collect all enabled connections from the genome.
2. Build an adjacency list (from -> to).
3. Topological sort using Kahn's algorithm (BFS with in-degree tracking).
4. **Handle cycles:** If cycles exist (recurrent connections), break them by removing the back-edge with the lowest innovation number (oldest recurrent link). This means the network is always feedforward within a single tick. Recurrence would require multi-step activation per tick, which is complex and slow. The one-tick delay from the simulation loop already provides temporal context.
5. Assign each node an index in the sorted order.
6. Remap all edges to use indices instead of NodeIds.
7. Group edges by destination node (so `incoming` is a contiguous range).

**Why feedforward only (no recurrence):**  Recurrent NEAT networks (CTRNN) require iterating the network multiple times per tick until values stabilize. This is 5-10x slower and adds a convergence parameter that's hard to tune. For our simulation, temporal state comes from the world itself - the prey can observe "I was climbing last tick" via the `is_climbing` input, and "I heard signal X recently" via signal persistence. External state memory substitutes for internal recurrence, and it's much easier to debug.

**Activation:**

```rust
impl NeatNetwork {
    pub fn activate(&mut self, inputs: &[f32]) -> Vec<f32> {
        // Set input node values
        for (i, &val) in inputs.iter().enumerate() {
            self.nodes[i].value = val;
        }

        // Forward pass through hidden + output nodes
        for i in self.input_count..self.nodes.len() {
            let node = &self.nodes[i];
            let mut sum = node.bias;
            for edge_idx in node.incoming.clone() {
                let edge = &self.edges[edge_idx];
                sum += self.nodes[edge.from_index].value * edge.weight;
            }
            self.nodes[i].value = apply_activation(node.activation, sum);
        }

        // Read output values
        let output_start = self.nodes.len() - self.output_count;
        self.nodes[output_start..]
            .iter()
            .map(|n| n.value)
            .collect()
    }
}
```

### 5.3 Mutation Operators

```rust
// evolution/selection.rs (mutation logic)

pub fn mutate(genome: &mut NeatGenome, config: &NeatConfig, rng: &mut SeededRng,
              innovations: &mut InnovationCounter) {

    // Weight mutation (most common, applied to many connections)
    if rng.gen_bool(config.weight_mutate_rate) {
        for conn in &mut genome.connections {
            if rng.gen_bool(config.weight_perturb_rate) {
                // Perturb: add small noise
                conn.weight += rng.gen_range(-1.0..1.0) * config.weight_perturb_strength;
                conn.weight = conn.weight.clamp(-4.0, 4.0); // Prevent explosion
            } else {
                // Replace: new random weight
                conn.weight = rng.gen_range(-2.0..2.0);
            }
        }
    }

    // Add connection mutation
    if rng.gen_bool(config.add_connection_rate) {
        add_connection_mutation(genome, rng, innovations);
    }

    // Add node mutation
    if rng.gen_bool(config.add_node_rate) {
        add_node_mutation(genome, rng, innovations);
    }

    // Disable gene mutation
    if rng.gen_bool(config.disable_gene_rate) {
        if let Some(conn) = genome.connections.choose_mut(rng) {
            conn.enabled = !conn.enabled; // Toggle
        }
    }
}
```

**Add Connection Mutation:**
1. Pick two random nodes (from, to) where from != to.
2. Ensure the connection doesn't already exist.
3. Ensure it doesn't create a cycle from input back to input, or from output forward to output.
4. Check the innovation counter: has (from, to) been seen this generation?
   - Yes: reuse the same innovation number (critical for crossover alignment).
   - No: assign a new innovation number, record it.
5. Create the connection with a small random weight.

**Add Node Mutation:**
1. Pick a random enabled connection.
2. Disable it.
3. Create a new hidden node.
4. Create two new connections:
   - from -> new_node (weight = 1.0, preserving original signal initially)
   - new_node -> to (weight = original connection's weight)
5. Both new connections get innovation numbers (same logic: check if this structural change has happened this generation).

**Why weight clamping at [-4.0, 4.0]:** Without bounds, weights drift to extreme values, making the sigmoid saturate and the network unresponsive to input changes. 4.0 through sigmoid gives 0.982, which is near-saturated but not completely dead. This is standard practice in neuroevolution.

### 5.4 Crossover

```rust
pub fn crossover(parent_a: &NeatGenome, parent_b: &NeatGenome,
                 rng: &mut SeededRng) -> NeatGenome {
    // parent_a is the more fit parent (or equal)
    let (fitter, weaker) = if parent_a.fitness >= parent_b.fitness {
        (parent_a, parent_b)
    } else {
        (parent_b, parent_a)
    };

    let mut child_connections = Vec::new();
    let mut child_nodes = fitter.nodes.clone(); // Start with fitter's node set

    let mut i = 0; // index into fitter
    let mut j = 0; // index into weaker

    // Both parents' connections are sorted by innovation number
    while i < fitter.connections.len() || j < weaker.connections.len() {
        let inn_a = fitter.connections.get(i).map(|c| c.innovation);
        let inn_b = weaker.connections.get(j).map(|c| c.innovation);

        match (inn_a, inn_b) {
            (Some(a), Some(b)) if a == b => {
                // Matching gene: randomly pick from either parent
                let gene = if rng.gen_bool(0.5) {
                    fitter.connections[i]
                } else {
                    weaker.connections[j]
                };
                child_connections.push(gene);
                i += 1;
                j += 1;
            }
            (Some(a), Some(b)) if a < b => {
                // Disjoint gene from fitter: include it
                child_connections.push(fitter.connections[i]);
                i += 1;
            }
            (Some(_), Some(_)) => {
                // Disjoint gene from weaker: SKIP (fitter parent dominates)
                j += 1;
            }
            (Some(_), None) => {
                // Excess gene from fitter: include it
                child_connections.push(fitter.connections[i]);
                i += 1;
            }
            (None, Some(_)) => {
                // Excess gene from weaker: SKIP
                j += 1;
            }
            (None, None) => break,
        }
    }

    // Ensure child has all nodes referenced by its connections
    // (weaker parent's matching genes might reference nodes not in fitter's set)
    for conn in &child_connections {
        ensure_node_exists(&mut child_nodes, conn.from, &fitter.nodes, &weaker.nodes);
        ensure_node_exists(&mut child_nodes, conn.to, &fitter.nodes, &weaker.nodes);
    }

    NeatGenome {
        id: new_genome_id(),
        nodes: child_nodes,
        connections: child_connections,
        fitness: 0.0,
        species_id: None,
    }
}
```

**Why disjoint/excess genes come only from the fitter parent:** This is the original NEAT paper's prescription. The fitter genome's structure is "proven" by its higher fitness. The weaker genome's unique structural innovations haven't proven themselves and would add noise. In the case of equal fitness, we could include from both, but keeping it simple (arbitrary tie-breaking) avoids genome bloat.

**Critical invariant: connections must be sorted by innovation number.** Both insertion during mutation and the crossover merge depend on this. Use `connections.sort_by_key(|c| c.innovation)` after mutations, or maintain sorted order during insertion.

### 5.5 Innovation Counter

```rust
// brain/innovation.rs

pub struct InnovationCounter {
    next_innovation: InnovationNumber,
    // Records for THIS generation only - reset each generation
    generation_innovations: HashMap<(NodeId, NodeId), InnovationNumber>,
    next_node_id: NodeId,
    generation_node_splits: HashMap<InnovationNumber, NodeId>,
}

impl InnovationCounter {
    pub fn get_connection_innovation(&mut self, from: NodeId, to: NodeId) -> InnovationNumber {
        if let Some(&inn) = self.generation_innovations.get(&(from, to)) {
            inn  // Same structural mutation happened already this generation
        } else {
            let inn = self.next_innovation;
            self.next_innovation = InnovationNumber(inn.0 + 1);
            self.generation_innovations.insert((from, to), inn);
            inn
        }
    }

    pub fn get_node_for_split(&mut self, split_connection: InnovationNumber) -> NodeId {
        if let Some(&node) = self.generation_node_splits.get(&split_connection) {
            node  // Same connection was split already this generation
        } else {
            let node = self.next_node_id;
            self.next_node_id = NodeId(node.0 + 1);
            self.generation_node_splits.insert(split_connection, node);
            node
        }
    }

    pub fn reset_generation(&mut self) {
        self.generation_innovations.clear();
        self.generation_node_splits.clear();
        // next_innovation and next_node_id persist across generations!
    }
}
```

**Why per-generation tracking:** If two different genomes independently mutate the same connection (from node 5 to node 12) in the same generation, they should get the same innovation number. This is the key insight of NEAT - it enables meaningful crossover between genomes that independently discovered the same structural innovation. But across generations, the same structural change is a different event in evolutionary history and gets a new number.

**Why `generation_node_splits`:** When connection X is split (add-node mutation), the new node should get the same ID if the same split happens independently in another genome this generation. This ensures the resulting node genes align during crossover.

### 5.6 Speciation

```rust
// evolution/species.rs

pub struct Species {
    pub id: SpeciesId,
    pub representative: NeatGenome,  // Random member from last generation
    pub members: Vec<GenomeId>,      // Genomes in this species this generation
    pub best_fitness: f32,           // Best fitness ever achieved
    pub generations_stagnant: u32,   // Generations since best_fitness improved
    pub average_fitness: f32,        // For offspring allocation
}

pub fn compatibility_distance(a: &NeatGenome, b: &NeatGenome, config: &NeatConfig) -> f32 {
    let mut matching = 0;
    let mut disjoint = 0;
    let mut excess = 0;
    let mut weight_diff_sum = 0.0;

    let mut i = 0;
    let mut j = 0;
    let max_inn_a = a.connections.last().map(|c| c.innovation).unwrap_or(InnovationNumber(0));
    let max_inn_b = b.connections.last().map(|c| c.innovation).unwrap_or(InnovationNumber(0));

    while i < a.connections.len() && j < b.connections.len() {
        let inn_a = a.connections[i].innovation;
        let inn_b = b.connections[j].innovation;

        if inn_a == inn_b {
            matching += 1;
            weight_diff_sum += (a.connections[i].weight - b.connections[j].weight).abs();
            i += 1;
            j += 1;
        } else if inn_a < inn_b {
            if inn_a > max_inn_b { excess += 1; } else { disjoint += 1; }
            i += 1;
        } else {
            if inn_b > max_inn_a { excess += 1; } else { disjoint += 1; }
            j += 1;
        }
    }
    // Remaining genes are excess
    excess += (a.connections.len() - i) + (b.connections.len() - j);

    let n = a.connections.len().max(b.connections.len()).max(1) as f32;
    let avg_weight_diff = if matching > 0 { weight_diff_sum / matching as f32 } else { 0.0 };

    // The NEAT formula
    (config.c1_excess * excess as f32 / n)
        + (config.c2_disjoint * disjoint as f32 / n)
        + (config.c3_weight * avg_weight_diff)
}
```

**Speciation algorithm (each generation):**

1. Each existing species keeps its representative from last generation.
2. Clear all member lists.
3. For each genome in the new population:
   - Compare against each species' representative.
   - If distance < `compatibility_threshold`: add to that species.
   - If no match: create a new species with this genome as representative.
4. Remove empty species.
5. Each species picks a new random representative from its current members.

**Dynamic threshold adjustment:** If the number of species drifts too far from a target (say, 8-15 species for a population of 150), nudge the threshold. Too many species? Increase threshold by 0.1. Too few? Decrease by 0.1. This prevents both speciation collapse (one mega-species) and fragmentation (50 species of 3 members each).

### 5.7 Fitness Sharing

```rust
pub fn compute_adjusted_fitness(species: &Species, genomes: &[NeatGenome]) -> Vec<f32> {
    let n = species.members.len() as f32;
    species.members.iter().map(|&gid| {
        let genome = &genomes[gid];
        genome.fitness / n  // Explicit fitness sharing: divide by species size
    }).collect()
}
```

**Why simple division by species size:** The original NEAT paper uses a more complex sharing function based on pairwise compatibility distances. For a Rust beginner, that's O(n^2) per species and hard to understand. Simple division by species size achieves the same goal: large species don't dominate, small species get a relative boost. The difference in evolutionary dynamics is negligible for populations under 500.

### 5.8 Offspring Allocation and Reproduction

```
For each generation:
1. Compute adjusted fitness for every genome
2. Sum adjusted fitness per species
3. Allocate offspring proportional to species' total adjusted fitness:
   species_offspring[i] = round(total_adjusted_fitness[i] / global_sum * population_size)
4. Adjust rounding so total == population_size

For each species:
5. Sort members by fitness (descending)
6. Keep top elitism_count unchanged (copy to next gen)
7. Remove bottom (1 - survival_rate) fraction
8. Fill remaining offspring slots:
   - 75% from crossover: pick two parents via tournament selection (size 3)
   - 25% from mutation-only: pick one parent, clone and mutate
9. Rare interspecies mating: with interspecies_mate_rate, pick second parent from different species
```

**Species stagnation:** If a species hasn't improved its `best_fitness` in `stagnation_limit` generations, it dies (all members removed, no offspring). Exception: the single best species is never killed, even if stagnant. This prevents total population collapse while still pruning dead ends.

---

## 6. Signal System

### 6.1 Vocabulary

**Fixed size, configurable, default 8 symbols.** Not evolving.

**Why fixed vocabulary:** An evolving vocabulary size (where mutations can add/remove symbols) creates a moving target for receiver neurons. If the vocabulary grows from 8 to 9 mid-evolution, all existing genomes suddenly have the wrong number of output neurons. This requires either dynamic network topology (complexity nightmare) or a max-vocabulary with unused symbols. The latter is what fixed vocabulary already is - symbol 7 being "unused" is equivalent to it being "possible but no agent has learned to use it." Let NEAT discover which symbols are worth using.

### 6.2 Broadcast Mechanics

When a prey's `signal_emit` output exceeds 0.5:
1. Determine which symbol: argmax over the `signal_symbol_*` outputs.
2. Deduct `signal_energy_cost` from the prey.
3. Create an `ActiveSignal`:
   - `origin`: prey's current position
   - `sender_id`: prey's ID
   - `symbol`: the chosen symbol
   - `tick_emitted`: current tick
   - `strength`: 1.0 (starts at full)

### 6.3 Reception Mechanics

During the sensor phase, for each prey:
1. Collect all `ActiveSignal`s within `hearing_range` of the prey.
2. For each symbol `s` in 0..vocab_size:
   - Find the strongest signal of type `s` within range.
   - Compute `strength * (1.0 - distance / hearing_range)` (distance attenuation).
   - Compute direction vector from prey to signal origin (normalized).
   - Feed into input neurons: `signal_s_strength` and `signal_s_direction_x`.

**Why direction matters:** Knowing that symbol 3 was broadcast to your north vs. south is critical information. "Eagle spotted to the north" is different from "eagle spotted to the south." Directionality enables spatial communication.

**Why only the strongest signal per symbol:** Multiple overlapping signals of the same symbol would require variable-length input, which neural networks can't handle. Taking the strongest is biologically plausible (you hear the loudest call) and keeps the input vector fixed-size.

### 6.4 Signal Cost and Honest Signaling

The energy cost of signaling is the enforcement mechanism for honest communication. Without cost, natural selection cannot distinguish between honest signaling and noise.

The math, applying Hamilton's rule (rB > C):
- C = 3.0 energy (signal cost)
- r = ~0.5 for siblings (same parent, asexual reproduction)
- B must exceed C/r = 6.0 per beneficiary

A prey that signals "eagle!" and causes 2 nearby siblings to hide under trees (each gaining survival benefit worth roughly 5.0 energy in avoided death probability) produces rB = 0.5 * (5.0 + 5.0) = 5.0, which just barely fails the threshold. But with 3 siblings nearby: rB = 0.5 * 15.0 = 7.5 > 3.0 = C. The signal is worth it.

This means honest signaling evolves only in kin clusters of 3+ relatives, which is exactly what spatial asexual reproduction produces. Isolated prey should stay quiet - and the simulation should show this pattern.

**Why signaling is non-exclusive with other actions:** A prey should be able to scream "eagle!" while also running. If signaling replaced movement, the cost would be dying from both the energy cost AND the lost movement, making signaling never worth it. Non-exclusivity lets the cost be tuned independently.

### 6.5 Deceptive Signaling

Deception emerges naturally when the cost-benefit equation shifts. A prey might signal "eagle!" (causing neighbors to freeze/hide) and then steal their food while they're hiding. For this to work:

1. The cheater must be far enough from the real threat that it isn't endangered.
2. The cheater must be near enough to competitors that they react.
3. The cheater must then exploit the reaction (eat food, move to better position).

The simulation doesn't need to encode deception - it needs to NOT prevent it. As long as the signal system has no built-in truth verification, dishonest signals are possible, and natural selection will find them if they're profitable. The stats system (section 10) detects deception post-hoc by correlating signals with actual threats vs. signaler benefit.

### 6.6 Signal Propagation Summary

```
Signal lifecycle:
  tick T:   prey emits symbol S at position (x, y), strength = 1.0
  tick T+1: nearby prey receive S with distance-attenuated strength
            signal strength decays by signal_decay_rate
  tick T+2: signal weaker, still audible to close prey
  tick T+3: if signal_lifetime=3, signal removed
```

No speed-of-sound delay (overcomplicated for a grid world). Signals are instantaneous within range but decay over time, modeling how a shout fades.

---

## 7. Fitness and Selection

### 7.1 Fitness Function

```rust
pub fn compute_fitness(prey: &Prey, world: &World, config: &EvolutionConfig,
                       kin_tracker: &KinTracker) -> f32 {
    let survival_score = prey.age as f32 * config.fitness_survival_weight;
    let energy_score = prey.energy * config.fitness_energy_weight;
    let offspring_score = prey.offspring_count as f32 * config.fitness_offspring_weight;

    // Kin bonus: reward for having relatives alive at end of generation
    let kin_alive = kin_tracker.count_living_kin(prey.lineage, &world.prey);
    let relatedness = kin_tracker.average_relatedness(prey.id);
    let kin_score = kin_alive as f32 * relatedness * config.fitness_kin_bonus;

    survival_score + energy_score + offspring_score + kin_score
}
```

**Why offspring gets the highest weight (2.0):** Reproduction is the entire point of evolution. A prey that lives long but never reproduces is an evolutionary dead end. The 2.0 weight ensures that reproductive success dominates fitness, while survival and energy still matter (you can't reproduce if you're dead).

**Why kin bonus exists as a separate term:** Without explicit kin bonus, kin selection only works indirectly through shared genes. By adding a small direct fitness bonus for having living relatives, we strengthen the selection pressure for alarm calling. This is a "simulation shortcut" - in real life, kin selection works through gene frequency math over thousands of generations. We don't have thousands of generations, so we boost the signal. The kin_bonus weight (0.5) is deliberately low - it nudges, it doesn't dominate.

### 7.2 Kin Tracking

```rust
// evolution/kin.rs

pub struct KinTracker {
    // Maps prey ID -> lineage info
    lineages: HashMap<PreyId, LineageInfo>,
}

pub struct LineageInfo {
    pub lineage_id: LineageId,
    pub parent_id: Option<PreyId>,
    pub generation_born: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageId(pub u32);
```

**Relatedness calculation (approximate, not exact):**
- Same parent: r = 0.5 (siblings from asexual reproduction share half the mutated genome)
- Same grandparent: r = 0.25
- Same lineage, unknown exact relation: r = 0.125 (conservative estimate)

**Why approximate:** Exact genetic relatedness requires comparing every gene between every pair of individuals - O(n^2 * genome_size). For 100 prey with genomes of 50+ genes, that's expensive every tick. Lineage-based approximation is O(1) per lookup and biologically reasonable (animals use spatial proximity and familiarity as kin recognition cues, not genome sequencing).

**Lineage assignment:**
- Initial population: each prey gets a unique lineage.
- Offspring: inherits parent's lineage.
- After N generations without improvement, a lineage ID may be "refreshed" (but this is optional and only for stat clarity).

### 7.3 Selection Method

**Tournament selection (size 3).**

```rust
pub fn tournament_select(species_members: &[&NeatGenome], rng: &mut SeededRng) -> &NeatGenome {
    let mut best = species_members[rng.gen_range(0..species_members.len())];
    for _ in 1..3 {
        let candidate = species_members[rng.gen_range(0..species_members.len())];
        if candidate.fitness > best.fitness {
            best = candidate;
        }
    }
    best
}
```

**Why tournament size 3:** Size 2 has weak selection pressure (the better of two random picks). Size 5+ is too strong (always picks near-best, reducing diversity). Size 3 is the standard for NEAT - it provides moderate selection pressure that balances exploitation and exploration.

**Why not roulette wheel:** Roulette wheel selection is sensitive to fitness scaling. If one individual has fitness 1000 and the rest have 1-10, roulette always picks the outlier. Tournament selection is rank-based and handles this naturally.

### 7.4 Population Collapse Prevention

Several mechanisms:
1. **Minimum species protection:** The best species cannot be killed for stagnation.
2. **Elitism:** Top N genomes always survive (default 2 per species).
3. **Minimum population:** If `min_prey_alive` is reached during simulation, the generation ends early. Surviving prey get a large fitness bonus (they beat the others).
4. **Adjusted fitness sharing:** Small species get disproportionately more offspring, preventing a single dominant species from taking over.
5. **Interspecies mating:** At a very low rate (0.1%), parents from different species can produce offspring. This injects novelty and prevents species from becoming too insular.

---

## 8. World Design

### 8.1 Grid Dimensions and Terrain

**Default: 80x60 grid (4800 cells).**

**Why this size:** With 100 prey and ~10 predators, that's ~1 entity per 44 cells. Prey will frequently encounter each other (necessary for communication) but aren't packed so tightly that movement is meaningless. The grid should be large enough that a prey can't see the whole world (vision range 8 covers ~200 cells, about 4% of the world), forcing local information and creating the need for communication about distant threats.

**Terrain generation algorithm:**

```
1. Fill everything with Open.
2. Place tree clusters: pick N random seeds, grow each into a cluster of 5-15 connected cells
   using random walk from the seed. (Clusters model forests - prey can hide in them.)
3. Place rock outcrops: same cluster algorithm, smaller clusters (3-8 cells).
4. Place water: generate 1-2 river-like features using random walk with momentum
   (each step has 70% chance of continuing in same direction). Width 1-2 cells.
5. Place bushes: scattered randomly (no clusters), modeling underbrush.
6. Fill remaining to target percentages, converting excess Open cells.
7. Validate: ensure no completely enclosed areas (flood-fill from a random Open cell
   should reach >90% of all passable cells). If not, regenerate.
```

**Why clusters for trees and rocks:** Isolated single trees are nearly useless for hiding - a prey needs to reach a tree when a predator appears, so trees must be reachable from nearby positions. Clusters create "safe zones" that prey can learn to stay near. This also creates interesting territorial dynamics.

### 8.2 Predator Behaviors (expanded)

**Aerial (Eagle) - 2 by default:**
- Speed: 3 cells/tick (very fast, prey moves 1)
- Vision: 15 cells (sees everything below from the sky)
- Weakness: cannot attack prey under trees (canopy blocks dive)
- Behavior: circles in large patrol arcs, locks onto the nearest exposed prey, dives straight at it
- After kill: rests for 30 ticks (digesting)
- Key trait for prey to learn: eagles are fast but can't reach you under trees. When you see an eagle, run for trees. Signal "eagle" to warn others to do the same.

**Ground (Snake) - 3 by default:**
- Speed: 1 cell/tick (same as prey)
- Vision: 4 cells (short, ambush predator)
- Weakness: cannot attack prey on rocks (can't climb)
- Behavior: lurks in grass/bushes, waits for prey to wander close, then strikes
- Ambush mechanic: if a snake hasn't been spotted (prey doesn't have it in vision cone), its first strike has +2 range (lunge). Once spotted, normal range.
- Key trait for prey to learn: snakes are invisible until close. Signal "snake at location" after spotting one helps others avoid the area. Climbing rocks provides safety.

**Pack (Wolves) - 4 in one pack by default:**
- Speed: 2 cells/tick (faster than prey)
- Vision: 10 cells
- Weakness: cannot coordinate when prey scatter (each wolf picks a different target, reducing effectiveness)
- Behavior: move as a group, fan out to surround a target cluster of prey
- Coordination mechanic: wolves share information. If any wolf sees prey, all wolves know.
- Scatter mechanic: wolves are effective against grouped prey (>3 prey within 5 cells of each other). If prey are scattered (no groups of 3+ within 5 cells), each wolf hunts independently with 50% reduced effectiveness (moves randomly half the time instead of toward target).
- Key trait for prey to learn: scatter when wolves approach. This is the OPPOSITE of the eagle response (cluster under trees). The conflict between these two responses is the core evolutionary challenge.

### 8.3 Food Distribution

- Initial density: 15% of cells have food (720 food items on 80x60 grid).
- Food spawns preferentially on Grass and Open terrain (not on Water, less on Rock).
- When eaten, the cell has a regrow timer (50 ticks). After that, food may respawn (80% chance per tick once timer expires).
- Food energy: 20.0 per item.
- Prey basal cost: 0.5/tick, so one food item sustains a prey for 40 ticks.
- With 100 prey each eating roughly every 40 ticks, the food system reaches equilibrium when about 60% of food cells are occupied.

**Why these numbers:** A prey needs to eat roughly every 40 ticks to survive. With 2000 ticks per generation, a prey must find ~50 food items. With 720 food items regenerating every ~50 ticks, there are roughly 720 * (2000/50) = 28,800 food-ticks available. For 100 prey needing 50 each = 5,000 food items needed. There's enough food that starvation isn't the main threat - predators are. This is important: if food is too scarce, prey evolve to be foragers, not communicators.

---

## 9. Configuration and Reproducibility

### 9.1 Configuration Loading

```rust
// config.rs

use serde::Deserialize;
use std::path::PathBuf;

impl SimConfig {
    pub fn load(path: Option<PathBuf>) -> Result<Self, ConfigError> {
        let default_toml = include_str!("../config/default.toml");
        let mut config: SimConfig = toml::from_str(default_toml)?;

        if let Some(path) = path {
            let user_toml = std::fs::read_to_string(path)?;
            let overrides: SimConfig = toml::from_str(&user_toml)?;
            config.merge(overrides); // User values override defaults
        }

        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.signal.vocab_size == 0 { return Err(ConfigError::Invalid("vocab_size must be > 0")); }
        if self.prey.initial_count < self.neat.population_size {
            return Err(ConfigError::Invalid("initial prey count must be >= population size"));
        }
        // ... more validation
        Ok(())
    }
}
```

**CLI with `clap`:**

```rust
// main.rs

#[derive(Parser)]
#[command(name = "predator-prey-evolution")]
struct Cli {
    /// Path to TOML config file
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Random seed (overrides config file)
    #[arg(short, long)]
    seed: Option<u64>,

    /// Maximum generations to run
    #[arg(short, long)]
    generations: Option<u32>,

    /// Enable terminal visualization
    #[arg(short, long)]
    viz: bool,

    /// Export stats path
    #[arg(short, long)]
    output: Option<String>,

    /// Load from checkpoint
    #[arg(long)]
    checkpoint: Option<PathBuf>,
}
```

**Why TOML over JSON or YAML:** TOML is Rust's native config format (Cargo.toml), has excellent serde support, allows comments (JSON doesn't), and has simpler syntax than YAML (no indentation-sensitivity, no gotchas like Norway == false). A Rust beginner will already be familiar with it from Cargo.toml.

### 9.2 Reproducibility

```rust
// rng.rs

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct SeededRng {
    inner: ChaCha8Rng,
    seed: u64,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
            seed,
        }
    }

    pub fn seed(&self) -> u64 { self.seed }

    // Delegate rand methods
    pub fn gen_range<T, R>(&mut self, range: R) -> T
    where
        T: rand::distributions::uniform::SampleUniform,
        R: rand::distributions::uniform::SampleRange<T>,
    {
        use rand::Rng;
        self.inner.gen_range(range)
    }

    pub fn gen_bool(&mut self, p: f64) -> bool {
        use rand::Rng;
        self.inner.gen_bool(p)
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        use rand::seq::SliceRandom;
        slice.shuffle(&mut self.inner);
    }
}
```

**Why `ChaCha8Rng` specifically:**
- Deterministic across all platforms (same seed -> same sequence on Windows, Linux, macOS, WASM).
- `ChaCha8` is the fast variant (8 rounds instead of 20). For simulation randomness (not cryptography), 8 rounds is more than sufficient and measurably faster.
- The `rand` crate's `StdRng` is `ChaCha12` and may change across versions. Pinning to `ChaCha8Rng` explicitly guarantees reproducibility even across `rand` crate updates.

**Single RNG instance:** The `World` struct owns the only `SeededRng`. Every system that needs randomness borrows it (`&mut world.rng`). This ensures deterministic execution order - the RNG state advances in a fixed sequence dictated by the tick loop order. Multiple RNG instances would make order-of-consumption non-deterministic.

### 9.3 Checkpointing

```rust
// Uses serde for serialization. Every struct that needs saving derives Serialize/Deserialize.

pub fn save_checkpoint(world: &World, population: &Population, path: &Path) -> Result<(), io::Error> {
    let checkpoint = Checkpoint {
        world_state: world.to_serializable(),
        population: population.clone(),
        rng_state: world.rng.seed(), // Save original seed + tick count to reconstruct
        tick: world.tick,
        generation: world.generation,
    };
    let bytes = bincode::serialize(&checkpoint)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn load_checkpoint(path: &Path) -> Result<Checkpoint, io::Error> {
    let bytes = std::fs::read(path)?;
    let checkpoint: Checkpoint = bincode::deserialize(&bytes)?;
    Ok(checkpoint)
}
```

**Why `bincode` and not JSON:** Checkpoints contain genome data for 150+ organisms, each with variable-length connection lists. Binary serialization is 5-10x smaller and 10-20x faster than JSON. Checkpoints are machine-readable, not human-readable - that's what the stats CSV export is for.

**RNG reconstruction:** Rather than serializing the full RNG state (which is opaque), we store the original seed and the current tick. To resume, we create an RNG with the same seed and advance it the correct number of times. This is slower than state serialization but simpler and crate-version-independent. For a faster approach, `ChaCha8Rng` implements `Serialize`/`Deserialize` if the `serde1` feature is enabled on `rand_chacha`.

---

## 10. Statistics and Analysis

### 10.1 Per-Generation Metrics

```rust
// stats/collector.rs

pub struct GenerationStats {
    pub generation: u32,
    pub ticks_elapsed: u64,

    // Population
    pub prey_alive_end: u32,
    pub prey_born: u32,
    pub prey_died_predation: u32,
    pub prey_died_starvation: u32,
    pub prey_died_old_age: u32, // (if we add age limit)

    // Fitness
    pub fitness_mean: f32,
    pub fitness_max: f32,
    pub fitness_std_dev: f32,
    pub fitness_by_species: Vec<(SpeciesId, f32)>,

    // NEAT
    pub species_count: u32,
    pub avg_genome_size: f32,   // Average connection count
    pub avg_hidden_nodes: f32,
    pub largest_genome: u32,    // Connection count of biggest genome

    // Predation
    pub kills_by_aerial: u32,
    pub kills_by_ground: u32,
    pub kills_by_pack: u32,

    // Signals
    pub total_signals_emitted: u32,
    pub signals_per_symbol: Vec<u32>,           // How often each symbol was used
    pub signal_predator_correlation: Vec<f32>,  // Per-symbol: correlation with nearby predator type
    pub signal_entropy: f32,                     // Shannon entropy of symbol distribution

    // Communication analysis
    pub topographic_similarity: f32,            // How structured is the signal-meaning mapping
    pub mutual_info_signal_predator: f32,       // Mutual information between signals and predator types
    pub deception_index: f32,                   // Signals emitted with no actual threat nearby

    // Kin selection
    pub avg_kin_cluster_size: f32,
    pub kin_signal_rate: f32,   // Signal rate when kin nearby vs. when not
}
```

### 10.2 Communication Analysis (the key metrics)

**Topographic Similarity (TopSim):**

Measures whether similar situations produce similar signals. Computed as the Spearman rank correlation between:
- Pairwise distances in "situation space" (what the prey was sensing when it signaled)
- Pairwise distances in "signal space" (what symbol was emitted)

```rust
pub fn topographic_similarity(signal_events: &[SignalEvent]) -> f32 {
    // SignalEvent records: (sensor_state_hash, symbol, tick)
    // For each pair of signal events:
    //   situation_distance = euclidean distance between sensor state vectors
    //   signal_distance = 0 if same symbol, 1 if different
    // TopSim = spearman_correlation(situation_distances, signal_distances)

    // High TopSim (>0.5): similar situations produce same signals = structured communication
    // Low TopSim (~0.0): random signal-situation mapping = no meaning
    // Negative TopSim: actively anti-correlated = confusing, likely noise
}
```

**Mutual Information between Signal and Predator Type:**

```rust
pub fn mutual_info_signal_predator(signal_events: &[SignalEvent]) -> f32 {
    // For each signal event, record:
    //   - Which symbol was emitted
    //   - Which predator type (if any) was nearest to the signaler
    // Compute I(Signal; PredatorType) = sum over s,p of P(s,p) * log(P(s,p) / (P(s) * P(p)))

    // High MI (>1.0 bit): signals carry information about predator types
    // Zero MI: signals are independent of predator presence = meaningless
}
```

**Deception Index:**

```rust
pub fn deception_index(signal_events: &[SignalEvent], world: &World) -> f32 {
    // A signal is "deceptive" if:
    //   1. No predator is within 2x the signaler's vision range (no real threat)
    //   2. The signaler benefits in the next 10 ticks (gains food, competitors flee)
    //
    // deception_index = deceptive_signals / total_signals
    //
    // 0.0 = all signals are honest
    // >0.3 = significant deception in the population
}
```

**Why these three metrics together:** TopSim tells you if communication has structure. MI tells you if that structure carries predator information. Deception index tells you if some agents are exploiting the system. Together, they paint a complete picture:
- High TopSim + High MI + Low Deception = honest alarm call system (the goal)
- High TopSim + High MI + High Deception = mixed strategy (some honest, some deceptive)
- Low TopSim + Low MI = no meaningful communication yet
- High TopSim + Low MI = structured signals, but about something other than predators (maybe food?)

### 10.3 Export

```rust
// stats/export.rs

pub enum ExportFormat {
    Csv,
    Json,
}

pub fn export_generation_stats(stats: &[GenerationStats], config: &StatsConfig) -> Result<(), io::Error> {
    match config.export_format {
        ExportFormat::Csv => {
            // One CSV file: generations.csv
            // One row per generation, columns for all metrics
            // Plus: signals_by_symbol.csv (generation, symbol, count, correlation)
            // Plus: species.csv (generation, species_id, size, avg_fitness, stagnation)
        }
        ExportFormat::Json => {
            // One JSON file per generation: gen_0042.json
            // Contains full GenerationStats struct
        }
    }
}
```

**Why CSV as default:** Researchers will want to plot metrics in Python/R/Excel. CSV is the universal interchange format. JSON is available for programmatic access but CSV is the primary output.

---

## 11. API Boundary for Future Frontends

### 11.1 The Snapshot Pattern

The core simulation never exposes `&mut` references to external code. Instead, it produces read-only snapshots.

```rust
// snapshot.rs

#[derive(Clone)]
pub struct WorldSnapshot {
    pub tick: u64,
    pub generation: u32,
    pub width: u32,
    pub height: u32,
    pub terrain: Vec<Terrain>,       // Shared (Arc) if cloning is expensive
    pub food: Vec<bool>,             // Simplified: is there food here?
    pub prey: Vec<AgentSnapshot>,
    pub predators: Vec<PredatorSnapshot>,
    pub signals: Vec<SignalSnapshot>,
}

#[derive(Clone)]
pub struct AgentSnapshot {
    pub id: u32,
    pub x: u32,
    pub y: u32,
    pub energy: f32,
    pub facing: Direction,
    pub last_action: Action,
    pub last_signal: Option<Symbol>,
    pub is_climbing: bool,
    pub is_hidden: bool,
    pub lineage: u32,
}

#[derive(Clone)]
pub struct PredatorSnapshot {
    pub id: u32,
    pub kind: PredatorKind,
    pub x: u32,
    pub y: u32,
    pub state: PredatorState,
}

#[derive(Clone)]
pub struct SignalSnapshot {
    pub x: u32,
    pub y: u32,
    pub symbol: u8,
    pub strength: f32,
}
```

### 11.2 The Observer Interface

```rust
// lib.rs

pub trait SimObserver {
    fn on_tick(&mut self, snapshot: &WorldSnapshot) {}
    fn on_generation_end(&mut self, stats: &GenerationStats) {}
    fn on_prey_death(&mut self, prey_id: u32, cause: DeathCause) {}
    fn on_signal_emitted(&mut self, sender_id: u32, symbol: u8, x: u32, y: u32) {}
}

pub struct Simulation {
    world: World,
    population: Population,
    config: SimConfig,
    observers: Vec<Box<dyn SimObserver>>,
}

impl Simulation {
    pub fn new(config: SimConfig) -> Self { /* ... */ }
    pub fn add_observer(&mut self, observer: Box<dyn SimObserver>) { /* ... */ }
    pub fn tick(&mut self) { /* ... notifies observers ... */ }
    pub fn run_generation(&mut self) { /* ... */ }
    pub fn run(&mut self) { /* run all generations */ }
    pub fn snapshot(&self) -> WorldSnapshot { /* ... */ }
}
```

**Why `SimObserver` trait and not channels:** Channels (`mpsc`) add async complexity and a Rust beginner will struggle with ownership of the receiver. The observer pattern is synchronous: the simulation calls `on_tick` at the end of each tick, the observer does its thing (render, log, whatever), and control returns. This is single-threaded and simple.

**Why `Box<dyn SimObserver>` (trait object) here and nowhere else:** This is the one place where runtime polymorphism is justified. The simulation core doesn't know or care what's observing it. It could be a terminal renderer, a Bevy scene updater, a WebSocket broadcaster, or nothing. The `dyn` dispatch cost is negligible (once per tick, not once per agent).

### 11.3 WASM Compatibility

The core compiles to WASM if:
1. No `std::fs` usage in the core (filesystem access only in `config.rs` and `stats/export.rs`, both behind `#[cfg(not(target_arch = "wasm32"))]`).
2. No threads (single-threaded by design).
3. No system time calls (tick counting is internal).
4. `rand_chacha` supports WASM out of the box.

For a web frontend:
```rust
// Exposed via wasm-bindgen
#[wasm_bindgen]
pub struct WasmSimulation {
    sim: Simulation,
}

#[wasm_bindgen]
impl WasmSimulation {
    pub fn new(config_toml: &str) -> Self { /* parse TOML, create sim */ }
    pub fn tick(&mut self) { self.sim.tick(); }
    pub fn snapshot_json(&self) -> String { /* serialize WorldSnapshot to JSON */ }
}
```

The WASM boundary serializes the snapshot to JSON (or a flat buffer) for JavaScript consumption. A Bevy/Three.js/Canvas renderer reads the snapshot and draws. The simulation logic is identical whether running native or in-browser.

---

## 12. Dependency Graph and Crate Layout

### External Dependencies

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
toml = "0.8"
rand = "0.8"
rand_chacha = "0.3"
clap = { version = "4", features = ["derive"] }
bincode = "1"                    # For checkpoints

[dev-dependencies]
criterion = "0.5"               # Benchmarks

[features]
default = ["terminal-viz"]
terminal-viz = ["dep:crossterm"]

[dependencies.crossterm]
version = "0.27"
optional = true
```

**Why these and nothing else:** Every dependency is a learning burden for a beginner. `serde` + `toml` for config, `rand` + `rand_chacha` for deterministic randomness, `clap` for CLI, `bincode` for checkpoints. That's it. No `itertools` (use standard iterators), no `rayon` (single-threaded is simpler to debug), no `log`/`tracing` (println with a verbosity flag is fine for v1). Add complexity when you need it, not before.

### Internal Dependency Graph

```
config.rs ─────────────────┐
rng.rs ────────────────────┤
                           ▼
                  brain/genome.rs
                  brain/activation.rs
                  brain/innovation.rs
                         │
                         ▼
                  brain/network.rs
                         │
       ┌─────────────────┼──────────────────┐
       ▼                 ▼                   ▼
  agent/prey.rs    agent/predator.rs    signal/message.rs
  agent/sensor.rs  agent/action.rs      signal/propagation.rs
       │                 │                   │
       └─────────┬───────┘───────────────────┘
                 ▼
            world/grid.rs
            world/terrain.rs
            world/food.rs
            world/entity.rs
                 │
                 ▼
          evolution/population.rs
          evolution/species.rs
          evolution/fitness.rs
          evolution/selection.rs
          evolution/reproduction.rs
          evolution/kin.rs
                 │
                 ▼
           stats/collector.rs
           stats/metrics.rs
           stats/export.rs
                 │
                 ▼
           snapshot.rs
                 │
                 ▼
        viz/terminal.rs (feature-gated)
        viz/dashboard.rs (feature-gated)
```

The critical insight: **dependencies flow downward.** `brain` knows nothing about `world`. `world` knows about `brain` (it contains prey with brains). `evolution` knows about both (it evaluates fitness in the world and manipulates genomes). `stats` knows about everything (it measures everything). `viz` only knows `snapshot` (it renders a frozen view).

---

## 13. Error Handling Strategy

**No `unwrap()` in the simulation hot loop.** Every error that can occur during a tick is either:
- **Prevented by construction:** Grid bounds are checked at the type level (Position is always valid). Genome construction ensures valid node references.
- **Handled as a no-op:** Trying to eat where there's no food? Action fails silently, prey wastes a tick. Trying to move off the grid? Stay in place.
- **Logged and continued:** A genome that somehow produces an invalid network (shouldn't happen, but defensive) gets a "null brain" that always outputs Idle. The prey will die quickly and its genes will be selected against.

**`Result` for setup operations:** Config loading, checkpoint loading, file I/O - these return `Result<T, Error>` and propagate with `?`. The `main()` function handles them with user-friendly error messages.

```rust
fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let config = SimConfig::load(cli.config)?;
    // ...
    Ok(())
}
```

**Custom error enum for the project:**

```rust
#[derive(Debug)]
pub enum SimError {
    Config(String),
    Io(std::io::Error),
    Checkpoint(String),
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SimError::Config(msg) => write!(f, "Configuration error: {msg}"),
            SimError::Io(e) => write!(f, "IO error: {e}"),
            SimError::Checkpoint(msg) => write!(f, "Checkpoint error: {msg}"),
        }
    }
}

impl std::error::Error for SimError {}
impl From<std::io::Error> for SimError { /* ... */ }
```

**Why a custom error enum and not `anyhow`:** `anyhow` is convenient but hides error types. For a library that might be consumed by other code (Bevy frontend, WASM wrapper), callers need to match on specific error variants. A beginner also benefits from seeing error handling explicitly rather than magic `?` propagation through `anyhow::Result`.

---

## 14. Beginner Pitfall Guide

### Ownership Issues You Will Hit

**1. Iterating prey while modifying the world.**

The simulation tick needs to iterate over all prey (to compute sensor readings) while also reading world state (terrain, food, other prey positions). In Rust, you can't have `&world.prey` (immutable borrow for iteration) and `&mut world` (mutable borrow for modifications) at the same time.

**Solution:** Phase separation. The tick loop is split into phases (sensor, brain, action) where each phase either reads the world or writes to it, never both simultaneously. Specifically:
- Sensor phase: read-only pass, produces `Vec<SensorReading>` (owned data, no borrows).
- Brain phase: reads `Vec<SensorReading>` (no world borrow), produces `Vec<Action>`.
- Action phase: writes to world using the `Vec<Action>` (no reading required since decisions are already made).

**2. Prey referencing other prey.**

A prey wants to "see" other prey. You can't hold references into the `Vec<Prey>` while iterating it.

**Solution:** Use indices, not references. `nearest_prey` is `Option<usize>` (index into `world.prey`), not `Option<&Prey>`. All inter-agent lookups go through the index.

**3. The innovation counter.**

The `InnovationCounter` must be shared across all mutations in a generation. If you try to pass `&mut InnovationCounter` while also iterating genomes, you'll hit borrow conflicts.

**Solution:** The evolution module runs mutations in a loop that owns the counter:
```rust
let mut innovations = InnovationCounter::new(/* ... */);
for genome in &mut next_generation {
    mutate(genome, &config.neat, &mut rng, &mut innovations);
}
innovations.reset_generation();
```
One mutable owner, lending it out one at a time. Simple.

**4. String formatting in hot loops.**

`format!()` and `println!()` allocate. In the tick loop (which runs millions of times per generation), never format strings. Stats collection should use numeric types and format only during the export phase.

### What Will Be Hard to Debug

**1. Silent genome corruption.** If a mutation creates a connection to a non-existent node, the network construction phase will either panic (good, fails fast) or silently produce a broken network (bad, fails late). The defense: `NeatGenome::validate()` method that checks all invariants. Call it in debug builds after every mutation.

```rust
impl NeatGenome {
    #[cfg(debug_assertions)]
    pub fn validate(&self) {
        let node_ids: HashSet<NodeId> = self.nodes.iter().map(|n| n.id).collect();
        for conn in &self.connections {
            assert!(node_ids.contains(&conn.from), "Connection references non-existent source node");
            assert!(node_ids.contains(&conn.to), "Connection references non-existent target node");
        }
        // Check innovation numbers are sorted
        for w in self.connections.windows(2) {
            assert!(w[0].innovation <= w[1].innovation, "Connections not sorted by innovation");
        }
    }
}
```

**2. Fitness landscape flatness.** If all prey die quickly (bad initial conditions), fitness differences are tiny, and selection has nothing to work with. The sign: fitness mean and max are nearly identical, species don't differentiate. The fix: make the world safer initially (fewer predators in generation 0, ramp up) or add a "grace period" where predators don't hunt for the first 100 ticks.

**3. Speciation collapse.** If the compatibility threshold is wrong, either everything is one species (no protection for novel topologies) or every genome is its own species (no crossover). The sign: species_count is 1 or equals population_size. The fix: dynamic threshold adjustment (section 5.6).

**4. Signal semantics not emerging.** If after 500 generations the mutual information between signals and predator types is still near zero, the simulation parameters need tuning. Most likely culprits: (a) signal cost too high (nobody signals), (b) hearing range too small (signals don't reach anyone), (c) predators too deadly (prey die before they can learn), (d) generation length too short (not enough time for communication patterns to matter within a single evaluation).

---

## Appendix A: Complete Input/Output Neuron Map

| Index | Type | Name | Range | Description |
|-------|------|------|-------|-------------|
| 0 | Input | nearest_aerial_dist | [-1, 1] | Distance to nearest eagle (-1 = none visible) |
| 1 | Input | nearest_aerial_dx | [-1, 1] | X direction to nearest eagle |
| 2 | Input | nearest_aerial_dy | [-1, 1] | Y direction to nearest eagle |
| 3 | Input | nearest_ground_dist | [-1, 1] | Distance to nearest snake |
| 4 | Input | nearest_ground_dx | [-1, 1] | X direction to nearest snake |
| 5 | Input | nearest_ground_dy | [-1, 1] | Y direction to nearest snake |
| 6 | Input | nearest_pack_dist | [-1, 1] | Distance to nearest wolf |
| 7 | Input | nearest_pack_dx | [-1, 1] | X direction to nearest wolf |
| 8 | Input | nearest_pack_dy | [-1, 1] | Y direction to nearest wolf |
| 9 | Input | on_tree | {0, 1} | Currently on tree cell |
| 10 | Input | on_rock | {0, 1} | Currently on rock cell |
| 11 | Input | nearest_tree_dist | [0, 1] | Normalized distance to nearest tree |
| 12 | Input | nearest_rock_dist | [0, 1] | Normalized distance to nearest rock |
| 13 | Input | nearest_food_dist | [0, 1] | Normalized distance to nearest food |
| 14 | Input | nearest_food_dx | [-1, 1] | X direction to nearest food |
| 15 | Input | prey_density | [0, 1] | Nearby prey count, normalized |
| 16 | Input | nearest_prey_dist | [0, 1] | Distance to nearest prey |
| 17-32 | Input | signal_s_strength/dir | [0, 1]/[-1, 1] | Per-symbol signal info (8 symbols x 2) |
| 33 | Input | own_energy | [0, 1] | Current energy / max energy |
| 34 | Input | is_protected | {0, 1} | Currently climbing or hidden |
| 35 | Input | ticks_since_signal | [0, 1] | Normalized cooldown |
| 36 | Output | move_north | [0, 1] | Movement desire north |
| 37 | Output | move_south | [0, 1] | Movement desire south |
| 38 | Output | move_east | [0, 1] | Movement desire east |
| 39 | Output | move_west | [0, 1] | Movement desire west |
| 40 | Output | eat | [0, 1] | Eat desire |
| 41 | Output | reproduce | [0, 1] | Reproduction desire |
| 42 | Output | climb | [0, 1] | Climb desire |
| 43 | Output | hide | [0, 1] | Hide desire |
| 44 | Output | idle | [0, 1] | Idle desire |
| 45 | Output | signal_emit | [0, 1] | Signal emission threshold |
| 46-53 | Output | signal_sym_0..7 | [0, 1] | Symbol selection (argmax) |

**Total: 36 inputs, 18 outputs, 54 fixed neurons.** Hidden neurons start at index 54 and grow from there.

---

## Appendix B: TOML Configuration Example

```toml
# predator-prey-evolution-communication default config

seed = 42

[world]
width = 80
height = 60
food_density = 0.15
food_energy = 20.0
food_regrow_ticks = 50
terrain_tree_pct = 0.10
terrain_rock_pct = 0.05
terrain_water_pct = 0.05
terrain_bush_pct = 0.08

[prey]
initial_count = 150
initial_energy = 100.0
max_energy = 200.0
energy_per_tick = 0.5
move_energy_cost = 1.0
signal_energy_cost = 3.0
reproduce_energy_cost = 60.0
reproduce_energy_threshold = 120.0
vision_range = 8
vision_angle = 60.0
hearing_range = 12

[predators]
aerial_count = 2
ground_count = 3
pack_count = 4
aerial_speed = 3
ground_speed = 1
pack_speed = 2
aerial_vision = 15
ground_vision = 4
pack_vision = 10
attack_cooldown = 30
kill_radius = 1

[neat]
population_size = 150
c1_excess = 1.0
c2_disjoint = 1.0
c3_weight = 0.4
compatibility_threshold = 3.0
weight_mutate_rate = 0.8
weight_perturb_rate = 0.9
weight_perturb_strength = 0.5
add_node_rate = 0.03
add_connection_rate = 0.05
disable_gene_rate = 0.01
interspecies_mate_rate = 0.001
stagnation_limit = 15
elitism_count = 2
survival_rate = 0.2

[signal]
vocab_size = 8
signal_range = 12
signal_decay_rate = 0.3
signal_lifetime = 3

[evolution]
generation_ticks = 2000
max_generations = 1000
min_prey_alive = 5
fitness_survival_weight = 1.0
fitness_energy_weight = 0.3
fitness_offspring_weight = 2.0
fitness_kin_bonus = 0.5
kin_relatedness_generations = 3

[stats]
export_every_n_generations = 10
export_format = "csv"
export_path = "output/"
track_signals = true
track_lineage = true
```

---

## Appendix C: Terminal Visualization Layout

```
Generation 42 | Tick 1,247/2,000 | Prey: 73/150 | Species: 8
═══════════════════════════════════════════════════════════════
  . . . . T T . . . . . . . # . . . . . . . . . . . . . . . .
  . . . * T T T . . . . . . . . . . . . . . . . . . . . . . .
  . . . . . T . . . o . . . . . . . . . W W . . . . . . . . .
  . . . . . . . . o o . . . . . . . . . . W . . . . . . . . .
  . R R . . . . . . o . . * . . . . . . . . . . . . . . . . .
  . R . . . . . A . . . . . . . . . . . . . . . . S . . . . .
  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

Legend: o=prey T=tree R=rock ~=water #=bush *=food
        A=eagle S=snake W=wolf !=signal
═══════════════════════════════════════════════════════════════
Fitness: avg=42.3 max=187.2 | Signals/tick: 3.2
Top species: #3 (28 members, avg_fit=51.2)
Signal usage: [0]=12% [1]=0% [2]=45% [3]=8% [4]=0% [5]=31% [6]=2% [7]=2%
MI(signal,predator): 0.34 bits | Deception: 8%
```

The terminal viz uses `crossterm` for cursor positioning and coloring. Refreshes every N ticks (configurable, default 5) to balance information density with performance.

---

## Appendix D: Build Order (Suggested Implementation Sequence)

For a Rust beginner, building this incrementally is critical. Here is the suggested order, where each step produces a runnable, testable artifact:

1. **config + rng** - Load TOML, create seeded RNG. Test: same seed produces same random sequence.

2. **world/grid + world/terrain** - Create grid, generate terrain, display as ASCII. Test: terrain percentages are correct, grid is connected.

3. **agent/prey (stub)** - Prey with position, energy, random movement. No brain yet. Test: prey wander, lose energy, die.

4. **world/food** - Food spawning and consumption. Test: prey eat food, energy increases, food regrows.

5. **agent/predator** - Predator types with behavior trees. No prey intelligence yet. Test: predators patrol, kill nearby prey, rest.

6. **brain/genome + brain/network** - Minimal NEAT: fixed topology network (no mutations yet), feedforward activation. Test: given inputs, network produces deterministic outputs.

7. **brain/innovation + evolution/selection** - Mutations (weight, add connection, add node). Test: mutated genomes produce valid networks.

8. **evolution/species** - Speciation, compatibility distance. Test: similar genomes cluster, different ones separate.

9. **evolution/population + evolution/reproduction** - Full generation loop: evaluate, speciate, select, reproduce. Test: fitness improves over 50 generations on a simple task (XOR or similar, not the full sim).

10. **Integration** - Wire brain to prey sensors/actions. Now prey are NEAT-controlled in the world. Test: prey survive longer over generations.

11. **signal/message + signal/propagation** - Add signaling. Test: signals propagate, decay, are received by nearby prey.

12. **evolution/kin** - Lineage tracking, kin bonus. Test: related prey cluster spatially.

13. **stats/collector + stats/metrics** - Communication analysis. Test: MI and TopSim computed correctly on synthetic data.

14. **stats/export** - CSV/JSON output. Test: files are written, parseable.

15. **viz/terminal** - ASCII visualization. Test: visual inspection.

16. **Checkpointing** - Save/load. Test: resume from checkpoint produces identical results.

Each step is roughly 2-6 hours of work for a beginner, totaling 50-100 hours for the full project.

---

## Design Decisions Summary

| Decision | Choice | Alternative Considered | Why |
|----------|--------|----------------------|-----|
| Architecture | Plain structs in Vecs | ECS (hecs, specs) | Beginner-friendly, entity count too small for ECS to matter |
| Crate layout | Single crate + features | Cargo workspace | Simpler, avoid inter-crate dependency management |
| RNG | ChaCha8Rng, single instance | StdRng, per-system RNGs | Cross-platform determinism, reproducibility |
| Config format | TOML | JSON, YAML | Rust-native, supports comments, familiar from Cargo |
| Predator AI | Scripted behavior trees | Co-evolved NEAT | Stable selection pressure, analyzable, halves complexity |
| Network type | Feedforward only | CTRNN (recurrent) | 5-10x faster, external state provides temporal context |
| Initial genome | Sparse random connectivity | Full input-output | Better initial diversity, more effective speciation |
| Reproduction | Asexual in-sim, sexual crossover between generations | Sexual reproduction in-sim | Simpler, natural kin clustering from spatial spawning |
| Kin tracking | Lineage IDs (approximate) | Full pedigree (exact) | O(1) vs O(n), biologically plausible |
| Vocabulary | Fixed size (8 symbols) | Evolving vocabulary | Stable network topology, unused symbols = evolved silence |
| Signal cost | 3.0 energy (6x movement) | Free, or 1.0 | Hamilton's rule math requires substantial cost for honest signaling |
| Error handling | Custom error enum | anyhow | Library-compatible, explicit for beginners |
| Serialization | serde + bincode (checkpoints) + CSV (stats) | All JSON | Binary checkpoints are 10x smaller, CSV is researcher-friendly |
| Frontend boundary | Observer trait + Snapshot structs | Direct world access | Decoupled, WASM-compatible, safe |
| Parallelism | Single-threaded | Rayon for per-agent compute | Determinism, debuggability, add Rayon later if slow |

---

**Sources consulted during design:**
- [NEAT-Python Documentation - Algorithm Overview](https://neat-python.readthedocs.io/en/latest/neat_overview.html)
- [Original NEAT Paper (Stanley & Miikkulainen, 2002)](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [CMU NEAT Algorithm Specification](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume21/stanley04a-html/node3.html)
- [Towards Data Science - Building NEAT from Scratch](https://towardsdatascience.com/from-genes-to-neural-networks-understanding-and-building-neat-neuro-evolution-of-augmenting-topologies-from-scratch/)
- [Speciation in Canonical NEAT (SharpNEAT)](https://sharpneat.sourceforge.io/research/speciation-canonical-neat.html)
- [Hamilton's Rule and Kin Selection (GeeksforGeeks)](https://www.geeksforgeeks.org/biology/hamiltons-rule-principle/)
- [Quantitative Test of Hamilton's Rule (PLOS Biology)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1000615)
- [Framework for Emergence and Analysis of Language in Social Learning Agents (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11366021/)
- [Learning and Communication Pressures in Neural Networks (arXiv)](https://arxiv.org/html/2403.14427v3)
- [Rust NEAT Implementations on GitHub](https://github.com/TLmaK0/rustneat)
- [Game-Theoretic Model of Predator-Prey Signaling (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0022519383710891)
- [Uumarrty: Agent-Based Predator-Prey with Game Theory (bioRxiv)](https://www.biorxiv.org/content/10.1101/2023.08.09.552686v1.full)