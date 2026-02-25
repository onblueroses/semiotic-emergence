WORK IN PROGRESS / LEARNING PROJECT
  public to hold myself accountable. 

# Predator-Prey Evolution & Communication

Prey agents evolve emergent alarm calls and deception under selection pressure from predators - implemented in Rust using NEAT neuroevolution.

This is an active research-style project. The simulation core runs. NEAT wiring is in progress.

---

## What It Does

A population of prey agents lives on an 80x60 grid with terrain (trees, rocks, water). Three types of predators hunt them: eagles (aerial, wide vision), wolves (pack coordination), and snakes/lions (ground, ambush). Prey that survive longer and eat more food reproduce.

Each prey has a neural network brain (36 inputs, 18 outputs). One output channel controls signal emission - prey can broadcast one of 8 symbols to neighbors within range 12. The goal is to see whether honest alarm calls, misdirection, or silence emerge as the dominant strategy, and whether that strategy has structure (specific symbols for specific threats).

The neural networks evolve via [NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf): genomes start minimal and grow topology through mutation. Speciation protects structural innovations while they develop. No hand-coded behaviors - everything the prey do comes from evolved weights.

## Architecture

```
config/           TOML-based configuration (seed, world params, NEAT rates)
src/
  world/          80x60 grid, terrain, food regrowth, entity tracking
  agent/          Prey and predator structs, sensor encoding, action resolution
  brain/          NEAT genome, network activation, innovation tracking
  signal/         Broadcast/receive, one-tick delay, distance decay
  evolution/      Speciation, fitness sharing, mutation, crossover, selection
  stats/          TopSim, mutual information, deception rate metrics
  viz/            Terminal dashboard (crossterm, feature-gated)
  simulation.rs   Generation loop, CLI entry point
```

Full design rationale: [`ARCHITECTURE.md`](ARCHITECTURE.md) (~4,000 words covering data flow, NEAT implementation details, signal economics, observability metrics, and pitfall guide).

## Status

| Component | State |
|---|---|
| World tick loop (9 phases) | Done |
| Predator AI (aerial / ground / pack behavior trees) | Done |
| Sensor encoding (36 inputs, 36-dim signal input) | Done |
| Signal propagation, decay, one-tick delay | Done |
| NEAT genome: topology mutation, crossover, innovation tracking | Done |
| Speciation, fitness sharing, stagnation protection | Done |
| Evolution loop wiring (NEAT into generation lifecycle) | In progress |
| Stats: TopSim, mutual information, deception rate | In progress |
| Terminal visualization | Stub |
| Curriculum configs (no-predator → multi-predator staging) | Done |

## Quick Start

```bash
# Build and run
cargo run --release

# Run with a specific config
cargo run --release -- --config config/stage1_evasion.toml

# Override seed and generation count
cargo run --release -- --seed 1234 --max-generations 500

# Fast iteration (debug, small world)
cargo run -- --config config/fast_test.toml
```

Requires Rust 2024 edition (`rustup update stable`).

## Metrics

The stats module tracks three signals of emergent communication:

- **Mutual information** between emitted symbol and predator type/distance - rising MI means signals carry real information about the environment
- **TopSim** (topographic similarity) - correlation between signal structure and referent structure; high TopSim indicates compositionality
- **Deception rate** - fraction of signals emitted when no predator is in range; tracks whether dishonest signaling persists or gets selected out

## Predator Types

| Type | Behavior | Counter-strategy |
|---|---|---|
| Eagle | Aerial, wide FOV (15), ignores terrain | Climb trees, coordinate warnings |
| Wolves | Pack coordination, medium vision (10) | Group defense, flee signals |
| Snake/Lion | Ground, short FOV (4), ambush | Proximity alarms, hide in rocks |

## Configuration

All parameters live in `config/default.toml`. Key levers:

```toml
[signal]
vocab_size = 8       # symbol count - affects neural input/output dimensions
signal_range = 12    # broadcast radius

[neat]
add_node_rate = 0.03        # structural complexity growth rate
compatibility_threshold = 3.0  # speciation sensitivity
stagnation_limit = 20       # generations before species is culled
```

`stage0_foraging.toml` - no predators, flat terrain, free signals (bootstraps foraging behavior)
`stage1_evasion.toml` - single ground predator (introduces basic evasion before communication)

## License

MIT
