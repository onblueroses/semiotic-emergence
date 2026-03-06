# predator-prey

I'm learning about genetic algorithms and emergent communication. This is a minimal simulation to test whether prey can evolve to use signals when hunted by a predator.

## What it does

- 20 prey with neural network brains live on a 20x20 grid
- A predator chases the nearest prey and kills on contact
- Prey can move, eat food, and broadcast one of 3 signals
- Signals decay with distance (range 8 cells) and have a 1-tick delay
- A genetic algorithm evolves the prey's neural network weights over generations
- Fitness = ticks survived + food eaten * 10

The question: do the prey evolve to use signals in a meaningful way?

## Run it

```bash
cargo run                # 200 generations, seed 42
cargo run -- 123 500     # seed 123, 500 generations
cargo run --release -- 42 1000  # faster
```

Output goes to `output.csv` with columns: `generation, avg_fitness, max_fitness, signals_emitted`.

## Architecture

~625 lines across 5 files:

- `brain.rs` - Fixed-topology neural network (10 inputs, 6 hidden tanh, 8 outputs). 122 weights per genome.
- `evolution.rs` - Tournament selection (size 3), single-point crossover, Gaussian mutation. Top 4 elites preserved.
- `world.rs` - Grid simulation. Prey sense predator, food, allies, and incoming signals. Predator uses simple chase-nearest logic.
- `signal.rs` - 3-symbol broadcast with linear distance decay and 1-tick delay.
- `main.rs` - Generation loop: evaluate groups of 4 prey, rank, evolve, repeat.

## What to look for

In `output.csv`:
- **avg_fitness increasing** means the population is learning to survive
- **signals_emitted > 0** means prey are using the signal outputs
- Whether signals correlate with predator proximity is the interesting part (not yet measured, future work)

## What's next

- Measure signal-predator correlation (are signals honest warnings?)
- Add mutual information metric between signals and predator distance
- Try multiple predators
- Visualize a generation in the terminal
