# semiotic-emergence

Forty-eight tiny neural networks live on a 20x20 grid. A predator hunts them. They can broadcast one of three symbols to each other. At generation zero, those symbols mean nothing.

I started this project because I wanted to watch meaning come into existence. We have theories for how meaning works once it already exists, but the moment before - when a world contains no signs, no reference, no communication, and then for the first time it does - nobody really has a good account of what that transition looks like. It happened deep in evolutionary history and left no fossil record.

So I built the simplest world I could think of. Prey, predator, food, three words, and evolution. The prey have 158-weight neural networks for brains. They can see about 4 cells around them. They can signal up to 8. The predator is three times faster than they are. The only reliable survival strategy is to cluster near the predator until it gets confused and stumbles randomly, which means coordinating, which in theory means communicating.

Whether they actually learn to talk to each other is still an open question. So far they've learned to shut up instead. They go silent when danger is near. Which might actually be the more interesting finding - silence as the first sign, the simplest possible meaning, requiring no coordination between sender and receiver about what any particular symbol means. Just the absence of background noise becoming informative.

[FRAMEWORK.md](FRAMEWORK.md) has the full intellectual framework behind the project. It governs all development decisions.

## Run it

```bash
cargo run --release -- [seed] [generations]       # normal run
cargo run --release -- 42 300                     # seed 42, 300 gens
cargo run --release -- 42 300 --no-signals        # counterfactual (signals suppressed)
cargo run --release -- --batch 10 300             # cross-population divergence
```

Output goes to `output.csv` and `trajectory.csv`. Batch mode also writes `divergence.csv`.

## What we can see

The CSV tracks per generation: fitness, signal count, mutual information (does symbol choice correlate with predator distance?), iconicity (are signals concentrated near the predator?), confusion ticks (how often prey successfully group up to confuse the predator), receiver JSD (does hearing a signal change what a prey does?), and silence correlation (do signals drop when the predator is close?).

The receiver-side instruments are the recent addition. We now have five instruments covering both halves of the communication channel:

1. **Receiver response spectrum** - JSD between action distributions with vs without signal. By gen 40, JSD > 0.2 - receivers genuinely change behavior in response to signals.
2. **Silence detection** - Pearson correlation between signal rate and predator proximity. Negative values confirm the temporal pattern: prey go quiet near danger.
3. **Semiotic trajectory** - How the signal-meaning mapping evolves generation by generation. Spikes in trajectory JSD mark phase transitions where meaning reorganizes.
4. **Cross-population divergence** - Do different seeds develop different conventions? Permutation-aware comparison accounts for arbitrary symbol labeling.
5. **Counterfactual value** - Run with `--no-signals` to suppress emission. Same seed, different fitness curves - the delta is the signal channel's contribution.

## The code

About 1200 lines of Rust across six files:

```
src/brain.rs      - Neural network (16 inputs, 6 hidden, 8 outputs)
src/evolution.rs  - Genetic algorithm (tournament, crossover, Gaussian mutation)
src/world.rs      - The grid, the physics, the predator, the food
src/signal.rs     - Three symbols, linear decay, short range, one-tick delay
src/metrics.rs    - All five FRAMEWORK instruments (MI, JSD, silence, trajectory, divergence)
src/main.rs       - Generation loop, batch mode, counterfactual mode, CSV output
```

## How the world works

The grid is toroidal - no edges, no corners. 48 prey get evaluated in groups of 8. The predator moves 3 cells per tick, prey move 1. It kills only on the same cell, but gets confused and stumbles randomly when 3 or more prey are within 4 cells of it. Prey vision is 4 cells, signals travel 8. That 2:1 gap between sight and signal range is the whole reason communication could matter here: signals carry information further than eyes can see.

Two of five evaluation rounds group prey by genetic similarity. This is how the bootstrap problem might get solved. You can't have senders without receivers or receivers without senders, but genetically similar individuals are more likely to share complementary wiring for both. Kin rounds give them a chance to discover that.

## How to think about changes

Every change should help us see what's actually happening, not push the simulation toward outcomes we expect. We went in looking for alarm calls and found meaningful silence instead. The framework in [FRAMEWORK.md](FRAMEWORK.md) has the full account, but the short version: measure before you optimize, look at receivers not just senders, and design for surprise.
