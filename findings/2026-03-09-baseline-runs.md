# Baseline Run Findings - 2026-03-09

Seed 42, default parameters (48 prey, 2 predators, 6 hidden neurons, 3 symbols).
Runs at 50, 200, 3000, and 10000 generations to characterize the system's behavior across timescales.

## Data Files

| File | Description |
|------|-------------|
| `findings/data/seed42-10k-output.csv` | Per-generation metrics, 10000 gens (15 cols) |
| `findings/data/seed42-10k-console.log` | Console output with sampled gen snapshots |
| `findings/data/seed42-3k-console.log` | Console output, 3000 gen run |
| `findings/data/seed42-10k-trajectory.csv` | Signal-context matrix evolution, 10000 gens (20 cols) |
| `findings/data/seed42-10k-input-mi.csv` | Per-input MI values, 10000 gens (17 cols) |

## Key Parameters

```
Grid:        20x20 toroidal
Population:  48 prey, 2 predators
Brain:       16 inputs -> 6 hidden (tanh) -> 8 outputs, 158 weights
Symbols:     3 (u8 0-2)
Vision:      4.0 cells
Signal range: 8.0 cells (2x vision)
Ticks/gen:   500
Elites:      8
Mutation:    Gaussian sigma=0.1
```

## Finding 1: Silence-as-sign is immediate and permanent

Silence correlation (`sil`) is negative from gen 0 onward across all seeds and all timescales. Range: -0.01 to -0.59. Prey suppress signal emission near predators.

**Caveat:** The correlation is already -0.37 at gen 0 before any evolution. This is likely an architectural spandrel - random weights in a 6-neuron hidden layer that's shared between movement and signal outputs create incidental predator-input-to-signal-suppression pathways. The correlation does not strengthen over 10k gens, which means evolution maintains but doesn't amplify it.

## Finding 2: Symbol 0 goes extinct

From gen ~700 onward, symbol 0 is never emitted (sym[0] JSD = 0.000 for 9300 consecutive generations). The population self-compressed from 3 symbols to 2 without any selection pressure for vocabulary size. The 6-neuron hidden layer apparently can't maintain three differentiated signal outputs.

## Finding 3: Five distinct semiotic epochs in 10k gens

The system cycles through high-activity states separated by long drift periods.

| Epoch | Gens | Duration | Peak MI | Notable |
|-------|------|----------|---------|---------|
| 1 | 600-900 | ~300 | 0.020 | JSD hits 0.693 (theoretical max) - peak receiver differentiation |
| 2 | 1670-1900 | ~230 | 0.130 | Sender-side coherence |
| 3 | 2300-2460 | ~160 | 0.213 | First high-MI epoch |
| 4 | 3100-3270 | ~170 | 0.185 | Repeat of epoch 3 pattern |
| 5 | 8890-9350 | ~460 | **0.237** | Longest, highest MI, positive iconicity |

Gap between epoch 4 and 5: ~5500 generations of drift with only brief MI spikes.

### Epoch 1 detail (gen 600-900)

Receiver-side dominated. `jsd_no_pred` reaches 0.693 (maximum JSD under natural log) at gens 830, 840, 880, 920. At gen 890, all three per-symbol JSD values simultaneously hit 0.693 while both `jsd_no_pred` and `jsd_pred` are at maximum. Receivers respond maximally differently to each symbol. This is the peak receiver differentiation moment in the entire run. Then it collapses.

### Epoch 5 detail (gen 8890-9350)

The strongest epoch by multiple measures:
- MI sustained above 0.1 for ~460 gens (3x longer than any prior epoch)
- Peak MI = 0.237 at gen 9330, all-time high
- High fitness: gen 9290 avg 40.6/max 117, gen 9320 avg 41.9/max 116
- **Positive iconicity** appears for the first time: gen 9290 (+0.027), gen 9370 (+0.039), gen 9400 (+0.088)

Positive iconicity means prey are signaling MORE near predators - alarm calling rather than silence. This is a qualitative departure from the silence-as-sign strategy. It appears only in epoch 5, after 9000 generations.

## Finding 4: No stable attractor

The system never reaches a stable high-communication state. Every epoch collapses back to low MI. Gen 9999 ends at MI = 0.005, mid-drift.

This suggests the 6-neuron hidden layer is the binding constraint. It can find weight configurations supporting communication but can't maintain them against evolutionary drift. The network must simultaneously serve:
1. Movement/survival behavior
2. Silence-near-predator suppression
3. Differentiated symbol emission

With 6 neurons and 158 weights, these demands compete.

## Finding 5: Receiver infrastructure matures independently

`jsd_pred` (receiver response when predator is visible) is consistently elevated (0.2-0.6) from gen ~4000 onward, even between sender-side MI epochs. This suggests receivers develop sensitivity to signal presence/absence that persists even when sender behavior drifts. The receiver side is more robust than the sender side.

## Finding 6: Signal count declines over time

Early gens: 700-1500 signals/gen. Late gens: 500-900. Fewer signals emitted but potentially more informative. Consistent with the silence strategy consolidating and signal cost creating pressure against noise.

## Finding 7: Fitness peaks during semiotic epochs (weak correlation)

High avg fitness episodes cluster near high-MI epochs:
- Gen 5990: avg 47.2 (highest avg in run)
- Gen 9320: avg 41.9 (during epoch 5)
- Gen 9290: avg 40.6 (during epoch 5)

But the highest individual fitness (max 149 at gen 6870) occurs during a low-MI period. The correlation between communication and fitness is suggestive but not conclusive.

## Finding 8: Counterfactual shows no clear signal benefit at 50 gens

Seed 42, gen 49:
- With signals: avg 24.4, max 67
- Without signals (--no-signals): avg 25.9, max 69

Fitness is comparable or slightly worse with signals at short timescales. The signal cost (0.01/emission) may offset any communication benefit before the system evolves effective signaling.

## Finding 9: response_fit_corr and silence onset metrics are data-starved

`response_fit_corr` produces non-zero values in only 1 out of 200 generations (seed 7 run). `silence_onset_jsd` and `silence_move_delta` are similarly sparse. The 30-sample-per-bucket minimum threshold is too high for the data density of 48 agents over 500 ticks.

## Open Questions

1. **Does HIDDEN=12 stabilize semiotic states?** The cycling/collapse pattern may be an architectural constraint. If 12 hidden neurons allow the system to maintain communication alongside movement, epochs should be longer or permanent.

2. **Is the silence correlation a spandrel or adaptive?** It's present at gen 0 and doesn't strengthen. A controlled experiment: evolve with signal outputs decoupled from movement outputs (separate hidden layers) and see if silence correlation still emerges.

3. **Why the 5500-gen gap?** Between epoch 4 (gen 3270) and epoch 5 (gen 8890), the system drifts for 5500 generations. What does the weight-space trajectory look like during this period?

4. **Is positive iconicity (alarm calling) a higher semiotic level?** It only appears in epoch 5. If it requires more evolutionary history to emerge, it may represent a more complex communication strategy than silence.

5. **Evolvable brain size:** Would a heritable hidden_size parameter (4-16 range) with energy cost proportional to neuron count produce co-evolution of brain complexity and communication?
