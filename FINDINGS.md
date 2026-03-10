# Findings: Runs 1-2 (84k and 68k generations)

Two runs analyzed: local (unknown seed, 84,139 gens) and VPS seed 42 (68,428 gens). Both at 8x defaults (pop=384, grid=56, pred=16, food=200, ticks=500). Signal cost = 0.0 (free). Evasion boost active.

## 1. Signals Are Fuel, Not Language

The evasion boost mechanic (+1 movement when receiving any signal near a predator) dominates the evolutionary dynamics. The signal channel evolved as a survival resource, not a communication medium.

Evidence:
- seed42: Pearson(signals_emitted, avg_fitness) = 0.989, survives detrending (r=0.989)
- Both runs: silence_corr is the #2 fitness predictor after signal volume
- response_fit_corr = 0 in both runs across all generations - behavioral response to signals has zero fitness coupling
- receiver_fit_corr is positive (0.36, 0.39) but is a spatial confound: Pearson(MI, receiver_fit_corr) = -0.220 in local (more information, LESS receiver fitness) and ~0 in seed42

Evolution maximized signal volume (~48k signals/gen, 30%+ of theoretical max) while minimizing emission near danger (negative silence_corr). The content is irrelevant - the presence of signal triggers the boost.

## 2. MI Is Confounded with Symbol Diversity

The local run's MI spike (0.68 at gen 37k) corresponded to a symbol transition period, not genuine encoding:

| Gen | sym0 | sym1 | sym2 | MI | Phase |
|-----|------|------|------|----|-------|
| 15k | 98% | 0% | 2% | 0.007 | sym0 monopoly |
| 35k | 66% | 0% | 34% | 0.172 | diversifying |
| 45k | 33% | 37% | 30% | 0.082 | max diversity |
| 60k | 0% | 0% | 100% | 0.008 | sym2 monopoly |

Pearson(HHI, MI) = -0.521. MI mechanically requires symbol variety - when one symbol dominates, MI collapses regardless of encoding quality. Only 1,320 gens showed monopoly + MI > 0.05, and those averaged only MI = 0.085.

## 3. The Causal Chain Almost Never Completes

For genuine communication, all three are required simultaneously:
- (A) Sender encodes context: MI > 0.05
- (B) Receiver changes behavior: jsd_pred > 0.05
- (C) Changed behavior helps fitness: response_fit_corr > 0.05

| Run | A+B+C | % of run | Interpretation |
|-----|-------|----------|---------------|
| Local | 4,527 gens | 5.4% | Scattered, transient |
| Seed42 | 29 gens | 0.0% | Essentially never |

Each component appears independently but they cannot be sustained together.

## 4. Two Radically Different Evolutionary Paths, Same Outcome

Despite identical mechanics, the runs diverged dramatically then converged to the same steady state.

| Metric | Local | Seed42 |
|--------|-------|--------|
| Brain peak | 31.6 (gen 80k, late) | 36.4 (gen 14.5k, early) |
| Final brain | 25.8 | 18.4 |
| Peak MI | 0.68 (gen 37k) | 0.11 (gen 1.4k) |
| Final MI | 0.000 | 0.000 |
| Lag direction | MI leads brain | Brain leads MI |
| Sustained fitness | 131 | 136 |
| Signals/gen | 48k | 47k |

Both converged: all metrics STABLE in the last 20k gens. Different brain sizes (25.8 vs 18.4), same fitness (~133). Seed42's smaller brain saves 0.05 energy/500 ticks - barely measurable.

## 5. Universal Silence Strategy

Both runs converge on silence near the predator: silence_corr = -0.24 (local), -0.36 (seed42). This is the most consistent emergent behavior - it appears across seeds and persists indefinitely.

## 6. The Silence Onset Effect Is Mechanical

When signals stop, prey FREEZE 80.5% of the time (negative silence_move_delta). This is not a learned information-processing response - it's the evasion boost turning off. Prey moved because the boost added +1 movement; when signals cease, their base movement rate takes over.

## 7. Vestigial Danger Symbol in Seed42

Seed42's rare sym1 (0.2% of signals) concentrates 88.1% in d0 (nearest predator distance bin), vs 27.6% for dominant sym0. This is a ghost of functional symbol differentiation - sym1 once meant "danger here" but was nearly driven extinct. Evidence of Level 3 semiotic potential, but not sustained.

## 8. Predator Saturation Undermines Information Asymmetry

With 16 predators on a 56x56 grid, 88% of the time prey have at least one predator within vision range. The 2:1 vision/signal ratio was designed to create information asymmetry (some prey see danger, others don't), but with this predator density almost everyone can see a predator. There's insufficient "safe" space for the signal channel to bridge an information gap.

## 9. Fitness Efficiency and Volatility

Both runs sustain only ~27% of theoretical maximum fitness (133/500). Fitness volatility is high (std ~40) and does not decrease over time. The fitness surface is noisy, not stabilizing.

During the local MI spike window (gen 30-55k), high-MI gens had +34 fitness over low-MI gens. But sender_fit_corr was -0.176 during high MI (senders were hurt). Classic altruism problem - population benefits but individual senders pay.

## 10. Encoding Collapse

Both runs end with zero sustained input MI across all 16 input dimensions. Signals encode nothing about any input by the end. Encoding stability is negative (Spearman: -0.53 local, -0.37 seed42 early vs late) - what signals encode keeps changing, preventing stable conventions.

---

## Diagnosis

Three structural features of the current parameter regime prevent genuine semiotic emergence:

1. **The evasion boost rewards signal presence, not signal content.** Any signal triggers the boost regardless of symbol or context. Evolution exploits this by maximizing volume.

2. **Free signals have no cost pressure.** With signal_cost = 0.0, there's no penalty for noise. The only selective pressure against meaningless signaling is indirect (noise could confuse neighbors), but the evasion boost overwhelms this.

3. **Predator saturation eliminates information asymmetry.** 16 predators on 56x56 means prey almost always see a predator. The signal channel can't bridge a gap that barely exists.

These interact: the boost makes signal presence valuable, free cost removes the penalty for noise, and predator saturation means there's no information to transmit anyway.

---

## Parameter Changes (Run 3+)

Based on findings above, three changes applied to address all three structural barriers:

### 1. Evasion boost removed

The +1 movement boost for signal reception rewarded signal presence regardless of content. Evolution exploited this by maximizing volume (~48k signals/gen) while encoding nothing. Removing it forces signals to compete on information value alone - receivers must learn useful behavioral responses to signal content, not just benefit from signal existence.

### 2. Signal cost: 0.0 -> 0.002

Free signals allowed noise to proliferate unchecked. At 0.002 per emission, a prey signaling every tick pays 1.0 energy over 500 ticks (entire starting energy). The observed ~0.25 signals/prey/tick rate would cost 0.0005/tick, roughly 60% of base metabolic drain. This creates selective pressure: signals that don't help the sender's kin (or the sender via reciprocity) are a net energy loss.

### 3. Predators: 16 -> 3, Food: 200 -> 100

16 predators on 56x56 gave 88% vision coverage - almost every prey could see a predator at any time. With 3 predators, vision coverage drops to ~33%, creating a ~55% information gap (prey within signal range but outside vision range of any predator). This is the gap the signal channel needs to bridge. Food halved to 100 to maintain resource pressure at the lower predator count.

### Expected effects

- Signal volume should drop dramatically (no boost incentive, cost penalty)
- Any signals that persist face genuine selection for content quality
- Information asymmetry creates real value for danger communication
- The altruism problem (senders pay, population benefits) may still limit sender evolution, but the absence of the boost removes the dominant exploitation pathway
