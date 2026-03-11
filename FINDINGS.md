# Findings

## Runs 1-2 (84k and 68k generations)

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

---

## Run 3 (100k generations, seed 42)

Parameters: pop=384, grid=56, pred=3, food=100, ticks=500. Evasion boost removed, signal cost 0.002, single hidden layer (4-124 neurons), 3 symbols.

Full analysis in [PERFORMANCE.md](PERFORMANCE.md). Key findings:

### 1. Brain size is the rate-limiting factor for semiotic emergence

Small brains (~6-10 neurons): MI peaks at 0.05. Large brains (~15 neurons): MI reaches 0.624. The hidden layer capacity determines whether evolution can isolate signaling from fleeing behavior. Brain size naturally evolved to the MAX_HIDDEN=16 ceiling, suggesting the constraint was artificial.

### 2. Brain expansion destroys then rebuilds semiotic structure

At gen 46k-50k, avg_hidden exploded from 10 to 15. Total semiotic collapse followed - MI, iconicity, sender_fit_corr all dropped to zero. The old 6-neuron signal strategy didn't transfer to 15 neurons. It took ~25k generations to rebuild, but the rebuilt system was far stronger (MI sustained above 0.2 for 6,187 consecutive generations).

### 3. Genuine encoding emerged during the MI surge (gen 75k-100k)

Input MI analysis showed signals primarily encoded predator information: predator distance (MI 0.472), predator dy (0.267), predator dx (0.199). Signal-0-strength MI of 0.179 suggested signal relaying - prey re-emitting received signals to extend warning range.

### 4. The coupling problem

With a single hidden layer, movement and signal outputs share all weights. Every movement adaptation changes signal behavior and vice versa. This creates:
- Spandrels (signal-context correlations from movement weight sharing, not communication intent)
- Fragility (movement optimization can destroy signal conventions)
- The convention collapse at gen 46k-50k was likely caused by brain expansion disrupting coupled weight configurations

### 5. Convention instability

MI peaked at 0.669 but the convention collapsed due to neutral drift - fitness barely changed whether prey communicated or not. The signal channel didn't matter enough to defend itself against genetic drift. This motivated the v2 architecture changes.

---

## Architecture v2

Five changes address the structural barriers identified in runs 1-3:

### 1. Split-head brain architecture

Single hidden layer replaced with base hidden (4-64, shared) + signal hidden (2-32, dedicated). Signal outputs now pass through their own hidden layer, giving evolution capacity for independent signal control. Two separate hidden size genes evolve independently. This directly addresses the coupling problem from run 3.

### 2. Six symbols (was 3)

More signal vocabulary enables richer encoding (food, zone proximity, direction) instead of just 3 coarse states. Harder for one symbol to monopolize - driving MI to zero by dominating 100% of 6 symbols is harder than 3.

### 3. Recurrent memory (8 cells)

Each prey has 8 memory cells updated via EMA (0.9 * old + 0.1 * tanh(output)). Memory is input to the brain, creating a recurrent loop. Enables temporal reasoning - prey can track signal patterns across ticks.

### 4. Cooperative food patches

50% of food requires 2+ prey within Chebyshev distance 2 to consume. Creates direct fitness incentive for spatial coordination, which signals can facilitate.

### 5. Kin fitness

Siblings (shared parent) get 0.5 bonus, cousins (shared grandparent) get 0.25 bonus added to selection fitness. Lineage tracked via parent and grandparent population indices. Supports altruistic signaling by making it individually advantageous to help relatives.

### Supporting changes

- Vision range halved (11.2 -> 5.6 at grid=56), signal range unchanged (22.4). 4:1 ratio forces heavy signal reliance.
- Neuron cost halved (0.00002 -> 0.00001). Allows complex signal processing without metabolic collapse.
- Predator speed reduced (round(1.5*scale) -> round(scale)). Gave prey more time to respond to warnings (predators later replaced by kill zones).
- Softmax emission replaces threshold-based. Emit if max(softmax) > 1/6.

### Early observations (1000 gen smoke test, with visible predators)

Brain compression: base hidden shrinks from 12 to ~4.4, signal hidden stays ~5-6. Evolution finds minimal base processing sufficient but retains signal capacity - the split architecture is working as intended. JSD rising steadily, MI climbing. Silence correlation consistently negative (-0.37 to -0.51).

---

## Kill Zones (current)

Visible predators replaced with invisible kill zones. This is the most significant architectural change since the split-head brain - it changes what communication is *for*.

### The problem with visible predators

Across all prior runs with visible predators, prey could see danger directly (brain inputs 0-2 encoded nearest predator dx/dy/distance). Communication was optional - prey could flee on their own visual information. The signal channel never became structurally necessary. MI correlated negatively with fitness across 5 seeds: communication was actively harmful. Prey that signaled paid the metabolic cost (0.002/emission) while giving away their position for no compensating survival advantage.

The fundamental issue: when prey can see the threat, signals are redundant. Evolution found that shutting up and running was strictly better than warning neighbors.

### The kill zone design

Three invisible circular zones (radius 8.0, ~19% grid coverage) drift randomly across the 56x56 grid. Zone speed is 0.5 (probabilistic - moves one cell ~every other tick in a random cardinal direction). Prey inside a zone lose 0.1 energy per tick from each overlapping zone. At starting energy 1.0, a zone kills in 10 ticks.

The critical change: zones are invisible. Brain inputs 0-2 are always zero (dead inputs, preserved for genome layout compatibility). Prey cannot see zones. The only self-signal of danger is energy loss (brain input 35) - but energy drops don't tell you *which direction* to flee.

This creates structural information asymmetry:
- A prey inside a zone knows only that energy is dropping, not where the zone boundary is
- Random fleeing has ~50% chance of going deeper into the zone
- Signals from nearby prey carry dx/dy directional information - the only source of escape direction
- A prey outside the zone can signal toward it, providing information the endangered prey cannot obtain alone

Communication is no longer optional. It's the difference between directed escape and a coin flip.

### Implementation details

- `KillZone { x: f32, y: f32, radius: f32, speed: f32 }` - f32 position for sub-cell precision
- Movement: probabilistic random walk, 1 cell per move, `speed` = probability of moving each tick
- Energy drain: `ZONE_DRAIN_RATE` (0.1) per tick per zone, stacks across overlapping zones
- Observer metrics use actual zone distance (prey can't see zones, but we measure signal-zone correlation)
- MI distance bins: `[zone_radius, signal_range, signal_range * 1.375]` - zone radius replaces vision range as the "close danger" boundary
- Receiver context is binary: in_zone vs not_in_zone (replaces predator_visible vs not_visible)
- `--pred N` CLI flag repurposed for zone count; `--zone-radius F` and `--zone-speed F` added
- signal.rs, brain.rs, evolution.rs unchanged - the change is purely in the world model

### Early observations (100 gen smoke test)

Promising results from the initial smoke test after implementation:

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Iconicity | positive | Prey signal more inside zones (alarm calling, not silence) |
| jsd_pred / jsd_no_pred | 6:1 | Receivers respond 6x more strongly inside zones |
| receiver_fit_corr | 0.76 | Strong positive correlation between hearing signals and surviving |
| Fitness | ~225 | Healthy population despite invisible danger |

The iconicity flip is the most striking change. With visible predators, iconicity was consistently negative (prey went silent near danger). With invisible zones, iconicity is positive from the start - prey signal *more* when in danger. This makes evolutionary sense: if you can't see the threat, broadcasting your distress is the only way to elicit help from those who might provide directional information.

Long-duration runs (open-ended, seeds 42 and 43) are accumulating data on the VPS to determine whether these early signals develop into sustained communication systems or collapse like prior runs.
