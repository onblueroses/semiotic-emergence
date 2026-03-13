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

Three invisible circular zones (radius 8.0, ~19% grid coverage) drift randomly across the 56x56 grid. Zone speed is 0.5 (probabilistic - moves one cell ~every other tick in a random cardinal direction). Prey inside a zone lose 0.02 energy per tick from each overlapping zone. At starting energy 1.0, a zone kills in 50 ticks.

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
- Energy drain: `ZONE_DRAIN_RATE` (0.02) per tick per zone, stacks across overlapping zones
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

---

## Parameter Changes (Post Kill-Zone Runs)

Three structural issues identified from the initial kill-zone runs (seeds 42/43, ~900 gens):

### 1. Dead silence vs behavioral silence

The silence_corr metric (Pearson between signals_per_tick and min_zone_dist) showed strong negative values (-0.58 to -0.90), suggesting prey go silent near zones. However, this is a mortality artifact: zones kill in 10 ticks, dead prey emit no signals, so signal volume drops when zones are nearby because there are fewer living prey - not because survivors choose silence.

**Fix:** Normalize signals_per_tick by alive_per_tick before computing the correlation. This isolates behavioral silence (living prey choosing to signal less) from dead silence (fewer prey alive to signal).

### 2. Zone lethality too fast for communication

At 0.1 energy/tick drain, zones kill in 10 ticks. A prey at the zone boundary (radius 8.0) needs ~8 ticks to walk out even knowing the right direction. The window between receiving a useful signal and dying is 2-3 ticks - too short for a meaningful signal-response-escape loop to provide fitness advantage.

**Fix:** ZONE_DRAIN_RATE reduced from 0.1 to 0.02 (50-tick kill). Prey now have 30-40 useful ticks to receive directional signals and escape. Communication becomes structurally valuable because there's time to act on information.

### 3. Brain size collapse to minimum

Both seeds showed avg base hidden = 4.2 (floor = 4), avg signal hidden = 2.2. With 4 shared base neurons, the split-head architecture has no capacity for independent signal control. Positive iconicity at ~900 gens is likely a spandrel (energy-drop → signal output through shared neurons), not intentional alarm calling. MI near zero (0.004) confirms symbols carry no zone information.

Prior runs at every neuron cost tested (0.0002, 0.00002, 0.00001) showed the same collapse. The cost doesn't matter - what matters is that larger brains provide no fitness advantage when communication hasn't emerged yet.

**Fix:** neuron_cost set to 0.0 (free brains). Under neutral drift with 5% hidden-size mutation rate, brain sizes will explore the full range [4-64] base / [2-32] signal. This gives the split-head architecture actual capacity, and if even a small fitness gradient emerges for useful signaling, evolution can exploit existing capacity instead of needing to build it from scratch.

### Expected effects

- Fitness should increase (longer survival in zones, no brain metabolic cost)
- Signal volume may change unpredictably (longer-lived prey emit more; but no metabolic incentive to signal)
- silence_corr values will change magnitude (normalized metric, different denominator)
- Brain sizes should drift above minimum over hundreds of generations
- MI may rise if brains develop enough capacity for zone-correlated signaling
- The critical test: does response_fit_corr ever leave zero? That's the real signal of communication.

---

## Kill Zone Runs: First Long-Duration Analysis

**Seeds: 42 (36,084 gens), 43 (37,441 gens), 99 (93,000 gens)**
Parameters: pop=384, grid=56, zones=3, radius=8.0, speed=0.5, food=100, ticks=500, signal_cost=0.002, kin_bonus=0.10, neuron_cost=0.0, 6 symbols, split-head brain.
seed99 run with `--metrics-interval 1000` (94 data points at ~1000-gen intervals); seed42/43 at default (one row per gen, full resolution).

### Universal Findings (all three seeds independently confirmed)

**1. Signal hidden layer converges to maximum.**

All three seeds independently evolved to near-maximum signal processing capacity: seed42 avg 29.0 [26-32], seed43 avg 30.8 [26-32], seed99 avg 25.3 [19-31]. Starting from random initialization around 6, all three sprinted to high signal_hidden by gen 18-25k. With neuron_cost=0.0, this is not energetically forced - it reflects genuine selection pressure for signal processing capacity.

**2. Zone encoding is zero, universally.**

`mi_zone_dist = 0.000` in all three seeds across the entire run. Zones create lethal pressure but never appear in signal content. Prey that survive zones do so through other means; no stable convention of danger signaling emerges.

**3. response_fit_corr = 0.000, universally.**

No individual prey benefits from changing its behavior based on signal content. The three-way coupling chain (encode → respond → survive) never closes. Signals influence the environment but not through content-sensitive behavioral responses.

**4. receiver_fit_corr ~0.79-0.87 in all runs.**

High but confirmed to be a spatial confound: center prey hear more signals AND survive more (better food access, further from zone edges on average). This metric has been consistent since run 1 with visible predators and does not indicate signal utility.

**5. Sender selection is real but moderate.**

sender_fit_corr = 0.36-0.46 across all runs. Signaling propensity correlates positively with fitness, likely through cooperative patch harvesting: prey that signal more are more active and more likely to be co-located with other active prey, satisfying the 2+ prey requirement for patch food.

### Divergent Attractors: Same Pressure, Three Solutions

Despite identical mechanics, the three seeds found completely different stable symbol systems:

| Seed | Final symbol distribution | Primary encoding |
|------|--------------------------|-----------------|
| seed42 | s1=33%, s3=32%, s4=25% (3-way split) | Food location (mi_food_dy=0.119, mi_food_dist=0.107) |
| seed43 | s3=78%, s5=19% (near monopoly + satellite) | Other signals (mi_sig5_str=0.072, mi_sig5_dx=0.020) |
| seed99 | s2=28%, s5=25%, s0=20%, s4=16% (4-way spread) | Food location (mi_food_dist=0.114, mi_food_dx=0.100) |

Symbol 0 is extinct in both VPS seeds; symbol 3 is extinct in seed99. The specific symbols that survive are arbitrary (index has no semantic prior), but the convergence on food encoding is not.

### The seed42/seed99 Pattern: Direct Food Encoding

Both seed42 (at gen 36k) and seed99 (at gen 93k) converge on the same solution: signals encode food location. Top sustained input MI in both runs has food_dy, food_dist, and food_dx as the dominant dimensions. Multiple symbols (3-4 active) all carry food proximity information with similar zone-proximity ratios (~1.4-1.5x), indicating this is a spandrel - the elevation near zones is because active/stressed prey are more prevalent there, not because prey are signaling about zones.

seed99's encoding is cleaner at 93k gens (food_dist=0.114) than seed42's at 36k gens (food_dist=0.107), suggesting the food encoding strategy consolidates further given more time. The VPS runs are mid-consolidation.

### The seed43 Anomaly: A Two-Tier Signal Relay

seed43 found a structurally distinct solution. Symbol 3 dominates (78%) and its top inputs are `mi_sig5_str=0.072` and `mi_sig5_dx=0.020` - the dominant symbol encodes the strength and direction of the minority symbol s5 (19%). Meanwhile s5 encodes food location (mi_food_dx and mi_food_dy appear in its encoding profile). The resulting chain:

**s5 → food proximity → s3 → where s5 activity is concentrated → transitively, where active prey near food are**

This is a second-order relay: prey emit s3 in response to hearing s5, re-broadcasting where s5 activity clusters. Functionally it achieves the same thing as direct food encoding but through an indirect route. Evidence that seed43 needed more signal processing capacity to sustain this: it has the highest signal_hidden (30.8 avg vs 29.0/25.3) and the lowest sustained fitness (620 vs 695/736) - the relay is less efficient than direct encoding.

seed43 also shows the most volatile evolutionary history: early vs late encoding stability Spearman = 0.188 (vs 0.461/0.461 for seed42/seed99), and the largest JSD spike cluster (gen 33k, values up to 0.375) - a late-run symbol reshuffling event not seen in the other seeds.

### Evolutionary Trajectory: Five Acts (seed99 detail)

seed99's 93k gens at sparse resolution reveals a consistent five-phase pattern likely present but compressed in the shorter runs:

1. **Gen 0-5k: Chaotic sweeps.** Signal hidden oscillates wildly (+16, -12 in single 1k-gen windows). One symbol briefly dominates then is permanently eliminated via selective sweep (symbol 3 hits 61%, then goes extinct and never recovers).

2. **Gen 15-25k: Low-complexity monopoly.** One symbol owns 72-79% of signals while signal_hidden stays small (2-6). Minimal processing, no real encoding. The cheapest viable signaling strategy.

3. **Gen 30-35k: The breakthrough.** Signal hidden explodes (+11.9 in a single window). A new genotype sweeps through with high signal processing capacity. Symbol reshuffling accompanies the sweep.

4. **Gen 35-55k: The turbulent semiotic window.** The closest the population gets to danger signaling: MI peaks at 0.006, silence_corr hits -0.231 (prey going quiet near zones), jsd_pred peaks at 0.311. All three couple briefly at gen ~45k. The coherence dissolves immediately - another sweep crashes signal_hidden from 17 to 10 in one step.

5. **Gen 55-93k: Food encoding consolidation.** Signal_hidden climbs steadily to 25-31. Food MI grows 0.15→0.34. Zone MI stays permanently at 0.000. The danger-signaling niche is abandoned; food coordination becomes the stable attractor.

### The Danger Signaling Problem

The turbulent semiotic window (Act 4) appears in all three seeds during the initial post-breakthrough phase. It is always transient. Several factors work against stable danger convention formation:

- **Silence near zones conflicts with food encoding near zones.** Prey are more active (and thus signal more) when near food, which correlates spatially with zones. A stable zone-signaling convention would require prey to signal *less* near food patches that happen to be near zones - evolutionarily unstable given the food encoding pressure.

- **The causal chain timing.** Zones kill in 50 ticks. A prey already inside a zone has limited time to receive signals, update behavior, and escape. For danger signaling to close the response_fit_corr gap, the signal-response loop needs to work faster than the simulation dynamics allow.

- **Free riding on food encoding.** Once food encoding is established, the signal channel is already occupied. Inserting zone information requires differentiated symbols - which appears in seed43's two-tier relay but at a significant fitness cost.

### Cross-Run Summary Table

| Metric | seed42 (36k) | seed43 (37k) | seed99 (93k) |
|--------|-------------|-------------|-------------|
| avg_fitness | 695.3 | 620.0 | 736.2 |
| sig_hidden | 29.0 | **30.8** | 25.3 |
| mutual_info (final) | 0.0001 | 0.0003 | 0.0001 |
| jsd_pred | 0.215 | 0.155 | 0.018 |
| silence_corr | +0.046 | -0.018 | -0.012 |
| sender_fit_corr | 0.360 | 0.419 | 0.464 |
| receiver_fit_corr | 0.826 | 0.873 | 0.789 |
| response_fit_corr | **0.000** | **0.000** | **0.000** |
| Zone MI | 0.000 | 0.000 | 0.000 |
| Top encoding | food location | signal relay | food location |
| Encoding stability | 0.461 | 0.188 | ~0.46 |

### What This Means for the Design

The kill-zone regime successfully replaced the evasion-boost exploit (signals as fuel) with signals that carry real information content. Food encoding with MI values of 0.09-0.12 is meaningful - these are genuine sender-world correlations. The architecture is working: signal_hidden at max, stable multi-symbol vocabularies, food information in the channel.

The open question is whether danger signaling is achievable in this regime or whether food encoding always outcompetes it. The turbulent semiotic windows suggest the population briefly tries danger signaling before abandoning it. Three possible interpretations:

1. **Insufficient selection pressure**: danger zones cover ~19% of the grid and kill in 50 ticks. The fitness advantage from danger signaling may be too small relative to food coordination gains to sustain.

2. **Convention fragility**: danger conventions require correlated behavior across many prey simultaneously. Random drift dissolves them faster than selection can reinforce them, as seen in run 3 with visible predators.

3. **Ecological niche exclusion**: food encoding and danger encoding compete for the same symbol vocabulary. Once food encoding occupies 4-5 symbols (seed42/99) or a two-tier relay occupies them (seed43), there's no room for stable danger conventions.

Distinguishing these requires a counterfactual: a run with cooperative food patches disabled (`--patch-ratio 0.0`) to remove the food coordination driver. If danger signaling emerges when food encoding pressure is absent, that confirms ecological niche exclusion.

---

## Overnight Counterfactual Experiment (2026-03-12/13)

Three runs from the same binary, same parameters except the signal channel. The first
controlled counterfactual in the project's history.

| Run | Seed | Signals | Threads | Gens reached |
|-----|------|---------|---------|-------------|
| baseline-s100 | 100 | enabled | 4 | 96,570 |
| mute-s100 | 100 | --no-signals | 4 | 148,970 |
| baseline-s101 | 101 | enabled | 4 | 95,270 |

Parameters: pop=384, grid=56, zones=3, radius=8.0, speed=0.5, food=100, ticks=500,
signal_cost=0.002, kin_bonus=0.10, neuron_cost=0.0. Binary commit 29c5f98.

### Critical parameter discovery

The binary ran with `ZONE_DRAIN_RATE = 0.10` (10-tick kill). The previous runs
(seeds 42/43/99 above, fitness 620-736) used 0.02 (50-tick kill). The fix documented
in "Parameter Changes (Post Kill-Zone Runs)" was either reverted during performance
optimization commits or never committed to main.

At 0.10 drain, 71% of prey (272/384) die to zones per generation. A prey at zone edge
needs ~8 ticks to walk out; with a 10-tick kill window, the communication window is
2-3 ticks. Too short for signal-response-escape loops to provide fitness advantage.

This does not invalidate the experiment - it tells us precisely that at high zone
lethality, signals cannot pay for themselves.

### 1. Signals are net negative (-25.5%)

| Metric | baseline-s100 | baseline-s101 | mute-s100 |
|--------|--------------|--------------|-----------|
| Sustained avg fitness | 252.5 | 256.9 | 306.2 |
| Final avg fitness | 239.7 | 279.9 | 334.7 |
| Peak avg fitness | 360.1 | 378.4 | 413.0 |
| Zone deaths/gen | 272 | 223 | 251 |

Counterfactual signal value integral: -3,936,072. The fitness gap is stable across
96k generations - not converging, not diverging. Mute found a higher plateau early
and stayed there. Level 1 of the evidence hierarchy is answered for this parameter
regime: signals do not have adaptive value at 0.10 zone drain.

### 2. Brain architecture parasitism

| Run | base_hidden | signal_hidden | total |
|-----|-------------|---------------|-------|
| baseline-s100 | 14.2 [10-16] | 13.8 [10-17] | 28.0 |
| baseline-s101 | 4.5 [4-7] | 21.7 [15-29] | 26.2 |
| mute-s100 | 6.6 [5-8] | 13.8 [8-18] | 20.4 |

The signal environment inflates base_hidden by 115% (14.2 vs mute's 6.6). With 18 of
36 brain inputs being signal channels, evolution selects for brains that process signal
noise even when that processing provides zero survival benefit.

baseline-s101 found a partial escape: collapse base_hidden to near-minimum (4.5), shunt
everything to signal_hidden (21.7). Marginally higher fitness (257 vs 253) but still
loses to mute (306). Mute's signal_hidden drifts randomly under zero selection pressure.

### 3. Signals encode memory, not world state

Previous runs at 0.02 drain encoded food location (mi_food_dist=0.107-0.114). The
overnight runs' top input MI dimensions are memory cells (mi_mem1-7, values 0.003-0.018)
and incoming signal strength (self-referential relay). Food and energy MI are minimal.
Zone MI is zero. An order of magnitude less information than previous runs.

### 4. Sender fitness correlation flipped negative

sender_fit_corr: -0.443 (s100), -0.558 (s101). Previously +0.36 to +0.46. At high
zone lethality, signaling is pure cost with no compensating benefit. The kin bonus
(0.10) is insufficient when information has no time to be useful.

### 5. Symbol monopoly returns

baseline-s100: symbol 4 at 97.9% (HHI 0.959). baseline-s101: symbol 3 at 79.7% with
symbol 2 satellite at 16% (HHI 0.663). Different dominant symbols (arbitrary convention).
Monopoly mechanically kills MI. Previous runs at 0.02 drain maintained 3-4 active symbols.

### 6. response_fit_corr remains zero

Still 0.000 across all runs. The three-way causal chain (encode -> respond -> survive)
has never closed in the project's history. Receivers change behavior (jsd_pred 0.19-0.23)
but that change does not predict survival. The in-zone/out-of-zone JSD ratio is only
1.14x (down from ~6x in early 50k salvaged data).

### 7. Reproducibility: fitness converges, everything else diverges

Sustained fitness differs by 1.7% between s100 and s101 (252.5 vs 256.9). base_hidden
differs by 3.2x, signal entropy by 50x, dominant symbol is different, encoding profile
is different. Fitness is constrained by world physics; brain architecture and symbol
system are contingent, path-dependent convention - but convention without function.

### 8. Computational cost of the signal channel

Mute reached 149k gens vs baseline's 96k in the same wall time (55% faster). The signal
channel consumes ~35% of total computation via the `receive_detailed()` inner loop
(O(alive_prey * active_signals)). Zero signals means trivial reception.

### 9. Epoch oscillations around a phase transition

baseline-s100 shows boom-bust cycles: signal_hidden grows, jsd_pred peaks (0.41-0.47),
then signal_hidden crashes because the response doesn't improve survival. MI leads
base_hidden by ~70 gens (r=0.502) - brief information episodes create selection for
brain capacity, but the capacity outlasts the information. The system orbits a phase
transition boundary without crossing it.

### What this means

The overnight experiment establishes a clear negative result at 0.10 drain: signals
cannot provide adaptive value when zones kill too fast for communication to help. The
next experiment should rerun the same counterfactual at 0.02 drain (the parameter used
for the successful food-encoding runs above) to test whether signals have adaptive value
when prey have time to act on information.

Combined with the 0.10-drain data, this would produce a 2x2 matrix (lethality x signals)
isolating the interaction between zone lethality and signal value.

Detailed analysis with charts: `findings/2026-03-13-overnight-analysis.md`
