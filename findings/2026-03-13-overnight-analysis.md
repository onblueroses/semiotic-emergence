# Overnight Experiment Analysis: Do Signals Have Adaptive Value?

Date: 2026-03-13
Runs: baseline-s100 (96k gens), mute-s100 (149k gens), baseline-s101 (95k gens)
Parameters: pop=384, grid=56, zones=3, radius=8.0, speed=0.5, food=100, ticks=500, signal_cost=0.002, kin_bonus=0.10, neuron_cost=0.0, ZONE_DRAIN_RATE=0.10
Binary: commit 29c5f98, started 2026-03-12T23:37Z, still running

## The Headline

**Signals are net negative.** The mute population outperforms both baselines by 20-25%.

| Metric | baseline-s100 | baseline-s101 | mute-s100 |
|--------|--------------|--------------|-----------|
| Sustained avg fitness | 252.5 | 256.9 | 306.2 |
| Final avg fitness | 239.7 | 279.9 | 334.7 |
| Peak avg fitness | 360.1 | 378.4 | 413.0 |
| Zone deaths/gen | 272 | 223 | 251 |
| Generations reached | 96k | 95k | 149k |

Counterfactual signal value: **-25.5%** (integral: -3,936,072). The fitness gap is stable
across the entire run - not converging, not diverging. Mute found a higher plateau early
and stayed there.

This is the "signals are net costly" outcome from the experiment plan's decision tree.
Level 1 of the evidence hierarchy is answered: signals do not have adaptive value at
these parameters.

## Critical Discovery: Wrong Zone Lethality

The overnight runs used `ZONE_DRAIN_RATE = 0.10` (10-tick kill). The previous successful
runs (seeds 42/43/99 in FINDINGS.md, fitness 620-736) used 0.02 (50-tick kill). The
FINDINGS.md explicitly identified 0.10 as "too fast for communication":

> A prey at the zone boundary (radius 8.0) needs ~8 ticks to walk out even knowing the
> right direction. The window between receiving a useful signal and dying is 2-3 ticks -
> too short for a meaningful signal-response-escape loop to provide fitness advantage.

This was the fix applied for runs 42/43/99, but the code currently has 0.10. Either the
fix was reverted during the performance optimization commits (signal_entropy, collect_metrics)
or it was never committed to main.

**This does not invalidate the experiment.** It tells us something precise: at 0.10 drain,
zones are too lethal for communication to pay for itself. The 71% zone mortality rate
(272/384 per gen) means most prey die before any signal-response loop can complete.

The counterfactual experiment should be RE-RUN at 0.02 drain to test whether signals
have adaptive value when prey have time to act on information. The current result
establishes a clear baseline: "at high lethality, signals are pure cost."

## What the Data Reveals

### 1. Brain Architecture Divergence (the most interesting finding)

The three runs evolved radically different brain architectures:

| | base_hidden | signal_hidden | total | strategy |
|--|-------------|---------------|-------|----------|
| baseline-s100 | 14.2 [10-16] | 13.8 [10-17] | 28.0 | balanced |
| baseline-s101 | 4.5 [4-7] | 21.7 [15-29] | 26.2 | tiny-base, huge-signal |
| mute-s100 | 6.6 [5-8] | 13.8 [8-18] | 20.4 | small-base, random-signal |

**Mute's base_hidden collapsed to 6.6.** Without signals to receive and process, brains
don't need capacity. The 6.6 neurons handle movement, food-finding, and energy monitoring.
This is the "minimum viable brain" for non-communicative survival.

**baseline-s100 maintains base_hidden at 14.2** - 115% larger than mute. The signal
environment inflates the base brain because signal inputs (18 of 36 input dimensions)
create selection pressure for processing capacity even when that processing doesn't
improve survival. This is a parasitic computational load.

**baseline-s101 found the opposite solution**: base_hidden collapsed to near-minimum (4.5)
while signal_hidden ballooned to 21.7. Yet s101 has slightly HIGHER fitness than s100
(257 vs 253). The tiny-base approach avoids the parasitic load by not trying to process
signals in the base layer - it just routes them to the signal head where they churn
uselessly but at least don't interfere with movement.

**Mute's signal_hidden drifts randomly** (oscillating 8-28, mean 13.8). With zero
selection pressure, it's pure genetic drift. In both baselines, signal_hidden is under
active selection - s100 holds it at 13.8 (matching mute's drift mean coincidentally),
s101 pushes it to 21.7. The difference: in baselines, signal_hidden MATTERS for
the organism even if it doesn't help survival. The brain is spending resources
processing signals that make it worse off.

### 2. Symbol Monopoly Returns

Both baselines converged on near-monopoly symbol distributions:

**baseline-s100**: symbol 4 at 97.9% (HHI 0.959)
- Started uniform, s2 dominated by gen 24k, s4 took over by gen 48k
- By gen 72k: s4 at 99.3%. Near-total monopoly.

**baseline-s101**: symbol 3 at 79.7%, symbol 2 at 16% (HHI 0.663)
- Different dominant symbol (arbitrary - index has no semantic prior)
- Retains a minority symbol, more diverse than s100

This is the same monopoly pattern from FINDINGS.md runs 1-2 (where one symbol dominated
98-100%). Monopoly mechanically kills MI (you need symbol variety for mutual information).
The sustained MI in both runs is near zero (0.0003 and 0.0004).

The monopoly is WORSE than the previous kill-zone runs (42/43/99) where 3-4 symbols
remained active and carried food information. At 0.10 drain, there's no fitness gradient
to maintain symbol diversity.

### 3. Signals Encode Memory, Not World State

Previous runs (at 0.02 drain) encoded food location: mi_food_dist=0.107-0.114.
The overnight runs encode something completely different:

**baseline-s100 top encodings (last 10%)**:
```
mi_mem1: 0.0039    mi_mem3: 0.0036    mi_mem4: 0.0033
mi_mem2: 0.0029    mi_mem0: 0.0027    mi_mem6: 0.0027
mi_sig5_str: 0.0023    mi_sig2_str: 0.0023    mi_sig0_str: 0.0021
```

**baseline-s101 top encodings**:
```
mi_mem2: 0.0182    mi_mem6: 0.0142    mi_mem7: 0.0125
mi_sig5_dy: 0.0070    mi_mem3: 0.0062    mi_sig1_str: 0.0051
```

Memory cells dominate in both. Signal inputs (relay) are secondary. Food and energy
are minimal. Zone distance is zero (as expected - prey can't see zones).

The MI values are also an order of magnitude lower than previous runs (0.004-0.018
vs 0.07-0.12). Signals carry almost no information about anything. What little they
carry is self-referential: memory state and incoming signal strength. This is the
"signals about signals about nothing" pattern - noise amplifying itself through
the recurrent loop without grounding in world state.

s101's higher memory MI (0.018 vs 0.004) combined with higher encoding stability
(mid-to-late Spearman 0.737 vs s100's 0.737) suggests s101 found a slightly more
stable self-referential encoding. But "more stable noise" is still noise.

### 4. The Fitness Coupling Pattern

| Metric | s100 | s101 | Previous (42/43/99) |
|--------|------|------|-------------------|
| sender_fit_corr | -0.443 | -0.558 | +0.36 to +0.46 |
| receiver_fit_corr | +0.744 | +0.767 | +0.79 to +0.87 |
| response_fit_corr | 0.000 | 0.000 | 0.000 |

**sender_fit_corr flipped negative.** In previous runs, senders benefited (food
encoding helped cooperative patch harvesting). Now senders are penalized - signaling
costs energy and provides no compensating advantage when zones kill too fast for
communication to help and food encoding hasn't emerged.

receiver_fit_corr remains the same spatial confound as always: center prey hear more
AND survive more. Not evidence of signal utility.

response_fit_corr = 0.000 is universal across ALL runs in the project's history.
The three-way causal chain (encode -> respond -> survive) has never closed.

### 5. Receiver Behavior Exists But Is Uncoupled

| Metric | s100 | s101 | Previous salvaged data |
|--------|------|------|----------------------|
| jsd_pred | 0.232 | 0.188 | ~6x jsd_no_pred |
| jsd_no_pred | 0.204 | 0.164 | - |
| jsd ratio | 1.14x | 1.15x | ~6x |

Receivers DO change behavior in response to signals (nonzero JSD). But the
in-zone/out-of-zone ratio is only 1.14x - receivers barely differentiate between
contexts. The previous 50k salvaged data showed 6x differentiation. The high
lethality regime suppresses context-sensitive response because prey die before
developing differentiated strategies.

### 6. Silence vs Alarm Calling

silence_corr: -0.207 (s100), -0.140 (s101) - consistently negative. Prey reduce
per-capita signaling near zones. This is behavioral silence, not dead silence
(the metric normalizes by alive count).

Iconicity from the raw logs: 0.412 (s100), 0.315 (s101) - positive. More signals
come from in-zone locations overall.

These aren't contradictory: more prey are active (and dying) in zones, inflating
total in-zone signal count, while individual survivors reduce their own signal rate.
The silence behavior persists across all runs in the project - it's the most
reliable emergent strategy, surviving every parameter regime change.

### 7. Computational Cost of the Signal Channel

Mute reached 149k gens vs baseline's 96k in the same wall time.
- **Mute is 55% faster** (149/96 = 1.55x)
- The signal channel consumes **35% of total computation** (1 - 1/1.55 = 0.355)

The experiment plan expected a "measurable" difference. 55% is enormous. The
`receive_detailed()` inner loop is O(alive_prey * active_signals) - with ~88k
signals per gen across 384 prey and 500 ticks, that's the dominant computational
cost center. Zero signals means the inner loop is trivial.

This has engineering implications: if signal processing is 35% of runtime, any
optimization to `receive_detailed()` or signal propagation yields large speedups.

### 8. Reproducibility: Fitness Converges, Everything Else Diverges

| What | s100 vs s101 | Interpretation |
|------|-------------|----------------|
| Sustained fitness | 252.5 vs 256.9 (+1.7%) | **Constrained** by world physics |
| base_hidden | 14.2 vs 4.5 (-68%) | **Contingent** on evolutionary path |
| signal_hidden | 13.8 vs 21.7 (+57%) | **Contingent** |
| Signal entropy | 0.015 vs 0.748 (+50x) | **Contingent** |
| Dominant symbol | s4 (98%) vs s3 (80%) | **Contingent** |
| Top encoding | memory cells | Shared (but different cells) |

The framework predicted this: "Convergence tells us what's constrained by the world.
Divergence tells us what's invented by the population." Fitness is constrained.
Everything else is convention - but convention without function, since signals are
net negative.

### 9. Epoch Structure

baseline-s100 shows 16 distinct epochs across 96k gens, alternating between brain
architecture states. The most interesting pattern:

- **gen 62-73k: "high-response"** - jsd_pred peaks at 0.408, signal_hidden at 28.8
- **gen 73-78k: "small-signal-brain"** - signal_hidden crashes to 12.5, jsd drops
- **gen 78-88k: "high-response"** - jsd recovers to 0.475 as signal_hidden rebuilds
- **gen 88-97k: "small-signal-brain"** - another crash to 13.2

This oscillation between high-response and brain-collapse epochs is the simulation
trying and failing to stabilize a communication system. Signal_hidden grows, receivers
respond more, but the response doesn't improve survival (response_fit_corr stays at 0),
so there's no selection pressure to maintain the brain size, and it collapses.

The lag correlation confirms this: MI LEADS base_hidden by ~70 gens (r=0.502). Brief
information-carrying episodes create selection for brain capacity, but the capacity
outlasts the information, creating a boom-bust cycle.

### 10. Phase Transition Signatures

The trajectory fluctuation ratio peaks at 15.76 in s100 - high instability. The top
trajectory JSD spikes cluster in gen 45-80k, coinciding with the epoch oscillations.
This matches the framework's prediction: "increasing fluctuations before the transition,
a sharp change at the critical point."

But the transition never completes. The population approaches the critical region
repeatedly and falls back. In the framework's metaphor, the system is orbiting a
phase transition boundary without enough energy to cross it.

## Diagnosis

Three factors combine to make signals net negative at current parameters:

### 1. Zone lethality prevents communication payoff

At 0.10 drain per tick, zones kill in 10 ticks. A prey at zone edge needs ~8 ticks
to escape even with perfect directional information. The communication window is
2-3 ticks - too short for signal-receive-respond-escape to complete. The information
value of signals can't exceed zero when there's no time to act on information.

This is the dominant factor. Previous runs at 0.02 drain (50-tick kill) had fitness
620-736 and signals encoding food location. At 0.10 drain, fitness drops to 250-310
and signals encode nothing useful.

### 2. Signal cost without signal benefit

At 0.002 per emission with ~88k-100k signals per gen across 384 prey, the population
pays ~0.5 energy per prey per generation in signaling costs. With sustained fitness
~250, that's 0.2% of the fitness budget - small but nonzero cost with zero benefit.

More importantly, the signal channel inflates base_hidden (14.2 vs mute's 6.6),
creating a larger genome under selection pressure with slower convergence.

### 3. Brain architecture parasitism

The signal environment creates an evolutionary trap: incoming signals are 18 of 36
brain inputs. Even when signals carry no useful information, the brain must process
them - and evolution selects for brains that handle the signal input noise rather
than ignoring it. This is visible in s100's base_hidden (14.2) being maintained
well above the mute baseline (6.6) without providing any survival advantage.

s101 found a partial escape: collapse base_hidden to 4.5 and shunt signal processing
to the dedicated signal head. This avoids base layer pollution at the cost of
over-investing in signal_hidden (21.7). s101's slightly higher fitness (257 vs 253)
suggests this escape is marginally better but still net negative vs mute (306).

## Comparison to Previous Results

| Metric | Previous (0.02 drain) | Overnight (0.10 drain) | Delta |
|--------|----------------------|----------------------|-------|
| avg_fitness | 620-736 | 252-306 | -60% |
| sender_fit_corr | +0.36 to +0.46 | -0.44 to -0.56 | **flipped** |
| Top encoding | food location (MI 0.10-0.12) | memory cells (MI 0.003-0.018) | collapse |
| Signal entropy | varied | 0.015-0.748 | - |
| response_fit_corr | 0.000 | 0.000 | unchanged |
| Zone deaths/gen | (lower, not recorded) | 223-272 (58-71% mortality) | - |
| base_hidden | collapsed to 4-5 | 4.5-14.2 | varied |
| signal_hidden | near-max (25-31) | 13.8-21.7 | lower |

The 0.02 drain runs had genuine food encoding, positive sender fitness correlation,
and signal_hidden pushed to maximum. The 0.10 drain runs have none of these. The
zone lethality parameter is the decisive variable.

## Recommendations

### Immediate: Fix ZONE_DRAIN_RATE and rerun

Change `ZONE_DRAIN_RATE` from 0.10 to 0.02 in `src/world.rs:207`. Rebuild the binary,
deploy to VPS, and rerun the exact same experiment (baseline-s100, mute-s100,
baseline-s101) at 0.02 drain.

This will answer the ACTUAL question: do signals have adaptive value when prey have
time to act on information? The overnight result answers a useful but different
question: do signals help when zones are instantly lethal? (No.)

### Keep current runs for comparison

The current 0.10-drain runs should keep running. They establish the "high lethality"
baseline. When the 0.02-drain counterfactual completes, we'll have a 2x2 matrix:

|  | 0.10 drain | 0.02 drain |
|--|-----------|-----------|
| signals enabled | current baseline runs | next experiment |
| signals disabled | current mute run | next experiment |

This isolates the interaction between zone lethality and signal value.

### After 0.02-drain experiment

If signals show adaptive value at 0.02 drain:
1. **Experiment B** from the plan: take evolved baseline population, disable signals
   mid-run. If fitness drops, the population DEPENDS on signals.
2. Focus on closing response_fit_corr (the chain that has never closed)
3. Test whether food encoding re-emerges and whether it extends to zone avoidance

If signals are still net negative at 0.02 drain:
1. Try signal_cost = 0.0 to isolate pure information value
2. Increase zone coverage (--zone-radius 10.0) to increase selection pressure
3. Consider whether the architecture can support communication at all without
   response_fit_corr ever leaving zero

## Evidence Hierarchy Update

| Level | Claim | Status after overnight |
|-------|-------|-----------------------|
| 1 | Signals have adaptive value | **NO at 0.10 drain. Untested at 0.02 drain.** |
| 2 | Receivers change behavior | Weak yes (JSD ratio 1.14x, down from 6x in prior data) |
| 3 | Different symbols carry different info | No (monopoly kills inter-symbol differentiation) |
| 4 | Responses are appropriate | No (response_fit_corr = 0.000) |
| 5 | Genuine reference | Not testable without Levels 1-4 |

## Raw Numbers Reference

### baseline-s100 (96,570 gens at metrics-interval 10)
- 9,657 data points, 16 fitness change points, 15 signal_hidden change points
- Epochs: large-base-brain -> low-response -> oscillating signal brain -> high-response cycles
- Symbol 4 monopoly (97.9%), HHI 0.959
- Encoding stability: early-late Spearman 0.053 (unstable)

### baseline-s101 (95,270 gens)
- 9,528 data points, 14 fitness change points, 18 signal_hidden change points
- Epochs: large-base-brain -> 20k gen small-signal-brain plateau -> turbulent mixing
- Symbol 3 dominant (79.7%) with symbol 2 satellite (16%), HHI 0.663
- Encoding stability: early-late Spearman 0.270 (marginally more stable)

### mute-s100 (148,970 gens)
- 14,897 data points, 16 fitness change points
- All signal metrics identically zero (correct - no signals emitted)
- base_hidden collapsed from 18.6 peak (gen 2k) to 6.6 stable
- signal_hidden drifts randomly (8-28), no selection pressure
- Flat metric health: 0% windows active for all signal metrics
