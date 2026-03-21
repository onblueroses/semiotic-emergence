# Findings

What we know, what we disproved, and what remains open. For the chronological experimental history (12 eras, 24 runs), see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Evidence Hierarchy

| Level | Claim | Rust (384-1k pop) | GPU (5k pop) |
|-------|-------|-------------------|-------------|
| 1 | Signals have adaptive value | **NO** at all configs (-8% to -25%) | **YES** (r=+0.51, +52 fitness) |
| 2 | Receivers change behavior | Weak yes (JSD 0.15-0.27) | Yes (JSD 0.033-0.066, rising) |
| 3 | Different symbols carry different info | Yes at 0.02 drain (food encoding) | Weak (PC1=89.9%, one channel) |
| 4 | Responses are appropriate | **NO** - symbol differentiation maladaptive (v11) | Metric fixed, needs GPU rebuild |
| 5 | Genuine reference | Not testable | Not testable |

**Critical gap:** Level 4 measured at 384 pop - negative both sighted (v11) and blind (v12). Blind mode disproven as a path to signal emergence. v10 at 2000 pop shows signal infrastructure selection (hidden layer growth, symbol diversity) but response_fit_corr unmeasurable (pre-fix architecture). The key open experiment is v12+ architecture at 2000 pop to measure response_fit_corr. Level 4 at 5k pop (GPU) also remains unmeasured.

---

## Standing Conclusions

### Universal patterns (every era, every seed)

- **Population scale is the key variable.** At 384-1000 agents, signals are net negative at every parameter configuration tested (8 eras, 15+ runs). At 5,000 agents (GPU), signals become adaptive (r=+0.51). The signal environment must be dense enough for statistical regularity to emerge from noisy senders.

- **response_fit_corr is negative at 384 pop.** The metric was fixed in commit 31a1516 (was always 0 due to measurement artifact). v11 data shows symbol differentiation is actively maladaptive at cap=6 (-0.13 to -0.28) and neutral at cap=32 (~0). Prey that respond differently to different symbols do worse because direct spatial inputs (food/ally direction) are more reliable than the noisy signal channel. This is the strongest evidence yet that population scale - not ecological conditions or metric artifacts - is the bottleneck.

- **receiver_fit_corr is a spatial confound.** Center prey hear more signals AND survive more. Consistently 0.48-0.87 across all eras. Not evidence of signal utility.

- **Silence near danger.** Prey reduce per-capita signaling near threats. Present from gen 0, maintained but not amplified by evolution. Likely an architectural spandrel of shared hidden layers, not a learned strategy.

- **Symbol monopoly under weak selection.** Without strong differentiation pressure, one symbol dominates. Seen in eras 1, 2 (phase 3), and 5. Only resisted when signals encode useful information (era 4 at 0.02 drain).

- **Fitness converges, conventions diverge.** Different seeds reach similar fitness but with completely different brain architectures, dominant symbols, and encoding profiles. Fitness is constrained by physics; everything else is contingent.

---

## Disproven Hypotheses

| Hypothesis | Era tested | Result |
|-----------|-----------|--------|
| Larger brains stabilize communication | 2 (phase 1) | Brain collapses to minimum when there's no fitness gradient for signal processing |
| Free signals enable communication | 2 (phase 3) | Free signals + evasion boost -> volume maximization, not content |
| Evasion boost creates receiver benefit | 2 (phase 3) | Rewards signal presence not content, evolution exploits |
| Visible predators create communication pressure | 3 | Prey see danger directly, signals are redundant, shutting up and running is strictly better |
| High zone lethality forces communication | 5 | Zones kill too fast for signal-response loops, signals become net cost |
| Neuron cost drives brain collapse | 4 (initial) | Same collapse at every cost tested (0.0002, 0.00002, 0.00001, 0.0) |
| Dying sound provides useful danger signal | 4 | Floods grid with noise, suppresses MI to ~0 |
| Freeze zones create richer communication | 6 | Heterogeneous threats make zone signaling harder (need two conventions simultaneously) |
| Death echo inputs help communication | 7 | Free directional info competes with signals, reducing signal value |
| Deme group selection rescues signaling | 7 | Too coarse (every 100 gens) to stabilize conventions that drift every gen |
| Higher signal threshold improves signal quality | 7 | Reduced signal diversity (entropy 1.17 vs 1.64), correlating with lower MI |
| 10x cheaper signals enable communication | 7 | Cost was never the bottleneck; signals fail because responses don't improve survival |
| Medium drain (0.05) is the sweet spot | 7 | Signals -12.8% at 0.05, worse than -8% at 0.02 |
| Stripping free info channels restores signal value | 8 | Food encoding persisted (input MI 0.10-0.18) but mute still +43% fitter |
| Reduced vision forces signal reliance | 8 | Vision 2.0 and 0.5 both had food encoding but signals still net negative |
| Demes enable altruistic food signaling | 8 (v9) | 4x4 demes + near-blindness still produced mute +56% fitter |
| Ecological conditions are the bottleneck | 8+GPU | **Disproven: population scale is the bottleneck** |
| Constraining signal capacity improves encoding quality | 9 (v11) | Cap=6 produces more food encoding but symbol differentiation is maladaptive (-0.13 to -0.28 response_fit_corr). Direct spatial inputs outcompete signals. |
| Removing spatial perception forces signal dependence | 12 (v12) | Blind mode: MI~0, 2 symbols extinct, fitness halved. Prey can't signal about things they can't perceive. Memory replaces perception, not signals. |

---

## What Works

| Feature | Era introduced | Status |
|---------|---------------|--------|
| Split-head brain | 3 | Working. signal_hidden independently selected, reaches near-max |
| Kill zones (invisible danger) | 4 | Working. Creates structural information asymmetry |
| Free brains (neuron_cost=0) | 4 | Working. Brain sizes explore freely, signal capacity grows |
| Cooperative food patches | 3 | Working. Creates coordination incentive that signals exploit |
| 4:1 vision:signal ratio | 3 | Working. Forces reliance on social information |
| Zone drain 0.02 (50-tick kill) | 4 | Working. Enough time for signal-response loops |
| Food encoding | 4 | Emerged independently in 3 seeds. MI 0.10-0.12 sustained |
| Signal relay (seed43) | 4 | Emerged spontaneously as alternative to direct encoding |

---

## The Metric Problem

### Discovery: MI was measuring the wrong thing

The headline metric I(Signal; ZoneDistance) measures whether signals encode zone proximity. It does not measure whether signals encode food location, ally position, or any other world-state information.

v6 achieved input MI of 0.137 on food_dx - the strongest structured encoding in the project's history - while headline MI showed ~0. We spent two eras (v6, v7) trying to "fix" communication that was already working, because the metric was blind to it.

### The information channel competition

Every brain input that provides world-state information without signals reduces signal value:

| Input | What it tells prey | Added in | Effect |
|-------|-------------------|----------|--------|
| zone_damage (0) | "I'm hurting" | Era 4 | Necessary - drives zone avoidance |
| energy_delta (1) | "I'm gaining/losing energy" | Era 4 | Disambiguates zone from metabolism |
| freeze_pressure (2) | "I'm in a freeze zone, this deep" | v6 | Reduces signal value for freeze zones |
| death_nearby (36) | "Something died nearby, intensity" | v7 | Free directional danger info |
| death_dx/dy (37-38) | "Death was in this direction" | v7 | Makes signals redundant for zone avoidance |

Inputs 0-1 are justified: prey need body-state awareness. Inputs 2, 36-38 give away information that would otherwise require signals - they actively compete with the communication channel.

### What this taught us

1. **Measure what signals actually encode**, not what you think they should encode. The food_mi metric and input MI analysis revealed the real signal content.
2. **Every "helpful" feature is a competing information channel.** Each addition degraded the signal environment it was meant to improve.
3. **Stop adding features.** Era 4's food encoding happened with the simplest feature set and was degraded by subsequent additions.

---

## Open Questions

1. **What is the minimum population for signal emergence?** PARTIALLY ANSWERED. Bracketed between 384 (all negative) and 5,000 (positive). v10-2k-42 (100k gens, pop=2000) shows intermediate regime: signal infrastructure selected but response_fit_corr unmeasurable (pre-fix). Next test: v12+ architecture at 2000 pop with fixed metric.

2. **Is response_fit_corr positive at 5k pop?** The GPU run used the pre-fix architecture. The metric was broken (measurement artifact). Needs a GPU rebuild with the fixed per-symbol JSD metric.

3. **Can the Rust simulation scale to 5k+ efficiently?** Current architecture is CPU-bound. The GPU mirror spec (Python/JAX) is the planned path for 100k-prey runs.

---

## Resolved Questions

1. **Do signals have adaptive value at 0.02 drain?** Not at 384 pop. v6 counterfactual shows signals -8%. But at 5k pop (GPU), signals are adaptive at drain 0.15.

2. **Can danger signaling coexist with food encoding?** No at small scale. At large scale (GPU), food encoding vanishes entirely - the signal environment itself becomes the primary information source.

3. **Why is response_fit_corr always zero?** Measurement artifact. Signal coverage is so high that every prey hears signals every tick. The "no signal" bucket never reaches the 10-sample threshold. Fixed in commit 31a1516.

4. **Can stripping redundant inputs restore signal value?** No. Era 8 stripped death echoes and freeze pressure. Food encoding persisted but mute still +43% fitter. Population scale, not competing information channels.

5. **Does making food harder to find amplify signal value?** Not at 384 pop. Era 8 (vision=2.0, vision=0.5) had food encoding but signals still net negative.

6. **Can the response_fit_corr metric be fixed?** Fixed and measured (commit 31a1516, v11 data). Metric works. Biological result: symbol differentiation is maladaptive at 384 pop.

7. **Does removing spatial perception flip response_fit_corr positive?** No. v12-blind6-42 shows response_fit_corr=-0.044, MI~0, 2 symbols extinct. Removing perception destroys information asymmetry rather than redirecting it through signals.
