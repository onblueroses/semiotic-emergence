# Findings

What we know, what we disproved, and what remains open. For the chronological experimental history (13 eras, 25 runs), see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Evidence Hierarchy

| Level | Claim | Rust (384 pop) | Rust (2k pop) | GPU (5k pop) |
|-------|-------|----------------|---------------|-------------|
| 1 | Signals have adaptive value | **NO** (-8% to -25%) | Untested (needs counterfactual) | **YES** (r=+0.51, +52 fitness) |
| 2 | Receivers change behavior | Weak yes (JSD 0.15-0.27) | Yes (JSD 0.29-0.32) | Yes (JSD 0.033-0.066, rising) |
| 3 | Different symbols carry different info | Yes at 0.02 drain (food encoding) | **Yes** (food_mi=0.14, sig strengths interleaved with food inputs) | Weak (PC1=89.9%, one channel) |
| 4 | Responses are appropriate | **NO** (v11: -0.13 to -0.28) | **NO** (v13: -0.29) | Metric fixed, needs GPU rebuild |
| 5 | Genuine reference | Not testable | Not testable | Not testable |

**Critical finding (v13):** Level 4 now measured at 2000 pop - still negative (response_fit_corr=-0.29). The receiver paradox: receiver_fit_corr=0.74 (being near signals correlates with fitness) but responding to signal content is maladaptive. This narrows the emergence threshold: response_fit_corr is negative at 2000 and positive at 5000, so the crossover is somewhere in 2000-5000. A counterfactual (--no-signals) at 2000 pop is needed to test Level 1. Level 4 at 5k pop (GPU) also remains unmeasured.

---

## Standing Conclusions

### Universal patterns (every era, every seed)

- **Population scale is the key variable.** At 384-1000 agents, signals are net negative at every parameter configuration tested (8 eras, 15+ runs). At 2000, signals carry real information (food_mi=0.14) but responses are maladaptive (response_fit_corr=-0.29). At 5,000 agents (GPU), signals become adaptive (r=+0.51). The emergence threshold lies between 2000 and 5000.

- **response_fit_corr is negative up to 2000 pop.** At 384 pop: -0.13 to -0.28 (v11). At 2000 pop: -0.29 (v13). Symbol differentiation is maladaptive because direct spatial inputs (food/ally direction) remain more reliable than the signal channel. The signal environment at 2000 pop is informationally richer than at 384 (food_mi 0.14 vs 0.01) but still too noisy for symbol-differentiated responses to outperform direct perception.

- **The receiver paradox.** receiver_fit_corr is consistently positive (0.48-0.87 across all eras, 0.74 at 2000 pop), but response_fit_corr is consistently negative. Being near signals correlates with fitness; acting on signal content reduces fitness. The positive receiver correlation is a spatial confound: center prey hear more signals AND encounter more food AND have more escape routes. The negative response correlation is the real signal: the channel is too noisy for content to be actionable.

- **Silence near danger.** Prey reduce per-capita signaling near threats. Present from gen 0, maintained but not amplified by evolution. Likely an architectural spandrel of shared hidden layers, not a learned strategy.

- **Symbol reduction, not monopoly, at scale.** At 384 pop, one symbol dominates (monopoly). At 2000 pop (v13), the vocabulary reduces from 6 to 4 active symbols with a relatively even distribution (HHI=0.28 vs monopoly ~0.5+). The surviving symbols carry real information (input MI 0.08-0.10); the extinct ones don't. This is vocabulary optimization, not collapse.

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
| Food encoding | 4 | Emerged independently in 3 seeds. MI 0.10-0.12 sustained. Strongest at 2k pop (0.14) |
| Signal relay (seed43) | 4 | Emerged spontaneously as alternative to direct encoding |
| Metrics-interval=10 | 13 | 10x finer resolution reveals dynamics masked at 200 (v10 vs v13 signal hidden trajectories) |

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

1. **What is the minimum population for signal emergence?** NARROWED. Bracketed between 2000 (response_fit_corr=-0.29, v13) and 5,000 (receiver_fit_corr=+0.51, GPU). The crossover where symbol differentiation becomes adaptive is somewhere in this range. Next tests: pop=3000 or 4000 bracket runs, and/or counterfactual at 2000.

2. **Is receiver_fit_corr a pure spatial confound at 2000 pop?** v13 shows receiver_fit_corr=0.74 but we can't separate signal utility from spatial advantage without a counterfactual. Next test: v13-mute-2k-42 (same params, --no-signals). If mute prey show similar center-survival advantage, the 0.74 is spatial confound.

3. **Is response_fit_corr positive at 5k pop?** The GPU run used the pre-fix architecture. The metric was broken (measurement artifact). Needs a GPU rebuild with the fixed per-symbol JSD metric.

4. **Can the Rust simulation scale to 5k+ efficiently?** At 2000 pop: 25 gen/min (CPU-bound on signal reception at 41% of runtime). 5000 pop would be ~4 gen/min, making 100k gens impractical on VPS. GPU mirror (Python/JAX) is the planned path.

---

## Resolved Questions

1. **Do signals have adaptive value at 0.02 drain?** Not at 384 pop. v6 counterfactual shows signals -8%. But at 5k pop (GPU), signals are adaptive at drain 0.15.

2. **Can danger signaling coexist with food encoding?** No at small scale. At large scale (GPU), food encoding vanishes entirely - the signal environment itself becomes the primary information source.

3. **Why is response_fit_corr always zero?** Measurement artifact. Signal coverage is so high that every prey hears signals every tick. The "no signal" bucket never reaches the 10-sample threshold. Fixed in commit 31a1516.

4. **Can stripping redundant inputs restore signal value?** No. Era 8 stripped death echoes and freeze pressure. Food encoding persisted but mute still +43% fitter. Population scale, not competing information channels.

5. **Does making food harder to find amplify signal value?** Not at 384 pop. Era 8 (vision=2.0, vision=0.5) had food encoding but signals still net negative.

6. **Can the response_fit_corr metric be fixed?** Fixed and measured (commit 31a1516, v11 data). Metric works. Biological result: symbol differentiation is maladaptive at 384 pop.

7. **Does removing spatial perception flip response_fit_corr positive?** No. v12-blind6-42 shows response_fit_corr=-0.044, MI~0, 2 symbols extinct. Removing perception destroys information asymmetry rather than redirecting it through signals.

8. **Is response_fit_corr positive at 2000 pop?** No. v13-2k-42 (100k gens, fixed metrics) shows response_fit_corr=-0.29. Symbol differentiation remains maladaptive at 2000 pop despite strong food encoding (food_mi=0.14) and high receiver_fit_corr (0.74). The emergence threshold is above 2000.
