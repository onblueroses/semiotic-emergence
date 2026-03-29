# Findings

What we know, what we disproved, and what remains open. For the chronological experimental history (13 eras, 25 runs), see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Evidence Hierarchy

| Level | Claim | Rust (384 pop) | Rust (2k pop) | GPU (5k pop) |
|-------|-------|----------------|---------------|-------------|
| 1 | Signals have adaptive value | **NO** (-7% to -25%) | **NO** (-7.3% seed 42, -5.3% seed 43) | **YES** (r=+0.51, +52 fitness) |
| 2 | Receivers change behavior | Weak yes (JSD 0.15-0.27) | **Yes** (JSD 0.29-0.32; silence_move_delta +0.24) | Yes (JSD 0.033-0.066, rising) |
| 3 | Different symbols carry different info | Yes at 0.02 drain (food encoding) | **Yes** (vocabulary stratification: sym5 beacon, sym1 poison-correlated r=+0.291, sym2 rare alarm JSD=0.109) | Weak (PC1=89.9%, one channel) |
| 4 | Responses are appropriate | **PARTIAL** (v15: +0.078 to +0.125 with poison) | **SEED-DEPENDENT** (seed 42: rfc=+0.12; seed 43: rfc=-0.13). Positive rfc does not translate to adaptive value. | Metric fixed, needs GPU rebuild |
| 5 | Genuine reference | Not testable | Not testable | Not testable |

**Critical finding (v15-2k counterfactual):** Level 1 is now answered at 2k pop: **signals are NOT adaptive**. Mute prey are 7.3% fitter (seed 42) and 5.3% fitter (seed 43) in every time window tested. The integral signal value is deeply negative (-12M). Even during Regime C where rfc=+0.06 (seed 42's best period), the signal population is 8.8% worse than mute. Seed 43 failed to reproduce positive rfc entirely (sustained rfc=-0.13), making the vocabulary stratification and altruistic signaling pattern seed-specific rather than general. The 5k GPU result remains the only positive Level 1 evidence.

---

## Standing Conclusions

### Universal patterns (every era, every seed)

- **Population scale is the key variable, but 2000 is insufficient.** At 384-2000 agents, signals are net negative at every parameter configuration tested (8 eras, 20+ runs, 2 seeds at 2k). The 2k counterfactual (v15-mute-psn30-2k-42) definitively shows mute prey 7-9% fitter than signal prey, consistent across all time windows. Even with poison creating positive rfc in one seed, the population-level cost of signaling exceeds the individual benefit. At 5,000 agents (GPU), signals become adaptive (r=+0.51, +52 fitness) - but with a different architecture and unverified rfc metric. The threshold for adaptive signaling lies somewhere between 2k and 5k, likely requiring both population scale AND architectural changes.

- **response_fit_corr is seed-dependent at 2k, not reliably positive.** Without poison: -0.13 to -0.29 at all scales. With 30% poison at 384 pop: +0.078 to +0.125 (dose-response confirmed). With 30% poison at 2000 pop: **seed 42 rfc=+0.12 (95% positive), seed 43 rfc=-0.13 (3% positive)**. The positive rfc is NOT reproducible across seeds. Furthermore, even seed 42's positive rfc does not translate to adaptive value - mute prey are 8.8% fitter during the period of strongest rfc. Positive rfc means "among signal users, those who differentiate do slightly better" - it does not mean signaling helps the population.

- **The receiver paradox (resolved with poison).** Without poison: receiver_fit_corr is consistently positive (0.48-0.87 across all eras, 0.74 at v13-2k) but response_fit_corr is consistently negative. The positive receiver correlation is a spatial confound. With poison at 2k (v15-2k): recv_fit actually went NEGATIVE (-0.39 in Regime B) - unprecedented, showing the confound can be overwhelmed - while rfc turned strongly positive. Poison breaks the receiver paradox: it makes signal content adaptive while removing the spatial confound that inflated recv_fit.

- **The beacon attractor bends but doesn't break.** Every architecture tested converges to one dominant signal. At 2k+poison (v15-2k), the beacon (sym5, 82%) still dominates, but minority symbols now carry distinct information: sym1 is poison-correlated (r=+0.291), sym2 is a rare alarm (JSD 0.109). This is the first evidence of functional stratification - a beacon with annotations, not a pure monopoly. The evolutionary path to a true multi-convention vocabulary may require the brain collapse/regrowth dynamic to run longer, or higher poison ratios.

- **Brain collapse as creative destruction.** At 2k+poison (v15-2k), brains grew to 24, crashed to 7, then regrew to 17 over 90k gens. The regrown brain produced 10x higher signal channel MI (0.50 vs 0.05), more active symbols (4 vs 2-3), and higher entropy (0.72 vs 0.12). The collapse destroyed an architecture that couldn't support multi-symbol communication and rebuilt one that could. This dynamic has not been observed at 384 pop or at 2k without poison.

- **Altruistic signaling is seed-specific, not universal.** At 2k+poison, seed 42 shows the classic altruistic pattern: sender_fit negative (-0.03 to -0.11), rfc positive, gap widening to +0.13. But seed 43 shows the opposite: sender_fit POSITIVE (+0.05), rfc negative (-0.08), gap=-0.10. Seed 43 senders benefit from signaling (aggregation/selfish strategy) while receivers are worse off. The altruistic pattern is not a general consequence of population scale + poison - it depends on which evolutionary trajectory the population follows.

- **Quality vs quantity is seed-dependent.** Seed 42 (v15-2k): 111 signals/gen, positive rfc - quality strategy. Seed 43 (v15-2k): 185k signals/gen, negative rfc - volume strategy. Both are 7-8% worse than mute. Different seeds with identical parameters evolve opposite strategies (1,700x volume difference), and neither achieves adaptive value. The quality strategy narrows the gap slightly (-7.3% vs -5.3%) but doesn't close it.

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
| Shared-layer spandrels bootstrap vocabulary | 14 (v14) | Spandrel mechanism creates transient signal-context correlations (MI spikes at gen 7k, 49k) but they collapse to beacon attractor. Gate neuron becomes volume knob, not context switch. Same outcome at 384 and 2000 pop. |
| Poison food breaks the beacon attractor | 15 (v15) | Beacon bends but doesn't break. At 384 pop: rfc positive but signals net negative. At 2k pop: rfc=+0.12 (95% pos), vocabulary stratification (sym1 poison-correlated, sym2 rare alarm), but sym5 beacon still at 82%. energy_delta_mi = 0 - prey don't signal about individual poison encounters. Poison creates functional annotations on the beacon, not a second convention. |

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
| Shared-layer + gate neuron | 14 | Simpler architecture (3860 vs 5683 weights), spandrel mechanism bootstraps signal-context correlations. Gate separates emission decision from symbol selection. |
| Poison food (vocabulary pressure) | 15 | Creates positive rfc at 384 pop (dose-response confirmed). At 2k pop: seed-dependent (rfc=+0.12 seed 42, -0.13 seed 43). Does NOT make signals adaptive (mute still 7-9% fitter). Effect is real but insufficient. |

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

1. **Is response_fit_corr positive at 5k pop?** The GPU run used the pre-fix architecture. The metric was broken (measurement artifact). Needs a GPU rebuild with the fixed per-symbol JSD metric. This is now the most important question - the 5k GPU result is the ONLY positive Level 1 evidence in the entire project.

2. **Can the Rust simulation scale to 5k+ efficiently?** At 2000 pop: ~84 gen/min with 6 cores. 5000 pop would be ~15-20 gen/min, making 100k gens require ~4 days. Feasible on VPS but slow. GPU mirror (Python/JAX) is the planned path for larger scales.

3. **Does 50% poison at 2k close the fitness gap?** The 384-pop dose-response (30%->50% increased rfc from +0.078 to +0.125). At 2k the deficit is 7-9% - could higher poison pressure narrow it? Worth testing, but the seed 43 result suggests the quality strategy that makes positive rfc possible may not be reliably evolvable.

4. **What architectural change would make signals adaptive at 2k?** The shared-layer architecture imposes a brain size tax (signal runs grow 7-17 neurons vs mute's 6). A dedicated signal pathway (return to split-head?) might reduce the overhead. The GPU architecture differs significantly - understanding what it does differently could be key.

5. **Why did seed 42 and 43 diverge so dramatically?** Same parameters, different seeds: quality strategy (110 sig/gen, positive rfc) vs volume strategy (185k sig/gen, negative rfc). Understanding this bifurcation could reveal what conditions favor quality signaling.

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

9. **Can shared-layer spandrels bootstrap vocabulary?** No. v14 (shared-layer + gate neuron) creates transient signal-context correlations via the spandrel mechanism, but they collapse to beacon attractor at 384 pop and silence at 2000 pop. Architecture is not the bottleneck - the environment supports only one useful message.

10. **Can poison food create vocabulary pressure?** Yes, but only in one seed. v15 at 384 pop: rfc +0.078 to +0.125 (dose-response confirmed), signals net negative vs mute. v15 at 2k pop, seed 42: rfc=+0.12, vocabulary stratification (sym1 poison, sym2 alarm, sym5 beacon). v15 at 2k pop, seed 43: rfc=-0.13, no stratification, volume maximizer. The vocabulary pressure is real but its expression is seed-dependent. Counterfactual confirms signals net negative at 2k in both seeds (-7.3% seed 42, -5.3% seed 43).

11. **Is response_fit_corr positive at 2000 pop with poison?** Seed-dependent. Seed 42: rfc=+0.12 (95% positive last 10%). Seed 43: rfc=-0.13 (3% positive). Not reproducible across seeds at 2k pop.

12. **Are signals net adaptive at 2k with poison?** No. v15-mute-psn30-2k-42 (278k+ gens) is 7.3% fitter than signal seed 42, 5.3% fitter than signal seed 43, in every time window. Integral signal value: -12M (seed 42), -14M (seed 43). Even during Regime C (strongest rfc period), signals are 8.8% worse. This mirrors the 384-pop result (-7 to -10%) and extends to the strongest signal configuration tested.

13. **Is vocabulary stratification reproducible?** No. Seed 43 evolved a two-symbol monopoly (sym3 71%, sym1 29%) instead of seed 42's functional stratification. The brain collapse/regrowth cycle that produced 10x signal MI at seed 42 did not occur at seed 43. The quality signaling trajectory is contingent on early evolutionary decisions, not a general attractor.

14. **Does positive rfc imply adaptive signaling?** No. Seed 42's Regime C has rfc=+0.06 but is 8.8% worse than mute. Positive rfc means symbol-differentiating individuals are fitter *among signal users*, but the entire signaling class is less fit than non-signalers. This is the key conceptual distinction: rfc measures within-group fitness coupling, not between-group adaptive value.
