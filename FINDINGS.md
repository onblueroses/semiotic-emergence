# Findings

What we know, what we disproved, and what remains open. For the chronological experimental history (13 eras, 25 runs), see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Evidence Hierarchy

| Level | Claim | Rust (384 pop) | Rust (2k pop) | GPU (5k pop) |
|-------|-------|----------------|---------------|-------------|
| 1 | Signals have adaptive value | **NO** (-7% to -25%) | Untested (needs v15-2k counterfactual) | **YES** (r=+0.51, +52 fitness) |
| 2 | Receivers change behavior | Weak yes (JSD 0.15-0.27) | **Yes** (JSD 0.29-0.32; silence_move_delta +0.24) | Yes (JSD 0.033-0.066, rising) |
| 3 | Different symbols carry different info | Yes at 0.02 drain (food encoding) | **Yes** (vocabulary stratification: sym5 beacon, sym1 poison-correlated r=+0.291, sym2 rare alarm JSD=0.109) | Weak (PC1=89.9%, one channel) |
| 4 | Responses are appropriate | **PARTIAL** (v15: +0.078 to +0.125 with poison) | **YES with poison** (v15-2k: rfc=+0.12, 95% positive last 10%), **NO without** (v13: -0.29) | Metric fixed, needs GPU rebuild |
| 5 | Genuine reference | Not testable | Not testable | Not testable |

**Critical finding (v15-2k):** Level 4 is now strongly positive at 2000 pop with 30% poison. response_fit_corr=+0.12 (95% positive in last 10%), up from +0.078 at 384 pop. First evidence of vocabulary stratification: sym5=beacon (82%, anti-poison r=-0.143), sym1=poison-correlated (9%, r=+0.291), sym2=rare alarm (2%, JSD 0.109). Brain collapse/regrowth cycle at 90-155k gens destroyed an architecture unable to support multi-symbol communication and rebuilt one that could (signal MI 10x higher post-collapse). The altruism gap (rfc - sender_fit) is widening: senders pay increasing cost, receivers benefit increasingly from content. Mute counterfactual still needed for Level 1.

---

## Standing Conclusions

### Universal patterns (every era, every seed)

- **Population scale is the key variable, environmental complexity is the catalyst.** At 384-1000 agents, signals are net negative at every parameter configuration tested (8 eras, 15+ runs). At 2000 without poison (v13), signals carry real information (food_mi=0.14) but responses are maladaptive (rfc=-0.29). At 2000 WITH poison (v15-2k), rfc flips to +0.12 (95% positive last 10%) with vocabulary stratification. At 5,000 agents (GPU), signals become adaptive without poison (r=+0.51). The interaction of population scale x environmental complexity produces signal quality that neither achieves alone.

- **response_fit_corr is negative without environmental complexity, positive with it.** Without poison: -0.13 to -0.28 at 384 pop (v11), -0.29 at 2000 pop (v13). With 30% poison at 384 pop: +0.078 (65% positive). With 50% poison at 384: +0.125 (77% positive). With 30% poison at 2000 pop: +0.12 (95% positive last 10%), strengthening through 188k gens. Dose-response confirmed at both scales. However, positive rfc among signal users does not confirm net adaptive value - mute counterfactual still needed at 2k.

- **The receiver paradox (resolved with poison).** Without poison: receiver_fit_corr is consistently positive (0.48-0.87 across all eras, 0.74 at v13-2k) but response_fit_corr is consistently negative. The positive receiver correlation is a spatial confound. With poison at 2k (v15-2k): recv_fit actually went NEGATIVE (-0.39 in Regime B) - unprecedented, showing the confound can be overwhelmed - while rfc turned strongly positive. Poison breaks the receiver paradox: it makes signal content adaptive while removing the spatial confound that inflated recv_fit.

- **The beacon attractor bends but doesn't break.** Every architecture tested converges to one dominant signal. At 2k+poison (v15-2k), the beacon (sym5, 82%) still dominates, but minority symbols now carry distinct information: sym1 is poison-correlated (r=+0.291), sym2 is a rare alarm (JSD 0.109). This is the first evidence of functional stratification - a beacon with annotations, not a pure monopoly. The evolutionary path to a true multi-convention vocabulary may require the brain collapse/regrowth dynamic to run longer, or higher poison ratios.

- **Brain collapse as creative destruction.** At 2k+poison (v15-2k), brains grew to 24, crashed to 7, then regrew to 17 over 90k gens. The regrown brain produced 10x higher signal channel MI (0.50 vs 0.05), more active symbols (4 vs 2-3), and higher entropy (0.72 vs 0.12). The collapse destroyed an architecture that couldn't support multi-symbol communication and rebuilt one that could. This dynamic has not been observed at 384 pop or at 2k without poison.

- **Altruistic signaling at scale.** At 2k+poison, sender_fit_corr is consistently negative and getting more negative (-0.03 to -0.11 across Regime C), while rfc simultaneously strengthens (+0.03 to +0.12). Senders pay increasing fitness costs; receivers benefit increasingly from signal content. The altruism gap (rfc - sender_fit) widened from +0.16 to +0.22. This is the classic sender-cost/receiver-benefit pattern predicted by costly signaling theory.

- **Quality over quantity.** v13 (2k, no poison) floods 846k signals/gen with negative rfc. v15-2k (2k, 30% poison) emits 111 signals/gen (7,600x fewer) with positive rfc. The correlation holds within the v15-2k run: quieter generations have higher fitness AND better signal quality. Signal cost per agent is negligible at 111 signals/2000 agents - the quality difference is about information content, not energy cost.

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
| Poison food (vocabulary pressure) | 15 | First positive response_fit_corr at 384 pop. At 2k pop: rfc=+0.12 (95% pos), vocabulary stratification, brain collapse/regrowth producing 10x signal MI. Strongest signal quality in project history. |

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

1. **Are signals net adaptive at 2k+poison?** The single most important remaining question. v15-psn30-2k-42 shows rfc=+0.12 (95% pos) with vocabulary stratification, but we don't know if this translates to positive counterfactual signal value. At 384 pop, rfc was positive but signals were -7 to -10% vs mute. The 2k run has much stronger rfc AND much lower signal volume (111/gen vs 846k at v13) - this could be the configuration where signals tip positive. Next test: v15-mute-psn30-2k-42 (same params, --no-signals).

2. **Is the vocabulary stratification reproducible?** One seed (42), one run. The sym1-poison / sym5-beacon pattern could be contingent. Different seeds at 2k+poison needed.

3. **Is response_fit_corr positive at 5k pop?** The GPU run used the pre-fix architecture. The metric was broken (measurement artifact). Needs a GPU rebuild with the fixed per-symbol JSD metric.

4. **Can the Rust simulation scale to 5k+ efficiently?** At 2000 pop: ~132 gen/min with 12 cores (improved from prior 25 gen/min). 5000 pop would be ~20 gen/min, making 100k gens require ~3.5 days. Feasible on VPS but slow. GPU mirror (Python/JAX) is the planned path for larger scales.

5. **Would the brain collapse/regrowth dynamic continue?** The v15-2k run was terminated at 188k gens with rfc still rising. The 90k-155k evolutionary winter ended with a 10x improvement in signal MI. Would extending produce another cycle? Or has the architecture stabilized?

6. **Does 50% poison at 2k pop strengthen the effect?** The 384-pop dose-response (30%->50% poison increased rfc from +0.078 to +0.125) suggests higher poison would strengthen vocabulary differentiation further at 2k.

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

10. **Can poison food create vocabulary pressure?** Yes. v15 at 384 pop: rfc +0.078 to +0.125 (dose-response confirmed), but signals still net negative vs mute. v15 at 2k pop: rfc=+0.12 (95% positive last 10%), vocabulary stratification (sym1 poison-correlated r=+0.291, sym2 rare alarm JSD=0.109, sym5 beacon at 82%). Brain collapse/regrowth dynamic produced 10x signal MI improvement. Poison creates genuine vocabulary pressure that strengthens with population - but the beacon persists as the dominant symbol and energy_delta_mi remains zero (prey don't signal about individual poison encounters). Counterfactual needed for Level 1.

11. **Is response_fit_corr positive at 2000 pop with poison?** Yes. v15-psn30-2k-42 (188k gens) shows rfc=+0.04 overall (67% positive), +0.12 in last 10% (95% positive), with the trend still strengthening at termination. Compare v13-2k without poison: rfc=-0.29 (6% positive). Poison is confirmed as the causal variable for positive rfc at both 384 and 2000 pop.
