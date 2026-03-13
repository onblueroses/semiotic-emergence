# Findings

Experimental history of semiotic-emergence. Each era documents what we tested, what we found, and what we changed in response. Standing conclusions at the bottom.

## Table of Contents

- [Run Registry](#run-registry)
- [Era 1: Baseline](#era-1-baseline-20x20-48-prey-fixed-brain) - 20x20, 48 prey, fixed brain
- [Era 2: 8x Scale](#era-2-8x-scale-56x56-384-prey-evolvable-brain) - 56x56, 384 prey, evolvable brain
- [Era 3: Architecture v2](#era-3-architecture-v2) - Split-head brain, 6 symbols, memory, patches, kin
- [Era 4: Kill Zones](#era-4-kill-zones) - Invisible danger, zone damage, food encoding
- [Era 5: Counterfactual Testing](#era-5-counterfactual-testing) - Do signals have adaptive value?
- [Standing Conclusions](#standing-conclusions)

---

## Run Registry

Every significant run, its parameters, and headline result.

| ID | Era | Seeds | Gens | Key params | Headline result |
|----|-----|-------|------|-----------|----------------|
| baseline-10k | 1 | 42 | 10,000 | 48 prey, 20x20, 2 pred, fixed 6 hidden, 3 sym | 5 semiotic epochs, peak MI 0.237, no stability |
| 8x-100k | 2 | 42 | 100,000 | 384 prey, 56x56, 16 pred, 4-16 hidden, 3 sym, cost 0.0002/neuron, signal cost 0.01 | Brain collapse to 4 neurons, signals are spandrels |
| boost-84k | 2 | unknown | 84,139 | + evasion boost, free signals, cheap neurons | Signals are fuel not language |
| boost-68k | 2 | 42 | 68,428 | same as boost-84k | Same outcome, different evolutionary path |
| run3-100k | 2 | 42 | 100,000 | boost removed, signal cost 0.002, pred=3, 4-124 hidden | Genuine encoding emerged then collapsed (MI 0.669 peak) |
| kz-42 | 4 | 42 | 36,084 | Kill zones, drain 0.02, free brains, split-head, 6 sym | Food encoding (MI 0.107) |
| kz-43 | 4 | 43 | 37,441 | same | Two-tier signal relay |
| kz-99 | 4 | 99 | 93,000 | same, metrics-interval 1000 | Food encoding (MI 0.114), five-act trajectory |
| cf-s100 | 5 | 100 | 96,570 | drain 0.10 (accidental), signals on | Signals net negative (-25.5% vs mute) |
| cf-s101 | 5 | 101 | 95,270 | same | Reproduces cf-s100 within 1.7% fitness |
| mute-s100 | 5 | 100 | 148,970 | drain 0.10, --no-signals | 20-25% fitter than baselines |

Detailed per-run analysis: `findings/` directory. Data files: `analysis/` directory.

---

## Era 1: Baseline (20x20, 48 prey, fixed brain)

*Question: Can communication emerge in a minimal system?*

**Parameters:** 48 prey, 20x20 toroidal grid, 2 visible predators, fixed 6-neuron hidden layer, 3 symbols, 500 ticks/gen, vision 4.0, signal range 8.0 (2:1 ratio), signal cost 0.01.

**Runs:** seed 42, 50/200/3k/10k generations.

### Findings

**1. Silence-as-sign is immediate.** Silence correlation is negative from gen 0 (-0.37) and stays negative across all timescales. Prey suppress signals near predators. But the gen-0 value suggests this is an architectural spandrel - random weights in a shared 6-neuron layer create incidental predator-to-signal-suppression pathways. Evolution maintains but does not amplify it.

**2. Symbol 0 goes extinct.** From gen ~700 onward, sym0 is never emitted. The population self-compressed from 3 to 2 symbols without any selection pressure for vocabulary size. The 6-neuron layer can't maintain three differentiated outputs.

**3. Five distinct semiotic epochs appear in 10k gens.**

| Epoch | Gens | Duration | Peak MI | Notable |
|-------|------|----------|---------|---------|
| 1 | 600-900 | ~300 | 0.020 | JSD hits 0.693 (theoretical max) |
| 2 | 1670-1900 | ~230 | 0.130 | Sender-side coherence |
| 3 | 2300-2460 | ~160 | 0.213 | First high-MI epoch |
| 4 | 3100-3270 | ~170 | 0.185 | Repeat of epoch 3 pattern |
| 5 | 8890-9350 | ~460 | 0.237 | Longest, highest MI, positive iconicity |

Gap between epoch 4 and 5: 5,500 generations of drift.

Epoch 1 was receiver-dominated - all per-symbol JSDs simultaneously hit 0.693 (maximum), then collapsed. Epoch 5 was the strongest by multiple measures, and the first time positive iconicity appeared (prey signaling MORE near predators - alarm calling instead of silence). It appeared only after 9,000 generations.

**4. No stable attractor.** Every epoch collapses back to low MI. The 6-neuron hidden layer can find weight configurations supporting communication but can't maintain them against drift. Movement, silence suppression, and differentiated emission compete for the same 6 neurons.

**5. Receiver infrastructure matures independently.** jsd_pred stays elevated (0.2-0.6) from gen ~4,000 even between sender-side MI epochs. Receivers develop signal sensitivity that persists when sender behavior drifts. The receiver side is more robust than the sender side.

**6. response_fit_corr and silence onset metrics are data-starved.** The 30-sample minimum threshold is too high for 48 agents over 500 ticks. response_fit_corr produces non-zero values in 1 of 200 generations.

### What we changed (for Era 2)

Moved to 8x scale with an evolvable hidden layer to test whether larger brains and populations stabilize semiotic states. Hidden size became a heritable gene (4-16 range) with energy cost proportional to neuron count.

Detailed analysis: `findings/2026-03-09-baseline-runs.md`

---

## Era 2: 8x Scale (56x56, 384 prey, evolvable brain)

*Question: Does scale + evolvable brains stabilize communication?*

This era had three sub-phases as we diagnosed and addressed structural barriers.

### Phase 1: Initial 8x run (100k gens)

**Parameters:** 384 prey, 56x56, 16 predators (pred speed 8 cells/tick), 4-16 evolvable hidden, 3 symbols, neuron cost 0.0002/tick, signal cost 0.01, food 200.

**Run:** seed 42, 100,000 generations.

**1. Brain collapse is the root problem.** Evolution drives hidden size to 4 (minimum). The cost difference between 4 and 6 neurons is 0.2 energy over 500 ticks - equivalent to two-thirds of a food pellet. With 4 neurons shared between 16 inputs and 8 outputs, there's zero spare capacity for signal processing.

**2. Signals are spandrels, not communication.** 70-91% of signals come from prey that can already see the predator. Neural crosstalk from flee behavior activating shared hidden neurons, not intentional warning.

**3. Signaling actively hurts fitness.** sender_fit_corr stabilizes at -0.34 by gen 30k. Active signalers pay metabolic cost with no kin-selection benefit.

**4. Receiver benefit is a spatial confound.** receiver_fit_corr is rock-solid at ~0.48 for all 100k gens. Entirely explained by position: center prey hear more AND have more neighbors between them and predators. response_fit_corr = 0.0 throughout.

**5. Symbol vocabulary collapses to monopoly.** 3 symbols -> 1 dominant + 1 vestigial + 1 extinct. Without differentiation pressure, the system collapses to a single signal type.

**6. Speed mismatch makes warning useless.** Predator moves 8 cells/tick, prey 1 cell/tick. A warned prey at signal range can only move 3 cells in the ~3 ticks before the predator arrives.

**7. Trajectory freezes completely.** Trajectory JSD declines from 0.016 (0-10k) to 0.0006 (90-100k). The semiotic landscape is frozen by end of run.

Detailed analysis: `findings/2026-03-09-100k-8x-scale.md`

### Phase 2: Pro-communication tuning

Four changes to address the structural barriers identified in Phase 1:

| Change | From | To | Why |
|--------|------|----|-----|
| Neuron cost | 0.0002 | 0.00002 | Brain of 16 now costs 0.56 vs 0.44 for brain of 4 - negligible delta |
| Signal cost | 0.01 | 0.0 (free) | Removes -0.34 sender fitness penalty |
| Predator speed | round(3*scale) | round(1.5*scale) | Warned prey now have ~6 ticks to react instead of ~3 |
| Evasion boost | none | +1 movement when receiving signal near predator | Direct fitness benefit for signal-responsive prey |

Rationale: `findings/2026-03-09-pro-communication-tuning.md`

### Phase 3: Evasion boost era (runs 1-2)

**Parameters:** 384 prey, 56x56, 16 predators (speed 4), free signals, neuron cost 0.00002, evasion boost active.

**Runs:** local (unknown seed, 84,139 gens), VPS seed 42 (68,428 gens).

**1. Signals are fuel, not language.** The evasion boost (+1 movement on signal reception near predator) dominates. Pearson(signals_emitted, avg_fitness) = 0.989. Evolution maximized signal volume (~48k signals/gen, 30%+ of theoretical max) while the content remained irrelevant.

**2. MI is confounded with symbol diversity.** The local run's MI spike (0.68 at gen 37k) corresponded to a symbol transition, not genuine encoding. Pearson(HHI, MI) = -0.521. MI mechanically requires symbol variety.

| Gen | sym0 | sym1 | sym2 | MI | Phase |
|-----|------|------|------|----|-------|
| 15k | 98% | 0% | 2% | 0.007 | sym0 monopoly |
| 35k | 66% | 0% | 34% | 0.172 | diversifying |
| 45k | 33% | 37% | 30% | 0.082 | max diversity |
| 60k | 0% | 0% | 100% | 0.008 | sym2 monopoly |

**3. The causal chain almost never completes.** Requiring MI > 0.05, jsd_pred > 0.05, and response_fit_corr > 0.05 simultaneously: local achieved this for 5.4% of its run (scattered, transient), seed42 for 0.0% (29 gens total).

**4. Two different evolutionary paths, same outcome.** The runs diverged dramatically (local: brain peaked late at 31.6, MI peaked at 0.68; seed42: brain peaked early at 36.4, MI peaked at 0.11) then converged to the same steady state (~133 fitness, ~48k signals/gen, zero MI).

**5. Silence onset effect is mechanical.** When signals stop, prey freeze 80.5% of the time - not a learned response but the evasion boost turning off.

**6. Predator saturation undermines information asymmetry.** 16 predators on 56x56: 88% of the time prey have a predator within vision. The signal channel can't bridge a gap that barely exists.

**7. Vestigial danger symbol in seed42.** Rare sym1 (0.2%) concentrates 88.1% in d0 (nearest predator bin), vs 27.6% for dominant sym0. A ghost of functional differentiation - sym1 once meant "danger here" but was nearly extinct.

**Diagnosis:** Three structural features prevent emergence: (1) evasion boost rewards signal presence not content, (2) free signals have no cost pressure against noise, (3) predator saturation eliminates information asymmetry. These interact: the boost makes presence valuable, free cost allows noise to proliferate, and saturation means there's nothing to communicate.

### What we changed (for run 3)

| Change | From | To | Why |
|--------|------|----|-----|
| Evasion boost | active | removed | Rewarded presence not content |
| Signal cost | 0.0 | 0.002 | Create pressure against noise |
| Predator count | 16 | 3 | Create information gap (~55% of prey outside vision but within signal range) |
| Food | 200 | 100 | Maintain resource pressure at lower predator count |

### Phase 4: Run 3 (100k gens, single hidden layer)

**Parameters:** 384 prey, 56x56, 3 visible predators (speed 4), signal cost 0.002, single hidden layer 4-124, 3 symbols.

**Run:** seed 42, 100,000 generations.

**1. Brain size is the rate-limiting factor.** Small brains (~6-10 neurons): MI peaks at 0.05. Large brains (~15 neurons): MI reaches 0.624. Brain size naturally evolved to the MAX_HIDDEN=16 ceiling, suggesting the constraint was artificial. (Brain capacity later expanded to 124.)

**2. Brain expansion destroys then rebuilds semiotic structure.** At gen 46-50k, avg_hidden exploded from 10 to 15. Total semiotic collapse followed - MI, iconicity, sender_fit_corr all dropped to zero. The old 6-neuron signal strategy didn't transfer to 15 neurons. Took ~25k gens to rebuild, but the rebuilt system was stronger (MI sustained above 0.2 for 6,187 consecutive gens).

**3. Genuine encoding emerged during MI surge (gen 75-100k).** Input MI showed signals encoded predator information: predator distance (MI 0.472), predator dy (0.267), predator dx (0.199). Signal-0-strength MI of 0.179 suggested signal relaying.

**4. The coupling problem.** Single hidden layer means movement and signal outputs share all weights. Every movement adaptation changes signal behavior. This creates spandrels, fragility, and the convention collapse at gen 46-50k.

**5. Convention instability.** MI peaked at 0.669 but collapsed due to neutral drift - fitness barely changed whether prey communicated or not. The signal channel didn't matter enough to defend itself against genetic drift.

### What we changed (for Era 3)

The coupling problem and convention instability motivated the v2 architecture: split-head brain to decouple movement from signaling, plus supporting changes to create stronger selection pressure for communication.

---

## Era 3: Architecture v2

*Question: Does decoupled signal processing + richer vocabulary enable stable communication?*

Five changes to address structural barriers from eras 1-2:

### 1. Split-head brain

Single hidden layer replaced with base hidden (4-64, shared across all outputs) + signal hidden (2-32, dedicated to signal outputs). Two separate hidden size genes evolve independently. Addresses the coupling problem: signal outputs now pass through their own hidden layer.

### 2. Six symbols (was 3)

Richer vocabulary for encoding food, zone proximity, direction. Harder for one symbol to monopolize - driving MI to zero by dominating 6 symbols is harder than 3.

### 3. Recurrent memory (8 cells)

EMA update (0.9*old + 0.1*tanh(output)). Memory feeds back as input, creating a recurrent loop. Enables temporal reasoning across ticks.

### 4. Cooperative food patches

50% of food requires 2+ prey within Chebyshev distance 2. Creates direct fitness incentive for spatial coordination via signals.

### 5. Kin fitness

Siblings (+0.5) and cousins (+0.25) bonus on selection fitness. Lineage tracked via parent/grandparent indices. Supports altruistic signaling.

### Supporting changes

- Vision halved (11.2 -> 5.6 at grid=56), signal range unchanged (22.4). 4:1 ratio forces signal reliance.
- Neuron cost halved (0.00002 -> 0.00001).
- Predator speed reduced (round(1.5*scale) -> round(scale)). More time to respond to warnings.
- Softmax emission replaces threshold-based. Emit if max(softmax) > 1/6.

### Early observations (1000 gen smoke test, visible predators)

Brain compression: base hidden shrinks from 12 to ~4.4, signal hidden stays ~5-6. The split architecture is working as intended - evolution finds minimal base processing sufficient but retains signal capacity. JSD rising, MI climbing, silence correlation consistently negative (-0.37 to -0.51).

But across 5 seeds with visible predators, MI correlated negatively with fitness. Communication was actively harmful. Prey that signaled paid the metabolic cost (0.002/emission) while giving away their position for no compensating advantage.

**The fundamental issue:** when prey can see the threat, signals are redundant. Evolution found that shutting up and running was strictly better than warning neighbors. This motivated the most significant change since the split-head brain.

---

## Era 4: Kill Zones

*Question: Does making danger invisible force communication to become structurally necessary?*

### The design

Three invisible circular zones (radius 8.0, ~19% grid coverage) drift randomly across the 56x56 grid. Zone speed 0.5 (probabilistic). Brain inputs 0-2 are always zero - prey cannot see zones. The only self-signal of danger is energy loss (input 35), which doesn't indicate direction.

This creates structural information asymmetry:
- A prey inside a zone knows only that energy is dropping, not where the boundary is
- Random fleeing has ~50% chance of going deeper into the zone
- Signals from nearby prey carry dx/dy directional information - the only source of escape direction
- Communication is the difference between directed escape and a coin flip

### Initial implementation and immediate problems (drain 0.10)

First kill zone runs (seeds 42/43, ~900 gens) used ZONE_DRAIN_RATE = 0.10, killing in 10 ticks.

Early smoke test showed promising metrics: positive iconicity (alarm calling), 6:1 jsd_pred/jsd_no_pred ratio, receiver_fit_corr 0.76. But three structural issues emerged:

**1. Dead silence vs behavioral silence.** silence_corr showed strong negatives (-0.58 to -0.90), but this was a mortality artifact: zones killing prey reduces total signal volume regardless of behavior. Fix: normalize signals_per_tick by alive_per_tick.

**2. Zone lethality too fast for communication.** At 0.10 drain, zones kill in 10 ticks. A prey at zone boundary needs ~8 ticks to walk out. Communication window is 2-3 ticks - too short for signal-response-escape. Fix: ZONE_DRAIN_RATE reduced to 0.02 (50-tick kill).

**3. Brain collapse to minimum (again).** avg base hidden = 4.2, signal hidden = 2.2. Same collapse seen at every neuron cost tested (0.0002, 0.00002, 0.00001). The cost doesn't matter - larger brains provide no fitness advantage when communication hasn't emerged yet. Fix: neuron_cost set to 0.0 (free brains).

### Zone damage separated from energy

Initially, zone drain reduced the same energy pool that food replenished. This meant prey could offset zone damage by eating, undermining the lethality model. Zone damage was separated into its own accumulator (zone_damage field, death at >= 1.0). Food cannot heal zone damage. Energy and zone damage are independent threats.

### The dying sound experiment (added then removed)

When a prey died to zone damage, it emitted all 6 symbols simultaneously - a "dying scream." The hypothesis: dying sounds would create a strong spatial signal that survivors could learn to flee from.

**Result: catastrophic.** With 3 zones and high death rates, dying bursts created near-constant grid-wide signal coverage (3 zones * high death rate * 22-cell signal range). This:
- Suppressed MI to ~0 (constant signal noise drowns out any structured information)
- Prevented silence transitions (the grid is never quiet)
- Made the signal channel useless by flooding it

The dying sound was removed. Dying prey now signal only via their brain's normal final emission before zone_drain runs. This was an important lesson: more signal != more communication. The channel needs quiet periods for structured information to have value.

### Long-duration results (seeds 42/43/99, drain 0.02, free brains)

**Parameters:** pop=384, grid=56, zones=3, radius=8.0, speed=0.5, food=100, ticks=500, signal_cost=0.002, kin_bonus=0.10, neuron_cost=0.0, 6 symbols, split-head brain. ZONE_DRAIN_RATE=0.02.

seed99 at metrics-interval 1000 (94 data points); seeds 42/43 at full resolution.

#### Universal findings (all three seeds)

**1. Signal hidden converges to maximum.** All three independently evolved near-max signal processing: seed42 avg 29.0, seed43 avg 30.8, seed99 avg 25.3. Starting from ~6, all sprinted to high signal_hidden by gen 18-25k. With free brains, this reflects genuine selection pressure for signal processing capacity.

**2. Zone encoding is zero, universally.** mi_zone_dist = 0.000 in all three seeds across entire runs. Zones create lethal pressure but never appear in signal content.

**3. response_fit_corr = 0.000, universally.** The three-way coupling chain (encode -> respond -> survive) never closes.

**4. receiver_fit_corr ~0.79-0.87.** Confirmed spatial confound. Consistent since run 1, does not indicate signal utility.

**5. Sender selection is real but moderate.** sender_fit_corr = 0.36-0.46. Likely through cooperative patch harvesting - active signalers are co-located with others, satisfying the 2+ prey requirement for patch food.

#### Divergent attractors: same pressure, three solutions

| Seed | Final symbol distribution | Primary encoding |
|------|--------------------------|-----------------|
| 42 | s1=33%, s3=32%, s4=25% (3-way split) | Food location (mi_food_dy=0.119, mi_food_dist=0.107) |
| 43 | s3=78%, s5=19% (near monopoly + satellite) | Signal relay (mi_sig5_str=0.072) |
| 99 | s2=28%, s5=25%, s0=20%, s4=16% (4-way spread) | Food location (mi_food_dist=0.114, mi_food_dx=0.100) |

The specific surviving symbols are arbitrary, but the convergence on food encoding is not.

**The seed42/seed99 pattern: direct food encoding.** Both converge on signals encoding food location. Multiple active symbols carry food proximity information. seed99's encoding is cleaner at 93k (MI 0.114) than seed42's at 36k (MI 0.107), suggesting consolidation over time.

**The seed43 anomaly: two-tier signal relay.** Symbol 3 (78%) encodes the strength and direction of symbol 5 (19%), which encodes food location. A second-order relay: s5 -> food proximity -> s3 -> where s5 activity is concentrated. Achieves the same result as direct food encoding through an indirect route, but at lower fitness (620 vs 695/736) and higher signal_hidden (30.8). The relay is less efficient.

#### Evolutionary trajectory: five acts (seed99)

1. **Gen 0-5k: Chaotic sweeps.** Signal hidden oscillates wildly. Symbol 3 hits 61% then goes permanently extinct.
2. **Gen 15-25k: Low-complexity monopoly.** One symbol at 72-79%, signal_hidden small (2-6). Cheapest viable strategy.
3. **Gen 30-35k: Breakthrough.** Signal hidden explodes (+11.9 in one window). Genotype sweep with high signal processing capacity.
4. **Gen 35-55k: Turbulent semiotic window.** Closest to danger signaling: MI peaks at 0.006, silence_corr hits -0.231, jsd_pred peaks at 0.311. All three couple briefly at gen ~45k. Coherence dissolves immediately.
5. **Gen 55-93k: Food encoding consolidation.** Signal_hidden climbs to 25-31. Food MI grows 0.15 -> 0.34. Zone MI stays at 0.000. Danger signaling abandoned; food coordination is the stable attractor.

The turbulent semiotic window (act 4) appears in all three seeds during the post-breakthrough phase. Always transient. Danger signaling is attempted and abandoned in favor of food encoding.

#### Why danger signaling fails

- **Food encoding competes.** Prey are more active near food, which correlates spatially with zones. A zone-signaling convention would require reducing signals near food patches that happen to overlap zones - unstable.
- **Timing.** Even at 0.02 drain (50-tick kill), the signal-response-escape loop may be too slow to generate fitness advantage.
- **Ecological niche exclusion.** Once food encoding occupies 4-5 symbols, there's no room for danger conventions.

#### Cross-run summary

| Metric | seed42 (36k) | seed43 (37k) | seed99 (93k) |
|--------|-------------|-------------|-------------|
| avg_fitness | 695.3 | 620.0 | 736.2 |
| signal_hidden | 29.0 | 30.8 | 25.3 |
| mutual_info (final) | 0.0001 | 0.0003 | 0.0001 |
| jsd_pred | 0.215 | 0.155 | 0.018 |
| silence_corr | +0.046 | -0.018 | -0.012 |
| sender_fit_corr | 0.360 | 0.419 | 0.464 |
| receiver_fit_corr | 0.826 | 0.873 | 0.789 |
| response_fit_corr | **0.000** | **0.000** | **0.000** |
| Zone MI | 0.000 | 0.000 | 0.000 |
| Top encoding | food location | signal relay | food location |
| Encoding stability | 0.461 | 0.188 | ~0.46 |

---

## Era 5: Counterfactual Testing

*Question: Do signals actually improve fitness, or are they noise that evolution tolerates?*

### Experimental design

First controlled counterfactual in the project's history. Three runs, same binary, same parameters except the signal channel.

| Run | Seed | Signals | Threads | Gens reached |
|-----|------|---------|---------|-------------|
| baseline-s100 | 100 | enabled | 4 | 96,570 |
| mute-s100 | 100 | --no-signals | 4 | 148,970 |
| baseline-s101 | 101 | enabled | 4 | 95,270 |

Parameters: pop=384, grid=56, zones=3, radius=8.0, speed=0.5, food=100, ticks=500, signal_cost=0.002, kin_bonus=0.10, neuron_cost=0.0. Binary commit 29c5f98.

### Critical discovery: wrong zone lethality

The binary ran with ZONE_DRAIN_RATE = 0.10 (10-tick kill). The Era 4 runs (seeds 42/43/99, fitness 620-736) used 0.02 (50-tick kill). The fix was either reverted during performance optimization commits or never committed to main.

This does not invalidate the experiment. It answers a different but useful question: at high zone lethality, can signals pay for themselves? The 71% per-generation zone mortality rate (272/384) means most prey die before any signal-response loop can complete.

### Findings

**1. Signals are net negative (-25.5%).** Mute population outperforms both baselines.

| Metric | baseline-s100 | baseline-s101 | mute-s100 |
|--------|--------------|--------------|-----------|
| Sustained avg fitness | 252.5 | 256.9 | 306.2 |
| Final avg fitness | 239.7 | 279.9 | 334.7 |
| Peak avg fitness | 360.1 | 378.4 | 413.0 |
| Zone deaths/gen | 272 | 223 | 251 |

Counterfactual signal value integral: -3,936,072. The gap is stable across 96k gens - not converging, not diverging. Level 1 of the evidence hierarchy answered for this parameter regime: signals do not have adaptive value at 0.10 drain.

**2. Brain architecture parasitism.** The signal environment inflates base_hidden by 115% (14.2 vs mute's 6.6). With 18 of 36 inputs being signal channels, evolution selects for processing capacity even when it provides zero survival benefit.

| Run | base_hidden | signal_hidden | total |
|-----|-------------|---------------|-------|
| baseline-s100 | 14.2 [10-16] | 13.8 [10-17] | 28.0 |
| baseline-s101 | 4.5 [4-7] | 21.7 [15-29] | 26.2 |
| mute-s100 | 6.6 [5-8] | 13.8 [8-18] | 20.4 |

baseline-s101 found a partial escape: collapse base_hidden to near-minimum, shunt to signal_hidden. Marginally higher fitness (257 vs 253) but still loses to mute (306).

**3. Signals encode memory, not world state.** Previous runs at 0.02 drain encoded food location (MI 0.107-0.114). Overnight runs encode memory cells (MI 0.003-0.018) and incoming signal strength (self-referential relay). An order of magnitude less information. "Signals about signals about nothing."

**4. Sender fitness correlation flipped negative.** sender_fit_corr: -0.443 (s100), -0.558 (s101). Previously +0.36 to +0.46. At high lethality, signaling is pure cost with no compensating benefit. Kin bonus (0.10) is insufficient.

**5. Symbol monopoly returns.** s100: symbol 4 at 97.9% (HHI 0.959). s101: symbol 3 at 79.7% (HHI 0.663). Monopoly kills MI. Previous runs at 0.02 drain maintained 3-4 active symbols.

**6. response_fit_corr remains zero.** Still 0.000. The three-way causal chain has never closed in the project's history.

**7. Fitness converges, everything else diverges.** Sustained fitness differs by 1.7% between s100 and s101. base_hidden differs by 3.2x, signal entropy by 50x, dominant symbol is different. Fitness is constrained by world physics; brain architecture and symbol system are contingent path-dependent convention without function.

**8. Signal channel costs 35% of computation.** Mute reached 149k gens vs baseline's 96k in the same wall time. The receive_detailed() inner loop is O(alive_prey * active_signals).

**9. Epoch oscillations around a phase transition.** baseline-s100 shows boom-bust cycles: signal_hidden grows, jsd_pred peaks (0.41-0.47), then signal_hidden crashes because response doesn't improve survival. MI leads base_hidden by ~70 gens (r=0.502). The system orbits a phase transition boundary without crossing it.

### What this means

The experiment establishes a clear negative result at 0.10 drain. The next experiment: rerun at 0.02 drain (the parameter used for food-encoding runs) to test whether signals have adaptive value when prey have time to act on information.

Combined, this would produce a 2x2 matrix isolating the interaction:

|  | 0.10 drain | 0.02 drain |
|--|-----------|-----------|
| signals enabled | net negative (-25.5%) | not yet tested |
| signals disabled | baseline (mute-s100) | not yet tested |

Detailed analysis: `findings/2026-03-13-overnight-analysis.md`
Experimental design: `findings/2026-03-12-overnight-experiment.md`

---

## Standing Conclusions

What holds true across all runs, what's been disproven, and what remains open.

### Universal patterns (every era, every seed)

- **response_fit_corr = 0.** The three-way causal chain (encode -> respond -> survive) has never closed. Receivers change behavior in response to signals but that change does not predict survival. This is the single most persistent negative result.
- **receiver_fit_corr is a spatial confound.** Center prey hear more signals AND survive more. Consistently 0.48-0.87 across all eras. Not evidence of signal utility.
- **Silence near danger.** Prey reduce per-capita signaling near threats. Present from gen 0, maintained but not amplified by evolution. Likely an architectural spandrel of shared hidden layers, not a learned strategy.
- **Symbol monopoly under weak selection.** Without strong differentiation pressure, one symbol dominates. Seen in eras 1, 2 (phase 3), and 5. Only resisted when signals encode useful information (era 4 at 0.02 drain).
- **Fitness converges, conventions diverge.** Different seeds reach similar fitness but with completely different brain architectures, dominant symbols, and encoding profiles. Fitness is constrained by physics; everything else is contingent.

### Disproven hypotheses

| Hypothesis | Era tested | Result |
|-----------|-----------|--------|
| Larger brains stabilize communication | 2 (phase 1) | Brain collapses to minimum when there's no fitness gradient for signal processing |
| Free signals enable communication | 2 (phase 3) | Free signals + evasion boost -> volume maximization, not content |
| Evasion boost creates receiver benefit | 2 (phase 3) | Rewards signal presence not content, evolution exploits |
| Visible predators create communication pressure | 3 | Prey see danger directly, signals are redundant, shutting up and running is strictly better |
| High zone lethality forces communication | 5 | Zones kill too fast for signal-response loops, signals become net cost |
| Neuron cost drives brain collapse | 4 (initial) | Same collapse at every cost tested (0.0002, 0.00002, 0.00001, 0.0) |
| Dying sound provides useful danger signal | 4 | Floods grid with noise, suppresses MI to ~0 |

### What works

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

### Open questions

1. **Do signals have adaptive value at 0.02 drain?** The 2x2 counterfactual matrix is half-complete. The 0.10-drain result (net negative) doesn't predict the 0.02-drain result where food encoding already works.

2. **Can danger signaling coexist with food encoding?** The turbulent semiotic windows (era 4, all seeds) show populations briefly attempting danger signaling before abandoning it. Is this niche exclusion (food encoding occupies the channel) or insufficient selection pressure?

3. **Why is response_fit_corr always zero?** Three possible explanations: (a) the metric threshold is too strict (10+ samples per bucket per prey), (b) receiver behavioral changes are real but orthogonal to fitness, (c) the architecture genuinely cannot close the loop. A counterfactual at 0.02 drain with and without patch food would distinguish (a) from (c).

4. **Publication readiness (ALIFE 2026, August, Waterloo).** Need MI > 0.3 sustained across 5+ seeds. Currently have MI 0.10-0.12 across 3 seeds at 0.02 drain. The food-encoding result is publishable as-is if framed correctly (emergent coordination rather than referential communication).

### Evidence hierarchy status

| Level | Claim | Status |
|-------|-------|--------|
| 1 | Signals have adaptive value | NO at 0.10 drain. Untested at 0.02 drain (where food encoding exists) |
| 2 | Receivers change behavior | Weak yes (JSD nonzero, but in-zone ratio degraded from 6x to 1.14x at high lethality) |
| 3 | Different symbols carry different info | Yes at 0.02 drain (3-4 active symbols with food encoding). No at 0.10 drain (monopoly) |
| 4 | Responses are appropriate | Never observed (response_fit_corr = 0) |
| 5 | Genuine reference | Not testable without levels 1-4 |

### Parameter history

Tracks every significant parameter change and why.

| Parameter | Era 1 | Era 2 (phase 1) | Era 2 (phase 2) | Era 2 (phase 4) | Era 3 | Era 4 (current) |
|-----------|-------|-----------------|-----------------|-----------------|-------|-----------------|
| Population | 48 | 384 | 384 | 384 | 384 | 384 |
| Grid | 20x20 | 56x56 | 56x56 | 56x56 | 56x56 | 56x56 |
| Threat | 2 pred (vis) | 16 pred (vis) | 16 pred (vis) | 3 pred (vis) | 3 pred (vis) | 3 zones (invis) |
| Threat speed | - | 8 | 4 | 4 | round(scale) | 0.5 prob |
| Hidden layer | 6 fixed | 4-16 evolv | 4-16 evolv | 4-124 evolv | 4-64 base + 2-32 sig | same |
| Symbols | 3 | 3 | 3 | 3 | 6 | 6 |
| Signal cost | 0.01 | 0.01 | 0.0 | 0.002 | 0.002 | 0.002 |
| Neuron cost | 0 (fixed) | 0.0002 | 0.00002 | 0.00002 | 0.00001 | 0.0 |
| Evasion boost | no | no | yes | no | no | no |
| Vision | 4.0 | 11.2 | 11.2 | 11.2 | 5.6 | 5.6 |
| Signal range | 8.0 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 |
| Memory | no | no | no | no | 8 cells | 8 cells |
| Patches | no | no | no | no | 50% | 50% |
| Kin fitness | no | no | no | no | 0.5/0.25 | 0.5/0.25 |
| Zone drain | - | - | - | - | - | 0.02 (code: 0.10) |
| Food | 25 | 200 | 200 | 100 | 100 | 100 |
