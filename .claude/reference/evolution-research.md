# Evolution Simulation Research
Living reference document. Updated: 2026-02-25 | Rounds: 1 (5 agents), 2a (4 agents), 2b (3 agents)

Read this before making architectural decisions about evolution, fitness, communication, or simulation parameters.
For language evolution specifics (Lewis games, ILM, deep RL findings, paper strategy): see `language-evolution-research.md`.

---

## Critical Findings

**Kin clustering is prerequisite for communication.** Random prey placement at generation boundary destroys spatial kin structure. Without r >= 0.25 among neighbors, deception dominates over honest signaling (Mitri 2009/2011). Fix: inherit parent position + small jitter (stdev 3-5 cells) at generation boundary. See "Kin Clustering" section.

**Fitness normalization is essential.** Raw survival ticks (0-2000) overwhelm offspring count (0-5) regardless of weights. Normalize all components to [0,1] before weighting.

**Signal cost of 3.0 is well-calibrated.** 15% of food value, 3x move cost. Spam is self-punishing (drains 100 energy in 22 ticks). Hamilton's rule easily satisfied for kin (c=3 < r*b = 0.5*120 = 60). No change needed.

**Start simple.** biosim4: 0 predators, 0 food system, 0 terrain. Still got interesting behavior. Every successful project started simpler than we are now.

---

## NEAT Algorithm

### Initial Population
- Start minimal: 0 hidden nodes, inputs + bias + outputs. Random-topology NEAT was 7x slower (Stanley).
- **Bias node essential**: constant 1.0 node breaks output symmetry (fixes our North bias problem). Evolution decides which outputs need bias by evolving connections FROM the bias node.
- Sparse initial connections: for 36 inputs, starting with ~20% connections lets NEAT decide which matter.
- Weight init: Gaussian mean 0, stdev 1.0 (Stanley). Range +/-30.0.

### Stanley's Mutation Rates
| Mutation | Rate |
|----------|------|
| Weight perturbation | 0.80 |
| Uniform perturb (vs new random) | 0.90 / 0.10 |
| Add connection | 0.10 |
| Add node | 0.01 |
| Disabled gene re-enable | 0.25 |

### Top Implementation Bugs (from NEAT-Python, SharpNEAT, JS implementations)
1. Topological sort must include ALL nodes (not just input-reachable)
2. Innovation numbers not deduplicated within generation
3. Node split weights: A->C=1.0, C->B=old_weight (minimizes disruption)
4. Disabled genes in crossover: 75% remain disabled (not always toggle)
5. Missing fitness sharing (adjusted_fitness = fitness / species_size)
6. Not resetting node values between activations
7. Global best genome not tracked separately from species best
8. Structural mutation rates too high (node=1%, connection=5-10%, weight=80%)
9. Crossover: excess/disjoint from fitter parent only
10. Weight mutation power too high for sigmoid's sensitive range

### Speciation
- Stanley's code doesn't divide by N in compatibility distance (despite paper)
- Dynamic threshold: target 5-15 species for pop 300. Adjust +/-0.3/gen
- Stagnation: never kill species with global best genome
- Weight coefficient too high -> speciation by weight alone, not topology

---

## Energy Economics

### Our Budget Math (actual config values)
| Parameter | Value |
|-----------|-------|
| initial_energy | 100.0 |
| max_energy | 200.0 |
| energy_per_tick (metabolism) | 0.5 |
| move_energy_cost | 1.0 |
| food_energy | 20.0 |
| signal_energy_cost | 3.0 |
| reproduce_threshold | 120.0 |
| reproduce_cost | 60.0 |

- Idle survival: 100 / 0.5 = **200 ticks**
- Moving survival: 100 / 1.5 = **67 ticks**
- Realistic (70% moving): 100 / 1.2 = **83 ticks**
- Food needed to sustain: 6 items per 100 ticks (1 every ~17 ticks)
- Food-energy ratio: 20:1.5 = **13.3:1** (tight end - Beltoforion uses 40:1)

### Food Density
- 720 food items on 4800 cells (15% density)
- Regrow every 50 ticks -> 28,800 food produced per 2000-tick generation
- Total demand: 150 prey * 120 items = 18,000
- Supply/demand ratio: ~1.6 (low pressure - food is abundant)
- Sweet spot for foraging pressure: 0.8-1.2 ratio
- **Recommendation**: increase food_regrow_ticks from 50 to 70-80

### Reproduction
- Gap from start to threshold: 120 - 100 = 20 energy = just 1 food item. **Too easy.**
- **Recommendation**: raise threshold to 150 (requires ~3 food items above maintenance)
- Parent after reproducing at 150: 150-60 = 90 energy (viable)
- Successful forager can reproduce 4-5 times per generation

### Predator Kill Rate
- 9 predators * (2000/30 cooldown) = ~600 max attacks/gen
- At 30% success rate: ~180 kills against 150 prey = **50-70% mortality (too high)**
- Target: 30-50% mortality for communication to evolve
- **Recommendation**: increase attack_cooldown from 30 to 45-50

### Parameter Change Summary
| Parameter | Current | Recommended | Reason |
|-----------|---------|-------------|--------|
| food_regrow_ticks | 50 | 70-80 | Push supply/demand closer to 1.0 |
| reproduce_threshold | 120 | 150 | Require meaningful foraging investment |
| attack_cooldown | 30 | 45-50 | Reduce mortality from ~60% to ~40% |
| signal_energy_cost | 3.0 | 3.0 (keep) | Well-calibrated per Hamilton's rule |
| generation_ticks | 2000 | 1200-2000 | Diminishing returns after tick 500 |

---

## Kin Clustering and Spatial Structure

### THE CRITICAL PROBLEM

Our `spawn_prey` uses `random_passable_pos()` every generation. **This destroys ALL spatial kin clustering.** Communication CANNOT evolve without kin structure (Mitri 2009).

### What the Literature Says
- **Reggia 2001**: Spatial locality of reproduction is the #1 requirement. Both parent selection AND child placement must be spatially local.
- **Mitri 2011**: Quantified the relatedness threshold:
  - r=0: deception dominates
  - r=0.25: mixed results
  - r >= 0.5: honest signaling evolves
  - No difference between r=0.75 and r=1.0
- **Avida**: 1-cell offspring dispersal, NO generational reset. Kin clusters form naturally.
- **The Bibites**: Continuous sim, eggs hatch near parent. Natural kin clusters.
- **biosim4**: Random placement each gen, but does NOT try to evolve communication.

### Taylor Cancellation Problem
In viscous populations, kin cooperation benefits are cancelled by kin competition (Taylor 1992). Three mechanisms break this:
1. **Population elasticity**: alarm calls increase local survival (our mechanism)
2. **Scale mismatch**: food competition is global, alarm benefit is local (partially applies)
3. **Budding/group dispersal**: related individuals disperse together

### Fix: Inherited Position + Jitter
At generation boundary:
- Store parent's end-of-generation position alongside genome
- Place offspring within Gaussian jitter (stdev 3-5 cells) of parent position
- For crossover offspring: use fitter parent's position
- 5-10% placed randomly (genetic flow between regions)
- Kin neighborhoods of radius ~10 cells, overlapping with hearing_range

### Implicit Kin Radius
| Distance | Implicit r | Mechanism |
|----------|-----------|-----------|
| 1-3 cells | ~0.5 | Parent-offspring, siblings |
| 4-6 cells | ~0.125-0.25 | Cousins, grand-offspring |
| 7+ cells | ~0 | Random chance |

hearing_range should be <= 2x kin cluster radius (~10-15 cells).

---

## Communication/Signal Evolution

### Conditions for Emergence
- Predator alarm calls evolve easily when predation kills 30-50% per generation
- Food calls only with scarce, clumped food
- **Spatial reproduction locality is #1** (Reggia 2001)
- Agent density must be sufficient (signals useless without receivers)
- Without kin clustering: deception evolves instead (Mitri 2009)

### Signal System Design
- 8 discrete symbols good (agents self-limit to 2-4 meaningful ones)
- Hearing range = 1.5-2x vision range
- Signal lifetime = 2-3 ticks sufficient
- One-tick delay is a feature (Beer & Williams 2019)
- Cost 3.0 energy: 15% of food value, spam is self-punishing

### Expected Timeline
| Generation | What Happens |
|-----------|-------------|
| 0-50 | No meaningful signaling |
| 50-200 | Signal spam (high rate, no information content) |
| 200-500 | Symbol differentiation begins |
| 500-1000 | Stable communication protocol in kin groups |
| 1000+ | Refinement, equilibrium, or complex strategies |

### Measuring Communication
| Metric | Formula | Meaningful Value | When to Check |
|--------|---------|-----------------|---------------|
| MI(Signal; PredatorType) | Joint entropy - marginals | > 0.05 after N>1000 signals | Gen 50+ |
| TopSim | Spearman(situation_dist, signal_dist) | > 0.3 = compositional | Gen 200+ |
| Deception rate | signals_no_predator / total | < 0.15 normal, > 0.30 concerning | After MI > 0.1 |
| Gini coefficient | Symbol usage concentration | 0.3-0.6 healthy | Gen 100+ |
| Receiver survival boost | survival_near_signal / survival_far | > 1.0 = signals help | Gen 500+ |

**Anti-metrics** (misleading): raw signal entropy, signal frequency alone, survival correlation without controls.

---

## Fitness Function

### Current Formula Problems
Raw survival (0-2000) massively dominates offspring (0-5 * 2.0 = 0-10). Weights are meaningless without normalization.

### Recommendations
1. Normalize to [0,1] before weighting: `survival_norm = age / max_ticks`
2. Drop energy weight (redundant with survival + offspring)
3. Simplest effective: `survival_ticks/max_ticks + offspring_count/max_offspring * weight`
4. Add signal utility bonus when communication is active (reward signals that help receivers survive)
5. Tournament selection size 3, elitism 2 per species
6. Consider sqrt(survival) to compress late-generation diminishing returns

### Pitfalls
- Pure survival -> "couch potato" strategy (stand still forever)
- Energy at evaluation rewards hoarding, not foraging
- Kin bonus can dominate in later generations (cap it)
- Don't reward signaling directly - reward outcomes of signaling

---

## Observability

### Minimum Viable Console Output (7 numbers per generation)
```
Gen 42 | alive: 87/150 | fit: 312.4/1847.0/198.7 | spc: 7 | gsz: 14.2 | sig: 2341 (H=1.82)
```
Fields: survival count, mean/max/std fitness, species count, avg genome size, signal count + entropy.

### Alarm Thresholds
| Condition | Threshold | Meaning |
|-----------|-----------|---------|
| Population collapse | alive < 20 for 3 gens | Predation too strong |
| Species collapse | species <= 2 for 10 gens | Premature convergence |
| Fitness stagnation | max unchanged 30 gens | No selection signal |
| Convergence | fitness CV < 5% for 10 gens | Population homogeneous |
| Signal spam | signals/alive > 100 AND MI < 0.05 | Noise, not communication |
| Genome bloat | avg connections > 100 | Runaway complexification |

### Signs of Broken Evolution
- **Flat fitness**: max/mean ratio < 1.10 for 50 gens (no differentiation)
- **Genetic drift**: selection_differential/fitness_sd < 0.2 (selection barely acts)
- **Specification gaming**: high fitness + action_entropy < 0.5 (one action dominates)
- **Death cause ratio**: predation/starvation > 10:1 (predators too strong) or < 0.5:1 (predators irrelevant)

### Behavioral Milestones
| Milestone | Detection | Expected Gen |
|-----------|-----------|-------------|
| Food-seeking | food_eaten/alive > 3 | 10-50 |
| Predator avoidance | predation deaths drop 40%+ | 30-200 |
| Signal spam phase | signal count rises, MI ~ 0 | 50-150 |
| Symbol differentiation | MI > 0.05, entropy drops | 150-500 |
| Communication protocol | MI > 0.3, TopSim > 0.3 | 500-1000 |

---

## Progressive Difficulty / Curriculum

### Bootstrap Problem
Random brains die immediately. The Bibites: "most will be useless and die almost immediately - typically takes several hours at 5x speed."

### Solution: Staged Configs
| Phase | Predators | Terrain | Food | Signal Cost | Expected Gens |
|-------|-----------|---------|------|-------------|---------------|
| 0 - Foraging | 0 | Flat | 0.30 | 0.0 | 50-100 |
| 1 - One Predator | 1 ground | Flat | 0.25 | 0.0 | 100-200 |
| 2 - Pressure | 3 mixed | Flat | 0.20 | 0.0 | 200-500 |
| 3 - Communication | 5+ mixed | Flat | 0.15 | 3.0 | 500-2000+ |

### Key Curriculum Insights
- Include signal neurons from gen 0 (avoid architecture change) but set signal_cost = 0 until communication phase
- Carry genomes forward across phase transitions (never reset population)
- Transition metric: survival rate > 70% for 10 consecutive gens -> advance
- Never hot-swap: population_size, vocab_size, sensor/output counts (structural)
- Safe to hot-swap: predator counts, speeds, food density, energy costs

### From Literature
- NEAT XOR: ~32 generations at pop 150 (Stanley 2002)
- Food-seeking in grid: 20-50 gens
- Predator avoidance: 50-200 gens
- Swarming from survival pressure: ~1200 gens (Li et al. 2023)
- Communication protocol: 500-1000 gens (Reggia 2001)

---

## Predator-Prey Sim Design

### Successful Reference Projects
| Project | Grid | Pop | Steps/gen | Predators | Selection | Communication? |
|---------|------|-----|-----------|-----------|-----------|----------------|
| biosim4 | 128x128 | 3000 | 300 | 0 | Positional | No |
| The Bibites | Continuous | Variable | Continuous | Evolved | Natural | No |
| skaramicke | 50x50 | 50+50 | Continuous | Co-evolved | Age at death | No |
| Hunting for Insights | Unity | 3+3 | Episodic | RL | Reward | No |
| Grouchy 2016 | 40x40 cont. | 50x100 islands | 100K/era | Implicit (mate-finding) | Natural | Yes (icon->symbol) |
| Cangelosi 1998 | 2D grid | ~100 | Generational | 0 | Fitness | Yes (mushroom categories) |
| Steels 1995 | None | 2-N | N/A | 0 | Naming game | Yes (shared vocabulary) |
| Mordatch 2018 | Continuous 2D | 2-10 | Episodic | 0 | RL reward | Yes (compositional) |

### The Grouchy Result (only published icon-to-symbol transition)
- 50 agents per island, 100 islands, 40x40 continuous, toroidal
- Genetic programming trees (max 200 nodes), single continuous sound channel
- **Indexical communication in 99% of runs** (by era ~50)
- **Symbolic communication in only 9% of runs** (by era ~937)
- The index-to-symbol transition took 18x longer than nothing-to-index
- Our simulation could be the second to demonstrate this computationally

### What Can Be Stripped (start simple)
- Complex terrain, multiple predator types, communication, speciation, crossover

### What Cannot Be Stripped
- Vision/sensing, energy/metabolism, population pressure (death)

### Design Rules
- Scripted predators correct for communication-focused project
- Prey should be slightly faster than predators
- Predator-to-prey ratio: 5:1 to 20:1
- Immortal scripted predators are fine (literature validated)

---

## Semiotic Measurement Framework

For a paper about emergent semiotics (not just "communication evolves"), demonstrate the Peircean sign hierarchy:
1. **Icon**: signal resembles referent (strength proportional to danger proximity)
2. **Index**: signal causally correlated with referent (symbol X always near predator)
3. **Symbol**: arbitrary but conventional association (symbol 3 = "aerial" with no physical reason)

### Core Metrics (per generation)
| Metric | What it measures | Meaningful threshold |
|--------|-----------------|---------------------|
| Symbol-context matrix (8xK) | P(symbol \| context) - the emergent lexicon | Track over time |
| MI(Symbol; Context) | Information content of signals | > 0.05 = indexical, > 0.20 = symbolic |
| TopSim | Compositionality (similar situations -> similar signals) | > 0.3 = compositional |
| Lexicon stability (KL div) | Meaning change rate across generations | Decreasing over time = stabilizing |
| Iconicity score | Signal intensity correlation with stimulus proximity | Decreasing while MI stays high = icon->symbol transition |
| Arbitrariness (cross-run) | Do different seeds assign different symbols to same contexts? | Different = true symbols |

### Per-signal-event data required
Emitter ID, position, energy, species ID, symbol, nearest predator (type+distance), tick, list of receiver IDs, each receiver's subsequent action (within 3 ticks) and survival (next 10 ticks).

### The Publishable Result
Show: (1) MI increases from 0, (2) signals initially iconic/indexical, (3) iconicity drops while MI stays high (symbol emergence), (4) different runs produce different conventions (arbitrariness), (5) conventions stabilize, (6) receivers respond appropriately. This is the complete Peircean trajectory.

---

## Sources

### Core Papers
- Stanley 2002 (NEAT): https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
- Reggia et al. 2001 (communication conditions): https://direct.mit.edu/artl/article/7/1/3/2365/
- Mitri et al. 2009 (deception): https://www.pnas.org/doi/10.1073/pnas.0903152106
- Mitri et al. 2011 (relatedness thresholds): https://pmc.ncbi.nlm.nih.gov/articles/PMC3013414/
- Taylor 1992 (kin competition cancellation): viscous population kin selection limits
- Beer & Williams 2019 (signal delay): https://direct.mit.edu/artl/article/25/4/315/93258/
- Lowe et al. 2019 (communication measurement pitfalls): https://arxiv.org/abs/1903.05168
- Li et al. 2023 (swarming from survival): https://arxiv.org/abs/2308.12624

### Energy and Fitness
- Waibel et al. 2011 (Hamilton's rule test): https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1000615
- Torney et al. 2011 (signal cost calibration): https://pmc.ncbi.nlm.nih.gov/articles/PMC3178622/
- DeepMind specification gaming: https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/

### Curriculum and Progressive Difficulty
- Milano & Nolfi 2021 (neuroevolution curriculum): https://pmc.ncbi.nlm.nih.gov/articles/PMC8076209/
- Gomez & Miikkulainen 1997 (incremental evolution): https://journals.sagepub.com/doi/10.1177/105971239700500305
- OpenAI Hide-and-Seek (autocurricula): https://arxiv.org/abs/1909.07528
- POET (environment co-evolution): https://arxiv.org/abs/1901.01753

### Communication/Semiotics
- Grouchy et al. 2016 (icon-to-symbol): https://www.nature.com/articles/srep34615
- Cangelosi & Parisi 1998 (mushroom signals): https://langev.com/pdf/cangelosi01evolutionOf.pdf
- Mordatch & Abbeel 2018 (compositional language): https://arxiv.org/abs/1703.04908
- Steels 1995 (naming game): https://www.academia.edu/2936171/The_naming_game
- Witkowski & Ikegami 2016 (swarming+signals): https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152756
- Lazaridou et al. 2017 (multi-agent language): https://arxiv.org/abs/1612.07182

### Reference Implementations
- biosim4: https://github.com/davidrmiller/biosim4
- NEAT-Python: https://neat-python.readthedocs.io/en/latest/
- SharpNEAT: https://sharpneat.sourceforge.io/phasedsearch.html
- The Bibites: https://the-bibites.fandom.com/
- Hunting for Insights: https://arvind6599.github.io/PredatorPreyWebsite/
- Avida: https://alife.org/encyclopedia/digital-evolution/avida/
- Aquarium MARL: https://arxiv.org/html/2401.07056v1
