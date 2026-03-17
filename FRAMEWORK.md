# A Framework for Semiotic Emergence in Minimal Worlds

This document is the governing intellectual framework for the project. Every change to the simulation - new metrics, parameter tuning, architectural decisions - should be evaluated against the questions and principles here. The simulation is a tool for exploring these ideas; the ideas are not decorations on the simulation.

## The core question

Every existing theory of signs starts from a world where meaning already exists and asks how it works. Peirce classifies signs that are already signs. Saussure analyzes a language that's already a language. Even biosemiotics begins from cells that already interpret molecules. The theoretical apparatus is built for the middle of the story.

We don't have good theory for the beginning - the moment when the universe contained no meaning and then, for the first time, it did. This moment is empirically inaccessible in biology. It happened deep in evolutionary time and left no fossil record.

This simulation is a telescope pointed at exactly that gap. Hundreds of neural networks on a toroidal grid, under invisible lethal pressure, with a 6-symbol broadcast channel. At generation 0, signals are noise. If meaning emerges at all, it emerges here, and we can watch every step of it.

**The question is not "do prey evolve alarm calls?" That's a biology question with known answers. The question is: what does it look like when meaning comes into existence for the first time in a world that previously contained none?**

## What meaning requires

Strip away consciousness, intention, culture, syntax, shared history. Three things remain:

**Differential production.** The signal varies with context. If the sender emits the same signal regardless of world state, it carries no information. This is the sender side. Mutual information measures it.

**Differential reception.** The receiver acts differently depending on the signal. If the receiver responds identically to all signals or ignores them entirely, the signal makes no difference in the world. This is the receiver side. Receiver JSD measures it - comparing action distributions with vs without signal, split by context.

**Fitness coupling.** The differential production and reception are linked to outcomes. The combination must matter for survival. Otherwise the pattern is an epiphenomenon - two clocks synchronized by coincidence, not connected by mechanism. The evolutionary feedback loop provides this coupling: signal-response patterns that improve survival get reinforced.

What meaning does NOT require: consciousness, intention, shared knowledge, convention. These are features of mature semiotic systems, not prerequisites. A thermostat differentially responds to temperature, but nothing in the system benefits from that response in a way that feeds back into maintaining it. In the simulation, natural selection provides exactly this feedback. Meaning maintains itself.

## The pre-semiotic zone

Between pure noise and genuine meaning, there is a region with no good name. These are the forms that occupy it:

**Spandrels.** Signals that correlate with world states as a side effect of neural architecture. In a single-layer network, the same weights that drive fleeing happen to drive signaling because the network cannot isolate signal outputs from movement outputs. The split-head architecture (v2) addresses this by routing signals through a dedicated hidden layer, but spandrels can still arise through the shared base layer. With invisible kill zones, the spandrel risk shifts: prey can't see zones, so there's no direct sensory input to leak through shared weights. Any zone-correlated signaling must arise through energy-level encoding or social relay - both more indicative of genuine information processing.

**Ghosts.** Signal-context correlations that exist in the observer's analysis but not in the agents' behavior. When we compute mutual information using zone distance bins, we are the ones creating the meaning. The prey might be encoding something entirely different, or nothing at all. The meaning is in our analysis pipeline, not in the world.

**Fossils.** Signal patterns that were once functional but have been disconnected from their original context through genetic drift. Vestigial meaning.

**Seeds.** Signal-response patterns too weak to be detected by current metrics but potentially amplifiable by selection. The precursor to meaning, not yet meaning itself.

Mutual information alone cannot distinguish between these. The receiver-side instruments (receiver JSD, silence onset, three-way coupling) help separate genuine communication from spandrels and ghosts by checking whether receivers actually change behavior in response to signals, and whether that change correlates with survival.

## The observer's Umwelt

We inhabit a fundamentally different semiotic space than the prey. We see the entire grid, all positions, all statistics. We compute cross-generational trends. The prey see 39 floating-point numbers.

When we say "symbol 0 means danger-nearby," we are translating from their Umwelt into ours. But what does symbol 0 mean in their Umwelt? Not "danger nearby" - they don't have that concept. It means: a specific activation pattern in a 5683-weight split-head network produced this output given these inputs. That is what it "means" inside the network.

Any claim about what signals mean is a translation, and translation always loses something. We project our categories (zone, distance, danger) onto their signal-context relationships. The actual semiotic structure of the prey's world might be organized along entirely different dimensions.

This is why unsupervised analysis matters. Instead of asking "do signals encode zone distance?" (our question, our categories), ask "what structure exists in the relationship between signals, contexts, and responses?" and let the patterns reveal whatever categories the prey have actually evolved.

The neural network's 39 inputs include 3 body-state inputs (zone damage accumulation, energy delta, freeze pressure - the prey feel pain but can't see its source), food direction and distance, ally direction and distance, 18 signal channels (6 symbols x strength/direction), 8 recurrent memory cells, energy level, and 3 death witness inputs (intensity, direction to nearest recent zone death). Signal outputs pass through a dedicated signal hidden layer, giving evolution capacity for independent signal control. Our mutual information metric only checks correlation with one dimension (zone distance - an observer metric, not a prey input). The prey might be signaling about energy level, food proximity, ally density, memory state, the incoming signals themselves, death witness information, or some nonlinear combination that doesn't decompose into any single variable. Input MI - I(Symbol; X_i) for each of the 39 input dimensions - partially addresses this by letting the data reveal which dimensions signals actually correlate with. But it still decomposes into single variables; nonlinear combinations remain in the dark.

The invisibility of kill zones deepens the observer's Umwelt problem. We measure MI against zone distance because *we* can see the zones. The prey cannot. If signals correlate with zone distance, that correlation must be mediated through observable channels - energy loss, received signals from other prey, or memory of recent energy trajectories. This indirection makes any observed MI more meaningful: it can't be a trivial spandrel of shared sensory input.

The earliest forms of meaning might be self-referential: "I'm scared" rather than "there's a predator." Both carry information about the predator, but the semiotic structure is completely different.

## Silence as the first sign

With visible predators, we discovered that prey go silent near danger. We initially treated this as a failure. It might be the most important finding of the early runs.

Silence is the simplest possible sign. It requires no coordination between sender and receiver about what specific symbols mean. It requires only that the receiver is calibrated to a baseline rate of signaling. When signals are constant and then stop, the absence carries information.

The evolutionary path to silence is easy. It doesn't require the bootstrap. A sender that stops signaling near danger needs no complementary receiver mutation - any receiver that notices the decrease in ambient signal strength gets free information. The receiver doesn't need to "understand" the silence. It just needs to notice the statistical anomaly.

This might be semiotic ground zero - the first meaning that can emerge without solving the bootstrap problem. And if silence is functional (if receivers respond to it), then it creates a platform for more complex signaling. The population has crossed the boundary from noise into structured absence. From structured absence, it's a shorter step to structured presence than from pure noise.

With invisible kill zones, the dynamics shifted. Early observations show positive iconicity - prey signal *more* inside zones, not less. This makes evolutionary sense: when you can't see the threat, broadcasting distress is more valuable than going silent. Silence was the optimal strategy when prey could see danger directly and had nothing to gain from alerting neighbors. When danger is invisible, the calculus inverts - the endangered prey benefits from eliciting directional information from others, and neighbors benefit from knowing where the danger is.

Whether silence or alarm calling stabilizes as the dominant strategy over long evolutionary runs remains an open question. The answer may depend on whether the population discovers that signals can carry directional information (pointing away from zones) or merely presence information (danger exists somewhere nearby).

## The semiotic landscape

The genome defines a point in high-dimensional weight space (5683 dimensions in the current architecture). Different regions produce agents with different semiotic configurations. This is not a metaphor - the weight space is literally a latent space, and its topology determines what meaning-making systems are possible.

**Basins.** Regions where populations converge on stable signal-meaning mappings. Different basins represent different possible communication systems.

**Barriers.** Weight-space regions separating one communication system from another. How high these are determines whether evolution can switch between systems or gets stuck.

**The noise floor.** The vast region where signals are pure noise. Most of the space. Evolution must find its way out of this region to reach any semiotic basin.

**Attractors.** Certain communication systems that evolution gravitates toward regardless of starting conditions. If all populations from different seeds converge on the same system, there's a dominant attractor. If they find different systems, the landscape has multiple attractors and outcomes are path-dependent - contingent, conventional, semiotically sticky.

Running multiple populations from different seeds samples this landscape. Convergence tells us what's constrained by the world. Divergence tells us what's invented by the population.

**The semiotic landscape is now 5683-dimensional** (up from 158 in early runs). The split-head architecture adds two independent hidden size genes that control the effective dimensionality: a population with base_hidden=4 and signal_hidden=6 uses a different region of weight space than one with base_hidden=12 and signal_hidden=2, even if total capacity is similar. Brain compression (evolution shrinking hidden layers to save metabolic cost) is itself a semiotic phenomenon - the population is discovering which dimensions of the landscape matter.

## The bootstrap as phase transition

The bootstrap problem - no sender without receiver, no receiver without sender - might be better understood as a phase transition.

Below some critical level of genetic similarity, spatial proximity, or selection pressure, signal-response correlations are random fluctuations that don't persist. Above it, they lock in and become self-reinforcing. Spatial reproduction is the mechanism for local correlation: offspring inherit positions near their parents, creating natural kin clusters where genetically similar individuals share overlapping signal ranges. No artificial grouping is needed - the spatial structure itself generates the conditions for complementary sender-receiver weights to co-occur.

If this is right, the emergence of meaning should show phase transition signatures: increasing fluctuations before the transition, a sharp change at the critical point, and hysteresis (once established, the system persists even if conditions partially revert).

Deme-based group selection provides a second mechanism for crossing the phase boundary. When the grid is divided into demes (rectangular subpopulations), evolution operates at two levels: individual fitness within demes, and deme-level selection between demes. Demes where communication emerges outcompete silent demes, providing group-level reinforcement that individual selection alone cannot. Migration between adjacent demes prevents complete isolation while preserving deme identity. This multi-level selection is the theoretical mechanism most likely to overcome the bootstrap problem - it rewards the sender-receiver pair rather than punishing the altruistic sender.

## What makes a meaning-making system survive

Some populations might evolve communication that persists for hundreds of generations. Others might evolve it briefly and lose it. The difference tells us about semiotic resilience.

**Redundancy.** Systems where multiple signals carry overlapping information are more robust than single-point-of-failure systems.

**Integration depth.** If the signal-response mapping is deeply woven into survival behavior rather than a separate module, it's harder for drift to dislodge.

**Symmetry.** Systems where the sender benefits as much as the receiver are stickier than altruistic systems. Selfish communication is more durable than generous communication.

**Population structure.** Meaning-making maintained by spatial kin clusters may be more fragile than population-wide convention, because local clusters are vulnerable to demographic fluctuation and zone pressure. But spatial reproduction continuously regenerates kin proximity, which may provide more durable support than artificial grouping. Deme structure adds a second layer: communication systems that help entire demes survive get reinforced through group selection, even if individual signalers pay a cost.

## Dark semiotics

The semiotic processes happening in the simulation that our metrics cannot see.

**Behavioral semiotics.** A prey fleeing is a signal to other prey, even without the signal channel. A sudden increase in nearest-ally distance means someone fled, which means danger. This is semiotic activity in the movement channel, completely unmeasured.

**Temporal semiotics.** Not what symbol is emitted, but when. A burst of signals might mean something different from a steady trickle. The rhythm of signaling over ticks might carry information that per-signal metrics miss.

**Negative semiotics.** Silence. The absence of a signal in a context where signals are normally expected. The system is meaningful precisely because it can be not-used. Silence is only meaningful against a background of expected speech.

**Relational semiotics.** Not what a signal means in isolation, but relative to other signals. If symbol 0 is usually emitted at distance 3 and symbol 1 at distance 7, the contrast carries meaning even if neither one alone does.

**Social semiotics.** Differences in signaling between spatially proximate kin clusters and interactions with distant non-relatives. If prey signal differently when surrounded by genetic relatives (which spatial reproduction creates naturally), that's audience-dependent communication.

## The instruments

The original five measurements have grown into a fuller observatory as the project matured. They organize by the question they answer.

### What are signals encoding?

**Mutual information.** I(Signal; ZoneDistance) - the foundational sender-side metric. But it uses observer-imposed bins, measuring correlation with *our* categories. **Input MI** addresses this by computing I(Symbol; X_i) for all 39 input dimensions without imposing categories - letting the data reveal what signals actually encode. **Signal entropy** (Shannon entropy of symbol frequencies) captures pre-semiotic organization that MI misses: a population converging on specific symbols even before those symbols carry measurable information. **Iconicity** measures whether signal rate rises or falls inside zones - alarm calling vs silence.

### How do receivers respond?

**Receiver response spectrum.** JSD between action distributions with and without signal, split by context. The core receiver-side metric. **Silence onset** detects whether receivers change behavior when signals vanish - increased movement at silence onset means signals were functioning as "all clear." **Contrast telescope** (pairwise inter-symbol JSD) checks whether different symbols carry different information - the prerequisite for functional reference.

### Does the signal-response loop affect fitness?

**Fitness coupling.** Correlation between signaling rate and survival (sender side) and between signal reception and survival (receiver side). **Response fitness correlation** is the cleanest: does *responding differently* to signals predict survival? A prey that hears but ignores signals is its own control.

**Counterfactual value.** The definitive test. Paired simulations from the same seed, one with signals, one without. The fitness divergence is the total value of the semiotic system. Known confounds bound interpretation: mute prey save emission energy (~62% of base metabolic drain), creating a cost advantage; signal emission changes the energy input, creating a feedback loop absent in the mute run; identical seeds diverge within ticks as different survival patterns alter the RNG stream. A small baseline advantage could be cost-offset. A large advantage is strong evidence.

### How does the system evolve?

**Semiotic trajectory.** The signal-context mapping tracked over evolutionary time. Phase transitions appear as sudden shifts in trajectory JSD. **Phase transition detection** (rolling fluctuation ratio) identifies destabilization before reorganization. **Cross-population divergence** compares signal-meaning mappings across seeds: high divergence means convention, low means constraint.

## Hierarchy of semiotic phenomena

Not a ladder to climb, but a map of what could exist in this world:

**Level 0 - Cue.** Information exists but nobody produced it for anyone. A prey's position is a cue. Coordination without communication.

**Level 1 - Index.** Signal correlates with world state because of shared causal origin. The same network activation that drives fleeing also drives signaling. A side effect, not a message. This is what we currently observe.

**Level 2 - Ritualized signal.** A side effect that has been enhanced by selection for its communicative function. More consistent, more conspicuous, more costly than the original behavior warrants.

**Level 3 - Functional reference.** Different signals trigger different, appropriate receiver behaviors. Not just "alert vs. calm" but specific, differentiated responses. This is what our 6-symbol system with dedicated signal processing is architecturally capable of.

**Level 4 - Convention.** The mapping between signal and referent is arbitrary - it could have been otherwise. Testable by comparing across independently evolved populations.

**Levels 5+ - Compositionality, displacement, metalanguage.** Beyond what fixed-topology networks can achieve. Not a target for this simulation.

## The evidence hierarchy

The semiotic phenomena hierarchy describes what *could* exist. This hierarchy describes what we need to *measure* to know we've found it. Each level builds on the previous - skipping ahead produces ambiguous results.

| Level | Claim | Evidence needed |
|-------|-------|----------------|
| 1 | Signals have adaptive value | Baseline vs mute fitness divergence |
| 2 | Receivers change behavior | JSD(action\|signal) vs JSD(action\|no signal) > 0 |
| 3 | Different symbols carry different info | Inter-symbol JSD > 0, input MI shows symbol-specific encoding |
| 4 | Responses are appropriate | MI(symbol; action \| context) > 0 |
| 5 | Genuine reference | Stable symbol-referent mapping, robust to perturbation |

Level 1 is the prerequisite. Without it, signal-response patterns could be spandrels - correlated with fitness but not causal. The counterfactual experiment isolates causation by removing the signal channel and measuring whether the evolutionary trajectory changes.

Early data shows promising Level 2/3 signals (receiver JSD ~6x baseline, sender fitness correlation -0.39 / receiver fitness correlation +0.75, consistent with costly kin-selected signaling). But without Level 1, these remain ambiguous.

The gap between Level 3 and Level 4 is the gap between information and meaning. A signal carrying different information than another (Level 3) might not elicit the *right* response. Level 4 conditions on context: does the receiver take the survival-improving action for *this* signal in *this* situation?

Level 5 is the project's ultimate question: reference emerging from scratch in a system simple enough to fully inspect.

## Principles for development

Every change to the simulation should be evaluated against these questions:

1. **Does this change help us see something we couldn't see before?** New metrics, new logging, new analysis tools are valuable. Complexity for its own sake is not.

2. **Are we measuring what the prey are doing, or what we think they should be doing?** Beware of imposing our categories. Let the data reveal its structure.

3. **Does this change preserve the simplicity that makes the system legible?** The simulation's value is that we can inspect every component. Adding complexity trades legibility for realism. Prefer legibility.

4. **Are we looking at the sender, the receiver, or the relationship?** The relationship is where meaning lives. Sender-only metrics (MI, iconicity) are necessary but insufficient.

5. **Would this change help us detect meaning we aren't expecting?** The most interesting findings will be things we didn't predict. Design instruments for surprise.
