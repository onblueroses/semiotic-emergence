# A Framework for Semiotic Emergence in Minimal Worlds

This document is the governing intellectual framework for the project. Every change to the simulation - new metrics, parameter tuning, architectural decisions - should be evaluated against the questions and principles here. The simulation is a tool for exploring these ideas; the ideas are not decorations on the simulation.

## The core question

Every existing theory of signs starts from a world where meaning already exists and asks how it works. Peirce classifies signs that are already signs. Saussure analyzes a language that's already a language. Even biosemiotics begins from cells that already interpret molecules. The theoretical apparatus is built for the middle of the story.

We don't have good theory for the beginning - the moment when the universe contained no meaning and then, for the first time, it did. This moment is empirically inaccessible in biology. It happened deep in evolutionary time and left no fossil record.

This simulation is a telescope pointed at exactly that gap. Forty-eight neural networks on a toroidal grid, under predation pressure, with a 3-symbol broadcast channel. At generation 0, signals are noise. If meaning emerges at all, it emerges here, and we can watch every step of it.

**The question is not "do prey evolve alarm calls?" That's a biology question with known answers. The question is: what does it look like when meaning comes into existence for the first time in a world that previously contained none?**

## What meaning requires

Strip away consciousness, intention, culture, syntax, shared history. Three things remain:

**Differential production.** The signal varies with context. If the sender emits the same signal regardless of world state, it carries no information. This is the sender side. Mutual information measures it.

**Differential reception.** The receiver acts differently depending on the signal. If the receiver responds identically to all signals or ignores them entirely, the signal makes no difference in the world. This is the receiver side. Receiver JSD measures it - comparing action distributions with vs without signal, split by context.

**Fitness coupling.** The differential production and reception are linked to outcomes. The combination must matter for survival. Otherwise the pattern is an epiphenomenon - two clocks synchronized by coincidence, not connected by mechanism. The evolutionary feedback loop provides this coupling: signal-response patterns that improve survival get reinforced.

What meaning does NOT require: consciousness, intention, shared knowledge, convention. These are features of mature semiotic systems, not prerequisites. A thermostat differentially responds to temperature, but nothing in the system benefits from that response in a way that feeds back into maintaining it. In the simulation, natural selection provides exactly this feedback. Meaning maintains itself.

## The pre-semiotic zone

Between pure noise and genuine meaning, there is a region with no good name. These are the forms that occupy it:

**Spandrels.** Signals that correlate with world states as a side effect of neural architecture. The same weights that drive fleeing happen to drive signaling because the network is small (6 hidden neurons) and cannot isolate signal outputs from movement outputs. The correlation is real but accidental - an architectural artifact, not a semiotic achievement.

**Ghosts.** Signal-context correlations that exist in the observer's analysis but not in the agents' behavior. When we compute mutual information using predator distance bins, we are the ones creating the meaning. The prey might be encoding something entirely different, or nothing at all. The meaning is in our analysis pipeline, not in the world.

**Fossils.** Signal patterns that were once functional but have been disconnected from their original context through genetic drift. Vestigial meaning.

**Seeds.** Signal-response patterns too weak to be detected by current metrics but potentially amplifiable by selection. The precursor to meaning, not yet meaning itself.

Mutual information alone cannot distinguish between these. The receiver-side instruments (receiver JSD, silence onset, three-way coupling) help separate genuine communication from spandrels and ghosts by checking whether receivers actually change behavior in response to signals, and whether that change correlates with survival.

## The observer's Umwelt

We inhabit a fundamentally different semiotic space than the prey. We see the entire grid, all positions, all statistics. We compute cross-generational trends. The prey see 16 floating-point numbers.

When we say "symbol 0 means predator-nearby," we are translating from their Umwelt into ours. But what does symbol 0 mean in their Umwelt? Not "predator nearby" - they don't have that concept. It means: a specific activation pattern in a 158-weight network produced this output given these inputs. That is what it "means" inside the network.

Any claim about what signals mean is a translation, and translation always loses something. We project our categories (predator, distance, danger) onto their signal-context relationships. The actual semiotic structure of the prey's world might be organized along entirely different dimensions.

This is why unsupervised analysis matters. Instead of asking "do signals encode predator distance?" (our question, our categories), ask "what structure exists in the relationship between signals, contexts, and responses?" and let the patterns reveal whatever categories the prey have actually evolved.

The neural network's 16 inputs include predator proximity, food direction, ally distance, incoming signals, and energy level. Signal outputs are driven by all of them. Our mutual information metric only checks correlation with one dimension (predator distance). The prey might be signaling about energy level, food proximity, ally density, the incoming signals themselves, or some nonlinear combination that doesn't decompose into any single variable.

The earliest forms of meaning might be self-referential: "I'm scared" rather than "there's a predator." Both carry information about the predator, but the semiotic structure is completely different.

## Silence as the first sign

We discovered that prey go silent near the predator. We initially treated this as a failure. It might be the most important finding.

Silence is the simplest possible sign. It requires no coordination between sender and receiver about what specific symbols mean. It requires only that the receiver is calibrated to a baseline rate of signaling. When signals are constant and then stop, the absence carries information.

The evolutionary path to silence is easy. It doesn't require the bootstrap. A sender that stops signaling near the predator needs no complementary receiver mutation - any receiver that notices the decrease in ambient signal strength gets free information. The receiver doesn't need to "understand" the silence. It just needs to notice the statistical anomaly.

This might be semiotic ground zero - the first meaning that can emerge without solving the bootstrap problem. And if silence is functional (if receivers respond to it), then it creates a platform for more complex signaling. The population has crossed the boundary from noise into structured absence. From structured absence, it's a shorter step to structured presence than from pure noise.

Or silence might be the stable attractor. The simplest meaning-making system in this world might not be alarm calls but their inverse: a background hum that goes quiet when danger arrives. Not "I'm screaming fire" but "I've stopped humming." This is a meaning-making system that nobody has looked for because everyone was looking for alarm calls.

## The semiotic landscape

The 158-weight genome defines a point in 158-dimensional space. Different regions produce agents with different semiotic configurations. This is not a metaphor - the weight space is literally a latent space, and its topology determines what meaning-making systems are possible.

**Basins.** Regions where populations converge on stable signal-meaning mappings. Different basins represent different possible communication systems.

**Barriers.** Weight-space regions separating one communication system from another. How high these are determines whether evolution can switch between systems or gets stuck.

**The noise floor.** The vast region where signals are pure noise. Most of the space. Evolution must find its way out of this region to reach any semiotic basin.

**Attractors.** Certain communication systems that evolution gravitates toward regardless of starting conditions. If all populations from different seeds converge on the same system, there's a dominant attractor. If they find different systems, the landscape has multiple attractors and outcomes are path-dependent - contingent, conventional, semiotically sticky.

Running multiple populations from different seeds samples this landscape. Convergence tells us what's constrained by the world. Divergence tells us what's invented by the population.

## The bootstrap as phase transition

The bootstrap problem - no sender without receiver, no receiver without sender - might be better understood as a phase transition.

Below some critical level of genetic similarity, spatial proximity, or selection pressure, signal-response correlations are random fluctuations that don't persist. Above it, they lock in and become self-reinforcing. Spatial reproduction is the mechanism for local correlation: offspring inherit positions near their parents, creating natural kin clusters where genetically similar individuals share overlapping signal ranges. No artificial grouping is needed - the spatial structure itself generates the conditions for complementary sender-receiver weights to co-occur.

If this is right, the emergence of meaning should show phase transition signatures: increasing fluctuations before the transition, a sharp change at the critical point, and hysteresis (once established, the system persists even if conditions partially revert).

## What makes a meaning-making system survive

Some populations might evolve communication that persists for hundreds of generations. Others might evolve it briefly and lose it. The difference tells us about semiotic resilience.

**Redundancy.** Systems where multiple signals carry overlapping information are more robust than single-point-of-failure systems.

**Integration depth.** If the signal-response mapping is deeply woven into survival behavior rather than a separate module, it's harder for drift to dislodge.

**Symmetry.** Systems where the sender benefits as much as the receiver are stickier than altruistic systems. Selfish communication is more durable than generous communication.

**Population structure.** Meaning-making maintained by spatial kin clusters may be more fragile than population-wide convention, because local clusters are vulnerable to demographic fluctuation and predator pressure. But spatial reproduction continuously regenerates kin proximity, which may provide more durable support than artificial grouping.

## Dark semiotics

The semiotic processes happening in the simulation that our metrics cannot see.

**Behavioral semiotics.** A prey fleeing is a signal to other prey, even without the signal channel. A sudden increase in nearest-ally distance means someone fled, which means danger. This is semiotic activity in the movement channel, completely unmeasured.

**Temporal semiotics.** Not what symbol is emitted, but when. A burst of signals might mean something different from a steady trickle. The rhythm of signaling over ticks might carry information that per-signal metrics miss.

**Negative semiotics.** Silence. The absence of a signal in a context where signals are normally expected. The system is meaningful precisely because it can be not-used. Silence is only meaningful against a background of expected speech.

**Relational semiotics.** Not what a signal means in isolation, but relative to other signals. If symbol 0 is usually emitted at distance 3 and symbol 1 at distance 7, the contrast carries meaning even if neither one alone does.

**Social semiotics.** Differences in signaling between spatially proximate kin clusters and interactions with distant non-relatives. If prey signal differently when surrounded by genetic relatives (which spatial reproduction creates naturally), that's audience-dependent communication.

## The five instruments

The framework points to five measurements the simulation needs:

**1. Receiver response spectrum.** For each signal-context pair, what does the receiver do? Compare against the null (same context, no signal). The difference is the semiotic effect.

**2. Silence detection.** Measure signal rate over time within a run. When does it drop? Does the drop correlate with predator proximity to any prey? Do receivers near the silent zone behave differently?

**3. Semiotic trajectory.** Track the signal-context-response mapping over evolutionary time. Not snapshots but the full trajectory. Look for phase-transition signatures.

**4. Cross-population divergence.** Run N populations from different seeds. Compare signal-meaning mappings at generation G. High divergence means convention. Low divergence means constraint.

**5. Counterfactual value.** Run paired simulations: one with signals active, one with them disabled. Same seed, same conditions. The fitness difference is the value of the entire semiotic system.

## Hierarchy of semiotic phenomena

Not a ladder to climb, but a map of what could exist in this world:

**Level 0 - Cue.** Information exists but nobody produced it for anyone. A prey's position is a cue. Coordination without communication.

**Level 1 - Index.** Signal correlates with world state because of shared causal origin. The same network activation that drives fleeing also drives signaling. A side effect, not a message. This is what we currently observe.

**Level 2 - Ritualized signal.** A side effect that has been enhanced by selection for its communicative function. More consistent, more conspicuous, more costly than the original behavior warrants.

**Level 3 - Functional reference.** Different signals trigger different, appropriate receiver behaviors. Not just "alert vs. calm" but specific, differentiated responses. This is what our 3-symbol system is architecturally capable of.

**Level 4 - Convention.** The mapping between signal and referent is arbitrary - it could have been otherwise. Testable by comparing across independently evolved populations.

**Levels 5+ - Compositionality, displacement, metalanguage.** Beyond what 158-weight fixed-topology networks can achieve. Not a target for this simulation.

## Principles for development

Every change to the simulation should be evaluated against these questions:

1. **Does this change help us see something we couldn't see before?** New metrics, new logging, new analysis tools are valuable. Complexity for its own sake is not.

2. **Are we measuring what the prey are doing, or what we think they should be doing?** Beware of imposing our categories. Let the data reveal its structure.

3. **Does this change preserve the simplicity that makes the system legible?** The simulation's value is that we can inspect every component. Adding complexity trades legibility for realism. Prefer legibility.

4. **Are we looking at the sender, the receiver, or the relationship?** The relationship is where meaning lives. Sender-only metrics (MI, iconicity) are necessary but insufficient.

5. **Would this change help us detect meaning we aren't expecting?** The most interesting findings will be things we didn't predict. Design instruments for surprise.
