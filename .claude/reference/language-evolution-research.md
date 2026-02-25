# Language Evolution Research for Predator-Prey Communication Simulation

Structured literature review covering computational models of language evolution, emergent
communication, and measurement methodology. Written for the predator-prey-evolution-communication
project (NEAT neuroevolution, 8 discrete symbols, 3 predator types, spatial kin structure).

Updated: 2026-02-25

---

## Table of Contents

1. Lewis Signaling Games
2. Iterated Learning Model (Kirby, Edinburgh)
3. Naming Games (Steels, Baronchelli)
4. Referential Games in Deep Learning
5. What Our Simulation Adds vs Existing Work
6. Metrics for Language Properties (Hockett's Design Features)
7. Paper Strategy: Venues, Narrative, Surprising Results

---

## 1. Lewis Signaling Games

### What They Are

David Lewis (1969, "Convention") formalized the simplest signaling scenario as a coordination
game between a Sender and a Receiver:

1. Nature selects a **state** (e.g., predator type) from a set of possible states.
2. The **Sender** observes the state and chooses a **signal** from a fixed vocabulary.
3. The **Receiver** observes the signal (not the state) and chooses an **action**.
4. Both players receive a payoff that depends on whether the action matches the state.

A **signaling system** is a strategy pair where every state maps to a unique signal, and
every signal maps to the correct action - a perfect bijection between states and responses.

### How This Maps to Our Simulation

Our setup is a multi-agent, multi-state Lewis game:

| Lewis Game Component | Our Simulation |
|---------------------|----------------|
| State | PredatorKind (Aerial, Ground, Pack) + distance/direction |
| Sender | Prey that detects a predator and emits a Symbol(0..7) |
| Signal | One of 8 discrete symbols |
| Receiver | Nearby prey within hearing_range (12 cells) |
| Action | Flee, climb, hide, scatter (context-dependent survival) |
| Payoff | Survival (fitness) - both sender and receiver benefit |

Key difference from classic Lewis: we have **common interest** (both players benefit from
correct coordination) and signals are **costly** (3.0 energy per emission), which addresses
the cheap-talk problem that makes Lewis games trivially uninteresting without cost.

### Equilibrium Structure

**Signaling system equilibria** - In the standard Lewis game with N states, N signals, and
N acts (where N matches), there are N! signaling system equilibria (every permutation of
signal-to-state mappings works equally well). With 3 predator types and 8 symbols, there
are 8 * 7 * 6 = 336 possible injective mappings from predator types to symbols.

**Pooling equilibria** - There are always equilibria where the sender ignores the state and
the receiver ignores the signal. These are bad (no information transfer) but they are Nash
equilibria. In evolutionary dynamics, pooling equilibria are unstable under most dynamics
but can be sticky traps in finite populations.

**Partial pooling** - Some states share a signal while others don't. Example: using one
symbol for "aerial predator" and another for "ground OR pack predator." These equilibria
are intermediate between full signaling and full pooling. Our simulation could settle here
if two predator types require similar responses (both ground and pack might warrant hiding
behind rocks).

### Evolutionary Dynamics (Skyrms, 2010)

Brian Skyrms' "Signals: Evolution, Learning, and Information" is the key reference for
evolutionary analysis of Lewis games:

- Under **replicator dynamics** (continuous, infinite population), signaling system equilibria
  are the only **attractors** in N-state, N-signal, N-act games. Their combined basins of
  attraction cover almost all initial conditions. This is excellent news: evolution almost
  certainly converges to some signaling system.

- Under **replicator-mutator dynamics** (with mutation), signaling systems are still
  attractors but convergence is slower and the system can temporarily get stuck in partial
  pooling.

- In **finite populations** (our case: 150 prey), stochastic effects matter. Drift can
  push the population away from signaling systems. However, the strong selection pressure
  from predator kills should dominate drift at N=150.

- **Multiple senders and receivers** (our case): When multiple senders can observe the same
  state, there's potential for redundancy. Skyrms (with collaborators) showed that with
  multiple senders, stable equilibria emerge where different senders specialize or provide
  redundant signals. In our simulation, multiple prey near the same predator all sense it
  and all can signal - the question is whether they converge on the same symbol for the
  same predator type.

### Key Prediction for Our Simulation

Game theory predicts that *some* signaling system will emerge (under reasonable evolutionary
dynamics), but not *which* one. The specific mapping of symbols to predator types is
arbitrary - what matters is convergence. The simulation should show:

1. Initial random signaling (pooling or near-pooling).
2. A transition period where partial pooling forms.
3. Convergence to a signaling system (possibly with unused symbols).

The timescale of convergence depends on selection pressure strength, population size, and
mutation rate. With strong predator pressure, it should be fast (10s-100s of generations,
not 1000s).

### Key Papers

- Lewis, D. (1969). *Convention: A Philosophical Study*. Harvard University Press.
- Skyrms, B. (2010). [*Signals: Evolution, Learning, and Information*](https://sites.socsci.uci.edu/~bskyrms/bio/books/signals.pdf). Oxford University Press.
- Huttegger, S., Skyrms, B., Smead, R., & Zollman, K. (2010). [Evolutionary dynamics of Lewis signaling games](https://bpb-us-e2.wpmucdn.com/faculty.sites.uci.edu/dist/c/190/files/2011/03/Hutteggeral2009.pdf). *Synthese*, 172, 177-191.
- Skyrms, B. (2014). [Some dynamics of signaling games](https://www.pnas.org/doi/10.1073/pnas.1400838111). *PNAS*.
- Skyrms, B. & Barrett, J. [Evolution of Signaling Systems with Multiple Senders and Receivers](https://sites.socsci.uci.edu/~bskyrms/bio/papers/signalingwithmultiplesendersandreceivers.pdf).

---

## 2. Iterated Learning Model (Kirby, Edinburgh)

### What It Is

The **Iterated Learning Model** (ILM), developed by Simon Kirby and colleagues at the
University of Edinburgh, studies how language changes when it is transmitted from one
generation of learners to the next. The core setup:

1. Generation N has a language (mapping from meanings to signals).
2. Generation N produces *utterances* - a sample of meaning-signal pairs.
3. Generation N+1 observes only this sample (not the full language).
4. Generation N+1 *generalizes* from the sample to reconstruct a full language.
5. Generation N+1 then produces utterances for Generation N+2.

The crucial element is the **transmission bottleneck**: learners see only a subset of the
language. They must generalize to unseen meanings.

### The Bottleneck Effect

This is the ILM's deepest finding:

- When the bottleneck is **tight** (learners see few examples), only **compositional**
  languages survive transmission. A holistic language (where each meaning-signal pair is
  arbitrary and unrelated) cannot be faithfully reconstructed from a small sample. A
  compositional language (where signal parts systematically correspond to meaning parts)
  *can* be reconstructed because learners can generalize the rules.

- When the bottleneck is **loose** (learners see most of the language), holistic languages
  survive fine. There's no pressure toward compositionality.

- The bottleneck doesn't just allow compositionality - it *forces* it. Over iterated
  generations, any initially holistic language gradually becomes compositional because
  each generation's reconstruction errors are biased toward regularity.

### Analogy to Our Simulation

In our simulation, the "language" is encoded in NEAT genomes (neural network topology and
weights), not in an explicit mapping table. The "transmission" happens through:

1. **Selection**: only fit prey survive to reproduce.
2. **Crossover**: offspring genomes combine parent genomes.
3. **Mutation**: random changes to topology and weights.

The **bottleneck** in our system is the genome itself - a NEAT genome with ~50-200
connections cannot encode 8^(all possible world states) arbitrary mappings. The genome
must learn general rules (e.g., "if sensor pattern X, emit symbol Y") rather than
memorizing every specific situation.

This is analogous to the ILM bottleneck: **genome complexity limits expressivity, forcing
compositionality**. A genome that encodes "emit symbol 3 when aerial predator is at
distance 4 in direction NE" wastes connections. A genome that encodes "emit symbol 3 when
any aerial predator is nearby" is more compact and more likely to survive crossover intact.

### What ILM Predicts for Our System

1. **Regularity will increase over generations.** Early signaling will be noisy and
   inconsistent. Over time, the population will converge on regular, predictable mappings
   because irregular mappings are harder to maintain across genomic "transmission."

2. **Compositionality will emerge if the signal space is rich enough relative to the
   meaning space.** With 8 symbols and 3 predator types, there's room for compositionality
   (e.g., symbol encodes both type AND distance bin). Whether this happens depends on
   whether the fitness landscape rewards distance-encoding.

3. **Simplification before complexification.** ILM predicts that languages first simplify
   (lose irregularity) then complexify (add productive rules). In our system: first, symbols
   collapse to a few used symbols. Then, if selection pressure demands it, the usage patterns
   become more structured.

### Key Papers

- Kirby, S. (2001). [Spontaneous evolution of linguistic structure: An iterated learning model of the emergence of regularity and irregularity](https://cocosci.princeton.edu/tom/papers/IteratedLearningEvolutionLanguage.pdf). *IEEE Transactions on Evolutionary Computation*.
- Smith, K., Kirby, S., & Brighton, H. (2003). [Iterated Learning: A Framework for the Emergence of Language](https://www.lel.ed.ac.uk/~kenny/publications/smith_03_iterated.pdf). *Artificial Life*, 9(4), 371-386.
- Kirby, S., Griffiths, T., & Smith, K. (2014). [Iterated learning and the evolution of language](https://www.sciencedirect.com/science/article/abs/pii/S0959438814001421). *Current Opinion in Neurobiology*, 28, 108-114.
- Griffiths, T. & Kalish, M. (2007). [Language Evolution by Iterated Learning With Bayesian Agents](https://cocosci.princeton.edu/tom/papers/iteratedcogsci.pdf). *Cognitive Science*, 31, 441-480.
- Oliphant, M. & Batali, J. (1997). [Learning and the emergence of coordinated communication](https://www.semanticscholar.org/paper/Learning-and-the-Emergence-of-Coordinated-Oliphant-Batali/dc3ff53e5f123c86ce9254af7f89bf0b74242ba1). *Center for Research on Language Newsletter*.

---

## 3. Naming Games (Steels, Baronchelli)

### What They Are

Luc Steels' **Naming Game** models how a population of agents, with no central authority,
converge on a shared vocabulary through purely local interactions:

1. Two agents are randomly paired (speaker and hearer).
2. The speaker picks a topic (object/meaning) and a name for it from their personal
   vocabulary. If they have no name for the topic, they invent one.
3. The hearer checks if they know the name for that topic.
   - **Success**: both agents keep only that name for the topic (delete synonyms).
   - **Failure**: the hearer adds the speaker's name to their vocabulary.
4. Repeat for many rounds.

### Dynamics

The naming game exhibits a characteristic three-phase trajectory:

**Phase 1 - Vocabulary explosion** (early rounds): Agents invent many different names for
the same topic. Synonymy is rampant. Total number of words in the system grows rapidly.

**Phase 2 - Consensus nucleation** (middle): Local agreements start to spread. Some names
"win" in local neighborhoods and begin to dominate. This is where the **S-curve** appears:
the success rate of communication (fraction of successful interactions) transitions sharply
from low to high.

**Phase 3 - Global consensus** (late): One name per topic dominates the entire population.
Synonyms are eliminated. The system reaches a stable shared vocabulary.

### Quantitative Scaling Laws (Baronchelli et al., 2006)

Andrea Baronchelli and colleagues provided analytical results for convergence:

- **Mean-field (well-mixed population)**: convergence time scales as **N^(3/2)** where N
  is population size. Memory per agent also scales as N^(3/2) at peak.

- **Low-dimensional spatial grids** (d dimensions): convergence time scales as **N^(1+2/d)**.
  In 2D (our grid), that's N^2 - significantly slower than well-mixed.

- **Complex networks**: convergence time approximately N^(1.4).

For our simulation with N=150 prey on a 2D grid:
- Well-mixed prediction: ~150^1.5 = ~1,837 interactions to converge.
- 2D grid prediction: ~150^2 = ~22,500 interactions.
- Per generation (2000 ticks, ~150 agents, multiple interactions per tick), convergence
  on a shared vocabulary could take 5-20 generations.

### Synonymy Before Convergence

Critically, the naming game predicts that **synonymy is a transient feature** that appears
before convergence. In our simulation, we might see:
- Early: symbols 2, 5, and 7 all used for "aerial predator" (different lineages, different
  conventions).
- Middle: symbol 5 dominates but symbol 2 persists in a spatial pocket.
- Late: symbol 5 universally means "aerial predator."

The Zipf-like distribution of synonyms during the transient phase (a few dominant names,
many rare ones) would be measurable in our signal logs.

### Relevance to Our Simulation

Our simulation differs from the standard naming game in that:
- Agents don't explicitly negotiate (no success/failure feedback per interaction).
- "Agreement" emerges through differential survival, not explicit vocabulary update.
- The topology is a 2D spatial grid, not well-mixed.

But the core dynamics should be similar: local conventions form first (kin clusters share
signal-meaning mappings), then spread as successful lineages expand.

### Key Papers

- Steels, L. (1995). [A self-organizing spatial vocabulary](https://www.academia.edu/2936171/The_naming_game). *Artificial Life*, 2(3), 319-332.
- Baronchelli, A., Felici, M., Loreto, V., Caglioti, E., & Steels, L. (2006). [Sharp transition towards shared vocabularies in multi-agent systems](https://www.stat.berkeley.edu/~aldous/260-FMIE/Papers/baronchelli.pdf). *Journal of Statistical Mechanics*.
- Baronchelli, A., Loreto, V., & Steels, L. (2008). [In-depth analysis of the Naming Game dynamics: the homogeneous mixing case](https://arxiv.org/abs/0803.0398). *Int. J. Mod. Phys. C*, 19(5).
- Steels, L. & Loetzsch, M. (2012). [The Grounded Naming Game](https://martin-loetzsch.de/publications/steels12grounded.pdf). *Advances in Interaction Studies*, 3.

---

## 4. Referential Games in Deep Learning

### The Standard Setup

Since ~2016, deep learning researchers have studied emergent communication using
**referential games**: two neural network agents (Speaker and Listener) trained to
communicate about images or structured inputs.

- Speaker sees a target object, produces a discrete message (sequence of symbols).
- Listener sees several candidates, receives the message, and must identify the target.
- Both are trained end-to-end with reinforcement learning (REINFORCE) or a differentiable
  relaxation (Gumbel-softmax).

### Key Findings

**Lazaridou et al. (2017, 2018)** - "Emergence of Linguistic Communication from Referential
Games":
- Agents develop communication protocols that achieve high task success.
- The degree of structure in the protocol mirrors the structure of the input: structured
  inputs (feature vectors) produce more structured languages than raw pixels.
- Critical finding: compositionality doesn't emerge easily. Most emergent languages are
  **holistic** (each message is an opaque label for each input, no internal structure).

**Havrylov & Titov (2017)** - "Emergence of Language with Multi-agent Games":
- Used variable-length sequences of discrete symbols (closer to real language).
- Gumbel-softmax (differentiable) converges much faster than REINFORCE.
- Observed some compositionality and variability (same meaning expressed multiple ways).
- But compositionality was weak and inconsistent across runs.

**Chaabouni et al. (2019, NeurIPS)** - "Anti-efficient encoding in emergent communication":
- Tested whether Zipf's Law of Abbreviation (frequent meanings get shorter signals) emerges.
- **Result: the opposite happens.** Neural agents produce ANTI-efficient encoding - frequent
  meanings get the LONGEST messages.
- Reason: neural listeners find longer messages easier to discriminate, and speakers have
  no physiological pressure toward brevity (unlike humans).
- When a length penalty is added to the speaker's loss, Zipf's law re-emerges.
- **Implication for our system**: NEAT genomes DO have a "brevity pressure" - shorter,
  simpler networks are easier to evolve and maintain. This could naturally produce more
  efficient encoding than deep RL agents.

**Mordatch & Abbeel (2018, AAAI)** - "Emergence of Grounded Compositional Language in
Multi-Agent Populations":
- Multi-agent environment with physical goals (navigate to locations).
- Agents develop compositional language with a defined vocabulary and syntax.
- Also observed non-verbal communication (pointing, guiding) when verbal channel removed.
- Environmental pressure (needing to achieve goals) drives compositionality more than
  purely referential tasks.

### What Doesn't Emerge Without Specific Pressures

The deep learning literature has identified that the following do NOT emerge naturally:

1. **Compositionality** - requires either:
   - A learning bottleneck (Ren et al., 2020) - analogous to ILM.
   - Ease-of-teaching pressure (Li & Bowling, 2019) - agents rewarded for being easy
     for new agents to understand.
   - Environmental pressure requiring generalization (Mordatch & Abbeel, 2018).
   - Interaction history / dialogue (Cogswell et al., 2020).

2. **Zipf's law / efficient encoding** - requires a cost on message length (Chaabouni
   et al., 2019). Without cost, anti-efficient encoding dominates.

3. **Natural language-like properties** - emergent protocols consistently lack:
   - Recursion or embedding.
   - Displacement (referring to non-present things).
   - Open-ended productivity.
   - These require more complex environments and pressures than referential games provide.

### What DOES Emerge Reliably

- **Referentiality** - symbols consistently map to meanings (high mutual information).
- **Arbitrary conventions** - different random seeds produce different but equally
  functional mappings.
- **Some degree of systematicity** - especially with structured inputs.
- **Redundancy** - messages are longer than strictly necessary for discrimination.

### Pressures That Drive Compositionality (Ranked by Effectiveness)

1. **Information bottleneck / limited channel** - restricting message length forces
   compositional encoding (each position carries independent meaning).
2. **Generalization pressure** - agents must communicate about novel combinations of
   features not seen during training. Holistic codes fail; compositional ones generalize.
3. **Population turnover** - new agents must quickly learn the protocol, favoring simpler,
   more regular systems (ease-of-teaching).
4. **Environmental grounding** - signals tied to physical consequences (survival) produce
   more structured communication than abstract referential games.
5. **Interaction diversity** - communicating with many different partners prevents
   overfitting to one partner's idiosyncrasies.

### Key Papers

- Lazaridou, A., Peysakhovich, A., & Baroni, M. (2017). Multi-Agent Cooperation and the Emergence of (Natural) Language. *ICLR*.
- Lazaridou, A., Hermann, K.M., Tuyls, K., & Clark, S. (2018). [Emergence of Linguistic Communication from Referential Games with Symbolic and Pixel Input](https://arxiv.org/abs/1804.03984). *ICLR*.
- Havrylov, S. & Titov, I. (2017). [Emergence of Language with Multi-agent Games: Learning to Communicate with Sequences of Symbols](https://arxiv.org/abs/1705.11192). *NeurIPS*.
- Chaabouni, R., Kharitonov, E., Dupoux, E., & Baroni, M. (2019). [Anti-efficient encoding in emergent communication](https://arxiv.org/abs/1905.12561). *NeurIPS*.
- Mordatch, I. & Abbeel, P. (2018). [Emergence of Grounded Compositional Language in Multi-Agent Populations](https://arxiv.org/abs/1703.04908). *AAAI*.
- Lazaridou, A. & Baroni, M. (2020). [Emergent Multi-Agent Communication in the Deep Learning Era](https://arxiv.org/abs/2006.02419). Survey paper.
- Li, F. & Bowling, M. (2019). [Ease-of-Teaching and Language Structure from Emergent Communication](http://papers.neurips.cc/paper/9714-ease-of-teaching-and-language-structure-from-emergent-communication.pdf). *NeurIPS*.

---

## 5. What Our Simulation Adds vs Existing Work

### The Standard Setup in Existing Literature

Most emergent communication research uses:

| Feature | Standard Approach | Our Simulation |
|---------|------------------|----------------|
| Task | Referential game (identify object) | Survival (avoid being eaten) |
| Pressure | Task reward (accuracy) | Life or death (fitness = survival time) |
| Agents | Fixed roles (sender/receiver) | Emergent roles (any prey can send/receive) |
| Learning | Deep RL (gradient descent) | NEAT neuroevolution (topology + weight evolution) |
| Population | 2 agents, or N with shared weights | 150 individuals, each with unique genome |
| Space | No spatial structure | 80x60 grid with terrain |
| Kin structure | None | Spatial kin clusters from local reproduction |
| Time | Discrete referential rounds | Continuous tick-based simulation |
| Signal cost | Free or fixed penalty | Calibrated to Hamilton's rule (3.0 energy) |
| Signal semantics | Pre-defined meaning space (features) | Open-ended (predator type + distance + direction) |
| Multiple meanings | Usually one object per round | Multiple simultaneous predators possible |

### What Is Genuinely Novel

**1. Survival pressure creates honest-signaling dynamics.**
In referential games, there's no cost to signaling and no concept of deception. In our
simulation, signaling costs energy and attracts predator attention. This creates the
honest-signaling problem studied in biological signaling theory (Zahavi's handicap
principle, Hamilton's rule). The signal must be costly enough to prevent cheating but
beneficial enough to be worth the cost. This is a genuine gap in the deep RL literature
- they study communication *efficiency* but not communication *honesty*.

**2. Kin selection drives altruistic signaling.**
The combination of spatial reproduction (offspring spawn near parents) and kin fitness
bonuses means that signaling evolves through kin selection, not individual rationality.
This is the biological mechanism for alarm call evolution in nature (vervet monkeys,
ground squirrels). No deep RL emergent communication paper models this because they lack
population genetics and spatial structure.

**3. NEAT evolves the communication architecture itself.**
Deep RL agents have fixed architectures (encoder-decoder with attention, etc.). NEAT
evolves both the network topology and weights simultaneously. This means the "language
faculty" itself evolves - some prey might develop complex signal-processing circuitry
while others remain simple. This mirrors biological neural evolution more closely than
fixed-architecture training.

**4. Emergent sender-receiver roles.**
In standard referential games, one agent is the Speaker and one is the Listener. In our
simulation, every prey is both a potential sender and receiver. Whether a prey signals
depends on its evolved brain, its current energy, and whether it perceives a predator.
This creates the possibility of:
- **Sentinel behavior**: some prey specialize in watching and signaling.
- **Free-riding**: some prey listen but never signal (save energy).
- **Context-dependent roles**: prey signal when they have energy to spare but stay quiet
  when energy-poor.

**5. Multiple simultaneous threats require disambiguation.**
Standard referential games have one target per round. Our prey face 3 predator types
simultaneously, each requiring a different response. This creates natural pressure for
a *structured* signal vocabulary (not just "danger!" but "what kind of danger").

**6. Discrete grid + terrain creates spatial semantics.**
The terrain structure (trees protect from aerial, rocks from ground) means signals can
carry implicit spatial information. A prey near trees that signals "aerial" is implicitly
saying "come here, there's cover." This spatial grounding is absent from abstract
referential games.

### What Existing Work Has That We Lack (Limitations)

- No variable-length messages (our symbols are single-shot, not sequences).
- No recursive structure possible (8 symbols, no grammar).
- No within-lifetime learning (NEAT is evolutionary, not developmental).
- No explicit semantic structure in the meaning space (predator types are unstructured).
- No cultural transmission in the ILM sense (genomes are biological, not learned).

### Prior Art Closest to Our Setup

- **Mitri et al. (2009, 2011)** - Evolved communication in predator-prey with
  neuroevolution. Closest existing work. Used simple feed-forward networks (not NEAT),
  binary signals (not 8 symbols), and showed that kin clustering is necessary for honest
  signaling. Our 8-symbol vocabulary and NEAT topology evolution are direct extensions.

- **Marocco & Nolfi (2007)** - Evolved robots that develop alarm calls. Used evolutionary
  robotics with simple neural controllers. Showed that alarm calls emerge when agents
  are genetically related and signals are costly.

- **Floreano, Mitri et al. (2007)** - "Evolution of communication in a population of
  mobile robots." Showed that deceptive signaling evolves when agents are unrelated, but
  honest signaling evolves with kin selection. Key paper for our kin bonus justification.

- **Quinn (2001)** - [Conditions enabling the evolution of inter-agent signaling in an
  artificial world](https://direct.mit.edu/artl/article/7/1/3/2365/Conditions-Enabling-the-Evolution-of-Inter-Agent). Found that if agents can sense danger directly, communication
  doesn't evolve (evolution finds the simpler solution). Critical insight: communication
  must provide information that can't be obtained otherwise. In our simulation, hearing
  range (12) exceeds vision range (8), so signals provide early warning that vision alone
  cannot.

---

## 6. Metrics for Language Properties (Hockett's Design Features)

### Background: Hockett's Design Features

Charles Hockett (1960) proposed 13 (later 16) design features that characterize human
language and distinguish it from animal communication. Not all are achievable or
measurable in our simulation. Here's the full list with assessments:

| Feature | Definition | Achievable? | Measurable? |
|---------|-----------|-------------|-------------|
| Vocal-auditory channel | Sound-based | N/A (abstract symbols) | N/A |
| Broadcast transmission | Signal goes to all nearby | Yes (built-in) | Trivially yes |
| Rapid fading | Signals are temporary | Yes (signal_lifetime=3 ticks) | Trivially yes |
| Interchangeability | Any agent can send or receive | Yes (all prey can) | Check if some always/never signal |
| Total feedback | Sender hears own signal | Partially (sender can sense own emission) | Log self-reception events |
| Specialization | Signals have no physical effect beyond information | Yes (signals don't move agents) | Trivially yes |
| **Semanticity** | Signals have meaning | **Target** | Mutual Information (below) |
| **Arbitrariness** | Signal form is unrelated to meaning | **Target** | Check if symbol encoding is systematic |
| **Discreteness** | Finite set of distinct signals | **Built-in** (8 symbols) | Trivially yes |
| **Displacement** | Reference to non-present things | **Stretch goal** | Measure signaling when predator is outside vision |
| **Productivity** | Novel combinations convey new meanings | **Unlikely** (single symbols) | Would need sequences |
| Cultural transmission | Language is learned, not innate | **Partially** (genetic, not cultural) | Compare lineages |
| Duality of patterning | Meaningless elements combine into meaningful units | **No** (single symbols) | N/A |
| Prevarication | Ability to lie | **Yes** (deceptive signaling) | Deception rate metric |
| Reflexiveness | Talk about language | No | N/A |
| Learnability | Can learn other languages | No (fixed genome) | N/A |

### Measurable Properties and How to Compute Them

#### 6.1 Referentiality (Semanticity)

**Question:** Do symbols consistently map to environmental states?

**Metric: Mutual Information I(S; C)**

Where S = emitted symbol, C = context (predator type / distance bin / no predator).

```
Computation:
1. Collect all SignalEvents for a generation.
2. For each event, record:
   - s = symbol emitted
   - c = context category (aerial_near, aerial_far, ground_near, ground_far,
     pack_near, pack_far, no_predator)
3. Build joint distribution P(S, C) from counts.
4. Compute marginals P(S), P(C).
5. MI = sum over s, c of P(s,c) * log(P(s,c) / (P(s) * P(c)))
6. Normalize by H(S) to get Normalized MI in [0, 1].
```

**NMI = 0**: symbols are independent of context (random or no signaling).
**NMI > 0.3**: meaningful signal-context association.
**NMI > 0.7**: strong referential system.

**Baseline**: compute NMI with shuffled symbols to establish chance level.

#### 6.2 Topographic Similarity (TopSim)

**Question:** Do similar situations produce similar signals?

**Metric: Spearman correlation between meaning distances and signal distances.**

Introduced by Brighton & Kirby (2006). Standard metric in emergent communication.

```
Computation:
1. Define a meaning vector for each signal event:
   m = (predator_type_onehot, distance_normalized, direction_x, direction_y,
        own_energy, nearby_prey_count)
2. Compute pairwise meaning distances: d_m(i,j) = Euclidean(m_i, m_j)
3. Compute pairwise signal distances: d_s(i,j) = (s_i != s_j) ? 1 : 0
   (Hamming distance for single symbols)
4. TopSim = Spearman_correlation(d_m, d_s)
```

**TopSim ~ 0.0**: no structure (random mapping).
**TopSim > 0.3**: moderate systematicity.
**TopSim > 0.5**: strong compositional structure.

**Limitation**: With single symbols (not sequences), TopSim mainly measures whether
the symbol vocabulary partitions the meaning space into coherent clusters. True
compositionality metrics (PosDis, BosDis) require multi-symbol messages.

#### 6.3 Positional Disentanglement (PosDis) and Bag-of-Symbols Disentanglement (BosDis)

These are designed for multi-symbol messages. With our single-symbol setup, they
degenerate. **If we extend to 2-symbol messages** (possible with output neurons
for position 1 + position 2), these become applicable:

- **PosDis**: measures whether each position in the message encodes a different attribute.
  E.g., position 1 encodes predator type, position 2 encodes distance bin. Computed as
  the gap between mutual information of the best-predicted attribute and the second-best
  at each position.

- **BosDis**: same as PosDis but ignores position (bag of symbols). High BosDis + low
  PosDis = compositionality without fixed word order.

Reference: Chaabouni et al. (2020), "Compositionality and Generalization in Emergent Languages."

#### 6.4 Displacement

**Question:** Can signals refer to things not currently perceived by the sender?

This is an advanced feature unlikely to emerge in generation 1-100, but measurable:

```
Computation:
1. For each signal event, record sender's vision state.
2. Flag events where:
   - Signal is context-appropriate (matches a predator type by NMI analysis)
   - BUT sender cannot currently see the relevant predator.
3. Displacement rate = flagged events / total context-appropriate signals.
```

A non-zero displacement rate could mean:
- Signal was triggered by a heard signal from another prey (relay/echo behavior).
- Signal was triggered by recent memory (predator was visible last tick but not now).
- Coincidence (baseline noise).

To distinguish these, compare displacement rate against a shuffled baseline.

#### 6.5 Prevarication (Deception)

**Question:** Do prey emit misleading signals?

```
Computation:
1. For each signal event, check:
   - Is there a predator matching the signal's "usual meaning" (by NMI) nearby?
   - If not: this is a potentially deceptive signal.
2. Deception rate = signals-without-matching-predator / total-signals
3. Track over generations to see if deception increases or decreases.
```

Expected trajectory: high deception initially (random signaling), drops as referentiality
emerges, may spike if deceptive mutants gain short-term advantage, then re-drops as
receivers evolve to ignore unreliable signalers.

Deceptive signaling that persists indicates a breakdown of honest signaling - likely
caused by insufficient kin structure (relatedness too low for Hamilton's rule).

#### 6.6 Vocabulary Usage Entropy

**Question:** How much of the symbol vocabulary is actually used?

```
H(S) = -sum over s of P(s) * log2(P(s))
```

**H(S) = 0**: all signals use the same symbol (maximally redundant).
**H(S) = log2(8) = 3.0**: all symbols used equally (maximally diverse).

Expected trajectory:
- Early: moderate entropy (random signaling).
- Middle: entropy drops (population converges on a few symbols).
- Late: entropy settles at log2(k) where k = number of functionally distinct signals.
  With 3 predator types, expect H ~ 1.5-2.0 bits.

#### 6.7 Signal Usage Rate

**Question:** How often do prey signal at all?

```
Signal rate = signals_emitted / (prey_alive * ticks)
```

If signal rate collapses to 0, communication has been selected against (cost too high,
no benefit). If it stays near 1.0, signaling is too cheap or there's a bug.

Expected: 5-15% of ticks involve signaling, concentrated around predator encounters.

#### 6.8 Receiver Response Metric

**Question:** Do receivers change behavior after hearing a signal?

```
1. For each prey that receives a signal on tick T:
   - Record action on tick T (before receiving any signal).
   - Record action on tick T+1 (after receiving signal).
2. Compare action distributions for:
   - Ticks following signal reception vs. ticks without signals.
3. Chi-squared test: is the action distribution significantly different?
```

Strong receiver response + high referentiality = functional communication system.

### Recommended Metric Dashboard (Per Generation)

| Metric | Range | What It Tells You |
|--------|-------|-------------------|
| NMI(Symbol; PredatorType) | [0, 1] | Are symbols referential? |
| NMI(Symbol; PredatorDistance) | [0, 1] | Do symbols encode distance? |
| TopSim | [-1, 1] | Is there systematic structure? |
| Vocab entropy H(S) | [0, 3.0] | How many symbols are in active use? |
| Signal rate | [0, 1] | How often do prey signal? |
| Deception rate | [0, 1] | How many signals are "lies"? |
| Receiver response delta | p-value | Do receivers react to signals? |
| Mean fitness with signaling | f32 | Does signaling correlate with survival? |
| Displacement rate | [0, 1] | Are there signals about non-visible threats? |

---

## 7. Paper Strategy: Venues, Narrative, Surprising Results

### Venue Analysis

| Venue | Fit | Type | Typical Acceptance | Notes |
|-------|-----|------|-------------------|-------|
| **ALIFE Conference** | Excellent | Conference (MIT Press proceedings) | ~40-50% | Primary target. Next: ALIFE 2026 in Waterloo, Aug 17-21. 8-page full paper or 2-page extended abstract. Theme: "Living and Lifelike Complex Adaptive Systems." Our project fits perfectly. |
| **Artificial Life Journal** | Excellent | Journal (MIT Press, quarterly) | ~30% | For a full-length paper with thorough analysis. Longer format (15-25 pages) allows deeper treatment of metrics and evolutionary dynamics. |
| **CogSci** | Good | Conference | ~25-30% | Framing: "what do these results tell us about the evolution of language?" Requires cognitive science angle, not just simulation results. 6-page papers. |
| **EMNLP Workshop (EmeCom)** | Good | Workshop | ~50% | Emergent Communication workshop has appeared at NeurIPS and ICLR. Position paper comparing neuroevolution vs. deep RL for language emergence. |
| **GECCO** | Good | Conference (ACM) | ~35% | Genetic and Evolutionary Computation Conference. Focus on the NEAT methodology aspects. |
| **EvoLang** | Excellent | Conference | ~40% | Evolution of Language conference (biennial). The core venue for language evolution research. Check schedule. |

**Recommended strategy**: Submit to ALIFE 2026 (deadline likely Feb-Apr 2026) as primary
venue. Simultaneously prepare a journal-length version for Artificial Life journal. If
results on Hockett features are strong, a CogSci submission would also work.

### Compelling Narrative

The strongest narrative is **not** "we built a cool simulation" but rather:

**"When does honest, structured communication emerge from meaningless signals, and what
are the necessary and sufficient conditions?"**

Frame it as a **negative-result-resistant** investigation:

1. If structured communication emerges: "We show that survival pressure + kin structure +
   costly signaling are sufficient for proto-language emergence, even without gradient-based
   learning or explicit teaching."

2. If only partial structure emerges (referential but not compositional): "We identify a
   boundary condition - single-symbol signals with evolutionary learning achieve referentiality
   but not compositionality, suggesting that compositional structure requires either multi-symbol
   messages, cultural (non-genetic) transmission, or both."

3. If communication fails to emerge: "We demonstrate that [specific missing condition] is
   necessary for honest signaling to evolve, contradicting the assumption that survival
   pressure alone suffices."

All three are publishable. The third might be the most interesting.

### What Results Would Be Genuinely Surprising

Ranked from most to least surprising (and therefore impactful):

**1. Displacement emerges.** If prey signal about predators they can no longer see (e.g.,
relaying heard signals), this would be remarkable. Displacement is considered a uniquely
human language feature. Demonstrating it in a simple neuroevolution simulation would
challenge the "only humans have displacement" narrative. **High impact if demonstrated
rigorously with controls.**

**2. Deception-honest signaling arms race.** If the simulation shows oscillating dynamics
where deceptive mutants invade, receivers evolve skepticism, then honest signalers return,
this would demonstrate an evolutionary arms race in communication. This has been theorized
but rarely demonstrated in simulation with both spatial and kin structure.

**3. NEAT topology correlates with communication role.** If "sentinel" prey (frequent
signalers) evolve measurably different network topologies from "listener" prey (infrequent
signalers, strong signal-to-action pathways), this would show that evolution differentiated
communication roles through neural architecture, not just weights. This has no precedent
in the emergent communication literature.

**4. Predator-specific alarm calls emerge with >3 distinct symbols.** Vervet monkey-like
alarm calls (distinct signals for aerial, ground, pack) would be the expected "success"
result. This would replicate the biological finding in simulation. Solid but not surprising.

**5. Symbol meaning drifts over generations (language change).** If the mapping from
symbols to predator types shifts over time (symbol 3 meant "aerial" in generation 100 but
means "ground" by generation 500), this would demonstrate cultural drift in an evolutionary
(not cultural) transmission system. Interesting because it shows that even genetic encoding
doesn't prevent meaning drift.

**6. Compositionality in 8 single-symbol messages.** Compositional structure with only 8
symbols and no sequences would be very surprising. It would require symbols to encode
*combinations* (e.g., symbol 5 = aerial + close, symbol 6 = aerial + far). This is
theoretically possible if the fitness landscape strongly rewards distance information.

### Contribution Claims (for the paper's "contributions" section)

1. **First study combining NEAT neuroevolution, costly signaling, kin selection, and
   multi-predator disambiguation** in a single spatial simulation. Each component has been
   studied separately; the combination is novel.

2. **Quantitative measurement of Hockett's design features** (referentiality, arbitrariness,
   displacement, prevarication) in an evolutionary simulation. Most ALife communication
   papers report qualitative observations; we provide formal metrics.

3. **Comparison of evolved communication properties** against predictions from Lewis games,
   iterated learning, and deep RL emergent communication. Bridges game theory, cognitive
   science, and artificial life.

4. **Open-source, reproducible simulation** with deterministic seeding and configurable
   parameters. (Strong plus for any venue; reproducibility is a major concern.)

### Minimum Viable Results for a Paper

To submit to ALIFE or Artificial Life journal, you need at minimum:

1. **NMI > 0.3** for at least one predator type by generation ~200-500. This proves
   referential signaling emerged.

2. **Signal rate between 3% and 30%** (not zero, not always). This proves signaling is
   functionally used, not random.

3. **Deception rate trending downward** over generations. This proves honest signaling
   is being selected for.

4. **Receiver response delta is significant** (p < 0.05). This proves receivers are
   actually responding to signals, not ignoring them.

5. **Comparison across at least 5 random seeds** to show robustness. Different seeds may
   produce different symbol-meaning mappings but similar overall dynamics.

6. **Ablation**: run with signal_energy_cost = 0 (free signaling) and show that deception
   rate increases / communication quality decreases. This validates the costly signaling
   mechanism.

---

## Summary: Key Takeaways for Implementation

1. **The Lewis game framework predicts convergence to signaling systems.** With 3 predator
   types and 8 symbols, expect 3-5 symbols to become "active" with the rest unused.

2. **The ILM bottleneck effect applies through genome complexity.** NEAT genomes can't
   encode arbitrary mappings, so evolution will favor general rules - this is our pressure
   toward systematicity.

3. **Naming game dynamics predict an S-curve**: slow initial progress, sharp transition to
   shared vocabulary, then stability. In 2D grids, expect ~N^2 interactions to converge.

4. **Deep RL findings warn that compositionality is hard.** Don't expect rich compositional
   structure from single symbols. Focus the paper on referentiality and honest signaling.

5. **Our unique contribution is the combination**: survival pressure + kin selection +
   NEAT topology evolution + costly signaling. No existing work combines all four.

6. **Measure everything**: NMI, TopSim, deception rate, signal rate, vocabulary entropy,
   receiver response. These are the quantitative backbone of any paper.

7. **Best venue**: ALIFE 2026 (August, Waterloo) for conference paper. Artificial Life
   journal for the full treatment. CogSci if you find cognitive science implications.
