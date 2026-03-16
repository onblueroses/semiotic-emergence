# v5 Signal Run Findings

Two independent runs (seeds 200, 201) at 114k and 109k generations respectively.
384 agents, 56x56 toroidal grid, 3 kill zones, 6-symbol vocabulary, 500 ticks/gen.

## What evolved

A one-word language meaning "I exist here." The word has no fixed form (any symbol works),
no grammar, no vocabulary differentiation. It functions as a social coordination protocol
that produces aggregation, reducing zone mortality by ~36%.

## Key metrics

| Metric | s200 | s201 | Interpretation |
|--------|------|------|----------------|
| r(signals, fitness) | 0.83 | 0.83 | Strongest correlation. More signals = more survival |
| r(zone_deaths, signals) | -0.93 | - | More signals -> fewer zone deaths |
| r(zone_deaths, fitness) | -0.77 | - | Fewer zone deaths -> higher fitness |
| response_fit_corr | 0.00 | 0.00 | Not broken - population converged on single response strategy |
| MI (signal-zone) | ~0.003 | ~0.001 | Near zero. Symbols don't encode zone distance |
| MI CoV | 0.71 | 0.79 | Most unstable metric. MI is noise, not proto-language |
| Signal entropy (effective vocab) | 2.9 | 2.3 | 3 of 6 symbols used, but interchangeably |
| Pairwise symbol contrast | 0.0012 | - | All symbols functionally identical |
| Zone deaths Q1 -> Q4 | 460 -> 295 | - | 36% reduction over evolution |
| Receiver-fit (Q1 -> Q4) | 0.687 -> 0.667 | 0.699 -> 0.671 | Declining: spatial confound weakening |

## Input encoding budget (what the brain actually processes)

| Category | s200 | s201 |
|----------|------|------|
| Memory cells (recurrent) | 51.7% | 50.6% |
| Signal strength | 16.8% | 19.3% |
| Signal direction | 12.8% | 15.9% |
| Zone damage (pain) | 11.5% | 10.7% |
| Food perception | 6.9% | 3.3% |
| Ally perception | 0.3% | 0.1% |

The brain runs on internal temporal state, not external perception. Signals are processed as
"where is the swarm" (strength + direction), not "what is the swarm saying" (symbol identity).
Ally perception is vestigial - signals completely replaced the built-in social input.

## Finding-by-finding analysis

### 1. Signals are survival infrastructure

The causal chain: more signals -> more aggregation -> fewer zone deaths -> higher fitness.
Signal_cost (0.002/emission) creates selective pressure against noise, but the group benefit
of aggregation overwhelms the individual cost. This is a classic public goods dynamic.

### 2. Memory dominance (51%)

The 8 memory cells with EMA update (0.9 * old + 0.1 * new) act as slow-moving integrators.
The brain builds a temporal model of its own experience - primitive proprioception across time.
Decisions are primarily inertial, based on accumulated state rather than immediate perception.

### 3. Symbol interchangeability

All 15 pairwise symbol contrasts are within noise (mean=0.0012, std=0.0006). Effective
vocabulary is 2-3 symbols, but they all carry the same meaning. Why not all 6? Kin clusters
share weights through reproduction, so local populations converge on a preferred symbol
through drift. The global distribution reflects spatial kin structure, not semantic
differentiation.

### 4. MI spikes are drift noise

32 spike episodes (s200), 23 (s201). Duration: 10-630 gens. No fitness improvement follows
spikes. Autocorrelation shows no periodicity (first peak at lag 3-5, r=0.21). MI is white
noise with trivial short-memory. Spikes occur when drift temporarily aligns a kin cluster's
preferred symbol with zone proximity, creating momentary signal-zone correlation that
selection doesn't maintain.

### 5. Signal hidden oscillation: public goods cycle

Signal hidden size oscillates wildly (4.7-29.6 neurons, avg period 523 gens). With
neuron_cost = 0, brain size is free. The oscillation is a tragedy-of-the-commons cycle:
1. Large signal heads -> lots of signals -> group benefit -> high fitness
2. Free-rider mutants (small heads) benefit from others' signals at lower cost
3. Free-riders spread, signal volume drops, group fitness drops
4. High-signaling mutants regain advantage, cycle restarts

The signal head acts as a signal volume knob, not an information processor. Larger head
emits more, which helps the group through aggregation.

### 6. response_fit_corr = 0.00 is not a bug

This metric asks: "do prey that change behavior more when hearing signals survive better?"
When the entire population has converged on the same response (move toward signals),
per-prey JSD is uniform. No variance -> no correlation. response_fit = 0 is the signature
of a fully converged strategy, not the absence of signal value.

The correct test is counterfactual: fitness WITH signals minus fitness WITHOUT. The
`--no-signals` flag provides this but was not run for v5.

### 7. Zone-preferential symbols (closest to referential content)

Symbol 3 (s200) has +27.5% zone preference. Symbol 4 (s201) has +25.6%. Different symbols
in different seeds - convergent pattern, divergent convention. But:
- Different symbol in each seed (not structurally convergent)
- response_fit = 0 (receivers don't differentiate)
- MI is unstable (the preference isn't maintained by selection)

Most likely explanation: kin cluster near a zone happens to use its locally preferred symbol,
creating apparent zone-symbol correlation without semantic intent.

## Why referential communication did not emerge

The environment has exactly ONE information channel worth encoding through signals:
prey location (invisible beyond vision range 2.0, but detectable via signal range 8.0).
Evolution found exactly one signal meaning to fill this gap: presence.

Why no vocabulary diversification:
- Zones are homogeneous (all identical, no types to distinguish)
- Zones are self-perceptible (pain inputs 0-1, so neighbors don't need to warn you)
- Food is directly visible (inputs 3-5, no need to announce food)
- Zone response is uniform (flee in any direction, no directional vocabulary needed)
- There is nothing to say that a neighbor can't already feel for itself

## What would need to change

For multi-symbol communication to evolve, the environment needs multiple distinct situations
requiring different optimal responses, where the information is not self-perceptible:

1. **Heterogeneous zones** - two types with different optimal responses (flee vs freeze)
2. **Invisible food** - remove food perception to force social information sharing
3. **Directional zones** - predictable movement requiring directional vocabulary
4. **Active predators** - chase vs ambush behaviors requiring different escape strategies

The metric infrastructure is sound. The environment is informationally impoverished for
multi-symbol communication.
