# Performance Notes

## Benchmarks (2026-03-10, i7-12650H, Windows 11, release+LTO)

### Per-generation phase breakdown (500 ticks/gen)

| Pop | Grid | Zones | Food | eval | metrics | evolve | total/100g | gens/min |
|-----|------|-------|------|------|---------|--------|------------|----------|
| 384 | 56 | 16 | 200 | 3.43s | 0.20s | 0.46s | 4.31s | 1,392 |
| 2000 | 128 | 80 | 1040 | 37.04s | 0.71s | 5.58s | 43.55s | 138 |
| 5000 | 200 | 200 | 2600 | 39.80s | 0.35s | 6.67s | 46.95s/20g | 25.6 |

Eval (the 500-tick simulation loop) dominates at all sizes (~85% of time).
Evolve_spatial is the second cost center (~12%), scaling with population due to
spatial tournament selection with distance checks.

### Run times at 8x scale (384 prey)

Short benchmark (100 gens, no I/O): 1,392 gens/min.
Long run (100k gens, Tee-Object logging): 535 gens/min (~2.6x I/O overhead).
Estimated long-run without Tee-Object: ~1,200-1,400 gens/min.

| Generations | Projected (no I/O) | Actual |
|-------------|-------------------|--------|
| 10,000 | ~7 min | - |
| 100,000 | ~72 min | 187 min (Tee-Object) |
| 500,000 | ~6 hours | - |
| 1,000,000 | ~12 hours | - |

### Scaling with population

Going from 384 to 2000 prey (5.2x) costs ~10x in eval time per gen.
Going from 2000 to 5000 (2.5x) costs ~5.4x more in eval time per gen.
Scaling is slightly worse than linear due to signal processing: each prey
iterates all active signals in `receive_detailed()`, making signal reception
O(prey * active_signals). More prey emit more signals, so it's closer to O(n^2).

### Spatial grid optimization (implemented 2026-03-09)

Before the spatial grid, the 8x scale ran at ~13 gens/min. After: ~1,392 gens/min.
That's a **~107x speedup**, not the initially estimated 3x.

Key changes:
- CellGrid with Chebyshev ring search replaces O(n) scans for nearest ally, food, prey
- Food uses swap_remove (O(1)) instead of remove (O(n))
- Nearest zone distance cached once per prey per tick (was computed 2-3x)
- Pre-allocated buffers for shuffled indices and position snapshots

### Rayon parallelism (tested, marginal at 384 prey)

Added rayon par_iter for the compute phase (build_inputs + brain.forward).
At 384 prey, per-item work (~10us) is too small to amortize thread scheduling.
No measurable speedup. Kept in code but does not change behavior for the
sequential apply phase (food eating, signal emission, movement).

### Bash `time` on Windows is broken

Git Bash's `time` builtin adds ~37 seconds of fixed overhead on Windows.
Use PowerShell `Measure-Command` for accurate timing. This was discovered
when all benchmarks showed identical ~39s regardless of workload.

### What target-cpu=native does NOT help

Tested `RUSTFLAGS="-C target-cpu=native"` (AVX2 on i7-12650H).
No measurable difference - LLVM already auto-vectorizes with SSE2,
and the hot loops (Chebyshev ring search, small matrix multiply with
4-16 hidden neurons) are branch-heavy, not SIMD-friendly.

### 10k generation test run (small scale, 48 prey)

Completed 10,000 gens in ~36s actual compute. No NaN/Inf, no population
crashes (min avg fitness 82, max 551), no zero-signal gens. Brain size
evolved from 6.0 to 9-12 avg (max individual hit 16). Symbol 2 went
unused - population converged on 2-symbol vocabulary.

## 100k Generation Run (8x scale, seed 42)

**Runtime:** 11,207 seconds (~3.1 hours), ~535 gens/min. The 2.6x slowdown
vs benchmark (1,392 gens/min) is from PowerShell `Tee-Object` piping overhead.
Future runs should write directly to file.

### Three evolutionary eras

**Era 1 - Small Brain (Gen 0-46k):** avg_hidden oscillates 6-10, never settles.
Weak MI emerges early (~0.04 at gen 5k-10k) with positive sender_fit_corr (0.32) -
signaling weakly helps fitness. Iconicity is negative (prey go silent near predators).
Phase transition indicator peaks at 13.35 at gen 12,200. Then slow decline: MI drops
toward zero by gen 45k.

**Era 2 - Brain Explosion (Gen 46k-50k):** avg_hidden rockets from 10 to 15 in ~4,000
generations. Total semiotic collapse: MI, iconicity, sender_fit_corr, traj_fluct_ratio
all drop to zero simultaneously. Signals still emitted (25-40k/gen) but carry zero
information. The expanded neural architecture is unoptimized - weights that evolved for
6-10 neurons now control 15-16.

**Era 3 - Big Brain Renaissance (Gen 75k-100k):** avg_hidden locked at 14.8-15.5.
Strongest semiotic signal ever observed in this system. MI sustains above 0.2 for
6,187 consecutive generations (80,711-86,898). Peak MI = 0.6242 at gen 83,544.

### What signals encode during the MI surge (gen 78k-86k)

Input MI analysis reveals signals encode predator information:

| Input dimension | MI |
|-----------------|-----|
| Predator distance | 0.472 |
| Predator dy | 0.267 |
| Predator dx | 0.199 |
| Signal 0 strength | 0.179 (relay) |
| Signal 2 strength | 0.096 |
| Energy | 0.077 |
| Food dx/dy | ~0.001 (nothing) |

Signal-0-strength MI of 0.179 suggests signal relaying - prey that hear a signal
re-emit, creating relay chains.

### Key findings

**1. Brain size is the rate-limiting factor for semiotic emergence.** Small brains
(~6-10 neurons): MI peaks at 0.05. Big brains (~15 neurons): MI reaches 0.62. The
hidden layer capacity prediction was correct - small hidden layers can't isolate
signaling from fleeing behavior.

**2. Brain expansion destroys then rebuilds semiotic structure.** The 46k-50k
explosion creates a ~25k-generation semiotic desert before information-carrying
signals re-evolve with the new architecture. The old 6-neuron signal strategy
doesn't transfer to 15 neurons.

**3. Iconicity flips sign.** Early evolution: negative iconicity (silence near
predator). During the MI surge: positive iconicity (+0.03 to +0.08). Big brains
learned to signal toward the predator instead of going silent. The semiotic
strategy fundamentally changed.

**4. Altruistic signaling pattern.** Pearson(MI, sender_fit_corr) = -0.67 in the
final 25k gens. When signals carry the most information, senders who signal more
have lower fitness. Classic altruistic alarm calling.

**5. Receiver asymmetry reveals information value.** jsd_no_pred/jsd_pred ratio =
10.4x during the MI surge. Receivers change behavior 10x more when receiving signals
in safe contexts vs near a predator. Near danger they're already fleeing; far from
danger, the signal provides genuinely new information.

### Symbol differentiation

At peak MI (gen 83,544), inter-symbol contrasts show functional differentiation:

| Pair | Contrast (JSD) |
|------|---------------|
| Symbol 0 vs 2 | 0.622 |
| Symbol 0 vs 1 | 0.449 |
| Symbol 1 vs 2 | 0.170 |

Symbols are not interchangeable - they encode different aspects of the environment.
This approaches Level 3 semiotic emergence (functional reference).

### Phase averages

| Phase | avg_fit | MI | iconicity | sender_fit | silence_corr | avg_hidden |
|-------|---------|------|-----------|------------|--------------|------------|
| Bootstrap (0-100) | 30.7 | 0.005 | +0.004 | -0.005 | -0.41 | 6.4 |
| Early (100-1k) | 47.6 | 0.014 | -0.006 | -0.003 | -0.40 | 7.5 |
| Middle (1k-5k) | 77.4 | 0.047 | -0.086 | +0.102 | -0.41 | 7.1 |
| Transition (5k-10k) | 98.9 | 0.043 | -0.117 | +0.319 | -0.42 | 7.2 |
| Maturation (10k-25k) | 89.8 | 0.016 | -0.042 | +0.119 | -0.42 | 7.9 |
| Late (25k-50k) | 81.7 | 0.009 | -0.002 | +0.051 | -0.43 | 8.9 |
| Big brain (50k-75k) | 81.1 | 0.008 | -0.005 | -0.010 | -0.42 | 12.5 |
| MI surge (75k-100k) | 82.8 | 0.216 | -0.024 | -0.043 | -0.42 | 15.2 |

### Peaks

| Metric | Peak value | Generation |
|--------|-----------|------------|
| Mutual information | 0.6242 | 83,544 |
| avg_fitness | 266.1 | 97,204 |
| max_fitness | 710.0 | 99,661 |
| traj_fluct_ratio | 13.35 | 12,200 |
| Negative iconicity | -0.292 | 87,355 |
| Negative silence_corr | -0.564 | 58,034 |

### Broken metrics at this scale (fixed)

`response_fit_corr`, `silence_onset_jsd`, `silence_move_delta` were effectively
broken in the 100k run - per-prey sample thresholds were too high for 384 prey
x 500 ticks. Fixed by lowering thresholds:
- `MIN_RECEIVER_SAMPLES`: 30 -> 10 (response_fit_corr)
- Silence onset threshold: 10 -> 5 (silence_onset_jsd, silence_move_delta)

### Brain expansion (post-100k)

MAX_HIDDEN raised from 16 to 124, DEFAULT_HIDDEN from 6 to 18. The 100k run
showed brain size naturally evolved to the MAX_HIDDEN=16 ceiling. With the new
ceiling, energy costs create natural selection pressure: at hidden_size=60,
drain = 0.002/tick = 1.0 over 500 ticks (all starting energy). Evolution is
free to find the optimal brain size under the Social Brain Hypothesis.

Genome is now 3108 floats (was 408). Two optimizations keep overhead manageable:

1. **Scoped mutation**: Only mutates weights for active neurons + 4 headroom.
   At hidden_size=18, mutates 22/124 neurons (~82% RNG savings).
2. **Zero-copy parent pool**: `evolve_spatial` reborrows sorted `scored` slice
   instead of cloning 384 agents (saves 4.8MB/gen of allocation).

Benchmark with new constants: 5.27s/100gen (~1,140 gens/min), ~18% slower than
old 4.31s/100gen. The overhead is from larger genome copies in crossover/elite
cloning and wider forward() stride (124 vs 16).

### Split-head architecture (v2)

Single hidden layer replaced with split-head topology:
- **Base hidden** (4-64 neurons, default 12): shared across movement, signal, and memory outputs
- **Signal hidden** (2-32 neurons, default 6): dedicated layer between base and signal outputs

Genome expanded to 5491 weights (was 3108) with 10 named segments. Two independent
hidden size genes, each with 5% mutation rate and +/-1 step. Scoped mutation now
operates per pool: base weights scoped by `base_hidden_size + 4`, signal weights
by `signal_hidden_size + 4`.

New features added alongside: 8-cell recurrent memory (EMA update), cooperative
food patches (50% require 2+ nearby prey), kin fitness via lineage tracking
(parent/grandparent indices, 0.5 siblings, 0.25 cousins), softmax signal emission.

Input count: 25 -> 36 (added food_dist, ally_dx/dy, 8 memory cells).
Output structure: 5 movement + 6 signal (via signal hidden) + 8 memory (tanh bounded).
CSV columns: 18 -> 21 (split brain stats: avg/min/max for base and signal hidden).

Neuron cost halved (0.00002 -> 0.00001) and vision range halved (11.2 -> 5.6 at
grid=56) to increase signal dependency. Signal range unchanged at 22.4 (4:1 ratio).

Early benchmarks (1000 gen, seed 42): ~1 sec/gen at 8x scale. Brain compression
observed: base hidden shrinks from 12 to ~4-5 by gen 700, signal hidden stays ~5-6.
Evolution finds it can survive with minimal base processing but retains signal capacity.

### Kill zones (replaces visible predators)

Visible predators replaced with invisible kill zones. KillZone struct uses f32 position
for sub-cell precision. Zones drift via probabilistic random walk (speed 0.5 = moves
~every other tick). Brain inputs 0-2 are dead (always zero) - prey sense danger only
through energy loss. Zone drain is 0.1 energy/tick inside zone radius, stacking across
overlapping zones.

No performance impact from the change: zone movement is simpler than predator chase AI
(random walk vs. nearest-prey targeting). The per-tick zone distance computation uses
the existing CellGrid infrastructure. Signal processing and brain forward pass are
unchanged.

## Optimization Phase 2 (2026-03-12)

Five optimizations applied in commit 7644ca7:

| Change | File | Mechanism |
|--------|------|-----------|
| Fat LTO + codegen-units=1 + panic=abort | Cargo.toml | Full cross-crate inlining, single codegen unit, no unwinding tables |
| CLT sum-of-4-uniforms gaussian | evolution.rs | Eliminates ln/sqrt/cos from ~431k calls/gen |
| Pade [1/1] fast_tanh (clamped) | brain.rs | Replaces 3 std tanh calls in forward(), ~2.6% max error |
| Action argmax + emit to parallel phase | world.rs | Moves ~115ns/prey/tick pure computation from sequential to par_iter |
| Sparse kin fitness via HashMap | main.rs | O(relatives) instead of O(N^2) for kin bonus |

### VPS benchmark (2026-03-13, Hetzner 12 vCPU AMD EPYC 7B13 Zen 3)

Built with `RUSTFLAGS="-C target-cpu=znver3" cargo build --release`.
100 gens, seed 42, --metrics-interval 10, standard 8x config.

**Thread scaling (new binary):**

| Threads | Wall time | Gens/min |
|---------|-----------|----------|
| 2 | 63.5s | 95 |
| 4 | 41.8s | 143 |
| 6 | 36.0s | 167 |
| 12 | 27.7s | 217 |

**A/B comparison (old commit fe35822 vs new 7644ca7, same hardware + znver3):**

| Threads | Old | New | Speedup |
|---------|-----|-----|---------|
| 4 | 52.3s (115 gens/min) | 41.8s (143 gens/min) | **25.2%** |
| 12 | 32.1s (187 gens/min) | 27.7s (217 gens/min) | **15.9%** |

25% at 4 threads, 16% at 12 threads. Lower gain at higher thread counts is expected
(Amdahl's law - sequential phase dominates more with more parallel workers).

Note: absolute gens/min are lower than the i7-12650H benchmarks above because
Hetzner shared vCPUs have lower per-core frequency than a laptop i7.

## Bottleneck Analysis (2026-03-10, samply profile)

### Per-tick cost breakdown

Each tick runs for ~384 prey (minus dead). 500 ticks per generation.

| Component | % of runtime | Location | Complexity |
|-----------|-------------|----------|------------|
| `receive_detailed` | 41% | signal.rs:54-89 | O(prey * active_signals) |
| `CellGrid::nearest` | 6.7% | world.rs:126-179 | O(ring_area), early exit |
| `tanh` | 6.4% | brain.rs:79-132 | ~30 calls/prey/tick |
| Metrics computation | ~7% | metrics.rs | Once/gen (or per metrics-interval) |
| Sorting (evolution) | ~2% | evolution.rs | Once/gen |
| Everything else | ~37% | world.rs step() | Metabolism, signals, food, zones, memory |

### Why receive_detailed dominates

Inner loop (signal.rs:69-87) runs for every (prey, signal) pair per tick:
- ~96 active signals per tick (decays over 4 ticks, ~48k signals/gen / 500 ticks)
- 384 prey * ~96 signals = ~37k distance computations per tick
- 500 ticks = ~18M distance computations per generation
- Each: 2x wrap_delta (modular arithmetic), 2x mul, add, compare, conditional sqrt+div
- At 50k generations: ~900 billion distance computations total

No spatial filtering - every prey checks every signal. This is the O(n^2) term
that makes population scaling worse than linear (see "Scaling with population" above).

### Parallelism status

The par_iter block (world.rs:454-477) parallelizes build_inputs_fast + brain.forward +
action argmax + emit decision across cores via rayon. This covers receive_detailed +
CellGrid::nearest + forward + pure output computation. The sequential apply phase
(world.rs:480-524) mutates world state (movement, food, signals, memory) and cannot be
parallelized. On 12-core Hetzner VPS, the parallel phase scales well since each
prey's computation is independent.

### Optimization roadmap

**High impact:**

1. **Signal spatial grid.** Same CellGrid pattern as prey_grid/food_grid, rebuilt
   each tick for active signals. receive_detailed would only check signals within
   signal_range cells instead of all signals. With 96 signals on a 56x56 grid,
   most prey would check ~5-15 signals instead of ~96. Expected: 3-8x on
   receive_detailed = 1.5-3x overall. Moderate implementation complexity.

2. **SIMD distance batch.** After spatial filtering, batch remaining candidates
   into SSE/AVX lanes (4-8 signals at once). Pure arithmetic inner loop is ideal
   for vectorization. The conditional strongest-per-symbol logic needs horizontal
   reduction. Expected: 2-4x on remaining receive_detailed work. High complexity.

3. **Signal grid + SIMD combined.** Spatial filter to ~15 candidates, SIMD batch
   the survivors. Expected: 5-10x on receive_detailed = 2-4x overall.

**Medium impact:**

4. **wrap_delta lookup table.** 56x56 = 3136 entries, fits L1 cache. Replaces
   two modular arithmetic ops with a table lookup per distance calc. Small
   constant-factor win.

5. **tanh approximation.** ~~Fast polynomial (Pade) or lookup table for the 6.4%
   spent in tanh.~~ DONE (Phase 2) - Pade [1/1] fast_tanh, ~2.6% max error.

6. **Batch brain forward.** Restructure as matrix multiply across all prey.
   Enables BLAS-style optimization. Requires genome layout changes for
   row-major access.

**Not worth it:**

7. **I/O decoupling (ring buffer, async writes).** I/O is not the bottleneck.
   CSV writes happen once per generation. runs.tsv is one line per run.

8. **GPU offload.** Complex branching in step() (alive checks, food, zones)
   maps poorly to GPU. Brain forward pass at 384x12 neurons is too small to
   amortize transfer overhead.

9. **target-cpu=native.** Already tested - no measurable difference. LLVM
   auto-vectorizes with SSE2, and hot loops are branch-heavy.

### Key files for optimization work

| File | Lines | What |
|------|-------|------|
| signal.rs | 54-89 | receive_detailed - THE hot loop |
| world.rs | 387-532 | step() - the tick |
| world.rs | 444-455 | par_iter block |
| world.rs | 534-585 | build_inputs_fast (calls receive_detailed) |
| world.rs | 86-179 | CellGrid (spatial index pattern to copy for signals) |
| brain.rs | 79-132 | forward() |
