# Performance

## Baseline

**Hardware (VPS):** Shared Linux VPS, 12 vCPU AMD EPYC 7B13 (Zen 3), 24 GB RAM.

**Binary:** `RUSTFLAGS="-C target-cpu=znver3" cargo build --release`. Fat LTO, codegen-units=1, panic=abort.

### v15 (current) - pop=384, grid=56, 8x defaults

**Config:** 384 prey, 56x56 grid, 3 flee zones + 2 freeze zones, 100 food, 500 ticks/gen, `--metrics-interval 10`.

**Speed (2026-03-25, commit 8fceec5, taskset -c 6-11, RAYON_NUM_THREADS=6):**

| Metric | Value |
|--------|-------|
| Baseline (pre-autooptimize) | 7.25 gps |
| After SIMD distance (#9) | ~7.88 gps (+10.4% A/B) |
| After SIMD pad-tail (#10) | ~7.88 gps (+8.7% A/B, cumulative) |

A/B benchmarking with interleaved runs, 7 measurement pairs per experiment, CPU-pinned to cores 6-11 on znver3.

### v7 - pop=1000, grid=72, demes=3

**Config:** 1000 prey, 72x72 grid, 18% zone coverage (flee + freeze), 100 food, 500 ticks/gen, `--metrics-interval 10`, `--demes 3`, `--signal-threshold 0.3`.

**Speed (2026-03-17, commit 5cf1d66, predates spatial tournament bucketing #8):**

| Run | Threads | Gen/min | Notes |
|-----|---------|---------|-------|
| v7-signals-42 | 9 | ~210 | Full signals + demes + death witness |
| v7-mute-42 | 3 | ~230 | --no-signals counterfactual |

Deme logic (assign_deme, group_selection every 100 gens, migrate every gen) adds negligible overhead - gated behind `if deme_divisions > 1`. Death witness input computation (linear scan of recent_zone_deaths, typically <50 entries) is similarly lightweight.

v7 vs v5 at same thread count: roughly comparable. The 39-input brain (vs 36) and death echo tracking add minimal cost. The throughput difference between signal and mute runs (~10%) reflects signal propagation overhead, consistent with v3 measurements (~35% of wall time in receive_detailed).

### v5 - pop=1000, grid=72

**Config:** 1000 prey, 72x72 grid, ~10 zones (30% coverage), 100 food, 500 ticks/gen, `--metrics-interval 10`.

**Speed (6 threads, 2026-03-14, commit cd04504):** ~125 gens/min (~2.1 gen/sec). Estimated 100k gens in ~13 hours.

### v4 (previous) - pop=1000, grid=72

**Speed (6 threads, commit 29c5f98):** ~43 gens/min (~0.72 gen/sec). 100k gens took ~39 hours.

**v5 vs v4:** 2.9x speedup from hot loop optimizations (#7 below).

### v3 - pop=384, grid=56

**Config:** 384 prey, 56x56 grid, 3 zones, 100 food, 500 ticks/gen, `--metrics-interval 10`.

**Speed (4 threads, overnight 2026-03-13, commit 29c5f98):**

| Run | Seed | Generations | Wall time | Gen/sec |
|-----|------|-------------|-----------|---------|
| baseline-s100 | 100 | 96,560 | 12.25 h | 2.19 |
| baseline-s101 | 101 | 95,270 | 12.24 h | 2.16 |
| mute-s100 (--no-signals) | 100 | 148,960 | 12.28 h | 3.37 |

Signal processing accounts for ~35% of wall time (receive_detailed inner loop).

**Thread scaling (100 gens, same binary):**

| Threads | Wall time | Gen/sec |
|---------|-----------|---------|
| 2 | 63.5 s | 1.57 |
| 4 | 41.8 s | 2.39 |
| 6 | 36.0 s | 2.78 |
| 12 | 27.7 s | 3.61 |

Diminishing returns past 6 threads - Amdahl's law, sequential apply phase dominates.

---

## Optimization Log

### 1. CellGrid spatial index (2026-03-10, 3c11557)

**107x speedup.** Before: ~13 gens/min. After: ~1,392 gens/min (i7-12650H, Windows).

Replaced O(n) linear scans for nearest ally, food, and prey with a `CellGrid` using Chebyshev ring search. Food removal switched from `remove` (O(n)) to `swap_remove` (O(1)). Nearest zone distance cached once per prey per tick (was computed 2-3x). Pre-allocated buffers for shuffled indices and position snapshots.

### 2. Rayon parallelism (2026-03-10, 3c11557)

Added `par_iter` for the compute phase (build_inputs + brain.forward). At 384 prey, per-item work (~10 us) barely amortizes thread scheduling. Marginal improvement on laptop, meaningful on 12-core VPS. The sequential apply phase (food, signals, movement, memory) cannot be parallelized.

### 3. Per-tick allocation reduction (2026-03-10, 3a84576)

Moved scratch buffers (shuffled indices, position snapshots) out of the tick loop. Eliminated per-tick heap allocations in `world.step()`.

### 4. Hot path optimization (2026-03-11, 1a09e90)

Parallelized input MI computation. Fixed O(n*m) signal rate calculation (was iterating all events per prey). Enabled `-C target-cpu=native` for SIMD. The input MI parallelization later had to be reverted (0b57d90) due to rayon thread contention crashes - metrics are now single-threaded.

### 5. Metrics interval decoupling (2026-03-12, 89a943a)

Added `--metrics-interval N` flag. Metrics computation (~7% of runtime) runs every N generations instead of every generation. Observer bookkeeping (SignalEvent collection, receiver tracking, per-prey action matrices) skipped entirely on non-metrics generations (f6f0c23). Buffered CSV I/O.

### 6. Phase 2 optimizations (2026-03-12, 7644ca7)

Five changes in one commit:

| Change | Mechanism |
|--------|-----------|
| Fat LTO + codegen-units=1 + panic=abort | Full cross-crate inlining, no unwinding tables |
| CLT sum-of-4-uniforms gaussian | Eliminates ln/sqrt/cos from ~431k calls/gen |
| Pade [1/1] fast_tanh | Replaces std tanh in forward(), ~2.6% max error |
| Action argmax + emit to parallel phase | Moves ~115 ns/prey/tick from sequential to par_iter |
| Sparse kin fitness via HashMap | O(relatives) instead of O(N^2) for kin bonus |

**A/B on VPS (old fe35822 vs new 7644ca7, znver3):**

| Threads | Old (gen/sec) | New (gen/sec) | Speedup |
|---------|---------------|---------------|---------|
| 4 | 1.92 | 2.39 | 25% |
| 12 | 3.12 | 3.61 | 16% |

Lower gain at higher thread counts is expected (sequential phase dominates more with more parallel workers).

### 7. Hot loop optimizations (2026-03-14, cd04504)

**2.9x speedup** at pop=1000 (43 -> 125 gens/min on VPS, 6 threads).

| Change | Mechanism |
|--------|-----------|
| PreyGrid flat array | Replaced `Vec<Vec<u16>>` per-cell allocations with contiguous prefix-sum layout (same pattern as SignalGrid). Eliminates pointer chasing and per-cell heap allocations on rebuild. |
| zone_drain sqrt skip | Compute `dist_sq` before checking zone containment. `sqrt()` only called for prey actually inside a zone (~70% of prey are outside). Saves ~700 sqrt/tick at pop=1000 with 3 zones. |
| Kin bonus buffer reuse | Moved `Vec<bool>` allocations (2x pop_size) outside the per-agent loop. `.fill(false)` replaces re-allocation. Eliminated ~2M allocations/gen at pop=1000. |
| Observer metric caching | Min-zone-distance extracted from parallel phase results instead of redundant O(n) loop with `nearest_zone_edge_dist` calls. |

All changes preserve deterministic behavior (same seed = same results).

### 8. Cache density + computation reduction (2026-03-14)

Six optimizations targeting per-tick overhead:

| Change | Mechanism |
|--------|-----------|
| Slim Prey struct | Removed Brain from Prey (~22KB -> ~120 bytes). Brains stored in separate `Vec<Brain>`. Per-prey loops now fit in L2 cache. |
| CompactBrain dense forward | Packs only active weights contiguously (~700 vs 5683). Dense indexing gives 100% cache utilization vs ~12% for sparse genome layout. |
| sqrt skip on non-metrics gens | `is_in_zone()` uses dist_sq comparison (no sqrt). Full `nearest_zone_edge_dist` only on metrics generations. |
| Conditional wrap (rem_euclid elimination) | `wrap_coord()` replaces `rem_euclid()` in all ring loops, movement, and cooperative food checks. Compiles to CMOV (~2 cycles) vs division (~20-40 cycles). |
| Spatial zone_drain | Iterates zones, queries prey_grid cells within bounding box. ~150 checks vs ~3000 (3 zones x 1000 prey). |
| Allocation-free tournament selection | Reservoir sampling replaces `Vec<usize>` collection. Single pass, no heap allocation. |

All changes preserve deterministic behavior (same seed = same results within version).

### 9. AVX2 SIMD 8-wide distance (2026-03-24, a327bd7)

**+10.4% throughput** (A/B VPS benchmark, SNR 3.5, paired analysis runs 5-7).

Replaces scalar per-signal distance computation in `receive_detailed_grid` with AVX2 8-wide path:
- Runtime `is_x86_feature_detected!("avx2")` dispatch, scalar fallback for non-AVX2
- Loads 8 signal x/y coords via i16->i32 sign-extend in single SIMD op
- Branchless `wrap_delta` using SIMD comparison + bitwise AND for toroidal wrapping
- `_mm256_movemask_ps` extracts in-range bitmask, scalar finalization only for hits
- Common `finish_reception()` extracted for both paths

### 10. Per-cell SIMD padding with sentinels (2026-03-26, a458550)

**+8.7% throughput** (A/B VPS benchmark, SNR 3.22).

Eliminates scalar tail loop from #9 by padding each grid cell to multiples of 8:
- `rebuild()` allocates `total_padded` (each cell rounded up to 8), fills padding with sentinels
- Sentinels: `sig_x`/`sig_y` = `i16::MAX` (fails distance check), `sig_sym` = `0xFF` (fails `sym < NUM_SYMBOLS`)
- AVX2 path uses `end_padded = start + ((len + 7) & !7)` - pure 8-wide iteration, no scalar tail
- 42-line scalar tail loop removed entirely

---

## Bottleneck Analysis (samply profile, 2026-03-10)

Profiled at 384 prey / 56x56 grid. Relative weights shift at higher populations due to O(prey * signals) scaling in signal reception.

| Component | % runtime | Location | Complexity |
|-----------|-----------|----------|------------|
| `receive_detailed_grid` | 41% | signal.rs:157-271 | O(prey * nearby_signals) |
| `CellGrid::nearest` | 6.7% | world.rs:120-173 | O(ring_area), early exit |
| `tanh` | 6.4% | brain.rs (now fast_tanh) | ~26 calls/prey/tick |
| Metrics | ~7% | metrics.rs | Once per metrics-interval |
| Evolution sort | ~2% | evolution.rs | Once/gen |
| Everything else | ~37% | world.rs step() | Metabolism, signals, food, zones, memory |

### Why receive_detailed dominates

Signal reception uses `SignalGrid` spatial index for O(nearby) lookup instead of O(all). Each prey checks only signals in nearby grid cells (flat buffer with prefix-sum offsets). Still dominates because signal density is high relative to cell size. ~96 active signals per tick (4-tick persistence, configurable via `--signal-ticks`). Each check: 2x wrap_delta (inlined), 2x mul, add, compare, deferred sqrt (only 6 winners).

### Scaling with population

384 to 2,000 prey (5.2x) costs ~10x in eval time. 2,000 to 5,000 (2.5x) costs ~5.4x. Worse than linear because signal reception is O(prey * active_signals) and more prey emit more signals.

---

## Optimization Roadmap

### Done

1. ~~**Signal spatial grid.**~~ Implemented as `SignalGrid` with SoA layout, prefix-sum offsets, ring search.
2. ~~**Slim Prey struct.**~~ Brain moved to `World.brains` vec. Prey is ~120 bytes, fits in L2.
3. ~~**CompactBrain dense forward.**~~ Packs active weights contiguously (~700 vs 5683). 100% cache utilization.
4. ~~**sqrt skip on non-metrics gens.**~~ `is_in_zone()` uses dist_sq, no sqrt. Full distance only on metrics gens.
5. ~~**rem_euclid -> conditional wrap.**~~ `wrap_coord()` in all ring loops, movement, food checks. CMOV vs division.
6. ~~**Spatial zone_drain.**~~ Queries prey_grid cells within zone bounding box. ~150 checks vs ~3000.
7. ~~**Allocation-free tournament selection.**~~ Reservoir sampling, no Vec heap allocation.
8. ~~**Spatial bucketing for tournament selection.**~~ (651b7b2) O(nearby) parent lookup via prey_grid instead of O(N) linear scan. Fixed cell size to target radius/3 for effective ring search (5940345).

9. ~~**AVX2 SIMD 8-wide distance in receive_detailed_grid.**~~ (autoopt/004, a327bd7) Runtime AVX2 feature detection. Inner loop processes 8 signals simultaneously: i16->i32 sign-extend, branchless wrap_delta via SIMD comparison + bitwise AND, f32 dist_sq, movemask extraction. Scalar fallback for non-AVX2 CPUs. +10.4% A/B on VPS (SNR 3.5).
10. ~~**Per-cell SIMD padding with sentinels.**~~ (autoopt/007, a458550) Pads each SignalGrid cell's SoA arrays to multiples of 8 with i16::MAX/0xFF sentinels (always fail distance and symbol checks). Eliminates 42-line scalar tail from the AVX2 path - pure 8-wide iteration throughout. +8.7% A/B on VPS (SNR 3.22).

### Medium impact

11. **Batch brain forward.** Restructure as matrix multiply across all prey. Enables BLAS-style optimization. Requires genome layout changes for row-major access.

### Not worth it

12. **I/O decoupling.** CSV writes happen once per generation. Not the bottleneck.

13. **GPU offload.** Branch-heavy step() maps poorly to GPU. Brain forward at 1000x12 neurons is too small to amortize transfer overhead.

14. **target-cpu=native on laptop.** Tested on i7-12650H (AVX2). No measurable difference - LLVM auto-vectorizes with SSE2, hot loops are branch-heavy not SIMD-friendly. (VPS znver3 targeting does help via Zen 3 specific scheduling.)

15. **wrap_delta lookup table.** With conditional wrap replacing `rem_euclid`, the LUT approach is unnecessary. The branch predictor handles the conditional well.

16. **Branchless signal reception.** (dd01820) Two-pass approach: first pass collects candidates without branches, second pass selects winners. 7% regression - loss of early single-axis bailout and temp buffer overhead outweighed branch elimination. Well-predicted branches win here.

17. **Per-symbol signal index.** (ea3dc97, reverted fd4f0d5) Counting-sort signals by symbol within each grid cell, enabling 6 independent ring scans with per-symbol early exit. 23% regression at pop=2000 - 6x cell-iteration overhead from separate ring traversals per symbol outweighs reduced distance computations at high signal density.

18. **Brain SIMD padding.** (autoopt/008, discarded) Pad CompactBrain hidden layer from bh=12 to padded_bh=16 for scalar tail elimination. -7.1% regression. For bh=12, LLVM auto-vectorizes efficiently (1 AVX2 8-wide + 4 scalar). Padding adds 33% wasted iterations including unnecessary fast_tanh calls.

19. **Integer-only distance comparison.** (autoopt/001, discarded) Replace f32 distance with i32 in inner loop. +0.7% - in the noise.

20. **Flat scan full grid.** (autoopt/002, discarded) Skip ring iteration when search covers entire grid. -2.8% regression - ring early exit saves more than flat scan contiguous access gains.

21. **Fuse output heads.** (autoopt/003, discarded) Fuse 4 brain output head loops into single 20-wide loop. +0.9% - in the noise. base_hidden fits L1 cache line, no reload savings.

22. **wrap_coord in rebuild.** (autoopt/005, discarded) Replace rem_euclid with wrap_coord in SignalGrid::rebuild. Inconclusive (SNR 0.42).

23. **Split prey hot/cold.** (autoopt/006, discarded) Split Prey into hot (~50B) and cold (~180B) structs for cache density. Inconclusive after 14 paired A/B runs.

---

## Notes

**Bash `time` on Windows is broken.** Git Bash's `time` builtin adds ~37 s of fixed overhead. Use PowerShell `Measure-Command` for accurate timing.

**I/O overhead.** PowerShell `Tee-Object` piping costs ~2.6x (535 gens/min measured vs 1,392 benchmark on i7-12650H). VPS runs write directly to file, avoiding this.

**Laptop vs VPS.** i7-12650H (6P+4E cores) measures higher gens/min than shared VPS vCPUs at the same thread count due to higher per-core frequency. VPS numbers are the authoritative baseline for long runs.

**Determinism.** All optimizations preserve deterministic behavior. Same seed produces identical results regardless of optimization level. This is verified by regression tests.
