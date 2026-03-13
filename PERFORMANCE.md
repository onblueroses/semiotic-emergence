# Performance

## Baseline

**Hardware (VPS):** Hetzner 12 vCPU AMD EPYC 7B13 (Zen 3), shared. 24 GB RAM.

**Binary:** `RUSTFLAGS="-C target-cpu=znver3" cargo build --release`. Fat LTO, codegen-units=1, panic=abort.

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

---

## Bottleneck Analysis (samply profile, 2026-03-10)

500 ticks per generation, 384 prey per tick.

| Component | % runtime | Location | Complexity |
|-----------|-----------|----------|------------|
| `receive_detailed` | 41% | signal.rs:54-89 | O(prey * active_signals) |
| `CellGrid::nearest` | 6.7% | world.rs:126-179 | O(ring_area), early exit |
| `tanh` | 6.4% | brain.rs:79-132 | ~30 calls/prey/tick |
| Metrics | ~7% | metrics.rs | Once per metrics-interval |
| Evolution sort | ~2% | evolution.rs | Once/gen |
| Everything else | ~37% | world.rs step() | Metabolism, signals, food, zones, memory |

### Why receive_detailed dominates

Every (prey, signal) pair checked every tick. ~96 active signals per tick (4-tick persistence), 384 prey = ~37k distance computations per tick. Over 500 ticks that is ~18M per generation. Each computation: 2x wrap_delta, 2x mul, add, compare, conditional sqrt+div. No spatial filtering - every prey checks every signal.

### Scaling with population

384 to 2,000 prey (5.2x) costs ~10x in eval time. 2,000 to 5,000 (2.5x) costs ~5.4x. Worse than linear because signal reception is O(prey * active_signals) and more prey emit more signals.

---

## Optimization Roadmap

### High impact

1. **Signal spatial grid.** Same CellGrid pattern, rebuilt each tick for active signals. receive_detailed checks only signals within signal_range instead of all signals. With 96 signals on a 56x56 grid, most prey would check ~5-15 instead of ~96. Expected: 3-8x on receive_detailed, 1.5-3x overall.

2. **SIMD distance batch.** After spatial filtering, batch remaining candidates into SSE/AVX lanes (4-8 at once). Pure arithmetic inner loop is ideal for vectorization. Expected: 2-4x on remaining receive_detailed work.

3. **Signal grid + SIMD combined.** Spatial filter to ~15 candidates, SIMD batch the survivors. Expected: 5-10x on receive_detailed, 2-4x overall.

### Medium impact

4. **wrap_delta lookup table.** 56x56 = 3,136 entries, fits L1 cache. Replaces modular arithmetic with table lookup.

5. **Batch brain forward.** Restructure as matrix multiply across all prey. Enables BLAS-style optimization. Requires genome layout changes for row-major access.

### Not worth it

6. **I/O decoupling.** CSV writes happen once per generation. Not the bottleneck.

7. **GPU offload.** Branch-heavy step() maps poorly to GPU. Brain forward at 384x12 neurons is too small to amortize transfer overhead.

8. **target-cpu=native on laptop.** Tested on i7-12650H (AVX2). No measurable difference - LLVM auto-vectorizes with SSE2, hot loops are branch-heavy not SIMD-friendly. (VPS znver3 targeting does help via Zen 3 specific scheduling.)

---

## Notes

**Bash `time` on Windows is broken.** Git Bash's `time` builtin adds ~37 s of fixed overhead. Use PowerShell `Measure-Command` for accurate timing.

**I/O overhead.** PowerShell `Tee-Object` piping costs ~2.6x (535 gens/min measured vs 1,392 benchmark on i7-12650H). VPS runs write directly to file, avoiding this.

**Laptop vs VPS.** i7-12650H (6P+4E cores) measures higher gens/min than Hetzner shared vCPUs at the same thread count due to higher per-core frequency. VPS numbers are the authoritative baseline for long runs.
