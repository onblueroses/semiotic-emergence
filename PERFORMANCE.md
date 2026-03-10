# Performance Notes

## Benchmarks (2026-03-10, i7-12650H, Windows 11, release+LTO)

### Per-generation phase breakdown (500 ticks/gen)

| Pop | Grid | Pred | Food | eval | metrics | evolve | total/100g | gens/min |
|-----|------|------|------|------|---------|--------|------------|----------|
| 384 | 56 | 16 | 200 | 3.43s | 0.20s | 0.46s | 4.31s | 1,392 |
| 2000 | 128 | 80 | 1040 | 37.04s | 0.71s | 5.58s | 43.55s | 138 |
| 5000 | 200 | 200 | 2600 | 39.80s | 0.35s | 6.67s | 46.95s/20g | 25.6 |

Eval (the 500-tick simulation loop) dominates at all sizes (~85% of time).
Evolve_spatial is the second cost center (~12%), scaling with population due to
spatial tournament selection with distance checks.

### Projected run times at 8x scale (384 prey)

| Generations | Time |
|-------------|------|
| 10,000 | ~7 min |
| 50,000 | ~36 min |
| 100,000 | ~72 min |
| 500,000 | ~6 hours |
| 1,000,000 | ~12 hours |

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
- Nearest predator cached once per prey per tick (was computed 2-3x)
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
