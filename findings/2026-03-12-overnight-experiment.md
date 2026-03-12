# Overnight Experiment: Do Signals Have Adaptive Value?

Date: 2026-03-12
Status: PLANNED (not yet deployed)
VPS: 12 vCPU AMD EPYC Genoa (AVX-512), 22GB RAM, 87GB disk free

## The Question

Before investigating whether prey evolve "real" communication (reference, symbols
with meaning, receiver-appropriate responses), we need to answer the prerequisite:
**do signals matter at all for fitness?**

If signals don't improve survival, there's nothing to study. If they do, the size
and character of the effect tells us what to investigate next.

## Experimental Design

Three runs, same binary, same parameters except the signal channel:

| Run | Seed | Flag | Threads | Purpose |
|-----|------|------|---------|---------|
| baseline-s100 | 100 | (none) | 4 | Signals enabled - the treatment |
| mute-s100 | 100 | --no-signals | 4 | Signals disabled - the control |
| baseline-s101 | 101 | (none) | 4 | Replication seed |

All runs: `--metrics-interval 10`, default 8x params (384 prey, 56x56 grid,
3 zones radius 8.0 speed 0.5, 100 food, 500 ticks).

Generation target: 999999999 (run until killed). Expect 500k+ overnight depending
on gen/sec rate (benchmark before launching).

### Why these three

**baseline-s100 vs mute-s100** is the core comparison. Same seed means identical
initial population and zone positions. Only difference: whether signals propagate.
If baseline fitness diverges above mute, signals have adaptive value. If they track
together, signals are noise.

**baseline-s101** tests reproducibility. If both baselines hit similar epoch structure
(CUSUM change points at similar gen counts), the dynamics are deterministic - driven
by fitness landscape geometry. If they diverge, initial conditions matter and we need
more seeds.

### What `--no-signals` does

The flag suppresses signal EMISSION in `apply_outputs` (world.rs:670). The brain
still computes signal outputs via `forward()` every tick - it just never emits them.
This means:

- Signal hidden layer weights evolve under pure drift (no selection pressure through
  the signal channel)
- Base hidden weights shared between movement and signal heads are still selected on
- The mute population saves ~0.0005 energy/tick by not paying signal cost (0.002 per
  emission at ~0.25 emissions/prey/tick). This is 62% of base drain (0.0008/tick).
- Over long runs, expect signal_hidden to drift toward minimum (2) in mute - this
  itself is evidence the signal channel exerts selection pressure in baseline.

### Known confounds

**Energy confound**: Mute prey save signal cost. A small baseline fitness advantage
could be "signals help but cost offsets benefit." A large advantage is strong evidence.
If mute wins, signals are net negative regardless of mechanism.

**Motor noise confound**: Signal emission changes energy, which changes input 35
(own energy), which changes future brain behavior. The mute run lacks this feedback
loop. This is acceptable - we want the TOTAL effect of the signal channel, not just
the information channel in isolation.

**RNG divergence**: Same seed but the runs diverge within the first few ticks because
different survival patterns change the shuffle order, which changes the RNG stream.
We're testing "do signals change the evolutionary trajectory?" not "tick-for-tick
what changes?" This is the right test.

## Thread Allocation Rationale

Profiling shows ~54% of per-gen runtime is in the Rayon par_iter block (receive_detailed
41%, CellGrid::nearest 6.7%, tanh 6.4%). The sequential portion (apply_outputs, zone
movement, sorting) dominates at high thread counts.

By Amdahl's law:
- 4 threads/run: ~1.7x speedup over single-threaded
- 3 threads/run: ~1.6x speedup
- 6 threads/run: ~1.8x (diminishing returns)

With collect_metrics optimization, non-metrics gens (90% of all gens) have a higher
parallel fraction, so effective speedup is closer to 1.9x at 4 threads.

**3 runs x 4 threads = 12 = all cores.** Clean fit, near-optimal per-run throughput.
A 4th run at 3 threads each would cost ~10% per-run depth for 33% more experimental
coverage - not worth it for an exploratory first overnight.

## Metrics Interval Rationale

Measured: metrics-interval 10 gives 7.4% speedup over interval 1. Going to interval
50 or 100 gains only ~0.6% more - the observer bookkeeping gated by `collect_metrics`
is only ~8% of per-gen cost, and at interval 10 we already skip 90% of it.

At interval 10 with 500k gens: 50,000 data points. Plenty for CUSUM change-point
detection, epoch classification, and trend analysis.

**Interval 10 is near-optimal.** Higher intervals barely help but lose resolution.

## Stopping Strategy

**Minimum useful**: 50k gens (matches salvaged data depth - can compare directly).
**Good overnight**: 200k+ gens (enough for long-period dynamics to appear).
**Ideal**: 500k+ gens (reveals whether oscillations stabilize or are steady-state).

Don't kill runs in the morning unless they need to be restarted. Let them continue
running during the day if VPS isn't needed for other work. More depth is always better.

## Benchmark Notes

The 1000-gen benchmark before launch serves three purposes:

1. **Overnight yield estimate**: gen/sec * 36000 (10 hours) = expected depth
2. **Mute speed difference**: The mute run should be measurably faster because no
   signals exist in the world for receive_detailed to iterate over (the inner loop
   is O(alive_prey * active_signals) - zero signals means trivial reception). If mute
   is e.g. 15% faster per gen, that's information about how much of the simulation's
   computational cost comes from the signal channel itself.
3. **Sanity check**: If gen/sec is unexpectedly low, something is wrong with the build.

To get the mute comparison:
```bash
time RAYON_NUM_THREADS=4 ./target/release/semiotic-emergence 100 1000 --metrics-interval 10
time RAYON_NUM_THREADS=4 ./target/release/semiotic-emergence 100 1000 --metrics-interval 10 --no-signals
```

Record both times. The ratio tells us how much wall time the signal channel consumes.

## Deployment Steps

```bash
# 1. Update binary (VPS is 2 commits behind: needs signal_entropy + collect_metrics)
cd ~/semiotic-emergence
git pull origin main
source ~/.cargo/env
cargo build --release

# 2. Benchmark (critical: establishes gen/sec for overnight yield estimate)
time RAYON_NUM_THREADS=4 ./target/release/semiotic-emergence 100 1000 --metrics-interval 10
# Record: real time for 1000 gens. Calculate gen/sec.
# Expected overnight yield: gen/sec * 3600 * hours_overnight

# 3. Launch runs (order doesn't matter, launch all three quickly)
RAYON_NUM_THREADS=4 ./launch.sh baseline-s100 100 999999999 --metrics-interval 10
RAYON_NUM_THREADS=4 ./launch.sh mute-s100 100 999999999 --no-signals --metrics-interval 10
RAYON_NUM_THREADS=4 ./launch.sh baseline-s101 101 999999999 --metrics-interval 10

# 4. Verify all running
./status.sh

# 5. Spot-check after 5 minutes
wc -l runs/baseline-s100/output.csv runs/mute-s100/output.csv runs/baseline-s101/output.csv
tail -1 runs/baseline-s100/run.log
tail -1 runs/mute-s100/run.log
tail -1 runs/baseline-s101/run.log
```

## Morning Analysis Plan

### Getting the data

Analysis can run on VPS (if numpy/matplotlib installed) or locally. To pull data:
```bash
# From local machine (PowerShell SSH)
scp -r root@<VPS_IP>:~/semiotic-emergence/runs/baseline-s100/output.csv ./runs/baseline-s100/
scp -r root@<VPS_IP>:~/semiotic-emergence/runs/mute-s100/output.csv ./runs/mute-s100/
scp -r root@<VPS_IP>:~/semiotic-emergence/runs/baseline-s101/output.csv ./runs/baseline-s101/
# Add trajectory.csv and input_mi.csv if doing full analysis
```

### Quick triage (5 minutes)

```bash
# Are runs still alive?
./status.sh

# How far did we get?
tail -1 runs/baseline-s100/run.log
tail -1 runs/mute-s100/run.log
tail -1 runs/baseline-s101/run.log
```

### Headline result: does the signal channel have adaptive value?

```bash
python analyze.py runs/baseline-s100/output.csv \
    --counterfactual runs/mute-s100/output.csv --plot
```

This produces:
- Overlaid fitness curves with shaded delta region
- Integral of fitness difference (the "counterfactual signal value")
- CUSUM change-point detection on both runs

**Interpreting the result:**
- Baseline fitness clearly above mute → signals have adaptive value (Level 1 confirmed)
- Baseline fitness ≈ mute → signals are neutral (may still need more gens or more pressure)
- Baseline fitness below mute → signals are net costly (energy waste exceeds information value)

### Brain architecture divergence

Compare `avg_signal_hidden` across all three runs. Key question: does the mute run's
signal_hidden collapse to minimum (2)?

If yes: the signal channel exerts selection pressure on brain architecture in baseline.
If no: signal_hidden is maintained by indirect selection through shared base weights.

Also compare `avg_base_hidden` - does the mute population compensate with larger base?

### Reproducibility check

```bash
python analyze.py runs/baseline-s100/output.csv runs/baseline-s101/output.csv --plot
```

Compare epoch structure, oscillation periods, and final metric values between the two
baseline seeds.

### Detailed signal analysis (if Level 1 is confirmed)

```bash
# Full analysis with trajectory and input MI
python analyze.py runs/baseline-s100/output.csv \
    --all runs/baseline-s100/trajectory.csv runs/baseline-s100/input_mi.csv --plot

# What are signals encoding?
# Check input_mi.csv: which input dimensions have highest MI with signal symbol?
# If mi_energy is high: signals encode danger state
# If mi_food_dx/dy is high: signals encode food location
# If mi_sig*_str is high: signals are relaying other signals (gossip)
```

### What to investigate next (depends on overnight results)

**If signals have clear adaptive value:**
1. Experiment B: Take evolved baseline-s100 population at final gen, run 1000 gens
   with --no-signals. If fitness drops, the population DEPENDS on signals (not just
   coincidental).
2. MI(symbol; action | context) - do receivers take the RIGHT action for each symbol?
3. Per-lineage MI - do kin groups develop dialects?

**If signals are neutral:**
1. Increase zone pressure (--zone-coverage 0.15 or higher) to force communication
2. Increase signal cost (test whether removing cost makes signals emerge)
3. Consider whether the brain architecture can support communication (is base_hidden
   large enough to separate signal processing from movement?)

**If signals are net negative:**
1. The energy cost exceeds information value at current parameters
2. Try signal_cost = 0.0 to test pure information value without economic pressure
3. Consider whether zone speed/radius creates enough urgency for signaling

## Evidence Hierarchy

For reference: the levels of evidence this project is building toward.

| Level | Claim | Evidence needed | Status |
|-------|-------|----------------|--------|
| 1 | Signals have adaptive value | baseline vs mute fitness divergence | **This experiment** |
| 2 | Receivers change behavior | JSD(action\|signal vs action\|no_signal) > 0 | Already measured (jsd_pred ~6x jsd_no_pred in 50k data) |
| 3 | Different symbols carry different info | inter-symbol JSD > 0, input MI shows symbol-specific encoding | Partially measured (contrast telescope) |
| 4 | Responses are appropriate | MI(symbol; action \| context) | Not yet implemented |
| 5 | Genuine reference | Stable symbol-referent mapping across generations, robust to perturbation | Needs Levels 1-4 first |

The 50k salvaged data already showed promising Level 2/3 signals (jsd_pred ~6x baseline,
sender_fit=-0.39/receiver_fit=+0.75 consistent with kin selection). But without Level 1,
those could be spandrels - correlated with fitness but not causal.

## Key Metrics to Track

| Metric | What it tells us | Where | Watch for |
|--------|-----------------|-------|-----------|
| avg_fitness | Overall population health | output.csv col 2 | Divergence between baseline and mute |
| zone_deaths | Survival under zone pressure | output.csv col 22 | Fewer in baseline = signals help zone avoidance |
| signal_entropy | Symbol convergence | output.csv col 23 | Decreasing = population converging on specific symbols |
| avg_signal_hidden | Brain architecture | output.csv cols 19 | Collapse in mute = signal channel under selection |
| avg_base_hidden | Brain architecture | output.csv col 16 | Compensation if signal_hidden changes |
| mutual_info | Sender-world correlation | output.csv col 6 | Increasing = signals encoding zone info |
| response_fit_corr | Receiver benefit | output.csv col 13 | Positive = behavioral response to signals predicts fitness |
| silence_corr | Behavioral silence | output.csv col 9 | Negative = prey suppress signals in zones |

## Files Produced Per Run

| File | Size estimate | Content |
|------|--------------|---------|
| output.csv | ~10MB at 500k gens | 23 columns, one row per 10 gens |
| trajectory.csv | ~20MB | 47 columns (6x4 signal-context matrix + per-symbol JSD + contrasts) |
| input_mi.csv | ~15MB | 37 columns (MI between each input dimension and signal symbol) |
| run.log | ~50MB | Console output, one line per 10 gens |
| meta.txt | <1KB | Git commit, binary hash, command, timestamp |
| pid.txt | <1KB | Process ID |

Total across 3 runs: ~300MB. Trivial against 87GB free.
