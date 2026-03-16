"""Deep analysis of semiotic-emergence v5 runs at 110k+ gens."""
import numpy as np
from pathlib import Path

def load(path):
    raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    with open(path) as f:
        cols = f.readline().strip().split(",")
    return raw, cols

def col(data, cols, name):
    return data[:, cols.index(name)]

def window_avg(arr, w=500):
    """Rolling average with window w."""
    cs = np.cumsum(arr)
    cs = np.insert(cs, 0, 0)
    n = len(arr)
    result = np.zeros(n)
    for i in range(n):
        lo = max(0, i - w // 2)
        hi = min(n, i + w // 2)
        result[i] = (cs[hi] - cs[lo]) / (hi - lo)
    return result

def analyze_seed(name, out_path, traj_path, mi_path):
    print(f"\n{'='*70}")
    print(f"  DEEP ANALYSIS: {name}")
    print(f"{'='*70}")

    out, ocols = load(out_path)
    traj, tcols = load(traj_path)
    mi, mcols = load(mi_path)

    gen = col(out, ocols, "generation")
    fitness = col(out, ocols, "avg_fitness")
    mutual = col(out, ocols, "mutual_info")
    sig_hidden = col(out, ocols, "avg_signal_hidden")
    base_hidden = col(out, ocols, "avg_base_hidden")
    signals_emitted = col(out, ocols, "signals_emitted")
    silence_corr = col(out, ocols, "silence_corr")
    response_fit = col(out, ocols, "response_fit_corr")
    jsd_pred = col(out, ocols, "jsd_pred")
    jsd_no_pred = col(out, ocols, "jsd_no_pred")
    zone_deaths = col(out, ocols, "zone_deaths")
    signal_entropy = col(out, ocols, "signal_entropy")
    receiver_fit = col(out, ocols, "receiver_fit_corr")
    n = len(gen)

    # =====================================================================
    # 1. SIGNAL HIDDEN OSCILLATION ANALYSIS
    # =====================================================================
    print("\n  1. SIGNAL HIDDEN DYNAMICS")
    sig_smooth = window_avg(sig_hidden, 200)
    # Find local extrema
    peaks = []
    troughs = []
    for i in range(1, len(sig_smooth) - 1):
        if sig_smooth[i] > sig_smooth[i-1] and sig_smooth[i] > sig_smooth[i+1]:
            peaks.append((gen[i], sig_smooth[i]))
        if sig_smooth[i] < sig_smooth[i-1] and sig_smooth[i] < sig_smooth[i+1]:
            troughs.append((gen[i], sig_smooth[i]))

    if peaks and troughs:
        peak_vals = [p[1] for p in peaks]
        trough_vals = [t[1] for t in troughs]
        print(f"    Oscillation cycles: {len(peaks)} peaks, {len(troughs)} troughs")
        print(f"    Peak range: {min(peak_vals):.1f} - {max(peak_vals):.1f}")
        print(f"    Trough range: {min(trough_vals):.1f} - {max(trough_vals):.1f}")
        if len(peaks) >= 2:
            intervals = [peaks[i+1][0] - peaks[i][0] for i in range(len(peaks)-1)]
            print(f"    Avg cycle length: {np.mean(intervals):.0f} gens")
            print(f"    Cycle length range: {min(intervals):.0f} - {max(intervals):.0f} gens")

    # Does signal hidden size predict FUTURE fitness?
    offsets = [50, 100, 200, 500, 1000]
    print("\n    Predictive power of signal_hidden on future fitness:")
    for off in offsets:
        if off < n:
            r = np.corrcoef(sig_hidden[:n-off], fitness[off:])[0, 1]
            print(f"      +{off:4d} gens ahead: r={r:.3f}")

    # =====================================================================
    # 2. SIGNAL ECONOMY
    # =====================================================================
    print("\n  2. SIGNAL ECONOMY")
    # Signal emission rate over time
    q1 = signals_emitted[:n//4]
    q2 = signals_emitted[n//4:n//2]
    q3 = signals_emitted[n//2:3*n//4]
    q4 = signals_emitted[3*n//4:]
    print("    Signals emitted per gen (quartiles):")
    print(f"      Q1 (early):  {np.mean(q1):.1f} +/- {np.std(q1):.1f}")
    print(f"      Q2:          {np.mean(q2):.1f} +/- {np.std(q2):.1f}")
    print(f"      Q3:          {np.mean(q3):.1f} +/- {np.std(q3):.1f}")
    print(f"      Q4 (late):   {np.mean(q4):.1f} +/- {np.std(q4):.1f}")

    # Signal entropy (effective vocabulary size)
    print(f"\n    Signal entropy (bits, max={np.log2(6):.2f} for 6 symbols):")
    print(f"      Early (first 10%): {np.mean(signal_entropy[:n//10]):.3f}")
    print(f"      Mid (40-60%):      {np.mean(signal_entropy[2*n//5:3*n//5]):.3f}")
    print(f"      Late (last 10%):   {np.mean(signal_entropy[9*n//10:]):.3f}")

    eff_vocab_late = 2 ** np.mean(signal_entropy[9*n//10:])
    print(f"      Effective vocabulary (late): {eff_vocab_late:.1f} symbols")

    # =====================================================================
    # 3. MI SPIKE FORENSICS
    # =====================================================================
    print("\n  3. MI SPIKE FORENSICS")
    mi_smooth = window_avg(mutual, 50)
    threshold = np.percentile(mi_smooth, 95)
    spikes = mi_smooth > threshold
    spike_regions = []
    in_spike = False
    start = 0
    for i in range(len(spikes)):
        if spikes[i] and not in_spike:
            start = i
            in_spike = True
        elif not spikes[i] and in_spike:
            spike_regions.append((start, i))
            in_spike = False
    if in_spike:
        spike_regions.append((start, len(spikes)))

    print(f"    95th percentile MI threshold: {threshold:.4f}")
    print(f"    Number of MI spike episodes: {len(spike_regions)}")
    for j, (s, e) in enumerate(spike_regions[:8]):
        dur = gen[min(e, n-1)] - gen[s]
        peak_mi = np.max(mi_smooth[s:e])
        avg_fit = np.mean(fitness[s:e])
        avg_sig_h = np.mean(sig_hidden[s:e])
        avg_sil = np.mean(silence_corr[s:e])
        avg_entropy = np.mean(signal_entropy[s:e])
        print(f"    Spike {j+1}: gen {gen[s]:.0f}-{gen[min(e,n-1)]:.0f} ({dur:.0f} gens)")
        print(f"      peak MI={peak_mi:.4f}, fitness={avg_fit:.1f}, sig_h={avg_sig_h:.1f}, "
              f"silence_corr={avg_sil:.3f}, entropy={avg_entropy:.3f}")

    # What happens to fitness AFTER MI spikes?
    print("\n    Fitness trajectory after MI spikes:")
    for j, (s, e) in enumerate(spike_regions[:5]):
        pre_fit = np.mean(fitness[max(0, s-100):s]) if s > 100 else np.mean(fitness[:s]) if s > 0 else 0
        during_fit = np.mean(fitness[s:e])
        post_starts = [100, 500, 1000]
        post_fits = []
        for ps in post_starts:
            pe = min(e + ps, n)
            if pe > e:
                post_fits.append(f"+{ps}: {np.mean(fitness[e:pe]):.1f}")
        print(f"    Spike {j+1}: pre={pre_fit:.1f}, during={during_fit:.1f}, {', '.join(post_fits)}")

    # =====================================================================
    # 4. JSD PRED vs NO-PRED GAP (signal-dependent behavior near zones)
    # =====================================================================
    print("\n  4. BEHAVIORAL DIFFERENTIATION (JSD gap)")
    jsd_gap = jsd_pred - jsd_no_pred
    print("    JSD(pred) - JSD(no_pred) = signal-specific behavior near zones")
    print(f"    Overall avg gap: {np.mean(jsd_gap):.4f}")
    print(f"    Early gap:       {np.mean(jsd_gap[:n//10]):.4f}")
    print(f"    Mid gap:         {np.mean(jsd_gap[2*n//5:3*n//5]):.4f}")
    print(f"    Late gap:        {np.mean(jsd_gap[9*n//10:]):.4f}")
    print(f"    Max gap:         {np.max(jsd_gap):.4f} at gen {gen[np.argmax(jsd_gap)]:.0f}")
    # Correlation of gap with other metrics
    r_gap_mi = np.corrcoef(jsd_gap, mutual)[0, 1]
    r_gap_fit = np.corrcoef(jsd_gap, fitness)[0, 1]
    r_gap_sig = np.corrcoef(jsd_gap, sig_hidden)[0, 1]
    print(f"    Corr(gap, MI):      {r_gap_mi:.3f}")
    print(f"    Corr(gap, fitness): {r_gap_fit:.3f}")
    print(f"    Corr(gap, sig_h):   {r_gap_sig:.3f}")

    # =====================================================================
    # 5. INPUT MI: WHAT THE BRAIN ACTUALLY ENCODES
    # =====================================================================
    print("\n  5. INPUT ENCODING DEEP DIVE")
    mi_gen = col(mi, mcols, "generation")
    n_mi = len(mi_gen)

    # Signal strength encoding vs signal direction encoding
    sig_str_cols = [c for c in mcols if c.startswith("mi_sig") and c.endswith("_str")]
    sig_dir_cols = [c for c in mcols if c.startswith("mi_sig") and (c.endswith("_dx") or c.endswith("_dy"))]
    mem_cols = [c for c in mcols if c.startswith("mi_mem")]

    str_total = sum(np.mean(col(mi, mcols, c)[9*n_mi//10:]) for c in sig_str_cols)
    dir_total = sum(np.mean(col(mi, mcols, c)[9*n_mi//10:]) for c in sig_dir_cols)
    mem_total = sum(np.mean(col(mi, mcols, c)[9*n_mi//10:]) for c in mem_cols)
    zone_mi = np.mean(col(mi, mcols, "mi_zone_damage")[9*n_mi//10:])
    food_mi = sum(np.mean(col(mi, mcols, c)[9*n_mi//10:]) for c in ["mi_food_dx", "mi_food_dy", "mi_food_dist"])
    ally_mi = sum(np.mean(col(mi, mcols, c)[9*n_mi//10:]) for c in ["mi_ally_dx", "mi_ally_dy", "mi_ally_dist"])

    total_mi = str_total + dir_total + mem_total + zone_mi + food_mi + ally_mi
    print("    Late-stage encoding budget (% of total input MI):")
    print(f"      Memory cells:     {mem_total:.4f}  ({100*mem_total/total_mi:.1f}%)")
    print(f"      Zone damage:      {zone_mi:.4f}  ({100*zone_mi/total_mi:.1f}%)")
    print(f"      Signal strength:  {str_total:.4f}  ({100*str_total/total_mi:.1f}%)")
    print(f"      Signal direction: {dir_total:.4f}  ({100*dir_total/total_mi:.1f}%)")
    print(f"      Food:             {food_mi:.4f}  ({100*food_mi/total_mi:.1f}%)")
    print(f"      Ally:             {ally_mi:.4f}  ({100*ally_mi/total_mi:.1f}%)")

    # Per-symbol signal encoding (which symbols matter to the brain?)
    print("\n    Per-symbol encoding (strength MI, late):")
    for i in range(6):
        s = np.mean(col(mi, mcols, f"mi_sig{i}_str")[9*n_mi//10:])
        dx = np.mean(col(mi, mcols, f"mi_sig{i}_dx")[9*n_mi//10:])
        dy = np.mean(col(mi, mcols, f"mi_sig{i}_dy")[9*n_mi//10:])
        total_sym = s + dx + dy
        print(f"      sym{i}: str={s:.4f}  dir={dx+dy:.4f}  total={total_sym:.4f}")

    # Evolution of encoding over time
    print("\n    Encoding evolution (total MI by category):")
    quarters = [(0, n_mi//4, "Q1"), (n_mi//4, n_mi//2, "Q2"),
                (n_mi//2, 3*n_mi//4, "Q3"), (3*n_mi//4, n_mi, "Q4")]
    for lo, hi, label in quarters:
        s = sum(np.mean(col(mi, mcols, c)[lo:hi]) for c in sig_str_cols)
        d = sum(np.mean(col(mi, mcols, c)[lo:hi]) for c in sig_dir_cols)
        m = sum(np.mean(col(mi, mcols, c)[lo:hi]) for c in mem_cols)
        z = np.mean(col(mi, mcols, "mi_zone_damage")[lo:hi])
        print(f"      {label}: mem={m:.3f}  zone={z:.3f}  sig_str={s:.3f}  sig_dir={d:.3f}")

    # =====================================================================
    # 6. ZONE DEATHS & SURVIVAL STRATEGY
    # =====================================================================
    print("\n  6. ZONE DEATH DYNAMICS")
    print("    Zone deaths per gen (quartiles):")
    for lo, hi, label in [(0, n//4, "Q1"), (n//4, n//2, "Q2"),
                           (n//2, 3*n//4, "Q3"), (3*n//4, n, "Q4")]:
        zd = zone_deaths[lo:hi]
        print(f"      {label}: {np.mean(zd):.1f} +/- {np.std(zd):.1f}  (max {np.max(zd):.0f})")

    # Zone deaths vs fitness correlation
    r_zd_fit = np.corrcoef(zone_deaths, fitness)[0, 1]
    r_zd_mi = np.corrcoef(zone_deaths, mutual)[0, 1]
    r_zd_sig = np.corrcoef(zone_deaths, signals_emitted)[0, 1]
    print(f"    Corr(zone_deaths, fitness):  {r_zd_fit:.3f}")
    print(f"    Corr(zone_deaths, MI):       {r_zd_mi:.3f}")
    print(f"    Corr(zone_deaths, signals):  {r_zd_sig:.3f}")

    # =====================================================================
    # 7. TRAJECTORY: SYMBOL SPATIAL DISTRIBUTIONS
    # =====================================================================
    print("\n  7. SYMBOL-ZONE SPATIAL ANALYSIS")
    traj_gen = col(traj, tcols, "generation")
    n_traj = len(traj_gen)
    tail = slice(9*n_traj//10, n_traj)

    # For each symbol, compute the ratio of d0 (near zone) to d3 (far from zone)
    print("    Late-stage proximity ratios (d0/d3, >1 = overrepresented near zones):")
    for i in range(6):
        d0 = col(traj, tcols, f"s{i}d0")[tail]
        d3 = col(traj, tcols, f"s{i}d3")[tail]
        d0_mean = np.mean(d0)
        d3_mean = np.mean(d3)
        ratio = d0_mean / d3_mean if d3_mean > 0.001 else float('inf')
        total = d0_mean + np.mean(col(traj, tcols, f"s{i}d1")[tail]) + \
                np.mean(col(traj, tcols, f"s{i}d2")[tail]) + d3_mean
        share = total  # already proportional
        print(f"      sym{i}: d0={d0_mean:.4f} d3={d3_mean:.4f} ratio={ratio:.2f} share={share:.3f}")

    # Are ANY symbols preferentially used near zones vs far?
    # Compute chi-square-like statistic for each symbol
    print("\n    Zone-preference scores (positive = near-zone bias):")
    all_d0 = np.zeros(n_traj)
    all_d3 = np.zeros(n_traj)
    for i in range(6):
        all_d0 += col(traj, tcols, f"s{i}d0")
        all_d3 += col(traj, tcols, f"s{i}d3")

    for i in range(6):
        sym_d0 = np.mean(col(traj, tcols, f"s{i}d0")[tail])
        sym_d3 = np.mean(col(traj, tcols, f"s{i}d3")[tail])
        total_d0 = np.mean(all_d0[tail])
        total_d3 = np.mean(all_d3[tail])
        if total_d0 > 0 and total_d3 > 0:
            expected_ratio = total_d0 / total_d3
            observed_ratio = sym_d0 / sym_d3 if sym_d3 > 0.001 else float('inf')
            preference = (observed_ratio - expected_ratio) / expected_ratio
            print(f"      sym{i}: observed_ratio={observed_ratio:.3f} expected={expected_ratio:.3f} "
                  f"preference={preference:+.3f}")

    # =====================================================================
    # 8. RECEIVER-FIT DECOMPOSITION
    # =====================================================================
    print("\n  8. RECEIVER-FIT TRAJECTORY")
    recv_smooth = window_avg(receiver_fit, 200)
    print("    Receiver-fit evolution:")
    for lo, hi, label in [(0, n//4, "Q1"), (n//4, n//2, "Q2"),
                           (n//2, 3*n//4, "Q3"), (3*n//4, n, "Q4")]:
        print(f"      {label}: {np.mean(receiver_fit[lo:hi]):.4f}")
    print(f"    Corr(receiver_fit, MI):      {np.corrcoef(receiver_fit, mutual)[0,1]:.3f}")
    print(f"    Corr(receiver_fit, fitness): {np.corrcoef(receiver_fit, fitness)[0,1]:.3f}")
    print(f"    Corr(receiver_fit, sig_h):   {np.corrcoef(receiver_fit, sig_hidden)[0,1]:.3f}")

    # Does high receiver-fit predict future fitness?
    for off in [100, 500]:
        if off < n:
            r = np.corrcoef(receiver_fit[:n-off], fitness[off:])[0, 1]
            print(f"    Receiver-fit -> fitness (+{off} gens): r={r:.3f}")

    # =====================================================================
    # 9. CONVERGENT DYNAMICS: do both seeds converge to same attractor?
    # =====================================================================
    print("\n  9. STABILITY ANALYSIS")
    # Compute coefficient of variation in late phase
    late = slice(9*n//10, n)
    metrics = [("fitness", fitness), ("MI", mutual), ("sig_hidden", sig_hidden),
               ("base_hidden", base_hidden), ("signals_emitted", signals_emitted),
               ("silence_corr", silence_corr), ("signal_entropy", signal_entropy)]
    print("    Late-phase stability (CoV = std/mean, lower = more stable):")
    for name_m, arr in metrics:
        m = np.mean(arr[late])
        s = np.std(arr[late])
        cov = s / m if m > 0 else float('inf')
        print(f"      {name_m:20s}: mean={m:.4f}  std={s:.4f}  CoV={cov:.3f}")

    # Autocorrelation of MI (is it periodic or random?)
    mi_centered = mutual - np.mean(mutual)
    acf = np.correlate(mi_centered, mi_centered, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / acf[0]
    # Find first peak after lag 0
    for i in range(2, min(500, len(acf)-1)):
        if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.05:
            print(f"    MI autocorrelation: first peak at lag {i} (r={acf[i]:.3f})")
            break
    else:
        print("    MI autocorrelation: no significant periodicity detected")

    # =====================================================================
    # 10. THE AGGREGATION HYPOTHESIS TEST
    # =====================================================================
    print("\n  10. AGGREGATION HYPOTHESIS")
    # If signals function as aggregation beacons:
    # - More signals emitted should correlate with higher fitness (group benefit)
    # - Signal direction encoding should be high (knowing WHERE others are)
    # - Symbol identity shouldn't matter (all symbols equivalent)
    r_sig_fit = np.corrcoef(signals_emitted, fitness)[0, 1]
    print(f"    Signals emitted <-> fitness: r={r_sig_fit:.3f}")

    # Signal direction vs strength encoding ratio
    if total_mi > 0:
        dir_str_ratio = dir_total / str_total if str_total > 0 else float('inf')
        print(f"    Direction/strength encoding ratio: {dir_str_ratio:.2f}")
        print("      (>1 means brain cares more about WHERE signals come from than WHAT they are)")

    # Symbol interchangeability: how similar are all pairwise contrasts?
    contrast_cols = [c for c in tcols if c.startswith("contrast_")]
    if contrast_cols:
        contrasts = [np.mean(col(traj, tcols, c)[tail]) for c in contrast_cols]
        print(f"    Pairwise symbol contrast (late): mean={np.mean(contrasts):.4f}, "
              f"std={np.std(contrasts):.4f}, max={np.max(contrasts):.4f}")
        print("      (low mean + low std = symbols are interchangeable)")


if __name__ == "__main__":
    base = Path("analysis-final")
    for seed, suffix in [("s200", "s200"), ("s201", "s201")]:
        analyze_seed(
            seed,
            base / f"{suffix}-output.csv",
            base / f"{suffix}-trajectory.csv",
            base / f"{suffix}-input_mi.csv",
        )
