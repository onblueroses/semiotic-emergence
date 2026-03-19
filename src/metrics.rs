use crate::brain::INPUTS;
use crate::signal::NUM_SYMBOLS;
use crate::world::SignalEvent;

pub fn compute_iconicity(
    signal_events: &[SignalEvent],
    ticks_in_zone: u32,
    total_ticks: u32,
    zone_radius: f32,
) -> f32 {
    if signal_events.is_empty() || total_ticks == 0 {
        return 0.0;
    }
    // Signals emitted while inside a zone (zone_dist <= 0 means inside)
    let signals_in_zone = signal_events
        .iter()
        .filter(|e| e.zone_dist <= zone_radius)
        .count() as f32;
    let signal_zone_rate = signals_in_zone / signal_events.len() as f32;
    let baseline_zone_rate = ticks_in_zone as f32 / total_ticks as f32;
    signal_zone_rate - baseline_zone_rate
}

pub fn compute_mutual_info(signal_events: &[SignalEvent], mi_bins: &[f32; 3]) -> f32 {
    if signal_events.len() < 20 {
        return 0.0;
    }
    let counts = signal_context_matrix(signal_events, mi_bins);
    mi_from_contingency(&counts)
}

/// I(Signal; `FoodDistance`) using adaptive quartile binning on inputs\[5\].
/// Bin edges derived from data each generation, avoiding degenerate bins when
/// food distances cluster near zero (which they do at every tested grid/food setting).
pub fn compute_food_mi(signal_events: &[SignalEvent]) -> f32 {
    adaptive_quartile_mi(signal_events, |e| e.inputs[5])
}

/// MI between emitted symbol and an arbitrary continuous variable, using
/// data-derived quartile bin edges. Handles any value distribution without
/// fixed-bin degeneracy.
fn adaptive_quartile_mi(
    signal_events: &[SignalEvent],
    value_fn: impl Fn(&SignalEvent) -> f32,
) -> f32 {
    if signal_events.len() < 20 {
        return 0.0;
    }
    let mut vals: Vec<f32> = signal_events.iter().map(&value_fn).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = vals[vals.len() / 4];
    let q2 = vals[vals.len() / 2];
    let q3 = vals[3 * vals.len() / 4];
    // If all values are identical, MI is zero (no variance to correlate with)
    if (q1 - q3).abs() < f32::EPSILON {
        return 0.0;
    }
    let mut counts = [[0u32; 4]; NUM_SYMBOLS];
    for e in signal_events {
        let sym = (e.symbol as usize).min(NUM_SYMBOLS - 1);
        let v = value_fn(e);
        let bin = if v <= q1 {
            0
        } else if v <= q2 {
            1
        } else if v <= q3 {
            2
        } else {
            3
        };
        counts[sym][bin] += 1;
    }
    mi_from_contingency(&counts)
}

/// Zone distance bin using configurable bin edges.
/// Bins: [0, bins[0]), [bins[0], bins[1]), [bins[1], bins[2]), [bins[2], +inf).
fn zone_dist_bin(dist: f32, bins: &[f32; 3]) -> usize {
    if dist < bins[0] {
        0
    } else if dist < bins[1] {
        1
    } else if dist < bins[2] {
        2
    } else {
        3
    }
}

/// Build the signal-context contingency matrix from signal events.
/// Rows = symbols, columns = zone distance bins.
pub fn signal_context_matrix(
    signal_events: &[SignalEvent],
    mi_bins: &[f32; 3],
) -> [[u32; 4]; NUM_SYMBOLS] {
    let mut counts = [[0u32; 4]; NUM_SYMBOLS];
    for e in signal_events {
        let sym = (e.symbol as usize).min(NUM_SYMBOLS - 1);
        counts[sym][zone_dist_bin(e.zone_dist, mi_bins)] += 1;
    }
    counts
}

/// MI from a contingency table. Shared by all MI computations.
fn mi_from_contingency(counts: &[[u32; 4]; NUM_SYMBOLS]) -> f32 {
    let n: f32 = counts.iter().flat_map(|row| row.iter()).sum::<u32>() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mut mi = 0.0_f32;
    for s in 0..NUM_SYMBOLS {
        let prob_s = counts[s].iter().sum::<u32>() as f32 / n;
        if prob_s == 0.0 {
            continue;
        }
        for (bin, &count) in counts[s].iter().enumerate() {
            let prob_bin: f32 = (0..NUM_SYMBOLS).map(|ss| counts[ss][bin]).sum::<u32>() as f32 / n;
            if prob_bin == 0.0 {
                continue;
            }
            let prob_joint = count as f32 / n;
            if prob_joint > 0.0 {
                mi += prob_joint * (prob_joint / (prob_s * prob_bin)).ln();
            }
        }
    }
    mi
}

fn kl_div(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q)
        .filter(|(&pi, &qi)| pi > 0.0 && qi > 0.0)
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum()
}

fn jsd(p: &[f32], q: &[f32]) -> f32 {
    let m: Vec<f32> = p.iter().zip(q).map(|(&a, &b)| (a + b) * 0.5).collect();
    0.5 * kl_div(p, &m) + 0.5 * kl_div(q, &m)
}

fn normalize_action_dist(counts: &[u32; 5]) -> Option<[f32; 5]> {
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return None;
    }
    let t = total as f32;
    Some([
        counts[0] as f32 / t,
        counts[1] as f32 / t,
        counts[2] as f32 / t,
        counts[3] as f32 / t,
        counts[4] as f32 / t,
    ])
}

/// Receiver Response Spectrum: JSD between action distributions with vs without signal,
/// split by context (not in zone, in zone).
pub fn compute_receiver_jsd(counts: &[[[u32; 5]; 2]; 1 + NUM_SYMBOLS]) -> (f32, f32) {
    let mut jsd_per_context = [0.0_f32; 2];
    for ctx in 0..2 {
        let baseline = &counts[0][ctx]; // no signal
        let Some(p_base) = normalize_action_dist(baseline) else {
            continue;
        };
        let mut total_jsd = 0.0_f32;
        let mut n_symbols = 0;
        for signal_counts in &counts[1..=NUM_SYMBOLS] {
            let Some(p_sig) = normalize_action_dist(&signal_counts[ctx]) else {
                continue;
            };
            total_jsd += jsd(&p_base, &p_sig);
            n_symbols += 1;
        }
        if n_symbols > 0 {
            jsd_per_context[ctx] = total_jsd / n_symbols as f32;
        }
    }
    (jsd_per_context[0], jsd_per_context[1])
}

/// Per-symbol JSD: how much each symbol shifts behavior vs no-signal baseline.
/// Pooled across both contexts.
pub fn compute_per_symbol_jsd(counts: &[[[u32; 5]; 2]; 1 + NUM_SYMBOLS]) -> [f32; NUM_SYMBOLS] {
    // Pool baseline across both contexts
    let mut base_pooled = [0u32; 5];
    for (a, bp) in base_pooled.iter_mut().enumerate() {
        *bp = counts[0][0][a] + counts[0][1][a];
    }

    let mut result = [0.0_f32; NUM_SYMBOLS];
    for sym in 0..NUM_SYMBOLS {
        let mut sig_pooled = [0u32; 5];
        for (a, sp) in sig_pooled.iter_mut().enumerate() {
            *sp = counts[sym + 1][0][a] + counts[sym + 1][1][a];
        }
        let Some(p_base) = normalize_action_dist(&base_pooled) else {
            continue;
        };
        let Some(p_sig) = normalize_action_dist(&sig_pooled) else {
            continue;
        };
        result[sym] = jsd(&p_base, &p_sig);
    }
    result
}

/// Pearson correlation between two f32 slices.
pub fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let nf = n as f32;
    let mean_x: f32 = xs[..n].iter().sum::<f32>() / nf;
    let mean_y: f32 = ys[..n].iter().sum::<f32>() / nf;
    let mut cov = 0.0_f32;
    let mut var_x = 0.0_f32;
    let mut var_y = 0.0_f32;
    for i in 0..n {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    cov / denom
}

fn jsd_4(p: &[f32; 4], q: &[f32; 4]) -> f32 {
    jsd(p.as_slice(), q.as_slice())
}

/// Normalize counts matrix to row-wise probability distributions.
pub fn normalize_matrix(counts: &[[u32; 4]; NUM_SYMBOLS]) -> Option<[[f32; 4]; NUM_SYMBOLS]> {
    let mut result = [[0.0_f32; 4]; NUM_SYMBOLS];
    for (s, row) in counts.iter().enumerate() {
        let total: u32 = row.iter().sum();
        if total == 0 {
            return None;
        }
        let t = total as f32;
        for (b, &c) in row.iter().enumerate() {
            result[s][b] = c as f32 / t;
        }
    }
    Some(result)
}

/// Cross-population divergence: minimum-over-permutations average row-wise JSD.
/// Accounts for arbitrary symbol index assignment across populations.
pub fn cross_population_divergence(
    a: &[[f32; 4]; NUM_SYMBOLS],
    b: &[[f32; 4]; NUM_SYMBOLS],
) -> f32 {
    let mut indices: Vec<usize> = (0..NUM_SYMBOLS).collect();
    let mut min_div = f32::MAX;
    // Iterate all permutations via Heap's algorithm
    heap_permute(&mut indices, NUM_SYMBOLS, a, b, &mut min_div);
    min_div
}

fn heap_permute(
    perm: &mut Vec<usize>,
    k: usize,
    a: &[[f32; 4]; NUM_SYMBOLS],
    b: &[[f32; 4]; NUM_SYMBOLS],
    min_div: &mut f32,
) {
    if k == 1 {
        let mut total = 0.0_f32;
        for (i, &pi) in perm.iter().enumerate() {
            total += jsd_4(&a[i], &b[pi]);
        }
        let avg = total / NUM_SYMBOLS as f32;
        if avg < *min_div {
            *min_div = avg;
        }
        return;
    }
    for i in 0..k {
        heap_permute(perm, k - 1, a, b, min_div);
        if k.is_multiple_of(2) {
            perm.swap(i, k - 1);
        } else {
            perm.swap(0, k - 1);
        }
    }
}

/// MI between each symbol and each input dimension at emission time.
/// Uses quartile-based binning (scale-invariant).
pub fn compute_input_mi(signal_events: &[SignalEvent]) -> [f32; INPUTS] {
    if signal_events.len() < 20 {
        return [0.0_f32; INPUTS];
    }

    let mis: Vec<(usize, f32)> = (0..INPUTS)
        .map(|dim| {
            let mut vals: Vec<f32> = signal_events.iter().map(|e| e.inputs[dim]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q1 = vals[vals.len() / 4];
            let q2 = vals[vals.len() / 2];
            let q3 = vals[3 * vals.len() / 4];

            let mut counts = [[0u32; 4]; NUM_SYMBOLS];
            for e in signal_events {
                let sym = (e.symbol as usize).min(NUM_SYMBOLS - 1);
                let bin = if e.inputs[dim] <= q1 {
                    0
                } else if e.inputs[dim] <= q2 {
                    1
                } else if e.inputs[dim] <= q3 {
                    2
                } else {
                    3
                };
                counts[sym][bin] += 1;
            }
            (dim, mi_from_contingency(&counts))
        })
        .collect();

    let mut result = [0.0_f32; INPUTS];
    for (dim, mi) in mis {
        result[dim] = mi;
    }
    result
}

/// Shannon entropy of emitted signal symbols. Max = ln(6) ≈ 1.79 (uniform).
/// Low entropy = population converging on specific symbols.
pub fn compute_signal_entropy(signal_events: &[SignalEvent]) -> f32 {
    if signal_events.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; NUM_SYMBOLS];
    for e in signal_events {
        let sym = (e.symbol as usize).min(NUM_SYMBOLS - 1);
        counts[sym] += 1;
    }
    let total = signal_events.len() as f32;
    let mut entropy = 0.0_f32;
    for &c in &counts {
        if c > 0 {
            let p = c as f32 / total;
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Rolling fluctuation ratio: std(recent window) / std(early window).
/// Rising ratio precedes phase transitions. Returns 0.0 if insufficient data.
pub fn rolling_fluctuation_ratio(series: &[f32], window: usize) -> f32 {
    if series.len() < window * 2 {
        return 0.0;
    }
    let early = &series[..window];
    let recent = &series[series.len() - window..];
    let std_early = std_dev(early);
    let std_recent = std_dev(recent);
    if std_early < 1e-12 {
        return 0.0;
    }
    std_recent / std_early
}

fn std_dev(xs: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    var.sqrt()
}

/// Pairwise JSD between symbols' context distributions.
/// Returns all pairwise values in lexicographic order.
pub fn inter_symbol_jsd(norm: &[[f32; 4]; NUM_SYMBOLS]) -> Vec<f32> {
    let mut result = Vec::with_capacity(NUM_SYMBOLS * (NUM_SYMBOLS - 1) / 2);
    for i in 0..NUM_SYMBOLS {
        for j in (i + 1)..NUM_SYMBOLS {
            result.push(jsd_4(&norm[i], &norm[j]));
        }
    }
    result
}

/// JSD between two consecutive generation matrices (for trajectory phase transitions).
pub fn trajectory_jsd(prev: &[[f32; 4]; NUM_SYMBOLS], curr: &[[f32; 4]; NUM_SYMBOLS]) -> f32 {
    let mut total = 0.0_f32;
    for s in 0..NUM_SYMBOLS {
        total += jsd_4(&prev[s], &curr[s]);
    }
    total / NUM_SYMBOLS as f32
}

/// Per-prey receiver JSD: how much one prey's actions differ with vs without signal.
/// Pools across both contexts. Returns 0.0 if either bucket has fewer than `min_samples` total.
pub fn per_prey_receiver_jsd(
    with: &[[u32; 5]; 2],
    without: &[[u32; 5]; 2],
    min_samples: u32,
) -> f32 {
    let mut with_pooled = [0u32; 5];
    let mut without_pooled = [0u32; 5];
    for ctx in 0..2 {
        for a in 0..5 {
            with_pooled[a] += with[ctx][a];
            without_pooled[a] += without[ctx][a];
        }
    }
    let total_with: u32 = with_pooled.iter().sum();
    let total_without: u32 = without_pooled.iter().sum();
    if total_with < min_samples || total_without < min_samples {
        return 0.0;
    }
    let Some(p_with) = normalize_action_dist(&with_pooled) else {
        return 0.0;
    };
    let Some(p_without) = normalize_action_dist(&without_pooled) else {
        return 0.0;
    };
    jsd(&p_with, &p_without)
}

/// Silence onset metrics: how receivers behave when signals disappear vs during signals.
/// `present` is the action distribution during signal reception (baseline).
/// Aggregates across all prey, pred-invisible context (idx 0) only to isolate signal-channel effect.
/// Returns `(onset_jsd, move_delta)`.
pub fn compute_silence_onset_metrics(
    onset: &[[[u32; 5]; 2]],
    present: &[[[u32; 5]; 2]],
) -> (f32, f32) {
    let mut onset_pooled = [0u32; 5];
    let mut present_pooled = [0u32; 5];
    for prey_onset in onset {
        for a in 0..5 {
            onset_pooled[a] += prey_onset[0][a];
        }
    }
    for prey_present in present {
        for a in 0..5 {
            present_pooled[a] += prey_present[0][a];
        }
    }

    let total_onset: u32 = onset_pooled.iter().sum();
    let total_present: u32 = present_pooled.iter().sum();
    if total_onset < 5 || total_present < 10 {
        return (0.0, 0.0);
    }

    let Some(p_onset) = normalize_action_dist(&onset_pooled) else {
        return (0.0, 0.0);
    };
    let Some(p_present) = normalize_action_dist(&present_pooled) else {
        return (0.0, 0.0);
    };

    let onset_jsd = jsd(&p_onset, &p_present);

    // Flight = movement actions (0-3), vs eat (4)
    let onset_flight: f32 = p_onset[0] + p_onset[1] + p_onset[2] + p_onset[3];
    let present_flight: f32 = p_present[0] + p_present[1] + p_present[2] + p_present[3];

    (onset_jsd, onset_flight - present_flight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsd_identical_is_zero() {
        let p = [0.2, 0.3, 0.1, 0.15, 0.25];
        assert!(jsd(&p, &p) < 1e-10);
    }

    #[test]
    fn jsd_different_is_positive() {
        let p = [1.0, 0.0, 0.0, 0.0, 0.0];
        let q = [0.0, 0.0, 0.0, 0.0, 1.0];
        assert!(jsd(&p, &q) > 0.5);
    }

    #[test]
    fn jsd_all_zeros_returns_zero() {
        // compute_receiver_jsd should handle empty bins gracefully
        let counts = [[[0u32; 5]; 2]; 1 + NUM_SYMBOLS];
        let (a, b) = compute_receiver_jsd(&counts);
        assert!(a.abs() < 1e-10);
        assert!(b.abs() < 1e-10);
    }

    #[test]
    fn pearson_perfect_positive() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson(&xs, &ys) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pearson_uncorrelated() {
        // Symmetric pattern: correlation cancels to zero
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [1.0, -1.0, 0.0, -1.0, 1.0];
        assert!(pearson(&xs, &ys).abs() < 1e-6);
    }

    #[test]
    fn cross_pop_divergence_identical_is_zero() {
        let m = [[0.25, 0.25, 0.25, 0.25]; NUM_SYMBOLS];
        assert!(cross_population_divergence(&m, &m) < 1e-10);
    }

    #[test]
    fn cross_pop_divergence_permuted_is_zero() {
        // First 3 symbols have distinct distributions, rest uniform
        let mut a = [[0.25, 0.25, 0.25, 0.25]; NUM_SYMBOLS];
        a[0] = [0.8, 0.1, 0.05, 0.05];
        a[1] = [0.1, 0.7, 0.1, 0.1];
        a[2] = [0.05, 0.05, 0.1, 0.8];
        // Same distributions but symbols 0-2 rotated
        let mut b = [[0.25, 0.25, 0.25, 0.25]; NUM_SYMBOLS];
        b[0] = [0.1, 0.7, 0.1, 0.1];
        b[1] = [0.05, 0.05, 0.1, 0.8];
        b[2] = [0.8, 0.1, 0.05, 0.05];
        assert!(cross_population_divergence(&a, &b) < 1e-10);
    }

    #[test]
    fn cross_pop_divergence_different_is_positive() {
        let mut a = [[0.25, 0.25, 0.25, 0.25]; NUM_SYMBOLS];
        a[0] = [0.9, 0.03, 0.03, 0.04];
        a[1] = [0.03, 0.9, 0.04, 0.03];
        a[2] = [0.04, 0.03, 0.9, 0.03];
        let b = [[0.25, 0.25, 0.25, 0.25]; NUM_SYMBOLS];
        assert!(cross_population_divergence(&a, &b) > 0.01);
    }

    #[test]
    fn trajectory_jsd_identical_is_zero() {
        let m = [[0.25, 0.25, 0.25, 0.25]; NUM_SYMBOLS];
        assert!(trajectory_jsd(&m, &m) < 1e-10);
    }

    #[test]
    fn rolling_fluct_insufficient_data_returns_zero() {
        assert!((rolling_fluctuation_ratio(&[1.0, 2.0, 3.0], 5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn rolling_fluct_stable_series_near_one() {
        // Same variance early and late
        let series: Vec<f32> = (0..40).map(|i| (i as f32 * 0.1).sin()).collect();
        let ratio = rolling_fluctuation_ratio(&series, 10);
        assert!(
            (ratio - 1.0).abs() < 0.5,
            "Expected ratio near 1.0, got {ratio}"
        );
    }

    #[test]
    fn rolling_fluct_increasing_variance() {
        // Small variance early, large variance late
        let mut series: Vec<f32> = (0..20).map(|i| 1.0 + (i as f32 * 0.01)).collect();
        series.extend((0..20).map(|i| if i % 2 == 0 { 0.0 } else { 2.0 }));
        let ratio = rolling_fluctuation_ratio(&series, 20);
        assert!(ratio > 1.0, "Expected ratio > 1.0, got {ratio}");
    }

    #[test]
    fn inter_symbol_jsd_identical_is_zero() {
        let m = [[0.25, 0.25, 0.25, 0.25]; NUM_SYMBOLS];
        let result = inter_symbol_jsd(&m);
        assert!(result.iter().all(|&v| v < 1e-10));
    }

    #[test]
    fn inter_symbol_jsd_distinct_is_positive() {
        let mut m = [[0.25, 0.25, 0.25, 0.25]; NUM_SYMBOLS];
        m[0] = [0.9, 0.03, 0.03, 0.04];
        m[1] = [0.03, 0.9, 0.04, 0.03];
        m[2] = [0.04, 0.03, 0.03, 0.9];
        let result = inter_symbol_jsd(&m);
        // At least the first 3 pairs (involving distinct rows) should be positive
        assert!(result[0] > 0.1);
    }

    #[test]
    fn input_mi_detects_correlated_dimension() {
        // Symbol 0 always emitted when dim 0 is low, symbol 1 when high
        let mut events = Vec::new();
        for i in 0..100 {
            let val = i as f32 / 100.0;
            let symbol = u8::from(val >= 0.5);
            let mut inputs = [0.5_f32; INPUTS];
            inputs[0] = val;
            events.push(SignalEvent {
                symbol,
                zone_dist: 5.0,
                inputs,
                emitter_idx: 0,
            });
        }
        let mi = compute_input_mi(&events);
        // Dim 0 should have high MI (strong correlation)
        assert!(mi[0] > 0.1, "Expected high MI for dim 0, got {}", mi[0]);
        // Other dims should have near-zero MI (no correlation)
        assert!(mi[1] < 0.01, "Expected low MI for dim 1, got {}", mi[1]);
    }

    #[test]
    fn input_mi_too_few_events_returns_zeros() {
        let events: Vec<SignalEvent> = (0..10)
            .map(|i| SignalEvent {
                symbol: (i % 3) as u8,
                zone_dist: 5.0,
                inputs: [0.0; INPUTS],
                emitter_idx: 0,
            })
            .collect();
        let mi = compute_input_mi(&events);
        assert!(mi.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn signal_entropy_empty_is_zero() {
        assert!(compute_signal_entropy(&[]).abs() < 1e-10);
    }

    #[test]
    fn signal_entropy_single_symbol_is_zero() {
        let events: Vec<SignalEvent> = (0..100)
            .map(|_| SignalEvent {
                symbol: 3,
                zone_dist: 5.0,
                inputs: [0.0; INPUTS],
                emitter_idx: 0,
            })
            .collect();
        assert!(compute_signal_entropy(&events).abs() < 1e-10);
    }

    #[test]
    fn signal_entropy_uniform_is_max() {
        let events: Vec<SignalEvent> = (0..600)
            .map(|i| SignalEvent {
                symbol: (i % 6) as u8,
                zone_dist: 5.0,
                inputs: [0.0; INPUTS],
                emitter_idx: 0,
            })
            .collect();
        let expected = (6.0_f32).ln(); // ln(6) ≈ 1.7918
        let result = compute_signal_entropy(&events);
        assert!(
            (result - expected).abs() < 1e-4,
            "Expected {expected:.4}, got {result:.4}"
        );
    }

    #[test]
    fn signal_entropy_two_symbols_equal() {
        let events: Vec<SignalEvent> = (0..200)
            .map(|i| SignalEvent {
                symbol: (i % 2) as u8,
                zone_dist: 5.0,
                inputs: [0.0; INPUTS],
                emitter_idx: 0,
            })
            .collect();
        let expected = (2.0_f32).ln(); // ln(2) ≈ 0.6931
        let result = compute_signal_entropy(&events);
        assert!(
            (result - expected).abs() < 1e-4,
            "Expected {expected:.4}, got {result:.4}"
        );
    }

    #[test]
    fn signal_context_matrix_bins_correctly() {
        let bins = [4.0, 8.0, 11.0];
        let events = vec![
            SignalEvent {
                symbol: 0,
                zone_dist: 2.0,
                inputs: [0.0; INPUTS],
                emitter_idx: 0,
            },
            SignalEvent {
                symbol: 1,
                zone_dist: 6.0,
                inputs: [0.0; INPUTS],
                emitter_idx: 0,
            },
            SignalEvent {
                symbol: 2,
                zone_dist: 10.0,
                inputs: [0.0; INPUTS],
                emitter_idx: 0,
            },
            SignalEvent {
                symbol: 0,
                zone_dist: 15.0,
                inputs: [0.0; INPUTS],
                emitter_idx: 0,
            },
        ];
        let m = signal_context_matrix(&events, &bins);
        assert_eq!(m[0][0], 1);
        assert_eq!(m[1][1], 1);
        assert_eq!(m[2][2], 1);
        assert_eq!(m[0][3], 1);
    }

    #[test]
    fn per_prey_jsd_different_distributions() {
        let with = [[50, 0, 0, 0, 0], [0; 5]]; // always action 0 with signal
        let without = [[0, 0, 0, 0, 50], [0; 5]]; // always action 4 without
        let result = per_prey_receiver_jsd(&with, &without, 10);
        assert!(result > 0.5, "Expected high JSD, got {result}");
    }

    #[test]
    fn per_prey_jsd_identical_distributions() {
        let dist = [[10, 10, 10, 10, 10], [0; 5]];
        let result = per_prey_receiver_jsd(&dist, &dist, 10);
        assert!(result < 1e-10, "Expected ~0 JSD, got {result}");
    }

    #[test]
    fn per_prey_jsd_below_threshold_returns_zero() {
        let with = [[1, 1, 1, 1, 1], [0; 5]]; // 5 total < 10 min
        let without = [[10, 10, 10, 10, 10], [0; 5]];
        assert!(per_prey_receiver_jsd(&with, &without, 10).abs() < 1e-10);
    }

    #[test]
    fn silence_onset_detects_behavioral_shift() {
        // At onset: all movement (action 0). During signal: all eating (action 4).
        let onset = vec![[[50, 0, 0, 0, 0], [0; 5]]];
        let present = vec![[[0, 0, 0, 0, 50], [0; 5]]];
        let (jsd_val, move_delta) = compute_silence_onset_metrics(&onset, &present);
        assert!(jsd_val > 0.5, "Expected high JSD, got {jsd_val}");
        assert!(
            move_delta > 0.5,
            "Expected positive flight delta, got {move_delta}"
        );
    }

    #[test]
    fn silence_onset_no_shift() {
        let dist = vec![[[10, 10, 10, 10, 10], [0; 5]]];
        let (jsd_val, move_delta) = compute_silence_onset_metrics(&dist, &dist);
        assert!(jsd_val < 1e-10, "Expected ~0 JSD, got {jsd_val}");
        assert!(
            move_delta.abs() < 1e-10,
            "Expected ~0 delta, got {move_delta}"
        );
    }

    #[test]
    fn silence_onset_insufficient_data() {
        let onset = vec![[[1, 1, 1, 1, 0], [0; 5]]]; // 4 total < 5 min
        let present = vec![[[10, 10, 10, 10, 10], [0; 5]]];
        let (jsd_val, move_delta) = compute_silence_onset_metrics(&onset, &present);
        assert!(jsd_val.abs() < 1e-10);
        assert!(move_delta.abs() < 1e-10);
    }

    #[test]
    fn food_mi_correlated_symbols() {
        // Symbol 0 when food is close (low inputs[5]), symbol 1 when far (high inputs[5])
        let mut events = Vec::new();
        for i in 0..100 {
            let food_dist = i as f32 / 100.0; // 0.0 to 0.99
            let symbol = u8::from(food_dist >= 0.5);
            let mut inputs = [0.5_f32; INPUTS];
            inputs[5] = food_dist;
            events.push(SignalEvent {
                symbol,
                zone_dist: 5.0,
                inputs,
                emitter_idx: 0,
            });
        }
        let mi = compute_food_mi(&events);
        assert!(
            mi > 0.1,
            "Expected food_mi > 0.1 for correlated signals, got {mi}"
        );
    }

    #[test]
    fn food_mi_uncorrelated_symbols() {
        // All symbols emit at all food distances equally - no correlation
        let mut events = Vec::new();
        for sym in 0..6_u8 {
            for bin in 0..4_u32 {
                let food_dist = bin as f32 * 0.25 + 0.1;
                for _ in 0..5 {
                    let mut inputs = [0.5_f32; INPUTS];
                    inputs[5] = food_dist;
                    events.push(SignalEvent {
                        symbol: sym,
                        zone_dist: 5.0,
                        inputs,
                        emitter_idx: 0,
                    });
                }
            }
        }
        let mi = compute_food_mi(&events);
        assert!(
            mi < 0.01,
            "Expected food_mi ~ 0 for uniform distribution, got {mi}"
        );
    }

    #[test]
    fn food_mi_clustered_near_zero() {
        // Regression: food distances cluster near zero at realistic params (food=520, grid=100).
        // Fixed bins put everything in [0, 0.25) giving MI=0. Adaptive bins should detect signal.
        let mut events = Vec::new();
        for i in 0..200 {
            // Symbol 0 when very close (0.01-0.02), symbol 1 when slightly further (0.03-0.05)
            let (symbol, food_dist) = if i % 2 == 0 {
                (0_u8, 0.01 + (i as f32 % 10.0) * 0.001)
            } else {
                (1_u8, 0.03 + (i as f32 % 10.0) * 0.002)
            };
            let mut inputs = [0.5_f32; INPUTS];
            inputs[5] = food_dist;
            events.push(SignalEvent {
                symbol,
                zone_dist: 5.0,
                inputs,
                emitter_idx: 0,
            });
        }
        let mi = compute_food_mi(&events);
        assert!(
            mi > 0.05,
            "Adaptive bins should detect correlation even when all values < 0.05, got {mi}"
        );
    }

    #[test]
    fn food_mi_too_few_events() {
        let events: Vec<SignalEvent> = (0..10)
            .map(|i| SignalEvent {
                symbol: (i % 3) as u8,
                zone_dist: 5.0,
                inputs: [0.5_f32; INPUTS],
                emitter_idx: 0,
            })
            .collect();
        let mi = compute_food_mi(&events);
        assert!(
            mi.abs() < 1e-10,
            "Expected 0.0 for too few events, got {mi}"
        );
    }
}
