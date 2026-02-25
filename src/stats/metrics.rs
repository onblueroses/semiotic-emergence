use crate::agent::predator::PredatorKind;
use crate::stats::collector::SignalEvent;

// ---------------------------------------------------------------------------
// Mutual Information: I(Signal; Context) (Step 6.3)
// ---------------------------------------------------------------------------

/// Compute mutual information between emitted symbols and predator context (D11).
///
/// Context: 4 predator types x 4 distance bins = 16 bins.
/// Distance bins: 0-3, 3-6, 6-10, 10+ (or no predator).
/// Uses Miller-Madow bias correction.
///
/// Returns 0.0 if fewer than 10 events (insufficient data).
pub(crate) fn mutual_information(events: &[SignalEvent], vocab_size: u32) -> f64 {
    if events.len() < 10 {
        return 0.0;
    }

    let num_symbols = vocab_size as usize;
    let num_context_bins = 16; // 4 predator types x 4 distance bins

    // Build contingency table: signal x context
    let mut joint = vec![0_u64; num_symbols * num_context_bins];
    let mut signal_counts = vec![0_u64; num_symbols];
    let mut context_counts = vec![0_u64; num_context_bins];
    let total = events.len() as f64;

    for event in events {
        let sym = (event.symbol as usize).min(num_symbols - 1);
        let ctx = context_bin(event.nearest_predator_kind, event.nearest_predator_dist);

        joint[sym * num_context_bins + ctx] += 1;
        signal_counts[sym] += 1;
        context_counts[ctx] += 1;
    }

    // H(S) - signal entropy
    let h_signal = entropy(&signal_counts, total);
    // H(C) - context entropy
    let h_context = entropy(&context_counts, total);
    // H(S,C) - joint entropy
    let h_joint = entropy(&joint, total);

    // MI = H(S) + H(C) - H(S,C)
    let mi = h_signal + h_context - h_joint;

    // Miller-Madow correction
    let bins_used = joint.iter().filter(|&&c| c > 0).count() as f64;
    let correction = (bins_used - 1.0) / (2.0 * total * 2.0_f64.ln());

    (mi - correction).max(0.0)
}

/// Map predator type + distance to one of 16 context bins (D11).
fn context_bin(kind: Option<PredatorKind>, dist: Option<f32>) -> usize {
    let type_idx = match kind {
        Some(PredatorKind::Aerial) => 0,
        Some(PredatorKind::Ground) => 1,
        Some(PredatorKind::Pack) => 2,
        None => 3,
    };

    let dist_idx = match dist {
        Some(d) if d < 3.0 => 0,
        Some(d) if d < 6.0 => 1,
        Some(d) if d < 10.0 => 2,
        _ => 3, // 10+ or no predator
    };

    type_idx * 4 + dist_idx
}

/// Compute Shannon entropy from counts.
fn entropy(counts: &[u64], total: f64) -> f64 {
    let mut h = 0.0;
    for &count in counts {
        if count > 0 {
            let p = count as f64 / total;
            h -= p * p.ln();
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Topographic Similarity (Step 6.4)
// ---------------------------------------------------------------------------

/// Compute topographic similarity: Spearman correlation between signal distance
/// and referent (context) distance for sampled pairs of signal events.
///
/// Signal distance: 0 if same symbol, 1 if different (Hamming for single symbols).
/// Referent distance: Euclidean distance between emitter positions.
///
/// Returns 0.0 if fewer than 10 pairs.
pub(crate) fn topographic_similarity(events: &[SignalEvent], _vocab_size: u32) -> f64 {
    if events.len() < 10 {
        return 0.0;
    }

    // Sample up to 500 pairs
    let max_pairs = 500;
    let mut signal_dists: Vec<f64> = Vec::with_capacity(max_pairs);
    let mut referent_dists: Vec<f64> = Vec::with_capacity(max_pairs);

    let step = if events.len() * (events.len() - 1) / 2 > max_pairs {
        events.len() / (max_pairs as f64).sqrt() as usize
    } else {
        1
    }
    .max(1);

    let mut count = 0;
    let mut idx_i = 0;
    while idx_i < events.len() && count < max_pairs {
        let mut idx_j = idx_i + 1;
        while idx_j < events.len() && count < max_pairs {
            let sig_dist = if events[idx_i].symbol == events[idx_j].symbol {
                0.0
            } else {
                1.0
            };
            let ref_dist = events[idx_i]
                .emitter_pos
                .distance_to(&events[idx_j].emitter_pos);

            signal_dists.push(sig_dist);
            referent_dists.push(f64::from(ref_dist));
            count += 1;

            idx_j += step;
        }
        idx_i += step;
    }

    if signal_dists.len() < 3 {
        return 0.0;
    }

    spearman_correlation(&signal_dists, &referent_dists)
}

// ---------------------------------------------------------------------------
// Iconicity (Step 6.5)
// ---------------------------------------------------------------------------

/// Compute iconicity: correlation between signal emission rate and danger proximity.
///
/// For each distance bin (close/medium/far/none), compute the emission rate.
/// Spearman correlation between emission rate and danger (inverse distance).
///
/// High iconicity (> 0.5) means signals are emitted more often near predators.
///
/// Returns 0.0 if fewer than 10 events.
pub(crate) fn iconicity(events: &[SignalEvent]) -> f64 {
    if events.len() < 10 {
        return 0.0;
    }

    // Count events per distance bin (4 bins: close/medium/far/none)
    let mut bin_counts = [0_u64; 4];
    for event in events {
        let bin = match event.nearest_predator_dist {
            Some(d) if d < 3.0 => 0,
            Some(d) if d < 6.0 => 1,
            Some(d) if d < 10.0 => 2,
            _ => 3,
        };
        bin_counts[bin] += 1;
    }

    // Danger proximity scores (inverse distance: close=high, far=low)
    let danger = [4.0, 3.0, 2.0, 1.0];
    let rates: Vec<f64> = bin_counts.iter().map(|&c| c as f64).collect();

    spearman_correlation(&rates, &danger)
}

// ---------------------------------------------------------------------------
// Spearman rank correlation
// ---------------------------------------------------------------------------

/// Compute Spearman rank correlation between two sequences.
fn spearman_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return 0.0;
    }

    let ranks_x = compute_ranks(xs);
    let ranks_y = compute_ranks(ys);

    pearson_correlation(&ranks_x, &ranks_y)
}

/// Compute fractional ranks (average rank for ties).
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut idx = 0;
    while idx < n {
        let mut end = idx + 1;
        while end < n && (indexed[end].1 - indexed[idx].1).abs() < 1e-12 {
            end += 1;
        }
        // Average rank for tied values (1-indexed)
        let avg_rank = (idx + end) as f64 / 2.0 + 0.5;
        for item in indexed.iter().take(end).skip(idx) {
            ranks[item.0] = avg_rank;
        }
        idx = end;
    }

    ranks
}

/// Compute Pearson correlation coefficient between two sequences.
fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mean_x: f64 = xs.iter().sum::<f64>() / n;
    let mean_y: f64 = ys.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (x_val, y_val) in xs.iter().zip(ys.iter()) {
        let dx = x_val - mean_x;
        let dy = y_val - mean_y;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::entity::Position;

    fn make_event(
        symbol: u8,
        pred_kind: Option<PredatorKind>,
        pred_dist: Option<f32>,
        pos: Position,
    ) -> SignalEvent {
        SignalEvent {
            emitter_id: crate::world::entity::PreyId(0),
            symbol,
            emitter_pos: pos,
            nearest_predator_kind: pred_kind,
            nearest_predator_dist: pred_dist,
            tick: 0,
        }
    }

    #[test]
    fn mi_random_signals_low() {
        // Random signals with no predator context -> MI near 0
        let events: Vec<SignalEvent> = (0..100)
            .map(|i| make_event((i % 4) as u8, None, None, Position::new(i % 10, i / 10)))
            .collect();

        let mi = mutual_information(&events, 4);
        assert!(mi < 0.5, "Random signals should have low MI, got {mi:.3}");
    }

    #[test]
    fn mi_correlated_signals_high() {
        // Symbol perfectly correlated with predator type
        let mut events = Vec::new();
        let kinds = [
            PredatorKind::Aerial,
            PredatorKind::Ground,
            PredatorKind::Pack,
        ];
        for (sym, kind) in kinds.iter().enumerate() {
            for dist in [1.0, 2.0, 5.0, 8.0] {
                for _ in 0..10 {
                    events.push(make_event(
                        sym as u8,
                        Some(*kind),
                        Some(dist),
                        Position::new(5, 5),
                    ));
                }
            }
        }

        let mi = mutual_information(&events, 4);
        assert!(
            mi > 0.3,
            "Correlated signals should have high MI, got {mi:.3}"
        );
    }

    #[test]
    fn mi_insufficient_data_zero() {
        let events = vec![make_event(0, None, None, Position::new(0, 0))];
        let mi = mutual_information(&events, 4);
        assert!(
            mi.abs() < f64::EPSILON,
            "Insufficient data should give 0.0, got {mi}"
        );
    }

    #[test]
    fn topsim_insufficient_data_zero() {
        let events = vec![make_event(0, None, None, Position::new(0, 0))];
        let ts = topographic_similarity(&events, 4);
        assert!(ts.abs() < f64::EPSILON);
    }

    #[test]
    fn iconicity_insufficient_data_zero() {
        let events = vec![make_event(0, None, None, Position::new(0, 0))];
        let icon = iconicity(&events);
        assert!(icon.abs() < f64::EPSILON);
    }

    #[test]
    fn iconicity_danger_correlated() {
        // More signals emitted near predators -> positive iconicity
        let mut events = Vec::new();
        for _ in 0..50 {
            events.push(make_event(
                0,
                Some(PredatorKind::Aerial),
                Some(1.0),
                Position::new(5, 5),
            ));
        }
        for _ in 0..10 {
            events.push(make_event(
                1,
                Some(PredatorKind::Aerial),
                Some(8.0),
                Position::new(5, 5),
            ));
        }
        for _ in 0..5 {
            events.push(make_event(2, None, None, Position::new(5, 5)));
        }

        let icon = iconicity(&events);
        assert!(
            icon > 0.3,
            "Danger-correlated signals should have positive iconicity, got {icon:.3}"
        );
    }

    #[test]
    fn spearman_perfect_correlation() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let corr = spearman_correlation(&xs, &ys);
        assert!(
            (corr - 1.0).abs() < 0.01,
            "Perfect monotone should give ~1.0, got {corr:.3}"
        );
    }
}
