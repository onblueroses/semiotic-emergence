use crate::brain::{softmax, SIGNAL_OUTPUTS};
use crate::world::wrap_delta;

pub const NUM_SYMBOLS: usize = 6;

#[derive(Clone, Debug)]
pub struct Signal {
    pub x: i32,
    pub y: i32,
    pub symbol: u8,
    pub tick_emitted: u32,
}

/// Compute received signal strengths for a prey at (rx, ry) on the given tick.
/// Returns strength per symbol `[0..NUM_SYMBOLS]`. Only signals from previous ticks
/// are receivable (1-tick delay). Strongest-per-symbol wins.
#[cfg(test)]
pub fn receive(
    signals: &[Signal],
    rx: i32,
    ry: i32,
    current_tick: u32,
    signal_range: f32,
    grid_size: i32,
) -> [f32; NUM_SYMBOLS] {
    let mut strengths = [0.0_f32; NUM_SYMBOLS];
    for sig in signals {
        if sig.tick_emitted >= current_tick {
            continue; // 1-tick delay: only past signals
        }
        let dx = wrap_delta(rx, sig.x, grid_size) as f32;
        let dy = wrap_delta(ry, sig.y, grid_size) as f32;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > signal_range {
            continue;
        }
        let strength = 1.0 - dist / signal_range;
        let sym = sig.symbol as usize;
        if sym < NUM_SYMBOLS && strength > strengths[sym] {
            strengths[sym] = strength;
        }
    }
    strengths
}

#[derive(Clone, Copy, Debug)]
pub struct ReceivedSignal {
    pub strength: f32,
    pub dx: f32,
    pub dy: f32,
}

/// Compute detailed received signals including direction to strongest emitter per symbol.
pub fn receive_detailed(
    signals: &[Signal],
    rx: i32,
    ry: i32,
    current_tick: u32,
    grid_size: f32,
    signal_range: f32,
) -> [ReceivedSignal; NUM_SYMBOLS] {
    let grid_size_i = grid_size as i32;
    let range_sq = signal_range * signal_range;
    let mut result = std::array::from_fn::<_, NUM_SYMBOLS, _>(|_| ReceivedSignal {
        strength: 0.0,
        dx: 0.0,
        dy: 0.0,
    });
    for sig in signals {
        if sig.tick_emitted >= current_tick {
            continue;
        }
        let dx = wrap_delta(rx, sig.x, grid_size_i) as f32;
        let dy = wrap_delta(ry, sig.y, grid_size_i) as f32;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq > range_sq {
            continue;
        }
        let dist = dist_sq.sqrt();
        let strength = 1.0 - dist / signal_range;
        let sym = sig.symbol as usize;
        if sym < NUM_SYMBOLS && strength > result[sym].strength {
            result[sym].strength = strength;
            result[sym].dx = dx / grid_size;
            result[sym].dy = dy / grid_size;
        }
    }
    result
}

/// Decide whether to emit a signal based on softmax of NN signal outputs.
/// Returns `Some(symbol)` if max softmax probability > `1/NUM_SYMBOLS`, None otherwise.
pub fn maybe_emit(signal_outputs: &[f32; SIGNAL_OUTPUTS]) -> Option<u8> {
    let probs = softmax(signal_outputs);
    let (max_idx, &max_prob) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
    let threshold = 1.0 / NUM_SYMBOLS as f32;
    if max_prob > threshold {
        Some(max_idx as u8)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SIGNAL_RANGE: f32 = 8.0;
    const TEST_GRID_SIZE: i32 = 20;

    #[test]
    fn delay_blocks_same_tick() {
        let signals = vec![Signal {
            x: 0,
            y: 0,
            symbol: 0,
            tick_emitted: 5,
        }];
        let strengths = receive(&signals, 1, 0, 5, TEST_SIGNAL_RANGE, TEST_GRID_SIZE);
        assert!(strengths[0].abs() < 1e-6);
    }

    #[test]
    fn previous_tick_received() {
        let signals = vec![Signal {
            x: 0,
            y: 0,
            symbol: 1,
            tick_emitted: 4,
        }];
        let strengths = receive(&signals, 0, 0, 5, TEST_SIGNAL_RANGE, TEST_GRID_SIZE);
        assert!((strengths[1] - 1.0).abs() < 1e-6); // same cell = max strength
    }

    #[test]
    fn out_of_range_not_received() {
        let signals = vec![Signal {
            x: 0,
            y: 0,
            symbol: 0,
            tick_emitted: 0,
        }];
        // Place receiver 10 cells away (> SIGNAL_RANGE of 8)
        let strengths = receive(&signals, 10, 0, 1, TEST_SIGNAL_RANGE, TEST_GRID_SIZE);
        assert!(strengths[0].abs() < 1e-6);
    }

    #[test]
    fn receive_detailed_returns_direction() {
        let signals = vec![Signal {
            x: 5,
            y: 3,
            symbol: 0,
            tick_emitted: 4,
        }];
        let result = receive_detailed(&signals, 2, 1, 5, 20.0, TEST_SIGNAL_RANGE);
        assert!(result[0].strength > 0.0);
        assert!((result[0].dx - 3.0 / 20.0).abs() < 1e-6);
        assert!((result[0].dy - 2.0 / 20.0).abs() < 1e-6);
        // Other symbols should be zero
        assert!(result[1].strength.abs() < 1e-6);
        assert!(result[2].strength.abs() < 1e-6);
    }

    #[test]
    fn emit_uniform_returns_none() {
        // Equal logits -> softmax = 1/6 each -> max == threshold -> no emission
        let outputs = [0.0_f32; SIGNAL_OUTPUTS];
        assert!(maybe_emit(&outputs).is_none());
    }

    #[test]
    fn emit_concentrated_returns_symbol() {
        let mut outputs = [0.0_f32; SIGNAL_OUTPUTS];
        outputs[3] = 5.0; // symbol 3 dominant
        assert_eq!(maybe_emit(&outputs), Some(3));
    }

    #[test]
    fn emit_slight_difference_returns_some() {
        let mut outputs = [0.0_f32; SIGNAL_OUTPUTS];
        outputs[1] = 0.01; // barely above others
        assert_eq!(maybe_emit(&outputs), Some(1));
    }
}
