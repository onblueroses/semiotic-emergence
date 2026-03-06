pub const SIGNAL_RANGE: f32 = 8.0;
pub const SIGNAL_THRESHOLD: f32 = 0.5;
pub const NUM_SYMBOLS: usize = 3;

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
pub fn receive(signals: &[Signal], rx: i32, ry: i32, current_tick: u32) -> [f32; NUM_SYMBOLS] {
    let mut strengths = [0.0_f32; NUM_SYMBOLS];
    for sig in signals {
        if sig.tick_emitted >= current_tick {
            continue; // 1-tick delay: only past signals
        }
        let dx = (sig.x - rx) as f32;
        let dy = (sig.y - ry) as f32;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > SIGNAL_RANGE {
            continue;
        }
        let strength = 1.0 - dist / SIGNAL_RANGE;
        let sym = sig.symbol as usize;
        if sym < NUM_SYMBOLS && strength > strengths[sym] {
            strengths[sym] = strength;
        }
    }
    strengths
}

/// Decide whether to emit a signal based on NN outputs 5-7.
/// Returns Some(symbol) if max output > threshold, None otherwise.
pub fn maybe_emit(outputs: &[f32], threshold: f32) -> Option<u8> {
    let signal_outs = &outputs[5..8];
    let (max_idx, &max_val) = signal_outs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
    if max_val > threshold {
        Some(max_idx as u8)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_blocks_same_tick() {
        let signals = vec![Signal {
            x: 0,
            y: 0,
            symbol: 0,
            tick_emitted: 5,
        }];
        let strengths = receive(&signals, 1, 0, 5);
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
        let strengths = receive(&signals, 0, 0, 5);
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
        let strengths = receive(&signals, 20, 0, 1);
        assert!(strengths[0].abs() < 1e-6);
    }

    #[test]
    fn emit_below_threshold() {
        let outputs = [0.0; 8];
        assert!(maybe_emit(&outputs, SIGNAL_THRESHOLD).is_none());
    }

    #[test]
    fn emit_above_threshold() {
        let mut outputs = [0.0; 8];
        outputs[6] = 0.8; // signal symbol 1
        assert_eq!(maybe_emit(&outputs, SIGNAL_THRESHOLD), Some(1));
    }
}
