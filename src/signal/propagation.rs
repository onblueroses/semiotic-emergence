use crate::config::SignalConfig;
use crate::signal::message::{ActiveSignal, Symbol};
use crate::world::entity::{Position, PreyId};

/// Received signal data: (symbol, strength, `direction_x`, `direction_y`).
pub(crate) struct ReceivedSignal {
    pub(crate) symbol: Symbol,
    pub(crate) strength: f32,
    pub(crate) direction_x: f32,
    #[expect(
        dead_code,
        reason = "will be wired to sensor inputs in Phase 2 expansion"
    )]
    pub(crate) direction_y: f32,
}

/// Get the strongest signal per symbol receivable at a position.
///
/// Enforces one-tick delay: only signals with `tick_emitted < current_tick` are receivable.
/// Returns strongest signal per symbol within hearing range.
pub(crate) fn receive_signals(
    signals: &[ActiveSignal],
    receiver_pos: Position,
    hearing_range: u32,
    current_tick: u64,
    config: &SignalConfig,
) -> Vec<ReceivedSignal> {
    let range_f = config.signal_range as f32;
    let hearing_f = hearing_range as f32;

    // Track strongest signal per symbol
    let mut best: Vec<Option<ReceivedSignal>> = (0..config.vocab_size).map(|_| None).collect();

    for signal in signals {
        // One-tick delay: only receive signals from previous ticks
        if signal.tick_emitted >= current_tick {
            continue;
        }

        let dist = receiver_pos.distance_to(&signal.origin);
        if dist > hearing_f {
            continue;
        }

        // Strength attenuated by distance
        let strength = signal.strength * (1.0 - dist / range_f).max(0.0);
        if strength <= 0.0 {
            continue;
        }

        let sym_idx = signal.symbol.0 as usize;
        if sym_idx >= config.vocab_size as usize {
            continue;
        }

        let (dx, dy) = receiver_pos.direction_to(&signal.origin);

        let is_stronger = best[sym_idx]
            .as_ref()
            .is_none_or(|existing| strength > existing.strength);

        if is_stronger {
            best[sym_idx] = Some(ReceivedSignal {
                symbol: signal.symbol,
                strength,
                direction_x: dx,
                direction_y: dy,
            });
        }
    }

    best.into_iter().flatten().collect()
}

/// Decay all signals and remove dead ones.
///
/// Reduces strength by `decay_rate` each tick. Removes signals that have
/// zero/negative strength or have exceeded their lifetime.
pub(crate) fn decay_signals(
    signals: &mut Vec<ActiveSignal>,
    current_tick: u64,
    config: &SignalConfig,
) {
    signals.retain_mut(|signal| {
        let age = current_tick.saturating_sub(signal.tick_emitted);
        if age > u64::from(config.signal_lifetime) {
            return false;
        }
        // Don't decay signals emitted this tick - they haven't been receivable yet
        if age == 0 {
            return true;
        }
        signal.strength -= config.signal_decay_rate;
        signal.strength > 0.0
    });
}

/// Create a new signal at a position.
pub(crate) fn create_signal(
    sender_id: PreyId,
    origin: Position,
    symbol: Symbol,
    tick: u64,
) -> ActiveSignal {
    ActiveSignal {
        origin,
        sender_id,
        symbol,
        tick_emitted: tick,
        strength: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SignalConfig;
    use crate::signal::message::Symbol;
    use crate::world::entity::{Position, PreyId};

    fn test_config() -> SignalConfig {
        SignalConfig {
            vocab_size: 8,
            signal_range: 12,
            signal_decay_rate: 0.3,
            signal_lifetime: 3,
        }
    }

    #[test]
    fn one_tick_delay_enforced() {
        let config = test_config();
        let signal = create_signal(PreyId(0), Position::new(5, 5), Symbol(0), 10);
        let signals = vec![signal];
        let recv_pos = Position::new(6, 5);

        // Same tick: should NOT receive
        let result = receive_signals(&signals, recv_pos, 12, 10, &config);
        assert!(result.is_empty());

        // Next tick: should receive
        let result = receive_signals(&signals, recv_pos, 12, 11, &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].symbol, Symbol(0));
    }

    #[test]
    fn strongest_per_symbol_wins() {
        let config = test_config();
        // Two signals with same symbol, different distances
        let close = ActiveSignal {
            origin: Position::new(5, 5),
            sender_id: PreyId(0),
            symbol: Symbol(1),
            tick_emitted: 0,
            strength: 1.0,
        };
        let far = ActiveSignal {
            origin: Position::new(10, 5),
            sender_id: PreyId(1),
            symbol: Symbol(1),
            tick_emitted: 0,
            strength: 1.0,
        };
        let signals = vec![close, far];
        let recv_pos = Position::new(5, 6); // 1 cell from close, 5 from far

        let result = receive_signals(&signals, recv_pos, 12, 1, &config);
        assert_eq!(result.len(), 1);
        // Closer signal should be stronger
        assert!(result[0].strength > 0.5);
    }

    #[test]
    fn signal_decay_removes_old() {
        let config = test_config();
        let mut signals = vec![
            create_signal(PreyId(0), Position::new(5, 5), Symbol(0), 0),
            create_signal(PreyId(1), Position::new(10, 10), Symbol(1), 3),
        ];

        // After 4 ticks, first signal is past lifetime (3), should be removed
        decay_signals(&mut signals, 4, &config);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].symbol, Symbol(1));
    }
}
