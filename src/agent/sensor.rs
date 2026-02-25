use crate::agent::action::Action;
use crate::signal::message::Symbol;
use crate::world::entity::Direction;

/// Aggregated sensor reading for a single prey agent.
/// All values normalized to [-1, 1] or [0, 1].
#[derive(Clone, Debug)]
pub(crate) struct SensorReading {
    pub(crate) inputs: Vec<f32>,
}

impl SensorReading {
    pub(crate) fn new(input_count: usize) -> Self {
        Self {
            inputs: vec![0.0; input_count],
        }
    }
}

// Input neuron index constants (with vocab_size=8)
pub(crate) const NEAREST_AERIAL_DIST: usize = 0;
pub(crate) const NEAREST_AERIAL_DX: usize = 1;
pub(crate) const NEAREST_AERIAL_DY: usize = 2;
pub(crate) const NEAREST_GROUND_DIST: usize = 3;
pub(crate) const NEAREST_GROUND_DX: usize = 4;
pub(crate) const NEAREST_GROUND_DY: usize = 5;
pub(crate) const NEAREST_PACK_DIST: usize = 6;
pub(crate) const NEAREST_PACK_DX: usize = 7;
pub(crate) const NEAREST_PACK_DY: usize = 8;
pub(crate) const ON_TREE: usize = 9;
pub(crate) const ON_ROCK: usize = 10;
pub(crate) const NEAREST_TREE_DIST: usize = 11;
pub(crate) const NEAREST_ROCK_DIST: usize = 12;
pub(crate) const NEAREST_FOOD_DIST: usize = 13;
pub(crate) const NEAREST_FOOD_DX: usize = 14;
pub(crate) const PREY_DENSITY: usize = 15;
pub(crate) const NEAREST_PREY_DIST: usize = 16;
pub(crate) const SIGNAL_INPUTS_START: usize = 17;
// Signal inputs: 17 + s*2 = strength, 17 + s*2 + 1 = direction_x (per symbol)
// With vocab_size=8: inputs 17..32 (16 signal inputs)
#[expect(
    dead_code,
    reason = "used once sensor encoding references these by name"
)]
pub(crate) const OWN_ENERGY: usize = 33;
#[expect(
    dead_code,
    reason = "used once sensor encoding references these by name"
)]
pub(crate) const IS_PROTECTED: usize = 34;
#[expect(
    dead_code,
    reason = "used once sensor encoding references these by name"
)]
pub(crate) const TICKS_SINCE_SIGNAL: usize = 35;

#[expect(dead_code, reason = "documents canonical input count for reference")]
pub(crate) const INPUT_COUNT_BASE: usize = 36; // With vocab_size=8

pub(crate) fn input_count(vocab_size: u8) -> usize {
    17 + (vocab_size as usize * 2) + 3
}

// Output neuron indices
pub(crate) const OUT_MOVE_NORTH: usize = 0;
pub(crate) const OUT_MOVE_SOUTH: usize = 1;
pub(crate) const OUT_MOVE_EAST: usize = 2;
pub(crate) const OUT_MOVE_WEST: usize = 3;
pub(crate) const OUT_EAT: usize = 4;
pub(crate) const OUT_CLIMB: usize = 5;
pub(crate) const OUT_HIDE: usize = 6;
pub(crate) const OUT_IDLE: usize = 7;
pub(crate) const OUT_SIGNAL_EMIT: usize = 8;
pub(crate) const OUT_SIGNAL_SYMBOL_START: usize = 9;

pub(crate) fn output_count(vocab_size: u8) -> usize {
    9 + vocab_size as usize
}

/// Decode neural network outputs into an action and optional signal.
///
/// Primary action: argmax over outputs[0..8] (movement dirs + eat/climb/hide/idle).
/// Signal: if outputs[8] > 0.5, also emit a signal (symbol = argmax of symbol outputs).
pub(crate) fn decode_outputs(outputs: &[f32], vocab_size: u8) -> (Action, Option<Symbol>) {
    // Primary action: argmax of first 8 outputs
    let action_outputs = &outputs[..8.min(outputs.len())];
    let (best_idx, _) = action_outputs.iter().enumerate().fold(
        (OUT_IDLE, f32::NEG_INFINITY),
        |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        },
    );

    let action = match best_idx {
        OUT_MOVE_NORTH => Action::Move(Direction::North),
        OUT_MOVE_SOUTH => Action::Move(Direction::South),
        OUT_MOVE_EAST => Action::Move(Direction::East),
        OUT_MOVE_WEST => Action::Move(Direction::West),
        OUT_EAT => Action::Eat,
        OUT_CLIMB => Action::Climb,
        OUT_HIDE => Action::Hide,
        _ => Action::Idle,
    };

    // Signal: non-exclusive, can co-occur with any primary action
    let signal = if outputs.len() > OUT_SIGNAL_EMIT && outputs[OUT_SIGNAL_EMIT] > 0.5 {
        let sym_start = OUT_SIGNAL_SYMBOL_START;
        let sym_end = (sym_start + vocab_size as usize).min(outputs.len());
        let sym_outputs = &outputs[sym_start..sym_end];
        let (sym_idx, _) =
            sym_outputs
                .iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                });
        Some(Symbol(sym_idx as u8))
    } else {
        None
    };

    (action, signal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_idle_on_zeros() {
        let outputs = vec![0.0; 17]; // 9 + vocab_size(8) = 17
        let (action, signal) = decode_outputs(&outputs, 8);
        // All zeros: first index (NORTH) wins in tie
        assert!(matches!(action, Action::Move(Direction::North)));
        assert!(signal.is_none());
    }

    #[test]
    fn decode_eat_action() {
        let mut outputs = vec![0.0; 17];
        outputs[OUT_EAT] = 1.0;
        let (action, _) = decode_outputs(&outputs, 8);
        assert!(matches!(action, Action::Eat));
    }

    #[test]
    fn decode_signal_when_above_threshold() {
        let mut outputs = vec![0.0; 17];
        outputs[OUT_EAT] = 1.0;
        outputs[OUT_SIGNAL_EMIT] = 0.6;
        outputs[OUT_SIGNAL_SYMBOL_START + 3] = 1.0; // Symbol 3
        let (action, signal) = decode_outputs(&outputs, 8);
        assert!(matches!(action, Action::Eat));
        assert_eq!(signal, Some(Symbol(3)));
    }

    #[test]
    fn decode_no_signal_below_threshold() {
        let mut outputs = vec![0.0; 17];
        outputs[OUT_SIGNAL_EMIT] = 0.4;
        outputs[OUT_SIGNAL_SYMBOL_START] = 1.0;
        let (_, signal) = decode_outputs(&outputs, 8);
        assert!(signal.is_none());
    }
}
