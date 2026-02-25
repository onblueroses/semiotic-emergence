#![expect(
    dead_code,
    reason = "sensor constants used when encoding is implemented; remove then"
)]

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
pub(crate) const OWN_ENERGY: usize = 33;
pub(crate) const IS_PROTECTED: usize = 34;
pub(crate) const TICKS_SINCE_SIGNAL: usize = 35;

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
pub(crate) const OUT_REPRODUCE: usize = 5;
pub(crate) const OUT_CLIMB: usize = 6;
pub(crate) const OUT_HIDE: usize = 7;
pub(crate) const OUT_IDLE: usize = 8;
pub(crate) const OUT_SIGNAL_EMIT: usize = 9;
pub(crate) const OUT_SIGNAL_SYMBOL_START: usize = 10;

pub(crate) fn output_count(vocab_size: u8) -> usize {
    10 + vocab_size as usize
}
