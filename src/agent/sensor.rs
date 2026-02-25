/// Aggregated sensor reading for a single prey agent.
/// All values normalized to [-1, 1] or [0, 1].
#[derive(Clone, Debug)]
pub struct SensorReading {
    pub inputs: Vec<f32>,
}

impl SensorReading {
    pub fn new(input_count: usize) -> Self {
        Self {
            inputs: vec![0.0; input_count],
        }
    }
}

// Input neuron index constants (with vocab_size=8)
pub const NEAREST_AERIAL_DIST: usize = 0;
pub const NEAREST_AERIAL_DX: usize = 1;
pub const NEAREST_AERIAL_DY: usize = 2;
pub const NEAREST_GROUND_DIST: usize = 3;
pub const NEAREST_GROUND_DX: usize = 4;
pub const NEAREST_GROUND_DY: usize = 5;
pub const NEAREST_PACK_DIST: usize = 6;
pub const NEAREST_PACK_DX: usize = 7;
pub const NEAREST_PACK_DY: usize = 8;
pub const ON_TREE: usize = 9;
pub const ON_ROCK: usize = 10;
pub const NEAREST_TREE_DIST: usize = 11;
pub const NEAREST_ROCK_DIST: usize = 12;
pub const NEAREST_FOOD_DIST: usize = 13;
pub const NEAREST_FOOD_DX: usize = 14;
pub const PREY_DENSITY: usize = 15;
pub const NEAREST_PREY_DIST: usize = 16;
pub const SIGNAL_INPUTS_START: usize = 17;
// Signal inputs: 17 + s*2 = strength, 17 + s*2 + 1 = direction_x (per symbol)
// With vocab_size=8: inputs 17..32 (16 signal inputs)
pub const OWN_ENERGY: usize = 33;
pub const IS_PROTECTED: usize = 34;
pub const TICKS_SINCE_SIGNAL: usize = 35;

pub const INPUT_COUNT_BASE: usize = 36; // With vocab_size=8

pub fn input_count(vocab_size: u8) -> usize {
    17 + (vocab_size as usize * 2) + 3
}

// Output neuron indices
pub const OUT_MOVE_NORTH: usize = 0;
pub const OUT_MOVE_SOUTH: usize = 1;
pub const OUT_MOVE_EAST: usize = 2;
pub const OUT_MOVE_WEST: usize = 3;
pub const OUT_EAT: usize = 4;
pub const OUT_REPRODUCE: usize = 5;
pub const OUT_CLIMB: usize = 6;
pub const OUT_HIDE: usize = 7;
pub const OUT_IDLE: usize = 8;
pub const OUT_SIGNAL_EMIT: usize = 9;
pub const OUT_SIGNAL_SYMBOL_START: usize = 10;

pub fn output_count(vocab_size: u8) -> usize {
    10 + vocab_size as usize
}
