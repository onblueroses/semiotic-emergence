use rand::Rng;

pub const INPUTS: usize = 39;

pub const MAX_BASE_HIDDEN: usize = 64;
pub const MIN_BASE_HIDDEN: usize = 4;
pub const DEFAULT_BASE_HIDDEN: usize = 12;

pub const MEMORY_SIZE: usize = 8;

pub const MOVEMENT_OUTPUTS: usize = 5;
pub const SIGNAL_OUTPUTS: usize = 6;
pub const GATE_OUTPUTS: usize = 1;
pub const MEMORY_OUTPUTS: usize = MEMORY_SIZE;

/// v14 shared-layer genome layout (forward pass order):
/// `[input->base, base_bias, base->move, move_bias, base->signal, signal_bias,
///  base->gate, gate_bias, base->mem, mem_bias]`
pub const MAX_GENOME_LEN: usize = INPUTS * MAX_BASE_HIDDEN
    + MAX_BASE_HIDDEN
    + MAX_BASE_HIDDEN * MOVEMENT_OUTPUTS
    + MOVEMENT_OUTPUTS
    + MAX_BASE_HIDDEN * SIGNAL_OUTPUTS
    + SIGNAL_OUTPUTS
    + MAX_BASE_HIDDEN * GATE_OUTPUTS
    + GATE_OUTPUTS
    + MAX_BASE_HIDDEN * MEMORY_OUTPUTS
    + MEMORY_OUTPUTS;

/// Pade [1/1] approximation of tanh, clamped to [-1, 1]. 3-5x faster than std.
#[allow(clippy::inline_always)]
#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    let x2 = x * x;
    (x * (27.0 + x2) / (27.0 + 9.0 * x2)).clamp(-1.0, 1.0)
}

/// Pade approximation of sigmoid: x/(2*(1+|x|)) + 0.5. Monotonic, bounded [0,1].
/// ~3x faster than 1/(1+exp(-x)). Max error ~7.8% at |x|~3.
#[allow(clippy::inline_always)]
#[inline(always)]
fn fast_sigmoid(x: f32) -> f32 {
    x / (2.0 * (1.0 + x.abs())) + 0.5
}

// Segment offsets for genome indexing and mutation scoping
pub const SEG_INPUT_BASE: usize = 0;
pub const SEG_BASE_BIAS: usize = SEG_INPUT_BASE + INPUTS * MAX_BASE_HIDDEN;
pub const SEG_BASE_MOVE: usize = SEG_BASE_BIAS + MAX_BASE_HIDDEN;
pub const SEG_MOVE_BIAS: usize = SEG_BASE_MOVE + MAX_BASE_HIDDEN * MOVEMENT_OUTPUTS;
pub const SEG_BASE_SIGNAL: usize = SEG_MOVE_BIAS + MOVEMENT_OUTPUTS;
pub const SEG_SIGNAL_BIAS: usize = SEG_BASE_SIGNAL + MAX_BASE_HIDDEN * SIGNAL_OUTPUTS;
pub const SEG_BASE_GATE: usize = SEG_SIGNAL_BIAS + SIGNAL_OUTPUTS;
pub const SEG_GATE_BIAS: usize = SEG_BASE_GATE + MAX_BASE_HIDDEN * GATE_OUTPUTS;
pub const SEG_BASE_MEM: usize = SEG_GATE_BIAS + GATE_OUTPUTS;
pub const SEG_MEM_BIAS: usize = SEG_BASE_MEM + MAX_BASE_HIDDEN * MEMORY_OUTPUTS;

/// Crossover groups for v14 shared-layer architecture.
/// Group 0 = LINKED: input->base + `base_bias` (`base_hidden_size` co-inherits from same parent).
/// Groups 1-4 = INDEPENDENT: each output projection + its biases.
pub const CROSSOVER_GROUPS: [(usize, usize); 5] = [
    (SEG_INPUT_BASE, SEG_BASE_MOVE),  // Linked: input->base + base_bias
    (SEG_BASE_MOVE, SEG_BASE_SIGNAL), // Independent: base->move + move_bias
    (SEG_BASE_SIGNAL, SEG_BASE_GATE), // Independent: base->signal + signal_bias
    (SEG_BASE_GATE, SEG_BASE_MEM),    // Independent: base->gate + gate_bias
    (SEG_BASE_MEM, MAX_GENOME_LEN),   // Independent: base->mem + mem_bias
];
/// Index of the linked crossover group (`base_hidden_size` inherits from this parent)
pub const LINKED_GROUP: usize = 0;

#[derive(Copy, Clone, Debug)]
pub struct ForwardResult {
    pub actions: [f32; MOVEMENT_OUTPUTS],
    pub signals: [f32; SIGNAL_OUTPUTS],
    pub gate_value: f32,
    pub memory_write: [f32; MEMORY_OUTPUTS],
}

mod weights_serde {
    use super::MAX_GENOME_LEN;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        weights: &[f32; MAX_GENOME_LEN],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        weights.as_slice().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; MAX_GENOME_LEN], D::Error> {
        let v: Vec<f32> = Vec::deserialize(d)?;
        v.try_into().map_err(|v: Vec<f32>| {
            serde::de::Error::custom(format!(
                "expected {} weights, got {}",
                MAX_GENOME_LEN,
                v.len()
            ))
        })
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Brain {
    #[serde(with = "weights_serde")]
    pub weights: [f32; MAX_GENOME_LEN],
    pub base_hidden_size: usize,
}

impl Brain {
    pub fn random(rng: &mut impl Rng) -> Self {
        let weights = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
        Self {
            weights,
            base_hidden_size: DEFAULT_BASE_HIDDEN,
        }
    }

    #[cfg(test)]
    pub fn zero() -> Self {
        Self {
            weights: [0.0; MAX_GENOME_LEN],
            base_hidden_size: DEFAULT_BASE_HIDDEN,
        }
    }

    /// Shared-layer forward pass: inputs -> base hidden (tanh) -> {movement (raw), signal (raw), gate (sigmoid), memory (tanh)}.
    /// Loop order: outer=source, inner=dest for contiguous weight access (LLVM auto-vectorization).
    #[allow(clippy::needless_range_loop, dead_code)]
    pub fn forward(&self, inputs: &[f32; INPUTS]) -> ForwardResult {
        let w = &self.weights;
        let bh = self.base_hidden_size;

        // 1. Input -> Base hidden (tanh)
        let mut base_hidden = [0.0_f32; MAX_BASE_HIDDEN];
        base_hidden[..bh].copy_from_slice(&w[SEG_BASE_BIAS..SEG_BASE_BIAS + bh]);
        for i in 0..INPUTS {
            let inp_val = inputs[i];
            let w_start = SEG_INPUT_BASE + i * MAX_BASE_HIDDEN;
            for h in 0..bh {
                base_hidden[h] += inp_val * w[w_start + h];
            }
        }
        for h in 0..bh {
            base_hidden[h] = fast_tanh(base_hidden[h]);
        }

        // 2. Base hidden -> Movement outputs (raw)
        let mut actions = [0.0_f32; MOVEMENT_OUTPUTS];
        actions.copy_from_slice(&w[SEG_MOVE_BIAS..SEG_MOVE_BIAS + MOVEMENT_OUTPUTS]);
        for h in 0..bh {
            let hidden_val = base_hidden[h];
            let w_start = SEG_BASE_MOVE + h * MOVEMENT_OUTPUTS;
            for o in 0..MOVEMENT_OUTPUTS {
                actions[o] += hidden_val * w[w_start + o];
            }
        }

        // 3. Base hidden -> Signal outputs (raw, symbol chosen by argmax)
        let mut signals = [0.0_f32; SIGNAL_OUTPUTS];
        signals.copy_from_slice(&w[SEG_SIGNAL_BIAS..SEG_SIGNAL_BIAS + SIGNAL_OUTPUTS]);
        for h in 0..bh {
            let hidden_val = base_hidden[h];
            let w_start = SEG_BASE_SIGNAL + h * SIGNAL_OUTPUTS;
            for o in 0..SIGNAL_OUTPUTS {
                signals[o] += hidden_val * w[w_start + o];
            }
        }

        // 4. Base hidden -> Gate (sigmoid, emit/suppress decision)
        let mut gate_raw = w[SEG_GATE_BIAS];
        for h in 0..bh {
            gate_raw += base_hidden[h] * w[SEG_BASE_GATE + h];
        }
        let gate_value = fast_sigmoid(gate_raw);

        // 5. Base hidden -> Memory outputs (tanh to bound [-1, 1])
        let mut memory_write = [0.0_f32; MEMORY_OUTPUTS];
        memory_write.copy_from_slice(&w[SEG_MEM_BIAS..SEG_MEM_BIAS + MEMORY_OUTPUTS]);
        for h in 0..bh {
            let hidden_val = base_hidden[h];
            let w_start = SEG_BASE_MEM + h * MEMORY_OUTPUTS;
            for o in 0..MEMORY_OUTPUTS {
                memory_write[o] += hidden_val * w[w_start + o];
            }
        }
        for o in 0..MEMORY_OUTPUTS {
            memory_write[o] = fast_tanh(memory_write[o]);
        }

        ForwardResult {
            actions,
            signals,
            gate_value,
            memory_write,
        }
    }
}

/// Dense weight packing for forward pass. Only active weights (determined by
/// `base_hidden_size`) are stored contiguously.
/// Built once per generation from Brain; Brain retains full genome for evolution.
pub struct CompactBrain {
    /// All active weights packed contiguously.
    // Layout: base_bias(bh), input_base(INPUTS*bh), base_move(bh*5), move_bias(5),
    //  base_signal(bh*6), signal_bias(6), base_gate(bh*1), gate_bias(1),
    //  base_mem(bh*8), mem_bias(8)
    w: Vec<f32>,
    bh: usize,
    // Offsets into w for each segment
    o_base_bias: usize,
    o_input_base: usize,
    o_base_move: usize,
    o_move_bias: usize,
    o_base_signal: usize,
    o_signal_bias: usize,
    o_base_gate: usize,
    o_gate_bias: usize,
    o_base_mem: usize,
    o_mem_bias: usize,
}

impl CompactBrain {
    pub fn from_brain(brain: &Brain) -> Self {
        let bh = brain.base_hidden_size;
        let total = bh
            + INPUTS * bh
            + bh * MOVEMENT_OUTPUTS
            + MOVEMENT_OUTPUTS
            + bh * SIGNAL_OUTPUTS
            + SIGNAL_OUTPUTS
            + bh * GATE_OUTPUTS
            + GATE_OUTPUTS
            + bh * MEMORY_OUTPUTS
            + MEMORY_OUTPUTS;
        let mut w = Vec::with_capacity(total);
        let g = &brain.weights;

        // base_bias
        let o_base_bias = 0;
        w.extend_from_slice(&g[SEG_BASE_BIAS..SEG_BASE_BIAS + bh]);
        // input_base: pack rows of bh from rows of MAX_BASE_HIDDEN
        let o_input_base = w.len();
        for i in 0..INPUTS {
            let src = SEG_INPUT_BASE + i * MAX_BASE_HIDDEN;
            w.extend_from_slice(&g[src..src + bh]);
        }
        // base_move: pack rows of MOVEMENT_OUTPUTS
        let o_base_move = w.len();
        for h in 0..bh {
            let src = SEG_BASE_MOVE + h * MOVEMENT_OUTPUTS;
            w.extend_from_slice(&g[src..src + MOVEMENT_OUTPUTS]);
        }
        // move_bias
        let o_move_bias = w.len();
        w.extend_from_slice(&g[SEG_MOVE_BIAS..SEG_MOVE_BIAS + MOVEMENT_OUTPUTS]);
        // base_signal: pack rows of SIGNAL_OUTPUTS from rows strided by SIGNAL_OUTPUTS
        let o_base_signal = w.len();
        for h in 0..bh {
            let src = SEG_BASE_SIGNAL + h * SIGNAL_OUTPUTS;
            w.extend_from_slice(&g[src..src + SIGNAL_OUTPUTS]);
        }
        // signal_bias
        let o_signal_bias = w.len();
        w.extend_from_slice(&g[SEG_SIGNAL_BIAS..SEG_SIGNAL_BIAS + SIGNAL_OUTPUTS]);
        // base_gate: bh weights (one column)
        let o_base_gate = w.len();
        for h in 0..bh {
            w.push(g[SEG_BASE_GATE + h]);
        }
        // gate_bias
        let o_gate_bias = w.len();
        w.push(g[SEG_GATE_BIAS]);
        // base_mem: pack rows of MEMORY_OUTPUTS
        let o_base_mem = w.len();
        for h in 0..bh {
            let src = SEG_BASE_MEM + h * MEMORY_OUTPUTS;
            w.extend_from_slice(&g[src..src + MEMORY_OUTPUTS]);
        }
        // mem_bias
        let o_mem_bias = w.len();
        w.extend_from_slice(&g[SEG_MEM_BIAS..SEG_MEM_BIAS + MEMORY_OUTPUTS]);

        debug_assert_eq!(w.len(), total);
        Self {
            w,
            bh,
            o_base_bias,
            o_input_base,
            o_base_move,
            o_move_bias,
            o_base_signal,
            o_signal_bias,
            o_base_gate,
            o_gate_bias,
            o_base_mem,
            o_mem_bias,
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, inputs: &[f32; INPUTS]) -> ForwardResult {
        let w = &self.w;
        let bh = self.bh;

        // 1. Input -> Base hidden (tanh)
        let mut base_hidden = [0.0_f32; MAX_BASE_HIDDEN];
        base_hidden[..bh].copy_from_slice(&w[self.o_base_bias..self.o_base_bias + bh]);
        for i in 0..INPUTS {
            let inp_val = inputs[i];
            let row = self.o_input_base + i * bh;
            for h in 0..bh {
                base_hidden[h] += inp_val * w[row + h];
            }
        }
        for h in 0..bh {
            base_hidden[h] = fast_tanh(base_hidden[h]);
        }

        // 2. Base hidden -> Movement outputs (raw)
        let mut actions = [0.0_f32; MOVEMENT_OUTPUTS];
        actions.copy_from_slice(&w[self.o_move_bias..self.o_move_bias + MOVEMENT_OUTPUTS]);
        for h in 0..bh {
            let hidden_val = base_hidden[h];
            let row = self.o_base_move + h * MOVEMENT_OUTPUTS;
            for o in 0..MOVEMENT_OUTPUTS {
                actions[o] += hidden_val * w[row + o];
            }
        }

        // 3. Base hidden -> Signal outputs (raw)
        let mut signals = [0.0_f32; SIGNAL_OUTPUTS];
        signals.copy_from_slice(&w[self.o_signal_bias..self.o_signal_bias + SIGNAL_OUTPUTS]);
        for h in 0..bh {
            let hidden_val = base_hidden[h];
            let row = self.o_base_signal + h * SIGNAL_OUTPUTS;
            for o in 0..SIGNAL_OUTPUTS {
                signals[o] += hidden_val * w[row + o];
            }
        }

        // 4. Base hidden -> Gate (sigmoid)
        let mut gate_raw = w[self.o_gate_bias];
        for h in 0..bh {
            gate_raw += base_hidden[h] * w[self.o_base_gate + h];
        }
        let gate_value = fast_sigmoid(gate_raw);

        // 5. Base hidden -> Memory outputs (tanh)
        let mut memory_write = [0.0_f32; MEMORY_OUTPUTS];
        memory_write.copy_from_slice(&w[self.o_mem_bias..self.o_mem_bias + MEMORY_OUTPUTS]);
        for h in 0..bh {
            let hidden_val = base_hidden[h];
            let row = self.o_base_mem + h * MEMORY_OUTPUTS;
            for o in 0..MEMORY_OUTPUTS {
                memory_write[o] += hidden_val * w[row + o];
            }
        }
        for o in 0..MEMORY_OUTPUTS {
            memory_write[o] = fast_tanh(memory_write[o]);
        }

        ForwardResult {
            actions,
            signals,
            gate_value,
            memory_write,
        }
    }
}

#[allow(dead_code)] // Kept for analysis/metrics use (Decision #7)
pub fn softmax(logits: &[f32; SIGNAL_OUTPUTS]) -> [f32; SIGNAL_OUTPUTS] {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut result = [0.0_f32; SIGNAL_OUTPUTS];
    let mut sum = 0.0_f32;
    for (i, &l) in logits.iter().enumerate() {
        result[i] = (l - max).exp();
        sum += result[i];
    }
    for v in &mut result {
        *v /= sum;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genome_length() {
        // 39*64 + 64 + 64*5 + 5 + 64*6 + 6 + 64*1 + 1 + 64*8 + 8
        // = 2496 + 64 + 320 + 5 + 384 + 6 + 64 + 1 + 512 + 8 = 3860
        assert_eq!(MAX_GENOME_LEN, 3860);
    }

    #[test]
    fn segment_offsets_contiguous() {
        assert_eq!(SEG_MEM_BIAS + MEMORY_OUTPUTS, MAX_GENOME_LEN);
    }

    #[test]
    fn crossover_groups_cover_genome() {
        assert_eq!(CROSSOVER_GROUPS[0].0, 0);
        assert_eq!(CROSSOVER_GROUPS[4].1, MAX_GENOME_LEN);
        for i in 0..4 {
            assert_eq!(
                CROSSOVER_GROUPS[i].1,
                CROSSOVER_GROUPS[i + 1].0,
                "gap between groups {i} and {}",
                i + 1
            );
        }
    }

    #[test]
    fn zero_weights_zero_output() {
        let brain = Brain::zero();
        let result = brain.forward(&[0.0; INPUTS]);
        for v in &result.actions {
            assert!(v.abs() < 1e-6);
        }
        for v in &result.signals {
            assert!(v.abs() < 1e-6);
        }
        // Gate with zero weights and zero bias: fast_sigmoid(0) = 0.5
        assert!((result.gate_value - 0.5).abs() < 1e-6);
        for v in &result.memory_write {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn forward_deterministic() {
        let mut rng = rand::thread_rng();
        let brain = Brain::random(&mut rng);
        let inputs = [0.5; INPUTS];
        let a = brain.forward(&inputs);
        let b = brain.forward(&inputs);
        for (x, y) in a.actions.iter().zip(&b.actions) {
            assert!((x - y).abs() < 1e-10);
        }
        for (x, y) in a.signals.iter().zip(&b.signals) {
            assert!((x - y).abs() < 1e-10);
        }
        assert!((a.gate_value - b.gate_value).abs() < 1e-10);
        for (x, y) in a.memory_write.iter().zip(&b.memory_write) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn forward_respects_base_hidden_size() {
        let mut brain = Brain {
            weights: [0.1; MAX_GENOME_LEN],
            base_hidden_size: MAX_BASE_HIDDEN,
        };
        let inputs = [1.0; INPUTS];
        let out_full = brain.forward(&inputs);

        brain.base_hidden_size = MIN_BASE_HIDDEN;
        let out_small = brain.forward(&inputs);

        let differs = out_full
            .actions
            .iter()
            .zip(&out_small.actions)
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "Different base_hidden_size should produce different action outputs"
        );
    }

    #[test]
    fn forward_with_min_hidden() {
        let brain = Brain {
            weights: [0.0; MAX_GENOME_LEN],
            base_hidden_size: MIN_BASE_HIDDEN,
        };
        let result = brain.forward(&[1.0; INPUTS]);
        for v in &result.actions {
            assert!(v.abs() < 1e-6);
        }
        for v in &result.signals {
            assert!(v.abs() < 1e-6);
        }
        assert!((result.gate_value - 0.5).abs() < 1e-6);
        for v in &result.memory_write {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn memory_outputs_bounded() {
        let mut rng = rand::thread_rng();
        let brain = Brain::random(&mut rng);
        let inputs: [f32; INPUTS] = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
        let result = brain.forward(&inputs);
        for v in &result.memory_write {
            assert!((-1.0..=1.0).contains(v), "Memory output {v} not in [-1, 1]");
        }
    }

    #[test]
    fn gate_output_bounded() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let brain = Brain::random(&mut rng);
            let inputs: [f32; INPUTS] = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
            let result = brain.forward(&inputs);
            assert!(
                (0.0..=1.0).contains(&result.gate_value),
                "Gate value {} not in [0, 1]",
                result.gate_value
            );
        }
    }

    #[test]
    fn softmax_uniform() {
        let logits = [0.0; SIGNAL_OUTPUTS];
        let probs = softmax(&logits);
        let expected = 1.0 / SIGNAL_OUTPUTS as f32;
        for p in &probs {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_concentrated() {
        let mut logits = [0.0; SIGNAL_OUTPUTS];
        logits[2] = 10.0;
        let probs = softmax(&logits);
        assert!(probs[2] > 0.99);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_no_nan() {
        let logits = [f32::MAX / 2.0; SIGNAL_OUTPUTS];
        let probs = softmax(&logits);
        for p in &probs {
            assert!(!p.is_nan(), "softmax produced NaN");
        }
    }

    #[test]
    fn zero_constructor_defaults() {
        let brain = Brain::zero();
        assert_eq!(brain.base_hidden_size, DEFAULT_BASE_HIDDEN);
        assert!(brain.weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn fast_tanh_accuracy() {
        for i in -30..=30 {
            let x = i as f32 / 10.0;
            let approx = fast_tanh(x);
            let exact = x.tanh();
            let err = (approx - exact).abs();
            assert!(
                err < 0.03,
                "fast_tanh({x}) = {approx}, std = {exact}, err = {err}"
            );
        }
        assert!((fast_tanh(10.0) - 1.0).abs() < 1e-6);
        assert!((fast_tanh(-10.0) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn fast_tanh_zero_and_odd() {
        assert!((fast_tanh(0.0)).abs() < 1e-10);
        for i in 1..=50 {
            let x = i as f32 / 10.0;
            assert!(
                (fast_tanh(x) + fast_tanh(-x)).abs() < 1e-6,
                "fast_tanh not odd at {x}"
            );
        }
    }

    #[test]
    fn fast_sigmoid_accuracy() {
        // Within [-3, 3], error is <8%
        for i in -30..=30 {
            let x = i as f32 / 10.0;
            let approx = fast_sigmoid(x);
            let exact = 1.0 / (1.0 + (-x).exp());
            let err = (approx - exact).abs();
            assert!(
                err < 0.08,
                "fast_sigmoid({x}) = {approx}, std = {exact}, err = {err}"
            );
        }
        // Midpoint
        assert!((fast_sigmoid(0.0) - 0.5).abs() < 1e-10);
        // Bounded
        assert!(fast_sigmoid(100.0) > 0.99);
        assert!(fast_sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn fast_sigmoid_monotonic() {
        let mut prev = fast_sigmoid(-10.0);
        for i in -99..=100 {
            let x = i as f32 / 10.0;
            let y = fast_sigmoid(x);
            assert!(y >= prev, "fast_sigmoid not monotonic at {x}");
            prev = y;
        }
    }

    #[test]
    fn compact_brain_matches_brain() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let mut brain = Brain::random(&mut rng);
            brain.base_hidden_size = rng.gen_range(MIN_BASE_HIDDEN..=MAX_BASE_HIDDEN);
            let inputs: [f32; INPUTS] = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
            let expected = brain.forward(&inputs);
            let compact = CompactBrain::from_brain(&brain);
            let actual = compact.forward(&inputs);
            for (i, (&e, &a)) in expected.actions.iter().zip(&actual.actions).enumerate() {
                assert!(
                    (e - a).abs() < 1e-5,
                    "action[{i}] mismatch: {e} vs {a} (bh={})",
                    brain.base_hidden_size
                );
            }
            for (i, (&e, &a)) in expected.signals.iter().zip(&actual.signals).enumerate() {
                assert!((e - a).abs() < 1e-5, "signal[{i}] mismatch: {e} vs {a}");
            }
            assert!(
                (expected.gate_value - actual.gate_value).abs() < 1e-5,
                "gate mismatch: {} vs {}",
                expected.gate_value,
                actual.gate_value
            );
            for (i, (&e, &a)) in expected
                .memory_write
                .iter()
                .zip(&actual.memory_write)
                .enumerate()
            {
                assert!((e - a).abs() < 1e-5, "memory[{i}] mismatch: {e} vs {a}");
            }
        }
    }
}
