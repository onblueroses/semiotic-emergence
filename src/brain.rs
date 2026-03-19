use rand::Rng;

pub const INPUTS: usize = 39;

pub const MAX_BASE_HIDDEN: usize = 64;
pub const MIN_BASE_HIDDEN: usize = 4;
pub const DEFAULT_BASE_HIDDEN: usize = 12;

pub const MAX_SIGNAL_HIDDEN: usize = 32;
pub const MIN_SIGNAL_HIDDEN: usize = 2;
pub const DEFAULT_SIGNAL_HIDDEN: usize = 6;

pub const MEMORY_SIZE: usize = 8;

pub const MOVEMENT_OUTPUTS: usize = 5;
pub const SIGNAL_OUTPUTS: usize = 6;
pub const MEMORY_OUTPUTS: usize = MEMORY_SIZE;

/// Genome layout follows forward pass order:
/// `[input->base, base_bias, base->move, move_bias, base->sighid, sighid_bias,
///  sighid->sigout, sigout_bias, base->mem, mem_bias]`
pub const MAX_GENOME_LEN: usize = INPUTS * MAX_BASE_HIDDEN
    + MAX_BASE_HIDDEN
    + MAX_BASE_HIDDEN * MOVEMENT_OUTPUTS
    + MOVEMENT_OUTPUTS
    + MAX_BASE_HIDDEN * MAX_SIGNAL_HIDDEN
    + MAX_SIGNAL_HIDDEN
    + MAX_SIGNAL_HIDDEN * SIGNAL_OUTPUTS
    + SIGNAL_OUTPUTS
    + MAX_BASE_HIDDEN * MEMORY_OUTPUTS
    + MEMORY_OUTPUTS;

/// Pade [1/1] approximation of tanh, clamped to [-1, 1]. 3-5x faster than std.
#[allow(clippy::inline_always)]
#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    let x2 = x * x;
    (x * (27.0 + x2) / (27.0 + 9.0 * x2)).clamp(-1.0, 1.0)
}

// Segment offsets for genome indexing and mutation scoping
pub const SEG_INPUT_BASE: usize = 0;
pub const SEG_BASE_BIAS: usize = SEG_INPUT_BASE + INPUTS * MAX_BASE_HIDDEN;
pub const SEG_BASE_MOVE: usize = SEG_BASE_BIAS + MAX_BASE_HIDDEN;
pub const SEG_MOVE_BIAS: usize = SEG_BASE_MOVE + MAX_BASE_HIDDEN * MOVEMENT_OUTPUTS;
pub const SEG_BASE_SIGHID: usize = SEG_MOVE_BIAS + MOVEMENT_OUTPUTS;
pub const SEG_SIGHID_BIAS: usize = SEG_BASE_SIGHID + MAX_BASE_HIDDEN * MAX_SIGNAL_HIDDEN;
pub const SEG_SIGHID_SIGOUT: usize = SEG_SIGHID_BIAS + MAX_SIGNAL_HIDDEN;
pub const SEG_SIGOUT_BIAS: usize = SEG_SIGHID_SIGOUT + MAX_SIGNAL_HIDDEN * SIGNAL_OUTPUTS;
pub const SEG_BASE_MEM: usize = SEG_SIGOUT_BIAS + SIGNAL_OUTPUTS;
pub const SEG_MEM_BIAS: usize = SEG_BASE_MEM + MAX_BASE_HIDDEN * MEMORY_OUTPUTS;

/// Genome segment boundaries as (start, end) pairs for segment-scoped crossover.
/// Each segment is a functional unit that should be inherited as a whole.
pub const SEGMENT_BOUNDARIES: [(usize, usize); 10] = [
    (SEG_INPUT_BASE, SEG_BASE_BIAS),      // input -> base hidden
    (SEG_BASE_BIAS, SEG_BASE_MOVE),       // base hidden biases
    (SEG_BASE_MOVE, SEG_MOVE_BIAS),       // base -> movement
    (SEG_MOVE_BIAS, SEG_BASE_SIGHID),     // movement biases
    (SEG_BASE_SIGHID, SEG_SIGHID_BIAS),   // base -> signal hidden
    (SEG_SIGHID_BIAS, SEG_SIGHID_SIGOUT), // signal hidden biases
    (SEG_SIGHID_SIGOUT, SEG_SIGOUT_BIAS), // signal hidden -> signal out
    (SEG_SIGOUT_BIAS, SEG_BASE_MEM),      // signal output biases
    (SEG_BASE_MEM, SEG_MEM_BIAS),         // base -> memory
    (SEG_MEM_BIAS, MAX_GENOME_LEN),       // memory biases
];

#[derive(Copy, Clone, Debug)]
pub struct ForwardResult {
    pub actions: [f32; MOVEMENT_OUTPUTS],
    pub signals: [f32; SIGNAL_OUTPUTS],
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
    pub signal_hidden_size: usize,
}

impl Brain {
    pub fn random(rng: &mut impl Rng) -> Self {
        let weights = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
        Self {
            weights,
            base_hidden_size: DEFAULT_BASE_HIDDEN,
            signal_hidden_size: DEFAULT_SIGNAL_HIDDEN,
        }
    }

    #[cfg(test)]
    pub fn zero() -> Self {
        Self {
            weights: [0.0; MAX_GENOME_LEN],
            base_hidden_size: DEFAULT_BASE_HIDDEN,
            signal_hidden_size: DEFAULT_SIGNAL_HIDDEN,
        }
    }

    /// Split forward pass: inputs -> base hidden (tanh) -> {movement (raw), signal hidden (tanh) -> signal (raw), memory (tanh)}.
    /// Loop order is flipped vs naive (outer=source, inner=dest) so weight access is
    /// contiguous in memory, enabling LLVM auto-vectorization (AVX2 on x86).
    #[allow(clippy::needless_range_loop, dead_code)]
    pub fn forward(&self, inputs: &[f32; INPUTS]) -> ForwardResult {
        let w = &self.weights;
        let bh = self.base_hidden_size;
        let sh = self.signal_hidden_size;

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

        // 3. Base hidden -> Signal hidden (tanh)
        let mut sig_hidden = [0.0_f32; MAX_SIGNAL_HIDDEN];
        sig_hidden[..sh].copy_from_slice(&w[SEG_SIGHID_BIAS..SEG_SIGHID_BIAS + sh]);
        for b in 0..bh {
            let hidden_val = base_hidden[b];
            let w_start = SEG_BASE_SIGHID + b * MAX_SIGNAL_HIDDEN;
            for h in 0..sh {
                sig_hidden[h] += hidden_val * w[w_start + h];
            }
        }
        for h in 0..sh {
            sig_hidden[h] = fast_tanh(sig_hidden[h]);
        }

        // 4. Signal hidden -> Signal outputs (raw, softmax applied in signal.rs)
        let mut signals = [0.0_f32; SIGNAL_OUTPUTS];
        signals.copy_from_slice(&w[SEG_SIGOUT_BIAS..SEG_SIGOUT_BIAS + SIGNAL_OUTPUTS]);
        for h in 0..sh {
            let hidden_val = sig_hidden[h];
            let w_start = SEG_SIGHID_SIGOUT + h * SIGNAL_OUTPUTS;
            for o in 0..SIGNAL_OUTPUTS {
                signals[o] += hidden_val * w[w_start + o];
            }
        }

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
            memory_write,
        }
    }
}

/// Dense weight packing for forward pass. Only active weights (determined by
/// `base_hidden_size` and `signal_hidden_size`) are stored contiguously.
/// Built once per generation from Brain; Brain retains full genome for evolution.
pub struct CompactBrain {
    /// All active weights packed contiguously.
    // Layout: base_bias(bh), input_base(INPUTS*bh), base_move(bh*5), move_bias(5),
    //  base_sighid(bh*sh), sighid_bias(sh), sighid_sigout(sh*6), sigout_bias(6),
    //  base_mem(bh*8), mem_bias(8)
    w: Vec<f32>,
    bh: usize,
    sh: usize,
    // Offsets into w for each segment
    o_base_bias: usize,
    o_input_base: usize,
    o_base_move: usize,
    o_move_bias: usize,
    o_base_sighid: usize,
    o_sighid_bias: usize,
    o_sighid_sigout: usize,
    o_sigout_bias: usize,
    o_base_mem: usize,
    o_mem_bias: usize,
}

impl CompactBrain {
    pub fn from_brain(brain: &Brain) -> Self {
        let bh = brain.base_hidden_size;
        let sh = brain.signal_hidden_size;
        let total = bh
            + INPUTS * bh
            + bh * MOVEMENT_OUTPUTS
            + MOVEMENT_OUTPUTS
            + bh * sh
            + sh
            + sh * SIGNAL_OUTPUTS
            + SIGNAL_OUTPUTS
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
        // base_move: pack rows of MOVEMENT_OUTPUTS from rows strided by MOVEMENT_OUTPUTS (already dense per row)
        let o_base_move = w.len();
        for h in 0..bh {
            let src = SEG_BASE_MOVE + h * MOVEMENT_OUTPUTS;
            w.extend_from_slice(&g[src..src + MOVEMENT_OUTPUTS]);
        }
        // move_bias
        let o_move_bias = w.len();
        w.extend_from_slice(&g[SEG_MOVE_BIAS..SEG_MOVE_BIAS + MOVEMENT_OUTPUTS]);
        // base_sighid: rows of sh from rows of MAX_SIGNAL_HIDDEN
        let o_base_sighid = w.len();
        for b in 0..bh {
            let src = SEG_BASE_SIGHID + b * MAX_SIGNAL_HIDDEN;
            w.extend_from_slice(&g[src..src + sh]);
        }
        // sighid_bias
        let o_sighid_bias = w.len();
        w.extend_from_slice(&g[SEG_SIGHID_BIAS..SEG_SIGHID_BIAS + sh]);
        // sighid_sigout: rows of SIGNAL_OUTPUTS from rows strided by SIGNAL_OUTPUTS (dense)
        let o_sighid_sigout = w.len();
        for h in 0..sh {
            let src = SEG_SIGHID_SIGOUT + h * SIGNAL_OUTPUTS;
            w.extend_from_slice(&g[src..src + SIGNAL_OUTPUTS]);
        }
        // sigout_bias
        let o_sigout_bias = w.len();
        w.extend_from_slice(&g[SEG_SIGOUT_BIAS..SEG_SIGOUT_BIAS + SIGNAL_OUTPUTS]);
        // base_mem: rows of MEMORY_OUTPUTS from rows strided by MEMORY_OUTPUTS (dense)
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
            sh,
            o_base_bias,
            o_input_base,
            o_base_move,
            o_move_bias,
            o_base_sighid,
            o_sighid_bias,
            o_sighid_sigout,
            o_sigout_bias,
            o_base_mem,
            o_mem_bias,
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, inputs: &[f32; INPUTS]) -> ForwardResult {
        let w = &self.w;
        let bh = self.bh;
        let sh = self.sh;

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

        // 3. Base hidden -> Signal hidden (tanh)
        let mut sig_hidden = [0.0_f32; MAX_SIGNAL_HIDDEN];
        sig_hidden[..sh].copy_from_slice(&w[self.o_sighid_bias..self.o_sighid_bias + sh]);
        for b in 0..bh {
            let hidden_val = base_hidden[b];
            let row = self.o_base_sighid + b * sh;
            for h in 0..sh {
                sig_hidden[h] += hidden_val * w[row + h];
            }
        }
        for h in 0..sh {
            sig_hidden[h] = fast_tanh(sig_hidden[h]);
        }

        // 4. Signal hidden -> Signal outputs (raw)
        let mut signals = [0.0_f32; SIGNAL_OUTPUTS];
        signals.copy_from_slice(&w[self.o_sigout_bias..self.o_sigout_bias + SIGNAL_OUTPUTS]);
        for h in 0..sh {
            let hidden_val = sig_hidden[h];
            let row = self.o_sighid_sigout + h * SIGNAL_OUTPUTS;
            for o in 0..SIGNAL_OUTPUTS {
                signals[o] += hidden_val * w[row + o];
            }
        }

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
            memory_write,
        }
    }
}

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
        // 39*64 + 64 + 64*5 + 5 + 64*32 + 32 + 32*6 + 6 + 64*8 + 8
        // = 2496 + 64 + 320 + 5 + 2048 + 32 + 192 + 6 + 512 + 8 = 5683
        assert_eq!(MAX_GENOME_LEN, 5683);
    }

    #[test]
    fn segment_offsets_contiguous() {
        assert_eq!(SEG_MEM_BIAS + MEMORY_OUTPUTS, MAX_GENOME_LEN);
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
        for (x, y) in a.memory_write.iter().zip(&b.memory_write) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn forward_respects_base_hidden_size() {
        let mut brain = Brain {
            weights: [0.1; MAX_GENOME_LEN],
            base_hidden_size: MAX_BASE_HIDDEN,
            signal_hidden_size: DEFAULT_SIGNAL_HIDDEN,
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
    fn forward_respects_signal_hidden_size() {
        let mut brain = Brain {
            weights: [0.1; MAX_GENOME_LEN],
            base_hidden_size: DEFAULT_BASE_HIDDEN,
            signal_hidden_size: MAX_SIGNAL_HIDDEN,
        };
        let inputs = [1.0; INPUTS];
        let out_full = brain.forward(&inputs);

        brain.signal_hidden_size = MIN_SIGNAL_HIDDEN;
        let out_small = brain.forward(&inputs);

        let differs = out_full
            .signals
            .iter()
            .zip(&out_small.signals)
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "Different signal_hidden_size should produce different signal outputs"
        );
    }

    #[test]
    fn forward_with_min_hidden() {
        let brain = Brain {
            weights: [0.0; MAX_GENOME_LEN],
            base_hidden_size: MIN_BASE_HIDDEN,
            signal_hidden_size: MIN_SIGNAL_HIDDEN,
        };
        let result = brain.forward(&[1.0; INPUTS]);
        for v in &result.actions {
            assert!(v.abs() < 1e-6);
        }
        for v in &result.signals {
            assert!(v.abs() < 1e-6);
        }
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
        assert_eq!(brain.signal_hidden_size, DEFAULT_SIGNAL_HIDDEN);
        assert!(brain.weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn fast_tanh_accuracy() {
        // Within [-3, 3] (where most NN activations fall), error is <3%
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
        // At extremes, clamped to [-1, 1]
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
    fn compact_brain_matches_brain() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let mut brain = Brain::random(&mut rng);
            brain.base_hidden_size = rng.gen_range(MIN_BASE_HIDDEN..=MAX_BASE_HIDDEN);
            brain.signal_hidden_size = rng.gen_range(MIN_SIGNAL_HIDDEN..=MAX_SIGNAL_HIDDEN);
            let inputs: [f32; INPUTS] = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
            let expected = brain.forward(&inputs);
            let compact = CompactBrain::from_brain(&brain);
            let actual = compact.forward(&inputs);
            for (i, (&e, &a)) in expected.actions.iter().zip(&actual.actions).enumerate() {
                assert!(
                    (e - a).abs() < 1e-5,
                    "action[{i}] mismatch: {e} vs {a} (bh={}, sh={})",
                    brain.base_hidden_size,
                    brain.signal_hidden_size
                );
            }
            for (i, (&e, &a)) in expected.signals.iter().zip(&actual.signals).enumerate() {
                assert!((e - a).abs() < 1e-5, "signal[{i}] mismatch: {e} vs {a}");
            }
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
