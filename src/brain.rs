use rand::Rng;

pub const INPUTS: usize = 36;

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

#[derive(Copy, Clone, Debug)]
pub struct ForwardResult {
    pub actions: [f32; MOVEMENT_OUTPUTS],
    pub signals: [f32; SIGNAL_OUTPUTS],
    pub memory_write: [f32; MEMORY_OUTPUTS],
}

#[derive(Clone, Debug)]
pub struct Brain {
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
    pub fn forward(&self, inputs: &[f32; INPUTS]) -> ForwardResult {
        let w = &self.weights;
        let bh = self.base_hidden_size;
        let sh = self.signal_hidden_size;

        // 1. Input -> Base hidden (tanh)
        let mut base_hidden = [0.0_f32; MAX_BASE_HIDDEN];
        for h in 0..bh {
            let mut sum = w[SEG_BASE_BIAS + h];
            for i in 0..INPUTS {
                sum += inputs[i] * w[SEG_INPUT_BASE + i * MAX_BASE_HIDDEN + h];
            }
            base_hidden[h] = fast_tanh(sum);
        }

        // 2. Base hidden -> Movement outputs (raw)
        let mut actions = [0.0_f32; MOVEMENT_OUTPUTS];
        for o in 0..MOVEMENT_OUTPUTS {
            let mut sum = w[SEG_MOVE_BIAS + o];
            for h in 0..bh {
                sum += base_hidden[h] * w[SEG_BASE_MOVE + h * MOVEMENT_OUTPUTS + o];
            }
            actions[o] = sum;
        }

        // 3. Base hidden -> Signal hidden (tanh)
        let mut sig_hidden = [0.0_f32; MAX_SIGNAL_HIDDEN];
        for h in 0..sh {
            let mut sum = w[SEG_SIGHID_BIAS + h];
            for b in 0..bh {
                sum += base_hidden[b] * w[SEG_BASE_SIGHID + b * MAX_SIGNAL_HIDDEN + h];
            }
            sig_hidden[h] = fast_tanh(sum);
        }

        // 4. Signal hidden -> Signal outputs (raw, softmax applied in signal.rs)
        let mut signals = [0.0_f32; SIGNAL_OUTPUTS];
        for o in 0..SIGNAL_OUTPUTS {
            let mut sum = w[SEG_SIGOUT_BIAS + o];
            for h in 0..sh {
                sum += sig_hidden[h] * w[SEG_SIGHID_SIGOUT + h * SIGNAL_OUTPUTS + o];
            }
            signals[o] = sum;
        }

        // 5. Base hidden -> Memory outputs (tanh to bound [-1, 1])
        let mut memory_write = [0.0_f32; MEMORY_OUTPUTS];
        for o in 0..MEMORY_OUTPUTS {
            let mut sum = w[SEG_MEM_BIAS + o];
            for h in 0..bh {
                sum += base_hidden[h] * w[SEG_BASE_MEM + h * MEMORY_OUTPUTS + o];
            }
            memory_write[o] = fast_tanh(sum);
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
        // 36*64 + 64 + 64*5 + 5 + 64*32 + 32 + 32*6 + 6 + 64*8 + 8
        // = 2304 + 64 + 320 + 5 + 2048 + 32 + 192 + 6 + 512 + 8 = 5491
        assert_eq!(MAX_GENOME_LEN, 5491);
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
}
