use rand::Rng;

pub const INPUTS: usize = 16;
pub const MAX_HIDDEN: usize = 16;
pub const MIN_HIDDEN: usize = 4;
pub const DEFAULT_HIDDEN: usize = 6;
pub const OUTPUTS: usize = 8;
/// Fixed genome length sized for `MAX_HIDDEN`. Dormant neurons are "junk DNA".
pub const MAX_GENOME_LEN: usize = INPUTS * MAX_HIDDEN + MAX_HIDDEN + MAX_HIDDEN * OUTPUTS + OUTPUTS;

#[derive(Clone, Debug)]
pub struct Brain {
    /// Weights: [input->hidden (I*MH=256), hidden biases (MH=16), hidden->output (MH*O=128), output biases (O=8)]
    pub weights: [f32; MAX_GENOME_LEN],
    /// Number of active hidden neurons (`MIN_HIDDEN..=MAX_HIDDEN`). Heritable, mutable.
    pub hidden_size: usize,
}

impl Brain {
    pub fn random(rng: &mut impl Rng) -> Self {
        let weights = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
        Self {
            weights,
            hidden_size: DEFAULT_HIDDEN,
        }
    }

    #[cfg(test)]
    pub fn zero() -> Self {
        Self {
            weights: [0.0; MAX_GENOME_LEN],
            hidden_size: DEFAULT_HIDDEN,
        }
    }

    /// Feed-forward: 16 inputs -> `hidden_size` hidden (tanh) -> 8 outputs (raw).
    /// Stride is `MAX_HIDDEN` so weight indices are stable regardless of active `hidden_size`.
    pub fn forward(&self, inputs: &[f32; INPUTS]) -> [f32; OUTPUTS] {
        let w = &self.weights;
        let h_size = self.hidden_size;

        // Input -> Hidden (only first h_size neurons)
        let mut hidden = [0.0_f32; MAX_HIDDEN];
        for h in 0..h_size {
            let mut sum = w[INPUTS * MAX_HIDDEN + h]; // bias
            for i in 0..INPUTS {
                sum += inputs[i] * w[i * MAX_HIDDEN + h];
            }
            hidden[h] = sum.tanh();
        }

        // Hidden -> Output (only active hidden neurons contribute)
        let offset = INPUTS * MAX_HIDDEN + MAX_HIDDEN;
        let bias_offset = offset + MAX_HIDDEN * OUTPUTS;
        let mut outputs = [0.0_f32; OUTPUTS];
        for o in 0..OUTPUTS {
            let mut sum = w[bias_offset + o]; // bias
            for h in 0..h_size {
                sum += hidden[h] * w[offset + h * OUTPUTS + o];
            }
            outputs[o] = sum;
        }
        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genome_length() {
        // 16*16 + 16 + 16*8 + 8 = 256 + 16 + 128 + 8 = 408
        assert_eq!(MAX_GENOME_LEN, 408);
    }

    #[test]
    fn zero_weights_zero_output() {
        let brain = Brain::zero();
        let out = brain.forward(&[0.0; INPUTS]);
        for v in &out {
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
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn forward_respects_hidden_size() {
        let mut brain = Brain {
            weights: [0.1; MAX_GENOME_LEN],
            hidden_size: MAX_HIDDEN,
        };
        let inputs = [1.0; INPUTS];
        let out_full = brain.forward(&inputs);

        brain.hidden_size = MIN_HIDDEN;
        let out_small = brain.forward(&inputs);

        // Fewer active neurons means different (generally smaller magnitude) outputs
        let differs = out_full
            .iter()
            .zip(&out_small)
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "Different hidden_size should produce different outputs"
        );
    }

    #[test]
    fn forward_with_min_hidden() {
        let brain = Brain {
            weights: [0.0; MAX_GENOME_LEN],
            hidden_size: MIN_HIDDEN,
        };
        let out = brain.forward(&[1.0; INPUTS]);
        // Zero weights, zero output regardless of hidden_size
        for v in &out {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn forward_with_max_hidden() {
        let mut rng = rand::thread_rng();
        let mut brain = Brain::random(&mut rng);
        brain.hidden_size = MAX_HIDDEN;
        let inputs = [0.5; INPUTS];
        let a = brain.forward(&inputs);
        let b = brain.forward(&inputs);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn zero_constructor_defaults() {
        let brain = Brain::zero();
        assert_eq!(brain.hidden_size, DEFAULT_HIDDEN);
        assert!(brain.weights.iter().all(|&w| w == 0.0));
    }
}
