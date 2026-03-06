use rand::Rng;

pub const INPUTS: usize = 10;
pub const HIDDEN: usize = 6;
pub const OUTPUTS: usize = 8;
pub const GENOME_LEN: usize = INPUTS * HIDDEN + HIDDEN + HIDDEN * OUTPUTS + OUTPUTS;

#[derive(Clone, Debug)]
pub struct Brain {
    /// Weights: [input->hidden (60), hidden biases (6), hidden->output (48), output biases (8)]
    pub weights: Vec<f32>,
}

impl Brain {
    pub fn random(rng: &mut impl Rng) -> Self {
        let weights = (0..GENOME_LEN).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self { weights }
    }

    /// Feed-forward: 10 inputs -> 6 hidden (tanh) -> 8 outputs (raw, no activation)
    pub fn forward(&self, inputs: &[f32; INPUTS]) -> [f32; OUTPUTS] {
        let w = &self.weights;
        let mut hidden = [0.0_f32; HIDDEN];
        for h in 0..HIDDEN {
            let mut sum = w[INPUTS * HIDDEN + h]; // bias
            for i in 0..INPUTS {
                sum += inputs[i] * w[i * HIDDEN + h];
            }
            hidden[h] = sum.tanh();
        }

        let offset = INPUTS * HIDDEN + HIDDEN;
        let bias_offset = offset + HIDDEN * OUTPUTS;
        let mut outputs = [0.0_f32; OUTPUTS];
        for o in 0..OUTPUTS {
            let mut sum = w[bias_offset + o]; // bias
            for h in 0..HIDDEN {
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
        assert_eq!(GENOME_LEN, 122);
    }

    #[test]
    fn zero_weights_zero_output() {
        let brain = Brain {
            weights: vec![0.0; GENOME_LEN],
        };
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
}
