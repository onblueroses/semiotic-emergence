#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActivationFn {
    Sigmoid,
    Tanh,
    Relu,
}

pub fn apply_activation(func: ActivationFn, x: f32) -> f32 {
    match func {
        ActivationFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        ActivationFn::Tanh => x.tanh(),
        ActivationFn::Relu => x.max(0.0),
    }
}
