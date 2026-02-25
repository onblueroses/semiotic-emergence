use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Debug)]
pub struct SeededRng {
    inner: ChaCha8Rng,
    seed: u64,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
            seed,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn gen_range<T, R>(&mut self, range: R) -> T
    where
        T: rand::distributions::uniform::SampleUniform,
        R: rand::distributions::uniform::SampleRange<T>,
    {
        use rand::Rng;
        self.inner.gen_range(range)
    }

    pub fn gen_bool(&mut self, p: f64) -> bool {
        use rand::Rng;
        self.inner.gen_bool(p)
    }

    pub fn gen_f32(&mut self) -> f32 {
        use rand::Rng;
        self.inner.r#gen::<f32>()
    }

    /// Generate a Gaussian-distributed f32 with given mean and standard deviation.
    /// Uses Box-Muller transform.
    pub fn gen_gaussian(&mut self, mean: f32, stdev: f32) -> f32 {
        // Box-Muller: need u1 in (0,1) exclusive of 0 to avoid ln(0)
        let u1 = loop {
            let v = self.gen_f32();
            if v > 0.0 {
                break v;
            }
        };
        let u2 = self.gen_f32();
        let z = (-2.0_f32 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + z * stdev
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        use rand::seq::SliceRandom;
        slice.shuffle(&mut self.inner);
    }
}
