use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

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
        self.inner.gen::<f32>()
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        use rand::seq::SliceRandom;
        slice.shuffle(&mut self.inner);
    }
}
