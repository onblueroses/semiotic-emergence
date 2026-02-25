#[derive(Clone, Debug)]
pub struct Food {
    pub energy: f32,
    pub regrow_timer: u32,
}

impl Food {
    pub fn new(energy: f32) -> Self {
        Self {
            energy,
            regrow_timer: 0,
        }
    }
}
