#[derive(Clone, Debug)]
#[expect(
    dead_code,
    reason = "used by world grid food placement; remove when food system is implemented"
)]
pub(crate) struct Food {
    pub(crate) energy: f32,
    pub(crate) regrow_timer: u32,
}

#[expect(
    dead_code,
    reason = "used by world grid food placement; remove when food system is implemented"
)]
impl Food {
    pub(crate) fn new(energy: f32) -> Self {
        Self {
            energy,
            regrow_timer: 0,
        }
    }
}
