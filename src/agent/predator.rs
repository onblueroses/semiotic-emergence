use crate::world::entity::{Position, PredatorId, PreyId};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PredatorKind {
    Aerial,
    Ground,
    Pack,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PredatorState {
    Roaming,
    Stalking(PreyId),
    Attacking,
    Resting,
}

#[expect(
    dead_code,
    reason = "fields read during predator AI phase 4; remove then"
)]
pub(crate) struct Predator {
    pub(crate) id: PredatorId,
    pub(crate) kind: PredatorKind,
    pub(crate) pos: Position,
    pub(crate) energy: f32,
    pub(crate) state: PredatorState,
    pub(crate) target: Option<PreyId>,
    pub(crate) cooldown: u32,
}
