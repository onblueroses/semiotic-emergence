use crate::world::entity::{PredatorId, Position, PreyId};

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

pub struct Predator {
    pub id: PredatorId,
    pub kind: PredatorKind,
    pub pos: Position,
    pub energy: f32,
    pub state: PredatorState,
    pub target: Option<PreyId>,
    pub cooldown: u32,
}
