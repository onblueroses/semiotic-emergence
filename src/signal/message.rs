use crate::world::entity::{Position, PreyId};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Symbol(pub u8);

pub struct ActiveSignal {
    pub origin: Position,
    pub sender_id: PreyId,
    pub symbol: Symbol,
    pub tick_emitted: u64,
    pub strength: f32,
}
