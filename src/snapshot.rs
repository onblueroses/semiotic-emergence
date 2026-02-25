use crate::agent::action::Action;
use crate::agent::predator::{PredatorKind, PredatorState};
use crate::signal::message::Symbol;
use crate::world::entity::Direction;
use crate::world::terrain::Terrain;

/// Read-only snapshot of the world state for rendering and analysis.
#[derive(Clone)]
pub struct WorldSnapshot {
    pub tick: u64,
    pub generation: u32,
    pub width: u32,
    pub height: u32,
    pub terrain: Vec<Terrain>,
    pub food: Vec<bool>,
    pub prey: Vec<AgentSnapshot>,
    pub predators: Vec<PredatorSnapshot>,
    pub signals: Vec<SignalSnapshot>,
}

#[derive(Clone)]
pub struct AgentSnapshot {
    pub id: u32,
    pub x: u32,
    pub y: u32,
    pub energy: f32,
    pub facing: Direction,
    pub last_action: Action,
    pub last_signal: Option<Symbol>,
    pub is_climbing: bool,
    pub is_hidden: bool,
    pub lineage: u32,
}

#[derive(Clone)]
pub struct PredatorSnapshot {
    pub id: u32,
    pub kind: PredatorKind,
    pub x: u32,
    pub y: u32,
    pub state: PredatorState,
}

#[derive(Clone)]
pub struct SignalSnapshot {
    pub x: u32,
    pub y: u32,
    pub symbol: u8,
    pub strength: f32,
}
