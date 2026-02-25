use crate::agent::action::Action;
use crate::brain::genome::{GenomeId, NeatGenome};
use crate::brain::network::NeatNetwork;
use crate::signal::message::Symbol;
use crate::world::entity::{Direction, LineageId, Position, PreyId};

pub struct Prey {
    pub id: PreyId,
    pub pos: Position,
    pub energy: f32,
    pub age: u32,
    pub genome: NeatGenome,
    pub brain: NeatNetwork,
    pub facing: Direction,
    pub last_action: Action,
    pub last_signal: Option<Symbol>,
    pub lineage: LineageId,
    pub parent_genome_id: Option<GenomeId>,
    pub generation_born: u32,
    pub offspring_count: u32,
    pub fitness_cache: Option<f32>,
    pub ticks_since_signal: u32,
    pub is_climbing: bool,
    pub is_hidden: bool,
}
