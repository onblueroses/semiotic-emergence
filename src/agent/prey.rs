use crate::agent::action::Action;
use crate::brain::genome::{GenomeId, NeatGenome};
use crate::brain::network::NeatNetwork;
use crate::signal::message::Symbol;
use crate::world::entity::{Direction, LineageId, Position, PreyId};

#[expect(
    dead_code,
    reason = "fields read during tick phases 2-5; remove as phases are implemented"
)]
pub(crate) struct Prey {
    pub(crate) id: PreyId,
    pub(crate) pos: Position,
    pub(crate) energy: f32,
    pub(crate) age: u32,
    pub(crate) genome: NeatGenome,
    pub(crate) brain: NeatNetwork,
    pub(crate) facing: Direction,
    pub(crate) last_action: Action,
    pub(crate) last_signal: Option<Symbol>,
    pub(crate) lineage: LineageId,
    pub(crate) parent_genome_id: Option<GenomeId>,
    pub(crate) generation_born: u32,
    pub(crate) offspring_count: u32,
    pub(crate) fitness_cache: Option<f32>,
    pub(crate) ticks_since_signal: u32,
    pub(crate) is_climbing: bool,
    pub(crate) is_hidden: bool,
}
