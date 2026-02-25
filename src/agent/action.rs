use crate::signal::message::Symbol;
use crate::world::entity::Direction;

#[derive(Clone, Copy, Debug)]
pub enum Action {
    Move(Direction),
    Eat,
    Signal(Symbol),
    Climb,
    Hide,
    Idle,
}
