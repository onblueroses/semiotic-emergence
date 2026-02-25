pub mod config;
pub mod rng;

pub mod agent;
pub mod brain;
pub mod evolution;
pub mod signal;
pub mod snapshot;
pub mod stats;
pub mod world;

pub mod simulation;

#[cfg(feature = "terminal-viz")]
pub mod viz;

pub use simulation::run_simulation;
