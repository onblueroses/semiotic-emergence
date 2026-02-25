use crate::agent::predator::PredatorKind;
use crate::world::entity::{Position, PreyId};

/// A single signal emission event, recorded per-tick for MI/TopSim analysis.
#[derive(Clone, Debug)]
pub(crate) struct SignalEvent {
    #[expect(
        dead_code,
        reason = "used by future per-event analysis and JSON export"
    )]
    pub(crate) emitter_id: PreyId,
    pub(crate) symbol: u8,
    pub(crate) emitter_pos: Position,
    pub(crate) nearest_predator_kind: Option<PredatorKind>,
    pub(crate) nearest_predator_dist: Option<f32>,
    #[expect(
        dead_code,
        reason = "used by future per-event analysis and JSON export"
    )]
    pub(crate) tick: u64,
}

/// Aggregate statistics for one generation.
#[derive(Clone, Debug)]
pub(crate) struct GenerationStats {
    pub(crate) generation: u32,
    pub(crate) avg_fitness: f32,
    pub(crate) max_fitness: f32,
    pub(crate) species_count: u32,
    pub(crate) prey_alive_end: u32,
    pub(crate) signal_count: u32,
    pub(crate) mutual_information: f64,
    pub(crate) topographic_similarity: f64,
    pub(crate) iconicity: f64,
}

/// Collects signal events during a generation and aggregated stats across generations.
pub(crate) struct StatsCollector {
    /// Signal events for the current generation (cleared each gen).
    pub(crate) current_events: Vec<SignalEvent>,
    /// Accumulated per-generation stats.
    pub(crate) generations: Vec<GenerationStats>,
}

impl StatsCollector {
    pub(crate) fn new() -> Self {
        Self {
            current_events: Vec::new(),
            generations: Vec::new(),
        }
    }

    /// Record a signal emission event.
    pub(crate) fn record_signal(&mut self, event: SignalEvent) {
        self.current_events.push(event);
    }

    /// Finalize the current generation: compute metrics and store stats.
    pub(crate) fn finalize_generation(
        &mut self,
        generation: u32,
        avg_fitness: f32,
        max_fitness: f32,
        species_count: u32,
        prey_alive_end: u32,
        vocab_size: u32,
    ) {
        let signal_count = self.current_events.len() as u32;

        let mi = super::metrics::mutual_information(&self.current_events, vocab_size);
        let topsim = super::metrics::topographic_similarity(&self.current_events, vocab_size);
        let icon = super::metrics::iconicity(&self.current_events);

        self.generations.push(GenerationStats {
            generation,
            avg_fitness,
            max_fitness,
            species_count,
            prey_alive_end,
            signal_count,
            mutual_information: mi,
            topographic_similarity: topsim,
            iconicity: icon,
        });

        self.current_events.clear();
    }
}
