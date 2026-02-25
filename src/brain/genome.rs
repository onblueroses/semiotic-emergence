use crate::brain::activation::ActivationFn;
use crate::brain::innovation::{InnovationCounter, InnovationNumber};
use crate::rng::SeededRng;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GenomeId(pub u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SpeciesId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeKind {
    Input,
    Hidden,
    Output,
}

#[derive(Clone, Debug)]
pub struct NodeGene {
    pub id: NodeId,
    pub kind: NodeKind,
    pub activation: ActivationFn,
    pub bias: f32,
}

#[derive(Clone, Debug)]
pub struct ConnectionGene {
    pub from: NodeId,
    pub to: NodeId,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: InnovationNumber,
}

#[derive(Clone, Debug)]
pub struct NeatGenome {
    pub id: GenomeId,
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
    pub fitness: f32,
    pub species_id: Option<SpeciesId>,
}

impl NeatGenome {
    pub fn new(id: GenomeId, nodes: Vec<NodeGene>, connections: Vec<ConnectionGene>) -> Self {
        Self {
            id,
            nodes,
            connections,
            fitness: 0.0,
            species_id: None,
        }
    }

    pub fn sort_connections(&mut self) {
        self.connections.sort_by_key(|c| c.innovation);
    }

    /// Create a minimal genome with sparse random connections (~20% of possible).
    ///
    /// Input nodes (`0..input_count`), then output nodes. No hidden nodes initially.
    /// Output nodes use Sigmoid activation.
    pub(crate) fn create_minimal(
        id: GenomeId,
        input_count: usize,
        output_count: usize,
        rng: &mut SeededRng,
        innovations: &mut InnovationCounter,
    ) -> Self {
        let mut nodes = Vec::with_capacity(input_count + output_count);

        // Input nodes: identity activation (Sigmoid is fine, they pass raw values)
        for i in 0..input_count {
            nodes.push(NodeGene {
                id: NodeId(i as u32),
                kind: NodeKind::Input,
                activation: ActivationFn::Sigmoid,
                bias: 0.0,
            });
        }

        // Output nodes
        for i in 0..output_count {
            nodes.push(NodeGene {
                id: NodeId((input_count + i) as u32),
                kind: NodeKind::Output,
                activation: ActivationFn::Sigmoid,
                bias: 0.0,
            });
        }

        // Sparse random connections: ~20% of input-to-output pairs
        let mut connections = Vec::new();
        for inp in 0..input_count {
            for out in 0..output_count {
                if rng.gen_f32() < 0.2 {
                    let from = NodeId(inp as u32);
                    let to = NodeId((input_count + out) as u32);
                    let innovation = innovations.get_connection_innovation(from, to);
                    connections.push(ConnectionGene {
                        from,
                        to,
                        weight: rng.gen_range(-2.0_f32..2.0),
                        enabled: true,
                        innovation,
                    });
                }
            }
        }

        let mut genome = Self::new(id, nodes, connections);
        genome.sort_connections();
        #[cfg(debug_assertions)]
        genome.validate();
        genome
    }

    #[cfg(debug_assertions)]
    pub fn validate(&self) {
        use std::collections::HashSet;
        let node_ids: HashSet<NodeId> = self.nodes.iter().map(|n| n.id).collect();
        for conn in &self.connections {
            assert!(
                node_ids.contains(&conn.from),
                "Connection references non-existent source node {:?}",
                conn.from
            );
            assert!(
                node_ids.contains(&conn.to),
                "Connection references non-existent target node {:?}",
                conn.to
            );
        }
        for w in self.connections.windows(2) {
            assert!(
                w[0].innovation <= w[1].innovation,
                "Connections not sorted by innovation number"
            );
        }
    }
}
