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
    Bias,
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
    /// Node layout: input nodes (`0..input_count`), bias node (`input_count`),
    /// then output nodes (`input_count+1..`). No hidden nodes initially.
    /// Output nodes use Sigmoid activation. Bias node uses Identity.
    pub(crate) fn create_minimal(
        id: GenomeId,
        input_count: usize,
        output_count: usize,
        rng: &mut SeededRng,
        innovations: &mut InnovationCounter,
    ) -> Self {
        let mut nodes = Vec::with_capacity(input_count + 1 + output_count);

        // Input nodes
        for i in 0..input_count {
            nodes.push(NodeGene {
                id: NodeId(i as u32),
                kind: NodeKind::Input,
                activation: ActivationFn::Identity,
                bias: 0.0,
            });
        }

        // Bias node at index input_count
        let bias_id = NodeId(input_count as u32);
        nodes.push(NodeGene {
            id: bias_id,
            kind: NodeKind::Bias,
            activation: ActivationFn::Identity,
            bias: 0.0,
        });

        // Output nodes start after bias
        let output_offset = input_count + 1;
        for i in 0..output_count {
            nodes.push(NodeGene {
                id: NodeId((output_offset + i) as u32),
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
                    let to = NodeId((output_offset + out) as u32);
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

        // Sparse connections from bias to outputs (~20%)
        for out in 0..output_count {
            if rng.gen_f32() < 0.2 {
                let to = NodeId((output_offset + out) as u32);
                let innovation = innovations.get_connection_innovation(bias_id, to);
                connections.push(ConnectionGene {
                    from: bias_id,
                    to,
                    weight: rng.gen_range(-2.0_f32..2.0),
                    enabled: true,
                    innovation,
                });
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
        use std::collections::{HashMap, HashSet};
        let node_ids: HashSet<NodeId> = self.nodes.iter().map(|n| n.id).collect();
        let node_kinds: HashMap<NodeId, NodeKind> =
            self.nodes.iter().map(|n| (n.id, n.kind)).collect();

        // At most one bias node
        let bias_count = self
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Bias)
            .count();
        assert!(
            bias_count <= 1,
            "Genome has {bias_count} bias nodes, expected 0 or 1"
        );

        // Bias nodes must use Identity activation
        for n in &self.nodes {
            if n.kind == NodeKind::Bias {
                assert!(
                    n.activation == ActivationFn::Identity,
                    "Bias node {:?} must use Identity activation",
                    n.id
                );
            }
        }

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
            // No incoming connections to bias or input nodes
            let to_kind = node_kinds[&conn.to];
            assert!(
                to_kind != NodeKind::Bias,
                "Connection targets bias node {:?} - bias nodes cannot have incoming connections",
                conn.to
            );
            assert!(
                to_kind != NodeKind::Input,
                "Connection targets input node {:?} - input nodes cannot have incoming connections",
                conn.to
            );
        }

        // Connections sorted by innovation number
        for w in self.connections.windows(2) {
            assert!(
                w[0].innovation <= w[1].innovation,
                "Connections not sorted by innovation number"
            );
        }
    }
}
