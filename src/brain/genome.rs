use crate::brain::activation::ActivationFn;
use crate::brain::innovation::InnovationNumber;

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
