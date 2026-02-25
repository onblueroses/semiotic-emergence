use crate::brain::activation::{ActivationFn, apply_activation};
use crate::brain::genome::{NeatGenome, NodeKind};
use std::ops::Range;

struct NetworkNode {
    kind: NodeKind,
    activation: ActivationFn,
    bias: f32,
    value: f32,
    incoming: Range<usize>,
}

struct NetworkEdge {
    from_index: usize,
    weight: f32,
}

pub(crate) struct NeatNetwork {
    nodes: Vec<NetworkNode>,
    edges: Vec<NetworkEdge>,
    input_count: usize,
    output_count: usize,
}

impl NeatNetwork {
    /// Build a feedforward network from a genome via topological sort.
    pub(crate) fn from_genome(genome: &NeatGenome) -> Self {
        use std::collections::{HashMap, VecDeque};

        let input_count = genome
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Input)
            .count();
        let output_count = genome
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Output)
            .count();

        // Build adjacency list from enabled connections
        let mut in_edges: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();
        let mut out_neighbors: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut in_degree: HashMap<u32, usize> = HashMap::new();

        for node in &genome.nodes {
            in_edges.entry(node.id.0).or_default();
            out_neighbors.entry(node.id.0).or_default();
            in_degree.entry(node.id.0).or_insert(0);
        }

        for conn in &genome.connections {
            if !conn.enabled {
                continue;
            }
            in_edges
                .entry(conn.to.0)
                .or_default()
                .push((conn.from.0, conn.weight));
            out_neighbors
                .entry(conn.from.0)
                .or_default()
                .push(conn.to.0);
            *in_degree.entry(conn.to.0).or_insert(0) += 1;
        }

        // Topological sort (Kahn's algorithm)
        let mut queue: VecDeque<u32> = VecDeque::new();
        for node in &genome.nodes {
            if in_degree.get(&node.id.0).copied().unwrap_or(0) == 0 {
                queue.push_back(node.id.0);
            }
        }

        let mut sorted_ids: Vec<u32> = Vec::new();
        while let Some(node_id) = queue.pop_front() {
            sorted_ids.push(node_id);
            let empty = Vec::new();
            for &neighbor in out_neighbors.get(&node_id).unwrap_or(&empty) {
                if let Some(deg) = in_degree.get_mut(&neighbor) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Any nodes not in sorted_ids are part of cycles - skip them (feedforward only)
        let node_map: HashMap<u32, &crate::brain::genome::NodeGene> =
            genome.nodes.iter().map(|n| (n.id.0, n)).collect();

        // Build index map: node_id -> position in sorted order
        let mut id_to_index: HashMap<u32, usize> = HashMap::new();
        for (i, &id) in sorted_ids.iter().enumerate() {
            id_to_index.insert(id, i);
        }

        // Build network nodes and edges
        let mut net_nodes: Vec<NetworkNode> = Vec::new();
        let mut net_edges: Vec<NetworkEdge> = Vec::new();

        for &id in &sorted_ids {
            let gene = node_map[&id];
            let edge_start = net_edges.len();

            // Add incoming edges for this node
            if let Some(edges) = in_edges.get(&id) {
                for &(from_id, weight) in edges {
                    if let Some(&from_idx) = id_to_index.get(&from_id) {
                        net_edges.push(NetworkEdge {
                            from_index: from_idx,
                            weight,
                        });
                    }
                }
            }

            net_nodes.push(NetworkNode {
                kind: gene.kind,
                activation: gene.activation,
                bias: gene.bias,
                value: 0.0,
                incoming: edge_start..net_edges.len(),
            });
        }

        Self {
            nodes: net_nodes,
            edges: net_edges,
            input_count,
            output_count,
        }
    }

    /// Feed inputs through the network, return output values.
    pub(crate) fn activate(&mut self, inputs: &[f32]) -> Vec<f32> {
        // Set input node values
        for (i, &val) in inputs.iter().enumerate().take(self.input_count) {
            self.nodes[i].value = val;
        }

        // Forward pass through bias + hidden + output nodes
        for i in self.input_count..self.nodes.len() {
            // Bias nodes always output 1.0 - skip activation
            if self.nodes[i].kind == NodeKind::Bias {
                self.nodes[i].value = 1.0;
                continue;
            }

            let incoming = self.nodes[i].incoming.clone();
            let bias = self.nodes[i].bias;
            let activation = self.nodes[i].activation;

            let mut sum = bias;
            for edge_idx in incoming {
                let edge = &self.edges[edge_idx];
                sum += self.nodes[edge.from_index].value * edge.weight;
            }
            self.nodes[i].value = apply_activation(activation, sum);
        }

        // Read output values (last output_count nodes in sorted order)
        let output_start = self.nodes.len() - self.output_count;
        self.nodes[output_start..].iter().map(|n| n.value).collect()
    }
}
