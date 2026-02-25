use crate::brain::genome::NodeId;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct InnovationNumber(pub u64);

pub(crate) struct InnovationCounter {
    next_innovation: InnovationNumber,
    generation_innovations: HashMap<(NodeId, NodeId), InnovationNumber>,
    next_node_id: NodeId,
    generation_node_splits: HashMap<InnovationNumber, NodeId>,
}

#[expect(
    dead_code,
    reason = "used by mutation operators in evolution/; remove when mutations are implemented"
)]
impl InnovationCounter {
    pub(crate) fn new(initial_innovation: u64, initial_node_id: u32) -> Self {
        Self {
            next_innovation: InnovationNumber(initial_innovation),
            generation_innovations: HashMap::new(),
            next_node_id: NodeId(initial_node_id),
            generation_node_splits: HashMap::new(),
        }
    }

    pub(crate) fn get_connection_innovation(
        &mut self,
        from: NodeId,
        to: NodeId,
    ) -> InnovationNumber {
        if let Some(&inn) = self.generation_innovations.get(&(from, to)) {
            inn
        } else {
            let inn = self.next_innovation;
            self.next_innovation = InnovationNumber(inn.0 + 1);
            self.generation_innovations.insert((from, to), inn);
            inn
        }
    }

    pub(crate) fn get_node_for_split(&mut self, split_connection: InnovationNumber) -> NodeId {
        if let Some(&node) = self.generation_node_splits.get(&split_connection) {
            node
        } else {
            let node = self.next_node_id;
            self.next_node_id = NodeId(node.0 + 1);
            self.generation_node_splits.insert(split_connection, node);
            node
        }
    }

    pub(crate) fn next_node_id(&self) -> NodeId {
        self.next_node_id
    }

    pub(crate) fn next_innovation(&self) -> InnovationNumber {
        self.next_innovation
    }

    pub(crate) fn reset_generation(&mut self) {
        self.generation_innovations.clear();
        self.generation_node_splits.clear();
    }
}
