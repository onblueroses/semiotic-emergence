use crate::brain::activation::ActivationFn;
use crate::brain::genome::{ConnectionGene, GenomeId, NeatGenome, NodeGene, NodeId, NodeKind};
use crate::brain::innovation::InnovationCounter;
use crate::config::SimConfig;
use crate::rng::SeededRng;

// ---------------------------------------------------------------------------
// Weight and bias mutation (Step 2.1)
// ---------------------------------------------------------------------------

/// Mutate connection weights and node biases in-place.
///
/// For each connection: `weight_mutate_rate` chance to mutate.
///   If mutating: `weight_perturb_rate` chance to perturb (Gaussian), else replace uniform [-1,1].
/// For each non-input/non-bias node: `bias_mutate_rate` chance to perturb bias (Gaussian stdev 0.5).
pub(crate) fn mutate_weights(genome: &mut NeatGenome, config: &SimConfig, rng: &mut SeededRng) {
    let neat = &config.neat;

    for conn in &mut genome.connections {
        if rng.gen_bool(neat.weight_mutate_rate) {
            if rng.gen_bool(neat.weight_perturb_rate) {
                conn.weight += rng.gen_gaussian(0.0, neat.weight_perturb_strength);
            } else {
                conn.weight = rng.gen_range(-1.0_f32..1.0);
            }
            conn.weight = conn.weight.clamp(-8.0, 8.0);
        }
    }

    for node in &mut genome.nodes {
        if matches!(node.kind, NodeKind::Input | NodeKind::Bias) {
            continue;
        }
        if rng.gen_bool(neat.bias_mutate_rate) {
            node.bias += rng.gen_gaussian(0.0, 0.5);
            node.bias = node.bias.clamp(-8.0, 8.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Add connection mutation (Step 2.2)
// ---------------------------------------------------------------------------

/// Attempt to add a new connection between two unconnected nodes.
///
/// Source: any non-output node. Target: any non-input, non-bias node with higher ID
/// (ensures feedforward). If a connection already exists, bail.
pub(crate) fn mutate_add_connection(
    genome: &mut NeatGenome,
    rng: &mut SeededRng,
    innovations: &mut InnovationCounter,
) -> bool {
    // Collect valid source nodes (input, bias, hidden - not output)
    let sources: Vec<NodeId> = genome
        .nodes
        .iter()
        .filter(|n| !matches!(n.kind, NodeKind::Output))
        .map(|n| n.id)
        .collect();

    // Collect valid target nodes (hidden, output - not input, not bias)
    let targets: Vec<NodeId> = genome
        .nodes
        .iter()
        .filter(|n| !matches!(n.kind, NodeKind::Input | NodeKind::Bias))
        .map(|n| n.id)
        .collect();

    if sources.is_empty() || targets.is_empty() {
        return false;
    }

    // Build set of existing connections for quick lookup
    let existing: std::collections::HashSet<(NodeId, NodeId)> =
        genome.connections.iter().map(|c| (c.from, c.to)).collect();

    // Try up to 20 random pairs to find a valid new connection
    for _ in 0..20 {
        let from = sources[rng.gen_range(0..sources.len())];
        let to = targets[rng.gen_range(0..targets.len())];

        // Feedforward: from.0 must be less than to.0 (lower layer to higher)
        if from.0 >= to.0 {
            continue;
        }

        if existing.contains(&(from, to)) {
            continue;
        }

        let innovation = innovations.get_connection_innovation(from, to);
        genome.connections.push(ConnectionGene {
            from,
            to,
            weight: rng.gen_range(-1.0_f32..1.0),
            enabled: true,
            innovation,
        });
        genome.sort_connections();
        return true;
    }

    false
}

// ---------------------------------------------------------------------------
// Add node mutation (Step 2.3)
// ---------------------------------------------------------------------------

/// Split an existing enabled connection by inserting a new hidden node.
///
/// Original connection is disabled. Two new connections added:
///   `from_node` -> `new_node` (weight 1.0)
///   `new_node` -> `to_node` (weight = original weight)
/// New node uses Sigmoid activation (D19).
pub(crate) fn mutate_add_node(
    genome: &mut NeatGenome,
    rng: &mut SeededRng,
    innovations: &mut InnovationCounter,
) -> bool {
    let enabled_indices: Vec<usize> = genome
        .connections
        .iter()
        .enumerate()
        .filter(|(_, c)| c.enabled)
        .map(|(i, _)| i)
        .collect();

    if enabled_indices.is_empty() {
        return false;
    }

    let idx = enabled_indices[rng.gen_range(0..enabled_indices.len())];
    let old_from = genome.connections[idx].from;
    let old_to = genome.connections[idx].to;
    let old_weight = genome.connections[idx].weight;
    let old_innovation = genome.connections[idx].innovation;

    // Disable original connection
    genome.connections[idx].enabled = false;

    // Get or create node for this split (same structural mutation in same gen -> same node ID)
    let new_node_id = innovations.get_node_for_split(old_innovation);

    // Add the new hidden node (only if not already present from a previous split of same connection)
    if !genome.nodes.iter().any(|n| n.id == new_node_id) {
        genome.nodes.push(NodeGene {
            id: new_node_id,
            kind: NodeKind::Hidden,
            activation: ActivationFn::Sigmoid,
            bias: 0.0,
        });
    }

    // Add two new connections
    let inn_in = innovations.get_connection_innovation(old_from, new_node_id);
    let inn_out = innovations.get_connection_innovation(new_node_id, old_to);

    // Only add if not already present
    if !genome
        .connections
        .iter()
        .any(|c| c.from == old_from && c.to == new_node_id)
    {
        genome.connections.push(ConnectionGene {
            from: old_from,
            to: new_node_id,
            weight: 1.0,
            enabled: true,
            innovation: inn_in,
        });
    }

    if !genome
        .connections
        .iter()
        .any(|c| c.from == new_node_id && c.to == old_to)
    {
        genome.connections.push(ConnectionGene {
            from: new_node_id,
            to: old_to,
            weight: old_weight,
            enabled: true,
            innovation: inn_out,
        });
    }

    genome.sort_connections();
    true
}

// ---------------------------------------------------------------------------
// Crossover (Step 2.4)
// ---------------------------------------------------------------------------

/// Produce a child genome from two parents via NEAT crossover.
///
/// Walk both parents' sorted connections by innovation number:
/// - Matching genes: 50/50 pick from either parent
/// - Disabled in either parent: 75% chance child inherits disabled (D18)
/// - Disjoint/excess: come from the fitter parent only
/// - If equal fitness: fitter = shorter genome (fewer connections)
///
/// Child gets all node genes from the fitter parent, plus any nodes referenced
/// by inherited connections from the other parent.
pub(crate) fn crossover(
    parent_a: &NeatGenome,
    fitness_a: f32,
    parent_b: &NeatGenome,
    fitness_b: f32,
    child_id: GenomeId,
    rng: &mut SeededRng,
) -> NeatGenome {
    // Determine fitter parent (tie-break: shorter genome)
    let (fitter, other) = if fitness_a > fitness_b
        || ((fitness_a - fitness_b).abs() < f32::EPSILON
            && parent_a.connections.len() <= parent_b.connections.len())
    {
        (parent_a, parent_b)
    } else {
        (parent_b, parent_a)
    };

    // Build connection-by-innovation lookup for the other parent
    let other_by_innovation: std::collections::HashMap<u64, &ConnectionGene> = other
        .connections
        .iter()
        .map(|c| (c.innovation.0, c))
        .collect();

    let mut child_connections: Vec<ConnectionGene> = Vec::new();

    for fitter_conn in &fitter.connections {
        if let Some(other_conn) = other_by_innovation.get(&fitter_conn.innovation.0) {
            // Matching gene: 50/50 pick
            let chosen = if rng.gen_bool(0.5) {
                fitter_conn
            } else {
                other_conn
            };
            let mut conn = chosen.clone();

            // D18: if disabled in either parent, 75% chance disabled in child
            if !fitter_conn.enabled || !other_conn.enabled {
                conn.enabled = !rng.gen_bool(0.75);
            }

            child_connections.push(conn);
        } else {
            // Disjoint/excess from fitter parent: always inherit
            child_connections.push(fitter_conn.clone());
        }
    }

    // Child nodes: start with all nodes from fitter parent
    let mut child_node_ids: std::collections::HashSet<NodeId> =
        fitter.nodes.iter().map(|n| n.id).collect();

    // Add any nodes from other parent that are referenced by inherited connections
    let other_node_map: std::collections::HashMap<NodeId, &NodeGene> =
        other.nodes.iter().map(|n| (n.id, n)).collect();

    let mut child_nodes: Vec<NodeGene> = fitter.nodes.clone();

    for conn in &child_connections {
        for node_id in [conn.from, conn.to] {
            if !child_node_ids.contains(&node_id)
                && let Some(node) = other_node_map.get(&node_id)
            {
                child_nodes.push((*node).clone());
                child_node_ids.insert(node_id);
            }
        }
    }

    let mut child = NeatGenome::new(child_id, child_nodes, child_connections);
    child.sort_connections();

    #[cfg(debug_assertions)]
    child.validate();

    child
}

// ---------------------------------------------------------------------------
// Tournament selection (Step 2.5)
// ---------------------------------------------------------------------------

/// Select a genome via tournament selection.
///
/// Pick `k` random members from the provided slice, return the one with highest fitness.
/// If the slice has only one element, return it.
pub(crate) fn tournament_select(
    members: &[(GenomeId, f32)],
    k: usize,
    rng: &mut SeededRng,
) -> GenomeId {
    let k = k.min(members.len()).max(1);
    let mut best_idx = rng.gen_range(0..members.len());
    let mut best_fitness = members[best_idx].1;

    for _ in 1..k {
        let idx = rng.gen_range(0..members.len());
        if members[idx].1 > best_fitness {
            best_idx = idx;
            best_fitness = members[idx].1;
        }
    }

    members[best_idx].0
}

/// Apply all mutations to a genome based on config rates.
pub(crate) fn mutate(
    genome: &mut NeatGenome,
    config: &SimConfig,
    rng: &mut SeededRng,
    innovations: &mut InnovationCounter,
) {
    mutate_weights(genome, config, rng);

    if rng.gen_bool(config.neat.add_connection_rate) {
        mutate_add_connection(genome, rng, innovations);
    }

    if rng.gen_bool(config.neat.add_node_rate) {
        mutate_add_node(genome, rng, innovations);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::innovation::InnovationCounter;

    #[expect(clippy::panic, reason = "test helper")]
    fn make_test_config() -> SimConfig {
        SimConfig::load(None).unwrap_or_else(|_| {
            // Fallback: load default.toml embedded in the binary
            let toml_str = include_str!("../../config/default.toml");
            toml::from_str(toml_str)
                .unwrap_or_else(|e| panic!("Failed to parse default config: {e}"))
        })
    }

    fn make_minimal_genome(rng: &mut SeededRng, innovations: &mut InnovationCounter) -> NeatGenome {
        // 4 inputs + 1 bias + 3 outputs = 8 nodes (IDs 0-7)
        NeatGenome::create_minimal(GenomeId(1), 4, 3, rng, innovations)
    }

    fn test_innovations() -> InnovationCounter {
        // 4 inputs + 1 bias + 3 outputs = 8 nodes, so start at 8
        InnovationCounter::new(0, 8)
    }

    #[test]
    fn mutate_weights_changes_at_least_one() {
        let mut rng = SeededRng::new(42);
        let mut innovations = test_innovations();
        let config = make_test_config();
        let mut genome = make_minimal_genome(&mut rng, &mut innovations);

        // Ensure we have some connections
        if genome.connections.is_empty() {
            // Force at least one connection
            let from = NodeId(0);
            let to = NodeId(5); // First output (4 inputs + 1 bias)
            let inn = innovations.get_connection_innovation(from, to);
            genome.connections.push(ConnectionGene {
                from,
                to,
                weight: 0.5,
                enabled: true,
                innovation: inn,
            });
            genome.sort_connections();
        }

        let original_weights: Vec<f32> = genome.connections.iter().map(|c| c.weight).collect();
        mutate_weights(&mut genome, &config, &mut rng);
        let new_weights: Vec<f32> = genome.connections.iter().map(|c| c.weight).collect();

        // With 80% mutation rate, very unlikely all weights stay the same
        assert!(
            original_weights != new_weights,
            "Expected at least one weight to change after mutation"
        );
    }

    #[test]
    fn mutate_weights_respects_clamp() {
        let mut rng = SeededRng::new(42);
        let mut innovations = test_innovations();
        let config = make_test_config();
        let mut genome = make_minimal_genome(&mut rng, &mut innovations);

        // Run many mutations
        for _ in 0..100 {
            mutate_weights(&mut genome, &config, &mut rng);
        }

        for conn in &genome.connections {
            assert!(
                conn.weight >= -8.0 && conn.weight <= 8.0,
                "Weight {} out of clamp range",
                conn.weight
            );
        }
        for node in &genome.nodes {
            assert!(
                node.bias >= -8.0 && node.bias <= 8.0,
                "Bias {} out of clamp range",
                node.bias
            );
        }
    }

    #[test]
    fn add_connection_increases_count() {
        let mut rng = SeededRng::new(42);
        let mut innovations = test_innovations();
        let mut genome = make_minimal_genome(&mut rng, &mut innovations);

        let original_count = genome.connections.len();
        // Try many times since it can fail to find valid pairs
        let mut added = false;
        for _ in 0..10 {
            if mutate_add_connection(&mut genome, &mut rng, &mut innovations) {
                added = true;
                break;
            }
        }

        if added {
            assert!(
                genome.connections.len() > original_count,
                "Expected more connections after add_connection"
            );
        }

        #[cfg(debug_assertions)]
        genome.validate();
    }

    #[test]
    fn add_node_splits_connection() {
        let mut rng = SeededRng::new(42);
        let mut innovations = test_innovations();
        let mut genome = make_minimal_genome(&mut rng, &mut innovations);

        // Ensure at least one connection
        if genome.connections.is_empty() {
            mutate_add_connection(&mut genome, &mut rng, &mut innovations);
        }

        if genome.connections.iter().any(|c| c.enabled) {
            let original_node_count = genome.nodes.len();
            let result = mutate_add_node(&mut genome, &mut rng, &mut innovations);
            assert!(result, "add_node should succeed with enabled connections");
            assert!(
                genome.nodes.len() > original_node_count,
                "Expected new hidden node"
            );

            // Check one connection was disabled
            assert!(
                genome.connections.iter().any(|c| !c.enabled),
                "Expected at least one disabled connection after node split"
            );

            #[cfg(debug_assertions)]
            genome.validate();
        }
    }

    #[test]
    fn crossover_produces_valid_child() {
        let mut rng = SeededRng::new(42);
        let mut innovations = test_innovations();
        let parent_a = make_minimal_genome(&mut rng, &mut innovations);
        let parent_b = NeatGenome::create_minimal(GenomeId(2), 4, 3, &mut rng, &mut innovations);

        let child = crossover(&parent_a, 1.0, &parent_b, 0.5, GenomeId(3), &mut rng);

        assert!(!child.nodes.is_empty(), "Child must have nodes");
        // Child should have at least as many nodes as the fitter parent (a)
        assert!(
            child.nodes.len() >= parent_a.nodes.len(),
            "Child should have at least fitter parent's nodes"
        );

        #[cfg(debug_assertions)]
        child.validate();
    }

    #[test]
    fn tournament_select_returns_fittest() {
        let mut rng = SeededRng::new(42);
        let members = vec![(GenomeId(1), 0.5), (GenomeId(2), 0.9), (GenomeId(3), 0.1)];

        // With k=members.len(), should always return the fittest
        let winner = tournament_select(&members, members.len(), &mut rng);
        assert_eq!(winner, GenomeId(2), "Should pick highest fitness");
    }

    #[test]
    fn tournament_select_single_member() {
        let mut rng = SeededRng::new(42);
        let members = vec![(GenomeId(7), 0.3)];

        let winner = tournament_select(&members, 3, &mut rng);
        assert_eq!(winner, GenomeId(7), "Single member must be selected");
    }
}
