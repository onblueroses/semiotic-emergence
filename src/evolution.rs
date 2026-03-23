use rand::Rng;

use crate::brain::{
    Brain, CROSSOVER_GROUPS, INPUTS, LINKED_GROUP, MAX_BASE_HIDDEN, MEMORY_OUTPUTS,
    MIN_BASE_HIDDEN, MOVEMENT_OUTPUTS, SEG_BASE_BIAS, SEG_BASE_GATE, SEG_BASE_MEM, SEG_BASE_MOVE,
    SEG_BASE_SIGNAL, SEG_GATE_BIAS, SEG_INPUT_BASE, SEG_MEM_BIAS, SEG_MOVE_BIAS, SEG_SIGNAL_BIAS,
    SIGNAL_OUTPUTS,
};
use crate::world::wrap_dist_sq;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Agent {
    pub brain: Brain,
    pub x: i32,
    pub y: i32,
    pub parent_indices: [Option<usize>; 2],
    pub grandparent_indices: [Option<usize>; 4],
}

/// Spatial deme defined by rectangular bounds on the toroidal grid.
#[derive(Clone, Debug)]
pub struct Deme {
    pub x_min: i32,
    pub x_max: i32,
    pub y_min: i32,
    pub y_max: i32,
}

/// Divide the grid into `divisions x divisions` rectangular demes.
#[allow(clippy::cast_possible_wrap)]
pub fn compute_demes(grid_size: i32, divisions: usize) -> Vec<Deme> {
    let d = divisions as i32;
    let cell_w = grid_size / d;
    let mut demes = Vec::with_capacity(divisions * divisions);
    for dy in 0..d {
        for dx in 0..d {
            demes.push(Deme {
                x_min: dx * cell_w,
                x_max: if dx == d - 1 {
                    grid_size
                } else {
                    (dx + 1) * cell_w
                },
                y_min: dy * cell_w,
                y_max: if dy == d - 1 {
                    grid_size
                } else {
                    (dy + 1) * cell_w
                },
            });
        }
    }
    demes
}

/// Return the deme index for a given position.
#[allow(clippy::cast_possible_wrap)]
pub fn assign_deme(x: i32, y: i32, grid_size: i32, divisions: usize) -> usize {
    let d = divisions as i32;
    let cell_w = grid_size / d;
    let dx = (x / cell_w).min(d - 1);
    let dy = (y / cell_w).min(d - 1);
    (dy * d + dx) as usize
}

const OFFSPRING_JITTER: i32 = 1;
const HIDDEN_SIZE_MUTATION_RATE: f32 = 0.05;
/// Mutate weights up to `hidden_size` + `MUTATION_HEADROOM` neurons.
/// Keeps dormant weights pre-seeded for when `hidden_size` grows.
const MUTATION_HEADROOM: usize = 4;

/// Spatial index for O(nearby) tournament selection instead of O(N).
/// Buckets scored-array indices into grid cells for fast radius queries.
struct SpatialBucket {
    cells: Vec<Vec<usize>>,
    cell_size: i32,
    cells_per_axis: i32,
    grid_size: i32,
}

impl SpatialBucket {
    fn new(population: &[Agent], scored: &[(usize, f32)], grid_size: i32, radius: f32) -> Self {
        // Cell size targets radius/3 so the search ring is ~7x7 cells (not the whole grid).
        // Must evenly divide grid_size.
        let target = (radius / 3.0).floor().max(1.0) as i32;
        let cell_size = (1..=target)
            .rev()
            .find(|&c| grid_size % c == 0)
            .unwrap_or(1);
        let cells_per_axis = grid_size / cell_size;
        let num_cells = (cells_per_axis * cells_per_axis) as usize;
        let mut cells = vec![Vec::new(); num_cells];
        for (scored_idx, &(pop_idx, _)) in scored.iter().enumerate() {
            let cx = population[pop_idx].x.rem_euclid(grid_size) / cell_size;
            let cy = population[pop_idx].y.rem_euclid(grid_size) / cell_size;
            let ci = (cy * cells_per_axis + cx) as usize;
            cells[ci].push(scored_idx);
        }
        Self {
            cells,
            cell_size,
            cells_per_axis,
            grid_size,
        }
    }

    #[inline]
    fn cells_radius(&self, radius: f32) -> i32 {
        (radius / self.cell_size as f32).ceil() as i32
    }

    #[inline]
    fn cell_idx(&self, cx: i32, cy: i32) -> usize {
        let wx = cx.rem_euclid(self.cells_per_axis);
        let wy = cy.rem_euclid(self.cells_per_axis);
        (wy * self.cells_per_axis + wx) as usize
    }
}

/// Relatedness between two agents based on shared ancestry.
/// Returns 0.5 for siblings (share a parent), 0.25 for cousins (share a grandparent).
#[cfg(test)]
pub fn relatedness(a: &Agent, b: &Agent) -> f32 {
    for &pa in &a.parent_indices {
        if let Some(pa) = pa {
            for &pb in &b.parent_indices {
                if pb == Some(pa) {
                    return 0.5;
                }
            }
        }
    }
    for &ga in &a.grandparent_indices {
        if let Some(ga) = ga {
            for &gb in &b.grandparent_indices {
                if gb == Some(ga) {
                    return 0.25;
                }
            }
        }
    }
    0.0
}

fn tournament_select(
    _population: &[Agent],
    scored: &[(usize, f32)],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> usize {
    let mut best_idx = rng.gen_range(0..scored.len());
    let mut best_fit = scored[best_idx].1;
    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..scored.len());
        if scored[idx].1 > best_fit {
            best_idx = idx;
            best_fit = scored[idx].1;
        }
    }
    scored[best_idx].0
}

#[allow(clippy::too_many_arguments)]
fn local_tournament_select(
    population: &[Agent],
    scored: &[(usize, f32)],
    bucket: &SpatialBucket,
    center_x: i32,
    center_y: i32,
    radius: f32,
    tournament_size: usize,
    grid_size: i32,
    rng: &mut impl Rng,
) -> Option<usize> {
    let radius_sq = radius * radius;
    let cells_r = bucket.cells_radius(radius);
    let cx = center_x.rem_euclid(bucket.grid_size) / bucket.cell_size;
    let cy = center_y.rem_euclid(bucket.grid_size) / bucket.cell_size;

    // Reservoir sampling over nearby cells only (O(nearby) instead of O(N))
    let mut reservoir = [0usize; 8];
    let ts = tournament_size.min(reservoir.len());
    let mut nearby_count = 0usize;

    for dy in -cells_r..=cells_r {
        for dx in -cells_r..=cells_r {
            let ci = bucket.cell_idx(cx + dx, cy + dy);
            for &scored_idx in &bucket.cells[ci] {
                let (pop_idx, _) = scored[scored_idx];
                let agent = &population[pop_idx];
                if wrap_dist_sq(agent.x, agent.y, center_x, center_y, grid_size) > radius_sq {
                    continue;
                }
                if nearby_count < ts {
                    reservoir[nearby_count] = scored_idx;
                } else {
                    let j = rng.gen_range(0..=nearby_count);
                    if j < ts {
                        reservoir[j] = scored_idx;
                    }
                }
                nearby_count += 1;
            }
        }
    }

    if nearby_count < 2 {
        return None;
    }

    let used = ts.min(nearby_count);
    let mut best_idx = reservoir[0];
    let mut best_fit = scored[best_idx].1;
    for &idx in &reservoir[1..used] {
        if scored[idx].1 > best_fit {
            best_idx = idx;
            best_fit = scored[idx].1;
        }
    }
    Some(scored[best_idx].0)
}

pub fn crossover(a: &Brain, b: &Brain, rng: &mut impl Rng) -> Brain {
    let mut weights = a.weights;
    // Linked group (0): base representation + base_hidden_size from same parent
    let linked_parent_is_b = rng.gen_bool(0.5);
    let (start, end) = CROSSOVER_GROUPS[LINKED_GROUP];
    if linked_parent_is_b {
        weights[start..end].copy_from_slice(&b.weights[start..end]);
    }
    // Independent output groups (1-4): each independently from either parent
    for &(start, end) in &CROSSOVER_GROUPS[1..] {
        if rng.gen_bool(0.5) {
            weights[start..end].copy_from_slice(&b.weights[start..end]);
        }
    }
    // base_hidden_size co-inherits with linked group
    let base_hidden_size = if linked_parent_is_b {
        b.base_hidden_size
    } else {
        a.base_hidden_size
    };
    Brain {
        weights,
        base_hidden_size,
    }
}

fn gaussian_noise(sigma: f32, rng: &mut impl Rng) -> f32 {
    // CLT: sum of 4 uniforms approximates Gaussian. No transcendentals.
    // Variance = 4 * (1/12) = 1/3, scale by sqrt(3) for unit variance.
    // Tails bounded at +/-3.46 sigma (vs infinite for Box-Muller) - fine for mutation.
    let sum: f32 = rng.gen::<f32>() + rng.gen::<f32>() + rng.gen::<f32>() + rng.gen::<f32>();
    (sum - 2.0) * 1.732_050_8 * sigma
}

/// Scoped Gaussian mutation. All weight pools scoped by `base_hidden_size`.
pub fn mutate(brain: &mut Brain, sigma: f32, rng: &mut impl Rng) {
    let bh_scope = (brain.base_hidden_size + MUTATION_HEADROOM).min(MAX_BASE_HIDDEN);
    let w = &mut brain.weights;

    // Input -> base hidden: scope by bh
    for i in 0..INPUTS {
        let row_start = SEG_INPUT_BASE + i * MAX_BASE_HIDDEN;
        for h in 0..bh_scope {
            w[row_start + h] += gaussian_noise(sigma, rng);
        }
    }

    // Base hidden biases: scope by bh
    for h in 0..bh_scope {
        w[SEG_BASE_BIAS + h] += gaussian_noise(sigma, rng);
    }

    // Base -> movement: scope rows by bh, all MOVEMENT_OUTPUTS cols
    for h in 0..bh_scope {
        for o in 0..MOVEMENT_OUTPUTS {
            w[SEG_BASE_MOVE + h * MOVEMENT_OUTPUTS + o] += gaussian_noise(sigma, rng);
        }
    }

    // Movement biases: always mutate all
    for o in 0..MOVEMENT_OUTPUTS {
        w[SEG_MOVE_BIAS + o] += gaussian_noise(sigma, rng);
    }

    // Base -> signal: scope rows by bh, all SIGNAL_OUTPUTS cols
    for h in 0..bh_scope {
        for o in 0..SIGNAL_OUTPUTS {
            w[SEG_BASE_SIGNAL + h * SIGNAL_OUTPUTS + o] += gaussian_noise(sigma, rng);
        }
    }

    // Signal biases: always mutate all
    for o in 0..SIGNAL_OUTPUTS {
        w[SEG_SIGNAL_BIAS + o] += gaussian_noise(sigma, rng);
    }

    // Base -> gate: scope by bh
    for h in 0..bh_scope {
        w[SEG_BASE_GATE + h] += gaussian_noise(sigma, rng);
    }

    // Gate bias: always mutate
    w[SEG_GATE_BIAS] += gaussian_noise(sigma, rng);

    // Base -> memory: scope rows by bh, all MEMORY_OUTPUTS cols
    for h in 0..bh_scope {
        for o in 0..MEMORY_OUTPUTS {
            w[SEG_BASE_MEM + h * MEMORY_OUTPUTS + o] += gaussian_noise(sigma, rng);
        }
    }

    // Memory biases: always mutate all
    for o in 0..MEMORY_OUTPUTS {
        w[SEG_MEM_BIAS + o] += gaussian_noise(sigma, rng);
    }
}

#[allow(clippy::cast_possible_wrap)]
pub fn mutate_hidden_size(brain: &mut Brain, rng: &mut impl Rng) {
    if rng.gen::<f32>() < HIDDEN_SIZE_MUTATION_RATE {
        let delta: i32 = if rng.gen_bool(0.5) { 1 } else { -1 };
        let new_size = (brain.base_hidden_size as i32 + delta)
            .clamp(MIN_BASE_HIDDEN as i32, MAX_BASE_HIDDEN as i32);
        brain.base_hidden_size = new_size as usize;
    }
}

#[allow(clippy::too_many_arguments)]
fn select_parent(
    population: &[Agent],
    scored: &[(usize, f32)],
    bucket: &SpatialBucket,
    sx: i32,
    sy: i32,
    tournament_size: usize,
    grid_size: i32,
    reproduction_radius: f32,
    fallback_radius: f32,
    rng: &mut impl Rng,
) -> usize {
    if let Some(idx) = local_tournament_select(
        population,
        scored,
        bucket,
        sx,
        sy,
        reproduction_radius,
        tournament_size,
        grid_size,
        rng,
    ) {
        return idx;
    }
    if let Some(idx) = local_tournament_select(
        population,
        scored,
        bucket,
        sx,
        sy,
        fallback_radius,
        tournament_size,
        grid_size,
        rng,
    ) {
        return idx;
    }
    tournament_select(population, scored, tournament_size, rng)
}

/// Spatial evolution:
/// - Sort by fitness descending
/// - Top `elite_count` agents keep brain AND position unchanged
/// - Bottom agents are replaced: their positions become offspring slots
/// - Offspring selected from nearby parents via local tournament
#[allow(clippy::too_many_arguments)]
pub fn evolve_spatial(
    population: &[Agent],
    scored: &mut [(usize, f32)],
    elite_count: usize,
    tournament_size: usize,
    sigma: f32,
    grid_size: i32,
    reproduction_radius: f32,
    fallback_radius: f32,
    deme_divisions: usize,
    rng: &mut impl Rng,
) -> Vec<Agent> {
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let scored = &*scored; // reborrow as immutable - no more mutation needed

    let pop_size = scored.len();
    let mut next_gen: Vec<Agent> = Vec::with_capacity(pop_size);

    // Elites keep brain AND position AND lineage
    for &(pop_idx, _) in scored.iter().take(elite_count) {
        next_gen.push(population[pop_idx].clone());
    }

    // Pre-compute per-deme scored lists and spatial buckets when demes are active
    let deme_scored: Vec<Vec<(usize, f32)>> = if deme_divisions > 1 {
        let num_demes = deme_divisions * deme_divisions;
        let mut by_deme: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_demes];
        for &(pop_idx, fit) in scored {
            let d = assign_deme(
                population[pop_idx].x,
                population[pop_idx].y,
                grid_size,
                deme_divisions,
            );
            by_deme[d].push((pop_idx, fit));
        }
        by_deme
    } else {
        Vec::new()
    };

    // Build spatial buckets for O(nearby) tournament selection
    let global_bucket = SpatialBucket::new(population, scored, grid_size, reproduction_radius);
    let deme_buckets: Vec<SpatialBucket> = if deme_divisions > 1 {
        deme_scored
            .iter()
            .map(|ds| SpatialBucket::new(population, ds, grid_size, reproduction_radius))
            .collect()
    } else {
        Vec::new()
    };

    // Fill remaining slots at dead agents' positions
    for &(pop_idx, _) in scored.iter().skip(elite_count) {
        let sx = population[pop_idx].x;
        let sy = population[pop_idx].y;

        // Use deme-filtered scored list and bucket when demes are active
        let (effective_scored, effective_bucket) = if deme_divisions > 1 {
            let d = assign_deme(sx, sy, grid_size, deme_divisions);
            (&deme_scored[d] as &[(usize, f32)], &deme_buckets[d])
        } else {
            (scored as &[(usize, f32)], &global_bucket)
        };

        let pa = select_parent(
            population,
            effective_scored,
            effective_bucket,
            sx,
            sy,
            tournament_size,
            grid_size,
            reproduction_radius,
            fallback_radius,
            rng,
        );
        let pb = select_parent(
            population,
            effective_scored,
            effective_bucket,
            sx,
            sy,
            tournament_size,
            grid_size,
            reproduction_radius,
            fallback_radius,
            rng,
        );
        let mut child_brain = crossover(&population[pa].brain, &population[pb].brain, rng);
        mutate(&mut child_brain, sigma, rng);
        mutate_hidden_size(&mut child_brain, rng);

        let jx = rng.gen_range(-OFFSPRING_JITTER..=OFFSPRING_JITTER);
        let jy = rng.gen_range(-OFFSPRING_JITTER..=OFFSPRING_JITTER);

        next_gen.push(Agent {
            brain: child_brain,
            x: (sx + jx).rem_euclid(grid_size),
            y: (sy + jy).rem_euclid(grid_size),
            parent_indices: [Some(pa), Some(pb)],
            grandparent_indices: [
                population[pa].parent_indices[0],
                population[pa].parent_indices[1],
                population[pb].parent_indices[0],
                population[pb].parent_indices[1],
            ],
        });
    }

    next_gen
}

/// Group selection: bottom 1/3 demes by avg fitness lose their lowest 20% agents,
/// replaced by offspring from top 1/3 demes.
pub fn group_selection(
    population: &mut [Agent],
    fitness: &[f32],
    grid_size: i32,
    deme_divisions: usize,
    sigma: f32,
    rng: &mut impl Rng,
) {
    let demes = compute_demes(grid_size, deme_divisions);
    let num_demes = demes.len();

    // Compute per-deme average fitness
    let mut deme_sums = vec![0.0_f32; num_demes];
    let mut deme_counts = vec![0usize; num_demes];
    for (i, agent) in population.iter().enumerate() {
        let d = assign_deme(agent.x, agent.y, grid_size, deme_divisions);
        deme_sums[d] += fitness[i];
        deme_counts[d] += 1;
    }
    let mut deme_avg: Vec<(usize, f32)> = (0..num_demes)
        .map(|d| {
            let avg = if deme_counts[d] > 0 {
                deme_sums[d] / deme_counts[d] as f32
            } else {
                0.0
            };
            (d, avg)
        })
        .collect();
    deme_avg.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let bottom_count = (num_demes / 3).max(1);
    let top_count = (num_demes / 3).max(1);
    let bottom_demes: Vec<usize> = deme_avg[..bottom_count].iter().map(|&(d, _)| d).collect();
    let top_demes: Vec<usize> = deme_avg[num_demes - top_count..]
        .iter()
        .map(|&(d, _)| d)
        .collect();

    // Collect top-deme agents as donor pool
    let donors: Vec<usize> = population
        .iter()
        .enumerate()
        .filter(|(_, a)| top_demes.contains(&assign_deme(a.x, a.y, grid_size, deme_divisions)))
        .map(|(i, _)| i)
        .collect();

    if donors.is_empty() {
        return;
    }

    // For each bottom deme, replace lowest 20% with offspring from donors
    for &bottom_d in &bottom_demes {
        let mut members: Vec<(usize, f32)> = population
            .iter()
            .enumerate()
            .filter(|(_, a)| assign_deme(a.x, a.y, grid_size, deme_divisions) == bottom_d)
            .map(|(i, _)| (i, fitness[i]))
            .collect();
        members.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let replace_count = (members.len() / 5).max(1);
        for &(victim_idx, _) in members.iter().take(replace_count) {
            let da = donors[rng.gen_range(0..donors.len())];
            let db = donors[rng.gen_range(0..donors.len())];
            let mut child = crossover(&population[da].brain, &population[db].brain, rng);
            mutate(&mut child, sigma, rng);
            mutate_hidden_size(&mut child, rng);
            // Place at victim's position
            population[victim_idx].brain = child;
            population[victim_idx].parent_indices = [Some(da), Some(db)];
            population[victim_idx].grandparent_indices = [
                population[da].parent_indices[0],
                population[da].parent_indices[1],
                population[db].parent_indices[0],
                population[db].parent_indices[1],
            ];
        }
    }
}

/// Swap `migration_rate` fraction of agents between adjacent demes.
pub fn migrate(
    population: &mut [Agent],
    grid_size: i32,
    deme_divisions: usize,
    migration_rate: f32,
    rng: &mut impl Rng,
) {
    let demes = compute_demes(grid_size, deme_divisions);
    let num_demes = demes.len();
    let d = deme_divisions;

    // Build per-deme member lists
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); num_demes];
    for (i, agent) in population.iter().enumerate() {
        let di = assign_deme(agent.x, agent.y, grid_size, deme_divisions);
        members[di].push(i);
    }

    // Swap between horizontally and vertically adjacent demes
    for dy in 0..d {
        for dx in 0..d {
            let di = dy * d + dx;
            // Right neighbor (with wrap)
            let right = dy * d + (dx + 1) % d;
            if right != di {
                swap_agents(
                    population,
                    &members[di],
                    &members[right],
                    &demes[right],
                    migration_rate,
                    rng,
                );
            }
            // Down neighbor (with wrap)
            let down = ((dy + 1) % d) * d + dx;
            if down != di {
                swap_agents(
                    population,
                    &members[di],
                    &members[down],
                    &demes[down],
                    migration_rate,
                    rng,
                );
            }
        }
    }
}

fn swap_agents(
    population: &mut [Agent],
    from_members: &[usize],
    to_members: &[usize],
    to_deme: &Deme,
    migration_rate: f32,
    rng: &mut impl Rng,
) {
    let swap_count =
        ((from_members.len().min(to_members.len()) as f32 * migration_rate) as usize).max(0);
    for _ in 0..swap_count {
        if from_members.is_empty() || to_members.is_empty() {
            break;
        }
        let fi = from_members[rng.gen_range(0..from_members.len())];
        let ti = to_members[rng.gen_range(0..to_members.len())];
        // Swap brains (positions stay - agents adopt new neighborhood)
        let tmp_brain = population[fi].brain.clone();
        let tmp_parent = population[fi].parent_indices;
        let tmp_grand = population[fi].grandparent_indices;
        population[fi].brain = population[ti].brain.clone();
        population[fi].parent_indices = population[ti].parent_indices;
        population[fi].grandparent_indices = population[ti].grandparent_indices;
        population[ti].brain = tmp_brain;
        population[ti].parent_indices = tmp_parent;
        population[ti].grandparent_indices = tmp_grand;
        // Move migrant to random position in target deme
        population[ti].x = rng.gen_range(to_deme.x_min..to_deme.x_max);
        population[ti].y = rng.gen_range(to_deme.y_min..to_deme.y_max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::MAX_GENOME_LEN;

    const TEST_GRID: i32 = 20;
    const TEST_REPRO_RADIUS: f32 = 6.0;
    const TEST_FALLBACK_RADIUS: f32 = 10.0;

    fn test_agent(rng: &mut impl Rng, x: i32, y: i32) -> Agent {
        Agent {
            brain: Brain::random(rng),
            x,
            y,
            parent_indices: [None, None],
            grandparent_indices: [None; 4],
        }
    }

    #[test]
    fn crossover_preserves_length() {
        let mut rng = rand::thread_rng();
        let a = Brain::random(&mut rng);
        let b = Brain::random(&mut rng);
        let child = crossover(&a, &b, &mut rng);
        assert_eq!(child.weights.len(), MAX_GENOME_LEN);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn crossover_inherits_hidden_size() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut a = Brain::random(&mut rng);
        let mut b = Brain::random(&mut rng);
        a.base_hidden_size = 8;
        b.base_hidden_size = 20;

        let mut got_parent_a = false;
        let mut got_parent_b = false;
        for _ in 0..100 {
            let child = crossover(&a, &b, &mut rng);
            if child.base_hidden_size == 8 {
                got_parent_a = true;
            }
            if child.base_hidden_size == 20 {
                got_parent_b = true;
            }
        }
        assert!(
            got_parent_a && got_parent_b,
            "Should inherit base_hidden_size from both parents"
        );
    }

    #[test]
    fn crossover_selects_whole_groups() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(77);

        let mut a = Brain::zero();
        let mut b = Brain::zero();
        // Fill parent A with 1.0, parent B with 2.0
        for w in a.weights.iter_mut() {
            *w = 1.0;
        }
        for w in b.weights.iter_mut() {
            *w = 2.0;
        }

        let child = crossover(&a, &b, &mut rng);
        // Each crossover group must be entirely from one parent (all 1.0 or all 2.0)
        for &(start, end) in &CROSSOVER_GROUPS {
            let first = child.weights[start];
            assert!(
                first == 1.0 || first == 2.0,
                "Group [{start},{end}) has unexpected value {first}"
            );
            for &w in &child.weights[start..end] {
                assert_eq!(w, first, "Group [{start},{end}) has mixed parents");
            }
        }
    }

    #[test]
    fn crossover_linked_group_coherence() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut a = Brain::zero();
        let mut b = Brain::zero();
        a.base_hidden_size = 8;
        b.base_hidden_size = 30;
        for w in a.weights.iter_mut() {
            *w = 1.0;
        }
        for w in b.weights.iter_mut() {
            *w = 2.0;
        }

        // Over many trials, the linked group (input_base + base_bias) and
        // base_hidden_size must always come from the same parent.
        for _ in 0..200 {
            let child = crossover(&a, &b, &mut rng);
            let (lg_start, lg_end) = CROSSOVER_GROUPS[LINKED_GROUP];
            let linked_val = child.weights[lg_start];
            if linked_val == 1.0 {
                assert_eq!(
                    child.base_hidden_size, 8,
                    "Linked group from A but hidden size from B"
                );
            } else {
                assert_eq!(
                    child.base_hidden_size, 30,
                    "Linked group from B but hidden size from A"
                );
            }
            // All weights in the linked group must be from the same parent
            for &w in &child.weights[lg_start..lg_end] {
                assert_eq!(w, linked_val, "Linked group has mixed parents");
            }
        }
    }

    #[test]
    fn mutate_hidden_size_stays_in_bounds() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(99);

        let mut brain = Brain::zero();
        brain.base_hidden_size = MIN_BASE_HIDDEN;
        for _ in 0..1000 {
            mutate_hidden_size(&mut brain, &mut rng);
            assert!(brain.base_hidden_size >= MIN_BASE_HIDDEN);
            assert!(brain.base_hidden_size <= MAX_BASE_HIDDEN);
        }

        brain.base_hidden_size = MAX_BASE_HIDDEN;
        for _ in 0..1000 {
            mutate_hidden_size(&mut brain, &mut rng);
            assert!(brain.base_hidden_size >= MIN_BASE_HIDDEN);
            assert!(brain.base_hidden_size <= MAX_BASE_HIDDEN);
        }
    }

    #[test]
    fn evolve_spatial_preserves_population_size() {
        let mut rng = rand::thread_rng();
        let coords: Vec<(i32, i32)> = (0..20)
            .map(|_| (rng.gen_range(0..TEST_GRID), rng.gen_range(0..TEST_GRID)))
            .collect();
        let population: Vec<Agent> = coords
            .into_iter()
            .map(|(x, y)| test_agent(&mut rng, x, y))
            .collect();
        let mut scored: Vec<(usize, f32)> = (0..20).map(|i| (i, i as f32)).collect();
        let next = evolve_spatial(
            &population,
            &mut scored,
            4,
            3,
            0.1,
            TEST_GRID,
            TEST_REPRO_RADIUS,
            TEST_FALLBACK_RADIUS,
            1,
            &mut rng,
        );
        assert_eq!(next.len(), 20);
    }

    #[test]
    fn evolve_spatial_elites_keep_positions() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let population: Vec<Agent> = (0..20)
            .map(|i| Agent {
                brain: Brain::random(&mut rng),
                x: i as i32,
                y: i as i32 + 1,
                parent_indices: [None, None],
                grandparent_indices: [None; 4],
            })
            .collect();
        let mut scored: Vec<(usize, f32)> = (0..20).map(|i| (i, (20 - i) as f32)).collect();
        let elite_count = 4;
        // Save elite positions before evolve
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let elite_positions: Vec<(i32, i32)> = scored
            .iter()
            .take(elite_count)
            .map(|&(idx, _)| (population[idx].x, population[idx].y))
            .collect();
        // Re-scramble so evolve_spatial does its own sort
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let next = evolve_spatial(
            &population,
            &mut scored,
            elite_count,
            3,
            0.1,
            TEST_GRID,
            TEST_REPRO_RADIUS,
            TEST_FALLBACK_RADIUS,
            1,
            &mut rng,
        );

        for (i, (ex, ey)) in elite_positions.iter().enumerate() {
            assert_eq!(next[i].x, *ex, "Elite {i} x mismatch");
            assert_eq!(next[i].y, *ey, "Elite {i} y mismatch");
        }
    }

    #[test]
    fn evolve_spatial_propagates_hidden_size() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let population: Vec<Agent> = (0..20)
            .map(|_| {
                let mut brain = Brain::random(&mut rng);
                brain.base_hidden_size = 10;
                Agent {
                    brain,
                    x: rng.gen_range(0..TEST_GRID),
                    y: rng.gen_range(0..TEST_GRID),
                    parent_indices: [None, None],
                    grandparent_indices: [None; 4],
                }
            })
            .collect();
        let mut scored: Vec<(usize, f32)> = (0..20).map(|i| (i, i as f32)).collect();

        let next = evolve_spatial(
            &population,
            &mut scored,
            4,
            3,
            0.1,
            TEST_GRID,
            TEST_REPRO_RADIUS,
            TEST_FALLBACK_RADIUS,
            1,
            &mut rng,
        );

        // Elites should keep exact hidden sizes
        for agent in next.iter().take(4) {
            assert_eq!(
                agent.brain.base_hidden_size, 10,
                "Elites should preserve base_hidden_size"
            );
        }
        // Non-elites: inherited from parents +/- mutation, within bounds
        for agent in &next {
            assert!(agent.brain.base_hidden_size >= MIN_BASE_HIDDEN);
            assert!(agent.brain.base_hidden_size <= MAX_BASE_HIDDEN);
        }
    }

    #[test]
    fn local_tournament_selects_nearby() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let population: Vec<Agent> = vec![
            Agent {
                brain: Brain::zero(),
                x: 0,
                y: 0,
                parent_indices: [None, None],
                grandparent_indices: [None; 4],
            },
            Agent {
                brain: Brain::zero(),
                x: 1,
                y: 0,
                parent_indices: [None, None],
                grandparent_indices: [None; 4],
            },
            Agent {
                brain: Brain::zero(),
                x: 15,
                y: 15,
                parent_indices: [None, None],
                grandparent_indices: [None; 4],
            },
        ];
        let scored: Vec<(usize, f32)> = vec![(0, 10.0), (1, 5.0), (2, 100.0)];
        let bucket = SpatialBucket::new(&population, &scored, TEST_GRID, 3.0);

        // Center at (0,0), radius 3.0 - should only see agents at (0,0) and (1,0)
        let result = local_tournament_select(
            &population,
            &scored,
            &bucket,
            0,
            0,
            3.0,
            5,
            TEST_GRID,
            &mut rng,
        );
        assert!(result.is_some());
        let selected_idx = result.unwrap();
        let selected = &population[selected_idx];
        // Should be one of the two nearby agents, never the far one at (15,15)
        assert!(
            (selected.x == 0 && selected.y == 0) || (selected.x == 1 && selected.y == 0),
            "Selected agent at ({}, {}) is not nearby",
            selected.x,
            selected.y
        );
    }

    #[test]
    fn offspring_get_lineage() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let population: Vec<Agent> = (0..10)
            .map(|i| Agent {
                brain: Brain::random(&mut rng),
                x: rng.gen_range(0..TEST_GRID),
                y: rng.gen_range(0..TEST_GRID),
                parent_indices: [Some(i + 100), Some(i + 200)],
                grandparent_indices: [None; 4],
            })
            .collect();
        let mut scored: Vec<(usize, f32)> = (0..10).map(|i| (i, i as f32)).collect();

        let next = evolve_spatial(
            &population,
            &mut scored,
            2,
            3,
            0.1,
            TEST_GRID,
            TEST_REPRO_RADIUS,
            TEST_FALLBACK_RADIUS,
            1,
            &mut rng,
        );

        // Non-elite offspring should have parent indices set
        for agent in next.iter().skip(2) {
            assert!(
                agent.parent_indices[0].is_some(),
                "Offspring should have parent_a"
            );
            assert!(
                agent.parent_indices[1].is_some(),
                "Offspring should have parent_b"
            );
        }
    }

    #[test]
    fn relatedness_siblings() {
        let a = Agent {
            brain: Brain::zero(),
            x: 0,
            y: 0,
            parent_indices: [Some(5), Some(8)],
            grandparent_indices: [None; 4],
        };
        let b = Agent {
            brain: Brain::zero(),
            x: 0,
            y: 0,
            parent_indices: [Some(5), Some(12)],
            grandparent_indices: [None; 4],
        };
        assert!((relatedness(&a, &b) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn relatedness_cousins() {
        let a = Agent {
            brain: Brain::zero(),
            x: 0,
            y: 0,
            parent_indices: [Some(10), Some(11)],
            grandparent_indices: [Some(5), None, None, None],
        };
        let b = Agent {
            brain: Brain::zero(),
            x: 0,
            y: 0,
            parent_indices: [Some(20), Some(21)],
            grandparent_indices: [None, Some(5), None, None],
        };
        assert!((relatedness(&a, &b) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn relatedness_unrelated() {
        let a = Agent {
            brain: Brain::zero(),
            x: 0,
            y: 0,
            parent_indices: [Some(1), Some(2)],
            grandparent_indices: [Some(10), Some(11), Some(12), Some(13)],
        };
        let b = Agent {
            brain: Brain::zero(),
            x: 0,
            y: 0,
            parent_indices: [Some(3), Some(4)],
            grandparent_indices: [Some(20), Some(21), Some(22), Some(23)],
        };
        assert!((relatedness(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn sparse_kin_matches_brute_force() {
        use std::collections::HashMap;

        // 10 agents: 0-1 siblings (share parent 100), 2-3 siblings (share parent 200),
        // 4-5 cousins of 0-1 (share grandparent 50), 6-9 unrelated
        let agents: Vec<Agent> = vec![
            Agent {
                brain: Brain::zero(),
                x: 0,
                y: 0,
                parent_indices: [Some(100), Some(101)],
                grandparent_indices: [Some(50), None, None, None],
            },
            Agent {
                brain: Brain::zero(),
                x: 0,
                y: 0,
                parent_indices: [Some(100), Some(102)],
                grandparent_indices: [Some(50), None, None, None],
            },
            Agent {
                brain: Brain::zero(),
                x: 0,
                y: 0,
                parent_indices: [Some(200), Some(201)],
                grandparent_indices: [None; 4],
            },
            Agent {
                brain: Brain::zero(),
                x: 0,
                y: 0,
                parent_indices: [Some(200), Some(202)],
                grandparent_indices: [None; 4],
            },
            Agent {
                brain: Brain::zero(),
                x: 0,
                y: 0,
                parent_indices: [Some(300), Some(301)],
                grandparent_indices: [Some(50), None, None, None],
            },
            Agent {
                brain: Brain::zero(),
                x: 0,
                y: 0,
                parent_indices: [Some(302), Some(303)],
                grandparent_indices: [None, Some(50), None, None],
            },
            Agent {
                brain: Brain::zero(),
                x: 0,
                y: 0,
                parent_indices: [Some(400), Some(401)],
                grandparent_indices: [None; 4],
            },
            Agent {
                brain: Brain::zero(),
                x: 0,
                y: 0,
                parent_indices: [Some(500), Some(501)],
                grandparent_indices: [None; 4],
            },
            Agent {
                brain: Brain::zero(),
                x: 0,
                y: 0,
                parent_indices: [Some(600), Some(601)],
                grandparent_indices: [None; 4],
            },
            Agent {
                brain: Brain::zero(),
                x: 0,
                y: 0,
                parent_indices: [Some(700), Some(701)],
                grandparent_indices: [None; 4],
            },
        ];
        let fitness: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let kin_bonus = 0.1_f32;
        let n = agents.len();

        // Brute force (O(N^2))
        let mut brute = fitness.clone();
        for i in 0..n {
            let mut kin_sum = 0.0_f32;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let r = relatedness(&agents[i], &agents[j]);
                if r > 0.0 && fitness[j] > 0.0 {
                    kin_sum += r;
                }
            }
            brute[i] += kin_bonus * kin_sum;
        }

        // Sparse (HashMap)
        let mut sparse = fitness.clone();
        let mut parent_map: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut grandparent_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, agent) in agents.iter().enumerate() {
            for &p in &agent.parent_indices {
                if let Some(p) = p {
                    parent_map.entry(p).or_default().push(i);
                }
            }
            for &g in &agent.grandparent_indices {
                if let Some(g) = g {
                    grandparent_map.entry(g).or_default().push(i);
                }
            }
        }
        for i in 0..n {
            let mut kin_sum = 0.0_f32;
            let mut seen_sibling = vec![false; n];
            for &p in &agents[i].parent_indices {
                if let Some(p) = p {
                    if let Some(siblings) = parent_map.get(&p) {
                        for &j in siblings {
                            if j != i && !seen_sibling[j] && fitness[j] > 0.0 {
                                kin_sum += 0.5;
                                seen_sibling[j] = true;
                            }
                        }
                    }
                }
            }
            let mut seen_cousin = vec![false; n];
            for &g in &agents[i].grandparent_indices {
                if let Some(g) = g {
                    if let Some(cousins) = grandparent_map.get(&g) {
                        for &j in cousins {
                            if j != i && !seen_sibling[j] && !seen_cousin[j] && fitness[j] > 0.0 {
                                kin_sum += 0.25;
                                seen_cousin[j] = true;
                            }
                        }
                    }
                }
            }
            sparse[i] += kin_bonus * kin_sum;
        }

        for (i, (&b, &s)) in brute.iter().zip(&sparse).enumerate() {
            assert!((b - s).abs() < 1e-6, "Agent {i}: brute={b}, sparse={s}");
        }
    }

    #[test]
    fn clt_gaussian_distribution() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let sigma = 0.1_f32;
        let n = 10_000;
        let mut sum = 0.0_f32;
        let mut sum_sq = 0.0_f32;
        for _ in 0..n {
            let v = gaussian_noise(sigma, &mut rng);
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n as f32;
        let variance = sum_sq / n as f32 - mean * mean;
        let std_dev = variance.sqrt();
        assert!(
            mean.abs() < 0.01,
            "CLT gaussian mean {mean} should be near 0"
        );
        assert!(
            (std_dev - sigma).abs() < 0.02,
            "CLT gaussian std {std_dev} should be near {sigma}"
        );
    }

    // --- Deme infrastructure ---

    #[test]
    fn compute_demes_produces_correct_count() {
        let demes = compute_demes(72, 3);
        assert_eq!(demes.len(), 9);
    }

    #[test]
    fn compute_demes_covers_grid() {
        let demes = compute_demes(72, 3);
        // Each deme should be 24 cells wide (72/3)
        assert_eq!(demes[0].x_min, 0);
        assert_eq!(demes[0].x_max, 24);
        assert_eq!(demes[2].x_min, 48);
        assert_eq!(demes[2].x_max, 72); // last column gets remainder
    }

    #[test]
    fn assign_deme_maps_correctly() {
        // 72x72 grid, 3 divisions -> 24-cell demes
        assert_eq!(assign_deme(0, 0, 72, 3), 0);
        assert_eq!(assign_deme(23, 0, 72, 3), 0);
        assert_eq!(assign_deme(24, 0, 72, 3), 1);
        assert_eq!(assign_deme(0, 24, 72, 3), 3);
        assert_eq!(assign_deme(71, 71, 72, 3), 8);
    }

    #[test]
    fn evolve_spatial_demes1_unchanged() {
        // With deme_divisions=1, behavior should be identical
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let coords: Vec<(i32, i32)> = (0..20)
            .map(|_| (rng.gen_range(0..TEST_GRID), rng.gen_range(0..TEST_GRID)))
            .collect();
        let population: Vec<Agent> = coords
            .into_iter()
            .map(|(x, y)| test_agent(&mut rng, x, y))
            .collect();
        let mut scored: Vec<(usize, f32)> = (0..20).map(|i| (i, i as f32)).collect();
        let next = evolve_spatial(
            &population,
            &mut scored,
            4,
            3,
            0.1,
            TEST_GRID,
            TEST_REPRO_RADIUS,
            TEST_FALLBACK_RADIUS,
            1,
            &mut rng,
        );
        assert_eq!(next.len(), 20);
    }

    #[test]
    fn evolve_spatial_with_demes_preserves_population() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        // Use a grid divisible by 2
        let grid = 20;
        let coords: Vec<(i32, i32)> = (0..20)
            .map(|_| (rng.gen_range(0..grid), rng.gen_range(0..grid)))
            .collect();
        let population: Vec<Agent> = coords
            .into_iter()
            .map(|(x, y)| test_agent(&mut rng, x, y))
            .collect();
        let mut scored: Vec<(usize, f32)> = (0..20).map(|i| (i, i as f32)).collect();
        let next = evolve_spatial(
            &population,
            &mut scored,
            4,
            3,
            0.1,
            grid,
            TEST_REPRO_RADIUS,
            TEST_FALLBACK_RADIUS,
            2, // 2x2 = 4 demes
            &mut rng,
        );
        assert_eq!(next.len(), 20);
    }

    #[test]
    fn group_selection_preserves_population_size() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let grid = 24; // divisible by 3
        let n = 27; // 3 per deme (3x3=9 demes)
        let coords: Vec<(i32, i32)> = (0..n)
            .map(|_| (rng.gen_range(0..grid), rng.gen_range(0..grid)))
            .collect();
        let mut population: Vec<Agent> = coords
            .into_iter()
            .map(|(x, y)| test_agent(&mut rng, x, y))
            .collect();
        let fitness: Vec<f32> = (0..n).map(|i| i as f32).collect();
        group_selection(&mut population, &fitness, grid, 3, 0.1, &mut rng);
        assert_eq!(population.len(), n);
    }

    #[test]
    fn migrate_preserves_population_size() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let grid = 20;
        let n = 20;
        let coords: Vec<(i32, i32)> = (0..n)
            .map(|_| (rng.gen_range(0..grid), rng.gen_range(0..grid)))
            .collect();
        let mut population: Vec<Agent> = coords
            .into_iter()
            .map(|(x, y)| test_agent(&mut rng, x, y))
            .collect();
        migrate(&mut population, grid, 2, 0.1, &mut rng);
        assert_eq!(population.len(), n);
    }
}
