use crate::agent::action::Action;
use crate::agent::predator::{Predator, PredatorKind, PredatorState};
use crate::agent::prey::Prey;
use crate::agent::sensor::{self, SensorReading};
use crate::brain::genome::{GenomeId, NeatGenome};
use crate::brain::network::NeatNetwork;
use crate::config::{SimConfig, SimError};
use crate::rng::SeededRng;
use crate::signal::message::{ActiveSignal, Symbol};
use crate::signal::propagation;
use crate::snapshot::{AgentSnapshot, PredatorSnapshot, SignalSnapshot, WorldSnapshot};
use crate::world::entity::{Direction, LineageId, Position, PredatorId, PreyId};
use crate::world::food::{self, Food};
use crate::world::terrain::{self, Terrain};

pub(crate) struct World {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) terrain: Vec<Terrain>,
    pub(crate) food: Vec<Option<Food>>,
    pub(crate) prey: Vec<Prey>,
    pub(crate) predators: Vec<Predator>,
    pub(crate) signals: Vec<ActiveSignal>,
    pub(crate) tick: u64,
    pub(crate) generation: u32,
    pub(crate) rng: SeededRng,
    pub(crate) next_prey_id: u32,
    pub(crate) next_predator_id: u32,
    /// Genomes+fitness collected from prey that died during the generation
    pub(crate) dead_genomes: Vec<(NeatGenome, f32)>,
}

/// Result of running one generation.
pub(crate) struct GenerationResult {
    pub(crate) genomes_with_fitness: Vec<(NeatGenome, f32)>,
    pub(crate) ticks_elapsed: u64,
    pub(crate) prey_alive_end: u32,
}

// ---------------------------------------------------------------------------
// Construction and initialization
// ---------------------------------------------------------------------------

impl World {
    /// Simple constructor for tests - flat Open terrain, no food/agents.
    #[expect(dead_code, reason = "used by future unit tests")]
    pub(crate) fn new(width: u32, height: u32, seed: u64) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            terrain: vec![Terrain::Open; size],
            food: vec![None; size],
            prey: Vec::new(),
            predators: Vec::new(),
            signals: Vec::new(),
            tick: 0,
            generation: 0,
            rng: SeededRng::new(seed),
            next_prey_id: 0,
            next_predator_id: 0,
            dead_genomes: Vec::new(),
        }
    }

    /// Full initialization: generate terrain, place food, spawn predators.
    /// Does NOT spawn prey (needs genomes from generation lifecycle).
    pub(crate) fn initialize(config: &SimConfig) -> Result<Self, SimError> {
        let w = config.world.width;
        let h = config.world.height;
        let mut rng = SeededRng::new(config.seed);

        let terrain_grid = terrain::generate_terrain(w, h, &config.world, &mut rng)?;
        let food_grid = food::place_food(
            &terrain_grid,
            config.world.food_density,
            config.world.food_energy,
            &mut rng,
        );

        let mut world = Self {
            width: w,
            height: h,
            terrain: terrain_grid,
            food: food_grid,
            prey: Vec::new(),
            predators: Vec::new(),
            signals: Vec::new(),
            tick: 0,
            generation: 0,
            rng,
            next_prey_id: 0,
            next_predator_id: 0,
            dead_genomes: Vec::new(),
        };

        world.spawn_predators(config);
        Ok(world)
    }

    pub(crate) fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    pub(crate) fn terrain_at(&self, x: u32, y: u32) -> Terrain {
        self.terrain[self.idx(x, y)]
    }

    #[expect(dead_code, reason = "used by future tick phases and tests")]
    pub(crate) fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height
    }

    fn random_passable_pos(&mut self) -> Position {
        loop {
            let x = self.rng.gen_range(0..self.width);
            let y = self.rng.gen_range(0..self.height);
            if self.terrain_at(x, y).is_passable() {
                return Position::new(x, y);
            }
        }
    }

    fn spawn_predators(&mut self, config: &SimConfig) {
        let kinds = [
            (PredatorKind::Aerial, config.predators.aerial_count),
            (PredatorKind::Ground, config.predators.ground_count),
            (PredatorKind::Pack, config.predators.pack_count),
        ];
        for (kind, count) in kinds {
            for _ in 0..count {
                let pos = self.random_passable_pos();
                let id = PredatorId(self.next_predator_id);
                self.next_predator_id += 1;
                self.predators.push(Predator {
                    id,
                    kind,
                    pos,
                    energy: 100.0,
                    state: PredatorState::Roaming,
                    target: None,
                    cooldown: 0,
                });
            }
        }
    }

    pub(crate) fn spawn_prey(&mut self, genomes: &[NeatGenome], config: &SimConfig) {
        for genome in genomes {
            let pos = self.random_passable_pos();
            let id = PreyId(self.next_prey_id);
            self.next_prey_id += 1;
            let brain = NeatNetwork::from_genome(genome);
            let lineage = LineageId(id.0);
            self.prey.push(Prey {
                id,
                pos,
                energy: config.prey.initial_energy,
                age: 0,
                genome: genome.clone(),
                brain,
                facing: Direction::North,
                last_action: Action::Idle,
                last_signal: None,
                lineage,
                parent_genome_id: None,
                generation_born: self.generation,
                offspring_count: 0,
                fitness_cache: None,
                ticks_since_signal: 0,
                is_climbing: false,
                is_hidden: false,
            });
        }
    }

    pub(crate) fn reset_for_generation(&mut self, config: &SimConfig) {
        self.food = food::place_food(
            &self.terrain,
            config.world.food_density,
            config.world.food_energy,
            &mut self.rng,
        );
        self.prey.clear();
        self.predators.clear();
        self.signals.clear();
        self.dead_genomes.clear();
        self.tick = 0;
        self.next_prey_id = 0;
        self.next_predator_id = 0;
        self.spawn_predators(config);
    }
}

// ---------------------------------------------------------------------------
// Tick lifecycle: 9 phases
// ---------------------------------------------------------------------------

impl World {
    /// Run one complete tick (all 9 phases).
    pub(crate) fn tick(&mut self, config: &SimConfig) {
        // Phase 1+2: Sensor + Brain -> decisions
        let decisions = self.sense_and_decide(config);

        // Phase 3: Signal emission
        self.emit_signals(&decisions, config);

        // Phase 4: Action resolution
        self.resolve_actions(&decisions, config);

        // Phase 5: Predator phase
        let kills = self.tick_predators(config);

        // Phase 6: Signal decay
        propagation::decay_signals(&mut self.signals, self.tick, &config.signal);

        // Phase 7: Food regrowth
        self.regrow_food(config);

        // Phase 8: Death phase
        self.death_phase(&kills, config);

        // Phase 9: Stats (placeholder)

        // Update tick counter and per-prey state
        self.tick += 1;
        for prey in &mut self.prey {
            prey.age += 1;
            prey.ticks_since_signal += 1;
        }
    }

    /// Phase 1+2: Sense environment and run brains for all prey.
    /// Returns (`prey_index`, action, `optional_signal`) for each prey.
    fn sense_and_decide(&mut self, config: &SimConfig) -> Vec<(usize, Action, Option<Symbol>)> {
        let vocab_size = config.signal.vocab_size;
        let n_inputs = sensor::input_count(vocab_size);
        let vision_range = config.prey.vision_range as f32;
        let hearing_range = config.prey.hearing_range;

        // Collect read-only state needed for sensors (avoids borrow conflict with &mut brain)
        let prey_positions: Vec<(PreyId, Position)> =
            self.prey.iter().map(|p| (p.id, p.pos)).collect();
        let pred_snapshot: Vec<(PredatorKind, Position)> =
            self.predators.iter().map(|p| (p.kind, p.pos)).collect();

        let ctx = SensorContext {
            all_prey: &prey_positions,
            predators: &pred_snapshot,
            signals: &self.signals,
            terrain: &self.terrain,
            food: &self.food,
            width: self.width,
            current_tick: self.tick,
            vision_range,
            hearing_range,
            signal_config: &config.signal,
            prey_config: &config.prey,
            n_inputs,
        };

        let mut decisions = Vec::with_capacity(self.prey.len());

        for i in 0..self.prey.len() {
            let reading = encode_sensors(&self.prey[i], &ctx);

            let outputs = self.prey[i].brain.activate(&reading.inputs);
            let (action, signal) = sensor::decode_outputs(&outputs, vocab_size);
            decisions.push((i, action, signal));
        }

        decisions
    }

    /// Phase 3: Emit signals for prey that decided to signal.
    fn emit_signals(&mut self, decisions: &[(usize, Action, Option<Symbol>)], config: &SimConfig) {
        for &(idx, _, ref signal_opt) in decisions {
            if let Some(symbol) = signal_opt {
                let prey = &mut self.prey[idx];
                prey.energy = (prey.energy - config.prey.signal_energy_cost).max(0.0);
                prey.last_signal = Some(*symbol);
                prey.ticks_since_signal = 0;
                let sig = propagation::create_signal(prey.id, prey.pos, *symbol, self.tick);
                self.signals.push(sig);
            }
        }
    }

    /// Phase 4: Resolve actions in shuffled random order.
    fn resolve_actions(
        &mut self,
        decisions: &[(usize, Action, Option<Symbol>)],
        config: &SimConfig,
    ) {
        // Shuffle processing order
        let mut order: Vec<usize> = (0..decisions.len()).collect();
        self.rng.shuffle(&mut order);

        // Collect offspring to add after the loop (avoids borrow conflict)
        let mut offspring: Vec<Prey> = Vec::new();

        let width = self.width;
        let height = self.height;

        for &decision_idx in &order {
            let (prey_idx, ref action, _) = decisions[decision_idx];

            match action {
                Action::Move(dir) => {
                    let px = self.prey[prey_idx].pos.x;
                    let py = self.prey[prey_idx].pos.y;
                    let nx = px as i32 + dir.dx();
                    let ny = py as i32 + dir.dy();
                    if nx >= 0 && ny >= 0 && (nx as u32) < width && (ny as u32) < height {
                        let ux = nx as u32;
                        let uy = ny as u32;
                        let ti = (uy * width + ux) as usize;
                        if self.terrain[ti].is_passable() {
                            self.prey[prey_idx].pos = Position::new(ux, uy);
                            self.prey[prey_idx].facing = *dir;
                        }
                    }
                    self.prey[prey_idx].energy -= config.prey.move_energy_cost;
                    self.prey[prey_idx].is_climbing = false;
                    self.prey[prey_idx].is_hidden = false;
                }
                Action::Eat => {
                    let px = self.prey[prey_idx].pos.x;
                    let py = self.prey[prey_idx].pos.y;
                    let food_idx = (py * width + px) as usize;
                    if let Some(food_item) = &mut self.food[food_idx]
                        && food_item.energy > 0.0
                    {
                        let gained = food_item.energy;
                        food_item.energy = 0.0;
                        food_item.regrow_timer = config.world.food_regrow_ticks;
                        self.prey[prey_idx].energy =
                            (self.prey[prey_idx].energy + gained).min(config.prey.max_energy);
                    }
                    self.prey[prey_idx].is_climbing = false;
                    self.prey[prey_idx].is_hidden = false;
                }
                Action::Reproduce => {
                    self.prey[prey_idx].is_climbing = false;
                    self.prey[prey_idx].is_hidden = false;
                    if self.prey[prey_idx].energy >= config.prey.reproduce_energy_threshold {
                        self.prey[prey_idx].energy -= config.prey.reproduce_energy_cost;
                        self.prey[prey_idx].offspring_count += 1;

                        let parent_pos = self.prey[prey_idx].pos;
                        let parent_lineage = self.prey[prey_idx].lineage;
                        let parent_genome_id = self.prey[prey_idx].genome.id;

                        let offspring_pos = find_adjacent_passable(
                            parent_pos,
                            &self.terrain,
                            width,
                            height,
                            &mut self.rng,
                        );

                        if let Some(pos) = offspring_pos {
                            let child_id = PreyId(self.next_prey_id);
                            self.next_prey_id += 1;
                            let mut child_genome = self.prey[prey_idx].genome.clone();
                            child_genome.id = GenomeId(u64::from(child_id.0));
                            for conn in &mut child_genome.connections {
                                conn.weight += self.rng.gen_range(-0.1_f32..0.1);
                            }
                            let brain = NeatNetwork::from_genome(&child_genome);
                            offspring.push(Prey {
                                id: child_id,
                                pos,
                                energy: config.prey.initial_energy * 0.5,
                                age: 0,
                                genome: child_genome,
                                brain,
                                facing: Direction::North,
                                last_action: Action::Idle,
                                last_signal: None,
                                lineage: parent_lineage,
                                parent_genome_id: Some(parent_genome_id),
                                generation_born: self.generation,
                                offspring_count: 0,
                                fitness_cache: None,
                                ticks_since_signal: 0,
                                is_climbing: false,
                                is_hidden: false,
                            });
                        }
                    }
                }
                Action::Climb => {
                    let px = self.prey[prey_idx].pos.x;
                    let py = self.prey[prey_idx].pos.y;
                    let ti = (py * width + px) as usize;
                    let cell_terrain = self.terrain[ti];
                    // Only trees can be climbed; rocks provide hiding via Hide action
                    self.prey[prey_idx].is_climbing = cell_terrain == Terrain::Tree;
                    self.prey[prey_idx].is_hidden = false;
                }
                Action::Hide => {
                    let px = self.prey[prey_idx].pos.x;
                    let py = self.prey[prey_idx].pos.y;
                    let ti = (py * width + px) as usize;
                    let terrain = self.terrain[ti];
                    self.prey[prey_idx].is_hidden =
                        matches!(terrain, Terrain::Tree | Terrain::Bush | Terrain::Rock);
                    self.prey[prey_idx].is_climbing = false;
                }
                Action::Idle | Action::Signal(_) => {
                    self.prey[prey_idx].is_climbing = false;
                    self.prey[prey_idx].is_hidden = false;
                }
            }

            self.prey[prey_idx].last_action = *action;
            self.prey[prey_idx].energy -= config.prey.energy_per_tick;
        }

        self.prey.extend(offspring);
    }

    /// Phase 7: Food regrowth.
    fn regrow_food(&mut self, config: &SimConfig) {
        // Separate RNG calls from food mutation to avoid borrow conflict
        let regrow_rolls: Vec<f32> = self
            .food
            .iter()
            .map(|food_opt| {
                if let Some(food_item) = food_opt
                    && food_item.energy <= 0.0
                    && food_item.regrow_timer == 0
                {
                    return 1.0; // Needs a roll
                }
                0.0 // No roll needed
            })
            .collect();

        let rolls: Vec<f32> = regrow_rolls
            .iter()
            .map(|&needs_roll| {
                if needs_roll > 0.5 {
                    self.rng.gen_f32()
                } else {
                    1.0
                }
            })
            .collect();

        for (i, food_opt) in self.food.iter_mut().enumerate() {
            if let Some(food_item) = food_opt
                && food_item.energy <= 0.0
            {
                if food_item.regrow_timer > 0 {
                    food_item.regrow_timer -= 1;
                } else if rolls[i] < 0.8 {
                    food_item.energy = config.world.food_energy;
                }
            }
        }
    }

    /// Phase 8: Remove dead prey (killed by predators or starvation).
    fn death_phase(&mut self, killed_by_predator: &[PreyId], config: &SimConfig) {
        // Mark dead prey indices (both predation and starvation)
        let mut dead_indices: Vec<usize> = Vec::new();

        for (i, prey) in self.prey.iter().enumerate() {
            if killed_by_predator.contains(&prey.id) || prey.energy <= 0.0 {
                dead_indices.push(i);
            }
        }

        // Process in reverse order for safe swap_remove
        dead_indices.sort_unstable();
        dead_indices.reverse();

        for &idx in &dead_indices {
            let prey = &self.prey[idx];
            let fitness = compute_fitness(prey, config);
            self.dead_genomes.push((prey.genome.clone(), fitness));
            self.prey.swap_remove(idx);
        }
    }
}

// ---------------------------------------------------------------------------
// Predator AI
// ---------------------------------------------------------------------------

impl World {
    /// Phase 5: Run all predator behavior trees, return kill list.
    fn tick_predators(&mut self, config: &SimConfig) -> Vec<PreyId> {
        let mut kills = Vec::new();

        // Collect prey positions for read-only access
        let prey_state: Vec<(PreyId, Position, bool, bool)> = self
            .prey
            .iter()
            .map(|p| (p.id, p.pos, p.is_climbing, p.is_hidden))
            .collect();

        // Process aerial and ground predators individually
        for pred in &mut self.predators {
            match pred.kind {
                PredatorKind::Aerial => {
                    if let Some(kill) = tick_aerial(
                        pred,
                        &prey_state,
                        &self.terrain,
                        self.width,
                        self.height,
                        config,
                        &mut self.rng,
                    ) {
                        kills.push(kill);
                    }
                }
                PredatorKind::Ground => {
                    if let Some(kill) = tick_ground(
                        pred,
                        &prey_state,
                        &self.terrain,
                        self.width,
                        self.height,
                        config,
                        &mut self.rng,
                    ) {
                        kills.push(kill);
                    }
                }
                PredatorKind::Pack => {
                    // Pack wolves handled separately below
                }
            }
        }

        // Pack predators: coordinate together
        let pack_kills = tick_pack(
            &mut self.predators,
            &prey_state,
            &self.terrain,
            self.width,
            self.height,
            config,
            &mut self.rng,
        );
        kills.extend(pack_kills);

        kills
    }
}

// ---------------------------------------------------------------------------
// Generation lifecycle
// ---------------------------------------------------------------------------

impl World {
    /// Run a full generation: tick loop until `generation_ticks` or `min_prey_alive`.
    pub(crate) fn run_generation(&mut self, config: &SimConfig) -> GenerationResult {
        let max_ticks = config.evolution.generation_ticks;
        let min_alive = config.evolution.min_prey_alive as usize;

        for _ in 0..max_ticks {
            if self.prey.len() < min_alive {
                break;
            }
            self.tick(config);
        }

        // Compute fitness for surviving prey
        for prey in &self.prey {
            let fitness = compute_fitness(prey, config);
            self.dead_genomes.push((prey.genome.clone(), fitness));
        }

        GenerationResult {
            genomes_with_fitness: std::mem::take(&mut self.dead_genomes),
            ticks_elapsed: self.tick,
            prey_alive_end: self.prey.len() as u32,
        }
    }

    /// Create a read-only snapshot of the current world state.
    #[expect(dead_code, reason = "consumed by terminal-viz feature gate")]
    pub(crate) fn snapshot(&self) -> WorldSnapshot {
        WorldSnapshot {
            tick: self.tick,
            generation: self.generation,
            width: self.width,
            height: self.height,
            terrain: self.terrain.clone(),
            food: self
                .food
                .iter()
                .map(|f| f.as_ref().is_some_and(|fi| fi.energy > 0.0))
                .collect(),
            prey: self
                .prey
                .iter()
                .map(|p| AgentSnapshot {
                    id: p.id.0,
                    x: p.pos.x,
                    y: p.pos.y,
                    energy: p.energy,
                    facing: p.facing,
                    last_action: p.last_action,
                    last_signal: p.last_signal,
                    is_climbing: p.is_climbing,
                    is_hidden: p.is_hidden,
                    lineage: p.lineage.0,
                })
                .collect(),
            predators: self
                .predators
                .iter()
                .map(|p| PredatorSnapshot {
                    id: p.id.0,
                    x: p.pos.x,
                    y: p.pos.y,
                    kind: p.kind,
                    state: p.state,
                })
                .collect(),
            signals: self
                .signals
                .iter()
                .map(|s| SignalSnapshot {
                    x: s.origin.x,
                    y: s.origin.y,
                    symbol: s.symbol.0,
                    strength: s.strength,
                })
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions: sensor encoding, predator AI, fitness
// ---------------------------------------------------------------------------

/// Shared read-only world state for sensor encoding (avoids too many function args).
struct SensorContext<'a> {
    all_prey: &'a [(PreyId, Position)],
    predators: &'a [(PredatorKind, Position)],
    signals: &'a [ActiveSignal],
    terrain: &'a [Terrain],
    food: &'a [Option<Food>],
    width: u32,
    current_tick: u64,
    vision_range: f32,
    hearing_range: u32,
    signal_config: &'a crate::config::SignalConfig,
    prey_config: &'a crate::config::PreyConfig,
    n_inputs: usize,
}

/// Encode sensor inputs for a single prey.
fn encode_sensors(prey: &Prey, ctx: &SensorContext<'_>) -> SensorReading {
    let mut reading = SensorReading::new(ctx.n_inputs);
    let pos = prey.pos;
    let vision_range = ctx.vision_range;
    let norm = |d: f32| -> f32 {
        if d >= f32::MAX {
            -1.0
        } else {
            d / vision_range
        }
    };

    encode_predator_sensors(pos, vision_range, ctx.predators, &norm, &mut reading);
    encode_environment_sensors(pos, vision_range, ctx, &norm, &mut reading);
    encode_social_sensors(prey, pos, vision_range, ctx, &norm, &mut reading);

    reading
}

/// Predator distance and direction sensors (9 inputs).
fn encode_predator_sensors(
    pos: Position,
    vision_range: f32,
    predators: &[(PredatorKind, Position)],
    norm: &dyn Fn(f32) -> f32,
    reading: &mut SensorReading,
) {
    let mut nearest_aerial = (f32::MAX, 0.0_f32, 0.0_f32);
    let mut nearest_ground = (f32::MAX, 0.0_f32, 0.0_f32);
    let mut nearest_pack = (f32::MAX, 0.0_f32, 0.0_f32);

    for &(kind, pred_pos) in predators {
        let dist = pos.distance_to(&pred_pos);
        if dist > vision_range {
            continue;
        }
        let (dx, dy) = pos.direction_to(&pred_pos);
        match kind {
            PredatorKind::Aerial if dist < nearest_aerial.0 => nearest_aerial = (dist, dx, dy),
            PredatorKind::Ground if dist < nearest_ground.0 => nearest_ground = (dist, dx, dy),
            PredatorKind::Pack if dist < nearest_pack.0 => nearest_pack = (dist, dx, dy),
            _ => {}
        }
    }

    reading.inputs[sensor::NEAREST_AERIAL_DIST] = norm(nearest_aerial.0);
    reading.inputs[sensor::NEAREST_AERIAL_DX] = nearest_aerial.1;
    reading.inputs[sensor::NEAREST_AERIAL_DY] = nearest_aerial.2;
    reading.inputs[sensor::NEAREST_GROUND_DIST] = norm(nearest_ground.0);
    reading.inputs[sensor::NEAREST_GROUND_DX] = nearest_ground.1;
    reading.inputs[sensor::NEAREST_GROUND_DY] = nearest_ground.2;
    reading.inputs[sensor::NEAREST_PACK_DIST] = norm(nearest_pack.0);
    reading.inputs[sensor::NEAREST_PACK_DX] = nearest_pack.1;
    reading.inputs[sensor::NEAREST_PACK_DY] = nearest_pack.2;
}

/// Terrain and food sensors (8 inputs).
fn encode_environment_sensors(
    pos: Position,
    vision_range: f32,
    ctx: &SensorContext<'_>,
    norm: &dyn Fn(f32) -> f32,
    reading: &mut SensorReading,
) {
    let width = ctx.width;
    let height = ctx.terrain.len() as u32 / width;
    let cell_idx = (pos.y * width + pos.x) as usize;
    let current_terrain = ctx.terrain.get(cell_idx).copied().unwrap_or(Terrain::Open);
    reading.inputs[sensor::ON_TREE] = if current_terrain == Terrain::Tree {
        1.0
    } else {
        0.0
    };
    reading.inputs[sensor::ON_ROCK] = if current_terrain == Terrain::Rock {
        1.0
    } else {
        0.0
    };

    let scan_range = vision_range as u32;
    let min_x = pos.x.saturating_sub(scan_range);
    let max_x = (pos.x + scan_range).min(width - 1);
    let min_y = pos.y.saturating_sub(scan_range);
    let max_y = (pos.y + scan_range).min(height - 1);

    let mut nearest_tree_dist = f32::MAX;
    let mut nearest_rock_dist = f32::MAX;
    let mut nearest_food_dist = f32::MAX;
    let mut nearest_food_dx = 0.0_f32;

    for sy in min_y..=max_y {
        for sx in min_x..=max_x {
            let ti = (sy * width + sx) as usize;
            let d = pos.distance_to(&Position::new(sx, sy));
            match ctx.terrain[ti] {
                Terrain::Tree if d < nearest_tree_dist => nearest_tree_dist = d,
                Terrain::Rock if d < nearest_rock_dist => nearest_rock_dist = d,
                _ => {}
            }
            if let Some(Some(f)) = ctx.food.get(ti)
                && f.energy > 0.0
                && d < nearest_food_dist
            {
                nearest_food_dist = d;
                let (dx, _) = pos.direction_to(&Position::new(sx, sy));
                nearest_food_dx = dx;
            }
        }
    }
    reading.inputs[sensor::NEAREST_TREE_DIST] = norm(nearest_tree_dist);
    reading.inputs[sensor::NEAREST_ROCK_DIST] = norm(nearest_rock_dist);
    reading.inputs[sensor::NEAREST_FOOD_DIST] = norm(nearest_food_dist);
    reading.inputs[sensor::NEAREST_FOOD_DX] = nearest_food_dx;
}

/// Prey density, signal, and internal state sensors.
fn encode_social_sensors(
    prey: &Prey,
    pos: Position,
    vision_range: f32,
    ctx: &SensorContext<'_>,
    norm: &dyn Fn(f32) -> f32,
    reading: &mut SensorReading,
) {
    let mut prey_nearby = 0_u32;
    let mut nearest_prey_dist = f32::MAX;
    let density_range = 5.0_f32;
    for &(other_id, other_pos) in ctx.all_prey {
        if other_id == prey.id {
            continue;
        }
        let d = pos.distance_to(&other_pos);
        if d <= density_range {
            prey_nearby += 1;
        }
        if d < nearest_prey_dist && d <= vision_range {
            nearest_prey_dist = d;
        }
    }
    reading.inputs[sensor::PREY_DENSITY] = (prey_nearby as f32 / 10.0).min(1.0);
    reading.inputs[sensor::NEAREST_PREY_DIST] = norm(nearest_prey_dist);

    // Signal inputs
    let received = propagation::receive_signals(
        ctx.signals,
        pos,
        ctx.hearing_range,
        ctx.current_tick,
        ctx.signal_config,
    );
    for rs in &received {
        let sym_idx = rs.symbol.0 as usize;
        let base = sensor::SIGNAL_INPUTS_START + sym_idx * 2;
        if base + 1 < ctx.n_inputs {
            reading.inputs[base] = rs.strength;
            reading.inputs[base + 1] = rs.direction_x;
        }
    }

    // Internal state
    let own_energy_idx = sensor::SIGNAL_INPUTS_START + ctx.signal_config.vocab_size as usize * 2;
    if own_energy_idx + 2 < ctx.n_inputs {
        reading.inputs[own_energy_idx] = prey.energy / ctx.prey_config.max_energy;
        reading.inputs[own_energy_idx + 1] = if prey.is_climbing || prey.is_hidden {
            1.0
        } else {
            0.0
        };
        reading.inputs[own_energy_idx + 2] = (prey.ticks_since_signal as f32 / 20.0).min(1.0);
    }
}

/// Aerial predator (eagle) behavior.
fn tick_aerial(
    predator: &mut Predator,
    prey: &[(PreyId, Position, bool, bool)],
    terrain: &[Terrain],
    width: u32,
    height: u32,
    config: &SimConfig,
    rng: &mut SeededRng,
) -> Option<PreyId> {
    if predator.cooldown > 0 {
        predator.cooldown -= 1;
        predator.state = PredatorState::Resting;
        return None;
    }

    let vision = config.predators.aerial_vision as f32;

    // Find nearest exposed prey (not climbing on tree)
    let mut best_target: Option<(PreyId, f32, Position)> = None;
    for &(id, pos, is_climbing, _) in prey {
        let dist = predator.pos.distance_to(&pos);
        if dist > vision {
            continue;
        }
        // Eagle defeated by tree cover
        let cell = (pos.y * width + pos.x) as usize;
        if is_climbing && terrain.get(cell).copied() == Some(Terrain::Tree) {
            continue;
        }
        if best_target.as_ref().is_none_or(|b| dist < b.1) {
            best_target = Some((id, dist, pos));
        }
    }

    if let Some((target_id, _, target_pos)) = best_target {
        predator.state = PredatorState::Attacking;
        predator.target = Some(target_id);

        // Move toward target
        let speed = config.predators.aerial_speed;
        move_toward(predator, target_pos, speed, terrain, width, height);

        // Check kill (use post-movement distance)
        let post_dist = predator.pos.distance_to(&target_pos);
        if post_dist <= config.predators.kill_radius as f32 {
            predator.cooldown = config.predators.attack_cooldown;
            predator.state = PredatorState::Resting;
            predator.target = None;
            return Some(target_id);
        }
    } else {
        // Patrol: random movement
        predator.state = PredatorState::Roaming;
        predator.target = None;
        let dir = rng.gen_range(0..4_u8);
        let (dx, dy) = dir_offsets(dir);
        try_move(predator, dx, dy, terrain, width, height);
    }

    None
}

/// Ground predator (snake) behavior.
fn tick_ground(
    predator: &mut Predator,
    prey: &[(PreyId, Position, bool, bool)],
    terrain: &[Terrain],
    width: u32,
    height: u32,
    config: &SimConfig,
    rng: &mut SeededRng,
) -> Option<PreyId> {
    if predator.cooldown > 0 && predator.state == PredatorState::Resting {
        predator.cooldown -= 1;
        return None;
    }

    let vision = config.predators.ground_vision as f32;

    match predator.state {
        PredatorState::Resting => {
            predator.state = PredatorState::Roaming;
            None
        }
        PredatorState::Roaming => {
            // Look for prey within vision, not protected by terrain
            for &(id, pos, is_climbing, is_hidden) in prey {
                let dist = predator.pos.distance_to(&pos);
                if dist > vision {
                    continue;
                }
                let cell = (pos.y * width + pos.x) as usize;
                let cell_terrain = terrain.get(cell).copied().unwrap_or(Terrain::Open);
                // Prey climbing a tree is visible but ground predator can't reach
                if is_climbing && cell_terrain == Terrain::Tree {
                    continue;
                }
                // Prey hidden behind rock is invisible
                if is_hidden && cell_terrain == Terrain::Rock {
                    continue;
                }
                predator.state = PredatorState::Stalking(id);
                predator.cooldown = 20; // Patience timer
                predator.target = Some(id);
                return None;
            }
            // Wander slowly
            let dir = rng.gen_range(0..4_u8);
            let (dx, dy) = dir_offsets(dir);
            try_move(predator, dx, dy, terrain, width, height);
            None
        }
        PredatorState::Stalking(target_id) => {
            // Check if target still visible and in range
            let target = prey.iter().find(|p| p.0 == target_id);
            if let Some(&(_, pos, is_climbing, is_hidden)) = target {
                let cell = (pos.y * width + pos.x) as usize;
                let cell_terrain = terrain.get(cell).copied().unwrap_or(Terrain::Open);
                // Lost target: prey climbed a tree or hid behind rock
                if (is_climbing && cell_terrain == Terrain::Tree)
                    || (is_hidden && cell_terrain == Terrain::Rock)
                {
                    predator.state = PredatorState::Roaming;
                    predator.target = None;
                    return None;
                }
                // Approach target (Bug 4 fix: snake now closes distance)
                move_toward(
                    predator,
                    pos,
                    config.predators.ground_speed,
                    terrain,
                    width,
                    height,
                );
                let dist = predator.pos.distance_to(&pos);
                if dist <= config.predators.kill_radius as f32 {
                    // Lunge and kill
                    predator.cooldown = config.predators.attack_cooldown;
                    predator.state = PredatorState::Resting;
                    predator.target = None;
                    return Some(target_id);
                }
            } else {
                // Target no longer exists
                predator.state = PredatorState::Roaming;
                predator.target = None;
                return None;
            }
            // Patience countdown
            if predator.cooldown > 0 {
                predator.cooldown -= 1;
            } else {
                predator.state = PredatorState::Roaming;
                predator.target = None;
            }
            None
        }
        PredatorState::Attacking => {
            predator.state = PredatorState::Roaming;
            predator.target = None;
            None
        }
    }
}

/// Pack predator (wolves) behavior. All pack wolves coordinate on a shared target.
fn tick_pack(
    predators: &mut [Predator],
    prey: &[(PreyId, Position, bool, bool)],
    terrain: &[Terrain],
    width: u32,
    height: u32,
    config: &SimConfig,
    rng: &mut SeededRng,
) -> Vec<PreyId> {
    let mut kills = Vec::new();
    let pack_vision = config.predators.pack_vision as f32;

    // Collect pack wolves
    let pack_indices: Vec<usize> = predators
        .iter()
        .enumerate()
        .filter(|(_, p)| p.kind == PredatorKind::Pack)
        .map(|(i, _)| i)
        .collect();

    if pack_indices.is_empty() || prey.is_empty() {
        return kills;
    }

    // Find pack center
    let mut cx = 0.0_f32;
    let mut cy = 0.0_f32;
    let count = pack_indices.len() as f32;
    for &i in &pack_indices {
        cx += predators[i].pos.x as f32;
        cy += predators[i].pos.y as f32;
    }
    cx /= count;
    cy /= count;
    let pack_center = Position::new(cx as u32, cy as u32);

    // Find shared target: nearest unprotected prey visible to any wolf
    let mut best_target: Option<(PreyId, f32, Position)> = None;
    for &i in &pack_indices {
        if predators[i].cooldown > 0 {
            continue;
        }
        for &(id, pos, is_climbing, is_hidden) in prey {
            let dist = predators[i].pos.distance_to(&pos);
            if dist > pack_vision {
                continue;
            }
            // Skip prey protected by terrain
            let cell = (pos.y * width + pos.x) as usize;
            let cell_terrain = terrain.get(cell).copied().unwrap_or(Terrain::Open);
            if is_climbing && cell_terrain == Terrain::Tree {
                continue;
            }
            if is_hidden && matches!(cell_terrain, Terrain::Rock | Terrain::Bush) {
                continue;
            }
            let center_dist = pack_center.distance_to(&pos);
            if best_target.as_ref().is_none_or(|b| center_dist < b.1) {
                best_target = Some((id, center_dist, pos));
            }
        }
    }

    // All wolves act on shared target
    if let Some((target_id, _, target_pos)) = best_target {
        for &i in &pack_indices {
            let wolf = &mut predators[i];
            if wolf.cooldown > 0 {
                wolf.cooldown -= 1;
                wolf.state = PredatorState::Resting;
                continue;
            }
            wolf.state = PredatorState::Attacking;
            wolf.target = Some(target_id);
            move_toward(
                wolf,
                target_pos,
                config.predators.pack_speed,
                terrain,
                width,
                height,
            );

            if wolf.pos.distance_to(&target_pos) <= config.predators.kill_radius as f32 {
                if !kills.contains(&target_id) {
                    kills.push(target_id);
                }
                wolf.cooldown = config.predators.attack_cooldown;
                wolf.state = PredatorState::Resting;
                wolf.target = None;
            }
        }
    } else {
        // No target: wander near pack center
        for &i in &pack_indices {
            let wolf = &mut predators[i];
            if wolf.cooldown > 0 {
                wolf.cooldown -= 1;
                wolf.state = PredatorState::Resting;
                continue;
            }
            wolf.state = PredatorState::Roaming;
            wolf.target = None;
            let dir = rng.gen_range(0..4_u8);
            let (dx, dy) = dir_offsets(dir);
            try_move(wolf, dx, dy, terrain, width, height);
        }
    }

    kills
}

/// Compute fitness for a prey (simplified, no kin bonus).
fn compute_fitness(prey: &Prey, config: &SimConfig) -> f32 {
    let evo = &config.evolution;
    prey.age as f32 * evo.fitness_survival_weight
        + prey.energy.max(0.0) * evo.fitness_energy_weight
        + prey.offspring_count as f32 * evo.fitness_offspring_weight
}

// ---------------------------------------------------------------------------
// Movement helpers
// ---------------------------------------------------------------------------

fn dir_offsets(dir: u8) -> (i32, i32) {
    match dir {
        0 => (0, -1), // North
        1 => (0, 1),  // South
        2 => (1, 0),  // East
        _ => (-1, 0), // West
    }
}

fn try_move(
    predator: &mut Predator,
    dx: i32,
    dy: i32,
    terrain: &[Terrain],
    width: u32,
    height: u32,
) {
    let nx = predator.pos.x as i32 + dx;
    let ny = predator.pos.y as i32 + dy;
    if nx >= 0 && ny >= 0 && (nx as u32) < width && (ny as u32) < height {
        let ti = (ny as u32 * width + nx as u32) as usize;
        let cell = terrain[ti];
        let passable = match predator.kind {
            // Aerial predators fly over everything except water
            PredatorKind::Aerial => cell != Terrain::Water,
            // Ground and pack predators cannot enter tree, rock, or water
            PredatorKind::Ground | PredatorKind::Pack => {
                !matches!(cell, Terrain::Tree | Terrain::Rock | Terrain::Water)
            }
        };
        if passable {
            predator.pos = Position::new(nx as u32, ny as u32);
        }
    }
}

fn move_toward(
    predator: &mut Predator,
    target: Position,
    speed: u32,
    terrain: &[Terrain],
    width: u32,
    height: u32,
) {
    for _ in 0..speed {
        let dx = (target.x as i32 - predator.pos.x as i32).signum();
        let dy = (target.y as i32 - predator.pos.y as i32).signum();
        if dx == 0 && dy == 0 {
            break;
        }
        // Prefer horizontal or vertical based on distance
        if dx.unsigned_abs() >= dy.unsigned_abs() {
            try_move(predator, dx, 0, terrain, width, height);
        } else {
            try_move(predator, 0, dy, terrain, width, height);
        }
    }
}

/// Find an adjacent passable cell for offspring spawning.
fn find_adjacent_passable(
    pos: Position,
    terrain: &[Terrain],
    width: u32,
    height: u32,
    rng: &mut SeededRng,
) -> Option<Position> {
    let mut candidates = Vec::new();
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let nx = pos.x as i32 + dx;
        let ny = pos.y as i32 + dy;
        if nx >= 0 && ny >= 0 && (nx as u32) < width && (ny as u32) < height {
            let idx = (ny as u32 * width + nx as u32) as usize;
            if terrain[idx].is_passable() {
                candidates.push(Position::new(nx as u32, ny as u32));
            }
        }
    }
    if candidates.is_empty() {
        None
    } else {
        let i = rng.gen_range(0..candidates.len());
        Some(candidates[i])
    }
}
