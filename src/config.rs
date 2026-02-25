use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
pub struct SimConfig {
    pub world: WorldConfig,
    pub prey: PreyConfig,
    pub predators: PredatorConfig,
    pub neat: NeatConfig,
    pub signal: SignalConfig,
    pub evolution: EvolutionConfig,
    pub stats: StatsConfig,
    pub seed: u64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct WorldConfig {
    pub width: u32,
    pub height: u32,
    pub food_density: f32,
    pub food_energy: f32,
    pub food_regrow_ticks: u32,
    pub terrain_tree_pct: f32,
    pub terrain_rock_pct: f32,
    pub terrain_water_pct: f32,
    pub terrain_bush_pct: f32,
}

#[derive(Deserialize, Clone, Debug)]
pub struct PreyConfig {
    pub initial_count: u32,
    pub initial_energy: f32,
    pub max_energy: f32,
    pub energy_per_tick: f32,
    pub move_energy_cost: f32,
    pub signal_energy_cost: f32,
    pub reproduce_energy_cost: f32,
    pub reproduce_energy_threshold: f32,
    pub vision_range: u32,
    pub vision_angle: f32,
    pub hearing_range: u32,
}

#[derive(Deserialize, Clone, Debug)]
pub struct PredatorConfig {
    pub aerial_count: u32,
    pub ground_count: u32,
    pub pack_count: u32,
    pub aerial_speed: u32,
    pub ground_speed: u32,
    pub pack_speed: u32,
    pub aerial_vision: u32,
    pub ground_vision: u32,
    pub pack_vision: u32,
    pub attack_cooldown: u32,
    pub kill_radius: u32,
}

#[derive(Deserialize, Clone, Debug)]
pub struct NeatConfig {
    pub population_size: u32,
    pub c1_excess: f32,
    pub c2_disjoint: f32,
    pub c3_weight: f32,
    pub compatibility_threshold: f32,
    pub weight_mutate_rate: f64,
    pub weight_perturb_rate: f64,
    pub weight_perturb_strength: f32,
    pub add_node_rate: f64,
    pub add_connection_rate: f64,
    pub disable_gene_rate: f64,
    pub interspecies_mate_rate: f64,
    pub stagnation_limit: u32,
    pub elitism_count: u32,
    pub survival_rate: f32,
}

#[derive(Deserialize, Clone, Debug)]
pub struct SignalConfig {
    pub vocab_size: u8,
    pub signal_range: u32,
    pub signal_decay_rate: f32,
    pub signal_lifetime: u32,
}

#[derive(Deserialize, Clone, Debug)]
pub struct EvolutionConfig {
    pub generation_ticks: u64,
    pub max_generations: u32,
    pub min_prey_alive: u32,
    pub fitness_survival_weight: f32,
    pub fitness_energy_weight: f32,
    pub fitness_offspring_weight: f32,
    pub fitness_kin_bonus: f32,
    pub kin_relatedness_generations: u32,
}

#[derive(Deserialize, Clone, Debug)]
pub struct StatsConfig {
    pub export_every_n_generations: u32,
    pub export_format: String,
    pub export_path: String,
    pub track_signals: bool,
    pub track_lineage: bool,
}

impl SimConfig {
    pub fn load(path: Option<std::path::PathBuf>) -> Result<Self, SimError> {
        let default_toml = include_str!("../config/default.toml");
        let mut config: SimConfig = toml::from_str(default_toml)
            .map_err(|e| SimError::Config(format!("Failed to parse default config: {e}")))?;

        if let Some(path) = path {
            let user_toml = std::fs::read_to_string(&path)
                .map_err(SimError::Io)?;
            let overrides: SimConfig = toml::from_str(&user_toml)
                .map_err(|e| SimError::Config(format!("Failed to parse {}: {e}", path.display())))?;
            config = overrides;
        }

        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), SimError> {
        if self.signal.vocab_size == 0 {
            return Err(SimError::Config("vocab_size must be > 0".into()));
        }
        if self.world.width == 0 || self.world.height == 0 {
            return Err(SimError::Config("world dimensions must be > 0".into()));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum SimError {
    Config(String),
    Io(std::io::Error),
    Checkpoint(String),
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SimError::Config(msg) => write!(f, "Configuration error: {msg}"),
            SimError::Io(e) => write!(f, "IO error: {e}"),
            SimError::Checkpoint(msg) => write!(f, "Checkpoint error: {msg}"),
        }
    }
}

impl std::error::Error for SimError {}

impl From<std::io::Error> for SimError {
    fn from(e: std::io::Error) -> Self {
        SimError::Io(e)
    }
}
