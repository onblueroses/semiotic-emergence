use clap::Parser;
use std::path::PathBuf;

use predator_prey_evolution_communication::config::SimConfig;
use predator_prey_evolution_communication::run_simulation;
use predator_prey_evolution_communication::simulation::SimOptions;

#[derive(Parser)]
#[command(name = "predator-prey-evolution")]
#[command(about = "Evolutionary simulation with emergent prey communication")]
struct Cli {
    /// Path to TOML config file
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Random seed (overrides config file)
    #[arg(short, long)]
    seed: Option<u64>,

    /// Maximum generations to run
    #[arg(short, long)]
    generations: Option<u32>,

    /// Enable terminal visualization
    #[arg(short, long)]
    viz: bool,

    /// Export stats path
    #[arg(short, long)]
    output: Option<String>,

    /// Load from checkpoint
    #[arg(long)]
    checkpoint: Option<PathBuf>,
}

#[expect(clippy::print_stderr)]
fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let mut config = SimConfig::load(cli.config)?;

    let options = SimOptions {
        seed: cli.seed,
        max_generations: cli.generations,
    };

    run_simulation(&mut config, &options)?;
    Ok(())
}
