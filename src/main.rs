use clap::Parser;
use std::path::PathBuf;

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

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let _cli = Cli::parse();
    println!("predator-prey-evolution-communication v0.1.0");
    println!("Simulation engine ready.");
    Ok(())
}
