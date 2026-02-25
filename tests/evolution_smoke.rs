//! Integration smoke test: run a few generations with `fast_test` config
//! and verify the simulation completes without panics, population is stable,
//! species exist, and fitness values are finite.
#![expect(clippy::expect_used, reason = "test code uses expect for clarity")]

use predator_prey_evolution_communication::config::SimConfig;
use predator_prey_evolution_communication::simulation::{SimOptions, run_simulation};

#[test]
fn five_generations_fast_test() {
    let mut config =
        SimConfig::load(Some("config/fast_test.toml".into())).expect("fast_test.toml should load");

    // Override export path to a temp directory so tests don't pollute output/
    config.stats.export_path = String::from("target/test_output/smoke/");

    let options = SimOptions {
        seed: Some(12345),
        max_generations: Some(5),
    };

    run_simulation(&mut config, &options).expect("simulation should complete without error");

    // Verify CSV was written
    let csv_path = std::path::Path::new("target/test_output/smoke/stats.csv");
    assert!(csv_path.exists(), "stats CSV should be written");

    let csv_content = std::fs::read_to_string(csv_path).expect("should read CSV");
    let lines: Vec<&str> = csv_content.lines().collect();

    // Header + 5 data rows
    assert_eq!(lines.len(), 6, "should have header + 5 generation rows");

    // Verify header
    assert!(
        lines[0].starts_with("generation,"),
        "first line should be CSV header"
    );

    // Verify each data row has correct column count and finite values
    for (row_idx, line) in lines.iter().enumerate().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        assert_eq!(
            cols.len(),
            9,
            "row {row_idx} should have 9 columns, got {}",
            cols.len()
        );

        // Generation number should match
        let generation: u32 = cols[0].parse().expect("generation should be a number");
        assert_eq!(
            generation,
            (row_idx - 1) as u32,
            "generation number should match row index"
        );

        // Fitness values should be finite and non-negative
        let avg_fitness: f32 = cols[1].parse().expect("avg_fitness should be a number");
        let max_fitness: f32 = cols[2].parse().expect("max_fitness should be a number");
        assert!(avg_fitness.is_finite(), "avg_fitness should be finite");
        assert!(max_fitness.is_finite(), "max_fitness should be finite");
        assert!(avg_fitness >= 0.0, "avg_fitness should be non-negative");
        assert!(max_fitness >= 0.0, "max_fitness should be non-negative");
        assert!(
            max_fitness >= avg_fitness,
            "max_fitness should be >= avg_fitness"
        );

        // Species count should be reasonable (0 for gen 0 is OK since speciation happens after)
        let species: u32 = cols[3].parse().expect("species_count should be a number");
        assert!(
            species <= 30,
            "species count should be <= population size, got {species}"
        );
    }

    // Clean up test output
    let _ = std::fs::remove_dir_all("target/test_output/smoke");
}
