use crate::config::SimConfig;

/// Compute normalized fitness for a prey agent (D6).
///
/// Formula: `age/max_ticks * w_survival + sqrt(energy)/sqrt(max_energy) * w_energy + kin_bonus * w_kin`
///
/// Uses sqrt normalization for energy to create diminishing returns (a prey with
/// 100 energy isn't twice as fit as one with 50).
pub(crate) fn compute_normalized_fitness(
    age: u64,
    max_ticks: u64,
    energy: f32,
    max_energy: f32,
    kin_bonus: f32,
    config: &SimConfig,
) -> f32 {
    let evo = &config.evolution;

    let survival_component = if max_ticks > 0 {
        age as f32 / max_ticks as f32 * evo.fitness_survival_weight
    } else {
        0.0
    };

    let energy_component = if max_energy > 0.0 {
        energy.max(0.0).sqrt() / max_energy.sqrt() * evo.fitness_energy_weight
    } else {
        0.0
    };

    let kin_component = kin_bonus * evo.fitness_kin_bonus;

    survival_component + energy_component + kin_component
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::panic, reason = "test helper")]
    fn make_test_config() -> SimConfig {
        let toml_str = include_str!("../../config/default.toml");
        toml::from_str(toml_str).unwrap_or_else(|e| panic!("Failed to parse default config: {e}"))
    }

    #[test]
    fn zero_inputs_zero_fitness() {
        let config = make_test_config();
        let fitness = compute_normalized_fitness(0, 500, 0.0, 200.0, 0.0, &config);
        assert!(
            fitness.abs() < f32::EPSILON,
            "Zero age/energy/kin should give zero fitness, got {fitness}"
        );
    }

    #[test]
    fn max_survival_gives_survival_weight() {
        let config = make_test_config();
        let fitness = compute_normalized_fitness(500, 500, 0.0, 200.0, 0.0, &config);
        let expected = config.evolution.fitness_survival_weight;
        assert!(
            (fitness - expected).abs() < 0.01,
            "Max survival should give {expected}, got {fitness}"
        );
    }

    #[test]
    fn sqrt_normalization_diminishing_returns() {
        let config = make_test_config();
        let fitness_low = compute_normalized_fitness(0, 500, 50.0, 200.0, 0.0, &config);
        let fitness_mid = compute_normalized_fitness(0, 500, 100.0, 200.0, 0.0, &config);
        let fitness_high = compute_normalized_fitness(0, 500, 150.0, 200.0, 0.0, &config);

        // Diminishing returns with equal-width intervals: each additional 50 energy
        // should give less fitness benefit than the previous 50 (sqrt marginal rate is
        // strictly decreasing).
        let gain_low_mid = fitness_mid - fitness_low;
        let gain_mid_high = fitness_high - fitness_mid;
        assert!(
            gain_mid_high < gain_low_mid,
            "Sqrt should give diminishing returns: gain 50->100={gain_low_mid:.3} should > gain 100->150={gain_mid_high:.3}"
        );
    }

    #[test]
    fn negative_energy_clamped_to_zero() {
        let config = make_test_config();
        let fitness = compute_normalized_fitness(0, 500, -50.0, 200.0, 0.0, &config);
        assert!(
            fitness >= 0.0,
            "Negative energy should not produce negative fitness, got {fitness}"
        );
    }

    #[test]
    fn kin_bonus_adds_to_fitness() {
        let config = make_test_config();
        let no_kin = compute_normalized_fitness(100, 500, 50.0, 200.0, 0.0, &config);
        let with_kin = compute_normalized_fitness(100, 500, 50.0, 200.0, 1.0, &config);
        assert!(
            with_kin >= no_kin,
            "Kin bonus should increase fitness: no_kin={no_kin}, with_kin={with_kin}"
        );
    }
}
