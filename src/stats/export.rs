use crate::stats::collector::StatsCollector;
use std::io::Write;
use std::path::Path;

/// Export generation stats to CSV file.
///
/// Creates parent directories if needed. Header row + one row per generation.
/// If the path ends with a separator or is an existing directory, appends `stats.csv`.
pub(crate) fn export_csv(collector: &StatsCollector, path: &str) -> std::io::Result<()> {
    let path = Path::new(path);

    // Determine if path is a directory (trailing separator or existing dir)
    let is_dir_path = path.as_os_str().to_string_lossy().ends_with('/')
        || path.as_os_str().to_string_lossy().ends_with('\\')
        || path.is_dir();

    let file_path = if is_dir_path {
        std::fs::create_dir_all(path)?;
        path.join("stats.csv")
    } else {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        path.to_path_buf()
    };

    let mut file = std::fs::File::create(&file_path)?;

    // Header
    writeln!(
        file,
        "generation,avg_fitness,max_fitness,species_count,prey_alive,signal_count,mutual_information,topographic_similarity,iconicity"
    )?;

    // Data rows
    for stats in &collector.generations {
        writeln!(
            file,
            "{},{:.4},{:.4},{},{},{},{:.6},{:.6},{:.6}",
            stats.generation,
            stats.avg_fitness,
            stats.max_fitness,
            stats.species_count,
            stats.prey_alive_end,
            stats.signal_count,
            stats.mutual_information,
            stats.topographic_similarity,
            stats.iconicity,
        )?;
    }

    Ok(())
}
