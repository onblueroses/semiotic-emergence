# GPU port (JAX/Python)

JAX/Python reimplementation of [semiotic-emergence](../) targeting large-scale GPU runs (Vast.ai, Colab). Same neural architecture, same 12 metric instruments, population sizes up to 100k prey.

## Results

**7k population, 83,860 generations (seed 42, Vast.ai RTX 4090):** mutual_info was exactly zero across all 83k generations. Signals collapsed to a single stereotyped symbol late in the run (0.0 bits entropy). The gap from the Rust version: signal_cost=0.015 (vs 0.002 in Rust) creates persistent selection pressure against senders. Receivers gain no fitness benefit from attending to signals that carry no information. Both sides of the communication channel have flat incentives. Language cannot emerge when signaling is a net fitness loss.

Results: `remote/results-7k/7k-seed42/` (output.csv, trajectory.csv, input_mi.csv). See [FINDINGS.md](FINDINGS.md).

## Setup

```bash
# CPU (development/testing)
pip install -e ".[dev]"

# GPU (CUDA 12, Linux)
pip install -e ".[dev]"
pip install "jax[cuda12-local]"

# Or use the setup script on a fresh GPU VPS:
./setup-gpu.sh
```

## Usage

```bash
# Single run
python -m semgpu.main <seed> <generations> [flags]

# Batch mode (multiple seeds + divergence matrix)
python -m semgpu.main --batch <n_seeds> <generations> [flags]

# Via launch script (isolated run directory, background process)
./launch.sh <name> <seed> <generations> [flags]
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--pop` | 384 | Population size |
| `--grid` | 56 | Grid dimension |
| `--pred` | 3 | Number of kill zones |
| `--food` | 100 | Food count |
| `--ticks` | 500 | Ticks per generation |
| `--zone-radius` | 8.0 | Kill zone radius |
| `--zone-speed` | 0.5 | Zone movement probability |
| `--zone-drain` | 0.02 | Zone drain rate |
| `--signal-cost` | 0.002 | Energy cost per signal |
| `--signal-range` | auto | Signal reception range |
| `--signal-ticks` | 4 | Signal persistence |
| `--patch-ratio` | 0.5 | Fraction of cooperative food |
| `--kin-bonus` | 0.1 | Kin selection strength |
| `--no-signals` | false | Counterfactual: disable signaling |
| `--zone-coverage` | - | Auto-scale zones by area fraction |

## Output

- `output.csv` - 23 columns: fitness, MI, JSD, iconicity, entropy, etc.
- `trajectory.csv` - 47 columns: contingency matrix, per-symbol JSD, contrast
- `input_mi.csv` - 37 columns: MI between each input dimension and emitted symbol
- `divergence.csv` - NxN cross-population JSD matrix (batch mode only)

## Tests

```bash
python -m pytest tests/ -v
```
