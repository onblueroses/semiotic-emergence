#!/usr/bin/env python3
"""Analyze semiotic-emergence simulation output.

Usage:
    python analyze.py output.csv                    # single run
    python analyze.py run1/output.csv run2/output.csv  # compare runs
    python analyze.py output.csv --trajectory trajectory.csv
    python analyze.py output.csv --input-mi input_mi.csv
    python analyze.py output.csv --all trajectory.csv input_mi.csv  # everything
"""

import csv
import argparse
from pathlib import Path
from dataclasses import dataclass


@dataclass
class RunStats:
    name: str
    gens: int
    # Fitness
    avg_fitness_final: float
    max_fitness_final: float
    avg_fitness_peak: float
    max_fitness_peak: float
    # Brain
    avg_hidden_final: float
    min_hidden_final: int
    max_hidden_final: int
    avg_hidden_peak: float
    brain_growth_rate: float  # gens to reach peak avg
    # Signals
    signals_final: int
    mi_final: float
    mi_peak: float
    mi_peak_gen: int
    mi_sustained: float  # avg MI over last 10% of run
    # Receiver
    jsd_pred_final: float
    jsd_pred_peak: float
    jsd_no_pred_final: float
    # Silence
    silence_corr_final: float
    silence_corr_min: float  # most negative = strongest silence
    # Fitness coupling
    sender_fit_final: float
    receiver_fit_final: float
    response_fit_final: float
    # Fixed metrics
    silence_onset_jsd_nonzero: int
    silence_move_delta_nonzero: int
    response_fit_nonzero: int
    # Phase transitions
    traj_fluct_max: float
    traj_fluct_max_gen: int


def read_output_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: try_float(v) for k, v in row.items()})
    return rows


def try_float(v):
    try:
        f = float(v)
        if f == int(f) and abs(f) < 1e9:
            return int(f)
        return f
    except (ValueError, TypeError):
        return v


def analyze_run(rows: list[dict], name: str = "") -> RunStats:
    n = len(rows)
    tail_start = max(0, int(n * 0.9))
    tail = rows[tail_start:]

    mi_peak_gen = max(range(n), key=lambda i: rows[i].get("mutual_info", 0))
    traj_peak_gen = max(range(n), key=lambda i: rows[i].get("traj_fluct_ratio", 0))

    # Find when avg_hidden peaked
    hidden_peak_gen = max(range(n), key=lambda i: rows[i].get("avg_hidden", 0))

    return RunStats(
        name=name or Path(rows[0].get("_path", "unknown")).stem,
        gens=n,
        avg_fitness_final=rows[-1].get("avg_fitness", 0),
        max_fitness_final=rows[-1].get("max_fitness", 0),
        avg_fitness_peak=max(r.get("avg_fitness", 0) for r in rows),
        max_fitness_peak=max(r.get("max_fitness", 0) for r in rows),
        avg_hidden_final=rows[-1].get("avg_hidden", 0),
        min_hidden_final=int(rows[-1].get("min_hidden", 0)),
        max_hidden_final=int(rows[-1].get("max_hidden", 0)),
        avg_hidden_peak=max(r.get("avg_hidden", 0) for r in rows),
        brain_growth_rate=hidden_peak_gen,
        signals_final=int(rows[-1].get("signals_emitted", 0)),
        mi_final=rows[-1].get("mutual_info", 0),
        mi_peak=rows[mi_peak_gen].get("mutual_info", 0),
        mi_peak_gen=int(rows[mi_peak_gen].get("generation", mi_peak_gen)),
        mi_sustained=avg([r.get("mutual_info", 0) for r in tail]),
        jsd_pred_final=rows[-1].get("jsd_pred", 0),
        jsd_pred_peak=max(r.get("jsd_pred", 0) for r in rows),
        jsd_no_pred_final=rows[-1].get("jsd_no_pred", 0),
        silence_corr_final=rows[-1].get("silence_corr", 0),
        silence_corr_min=min(r.get("silence_corr", 0) for r in rows),
        sender_fit_final=rows[-1].get("sender_fit_corr", 0),
        receiver_fit_final=rows[-1].get("receiver_fit_corr", 0),
        response_fit_final=rows[-1].get("response_fit_corr", 0),
        silence_onset_jsd_nonzero=sum(
            1 for r in rows if r.get("silence_onset_jsd", 0) != 0
        ),
        silence_move_delta_nonzero=sum(
            1 for r in rows if r.get("silence_move_delta", 0) != 0
        ),
        response_fit_nonzero=sum(
            1 for r in rows if r.get("response_fit_corr", 0) != 0
        ),
        traj_fluct_max=rows[traj_peak_gen].get("traj_fluct_ratio", 0),
        traj_fluct_max_gen=int(
            rows[traj_peak_gen].get("generation", traj_peak_gen)
        ),
    )


def avg(xs):
    return sum(xs) / len(xs) if xs else 0


def print_run_stats(s: RunStats):
    print(f"\n{'=' * 60}")
    print(f"  {s.name}  ({s.gens:,} generations)")
    print(f"{'=' * 60}")

    print("\n  FITNESS")
    print(f"    Final avg/max:  {s.avg_fitness_final:.1f} / {s.max_fitness_final:.1f}")
    print(f"    Peak avg/max:   {s.avg_fitness_peak:.1f} / {s.max_fitness_peak:.1f}")

    print("\n  BRAIN SIZE (Social Brain Hypothesis)")
    print(f"    Final avg:      {s.avg_hidden_final:.1f}  [{s.min_hidden_final}-{s.max_hidden_final}]")
    print(f"    Peak avg:       {s.avg_hidden_peak:.1f}  (at gen {s.brain_growth_rate:,})")
    drain_final = 0.0008 + s.avg_hidden_final * 0.00002
    print(f"    Energy drain:   {drain_final:.5f}/tick  ({drain_final * 500:.2f} over 500 ticks)")

    print("\n  MUTUAL INFORMATION (sender-world correlation)")
    print(f"    Final MI:       {s.mi_final:.4f}")
    print(f"    Peak MI:        {s.mi_peak:.4f}  (at gen {s.mi_peak_gen:,})")
    print(f"    Sustained MI:   {s.mi_sustained:.4f}  (avg last 10%)")

    print("\n  RECEIVER BEHAVIOR")
    print(f"    JSD (pred):     {s.jsd_pred_final:.4f}  (peak {s.jsd_pred_peak:.4f})")
    print(f"    JSD (no pred):  {s.jsd_no_pred_final:.4f}")
    print(f"    Silence corr:   {s.silence_corr_final:.4f}  (min {s.silence_corr_min:.4f})")

    print("\n  FITNESS COUPLING")
    print(f"    Sender-fitness: {s.sender_fit_final:.4f}")
    print(f"    Receiver-fit:   {s.receiver_fit_final:.4f}")
    print(f"    Response-fit:   {s.response_fit_final:.4f}")

    print(f"\n  METRIC HEALTH (non-zero counts / {s.gens:,} gens)")
    print(f"    response_fit:       {s.response_fit_nonzero:>6,}  ({100*s.response_fit_nonzero/s.gens:.1f}%)")
    print(f"    silence_onset_jsd:  {s.silence_onset_jsd_nonzero:>6,}  ({100*s.silence_onset_jsd_nonzero/s.gens:.1f}%)")
    print(f"    silence_move_delta: {s.silence_move_delta_nonzero:>6,}  ({100*s.silence_move_delta_nonzero/s.gens:.1f}%)")

    print("\n  PHASE TRANSITIONS")
    print(f"    Max fluct ratio: {s.traj_fluct_max:.4f}  (at gen {s.traj_fluct_max_gen:,})")


def analyze_trajectory(path: str):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: try_float(v) for k, v in row.items()})

    if not rows:
        print("\n  TRAJECTORY: empty file")
        return

    n = len(rows)
    print(f"\n{'=' * 60}")
    print(f"  TRAJECTORY ANALYSIS  ({n:,} generations)")
    print(f"{'=' * 60}")

    # Symbol differentiation over time
    # Check contrast columns - higher = more differentiated symbols
    contrasts = ["contrast_01", "contrast_02", "contrast_12"]
    tail_start = max(0, int(n * 0.9))
    tail = rows[tail_start:]

    print("\n  SYMBOL CONTRAST (pairwise JSD between symbol context distributions)")
    print("  Higher = symbols used in more different contexts = more differentiated")
    for c in contrasts:
        vals = [r.get(c, 0) for r in rows if r.get(c, 0) != 0]
        tail_vals = [r.get(c, 0) for r in tail if r.get(c, 0) != 0]
        if vals:
            print(f"    {c}: avg {avg(vals):.4f}, tail avg {avg(tail_vals):.4f}, max {max(vals):.4f}")

    # Phase transitions via trajectory_jsd spikes
    jsd_vals = [(int(r.get("generation", i)), r.get("trajectory_jsd", 0)) for i, r in enumerate(rows)]
    spikes = sorted(jsd_vals, key=lambda x: x[1], reverse=True)[:10]
    print("\n  TOP 10 TRAJECTORY JSD SPIKES (phase transition candidates)")
    for gen, jsd in spikes:
        if jsd > 0:
            print(f"    gen {gen:>7,}: {jsd:.4f}")

    # Symbol dominance over time
    print("\n  SYMBOL USAGE EVOLUTION (% of total emissions)")
    checkpoints = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    for idx in checkpoints:
        r = rows[idx]
        gen = int(r.get("generation", idx))
        totals = [
            sum(r.get(f"s{s}d{d}", 0) for d in range(4))
            for s in range(3)
        ]
        total = sum(totals) or 1
        pcts = [100 * t / total for t in totals]
        print(f"    gen {gen:>7,}: sym0={pcts[0]:5.1f}%  sym1={pcts[1]:5.1f}%  sym2={pcts[2]:5.1f}%")


def analyze_input_mi(path: str):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: try_float(v) for k, v in row.items()})

    if not rows:
        print("\n  INPUT MI: empty file")
        return

    n = len(rows)
    print(f"\n{'=' * 60}")
    print(f"  INPUT MI ANALYSIS  ({n:,} generations)")
    print(f"{'=' * 60}")
    print("  What are signals actually encoding? (higher MI = stronger encoding)")

    # Get all mi_ columns
    mi_cols = [k for k in rows[0] if k.startswith("mi_")]

    # Final values
    tail_start = max(0, int(n * 0.9))
    tail = rows[tail_start:]

    print("\n  SUSTAINED ENCODING (avg MI over last 10% of run)")
    rankings = []
    for col in mi_cols:
        tail_avg = avg([r.get(col, 0) for r in tail])
        rankings.append((col, tail_avg))
    rankings.sort(key=lambda x: x[1], reverse=True)
    for col, val in rankings:
        bar = "#" * int(val * 200)  # scale for visibility
        print(f"    {col:>15s}: {val:.4f}  {bar}")

    # Evolution of top encodings
    print("\n  TOP 3 ENCODING EVOLUTION")
    top3 = [col for col, _ in rankings[:3]]
    checkpoints = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    header = "    gen".ljust(12) + "".join(f"{c:>16s}" for c in top3)
    print(header)
    for idx in checkpoints:
        r = rows[idx]
        gen = int(r.get("generation", idx))
        vals = "".join(f"{r.get(c, 0):>16.4f}" for c in top3)
        print(f"    {gen:>7,}{vals}")


def compare_runs(stats_list: list[RunStats]):
    print(f"\n{'=' * 70}")
    print("  COMPARISON")
    print(f"{'=' * 70}")

    headers = [s.name[:20] for s in stats_list]
    w = max(len(h) for h in headers) + 2

    def row(label, getter, fmt=".4f"):
        vals = [getter(s) for s in stats_list]
        formatted = [f"{v:{fmt}}" if isinstance(v, float) else str(v) for v in vals]
        print(f"    {label:<25s}" + "".join(f"{v:>{w}s}" for v in formatted))

    print(f"\n    {'':25s}" + "".join(f"{h:>{w}s}" for h in headers))
    print(f"    {'-' * 25}" + ("-" * w) * len(headers))

    row("Generations", lambda s: s.gens, ",d")
    row("Final avg fitness", lambda s: s.avg_fitness_final, ".1f")
    row("Peak avg fitness", lambda s: s.avg_fitness_peak, ".1f")
    row("Final brain size", lambda s: s.avg_hidden_final, ".1f")
    row("Peak brain size", lambda s: s.avg_hidden_peak, ".1f")
    row("Brain peak gen", lambda s: s.brain_growth_rate, ",d")
    row("Final MI", lambda s: s.mi_final, ".4f")
    row("Peak MI", lambda s: s.mi_peak, ".4f")
    row("MI peak gen", lambda s: s.mi_peak_gen, ",d")
    row("Sustained MI", lambda s: s.mi_sustained, ".4f")
    row("JSD (pred)", lambda s: s.jsd_pred_final, ".4f")
    row("Silence corr", lambda s: s.silence_corr_final, ".4f")
    row("Sender-fitness", lambda s: s.sender_fit_final, ".4f")
    row("Response-fitness", lambda s: s.response_fit_final, ".4f")
    row("Max fluct ratio", lambda s: s.traj_fluct_max, ".4f")


def find_epochs(rows: list[dict], window: int = 500) -> list[tuple[int, int, str]]:
    """Identify distinct evolutionary epochs based on metric regime changes."""
    epochs = []
    n = len(rows)
    if n < window * 2:
        return [(0, n - 1, "single epoch")]

    prev_regime = None
    epoch_start = 0

    for i in range(0, n, window):
        chunk = rows[i : min(i + window, n)]
        mi = avg([r.get("mutual_info", 0) for r in chunk])
        hidden = avg([r.get("avg_hidden", 0) for r in chunk])
        silence = avg([r.get("silence_corr", 0) for r in chunk])

        # Classify regime
        if mi > 0.05:
            regime = "high-MI"
        elif hidden > 40:
            regime = "large-brain"
        elif silence < -0.3:
            regime = "silence-dominant"
        else:
            regime = "exploration"

        if regime != prev_regime and prev_regime is not None:
            gen = int(rows[i].get("generation", i))
            epochs.append((epoch_start, gen, prev_regime))
            epoch_start = gen

        prev_regime = regime

    epochs.append((epoch_start, int(rows[-1].get("generation", n - 1)), prev_regime))
    return epochs


def print_epochs(rows: list[dict]):
    epochs = find_epochs(rows)
    if not epochs:
        return

    print("\n  EVOLUTIONARY EPOCHS")
    for start, end, regime in epochs:
        print(f"    gen {start:>7,} - {end:>7,}: {regime}")


def main():
    parser = argparse.ArgumentParser(description="Analyze semiotic-emergence output")
    parser.add_argument("output_csvs", nargs="+", help="output.csv file(s)")
    parser.add_argument("--trajectory", "-t", help="trajectory.csv file")
    parser.add_argument("--input-mi", "-m", help="input_mi.csv file")
    parser.add_argument("--all", "-a", nargs=2, metavar=("TRAJ", "INPUT_MI"),
                        help="trajectory.csv and input_mi.csv")
    parser.add_argument("--epochs", "-e", action="store_true",
                        help="Show evolutionary epoch detection")
    args = parser.parse_args()

    all_stats = []
    for path in args.output_csvs:
        rows = read_output_csv(path)
        if not rows:
            print(f"Empty: {path}")
            continue
        # Tag rows with source
        for r in rows:
            r["_path"] = path
        name = Path(path).parent.name or Path(path).stem
        stats = analyze_run(rows, name=name)
        print_run_stats(stats)
        all_stats.append(stats)

        if args.epochs:
            print_epochs(rows)

    if len(all_stats) > 1:
        compare_runs(all_stats)

    traj = args.trajectory or (args.all[0] if args.all else None)
    input_mi = args.input_mi or (args.all[1] if args.all else None)

    if traj:
        analyze_trajectory(traj)
    if input_mi:
        analyze_input_mi(input_mi)


if __name__ == "__main__":
    main()
