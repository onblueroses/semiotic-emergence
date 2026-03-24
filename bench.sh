#!/usr/bin/env bash
# Benchmark semiotic-emergence binary. Runs N iterations, outputs median gens/sec.
# Usage: ./bench.sh [seed] [gens] [runs] [extra_args...]
# Example: ./bench.sh 42 50 3 --metrics-interval 10
#
# Output format (TSV):
#   run_1	12.345	4.05
#   run_2	12.100	4.13
#   run_3	12.567	3.98
#   median	12.345	4.05

set -euo pipefail

SEED="${1:-42}"
GENS="${2:-50}"
RUNS="${3:-3}"
shift 3 2>/dev/null || true
EXTRA="$*"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="$REPO_DIR/target/release/semiotic-emergence"

if [ ! -x "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY" >&2
    echo "Build first: RUSTFLAGS=\"-C target-cpu=znver3\" cargo build --release" >&2
    exit 1
fi

# Create temp dir for benchmark outputs
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

declare -a WALL_TIMES
declare -a GENS_PER_SEC

for i in $(seq 1 "$RUNS"); do
    cd "$TMPDIR"
    rm -f output.csv trajectory.csv input_mi.csv

    START=$(date +%s.%N)
    "$BINARY" "$SEED" "$GENS" $EXTRA > /dev/null 2>&1
    END=$(date +%s.%N)

    WALL=$(echo "$END - $START" | bc -l)
    GPS=$(echo "$GENS / $WALL" | bc -l)

    WALL_TIMES+=("$WALL")
    GENS_PER_SEC+=("$GPS")

    printf "run_%d\t%.3f\t%.2f\n" "$i" "$WALL" "$GPS"
done

# Compute median (sort, take middle element)
SORTED_WALL=($(printf '%s\n' "${WALL_TIMES[@]}" | sort -g))
SORTED_GPS=($(printf '%s\n' "${GENS_PER_SEC[@]}" | sort -g))

MID=$(( (RUNS - 1) / 2 ))
printf "median\t%.3f\t%.2f\n" "${SORTED_WALL[$MID]}" "${SORTED_GPS[$MID]}"
