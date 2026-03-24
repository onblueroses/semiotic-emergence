#!/bin/bash
set -euo pipefail
cd /root/semiotic-emergence
source ~/.cargo/env

git fetch origin
git checkout main
git reset --hard origin/main

echo "=== Building BASELINE (main) ==="
RUSTFLAGS="-C target-cpu=znver3" cargo build --release
cp target/release/semiotic-emergence /tmp/bench-baseline

echo "=== Building EXPERIMENT (autoopt/004-simd-distance-batch) ==="
git checkout autoopt/004-simd-distance-batch
git reset --hard origin/autoopt/004-simd-distance-batch
RUSTFLAGS="-C target-cpu=znver3" cargo build --release
cp target/release/semiotic-emergence /tmp/bench-experiment

git checkout main

SEED=42
GENS=50
WARMUP=2
RUNS=7
EXTRA="--metrics-interval 10"
TMPDIR=/tmp/bench_work
mkdir -p "$TMPDIR"

echo ""
echo "=== A/B BENCHMARK: $WARMUP warmup + $RUNS measurement per binary ==="
echo ""

echo "--- Warming up ---"
for w in $(seq 1 $WARMUP); do
    cd "$TMPDIR" && rm -f output.csv trajectory.csv input_mi.csv
    /tmp/bench-baseline $SEED $GENS $EXTRA > /dev/null 2>&1
    cd "$TMPDIR" && rm -f output.csv trajectory.csv input_mi.csv
    /tmp/bench-experiment $SEED $GENS $EXTRA > /dev/null 2>&1
    echo "warmup $w done"
done

echo "--- Measuring ---"
BASE_FILE=/tmp/base_times.txt
EXP_FILE=/tmp/exp_times.txt
> "$BASE_FILE"
> "$EXP_FILE"

for r in $(seq 1 $RUNS); do
    cd "$TMPDIR" && rm -f output.csv trajectory.csv input_mi.csv
    START=$(date +%s.%N)
    /tmp/bench-baseline $SEED $GENS $EXTRA > /dev/null 2>&1
    END=$(date +%s.%N)
    BT=$(echo "$END - $START" | bc -l)
    echo "$BT" >> "$BASE_FILE"
    BGPS=$(echo "$GENS / $BT" | bc -l)
    printf "run_%d\tBASE\t%.3f\t%.2f gps\n" "$r" "$BT" "$BGPS"

    cd "$TMPDIR" && rm -f output.csv trajectory.csv input_mi.csv
    START=$(date +%s.%N)
    /tmp/bench-experiment $SEED $GENS $EXTRA > /dev/null 2>&1
    END=$(date +%s.%N)
    ET=$(echo "$END - $START" | bc -l)
    echo "$ET" >> "$EXP_FILE"
    EGPS=$(echo "$GENS / $ET" | bc -l)
    printf "run_%d\tEXPR\t%.3f\t%.2f gps\n" "$r" "$ET" "$EGPS"
done

echo ""
echo "=== RESULTS ==="

SORTED_BASE=$(sort -g "$BASE_FILE")
SORTED_EXP=$(sort -g "$EXP_FILE")
MID=4

BASE_MED=$(echo "$SORTED_BASE" | sed -n "${MID}p")
EXP_MED=$(echo "$SORTED_EXP" | sed -n "${MID}p")
BASE_GPS=$(echo "$GENS / $BASE_MED" | bc -l)
EXP_GPS=$(echo "$GENS / $EXP_MED" | bc -l)
DELTA=$(echo "($EXP_GPS - $BASE_GPS) / $BASE_GPS * 100" | bc -l)

BASE_MEAN=$(awk '{s+=$1; n++} END{print s/n}' "$BASE_FILE")
EXP_MEAN=$(awk '{s+=$1; n++} END{print s/n}' "$EXP_FILE")
BASE_STD=$(awk -v m="$BASE_MEAN" '{d=$1-m; s+=d*d; n++} END{print sqrt(s/n)}' "$BASE_FILE")
EXP_STD=$(awk -v m="$EXP_MEAN" '{d=$1-m; s+=d*d; n++} END{print sqrt(s/n)}' "$EXP_FILE")
BASE_COV=$(echo "$BASE_STD / $BASE_MEAN * 100" | bc -l)
EXP_COV=$(echo "$EXP_STD / $EXP_MEAN * 100" | bc -l)

printf "BASELINE median: %.3fs (%.2f gps, CoV %.1f%%)\n" "$BASE_MED" "$BASE_GPS" "$BASE_COV"
printf "EXPERIMENT median: %.3fs (%.2f gps, CoV %.1f%%)\n" "$EXP_MED" "$EXP_GPS" "$EXP_COV"
printf "DELTA: %+.1f%%\n" "$DELTA"

echo ""
echo "All baseline times:"
cat "$BASE_FILE"
echo "All experiment times:"
cat "$EXP_FILE"
