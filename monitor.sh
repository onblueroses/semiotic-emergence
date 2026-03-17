#!/usr/bin/env bash
# Throughput monitor for semiotic-emergence runs.
# Spawned by launch.sh as a background companion. Samples every hour.
# Writes throughput.tsv in the run directory. Exits when the simulation PID dies.
#
# Usage (called by launch.sh, not directly):
#   monitor.sh <run-dir> <sim-pid> <start-epoch>

set -euo pipefail

RUN_DIR="$1"
SIM_PID="$2"
START_EPOCH="$3"
CSV="$RUN_DIR/output.csv"
TSV="$RUN_DIR/throughput.tsv"
INTERVAL=3600  # 1 hour

# Write header
echo -e "timestamp\telapsed_h\tgeneration\tgen_per_min\tavg_fitness\tmutual_info\tsignal_entropy" > "$TSV"

sample() {
    if [ ! -f "$CSV" ]; then
        return
    fi
    local last_line
    # Use second-to-last line to avoid partial writes (sim doesn't flush atomically)
    last_line=$(tail -2 "$CSV" 2>/dev/null | head -1) || return
    # Skip if header or empty
    case "$last_line" in generation*|"") return ;; esac

    local now
    now=$(date +%s)
    local elapsed_s=$(( now - START_EPOCH ))
    local elapsed_h
    elapsed_h=$(awk "BEGIN { printf \"%.2f\", $elapsed_s / 3600 }")

    local gen fitness mi entropy
    gen=$(echo "$last_line" | cut -d, -f1)
    fitness=$(echo "$last_line" | cut -d, -f2)
    mi=$(echo "$last_line" | cut -d, -f6)
    entropy=$(echo "$last_line" | cut -d, -f23)

    local gen_per_min
    if [ "$elapsed_s" -gt 0 ]; then
        gen_per_min=$(awk "BEGIN { printf \"%.1f\", $gen / ($elapsed_s / 60) }")
    else
        gen_per_min="0.0"
    fi

    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo -e "${ts}\t${elapsed_h}\t${gen}\t${gen_per_min}\t${fitness}\t${mi}\t${entropy}" >> "$TSV"
}

# Take initial sample after 5 minutes (let the run warm up)
sleep 300
sample

# Then sample every hour until the simulation dies
while kill -0 "$SIM_PID" 2>/dev/null; do
    sleep "$INTERVAL"
    # Check again after sleep - PID may have died during the wait
    if kill -0 "$SIM_PID" 2>/dev/null; then
        sample
    fi
done

# Final sample on exit
sample
