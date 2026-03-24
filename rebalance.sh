#!/usr/bin/env bash
# Redistribute CPU cores across active runs without restarting them.
# Usage: ./rebalance.sh                         # auto-distribute evenly
#        ./rebalance.sh run1=3 run2=9            # explicit core counts
#
# Examples:
#   ./rebalance.sh                               # split 12 cores evenly across 2 runs
#   ./rebalance.sh v14-baseline-42=3 v14-2k-42=9 # 3 for baseline, 9 for 2k

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$REPO_DIR/runs"
TOTAL_CORES=$(nproc)

# Collect active runs
declare -A ACTIVE_PIDS
for dir in "$RUNS_DIR"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    pid_file="$dir/pid.txt"
    [ -f "$pid_file" ] || continue
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        ACTIVE_PIDS[$name]=$pid
    fi
done

if [ ${#ACTIVE_PIDS[@]} -eq 0 ]; then
    echo "No active runs found."
    exit 0
fi

echo "Active runs: ${!ACTIVE_PIDS[*]}"
echo "Total cores: $TOTAL_CORES"
echo ""

# Build allocation map
declare -A ALLOC

if [ $# -gt 0 ]; then
    # Explicit: parse run=cores pairs
    total_requested=0
    for arg in "$@"; do
        name="${arg%=*}"
        cores="${arg#*=}"
        if [ -z "${ACTIVE_PIDS[$name]:-}" ]; then
            echo "ERROR: '$name' is not an active run. Active: ${!ACTIVE_PIDS[*]}"
            exit 1
        fi
        ALLOC[$name]=$cores
        total_requested=$((total_requested + cores))
    done
    if [ "$total_requested" -gt "$TOTAL_CORES" ]; then
        echo "WARN: Requested $total_requested cores but only $TOTAL_CORES available (oversubscribed)"
    fi
    # Allocate unmentioned runs with 0 (they keep current pinning)
    for name in "${!ACTIVE_PIDS[@]}"; do
        if [ -z "${ALLOC[$name]:-}" ]; then
            remaining=$((TOTAL_CORES - total_requested))
            if [ "$remaining" -gt 0 ]; then
                ALLOC[$name]=$remaining
                total_requested=$((total_requested + remaining))
            else
                echo "WARN: No cores left for $name (not mentioned in args)"
            fi
        fi
    done
else
    # Auto: distribute evenly, larger runs get the remainder
    n=${#ACTIVE_PIDS[@]}
    per_run=$((TOTAL_CORES / n))
    remainder=$((TOTAL_CORES % n))

    # Sort runs by name so allocation is deterministic
    mapfile -t sorted < <(printf '%s\n' "${!ACTIVE_PIDS[@]}" | sort)

    i=0
    for name in "${sorted[@]}"; do
        extra=0
        if [ $i -lt $remainder ]; then
            extra=1
        fi
        ALLOC[$name]=$((per_run + extra))
        ((i++))
    done
fi

# Assign core ranges sequentially, no overlap
core_offset=0
for name in $(printf '%s\n' "${!ALLOC[@]}" | sort); do
    cores=${ALLOC[$name]}
    pid=${ACTIVE_PIDS[$name]}
    run_dir="$RUNS_DIR/$name"

    lo=$core_offset
    hi=$((core_offset + cores - 1))
    if [ "$hi" -ge "$TOTAL_CORES" ]; then
        hi=$((TOTAL_CORES - 1))
    fi
    range="$lo-$hi"

    # Apply to all threads of the process
    taskset -apc "$range" "$pid" > /dev/null 2>&1

    # Update cores.txt
    echo "$range" > "$run_dir/cores.txt"

    # Show result
    old_range=""
    if [ -f "$run_dir/cores.txt.bak" ]; then
        old_range=" (was $(cat "$run_dir/cores.txt.bak"))"
    fi
    printf "  %-30s  PID %-8s  cores %s  (%d threads)%s\n" "$name" "$pid" "$range" "$cores" "$old_range"

    core_offset=$((hi + 1))
done

echo ""
echo "Done. Use './status.sh' to verify."
