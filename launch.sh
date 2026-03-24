#!/usr/bin/env bash
# Launch a semiotic-emergence run with proper isolation and CPU pinning.
# Usage: ./launch.sh <name> <seed> <generations> [--cores N] [extra flags...]
# Example: ./launch.sh baseline-seed42 42 999999999 --metrics-interval 10
# Example: ./launch.sh big-run 42 999999999 --cores 9 --pop 2000 --grid 100
#
# --cores N: pin this run to N CPU cores. Automatically selects cores not
#            used by other active runs. Without --cores, auto-allocates all
#            cores not claimed by other runs (or all cores if nothing else runs).
#
# Creates: runs/<name>/
#   - output.csv, trajectory.csv, input_mi.csv (from binary)
#   - run.log (stdout/stderr)
#   - meta.txt (git commit, binary hash, full command, timestamp)
#   - pid.txt (process ID for monitoring)
#   - cores.txt (pinned CPU core list, e.g. "3-11")
#   - throughput.tsv (hourly throughput + metrics snapshots from monitor.sh)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="$REPO_DIR/target/release/semiotic-emergence"
MONITOR="$REPO_DIR/monitor.sh"
RUNS_DIR="$REPO_DIR/runs"
TOTAL_CORES=$(nproc)

# --- Validate args ---
if [ $# -lt 3 ]; then
    echo "Usage: $0 <name> <seed> <generations> [--cores N] [extra flags...]"
    echo "Example: $0 baseline-seed42 42 999999999 --metrics-interval 10"
    echo "Example: $0 big-run 42 999999999 --cores 9 --pop 2000 --grid 100"
    exit 1
fi

NAME="$1"; SEED="$2"; GENS="$3"; shift 3

# --- Parse --cores from remaining args ---
REQUESTED_CORES=""
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --cores)
            REQUESTED_CORES="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done
EXTRA_FLAGS="${EXTRA_ARGS[*]:-}"

# --- Find cores claimed by other active runs ---
claimed_cores() {
    local claimed=""
    for dir in "$RUNS_DIR"/*/; do
        [ -d "$dir" ] || continue
        local pid_file="$dir/pid.txt"
        local cores_file="$dir/cores.txt"
        [ -f "$pid_file" ] && [ -f "$cores_file" ] || continue
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            local range
            range=$(cat "$cores_file")
            # Expand range like "3-11" or "0-2" to individual core numbers
            for part in $(echo "$range" | tr ',' ' '); do
                if [[ "$part" == *-* ]]; then
                    local lo=${part%-*}
                    local hi=${part#*-}
                    for ((i=lo; i<=hi; i++)); do
                        claimed="$claimed $i"
                    done
                else
                    claimed="$claimed $part"
                fi
            done
        fi
    done
    echo "$claimed" | tr ' ' '\n' | sort -un | tr '\n' ' '
}

allocate_cores() {
    local need=$1
    local claimed
    claimed=$(claimed_cores)

    # Build list of free cores
    local free=()
    for ((i=0; i<TOTAL_CORES; i++)); do
        local taken=false
        for c in $claimed; do
            [ "$c" -eq "$i" ] && taken=true && break
        done
        $taken || free+=("$i")
    done

    if [ ${#free[@]} -eq 0 ]; then
        echo "WARN: All $TOTAL_CORES cores claimed by other runs, using all cores" >&2
        echo "0-$((TOTAL_CORES-1))"
        return
    fi

    if [ "$need" -gt ${#free[@]} ]; then
        echo "WARN: Requested $need cores but only ${#free[@]} free, using all free" >&2
        need=${#free[@]}
    fi

    # Take the first N free cores, format as taskset range
    local selected=("${free[@]:0:$need}")
    local lo=${selected[0]}
    local hi=${selected[-1]}

    # Check if contiguous for clean range format
    local is_contiguous=true
    for ((i=1; i<${#selected[@]}; i++)); do
        if [ $((selected[i] - selected[i-1])) -ne 1 ]; then
            is_contiguous=false
            break
        fi
    done

    if $is_contiguous; then
        echo "$lo-$hi"
    else
        echo "${selected[*]}" | tr ' ' ','
    fi
}

# --- Check binary exists ---
if [ ! -x "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Build first: cargo build --release"
    exit 1
fi

# --- Check run directory doesn't already exist ---
RUN_DIR="$RUNS_DIR/$NAME"
if [ -d "$RUN_DIR" ]; then
    echo "ERROR: Run directory already exists: $RUN_DIR"
    echo "Pick a different name or remove the existing directory."
    exit 1
fi

# --- Create run directory ---
mkdir -p "$RUN_DIR"

# --- Allocate CPU cores ---
if [ -n "$REQUESTED_CORES" ]; then
    CORE_RANGE=$(allocate_cores "$REQUESTED_CORES")
else
    # Auto: take all unclaimed cores
    CLAIMED=$(claimed_cores)
    CLAIMED_COUNT=$(echo "$CLAIMED" | wc -w)
    FREE_COUNT=$((TOTAL_CORES - CLAIMED_COUNT))
    if [ "$FREE_COUNT" -le 0 ]; then
        FREE_COUNT=$TOTAL_CORES
    fi
    CORE_RANGE=$(allocate_cores "$FREE_COUNT")
fi

# Count cores in the range for RAYON_NUM_THREADS
count_cores() {
    local count=0
    for part in $(echo "$1" | tr ',' ' '); do
        if [[ "$part" == *-* ]]; then
            local lo=${part%-*}
            local hi=${part#*-}
            count=$((count + hi - lo + 1))
        else
            count=$((count + 1))
        fi
    done
    echo "$count"
}
THREADS=$(count_cores "$CORE_RANGE")

# --- Write metadata ---
GIT_COMMIT=$(cd "$REPO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(cd "$REPO_DIR" && git diff --quiet 2>/dev/null && echo "clean" || echo "dirty")
BINARY_HASH=$(sha256sum "$BINARY" | cut -d' ' -f1)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "$CORE_RANGE" > "$RUN_DIR/cores.txt"

cat > "$RUN_DIR/meta.txt" <<EOF
name: $NAME
seed: $SEED
generations: $GENS
extra_flags: $EXTRA_FLAGS
threads: $THREADS
cores: $CORE_RANGE
git_commit: $GIT_COMMIT ($GIT_DIRTY)
binary_sha256: $BINARY_HASH
started: $TIMESTAMP
command: taskset -c $CORE_RANGE RAYON_NUM_THREADS=$THREADS $BINARY $SEED $GENS $EXTRA_FLAGS
EOF

echo "=== Run: $NAME ==="
echo "Dir:    $RUN_DIR"
echo "Seed:   $SEED"
echo "Gens:   $GENS"
echo "Cores:  $CORE_RANGE ($THREADS threads)"
echo "Flags:  $EXTRA_FLAGS"
echo "Commit: $GIT_COMMIT ($GIT_DIRTY)"
echo "Binary: ${BINARY_HASH:0:16}..."

# --- Launch with correct CWD and CPU pinning ---
cd "$RUN_DIR"
nohup taskset -c "$CORE_RANGE" bash -c "RAYON_NUM_THREADS=$THREADS $BINARY $SEED $GENS $EXTRA_FLAGS" \
    >> run.log 2>&1 </dev/null &
PID=$!
echo "$PID" > pid.txt

# Verify it survived
sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "PID:    $PID (running)"

    # Start throughput monitor as companion process
    START_EPOCH=$(date +%s)
    if [ -x "$MONITOR" ]; then
        nohup "$MONITOR" "$RUN_DIR" "$PID" "$START_EPOCH" \
            > /dev/null 2>&1 </dev/null &
        MONITOR_PID=$!
        echo "$MONITOR_PID" > monitor_pid.txt
        echo "Monitor: $MONITOR_PID (throughput sampling every hour)"
    fi

    echo ""
    echo "Logs:    tail -f $RUN_DIR/run.log"
    echo "Speed:   cat $RUN_DIR/throughput.tsv"
    echo "Status:  ./status.sh"
    echo "Stop:    kill $PID"
else
    echo "ERROR: Process died immediately. Check $RUN_DIR/run.log"
    tail -20 "$RUN_DIR/run.log" 2>/dev/null
    exit 1
fi
