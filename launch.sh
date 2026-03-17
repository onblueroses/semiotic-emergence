#!/usr/bin/env bash
# Launch a semiotic-emergence run with proper isolation.
# Usage: ./launch.sh <name> <seed> <generations> [extra flags...]
# Example: ./launch.sh baseline-seed42 42 999999999 --metrics-interval 10
# Example: ./launch.sh high-coverage-42 42 100000 --metrics-interval 10 --zone-coverage 0.20
#
# Creates: runs/<name>/
#   - output.csv, trajectory.csv, input_mi.csv (from binary)
#   - run.log (stdout/stderr)
#   - meta.txt (git commit, binary hash, full command, timestamp)
#   - pid.txt (process ID for monitoring)
#   - throughput.tsv (hourly throughput + metrics snapshots from monitor.sh)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="$REPO_DIR/target/release/semiotic-emergence"
MONITOR="$REPO_DIR/monitor.sh"
RUNS_DIR="$REPO_DIR/runs"

# --- Validate args ---
if [ $# -lt 3 ]; then
    echo "Usage: $0 <name> <seed> <generations> [extra flags...]"
    echo "Example: $0 baseline-seed42 42 999999999 --metrics-interval 10"
    exit 1
fi

NAME="$1"; SEED="$2"; GENS="$3"; shift 3
EXTRA_FLAGS="$*"

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

# --- Write metadata ---
GIT_COMMIT=$(cd "$REPO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(cd "$REPO_DIR" && git diff --quiet 2>/dev/null && echo "clean" || echo "dirty")
BINARY_HASH=$(sha256sum "$BINARY" | cut -d' ' -f1)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
THREADS="${RAYON_NUM_THREADS:-$(nproc)}"

cat > "$RUN_DIR/meta.txt" <<EOF
name: $NAME
seed: $SEED
generations: $GENS
extra_flags: $EXTRA_FLAGS
threads: $THREADS
git_commit: $GIT_COMMIT ($GIT_DIRTY)
binary_sha256: $BINARY_HASH
started: $TIMESTAMP
command: RAYON_NUM_THREADS=$THREADS $BINARY $SEED $GENS $EXTRA_FLAGS
EOF

echo "=== Run: $NAME ==="
echo "Dir:    $RUN_DIR"
echo "Seed:   $SEED"
echo "Gens:   $GENS"
echo "Flags:  $EXTRA_FLAGS"
echo "Commit: $GIT_COMMIT ($GIT_DIRTY)"
echo "Binary: ${BINARY_HASH:0:16}..."

# --- Launch with correct CWD ---
cd "$RUN_DIR"
nohup bash -c "RAYON_NUM_THREADS=$THREADS $BINARY $SEED $GENS $EXTRA_FLAGS" \
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
