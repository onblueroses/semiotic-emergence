#!/usr/bin/env bash
# Stop a running semiotic-emergence run and record end time.
# Usage: ./stop.sh <run-name>

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$REPO_DIR/runs"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run-name>"
    exit 1
fi

RUN_DIR="$RUNS_DIR/$1"
if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Run directory not found: $RUN_DIR"
    exit 1
fi

PID_FILE="$RUN_DIR/pid.txt"
if [ ! -f "$PID_FILE" ]; then
    echo "No pid.txt found - run may not have been started with launch.sh"
    exit 1
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        echo "WARN: Process $PID didn't die, sending SIGKILL"
        kill -9 "$PID"
    fi
    echo "stopped: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "$RUN_DIR/meta.txt"
    FINAL_GEN=$(tail -1 "$RUN_DIR/output.csv" 2>/dev/null | cut -d, -f1)
    echo "final_gen: $FINAL_GEN" >> "$RUN_DIR/meta.txt"
    echo "Stopped $1 (PID $PID) at gen $FINAL_GEN"
else
    echo "Process $PID is not running (already stopped)"
fi
