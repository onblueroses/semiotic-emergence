#!/usr/bin/env bash
# Show status of all semiotic-emergence runs.
# Usage: ./status.sh [run-name]

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$REPO_DIR/runs"

show_run() {
    local dir="$1"
    local name=$(basename "$dir")
    local pid_file="$dir/pid.txt"
    local meta_file="$dir/meta.txt"
    local csv_file="$dir/output.csv"
    local log_file="$dir/run.log"

    # Skip directories without meta.txt (legacy runs)
    if [ ! -f "$meta_file" ]; then
        return
    fi

    local pid=$(cat "$pid_file" 2>/dev/null || echo "")
    local status="STOPPED"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        status="RUNNING"
    fi

    local gen="?"
    if [ -f "$csv_file" ]; then
        gen=$(tail -1 "$csv_file" 2>/dev/null | cut -d, -f1)
        local lines=$(wc -l < "$csv_file" 2>/dev/null)
    fi

    local seed=$(grep "^seed:" "$meta_file" | cut -d' ' -f2)
    local started=$(grep "^started:" "$meta_file" | cut -d' ' -f2)
    local flags=$(grep "^extra_flags:" "$meta_file" | cut -d' ' -f2-)
    local commit=$(grep "^git_commit:" "$meta_file" | cut -d' ' -f2-)

    local cores_file="$dir/cores.txt"
    local cores=""
    if [ -f "$cores_file" ]; then
        cores=" cores=$(cat "$cores_file")"
    fi

    local throughput_file="$dir/throughput.tsv"
    local speed=""
    if [ -f "$throughput_file" ] && [ "$(wc -l < "$throughput_file")" -gt 1 ]; then
        speed=$(tail -1 "$throughput_file" | cut -f4)
        speed=" ${speed} gen/min"
    fi

    printf "%-30s  %-8s  seed=%-6s  gen=%-8s%s%s  commit=%s\n" "$name" "$status" "$seed" "$gen" "$cores" "$speed" "$commit"

    if [ "$2" = "verbose" ]; then
        if [ -f "$throughput_file" ] && [ "$(wc -l < "$throughput_file")" -gt 1 ]; then
            echo "  Throughput:"
            column -t -s$'\t' "$throughput_file" | tail -5 | sed 's/^/    /'
            echo ""
        elif [ -f "$log_file" ]; then
            echo "  Latest:"
            tail -1 "$log_file" 2>/dev/null | sed 's/^/    /'
            echo ""
        fi
    fi
}

if [ $# -ge 1 ] && [ -d "$RUNS_DIR/$1" ]; then
    show_run "$RUNS_DIR/$1" verbose
else
    echo "=== Semiotic Emergence Runs ==="
    echo ""
    for dir in "$RUNS_DIR"/*/; do
        [ -d "$dir" ] && show_run "$dir" "${1:-brief}"
    done
fi
