#!/bin/bash

PROMPT="${1:-prompt here}"
MAX_RUNS=20

stop_requested=0
force_stop=0
current_pid=""

handle_sigint() {
    if [[ -n "$current_pid" ]] && kill -0 "$current_pid" 2>/dev/null; then
        if (( stop_requested == 0 )); then
            stop_requested=1
            echo "" >&2
            echo "Stop requested. The current Codex run will finish; press Ctrl-C again to stop immediately." >&2
        else
            force_stop=1
            echo "" >&2
            echo "Stopping current Codex run immediately..." >&2
            kill -TERM "$current_pid" 2>/dev/null || true
            kill -TERM -- "-$current_pid" 2>/dev/null || true
        fi
    else
        stop_requested=1
        force_stop=1
    fi
}

trap handle_sigint INT

for ((i = 1; i <= MAX_RUNS; i++)); do
    echo "=== Run $i/$MAX_RUNS ==="

    setsid codex --dangerously-bypass-approvals-and-sandbox exec "$PROMPT" &
    current_pid=$!

    run_status=0
    while true; do
        wait "$current_pid" || run_status=$?
        if ! kill -0 "$current_pid" 2>/dev/null; then
            break
        fi
    done
    current_pid=""

    echo ""

    if (( force_stop )); then
        exit 130
    fi

    if (( stop_requested )); then
        break
    fi
done
