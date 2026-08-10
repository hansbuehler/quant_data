#!/usr/bin/env bash
set -euo pipefail

ports=(8900 8901 8902 8903)

for port in "${ports[@]}"; do
    echo "Checking port $port..."

    # Find PIDs listening on this TCP port
    mapfile -t pids < <(
        sudo lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u
    )

    if [ "${#pids[@]}" -eq 0 ]; then
        echo "  No listener on port $port"
        continue
    fi

    for pid in "${pids[@]}"; do
        name="$(ps -p "$pid" -o comm= 2>/dev/null || true)"
        echo "  Killing PID $pid ${name:+($name)}"
        sudo kill "$pid"
    done
done

sleep 1

echo
echo "Remaining listeners:"
for port in "${ports[@]}"; do
    sudo lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
done

