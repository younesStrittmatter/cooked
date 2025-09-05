#!/usr/bin/env bash
set -euo pipefail
PORTS=(${PORTS:-8080 8081 8082 8083})  # edit as you like

for p in "${PORTS[@]}"; do
  echo ">>> Launching single worker on :$p"
  PORT="$p" ./run_debug.sh --single >"farm_${p}.log" 2>&1 &
  PIDS+=($!)
done

echo "PIDs: ${PIDS[*]}"
echo "Press Ctrl-C to stop"; trap 'kill "${PIDS[@]}"' INT
wait