#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Cloud bench via lobby → shards using your stress_testing/bots.py
# Usage:
#   dev-scripts/run_cloud_bench.sh <LOBBY_URL> [GAMES] [PLAYERS] [DURATION] [ROOM_PREFIX]
#
# Example:
#   dev-scripts/run_cloud_bench.sh https://game-lobby-xxxxx.run.app 20 2 30 bench-
#
# Notes:
# - Spawns one bots.py process per room so each connects to its assigned shard.
# - Requires: curl, jq, python, stress_testing/bots.py
# - Bots use WebSocket-only and print per-second stats as usual.
# ------------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOBBY_URL="${1:-}"
GAMES="${2:-10}"
PLAYERS="${3:-2}"
DURATION="${4:-15}"
ROOM_PREFIX="${5:-bench-}"

if [[ -z "$LOBBY_URL" ]]; then
  echo "Usage: $0 <LOBBY_URL> [GAMES] [PLAYERS] [DURATION] [ROOM_PREFIX]"
  exit 1
fi

command -v curl >/dev/null || { echo "curl not found"; exit 1; }
command -v jq   >/dev/null || { echo "jq not found";   exit 1; }

BOT="${ROOT_DIR}/stress_testing/bots.py"
[[ -f "$BOT" ]] || { echo "bots.py not found at $BOT"; exit 1; }

# Quick health probe (optional)
echo ">>> Lobby: $LOBBY_URL"
if ! curl -fsS "$LOBBY_URL/healthz" >/dev/null; then
  echo "WARNING: $LOBBY_URL/healthz not reachable; continuing anyway…"
fi

# Normalize lobby URL (strip trailing /)
LOBBY_URL="${LOBBY_URL%/}"

pids=()
rooms=()
shards=()

echo ">>> Launching $GAMES games x $PLAYERS players for ${DURATION}s"
for i in $(seq 1 "$GAMES"); do
  # We run one game per process. bots.py will use room = "<prefix>0".
  prefix="${ROOM_PREFIX}${i}-"
  room="${prefix}0"

  # Ask lobby where this room lives
  resp="$(curl -fsS "${LOBBY_URL}/where?room=${room}")" || {
    echo "✗ Failed to resolve shard for room=$room"; exit 1;
  }
  join_url="$(jq -r '.join_url' <<<"$resp")"
  shard_base="${join_url%%\?*}"   # strip query; keep scheme+host[/]

  echo "  - room=${room} → shard=${shard_base}"

  # Launch one bot process for this specific room on its shard
  (
    cd "$ROOT_DIR"
    # WS_ONLY=1 ensures websocket-only like your local tests
    WS_ONLY=1 \
    python "$BOT" \
      --server "$shard_base" \
      --games 1 \
      --players "$PLAYERS" \
      --duration "$DURATION" \
      --room-prefix "$prefix"
  ) &
  pids+=("$!")
  rooms+=("$room")
  shards+=("$shard_base")
done

echo ">>> Started ${#pids[@]} bot processes."
echo ">>> Waiting for completion…"
fails=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  room="${rooms[$idx]}"
  wait "$pid" || { echo "✗ bot for $room exited non-zero"; ((fails++)) || true; }
done

echo ">>> Done. Failures: $fails"
exit "$fails"
