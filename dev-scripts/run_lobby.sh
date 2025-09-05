#!/usr/bin/env bash
set -euo pipefail
export PORT="${PORT:-8081}"                         # lobby port
export SHARD_URLS="${SHARD_URLS:-http://localhost:8080}"

echo "Lobby on :${PORT}"
echo "Shards: ${SHARD_URLS}"
# Run with gunicorn so it stays up; path is module:app inside lobby/
exec gunicorn -k sync -w 1 main:app \
  --chdir "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lobby" && pwd)" \
  --bind 0.0.0.0:${PORT} \
  --access-logfile - --error-logfile -
