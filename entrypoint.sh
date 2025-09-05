#!/usr/bin/env bash
set -euo pipefail

# Cloud Run gives PORT
PORT="${PORT:-8080}"

# Keep 1 worker per shard instance (state stays in-process)
WORKERS="${WEB_CONCURRENCY:-1}"

# Eventlet hub: safe fallback
export EVENTLET_NO_GREENDNS=1
export PYTHONUNBUFFERED=1

APP_PATH="${APP_PATH:-wsgi:app}"

echo ">>> Booting shard on port ${PORT} | workers=${WORKERS} | app=${APP_PATH}"
echo ">>> MODE=${MODE} SOCKIO_FORCE_BASE=${SOCKIO_FORCE_BASE} SOCKIO_COMPRESS=${SOCKIO_COMPRESS}"

# Force BaseManager (no Redis) inside a shard; one process, many greenlets
export SOCKIO_FORCE_BASE=1
unset REDIS_URL

exec gunicorn -k eventlet -w "${WORKERS}" "${APP_PATH}" \
  --bind 0.0.0.0:"${PORT}" \
  --log-level info \
  --access-logfile - \
  --error-logfile - \
  --timeout 3600 \
  --graceful-timeout 30 \
  --keep-alive 25
