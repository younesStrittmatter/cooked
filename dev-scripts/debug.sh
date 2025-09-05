#!/usr/bin/env bash
set -euo pipefail

MODE="single"
APP_PATH="${APP_PATH:-}"
DEFAULT_APP_PATH="wsgi:app"

# ---- Args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --single) MODE="single"; shift;;
    --multi)  MODE="multi";  shift;;
    *)        APP_PATH="$1"; shift;;
  esac
done

# ---- App path ----
if [[ -z "${APP_PATH}" ]]; then
  if   [[ -f "wsgi.py"  ]]; then APP_PATH="wsgi:app";
  elif [[ -f "serve.py" ]]; then APP_PATH="serve:app";
  else                            APP_PATH="${DEFAULT_APP_PATH}"; fi
fi

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export PYTHONUNBUFFERED=1
export EVENTLET_NO_GREENDNS=1   # avoid slow DNS monkey-patch locally
export PORT="${PORT:-8080}"

py() { python - "$@"; }

verify_import() {
  py "$1" <<'PY'
import importlib, sys
arg=sys.argv[1]; mod,sep,attr=arg.partition(":")
assert sep, "APP_PATH must be module:variable (e.g. wsgi:app)"
m=importlib.import_module(mod); assert hasattr(m,attr), f"{mod} has no attr {attr}"
print(f"OK: imported {mod}:{attr}")
PY
}

port_open() {
  py "$@" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket(); s.settimeout(0.2)
try:
    s.connect((host, port)); print("open")
except Exception:
    print("closed")
finally:
    s.close()
PY
}

phys_cores() {
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.physicalcpu 2>/dev/null || echo 2
  else
    py <<'PY'
import os
c = os.cpu_count() or 2
print(max(1, c//2))
PY
  fi
}

choose_eventlet_hub() {
  # Respect pre-set value
  if [[ -n "${EVENTLET_HUB:-}" ]]; then
    echo "$EVENTLET_HUB"; return
  fi
  py <<'PY'
import sys, importlib
platform=sys.platform
candidates = ['selects']
if platform.startswith('darwin'):
    candidates = ['kqueue','selects']
elif platform.startswith('linux'):
    candidates = ['epolls','poll','selects']
try:
    import eventlet.hubs as hubs
except Exception:
    print('selects'); raise SystemExit
for name in candidates:
    try:
        importlib.import_module(f"eventlet.hubs.{name}")
        try:
            hubs.use_hub(name)
            print(name); break
        except Exception:
            continue
    except Exception:
        continue
else:
    print('selects')
PY
}

# Pick a correct Eventlet hub for this OS (does NOT import your app)
export EVENTLET_HUB="$(choose_eventlet_hub)"
echo ">>> EVENTLET_HUB=${EVENTLET_HUB}"

GUNICORN_FLAGS=(
  --bind 0.0.0.0:${PORT}
  --log-level info
  --access-logfile -
  --error-logfile -
  --timeout 120
  --graceful-timeout 30
  --keep-alive 25
)

if [[ "$MODE" == "single" ]]; then
  # Single worker, no Redis, no dotenv
  unset REDIS_URL
  export FLASK_SKIP_DOTENV=1
  export SOCKIO_FORCE_BASE=1
  export SOCKIO_COMPRESS=0

  # import ONCE, after env is set
  verify_import "$APP_PATH"

  WORKERS=1
  echo ">>> Mode: single | Workers: $WORKERS"
  echo ">>> Socket.IO: BaseManager (no Redis)"
  echo ">>> Connect: http://localhost:8080/"
  exec gunicorn -k eventlet -w "$WORKERS" "$APP_PATH" "${GUNICORN_FLAGS[@]}"

else
  # Multi-worker: Redis + RedisManager
  export FLASK_SKIP_DOTENV=0
  export SOCKIO_FORCE_BASE=0
  export SOCKIO_COMPRESS="${SOCKIO_COMPRESS:-0}"

  # Default workers based on physical cores; override with WEB_CONCURRENCY=N
  if [[ -z "${WEB_CONCURRENCY:-}" ]]; then
    CORES="$(phys_cores)"
    export WEB_CONCURRENCY="$(( CORES < 3 ? CORES : 3 ))"
  fi

  export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"

  # Start local Redis if needed
  if [[ "$REDIS_URL" =~ ^redis://(localhost|127\.0\.0\.1):([0-9]+)/ ]]; then
    host="${BASH_REMATCH[1]}"; port="${BASH_REMATCH[2]}"
    if [[ "$(port_open "$host" "$port")" != "open" ]]; then
      if command -v redis-server >/dev/null 2>&1; then
        echo ">>> Starting local redis-server on $host:$port ..."
        redis-server --daemonize yes
        sleep 1
      else
        echo "!!! redis-server missing. Install Redis or set REDIS_URL to a reachable instance."
        exit 1
      fi
    fi
  fi

  # import ONCE, after env/redis are set
  verify_import "$APP_PATH"

  echo ">>> Mode: multi"
  echo ">>> Workers: ${WEB_CONCURRENCY} (override with WEB_CONCURRENCY=N)"
  echo ">>> Redis: ${REDIS_URL}"
  echo ">>> Socket.IO: RedisManager (cross-process)"
  echo ">>> Connect: http://localhost:8080/   (client must use WebSocket-only: transports:['websocket'], upgrade:false)"

  exec gunicorn -k eventlet -w "${WEB_CONCURRENCY}" "$APP_PATH" "${GUNICORN_FLAGS[@]}"
fi
