#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-http://localhost:8080}"
GAMES="${GAMES:-10}"
PLAYERS="${PLAYERS:-2}"
DURATION="${DURATION:-30}"

# ensure client deps exist
python - <<'PY'
import sys, subprocess
try:
    import socketio  # noqa: F401
except Exception:
    print(">>> Installing python-socketio client ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-socketio[client]"])
print(">>> OK")
PY

exec python stress_testing/bots.py \
  --server "$SERVER" \
  --games "$GAMES" \
  --players "$PLAYERS" \
  --duration "$DURATION"
