#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./replays_raw

echo "Syncing missing replay files from gs://replay_files/replays ..."
echo "Tip: set DOWNLOAD_VERBOSE=1 to show every skipped/copied object."

if [[ "${DOWNLOAD_VERBOSE:-0}" == "1" ]]; then
  gsutil -o "GSUtil:parallel_process_count=1" -m -u cooked-455218 cp -rn gs://replay_files/replays ./replays_raw/
else
  gsutil -o "GSUtil:parallel_process_count=1" -m -q -u cooked-455218 cp -rn gs://replay_files/replays ./replays_raw/
fi

count="$(
  python3 - <<'PY'
from pathlib import Path
print(sum(1 for _ in Path("./replays_raw/replays").glob("*.json")))
PY
)"
echo "Replay download check complete: ${count} raw replay files available."
