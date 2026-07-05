#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON="${ROOT}/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

echo ">>> Using Python: ${PYTHON}"
mkdir -p "${ROOT}/.matplotlib"
mkdir -p "${ROOT}/.cache/fontconfig"
export MPLCONFIGDIR="${ROOT}/.matplotlib"
export XDG_CACHE_HOME="${ROOT}/.cache"
export PYTHONPATH="${ROOT}/engine/src:${ROOT}/games:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}/analysis"
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  echo ">>> Skipping replay download (SKIP_DOWNLOAD=1)"
else
  echo ">>> Downloading replay files"
  bash download.sh
fi

echo ">>> Filtering valid two-player replays"
"${PYTHON}" filter_replays.py

cd "${ROOT}"
# Default to (cores - 1) parallel workers, capped at 8; override with REPLAY_JOBS.
if [[ -z "${REPLAY_JOBS:-}" ]]; then
  CORES="$("${PYTHON}" -c 'import os; print(os.cpu_count() or 1)')"
  REPLAY_JOBS=$(( CORES > 1 ? CORES - 1 : 1 ))
  if (( REPLAY_JOBS > 8 )); then REPLAY_JOBS=8; fi
fi
echo ">>> Generating missing tick logs (headless, ${REPLAY_JOBS} workers)"
"${PYTHON}" replay_headless.py --jobs "${REPLAY_JOBS}"

cd "${ROOT}/analysis"
echo ">>> Creating per-game bundles"
"${PYTHON}" create_bundles.py

echo ">>> Creating item-touch histories"
"${PYTHON}" create_item_touches.py

echo ">>> Combining action CSVs"
"${PYTHON}" bundle.py

echo ">>> Plotting score distributions"
"${PYTHON}" plot_score.py --output "${ROOT}/analysis/score_distributions.png"

echo ">>> Done. Plot written to analysis/score_distributions.png"
