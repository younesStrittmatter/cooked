#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Load configs (project first, then engine)
# shellcheck disable=SC1091
[[ -f "${SCRIPT_DIR}/../config.env" ]] && source "${SCRIPT_DIR}/../config.env"
# shellcheck disable=SC1091
[[ -f "${SCRIPT_DIR}/config.env" ]] && source "${SCRIPT_DIR}/config.env"

: "${ENGINE_DIR:?ENGINE_DIR missing in dev-scripts/engine/config.env}"

ENGINE_ABS="${REPO_ROOT}/${ENGINE_DIR}"
PYPROJECT="${ENGINE_ABS}/pyproject.toml"

python "${SCRIPT_DIR}"/bump_version.py -path "${PYPROJECT}"
# Build and deploy using your existing scripts
echo
"${SCRIPT_DIR}/build.sh"
echo
"${SCRIPT_DIR}/deploy.sh"

# Stamp the commit that most recently touched the engine and the version we just released
echo "✅ Release complete."
