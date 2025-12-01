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
: "${PKG_NAME:?PKG_NAME missing in dev-scripts/engine/config.env}"

echo "🔧 Building ${PKG_NAME} from ${ENGINE_DIR} (repo: ${REPO_ROOT})"
python -m pip install -U pip build setuptools setuptools-scm >/dev/null

pushd "${REPO_ROOT}/${ENGINE_DIR}" >/dev/null
python -m build

WHEEL_PATH="$(ls -t dist/${PKG_NAME//-/_}-*.whl 2>/dev/null | head -n1 || true)"
if [[ -z "${WHEEL_PATH}" ]]; then
  echo "✗ No wheel in ${REPO_ROOT}/${ENGINE_DIR}/dist" >&2; exit 1
fi

ENGINE_VERSION="$(python - <<'PY'
import pathlib, re
wheels = sorted(pathlib.Path("dist").glob("*.whl"))
name = wheels[-1].name if wheels else ""
m = re.search(r"(.+?)-([0-9][^ -]*)-.*\.whl$", name)
print(m.group(2) if m else "")
PY
)"
if [[ -z "${ENGINE_VERSION}" ]]; then
  echo "✗ Could not parse version from ${WHEEL_PATH}" >&2; exit 1
fi

ABS_WHEEL="${REPO_ROOT}/${ENGINE_DIR}/${WHEEL_PATH}"
popd >/dev/null

echo "✅ Built: ${ABS_WHEEL}"
echo "   Version: ${ENGINE_VERSION}"
