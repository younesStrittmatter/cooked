#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Load configs (project first, then engine)
# shellcheck disable=SC1091
[[ -f "${SCRIPT_DIR}/../config.env" ]] && source "${SCRIPT_DIR}/../config.env"
# shellcheck disable=SC1091
[[ -f "${SCRIPT_DIR}/config.env" ]] && source "${SCRIPT_DIR}/config.env"

: "${PROJECT_ID:?PROJECT_ID missing in dev-scripts/config.env}"
: "${REGION:?REGION missing in dev-scripts/config.env}"
: "${PY_REPO:?PY_REPO missing in dev-scripts/config.env}"
: "${ENGINE_DIR:?ENGINE_DIR missing in dev-scripts/engine/config.env}"
: "${PKG_NAME:?PKG_NAME missing in dev-scripts/engine/config.env}"

DIST_DIR="${REPO_ROOT}/${ENGINE_DIR}/dist"
echo "🔎 Looking for wheel in: ${DIST_DIR}"

# Grab newest wheel regardless of basename (avoids subtle PKG_NAME mismatches/whitespace/etc.)
ENGINE_WHEEL="$(ls -t "${DIST_DIR}"/*.whl 2>/dev/null | head -n1 || true)"
if [[ -z "${ENGINE_WHEEL}" || ! -f "${ENGINE_WHEEL}" ]]; then
  echo "✗ No wheel found in ${DIST_DIR}. Run build.sh first." >&2
  exit 1
fi

# Parse version from wheel filename
ENGINE_VERSION="$(python -c 'import re,sys,pathlib as p; n=p.Path(sys.argv[1]).name; m=re.search(r"(.+?)-([0-9][^ -]*)-.*\.whl$", n); print(m.group(2) if m else "")' "${ENGINE_WHEEL}")"
if [[ -z "${ENGINE_VERSION}" ]]; then
  echo "✗ Could not parse version from ${ENGINE_WHEEL}" >&2
  exit 1
fi

echo "🧳 Wheel: ${ENGINE_WHEEL}"
echo "🔖 Version: ${ENGINE_VERSION}"
echo "🗂  Repository: ${PROJECT_ID}/${REGION}/${PY_REPO}"

# Ensure Artifact Registry (python) repo exists
if ! gcloud artifacts repositories describe "${PY_REPO}" --project "${PROJECT_ID}" --location "${REGION}" >/dev/null 2>&1; then
  echo "📦 Creating Artifact Registry (python) repo: ${PY_REPO} in ${REGION}"
  gcloud artifacts repositories create "${PY_REPO}" \
    --project "${PROJECT_ID}" \
    --repository-format=python \
    --location="${REGION}"
fi

# Install Twine + keyring + Artifact Registry keyring plugin (once per env)
python -m pip install -U twine keyring keyrings.google-artifactregistry-auth >/dev/null

# Upload to Artifact Registry (Python) via Twine
REPO_URL="https://${REGION}-python.pkg.dev/${PROJECT_ID}/${PY_REPO}/"
echo "🚀 Uploading ${PKG_NAME}==${ENGINE_VERSION} to ${REPO_URL}"
python -m twine upload \
  --repository-url "${REPO_URL}" \
  "${ENGINE_WHEEL}"

echo "✅ Published ${PKG_NAME}==${ENGINE_VERSION}"
echo "🔗 Extra index URL: https://${REGION}-python.pkg.dev/${PROJECT_ID}/${PY_REPO}/simple"
