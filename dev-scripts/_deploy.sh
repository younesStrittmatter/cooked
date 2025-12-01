#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GAME=spoiled_broth dev-scripts/deploy.sh
#   dev-scripts/deploy.sh spoiled_broth

# ───────────────────────── Paths & inputs ─────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GAME="${GAME:-${1-}}"
if [[ -z "${GAME}" ]]; then
  echo "✗ Provide a game: dev-scripts/deploy.sh <name>" >&2
  exit 1
fi

# Optional: re-release engine before building images (set RE_RELEASE_ENGINE=0 to skip)
RE_RELEASE_ENGINE="${RE_RELEASE_ENGINE:-1}"

# ───────────────────────── Load configs ──────────────────────────
# Global project config
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config.env"

# Per-game config (dev-scripts/games/<GAME>/config.env)
GAME_CFG="${SCRIPT_DIR}/games/${GAME}/config.env"
if [[ ! -f "${GAME_CFG}" ]]; then
  echo "✗ Missing per-game config: ${GAME_CFG}" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${GAME_CFG}"

# Required globals
: "${PROJECT_ID:?Missing PROJECT_ID in dev-scripts/config.env}"
: "${REGION:?Missing REGION in dev-scripts/config.env}"
: "${PY_REPO:?Missing PY_REPO in dev-scripts/config.env}"
: "${IMAGE_REPO:?Missing IMAGE_REPO in dev-scripts/config.env}"   # e.g. us-central1-docker.pkg.dev/${PROJECT_ID}/games

# Required per-game
: "${GAME_DIR:?Missing GAME_DIR in per-game config}"              # e.g. games/spoiled_broth
: "${ENGINE_VERSION:?Missing ENGINE_VERSION in per-game config}"  # e.g. 0.0.0.dev0 or "latest"
: "${SERVICE_PREFIX:?Missing SERVICE_PREFIX in per-game config}"  # e.g. sb-shard
: "${SHARD_COUNT:?}"
: "${SHARD_CPU:?}"
: "${SHARD_MEMORY:?}"
: "${SHARD_CONCURRENCY:?}"
: "${SHARD_MIN_INSTANCES:?}"
: "${SHARD_MAX_INSTANCES:?}"

# Optional gameplay/env knobs
GAME_CAPACITY="${GAME_CAPACITY:-2}"

# ─────────────── Service accounts (project-scoped) ────────────────
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
CR_PULL_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"   # Cloud Run runtime SA (default)

# ───────────────────── Enable required services ──────────────────
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  --project "${PROJECT_ID}" >/dev/null

# ─────────────────── Ensure AR docker repo exists ────────────────
DOCKER_REPO="${IMAGE_REPO##*/}"   # last segment (e.g. "games")
if ! gcloud artifacts repositories describe "${DOCKER_REPO}" \
     --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  echo "📦 Creating Artifact Registry (docker) repo: ${DOCKER_REPO} in ${REGION}"
  gcloud artifacts repositories create "${DOCKER_REPO}" \
    --repository-format=docker \
    --location "${REGION}" \
    --project "${PROJECT_ID}"
fi

# ─────────────── Grant AR access to build/pull SAs ───────────────
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/artifactregistry.writer" >/dev/null || true

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CR_PULL_SA}" \
  --role="roles/artifactregistry.reader" >/dev/null || true

# ───────────────────────── Assets (sprites) ───────────────────────
ASSET_BUCKET="${ASSET_BUCKET:-${PROJECT_ID}-${GAME}-assets}"
ASSET_VERSION="$(date +%Y%m%d%H%M%S)"

# 1) create bucket if missing
if ! gsutil ls -b "gs://${ASSET_BUCKET}" >/dev/null 2>&1; then
  gsutil mb -l "${REGION}" "gs://${ASSET_BUCKET}"
  # Public read (for simple static hosting from GCS)
  gsutil iam ch allUsers:objectViewer "gs://${ASSET_BUCKET}"
fi

# 2) upload sprites (game folder → gs://BUCKET/sprites)
gsutil -m rsync -r "${REPO_ROOT}/${GAME_DIR}/static/sprites" "gs://${ASSET_BUCKET}/sprites"

# 3) env to point shards at the asset bucket
ASSET_BASE_URL="https://storage.googleapis.com/${ASSET_BUCKET}/"

# ─────────────── State bucket (shared across shards) ──────────────
STATE_BUCKET="${STATE_BUCKET:-${PROJECT_ID}-${GAME}-state}"
STATE_PREFIX="${STATE_PREFIX:-shards}"   # objects go under gs://STATE_BUCKET/STATE_PREFIX/<SHARD_ID>/state/state.json

# Ensure state bucket exists once
if ! gsutil ls -b "gs://${STATE_BUCKET}" >/dev/null 2>&1; then
  gsutil mb -l "${REGION}" "gs://${STATE_BUCKET}"
fi
# Grant the Cloud Run runtime SA write access to the state bucket
gsutil iam ch "serviceAccount:${CR_PULL_SA}:roles/storage.objectAdmin" "gs://${STATE_BUCKET}" >/dev/null || true

# ─────────────── (Optional) engine re-release ─────────────────────
if [[ "${RE_RELEASE_ENGINE}" == "1" ]]; then
  if [[ -x "${SCRIPT_DIR}/engine/release.sh" ]]; then
    echo "♻️  Re-releasing cooked-engine (only bumps/builds if changes)…"
    "${SCRIPT_DIR}/engine/release.sh"
    ENGINE_VERSION="latest"
  else
    echo "⚠️  ${SCRIPT_DIR}/engine/release.sh not found or not executable; skipping re-release."
  fi
fi

# ─────────────── Resolve ENGINE_VERSION if "latest" ───────────────
if [[ "${ENGINE_VERSION}" == "latest" ]]; then
  echo "🔎 Resolving latest cooked-engine version from Artifact Registry…"
  ENGINE_VERSION="$(
    gcloud artifacts versions list \
      --project "${PROJECT_ID}" \
      --location "${REGION}" \
      --repository "${PY_REPO}" \
      --package "cooked-engine" \
      --format="value(name)" \
      --sort-by="~createTime" \
      --limit=1 | sed 's#.*/##'
  )"
  if [[ -z "${ENGINE_VERSION}" ]]; then
    echo "✗ Could not resolve latest cooked-engine version." >&2
    echo "  Tip: verify with:" >&2
    echo "       gcloud artifacts versions list --project \"${PROJECT_ID}\" --location \"${REGION}\" --repository \"${PY_REPO}\" --package cooked-engine" >&2
    exit 1
  fi
  echo "   → using ${ENGINE_VERSION}"
fi

# ─────────────── Download engine wheel locally (once) ─────────────
python -m pip install -U pip keyring keyrings.google-artifactregistry-auth >/dev/null

WHEELS_DIR="${REPO_ROOT}/${GAME_DIR}/wheels"
mkdir -p "${WHEELS_DIR}"

echo "⬇️  Downloading cooked-engine==${ENGINE_VERSION} wheel…"
pip download --no-deps \
  --dest "${WHEELS_DIR}" \
  --extra-index-url "https://${REGION}-python.pkg.dev/${PROJECT_ID}/${PY_REPO}/simple" \
  "cooked-engine==${ENGINE_VERSION}"

if ! ls "${WHEELS_DIR}"/cooked_engine-*.whl >/dev/null 2>&1; then
  echo "✗ Could not download cooked-engine==${ENGINE_VERSION} from Artifact Registry." >&2
  echo "  Check PY_REPO/REGION/PROJECT_ID and that this version is published." >&2
  exit 1
fi

# ───────────────────────── Build image ────────────────────────────
IMAGE_NAME="${IMAGE_NAME:-${IMAGE_REPO}/${SERVICE_PREFIX}}"
TAG="${IMAGE_NAME}:${ENGINE_VERSION}"

echo "🛠️  Building ${TAG} (context: ${GAME_DIR})"
gcloud builds submit "${REPO_ROOT}/${GAME_DIR}" \
  --project "${PROJECT_ID}" \
  --tag "${TAG}" \
  --quiet

# Clean wheels (optional)
rm -rf "${WHEELS_DIR}"

# ───────────────────────── Deploy shards ──────────────────────────
SHARD_URLS=()
for i in $(seq 1 "${SHARD_COUNT}"); do
  SVC="${SERVICE_PREFIX}-${i}"
  SHARD_ID="${i}"

  STATE_ENV="SHARD_ID=${SHARD_ID},SHARD_STATE_GS_URL=gs://${STATE_BUCKET}/${STATE_PREFIX},GAME_CAPACITY=${GAME_CAPACITY}"

  echo "🚀 Deploying shard ${SVC} → ${TAG}"
  gcloud run deploy "${SVC}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --image "${TAG}" \
    --platform managed \
    --allow-unauthenticated \
    --cpu "${SHARD_CPU}" \
    --memory "${SHARD_MEMORY}" \
    --concurrency "${SHARD_CONCURRENCY}" \
    --min-instances "${SHARD_MIN_INSTANCES}" \
    --max-instances "${SHARD_MAX_INSTANCES}" \
    --timeout 3600 \
    --set-env-vars "FLASK_SKIP_DOTENV=1,SOCKIO_FORCE_BASE=1,SOCKIO_COMPRESS=0,ASSET_BASE_URL=${ASSET_BASE_URL},ASSET_VERSION=${ASSET_VERSION},${STATE_ENV}" \
    --quiet

  URL="$(gcloud run services describe "${SVC}" \
        --project "${PROJECT_ID}" --region "${REGION}" \
        --format='value(status.url)')"
  SHARD_URLS+=("${URL}")
done

echo "✅ Shards for ${GAME}:"
printf '   - %s\n' "${SHARD_URLS[@]}"
