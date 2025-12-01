#!/usr/bin/env bash
set -euo pipefail

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GAME="${GAME:-${1-}}"
if [[ -z "${GAME}" ]]; then
  echo "✗ Provide a game: dev-scripts/deploy_lobby.sh <name>" >&2
  exit 1
fi

# Load configs
source "${SCRIPT_DIR}/config.env"
source "${SCRIPT_DIR}/games/${GAME}/config.env"

# Required
: "${PROJECT_ID:?Missing PROJECT_ID}"
: "${REGION:?Missing REGION}"
: "${LOBBY_CPU:?Missing LOBBY_CPU}"
: "${LOBBY_MEMORY:?Missing LOBBY_MEMORY}"
: "${LOBBY_CONCURRENCY:?Missing LOBBY_CONCURRENCY}"

# Derive names
LOBBY_DIR="${REPO_ROOT}/lobby"
LOBBY_SERVICE_NAME="${GAME//_/-}-game-lobby"
# Use Artifact Registry docker repo if you have one, else fallback to gcr.io
IMAGE_REPO="${IMAGE_REPO:-gcr.io/${PROJECT_ID}}"
LOBBY_IMAGE="${IMAGE_REPO}/${GAME//_/-}-lobby:latest"

SHARD_SERVICE_PREFIX="${GAME//_/-}-shard"

STATE_BUCKET="${STATE_BUCKET:-${PROJECT_ID}-${GAME}-state}"
STATE_PREFIX="${STATE_PREFIX:-shards}"

# SA for runtime (default)
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
CR_RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Discover shard URLs
echo "ℹ️  Discovering shard services with prefix '${SHARD_SERVICE_PREFIX}-' in ${REGION}..."
SHARD_URLS_CSV="$(
  gcloud run services list --project "${PROJECT_ID}" --region "${REGION}" \
    --filter="metadata.name~^${SHARD_SERVICE_PREFIX}-" \
    --format="value(status.url)" | awk 'NF' | paste -sd, -
)"
if [[ -z "${SHARD_URLS_CSV}" ]]; then
  echo "✗ No shards found. Deploy shards first or set SHARD_URLS." >&2
  exit 1
fi
echo "✅ Using shards: ${SHARD_URLS_CSV}"

# Ensure state bucket exists and lobby can read it
if ! gsutil ls -b "gs://${STATE_BUCKET}" >/dev/null 2>&1; then
  gsutil mb -l "${REGION}" "gs://${STATE_BUCKET}"
fi
# read-only is enough for lobby
gsutil iam ch "serviceAccount:${CR_RUNTIME_SA}:roles/storage.objectViewer" "gs://${STATE_BUCKET}" >/dev/null || true

# Build lobby image
echo "🛠️  Building lobby image: ${LOBBY_IMAGE}"
gcloud builds submit "${LOBBY_DIR}" --project "${PROJECT_ID}" --tag "${LOBBY_IMAGE}" --quiet

# Deploy lobby
echo "🚀 Deploying lobby ${LOBBY_SERVICE_NAME} …"
# Use alt delimiter so commas in SHARD_URLS don't break parsing
gcloud run deploy "${LOBBY_SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --image "${LOBBY_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --cpu "${LOBBY_CPU}" \
  --memory "${LOBBY_MEMORY}" \
  --concurrency "${LOBBY_CONCURRENCY}" \
  --max-instances=1 \
  --min-instances=1 \
  --set-env-vars="^|^SHARD_URLS=${SHARD_URLS_CSV}|SHARD_STATE_GS_URL=gs://${STATE_BUCKET}/${STATE_PREFIX}" \
  --quiet

LOBBY_URL="$(gcloud run services describe "${LOBBY_SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format="value(status.url)")"
echo "✅ Lobby deployed: ${LOBBY_URL}"
