#!/usr/bin/env bash
set -euo pipefail

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Load config (optional) ---
if [[ -f "${SCRIPT_DIR}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/config.env"
fi

# --- Config + defaults ---
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"

# Shards
SHARD_IMAGE="${SHARD_IMAGE:-gcr.io/${PROJECT_ID}/game-shard:latest}"
SHARD_SERVICE_PREFIX="${SHARD_SERVICE_PREFIX:-game-shard}"
SHARD_COUNT="${SHARD_COUNT:-5}"
SHARD_CPU="${SHARD_CPU:-2}"
SHARD_MEMORY="${SHARD_MEMORY:-1Gi}"
SHARD_CONCURRENCY="${SHARD_CONCURRENCY:-200}"
SHARD_MIN_INSTANCES="${SHARD_MIN_INSTANCES:-1}"
SHARD_MAX_INSTANCES="${SHARD_MAX_INSTANCES:-1}"  # keep 1 to ensure in-proc state
SHARD_APP_PATH="${SHARD_APP_PATH:-wsgi:app}"

# Lobby
LOBBY_DIR="${LOBBY_DIR:-${REPO_ROOT}/lobby}"          # must contain lobby Dockerfile + main.py
LOBBY_IMAGE="${LOBBY_IMAGE:-gcr.io/${PROJECT_ID}/game-lobby:latest}"
LOBBY_SERVICE_NAME="${LOBBY_SERVICE_NAME:-game-lobby}"
LOBBY_CPU="${LOBBY_CPU:-1}"
LOBBY_MEMORY="${LOBBY_MEMORY:-256Mi}"
LOBBY_CONCURRENCY="${LOBBY_CONCURRENCY:-80}"

# --- Sanity checks ---
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "✗ No GCP project set. Run: gcloud config set project <PROJECT_ID>"; exit 1
fi
if [[ ! -f "${REPO_ROOT}/Dockerfile" ]]; then
  echo "✗ Missing Dockerfile for shard at repo root (${REPO_ROOT}/Dockerfile)"; exit 1
fi
if [[ ! -d "${LOBBY_DIR}" || ! -f "${LOBBY_DIR}/Dockerfile" ]]; then
  echo "✗ Missing lobby Dockerfile at ${LOBBY_DIR}/Dockerfile"; exit 1
fi

# --- Build shard image ---
echo "🛠️  Building shard image: ${SHARD_IMAGE}"
gcloud builds submit "${REPO_ROOT}" --tag "${SHARD_IMAGE}" \
  --gcs-log-dir="gs://$PROJECT_ID-cloudbuild-logs" 2>/dev/null || \
gcloud builds submit "${REPO_ROOT}" --tag "${SHARD_IMAGE}"

# --- Deploy shards ---
SHARD_URLS=()
for i in $(seq 1 "${SHARD_COUNT}"); do
  SVC="${SHARD_SERVICE_PREFIX}-${i}"
  echo "🚀 Deploying shard ${SVC} (${i}/${SHARD_COUNT}) …"
  gcloud run deploy "${SVC}" \
    --image "${SHARD_IMAGE}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --cpu "${SHARD_CPU}" \
    --memory "${SHARD_MEMORY}" \
    --concurrency "${SHARD_CONCURRENCY}" \
    --min-instances "${SHARD_MIN_INSTANCES}" \
    --max-instances "${SHARD_MAX_INSTANCES}" \
    --timeout 3600 \
    --set-env-vars "APP_PATH=${SHARD_APP_PATH},MODE=single,SOCKIO_FORCE_BASE=1,SOCKIO_COMPRESS=0,FLASK_SKIP_DOTENV=1" \
    --quiet

  URL="$(gcloud run services describe "${SVC}" --region "${REGION}" --format="value(status.url)")"
  SHARD_URLS+=("${URL}")
done

SHARD_URLS_CSV="$(IFS=, ; echo "${SHARD_URLS[*]}")"
echo "✅ Shards ready:"
printf '   - %s\n' "${SHARD_URLS[@]}"

# --- Build lobby image ---
echo "🛠️  Building lobby image: ${LOBBY_IMAGE}"
gcloud builds submit "${LOBBY_DIR}" --tag "${LOBBY_IMAGE}" \
  --gcs-log-dir="gs://$PROJECT_ID-cloudbuild-logs" 2>/dev/null || \
gcloud builds submit "${LOBBY_DIR}" --tag "${LOBBY_IMAGE}"

# --- Deploy lobby with SHARD_URLS ---
echo
echo "🎉 Done!"
