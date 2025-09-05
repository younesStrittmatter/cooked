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

LOBBY_DIR="${LOBBY_DIR:-${REPO_ROOT}/lobby}"
LOBBY_SERVICE_NAME="${LOBBY_SERVICE_NAME:-game-lobby}"
LOBBY_IMAGE="${LOBBY_IMAGE:-gcr.io/${PROJECT_ID}/game-lobby:latest}"
LOBBY_CPU="${LOBBY_CPU:-1}"
LOBBY_MEMORY="${LOBBY_MEMORY:-256Mi}"
LOBBY_CONCURRENCY="${LOBBY_CONCURRENCY:-80}"
LOBBY_MIN_INSTANCES="${LOBBY_MIN_INSTANCES:-1}"

SHARD_SERVICE_PREFIX="${SHARD_SERVICE_PREFIX:-game-shard}"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "✗ No GCP project set. Run: gcloud config set project <PROJECT_ID>"; exit 1
fi
if [[ ! -d "${LOBBY_DIR}" || ! -f "${LOBBY_DIR}/Dockerfile" ]]; then
  echo "✗ Missing lobby Dockerfile at ${LOBBY_DIR}/Dockerfile"; exit 1
fi

# Helper: join lines by comma (portable to macOS bash 3.2)
join_by_comma() {
  # prefer paste -sd, if available; fallback to tr|sed
  if command -v paste >/dev/null 2>&1; then
    paste -sd, -
  else
    tr '\n' ',' | sed 's/,$//'
  fi
}

# --- Discover shard URLs unless provided ---
if [[ -z "${SHARD_URLS:-}" ]]; then
  echo "ℹ️  Discovering shard services with prefix '${SHARD_SERVICE_PREFIX}-' in ${REGION}..."
  SHARD_URLS_CSV="$(
    gcloud run services list --region "$REGION" \
      --filter="metadata.name~^${SHARD_SERVICE_PREFIX}-" \
      --format="value(status.url)" \
    | awk 'NF' \
    | join_by_comma
  )"

  if [[ -z "${SHARD_URLS_CSV}" ]]; then
    echo "✗ No shards found. Set SHARD_URLS env var or deploy shards first."; exit 1
  fi
else
  SHARD_URLS_CSV="${SHARD_URLS}"
fi

echo "✅ Using shards: ${SHARD_URLS_CSV}"

# --- Build lobby image ONLY ---
echo "🛠️  Building lobby image: ${LOBBY_IMAGE}"
gcloud builds submit "${LOBBY_DIR}" --tag "${LOBBY_IMAGE}"

# --- Deploy lobby with SHARD_URLS (via env-file to avoid quoting hell) ---


echo "🚀 Deploying lobby ${LOBBY_SERVICE_NAME} …"
gcloud run deploy "${LOBBY_SERVICE_NAME}" \
  --image "${LOBBY_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --cpu "${LOBBY_CPU}" \
  --memory "${LOBBY_MEMORY}" \
  --concurrency "${LOBBY_CONCURRENCY}" \
  --max-instances=1 \
  --min-instances=1 \
  --set-env-vars="^|^SHARD_URLS=${SHARD_URLS_CSV}" \
  --quiet




LOBBY_URL="$(gcloud run services describe "${LOBBY_SERVICE_NAME}" --region "${REGION}" --format="value(status.url)")"
echo "✅ Lobby deployed: ${LOBBY_URL}"
echo "   Health: ${LOBBY_URL}/healthz"
echo "   New:    ${LOBBY_URL}/new"
