#!/usr/bin/env bash
set -euo pipefail

# Reset shard state + force-roll all shard services for a game.
# Usage: dev-scripts/reset_game.sh <game_name>

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GAME="${GAME:-${1-}}"
if [[ -z "${GAME}" ]]; then
  echo "✗ Provide a game: dev-scripts/reset_game.sh <name>" >&2
  exit 1
fi

# --- Load configs (same pattern as deploy scripts) ---
# Global config (required)
source "${SCRIPT_DIR}/config.env"
# Engine config (optional)
if [[ -f "${SCRIPT_DIR}/engine/config.env" ]]; then
  source "${SCRIPT_DIR}/engine/config.env"
fi
# Game-specific config (required)
source "${SCRIPT_DIR}/games/${GAME}/config.env"

# --- Required vars ---
: "${PROJECT_ID:?Missing PROJECT_ID}"
: "${REGION:?Missing REGION}"

# --- Derivations aligned with deploy.sh ---
SERVICE_PREFIX="${GAME//_/-}-shard"                 # e.g., spoiled-broth-shard
STATE_BUCKET="${STATE_BUCKET:-${PROJECT_ID}-${GAME}-state}"
STATE_PREFIX="${STATE_PREFIX:-shards}"               # we only delete under this prefix

echo "▶ Resetting game: ${GAME}"
echo "  Project: ${PROJECT_ID}"
echo "  Region:  ${REGION}"
echo "  Shard svc prefix: ${SERVICE_PREFIX}-"
echo "  State path: gs://${STATE_BUCKET}/${STATE_PREFIX}/**"

# --- 1) Delete shard state under the prefix only ---
if gsutil ls -b "gs://${STATE_BUCKET}" >/dev/null 2>&1; then
  gsutil -m rm -r "gs://${STATE_BUCKET}/${STATE_PREFIX}/**" || true
else
  echo "  (state bucket gs://${STATE_BUCKET} not found; skipping delete)"
fi

# --- 2) Force-roll all shard services (drop all sockets) ---
SHARD_SERVICES="$(
  gcloud run services list \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --platform managed \
    --filter="metadata.name~^${SERVICE_PREFIX}-" \
    --format="value(metadata.name)" | awk 'NF'
)"

if [[ -z "${SHARD_SERVICES}" ]]; then
  echo "  (no shard services found matching ^${SERVICE_PREFIX}-)"
else
  STAMP="$(date +%s)"
  echo "  Rolling shard services:"
  while IFS= read -r svc; do
    [[ -z "${svc}" ]] && continue
    echo "   - ${svc}"
    gcloud run services update "${svc}" \
      --project "${PROJECT_ID}" \
      --region "${REGION}" \
      --platform managed \
      --set-env-vars="ROLL=${STAMP}" >/dev/null

    gcloud run services update-traffic "${svc}" \
      --project "${PROJECT_ID}" \
      --region "${REGION}" \
      --platform managed \
      --to-latest >/dev/null
  done <<< "${SHARD_SERVICES}"
fi

echo "✔ Reset complete."
