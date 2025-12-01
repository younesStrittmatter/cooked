#!/usr/bin/env bash
set -euo pipefail

GAME="${GAME:-${1-}}"
if [[ -z "${GAME}" ]]; then
  echo "✗ Provide a game: dev-scripts/deploy_lobby.sh <name>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Deploying game ${GAME}..."
"${SCRIPT_DIR}/deploy_game.sh" "${GAME}"

echo "Deploying lobby for ${GAME}..."
"${SCRIPT_DIR}/deploy_lobby.sh" "${GAME}"

