#!/bin/bash

# Get the absolute path to this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Load config from env file if it exists
if [ -f "$SCRIPT_DIR/config.env" ]; then
  source "$SCRIPT_DIR/config.env"
fi

# Defaults
SERVICE_NAME=${SERVICE_NAME:-overcooked-server}
REGION=${REGION:-us-central1}

echo "Creating requirements.txt..."

"$SCRIPT_DIR/build-requirements.sh"

echo "🚀 Deploying $SERVICE_NAME to Cloud Run in region $REGION..."

gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated