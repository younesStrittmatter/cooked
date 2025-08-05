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
PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project)}

#echo "Creating requirements.txt..."
#"$SCRIPT_DIR/build-requirements.sh"

echo "🛠️ Submitting build to Cloud Build..."
gcloud builds submit --tag "gcr.io/$PROJECT_ID/$SERVICE_NAME" .

echo "🚀 Deploying $SERVICE_NAME to Cloud Run in region $REGION..."
gcloud run deploy "$SERVICE_NAME" \
  --image "gcr.io/$PROJECT_ID/$SERVICE_NAME" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated
