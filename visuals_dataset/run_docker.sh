#!/usr/bin/env bash
set -euo pipefail

# Run the Waymo image fetcher inside the waymo-runner container without an interactive shell.
# Builds the image if it doesn't exist. Pass through any args to the Python script.

IMAGE="waymo-runner"
SCRIPT_ARGS=("$@")

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building Docker image $IMAGE ..."
  docker build -t "$IMAGE" .
fi

docker run --rm \
  -v "$PWD":/workspace \
  -w /workspace \
  -v "$HOME/.config/gcloud":/root/.config/gcloud \
  -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
  "$IMAGE" \
  python load_waymo_images.py "${SCRIPT_ARGS[@]}"
