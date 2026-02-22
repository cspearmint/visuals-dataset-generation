#!/usr/bin/env bash
set -euo pipefail

# Run the HiKER augmentation pipeline inside the hiker-augment container.
# Builds the image if it doesn't exist. Pass through any args to the Python script.

IMAGE="hiker-augment"
SCRIPT_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building Docker image $IMAGE ..."
  docker build -t "$IMAGE" "$SCRIPT_DIR"
fi

# Build docker run command
DOCKER_CMD=(docker run --rm -v "$REPO_ROOT:/workspace" -w /workspace/visuals_dataset)

# Only add gcloud mount if the directory exists and is accessible
if [ -d "$HOME/.config/gcloud" ]; then
  DOCKER_CMD+=(
    -v "$HOME/.config/gcloud:/root/.config/gcloud"
    -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json
  )
fi

# Run the container
"${DOCKER_CMD[@]}" "$IMAGE" python generate_dataset.py "${SCRIPT_ARGS[@]}"
