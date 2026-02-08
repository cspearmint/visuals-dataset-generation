#!/usr/bin/env bash
set -euo pipefail

# Run the HiKER augmentation pipeline inside the hiker-augment container.
# Builds the image if it doesn't exist. Pass through any args to the Python script.

IMAGE="hiker-augment"
SCRIPT_ARGS=("$@")

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building Docker image $IMAGE ..."
  docker build -t "$IMAGE" .
fi

# Build docker run command
DOCKER_CMD="docker run --rm -v $PWD:/workspace -w /workspace"

# Only add gcloud mount if the directory exists and is accessible
if [ -d "$HOME/.config/gcloud" ]; then
  DOCKER_CMD="$DOCKER_CMD -v $HOME/.config/gcloud:/root/.config/gcloud -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json"
fi

# Run the container
$DOCKER_CMD "$IMAGE" python load_waymo_images.py "${SCRIPT_ARGS[@]}"
