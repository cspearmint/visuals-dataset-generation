# Lightweight environment to run load_waymo_images.py with Waymo + TF 2.11
# Usage:
#   docker build -t waymo-runner .
#   docker run --rm -it \
#     -v "$PWD":/workspace \
#     -w /workspace \
#     -v "$HOME/.config/gcloud":/root/.config/gcloud \
#     -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
#     waymo-runner \
#     python load_waymo_images.py --count 10 --max-files 3

FROM tensorflow/tensorflow:2.11.0

WORKDIR /workspace

# Install deps once; cached unless requirements change.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Default command can be overridden; keep interactive container by default.
CMD ["/bin/bash"]
