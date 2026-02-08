# HiKER-based augmentation pipeline with Waymo dataset support
# Handles all corruptions: snow, frost, fog, rain, sunglare
# Usage:
#   docker build -t hiker-augment .
#   docker run --rm -it \
#     -v "$PWD":/workspace \
#     -w /workspace \
#     -v "$HOME/.config/gcloud":/root/.config/gcloud \
#     -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
#     hiker-augment \
#     python load_waymo_images.py --count 10 --max-files 3 --output-dir output

FROM tensorflow/tensorflow:2.11.0

WORKDIR /workspace

# Install system dependencies (ImageMagick for Wand, X11 for arcade rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    imagemagick \
    libmagickwand-dev \
    xvfb \
    libgl1-mesa-glx \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Default command can be overridden; keep interactive container by default.
CMD ["/bin/bash"]
