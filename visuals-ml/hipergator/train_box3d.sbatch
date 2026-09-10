#!/bin/bash
#
# Train the box3d baseline (amodal 3D box regressor) on HiPerGator.
#
# Conventions match train_monodetr.sbatch / train_positionnet.sbatch: GROUP and
# HPG_USER from the environment, CONFIG overridable, everything inside train.sif.
#
# Runs:
#   1. baselines.data.build_detection_index  (skipped if the index exists)
#   2. baselines.train --config configs/box3d.yaml
#   3. baselines.eval  --config configs/box3d.yaml  (--report)
#
# Split: by SEGMENT, with ALL 10 weather variants on both sides. A training
# segment contributes every one of its renderings; the held-out segments are
# entirely different drives, scored across every rendering. This mirrors the
# data the visuals agent will be trained on. Compute is cut with frame_stride
# (~10 Hz near-duplicates), never by dropping weathers.
#
# Usage:
#   export GROUP=<your-group> HPG_USER=$USER
#   sbatch --account=$GROUP --qos=$GROUP train_box3d.sbatch
#
#SBATCH --job-name=box3d
# NOTE: #SBATCH lines are read by SLURM before the shell runs, so they get NO
# parameter expansion. Either edit the placeholders below, or override them at
# submit time with `sbatch --account=... --qos=...` (flags win over these lines).
#SBATCH --account=<GROUP>
#SBATCH --qos=<GROUP>
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/box3d_%j.out
#SBATCH --error=logs/box3d_%j.err

set -euo pipefail

# ---- Configuration ----------------------------------------------------------
: "${GROUP:?set me - your HiPerGator group, e.g. export GROUP=ufhpc}"
: "${HPG_USER:?set me - your HiPerGator username, e.g. export HPG_USER=\$USER}"

REPO_ROOT=${REPO_ROOT:-/blue/${GROUP}/${HPG_USER}/visuals-dataset-generation}
DATASET_OUTPUT=${DATASET_OUTPUT:-/blue/${GROUP}/${HPG_USER}/waymo/output}
SIF=${SIF:-/blue/${GROUP}/${HPG_USER}/waymo/train.sif}
CONFIG=${CONFIG:-configs/box3d.yaml}
EVAL_CONFIG=${EVAL_CONFIG:-configs/box3d.yaml}
CAMERAS=${CAMERAS:-1}
# -----------------------------------------------------------------------------

mkdir -p logs reports

module load singularity

echo "[info] Host: $(hostname)"
echo "[info] Dataset output: ${DATASET_OUTPUT}"
echo "[info] SIF: ${SIF}"
echo "[info] Train config: ${CONFIG}   Eval config: ${EVAL_CONFIG}"

singularity exec --nv \
    --bind /blue/"${GROUP}" \
    --bind "${REPO_ROOT}":/workspace \
    "${SIF}" bash -lc "
set -euo pipefail
cd /workspace/visuals-ml

# Preflight the backbone weights. box3d builds TWO resnet18 encoders with
# weights=ResNet18_Weights.DEFAULT (IMAGENET1K_V1). Compute nodes have no
# internet and the SIF is read-only, so a cache miss surfaces as a confusing
# 'OSError: Read-only file system' several minutes in. Fail here, with the fix.
python -c \"
import os, warnings, torch, torchvision
warnings.filterwarnings('ignore')
print('[info] TORCH_HOME =', os.environ.get('TORCH_HOME', '<unset>'))
from torchvision.models import resnet18, ResNet18_Weights
try:
    resnet18(weights=ResNet18_Weights.DEFAULT)
except Exception as e:
    print('[error] resnet18 IMAGENET1K_V1 not in baked cache:', type(e).__name__, e)
    print('[error] rebuild train.sif; train.def must bake resnet18 too')
    raise SystemExit(1)
print('[info] resnet18 IMAGENET1K_V1 served from baked cache')
\"

INDEX=data/output/det_records.jsonl
if [ ! -f \"\$INDEX\" ]; then
    echo '[info] Building detection index'
    python -m baselines.data.build_detection_index \
        --source-dir '${DATASET_OUTPUT}' \
        --index-file \"\$INDEX\" \
        --cameras '${CAMERAS}'
else
    echo '[info] Reusing existing index'
fi

echo '[info] Training box3d (clear only)'
python -m baselines.train --config '${CONFIG}'

echo '[info] Evaluating best checkpoint across all 10 weather variants'
python -m baselines.eval --config '${EVAL_CONFIG}' \
    --report reports/final_box3d.json

echo '[info] === split fingerprints (the train and eval val= hashes MUST match) ==='
grep -h 'Group split fingerprint' /workspace/visuals-ml/logs/box3d_*.out 2>/dev/null | tail -4 || true
"

echo "[info] box3d done."
