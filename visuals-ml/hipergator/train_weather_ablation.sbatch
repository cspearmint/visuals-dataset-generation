#!/bin/bash
#
# Weather ablation: how much weather diversity in TRAINING buys robustness
# across all weather at test time?
#
# 10 array tasks = 2 models x 5 training-weather sets. Every task uses the SAME
# segment split, the same frame_stride and the same all-weather held-out
# segments; the only thing that varies is which weather variants the model is
# allowed to train on.
#
#   task  model      trains on
#   ----  ---------  -------------------------------------------------
#     0   box3d      all 10                     (the reference run)
#     1   box3d      clear only
#     2   box3d      clear + rain, snow, fog, dust
#     3   box3d      clear + rain, fog
#     4   box3d      clear + 4 random alterations (seeded, logged)
#     5   monodetr   all 10                     (the reference run)
#     6   monodetr   clear only
#     7   monodetr   clear + rain, snow, fog, dust
#     8   monodetr   clear + rain, fog
#     9   monodetr   clear + 4 random alterations (seeded, logged)
#
# THE FILTER APPLIES TO TRAINING ONLY. Validation always keeps all 10 variants,
# on the same held-out segments, for every task -- that is what makes the runs
# comparable. Filtering before the split would give each ablation its own test
# set and the numbers could not be lined up against each other.
#
# Tasks 0 and 5 duplicate the standalone train_box3d.sbatch /
# train_monodetr.sbatch runs. Keep them: they are the control arm, produced by
# exactly the same code path as the ablations, so any difference is the
# ablation and not the plumbing. Skip them with --array=1-4,6-9 if those runs
# are already finished and you would rather not spend the GPUs twice.
#
# Usage:
#   export GROUP=<your-group> HPG_USER=$USER
#   sbatch --account=$GROUP --qos=$GROUP train_weather_ablation.sbatch
#   sbatch --account=$GROUP --qos=$GROUP --array=1-4 train_weather_ablation.sbatch   # box3d only
#
#SBATCH --job-name=weather-abl
#SBATCH --account=<GROUP>
#SBATCH --qos=<GROUP>
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=logs/wabl_%A_%a.out
#SBATCH --error=logs/wabl_%A_%a.err

set -euo pipefail

# ---- Configuration ----------------------------------------------------------
: "${GROUP:?set me - your HiPerGator group, e.g. export GROUP=ufhpc}"
: "${HPG_USER:?set me - your HiPerGator username, e.g. export HPG_USER=\$USER}"

REPO_ROOT=${REPO_ROOT:-/blue/${GROUP}/${HPG_USER}/visuals-dataset-generation}
DATASET_OUTPUT=${DATASET_OUTPUT:-/blue/${GROUP}/${HPG_USER}/waymo/output}
SIF=${SIF:-/blue/${GROUP}/${HPG_USER}/waymo/train.sif}
CAMERAS=${CAMERAS:-1}
# -----------------------------------------------------------------------------

IDX=${SLURM_ARRAY_TASK_ID:-0}

# model | config | weather spec | short tag for paths
MODELS=(box3d box3d box3d box3d box3d monodetr monodetr monodetr monodetr monodetr)
CONFIGS=(configs/box3d.yaml configs/box3d.yaml configs/box3d.yaml configs/box3d.yaml configs/box3d.yaml \
         configs/monodetr.yaml configs/monodetr.yaml configs/monodetr.yaml configs/monodetr.yaml configs/monodetr.yaml)
WEATHERS=(all clear clear,rain,snow,fog,dust clear,rain,fog random4 \
          all clear clear,rain,snow,fog,dust clear,rain,fog random4)
TAGS=(all clear clear4 clear2 rand4 all clear clear4 clear2 rand4)

MODEL="${MODELS[$IDX]}"
CONFIG="${CONFIGS[$IDX]}"
WEATHER="${WEATHERS[$IDX]}"
TAG="${TAGS[$IDX]}"

RUN="${MODEL}_w-${TAG}"
CKPT_DIR="baselines/checkpoints/${RUN}"
REPORT="reports/final_${RUN}.json"

mkdir -p logs reports

module load singularity

echo "[info] Host:       $(hostname)"
echo "[info] Task:       ${IDX}  ->  ${RUN}"
echo "[info] Model:      ${MODEL}   Config: ${CONFIG}"
echo "[info] Train on:   ${WEATHER}   (validation always = all 10)"
echo "[info] Checkpoint: ${CKPT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

singularity exec --nv \
    --bind /blue/"${GROUP}" \
    --bind "${REPO_ROOT}":/workspace \
    "${SIF}" bash -lc "
set -euo pipefail
cd /workspace/visuals-ml

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

echo '[info] Training ${RUN}  (train weathers: ${WEATHER})'
python -m baselines.train \
    --config '${CONFIG}' \
    --train-weathers '${WEATHER}' \
    --checkpoint-dir '${CKPT_DIR}'

echo '[info] Evaluating ${RUN} across ALL 10 weather variants'
python -m baselines.eval \
    --config '${CONFIG}' \
    --checkpoint-dir '${CKPT_DIR}' \
    --report '${REPORT}'
"

echo "[info] ${RUN} done -> ${REPORT}"
