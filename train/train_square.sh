#!/bin/bash
# Train the base diffusion policy for the `square` task.
#
#   conda activate reguide
#   bash train/train_square.sh

set -euo pipefail

# run from the repo root (this script lives in train/)
cd "$(dirname "$0")/.." || exit 1

# use the active environment's python (override with: PYTHON=/path/to/python bash ...)
PYTHON=${PYTHON:-python}

export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true
export WANDB_START_METHOD=thread
export WANDB_PROJECT=reguide
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="/tmp/$USER"
mkdir -p "$TMPDIR"

TASK="square"
NUM_DEMOS="30_demos"
CATEGORY="base_policy"

DATASET="path/to/training_data.hdf5"
OUTPUT_ROOT="path/to/output_dir"

echo "Using: TASK=$TASK NUM_DEMOS=$NUM_DEMOS CATEGORY=$CATEGORY DATASET=$DATASET"

"$PYTHON" train.py \
  --config-dir=. --config-name="image_${TASK}_diffusion_policy_cnn.yaml" \
  training.seed=42 training.device=cuda:0 \
  logging.mode=offline \
  task.dataset.dataset_path="$DATASET" \
  task.env_runner.dataset_path="$DATASET" \
  task.dataset_path="$DATASET" \
  hydra.run.dir="${OUTPUT_ROOT}/\${now:%Y.%m.%d}/\${now:%H.%M.%S}" \
  logging.name="\${now:%Y.%m.%d}_\${now:%H.%M.%S}_train_diffusion_unet_hybrid_square_image_${NUM_DEMOS}_${CATEGORY}_final" \
  training.num_epochs=600
