#!/bin/bash
# Train the base diffusion policy for the LIBERO 10-task suite.
#
#   conda activate reguide          # LIBERO must also be installed / importable
#   bash train/train_libero.sh
#
# LIBERO trains on a *directory* of per-task hdf5 files (the loader globs them).

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
# LIBERO must be importable. If you have a local LIBERO checkout, set LIBERO_ROOT.
export PYTHONPATH="${LIBERO_ROOT:+$LIBERO_ROOT:}$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="/tmp/$USER"
mkdir -p "$TMPDIR"

# ---- experiment knobs ----
TASK="libero"            # matches image_libero_diffusion_policy_cnn.yaml
NUM_ROLLOUT="500"
SAMPLE="evenly"          # evenly | weighted
CATEGORY="rollout"

# directory of per-task hdf5 files (NOT a single merged hdf5)
DATASET_DIR="path/to/libero/dataset_dir"
OUTPUT_ROOT="path/to/output_dir"

echo "Using: TASK=$TASK NUM_ROLLOUT=$NUM_ROLLOUT SAMPLE=$SAMPLE CATEGORY=$CATEGORY"
echo "DATASET_DIR=$DATASET_DIR"

# sanity: the directory must exist and contain the per-task hdf5 files
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: dataset directory not found: $DATASET_DIR" >&2
  exit 1
fi
NUM_HDF5=$(find "$DATASET_DIR" -maxdepth 1 -name "*_demo.hdf5" | wc -l)
echo "Found $NUM_HDF5 *_demo.hdf5 files in $DATASET_DIR"

"$PYTHON" train.py \
  --config-dir=. --config-name="image_${TASK}_diffusion_policy_cnn.yaml" \
  training.seed=42 training.device=cuda:0 \
  logging.mode=offline \
  task.dataset.dataset_path="$DATASET_DIR" \
  task.env_runner.dataset_path="$DATASET_DIR" \
  task.dataset_path="$DATASET_DIR" \
  hydra.run.dir="${OUTPUT_ROOT}/${CATEGORY}_${SAMPLE}_${NUM_ROLLOUT}/\${now:%Y.%m.%d}/\${now:%H.%M.%S}" \
  logging.name="\${now:%Y.%m.%d}_\${now:%H.%M.%S}_train_diffusion_unet_hybrid_libero10_image_${CATEGORY}_${SAMPLE}_${NUM_ROLLOUT}" \
  training.num_epochs=200
