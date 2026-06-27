#!/bin/bash
# PCG step (a): encode observations into latents for a robomimic task.
#
#   conda activate reguide
#   bash pcg_data_generation/extract_latent.sh

set -eo pipefail

# run from the repo root (this script lives in pcg_data_generation/)
cd "$(dirname "$0")/.." || exit 1

# use the active environment's python (override with: PYTHON=/path/to/python bash ...)
PYTHON=${PYTHON:-python}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

TASK="transport"        # can | square | transport | tool_hang

# ---- paths ----
DATA_PATH="path/to/training_data.hdf5"
OUTPUT_FILE="path/to/encoded_obs_latents.hdf5"
DYN_CKPT="path/to/dyn_model.pth"
POLICY_CKPT="path/to/policy.ckpt"

echo "Using:"
echo "  TASK=$TASK"
echo "  DATA_PATH=$DATA_PATH"
echo "  DYN_CKPT=$DYN_CKPT"
echo "  POLICY_CKPT=$POLICY_CKPT"
echo "  OUTPUT_FILE=$OUTPUT_FILE"

"$PYTHON" extract_latents.py \
  --dynamics_ckpt "$DYN_CKPT" \
  --policy_ckpt_path "$POLICY_CKPT" \
  --in_hdf5 "$DATA_PATH" \
  --out_hdf5 "$OUTPUT_FILE" \
  --batch_size 64 \
  --device cuda
