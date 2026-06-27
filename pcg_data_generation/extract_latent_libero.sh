#!/bin/bash
# PCG step (a): encode observations into latents for a LIBERO task.
#
#   conda activate reguide          # LIBERO must also be installed / importable
#   bash pcg_data_generation/extract_latent_libero.sh

set -eo pipefail

# run from the repo root (this script lives in pcg_data_generation/)
cd "$(dirname "$0")/.." || exit 1

# use the active environment's python (override with: PYTHON=/path/to/python bash ...)
PYTHON=${PYTHON:-python}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# LIBERO must be importable. If you have a local LIBERO checkout, set LIBERO_ROOT.
export PYTHONPATH="${LIBERO_ROOT:+$LIBERO_ROOT:}$PWD${PYTHONPATH:+:$PYTHONPATH}"

TASK="libero"
CATEGORY="LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo"

# ---- paths ----
DATA_PATH="path/to/libero/${CATEGORY}.hdf5"
OUTPUT_FILE="path/to/libero/${CATEGORY}_encoded_obs_latents.hdf5"
DYN_CKPT="path/to/dyn_model.pth"
POLICY_CKPT="path/to/policy.ckpt"

echo "Using:"
echo "  TASK=$TASK"
echo "  CATEGORY=$CATEGORY"
echo "  DATA_PATH=$DATA_PATH"
echo "  DYN_CKPT=$DYN_CKPT"
echo "  POLICY_CKPT=$POLICY_CKPT"
echo "  OUTPUT_FILE=$OUTPUT_FILE"

"$PYTHON" pcg_data_generation/extract_latent_libero.py \
  --dynamics_ckpt "$DYN_CKPT" \
  --policy_ckpt_path "$POLICY_CKPT" \
  --in_hdf5 "$DATA_PATH" \
  --out_hdf5 "$OUTPUT_FILE" \
  --batch_size 64 \
  --device cuda
