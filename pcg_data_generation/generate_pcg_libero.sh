#!/bin/bash
# PCG step (b): cluster extracted latents into per-phase guidance targets (LIBERO task).
#
#   conda activate reguide
#   bash pcg_data_generation/generate_pcg_libero.sh

set -eo pipefail

# run from the repo root (this script lives in pcg_data_generation/)
cd "$(dirname "$0")/.." || exit 1

# use the active environment's python (override with: PYTHON=/path/to/python bash ...)
PYTHON=${PYTHON:-python}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLBACKEND=Agg
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# ---- task / clustering config ----
TASK="libero"
NUM_PHASES=4
PCA_DIM=128
NUM_TARGETS=100
CATEGORY="LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo"

K_CLUSTER=$(( NUM_PHASES * 10 ))

# extracted latents from step (a); output dir for the guidance targets
LATENTS_H5="path/to/libero/${CATEGORY}_encoded_obs_latents.hdf5"
OUTDIR="path/to/output/dir"
mkdir -p "$OUTDIR"

echo "TASK=$TASK  NUM_TARGETS=$NUM_TARGETS  CATEGORY=$CATEGORY  OUTDIR=$OUTDIR"

"$PYTHON" pcg_data_generation/latent_state_clustering.py \
  --latents_h5 "$LATENTS_H5" \
  --alpha_proprio 1.0 \
  --alpha_dvisual 0.3 \
  --alpha_dproprio 0.3 \
  --pca_dim "$PCA_DIM" --cluster_k "$K_CLUSTER" --n_phases "$NUM_PHASES" \
  --phase_merge time_bins \
  --n_prototypes_per_phase "$NUM_TARGETS" \
  --prototype_pick score --farthest_pool 100 \
  --task "$TASK" \
  --save_targets_h5 "${OUTDIR}/${CATEGORY}_${PCA_DIM}_k${K_CLUSTER}_ph${NUM_PHASES}_protos${NUM_TARGETS}.hdf5" \
  --do_plot \
  --color_by phase \
  --save_plot "${CATEGORY}_${NUM_PHASES}targets"
