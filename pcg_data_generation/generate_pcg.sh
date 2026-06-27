#!/bin/bash
# PCG step (b): cluster extracted latents into per-phase guidance targets (robomimic task).
#
#   conda activate reguide
#   bash pcg_data_generation/generate_pcg.sh

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
TASK="transport"        # can | square | transport | tool_hang
NUM_PHASES=5
PCA_DIM=128
NUM_TARGETS=100
CATEGORY="base_policy"

K_CLUSTER=$(( NUM_PHASES * 10 ))

# extracted latents from step (a); output dir for the guidance targets
LATENTS_H5="path/to/encoded_obs_latents.hdf5"
OUTDIR="path/to/output/dir"
mkdir -p "$OUTDIR"

echo "TASK=$TASK  NUM_TARGETS=$NUM_TARGETS  CATEGORY=$CATEGORY"
echo "K_CLUSTER=$K_CLUSTER  PCA_DIM=$PCA_DIM  NUM_PHASES=$NUM_PHASES  OUTDIR=$OUTDIR"

# Common clustering args; each variant only changes --prototype_pick.
COMMON_ARGS=(
  --latents_h5 "$LATENTS_H5"
  --alpha_proprio 1.0
  --alpha_dvisual 0.3
  --alpha_dproprio 0.3
  --pca_dim "$PCA_DIM"
  --cluster_k "$K_CLUSTER"
  --n_phases "$NUM_PHASES"
  --phase_merge time_bins
  --n_prototypes_per_phase "$NUM_TARGETS"
  --farthest_pool 100
  --task "$TASK"
  --do_plot
  --color_by phase
  --overwrite
  --diag_tau_percents 10 20 30 50 70 90
  --diag_threshold_percents 50 60 70 80 90
  --diag_use_runtime_proto_count
)

make_out_h5()   { echo "${OUTDIR}/${TASK}_${PCA_DIM}_k${K_CLUSTER}_ph${NUM_PHASES}_protos${NUM_TARGETS}_$1.hdf5"; }
make_plot_path() { echo "${OUTDIR}/${TASK}_${NUM_PHASES}targets_$1.png"; }

"$PYTHON" pcg_data_generation/latent_state_clustering.py \
  "${COMMON_ARGS[@]}" \
  --prototype_pick per_cluster_score \
  --save_targets_h5 "$(make_out_h5 per_cluster_score)" \
  --save_plot       "$(make_plot_path per_cluster_score)"

echo
echo "=========================================="
echo "Done. Outputs in $OUTDIR:"
ls -1 "$OUTDIR" | grep -E "${TASK}_${PCA_DIM}_k${K_CLUSTER}_ph${NUM_PHASES}_protos${NUM_TARGETS}" || true
echo "=========================================="
