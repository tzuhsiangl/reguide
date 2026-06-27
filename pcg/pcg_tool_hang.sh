#!/bin/bash
# ReGuide test-time guidance (Phase-Conditioned Guidance) for the `tool_hang` task.
#
#   conda activate reguide
#   bash pcg/pcg_tool_hang.sh
#
# Runs NUM_TEST guided episodes from each start-seed in SEEDS, sequentially on a single GPU.

set -eo pipefail

# run from the repo root (this script lives in pcg/)
cd "$(dirname "$0")/.." || exit 1

# use the active environment's python (override with: PYTHON=/path/to/python bash ...)
PYTHON=${PYTHON:-python}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLBACKEND=Agg
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

TASK="tool_hang"

# ---- checkpoints + guidance targets (or set in dyn_model/conf/planner/eval_tool_hang.yaml) ----
DYN_MODEL="path/to/dyn_model.pth"
POLICY="path/to/policy.ckpt"
PCG_DATA_PATH="path/to/pcg_targets.hdf5"   # phase-conditioned guidance targets (from Step 4)
LATENT_DIR="/targets/${TASK}"              # group inside PCG_DATA_PATH holding the targets

# ---- run settings ----
CATEGORY="reguide"     # label for this run; used in OUTDIR
NUM_TEST=50            # episodes per start-seed
SAVE_DATA=false        # true => save guided success rollouts (for self-improvement, Step 5)
SAVE_VIDEO=false

# ---- guidance hyperparameters (paper settings for this task) ----
GUIDING_STEPS=20
GUIDING_SCALE=0.01
SOFT_MIN=true
TAU=70
THRESHOLD_PERC=70
GUIDANCE_THRESHOLD_LOWER_PERC=20
GUIDANCE_THRESHOLD_UPPER_PERC=90
SWITCH_MARGIN=0.2
SWITCH_MIN_STEP=2
SWITCH_USE_THRESHOLD=true
NUM_TARGETS=100        # target prototypes per phase used at eval (caps what's loaded)
PCA_DIM=128

# Start seeds — each runs NUM_TEST episodes (the paper uses all 50).
# Trim this list (e.g. SEEDS=(2873)) for a quick single run.
SEEDS=(2873 8470 11236 852691 5831 30214 514237 1000300 9308 118609 266 366 737 3419 4762 17862 56428 143870 291604 1000400 21377 194025 128 521 1517 16850 68856 86810 135796 288567 719432 834017 1000000 1000100 86850 1000350 1000450 2208 41783 46810 70381 235796 246810 405733 812945 1000150 1000250 666 337451 6724)

start_time=$(date +%s)

for SEED in "${SEEDS[@]}"; do
  OUTDIR="outputs/inference/${TASK}/${CATEGORY}/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_seed${SEED}"
  mkdir -p "$OUTDIR"
  echo "[pcg] TASK=$TASK seed=$SEED n_test=$NUM_TEST -> $OUTDIR"

  "$PYTHON" eval_test_time_optimization.py \
    --config-name "eval_${TASK}" \
    guidance=true \
    guidance_scale="$GUIDING_SCALE" \
    guidance_start_timestep="$GUIDING_STEPS" \
    output_dir="$OUTDIR" \
    latent_dir="$LATENT_DIR" \
    pcg_data_path="$PCG_DATA_PATH" \
    pca_dim="$PCA_DIM" \
    targets_num="$NUM_TARGETS" \
    soft_min="$SOFT_MIN" \
    tau="$TAU" \
    threshold_perc="$THRESHOLD_PERC" \
    phase_switch_margin="$SWITCH_MARGIN" \
    phase_switch_min_steps="$SWITCH_MIN_STEP" \
    phase_switch_use_threshold="$SWITCH_USE_THRESHOLD" \
    guidance_threshold_lower_perc="$GUIDANCE_THRESHOLD_LOWER_PERC" \
    guidance_threshold_upper_perc="$GUIDANCE_THRESHOLD_UPPER_PERC" \
    n_test="$NUM_TEST" \
    save_hdf5="$SAVE_DATA" \
    save_video="$SAVE_VIDEO" \
    test_start_seed="$SEED" \
    dynamics_model_checkpoint="$DYN_MODEL" \
    policy_checkpoint="$POLICY" \
    "$@"
done

elapsed=$(( $(date +%s) - start_time ))
echo "=============================="
echo "Total runtime: $((elapsed / 60)) min $((elapsed % 60)) sec"
echo "=============================="
