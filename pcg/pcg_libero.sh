#!/bin/bash
# ReGuide test-time guidance (Phase-Conditioned Guidance) for a single LIBERO task.
#
#   conda activate reguide          # LIBERO must also be installed / importable
#   bash pcg/pcg_libero.sh
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
# LIBERO must be importable. If you have a local LIBERO checkout, set LIBERO_ROOT.
export PYTHONPATH="${LIBERO_ROOT:+$LIBERO_ROOT:}$PWD${PYTHONPATH:+:$PYTHONPATH}"

TASK="libero"

# ---- checkpoints + guidance targets (or set in dyn_model/conf/planner/eval_libero.yaml) ----
DYN_MODEL="path/to/dyn_model.pth"
POLICY="path/to/policy.ckpt"
PCG_DATA_PATH="path/to/pcg_targets.hdf5"        # phase-conditioned guidance targets (from Step 4)
LATENT_DIR="/targets/${TASK}_multitask"         # group inside PCG_DATA_PATH holding the targets

# ---- LIBERO task: CATEGORY is the task/scene name; DATASET_PATH is its demo hdf5 ----
CATEGORY="LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
DATASET_PATH="path/to/libero/${CATEGORY}_demo.hdf5"

# ---- run settings ----
NUM_TEST=50            # episodes per start-seed
SAVE_DATA=false        # true => save guided success rollouts (for self-improvement, Step 5)
SAVE_VIDEO=false

# ---- guidance hyperparameters (paper settings for this task) ----
GUIDING_STEPS=20
GUIDING_SCALE=0.005
SOFT_MIN=true
TAU=70
THRESHOLD_PERC=90
GUIDANCE_THRESHOLD_LOWER_PERC=50
GUIDANCE_THRESHOLD_UPPER_PERC=null
SWITCH_MARGIN=0.1
SWITCH_MIN_STEP=2
SWITCH_USE_THRESHOLD=true
NUM_TARGETS=100        # target prototypes per phase used at eval (caps what's loaded)
PCA_DIM=128

# ---- optional observation/action noise (off by default) ----
OBS_NOISE=false
ACTION_NOISE=false
OBS_IMAGE_NOISE_STD=0.05
OBS_STATE_NOISE_STD=0.01
ACTION_NOISE_STD=0.02

# Start seeds — each runs NUM_TEST episodes (the paper uses all 50).
# Trim this list (e.g. SEEDS=(266)) for a quick single run.
SEEDS=(5831 143870 4762 1517 2873 6724 473120 86850 521 337451 9308 86810 128 2208 737 288567 666 17862 346810 118609 3136 70381 405733 266 246810 719432 56428 624508 852691 1914 68856 1044 8470 135796 235796 1266 194025 14509 291604 46810 3419 812945 11236 21377 30214 366 16850 834017 41783 514237)

start_time=$(date +%s)

for SEED in "${SEEDS[@]}"; do
  OUTDIR="outputs/inference/${TASK}/${CATEGORY}/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_seed${SEED}"
  mkdir -p "$OUTDIR"
  echo "[pcg] TASK=$TASK category=$CATEGORY seed=$SEED n_test=$NUM_TEST -> $OUTDIR"

  "$PYTHON" eval_test_time_optimization_libero.py \
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
    success_hdf5_path="$OUTDIR/rollouts_success.hdf5" \
    failed_hdf5_path="$OUTDIR/rollouts_failed.hdf5" \
    save_video="$SAVE_VIDEO" \
    test_start_seed="$SEED" \
    obs_noise="$OBS_NOISE" \
    action_noise="$ACTION_NOISE" \
    obs_image_noise_std="$OBS_IMAGE_NOISE_STD" \
    obs_state_noise_std="$OBS_STATE_NOISE_STD" \
    action_noise_std="$ACTION_NOISE_STD" \
    dynamics_model_checkpoint="$DYN_MODEL" \
    policy_checkpoint="$POLICY" \
    dataset_path="$DATASET_PATH" \
    "$@"
done

elapsed=$(( $(date +%s) - start_time ))
echo "=============================="
echo "Total runtime: $((elapsed / 60)) min $((elapsed % 60)) sec"
echo "=============================="
