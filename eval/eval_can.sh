#!/bin/bash
# No-guidance evaluation for the `can` task.
# Works for any policy (base, fine-tuned, from-scratch, ...) — set CATEGORY to label the run.
#
#   conda activate reguide
#   bash eval/eval_can.sh
#
# Runs NUM_TEST episodes from each start-seed in SEEDS, sequentially on a single GPU.

set -eo pipefail

# run from the repo root (this script lives in eval/)
cd "$(dirname "$0")/.." || exit 1

# use the active environment's python (override with: PYTHON=/path/to/python bash ...)
PYTHON=${PYTHON:-python}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

TASK="can"

# ---- checkpoints (or set these in dyn_model/conf/planner/eval_can.yaml) ----
DYN_MODEL="path/to/dyn_model.pth"
POLICY="path/to/policy.ckpt"

# ---- eval settings ----
CATEGORY="base_policy" # label for this run (e.g. base_policy, reguide_ft, reguide_fs); used in OUTDIR
NUM_TEST=50            # episodes per start-seed
SAVE_DATA=false        # true => also save rollouts to HDF5
SAVE_VIDEO=false
ROLLOUT=false          # true => data-collection mode (save success + failed rollouts)

# Start seeds — each runs NUM_TEST episodes (the paper uses all 50).
# Trim this list (e.g. SEEDS=(5831)) for a quick single run.
SEEDS=(5831 143870 4762 1517 2873 6724 473120 86850 521 337451 9308 86810 128 2208 737 288567 666 17862 346810 118609 3136 70381 405733 266 246810 719432 56428 624508 852691 1914 68856 1044 8470 135796 235796 1266 194025 14509 291604 46810 3419 812945 11236 21377 30214 366 16850 834017 41783 514237)

start_time=$(date +%s)

for SEED in "${SEEDS[@]}"; do
  OUTDIR="outputs/inference/${TASK}/${CATEGORY}/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_seed${SEED}"
  mkdir -p "$OUTDIR"
  echo "[eval] TASK=$TASK seed=$SEED n_test=$NUM_TEST -> $OUTDIR"

  "$PYTHON" eval_test_time_optimization.py \
    --config-name "eval_${TASK}" \
    guidance=false \
    output_dir="$OUTDIR" \
    n_test="$NUM_TEST" \
    test_start_seed="$SEED" \
    save_hdf5="$SAVE_DATA" \
    save_video="$SAVE_VIDEO" \
    rollout="$ROLLOUT" \
    dynamics_model_checkpoint="$DYN_MODEL" \
    policy_checkpoint="$POLICY" \
    "$@"
done

elapsed=$(( $(date +%s) - start_time ))
echo "=============================="
echo "Total runtime: $((elapsed / 60)) min $((elapsed % 60)) sec"
echo "=============================="
