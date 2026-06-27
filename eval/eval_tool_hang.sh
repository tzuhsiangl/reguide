#!/bin/bash
# No-guidance evaluation for the `tool_hang` task.
# Works for any policy (base, fine-tuned, from-scratch, ...) — set CATEGORY to label the run.
#
#   conda activate reguide
#   bash eval/eval_tool_hang.sh
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

TASK="tool_hang"

# ---- checkpoints (or set these in dyn_model/conf/planner/eval_tool_hang.yaml) ----
DYN_MODEL="path/to/dyn_model.pth"
POLICY="path/to/policy.ckpt"

# ---- eval settings ----
CATEGORY="base_policy" # label for this run (e.g. base_policy, reguide_ft, reguide_fs); used in OUTDIR
NUM_TEST=50            # episodes per start-seed
SAVE_DATA=false        # true => also save rollouts to HDF5
SAVE_VIDEO=false
ROLLOUT=false          # true => data-collection mode (save success + failed rollouts)

# Start seeds — each runs NUM_TEST episodes (the paper uses all 50).
# Trim this list (e.g. SEEDS=(2873)) for a quick single run.
SEEDS=(2873 8470 11236 852691 5831 30214 514237 1000300 9308 118609 266 366 737 3419 4762 17862 56428 143870 291604 1000400 21377 194025 128 521 1517 16850 68856 86810 135796 288567 719432 834017 1000000 1000100 86850 1000350 1000450 2208 41783 46810 70381 235796 246810 405733 812945 1000150 1000250 666 337451 6724)

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
