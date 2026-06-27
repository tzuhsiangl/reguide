#!/bin/bash
# ReGuide Fine-Tune (FT) for the `transport` task: continue training the base policy on
# (expert demos + guided success rollouts).
#
#   conda activate reguide
#   bash train_reguide-ft/train_transport_ft.sh

set -euo pipefail

# run from the repo root (this script lives in train_reguide-ft/)
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

# ---- experiment knobs ----
TASK="transport"
NUM_DEMOS="10_demos_10"
NUM_ROLLOUT=50
CATEGORY="share_80_first_${NUM_ROLLOUT}"
EXPERT_SAMPLE_WEIGHT=0.2
ROLLOUT_REUSE_RATE=2

# ---- datasets, init checkpoint, output ----
EXPERT_DATASET="path/to/expert_demos.hdf5"
ROLLOUT_DATASET="path/to/rollouts_success.hdf5"
FINETUNE_CKPT="path/to/base_policy.ckpt"
OUTPUT_ROOT="path/to/output_dir"

RUN_NAME="\${now:%Y.%m.%d}_\${now:%H.%M.%S}_train_diffusion_unet_hybrid_transport_image_${NUM_DEMOS}_${CATEGORY}_expert_ratio_${EXPERT_SAMPLE_WEIGHT}"

echo "Using: TASK=$TASK NUM_DEMOS=$NUM_DEMOS CATEGORY=$CATEGORY"
echo " EXPERT_DATASET=$EXPERT_DATASET"
echo " ROLLOUT_DATASET=$ROLLOUT_DATASET"
echo " FINETUNE_CKPT=$FINETUNE_CKPT"

"$PYTHON" train.py \
  --config-dir=. --config-name="image_${TASK}_diffusion_policy_cnn_ft.yaml" \
  training.seed=42 training.device=cuda:0 \
  logging.mode=offline \
  task.dataset.expert_dataset_path="$EXPERT_DATASET" \
  task.dataset.rollout_dataset_path="$ROLLOUT_DATASET" \
  task.dataset.expert_sample_weight="$EXPERT_SAMPLE_WEIGHT" \
  task.env_runner.dataset_path="$EXPERT_DATASET" \
  task.dataset_path="$EXPERT_DATASET" \
  task.dataset.rollout_reuse_rate="$ROLLOUT_REUSE_RATE" \
  hydra.run.dir="${OUTPUT_ROOT}/\${now:%Y.%m.%d}/\${now:%H.%M.%S}" \
  logging.name="$RUN_NAME" \
  training.resume=false \
  training.init_ckpt_path="$FINETUNE_CKPT" \
  training.init_from_ema=true \
  training.num_epochs=10000 \
  training.rollout_every=5 \
  training.checkpoint_every=5 \
  training.max_train_steps=9000 \
  training.lr_warmup_steps=500 \
  training.lr_scheduler=cosine \
  optimizer.lr=3e-4
