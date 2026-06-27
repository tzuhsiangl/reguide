#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name=dyn_model
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
# run from the repo root
echo "PWD: $(pwd)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# ---- prevent ~/.local site-packages from leaking into the env ----
export PYTHONNOUSERSITE=1

# ---- mujoco / robosuite headless rendering ----
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1



export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true
export WANDB_START_METHOD=thread


export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR
# ---- wandb online ----
# TASK="square"
TASK="tool_hang"
# TASK="can"
# TASK="transport"


CATOGORY="80_demos"
export WANDB_DISABLE_SERVICE=true
export WANDB_START_METHOD=thread
export WANDB_PROJECT=reguide

export HYDRA_FULL_ERROR=1

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}"
which python




# VAL_DATASET="/path/to/validation/data.hdf5"
# TRAIN_DATASET="/path/to/training/data.hdf5"
# POLICY_CKPT='/path/to/diffusion/policy/checkpoint.ckpt'
# OUTDIR="/root/of/output"

VAL_DATASET="path/to/val_data.hdf5"
TRAIN_DATASET="path/to/train_data.hdf5"
POLICY_CKPT="path/to/policy.ckpt"


OUTDIR="path/to/output_dir"

echo "Python: $(which python)"
nvidia-smi || true

echo "Using:"
echo "  TASK=$TASK"
echo "  TRAIN_DATASET=$TRAIN_DATASET"
echo "  VAL_DATASET=$VAL_DATASET"
echo "  POLICY_CKPT=$POLICY_CKPT"
echo "  category=$CATOGORY"

python dyn_model/train.py --config-name train.yaml \
  env="${TASK}" \
  train_data_path="${TRAIN_DATASET}" \
  val_data_path="${VAL_DATASET}" \
  policy_ckpt_path="${POLICY_CKPT}" \
  env.train_data_path="${TRAIN_DATASET}" \
  env.val_data_path="${VAL_DATASET}" \
  env.policy_ckpt_path="${POLICY_CKPT}" \
  hydra.run.dir="${OUTDIR}/\${now:%Y.%m.%d}/\${now:%H.%M.%S}" \
  predictor_ckpt_path=null
