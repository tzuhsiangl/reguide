#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name=generate_data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1


set -euo pipefail

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl


TASK="square"
# TASK="tool_hang"
# TASK="transport"
# TASK="can"


IMAGE_DIM=140
DONE_MODE=1

NUM_DEMO=1

DATA_PATH="path/to/robomimic_data"
OUTPUT1="output/path/to/data/without/abs_action.hdf5"
OUTPUT2="output/path/to/data/with/abs_action.hdf5"



if [ "$TASK" = "transport" ]; then
  CAMERA_VIEW=(robot0_eye_in_hand robot1_eye_in_hand shouldercamera0 shouldercamera1)
elif [ "$TASK" = "tool_hang" ]; then
  CAMERA_VIEW=(sideview robot0_eye_in_hand)
else
  CAMERA_VIEW=(agentview robot0_eye_in_hand)
fi

NUM_WORKER=8

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

echo "Using:"
echo "  TASK=$TASK"
echo "  IMAGE_DIM=$IMAGE_DIM"
echo "  DONE_MODE=$DONE_MODE"
echo "  DATA_PATH=$DATA_PATH"
echo "  OUTPUT1=$OUTPUT1"
echo "  OUTPUT2=$OUTPUT2"
echo "  CAMERA_VIEW=${CAMERA_VIEW[*]}"
echo "  NUM_WORKER=$NUM_WORKER"


python -m robomimic.scripts.dataset_states_to_obs \
  --done_mode "$DONE_MODE" \
  --dataset "$DATA_PATH" \
  --output_name "$OUTPUT1" \
  --camera_names "${CAMERA_VIEW[@]}" \
  --camera_height "$IMAGE_DIM" \
  --camera_width "$IMAGE_DIM" \
  --n "$NUM_DEMO"


python diffusion_policy/scripts/robomimic_dataset_conversion.py \
  --input "$OUTPUT1" \
  --output "$OUTPUT2" \
  --num_workers "$NUM_WORKER"
