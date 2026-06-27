import os
import sys
import json
import math
import dill
import wandb
import pathlib
import collections
import numpy as np
import torch
import tqdm
import h5py

from dyn_model.datasets.language_goals import language_goals_list

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.recording_multistep_wrapper import RecordingMultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
    VideoRecorder,
)

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

from diffusion_policy.env.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
)
from diffusion_policy.env_runner.libero_bddl_mapping import bddl_file_name_dict

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from typing import Optional

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
libero_path = os.path.join(parent_dir, "LIBERO")
sys.path.append(libero_path)

if libero_path not in sys.path:
    sys.path.append(libero_path)

from libero.libero.envs.bddl_base_domain import TASK_MAPPING
from diffusion_policy.env_runner.libero_bddl_mapping import bddl_file_name_dict


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def create_env(
    env_meta,
    shape_meta,
    enable_render=True,
    render_obs_key='agentview_image',
    fps=10,
    crf=22,
    n_obs_steps=2,
    n_action_steps=8,
    max_steps=400,
    record=True,
):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    if env_meta["bddl_file"] not in bddl_file_name_dict.values():
        env_meta["bddl_file"] = bddl_file_name_dict[env_meta["bddl_file"]]
        env_meta["env_kwargs"]["bddl_file_name"] = env_meta["bddl_file"]

    raw_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    raw_env.env.hard_reset = False

    video_recorder = VideoRecorder.create_h264(
        fps=fps,
        codec='h264',
        input_pix_fmt='rgb24',
        crf=crf,
        thread_type='FRAME',
        thread_count=1
    )
    video_wrapper = VideoRecordingWrapper(
        env=RobomimicImageWrapper(
            env=raw_env,
            shape_meta=shape_meta,
            init_state=None,
            render_obs_key=render_obs_key,
        ),
        video_recoder=video_recorder,
        file_path=None,
        steps_per_render=max(20 // fps, 1)
    )

    # changed from MultiStepWrapper -> RecordingMultiStepWrapper
    env_wrapped = RecordingMultiStepWrapper(
        env=video_wrapper,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
        record=record,
        record_images_as_uint8=True,
    )
    return env_wrapped


class SequentialLiberoImageRunnerRecord(BaseImageRunner):
    """
    Sequential LIBERO runner with:
      - video saving
      - optional HDF5 rollout saving
      - robomimic-like output format

    Success detection uses base.get_success_label() (same as the robomimic runner)
    rather than `reward == 1.0`. This is required because RecordingMultiStepWrapper
    returns a chunk-summed scalar reward, which can never reliably equal 1.0 for
    LIBERO tasks (the env keeps emitting reward=1.0 after success, so the sum
    accumulates past 1.0).
    """

    def __init__(
        self,
        task_dir,
        output_dir,
        dataset_path,
        shape_meta: dict,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key="agentview_image",
        fps=10,
        crf=22,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        method_name="libero_seq_record",
        compress="gzip",
        save_hdf5: bool = False,
        success_hdf5_path: Optional[str] = None,
        failed_hdf5_path: Optional[str] = None,
        compress_lvl: int = 4,
        save_video: bool = True,
        guidance: bool = False,
        obs_noise: bool = False,
        action_noise: bool = False,
        obs_image_noise_std: float = 0.0,
        obs_state_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
        rollout: bool = False,
    ):
        super().__init__(output_dir)

        self.media_dir = pathlib.Path(output_dir).joinpath("media")
        self.media_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_path = os.path.expanduser(dataset_path)
        self.env_meta = FileUtils.get_env_metadata_from_dataset(self.dataset_path)
        self.shape_meta = shape_meta

        if self.env_meta["bddl_file"] not in bddl_file_name_dict.values():
            self.env_meta["bddl_file"] = bddl_file_name_dict[self.env_meta["bddl_file"]]
            self.env_meta["env_kwargs"]["bddl_file_name"] = self.env_meta["bddl_file"]

        self.rotation_transformer = None
        if abs_action:
            self.env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            from diffusion_policy.model.common.rotation_transformer import RotationTransformer
            self.rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        self.env_configs = []
        with h5py.File(self.dataset_path, "r") as f:
            if 'data' in f:
                data_group = f['data']
                self.env_args = data_group.attrs.get('env_args', None)

            for i in range(n_train):
                idx = train_start_idx + i
                init_state = f[f"data/demo_{idx}/states"][0]
                enable_render = (i < n_train_vis)
                self.env_configs.append({
                    "prefix": f"train/{self.env_meta['bddl_file'].split('/')[-1][:-5]}_",
                    "init_state": init_state,
                    "seed": None,
                    "enable_render": enable_render
                })

            for i in range(n_test):
                seed = test_start_seed + i
                print('seed:', seed)
                enable_render = (i < n_test_vis)
                self.env_configs.append({
                    "prefix": f"test/{self.env_meta['bddl_file'].split('/')[-1][:-5]}_",
                    "init_state": None,
                    "seed": seed,
                    "enable_render": enable_render
                })

        self.fps = fps
        self.crf = crf
        self.render_obs_key = render_obs_key
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.past_action = past_action
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.test_start_seed = test_start_seed
        self.output_path = None

        self.save_hdf5 = save_hdf5
        self.success_hdf5_path = success_hdf5_path
        self.failed_hdf5_path = failed_hdf5_path
        self.method_name = method_name
        self.compress = compress
        self.compress_lvl = compress_lvl
        self.save_video = save_video
        self.guidance = guidance
        self.rollout = rollout

        self.language_goal = " ".join(task_dir.split("/")[-1][:-10].split("_"))
        assert self.language_goal in language_goals_list, f"Language goal {self.language_goal} not found in language_goals"

        self.task_name = self.env_meta["bddl_file"].split("/")[-1][:-5]

    def _initialize_env(self, env, prefix, init_state, seed, enable_render, idx):
        assert isinstance(env.env, VideoRecordingWrapper)
        env.env.video_recoder.stop()
        env.env.file_path = None

        if enable_render and self.save_video:
            filename = self.media_dir.joinpath(f"eval_video_{idx}.mp4")
            env.env.file_path = str(filename)

        assert isinstance(env.env.env, RobomimicImageWrapper)
        if prefix.startswith("train"):
            env.env.env.init_state = init_state
        else:
            env.env.env.init_state = None
            env.seed(seed)

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3:3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

    def run(self, policy: BaseImagePolicy, **kwargs):
        device = policy.device

        n_inits = len(self.env_configs)
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_success = [0] * n_inits
        saved_episode_records = []

        success_h5 = None
        failed_h5 = None
        success_data_grp = None
        failed_data_grp = None

        success_total_steps = 0
        failed_total_steps = 0
        success_demo_write_idx = 0
        failed_demo_write_idx = 0

        if self.save_hdf5:
            if self.success_hdf5_path is None or self.failed_hdf5_path is None:
                raise ValueError("save_hdf5=True but success_hdf5_path or failed_hdf5_path is None")

            _ensure_dir_for_file(self.success_hdf5_path)
            _ensure_dir_for_file(self.failed_hdf5_path)

            success_h5 = h5py.File(self.success_hdf5_path, "w")
            failed_h5 = h5py.File(self.failed_hdf5_path, "w")

            success_data_grp = success_h5.create_group("data")
            failed_data_grp = failed_h5.create_group("data")

            for grp, split_name in [
                (success_data_grp, "success"),
                (failed_data_grp, "failed"),
            ]:
                grp.attrs["method_name"] = self.method_name
                grp.attrs["dataset_path"] = self.dataset_path
                grp.attrs["n_obs_steps"] = int(self.n_obs_steps)
                grp.attrs["n_action_steps"] = int(self.n_action_steps)
                grp.attrs["split"] = split_name

        try:
            for i, cfg in enumerate(self.env_configs):
                prefix = cfg["prefix"]
                init_state = cfg["init_state"]
                seed = cfg["seed"]
                enable_render = cfg["enable_render"]

                env = create_env(
                    env_meta=self.env_meta,
                    shape_meta=self.shape_meta,
                    enable_render=True,
                    render_obs_key=self.render_obs_key,
                    fps=self.fps,
                    crf=self.crf,
                    n_obs_steps=self.n_obs_steps,
                    n_action_steps=self.n_action_steps,
                    max_steps=self.max_steps,
                    record=True,
                )

                self._initialize_env(env, prefix, init_state, seed, enable_render, i)

                obs = env.reset()
                policy.reset()

                # Unwrap to RobomimicImageWrapper for get_success_label() access.
                # Stack at this point: RecordingMultiStepWrapper(VideoRecordingWrapper(RobomimicImageWrapper(...)))
                # so env.env is VideoRecordingWrapper and env.env.env is RobomimicImageWrapper.
                base = env.env.env if isinstance(env.env, VideoRecordingWrapper) else env.env

                # Sanity check the success-label API once per episode.
                if i == 0 and not hasattr(base, "get_success_label"):
                    raise RuntimeError(
                        f"Base env of type {type(base).__name__} has no get_success_label(); "
                        f"cannot determine success. Update the runner accordingly."
                    )

                past_action_list = []
                rewards = []

                pbar = tqdm.tqdm(
                    total=self.max_steps,
                    desc=f"Eval {self.task_name} {i+1}/{n_inits}",
                    leave=False,
                    mininterval=self.tqdm_interval_sec,
                )

                done = False
                success = False

                while not done:
                    np_obs_dict = dict(obs)
                    if self.past_action and len(past_action_list) > 0:
                        np_obs_dict["past_action"] = past_action_list[-1].astype(np.float32)

                    np_obs_dict = dict_apply(np_obs_dict, lambda x: np.expand_dims(x, axis=0))
                    obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device))

                    if "agentview_image" in obs_dict:
                        obs_dict["agentview_rgb"] = obs_dict.pop("agentview_image")
                    if "robot0_eye_in_hand_image" in obs_dict:
                        obs_dict["eye_in_hand_rgb"] = obs_dict.pop("robot0_eye_in_hand_image")
                    if "robot0_joint_pos" in obs_dict:
                        obs_dict["joint_states"] = obs_dict.pop("robot0_joint_pos")
                    if "robot0_eef_pos" in obs_dict:
                        obs_dict["ee_pos"] = obs_dict.pop("robot0_eef_pos")
                    if "robot0_eef_quat" in obs_dict:
                        obs_dict["ee_ori"] = obs_dict.pop("robot0_eef_quat")

                    if self.guidance:
                        action_dict = policy.predict_action_dyn_guided(
                            obs_dict,
                            language_goal=[self.language_goal] * obs_dict["agentview_rgb"].size(0),
                        )
                    else:
                        action_dict = policy.predict_action(
                            obs_dict,
                            language_goal=[self.language_goal] * obs_dict["agentview_rgb"].size(0),
                        )

                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy().squeeze(0))
                    action = np_action_dict["action"]

                    if not np.all(np.isfinite(action)):
                        raise RuntimeError("Nan or Inf action")

                    env_action = self.undo_transform_action(action) if self.abs_action else action
                    obs, reward, done_flag, info = env.step(env_action)

                    # Use base env's success label, NOT `reward == 1.0`.
                    # The RecordingMultiStepWrapper returns a chunk-summed scalar reward
                    # which is unreliable as a success indicator (see comment on class).
                    success = bool(base.get_success_label())

                    done = bool(np.all(done_flag)) or success

                    past_action_list.append(action[np.newaxis, ...])
                    if len(past_action_list) > 2:
                        past_action_list.pop(0)

                    rewards.append(reward)
                    pbar.update(action.shape[0])

                    if success:
                        break

                pbar.close()

                all_success[i] = int(success)
                all_rewards[i] = rewards

                print(f"[DEBUG] seed={seed} success={success}")

                if self.save_hdf5:
                    if self.rollout:
                        write_demo = True
                        target_split = "success"
                    elif not self.guidance:
                        if success:
                            write_demo = True
                            target_split = "success"
                        else:
                            write_demo = False
                            target_split = None
                    else:
                        write_demo = True
                        target_split = "success" if success else "failed"

                    if write_demo and target_split is not None:
                        traj = env.get_recorded_trajectory()
                        T = int(traj.actions.shape[0])

                        if target_split == "success":
                            target_grp = success_data_grp
                            target_demo_idx = success_demo_write_idx
                        else:
                            target_grp = failed_data_grp
                            target_demo_idx = failed_demo_write_idx

                        print(f"[DEBUG] writing {target_split} demo_{target_demo_idx} for seed={seed}")

                        record = {
                            "episode_idx": int(i),
                            "prefix": str(prefix),
                            "seed": None if seed is None else int(seed),
                            "success": int(success),
                            "split": target_split,
                            "hdf5_demo_key": f"demo_{target_demo_idx}",
                        }

                        demo = target_grp.create_group(f"demo_{target_demo_idx}")
                        demo.create_dataset("actions", data=traj.actions.astype(np.float32))
                        demo.create_dataset("abs_actions", data=traj.actions.astype(np.float32))
                        demo.create_dataset("rewards", data=np.array(traj.rewards).astype(np.float32))
                        demo.create_dataset("dones", data=np.array(traj.dones).astype(np.uint8))

                        if traj.states is not None:
                            demo.create_dataset("states", data=traj.states.astype(np.float32))

                        obs_grp = demo.create_group("obs")
                        next_obs_grp = demo.create_group("next_obs")

                        for k, arr in traj.obs.items():
                            if k.endswith("image") and (self.compress is not None):
                                obs_grp.create_dataset(
                                    k, data=arr,
                                    compression=self.compress,
                                    compression_opts=self.compress_lvl,
                                )
                            else:
                                obs_grp.create_dataset(k, data=arr)

                        for k, arr in traj.next_obs.items():
                            if k.endswith("image") and (self.compress is not None):
                                next_obs_grp.create_dataset(
                                    k, data=arr,
                                    compression=self.compress,
                                    compression_opts=self.compress_lvl,
                                )
                            else:
                                next_obs_grp.create_dataset(k, data=arr)

                        demo.attrs["prefix"] = prefix
                        demo.attrs["success"] = int(success)
                        demo.attrs["method_name"] = self.method_name
                        if seed is not None:
                            demo.attrs["seed"] = int(seed)

                        saved_episode_records.append(record)

                        if target_split == "success":
                            success_total_steps += T
                            success_demo_write_idx += 1
                        else:
                            failed_total_steps += T
                            failed_demo_write_idx += 1

                video_path = env.render()
                if isinstance(video_path, list) and len(video_path) > 0:
                    video_path = video_path[0]
                all_video_paths[i] = video_path

                env.close()
                del env

            if self.save_hdf5:
                success_data_grp.attrs["total_steps"] = int(success_total_steps)
                success_data_grp.attrs["n_demos_written"] = int(success_demo_write_idx)
                failed_data_grp.attrs["total_steps"] = int(failed_total_steps)
                failed_data_grp.attrs["n_demos_written"] = int(failed_demo_write_idx)

        finally:
            if success_h5 is not None:
                success_h5.close()
            if failed_h5 is not None:
                failed_h5.close()

        max_rewards = collections.defaultdict(list)
        log_data = {}

        for i, cfg in enumerate(self.env_configs):
            prefix = cfg["prefix"]
            seed = cfg["seed"]
            s = float(all_success[i])

            if prefix.startswith("train"):
                key = prefix + f"sim_max_reward_{i}"
            else:
                key = prefix + f"sim_max_reward_{seed}"

            log_data[key] = s
            max_rewards[prefix].append(s)

            vp = all_video_paths[i]
            if vp is not None and os.path.exists(vp):
                if prefix.startswith("train"):
                    log_data[f"{prefix}sim_video_{i}"] = wandb.Video(vp)
                else:
                    log_data[f"{prefix}sim_video_{seed}"] = wandb.Video(vp)

        for prefix, vals in max_rewards.items():
            log_data[f"{prefix}mean_score"] = np.mean(vals)

        if self.save_hdf5:
            manifest_path = os.path.join(self.output_dir, "saved_episode_manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(saved_episode_records, f, indent=2)
            print(f"[DEBUG] saved episode manifest to {manifest_path}")

        return log_data