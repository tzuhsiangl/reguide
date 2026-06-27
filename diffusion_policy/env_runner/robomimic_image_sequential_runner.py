#!/usr/bin/env python3
"""
robomimic_code2_semantics_with_step_recording.py

Keeps Code-2 rollout semantics:
  - policy predicts chunk actions (K, act_dim)
  - env.step(action_chunk) (NOT step-by-step)
  - pbar.update(K)
  - success check via env.env.get_success_label() (VideoRecordingWrapper forwards to base env in DP codebase)
  - test seeding via env.seed(seed) (outer wrapper)
  - video path via env.render()

Adds: per-primitive-step recording (T, ...) using RecordingMultiStepWrapper,
      and optional HDF5 saving in robomimic-like format.

HDF5 format (if save_hdf5=True):
/data/demo_k/
  actions        (T, act_dim)
  abs_actions    (T, act_dim)
  rewards        (T,)
  dones          (T,)
  states         (T, state_dim)     # if provided by wrapper
  obs/<key>      (T, ...)
  next_obs/<key> (T, ...)
"""

from __future__ import annotations

import os
import pathlib
import collections
from typing import Any, Optional, Dict

import numpy as np
import torch
import tqdm
import h5py
import wandb

from diffusion_policy.gym_util.recording_multistep_wrapper import RecordingMultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

# IMPORTANT: use the OLD wrapper to match Code 2 exactly (model_file reset_to logic)
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import json


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


class SequentialRobomimicImageRunner(BaseImageRunner):
    """
    Code2 semantics + per-primitive-step recording + optional HDF5 writer.
    """

    def __init__(
        self,
        output_dir: str,
        dataset_path: str,
        shape_meta: dict,
        *,
        n_train: int = 10,
        n_train_vis: int = 3,
        train_start_idx: int = 0,
        n_test: int = 22,
        n_test_vis: int = 6,
        test_start_seed: int = 10000,
        max_steps: int = 400,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        render_obs_key: str = "agentview_image",
        fps: int = 10,
        crf: int = 22,
        past_action: bool = False,
        abs_action: bool = False,
        tqdm_interval_sec: float = 5.0,
        n_envs=None,
        # saving
        save_hdf5: bool = False,
        success_hdf5_path: Optional[str] = None,
        method_name: str = "code2_step_record",
        compress: Optional[str] = "gzip",
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

        dataset_path = os.path.expanduser(dataset_path)
        self.dataset_path = dataset_path
        if "square" in dataset_path:
            max_steps = 600

        self.media_dir = pathlib.Path(output_dir).joinpath("media")
        self.media_dir.mkdir(parents=True, exist_ok=True)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta["env_kwargs"]["use_object_obs"] = False

        rotation_transformer = None
        if abs_action:
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        # --- same env_configs structure as Code 2 ---
        self.env_configs = []
        with h5py.File(dataset_path, "r") as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                init_state = f[f"data/demo_{train_idx}/states"][0]
                enable_render = (i < n_train_vis)
                self.env_configs.append(
                    dict(prefix="train/", init_state=init_state, seed=None, enable_render=enable_render)
                )

        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = (i < n_test_vis)
            self.env_configs.append(
                dict(prefix="test/", init_state=None, seed=seed, enable_render=enable_render)
            )

        self.env_meta = env_meta
        self.shape_meta = shape_meta
        self.render_obs_key = render_obs_key
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.past_action = past_action
        self.abs_action = abs_action
        self.rotation_transformer = rotation_transformer
        self.tqdm_interval_sec = tqdm_interval_sec

        self.save_hdf5 = save_hdf5
        self.success_hdf5_path = success_hdf5_path
        self.method_name = method_name
        self.compress = compress
        self.compress_lvl = compress_lvl
        self.save_video = save_video
        self.guidance = guidance
        self.obs_noise = obs_noise
        self.action_noise = action_noise
        self.image_noise_std=obs_image_noise_std
        self.state_noise_std=obs_state_noise_std
        self.action_noise_std = action_noise_std
        self.rollout = rollout

    def create_env(self) -> RecordingMultiStepWrapper:
        """
        Create env EXACTLY like Code 2 in terms of robomimic env + wrapper stack,
        but using RecordingMultiStepWrapper to record per-primitive-step transitions.
        """
        modality_mapping = collections.defaultdict(list)
        for key, attr in self.shape_meta["obs"].items():
            modality_mapping[attr.get("type", "low_dim")].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        # Code 2 created env with enable_render=True always in the call site.
        # We keep that behavior here (always True).
        enable_render = True

        env = EnvUtils.create_env_from_metadata(
            env_meta=self.env_meta,
            render=False,
            render_offscreen=enable_render,
            use_image_obs=enable_render,
        )
        env.env.hard_reset = False

        base = RobomimicImageWrapper(
            env=env,
            shape_meta=self.shape_meta,
            init_state=None,
            render_obs_key=self.render_obs_key,
        )

        if self.save_video:
            video_recorder = VideoRecorder.create_h264(
                fps=self.fps,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=self.crf,
                thread_type="FRAME",
                thread_count=1,
            )
            wrapped = VideoRecordingWrapper(
                env=base,
                video_recoder=video_recorder,
                file_path=None,
                steps_per_render=max(20 // self.fps, 1),
            )
        else:
            wrapped = base

        # Key change vs Code 2: RecordingMultiStepWrapper (records per primitive step)
        image_keys = []
        state_keys = []
        for key, attr in self.shape_meta["obs"].items():
            if key == "language":
                continue
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                image_keys.append(key)
            elif obs_type == "low_dim":
                state_keys.append(key)

        env_wrapped = RecordingMultiStepWrapper(
            env=wrapped,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_episode_steps=self.max_steps,
            record=True,
            record_images_as_uint8=True,
            obs_noise=self.obs_noise,
            image_keys=image_keys,
            state_keys=state_keys,
            image_noise_std=self.image_noise_std,
            state_noise_std=self.state_noise_std,
            image_range="auto",
        )
        return env_wrapped



    def add_action_noise(self, action, noise_std=0.0, clip_min=None, clip_max=None):
        if noise_std <= 0:
            return action
        noisy_action = action + np.random.randn(*action.shape) * noise_std
        if clip_min is not None or clip_max is not None:
            min_val = -np.inf if clip_min is None else clip_min
            max_val =  np.inf if clip_max is None else clip_max
            noisy_action = np.clip(noisy_action, min_val, max_val)
        return noisy_action

    def _initialize_env(self, env: RecordingMultiStepWrapper, prefix: str, init_state, seed, enable_render: bool, i: int) -> None:
        """
        Match Code 2 initialization semantics as closely as possible.
        """
        inner = env.env  # VideoRecordingWrapper or RobomimicImageWrapper
        if self.save_video:
            assert isinstance(inner, VideoRecordingWrapper)
            inner.video_recoder.stop()
            inner.file_path = None
            if enable_render:
                filename = self.media_dir.joinpath(f"eval_video_{i}.mp4")
                inner.file_path = str(filename)
            assert isinstance(inner.env, RobomimicImageWrapper)
            base = inner.env
        else:
            assert isinstance(inner, RobomimicImageWrapper)
            base = inner

        if prefix.startswith("train"):
            base.init_state = init_state
        else:
            base.init_state = None
            # Code 2 seeds the OUTER wrapper
            env.seed(seed)

    def undo_transform_action(self, action: np.ndarray) -> np.ndarray:
        if self.rotation_transformer is None:
            raise RuntimeError("abs_action=True but rotation_transformer is None")

        raw_shape = action.shape
        if raw_shape[-1] == 20:
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]

        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

    def run(self, policy: BaseImagePolicy) -> Dict[str, Any]:
        device = policy.device
        n_inits = len(self.env_configs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_success = [0] * n_inits
        # saved_episode_records = []

        # HDF5 setup
        success_h5 = None
        success_data_grp = None

        success_total_steps = 0
        success_demo_write_idx = 0

        print("obs noise", self.obs_noise)
        print("action noise", self.action_noise)

        if self.save_hdf5:
            if self.success_hdf5_path is None:
                raise ValueError("save_hdf5=True but success_hdf5_path is None")

            _ensure_dir_for_file(self.success_hdf5_path)

            success_h5 = h5py.File(self.success_hdf5_path, "w")
            success_data_grp = success_h5.create_group("data")

            success_data_grp.attrs["method_name"] = self.method_name
            success_data_grp.attrs["dataset_path"] = self.dataset_path
            success_data_grp.attrs["n_obs_steps"] = int(self.n_obs_steps)
            success_data_grp.attrs["n_action_steps"] = int(self.n_action_steps)
            success_data_grp.attrs["split"] = "success"

        try:
            for i, cfg in enumerate(self.env_configs):
                prefix = cfg["prefix"]
                init_state = cfg["init_state"]
                seed = cfg["seed"]
                enable_render = cfg["enable_render"]

                env = self.create_env()
                self._initialize_env(env, prefix, init_state, seed, enable_render, i)

                base = env.env.env if isinstance(env.env, VideoRecordingWrapper) else env.env

                obs = env.reset()
                policy.reset()
                past_action = None
                rewards = []

                env_name = self.env_meta.get("env_name", "robomimic")
                pbar = tqdm.tqdm(
                    total=self.max_steps,
                    desc=f"Eval {env_name} {i+1}/{n_inits}",
                    leave=False,
                    mininterval=self.tqdm_interval_sec,
                )

                done = False
                success = False

                while not done:
                    np_obs_dict = dict(obs)
                    if self.past_action and (past_action is not None):
                        np_obs_dict["past_action"] = past_action[:, -(self.n_obs_steps - 1) :].astype(np.float32)
                    
                  
                    np_obs_dict = dict_apply(np_obs_dict, lambda x: np.expand_dims(x, axis=0))
                    obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

                    if self.guidance:
                        action_dict = policy.predict_action_dyn_guided(obs_dict)
                    else:
                        action_dict = policy.predict_action(obs_dict)
                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy().squeeze(0))
                    action = np_action_dict["action"]  # (K, act_dim)

                    #adding noise to actions before stepping into the environment
                    if self.action_noise:
                        action = self.add_action_noise(action, noise_std=self.action_noise_std)

                    if not np.all(np.isfinite(action)):
                        raise RuntimeError("NaN/Inf action")

                    env_action = action
                    if self.abs_action:
                        env_action = self.undo_transform_action(action)

                    obs, reward, done, info = env.step(env_action)

                    success = bool(base.get_success_label())

                    done = bool(np.all(done))
                    past_action = action
                    rewards.append(reward)

                    # Code 2 updates by chunk size
                    pbar.update(action.shape[0])
                    if success:
                        break

                pbar.close()

                all_success[i] = int(success)
                all_rewards[i] = rewards

                print(f"[DEBUG] seed={seed} success={success} ")

                # Write per-primitive-step data from RecordingMultiStepWrapper
                if self.save_hdf5:
                    assert success_data_grp is not None

                    # correction diagnostics from the planner (if any)
                    planner = getattr(policy, "planner", None)

                    corr_summary = None
                    corr_score = None

                    if planner is not None:
                        if hasattr(planner, "get_current_compact_summary"):
                            corr_summary = planner.get_current_compact_summary(
                                weak_margin_eps=0.05,
                            )

                        if corr_summary is not None:
                            if success and hasattr(planner, "compute_episode_selection_score_success"):
                                corr_score = planner.compute_episode_selection_score_success(corr_summary)
                            elif (not success) and hasattr(planner, "compute_episode_selection_score_compact"):
                                corr_score = planner.compute_episode_selection_score_compact(corr_summary)

                    episode_quality = None
                    if planner is not None and corr_summary is not None:
                        if hasattr(planner, "compute_episode_quality_compact"):
                            episode_quality = planner.compute_episode_quality_compact(corr_summary, success=bool(success))

                    if self.rollout:
                        # rollout: save BOTH success and failed episodes into the same folder
                        write_demo = True
                    else:
                        # not rollout (data collection): only save successful episodes
                        write_demo = bool(success)

                    if not write_demo:
                        print(f"[DEBUG] skipping write for seed={seed} (failed)")
                    else:
                        traj = env.get_recorded_trajectory()
                        T = int(traj.actions.shape[0])

                        demo_idx = success_demo_write_idx
                        print(f"[DEBUG] writing demo_{demo_idx} for seed={seed}")

                        # record = {
                        #     "episode_idx": int(i),
                        #     "prefix": str(prefix),
                        #     "seed": None if seed is None else int(seed),
                        #     "success": int(success),
                        #     "episode_quality": None if episode_quality is None else float(episode_quality),
                        #     "hdf5_demo_key": f"demo_{demo_idx}",
                        #     "corr_score": None if corr_score is None else float(corr_score),
                        #     "corr_summary": corr_summary,
                        # }

                        demo = success_data_grp.create_group(f"demo_{demo_idx}")
                        demo.create_dataset("actions", data=traj.actions.astype(np.float32))
                        demo.create_dataset("abs_actions", data=traj.actions.astype(np.float32))
                        demo.create_dataset("rewards", data=traj.rewards.astype(np.float32))
                        demo.create_dataset("dones", data=traj.dones.astype(np.uint8))

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
                        if episode_quality is not None:
                            demo.attrs["episode_quality"] = float(episode_quality)

                        if corr_score is not None:
                            demo.attrs["corr_score"] = float(corr_score)

                        if corr_summary is not None:
                            for k, v in corr_summary.items():
                                if v is not None:
                                    demo.attrs[f"corr_{k}"] = v
                        if seed is not None:
                            demo.attrs["seed"] = int(seed)

                        success_total_steps += T
                        success_demo_write_idx += 1

                        # saved_episode_records.append(record)

                # Video path EXACTLY like Code 2
                video_path = env.render()
                if isinstance(video_path, list) and len(video_path) > 0:
                    video_path = video_path[0]
                all_video_paths[i] = video_path

                env.close()
                del env

            if self.save_hdf5:
                assert success_data_grp is not None

                success_data_grp.attrs["total_steps"] = int(success_total_steps)
                success_data_grp.attrs["n_demos_written"] = int(success_demo_write_idx)

        finally:
            if success_h5 is not None:
                success_h5.close()

        # ---- Logging EXACTLY like Code 2 (max reward) ----
        max_rewards = collections.defaultdict(list)
        log_data: Dict[str, Any] = {}

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

            video_path = all_video_paths[i]
            if video_path is not None and isinstance(video_path, str) and os.path.exists(video_path):
                sim_video = wandb.Video(video_path)
                if prefix.startswith("train"):
                    log_data[prefix + f"sim_video_{i}"] = sim_video
                else:
                    log_data[prefix + f"sim_video_{seed}"] = sim_video

        for prefix, vals in max_rewards.items():
            log_data[prefix + "mean_score"] = float(np.mean(vals)) if len(vals) > 0 else 0.0
        #================save recorded episode metadata==============    
        # if self.save_hdf5:
        #     manifest_path = os.path.join(self.output_dir, "saved_episode_manifest.json")
        #     with open(manifest_path, "w") as f:
        #         json.dump(saved_episode_records, f, indent=2)
        #     #============================================================
        #     print(f"[DEBUG] saved episode manifest to {manifest_path}")
        return log_data
