from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gym


def _as_float32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _img_to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert image to HWC uint8 [0,255].
    Accepts CHW/HWC, float/uint8.
    """
    x = np.asarray(img)
    if x.ndim != 3:
        raise ValueError(f"_img_to_hwc_uint8 expected 3D, got {x.shape}")

    # CHW -> HWC
    if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.moveaxis(x, 0, -1)

    if np.issubdtype(x.dtype, np.floating):
        vmax = float(x.max()) if x.size else 0.0
        if vmax <= 1.5:
            x = x * 255.0
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def _maybe_take_last_obs(x: np.ndarray, n_obs_steps: int) -> np.ndarray:
    """
    If x is stacked (n_obs_steps, ...), take last frame. Else return as-is.
    """
    arr = np.asarray(x)
    if arr.ndim >= 1 and arr.shape[0] == n_obs_steps:
        return arr[-1]
    return arr


@dataclass
class RecordedTrajectory:
    # primitive-step sequences of length T
    obs: Dict[str, np.ndarray]
    next_obs: Dict[str, np.ndarray]
    actions: np.ndarray        # (T, action_dim)
    rewards: np.ndarray        # (T,)
    dones: np.ndarray          # (T,)
    states: Optional[np.ndarray]  # (T, state_dim) if available else None


class RecordingMultiStepWrapper(gym.Wrapper):
    """
    A wrapper that:
      - accepts action chunks of shape (n_action_steps, action_dim) OR (action_dim,)
      - executes the underlying env step-by-step for n_action_steps primitive steps
      - returns the same kind of output as a standard "multistep" wrapper would return
      - BUT additionally records every primitive transition so you can save a robomimic-style dataset.

    Assumptions:
      - underlying env returns obs dict
      - underlying env has:
          - get_flattened_state()  (optional; used for states)
    """

    def __init__(
        self,
        env: gym.Env,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        max_episode_steps: int = 400,
        record: bool = True,
        record_images_as_uint8: bool = True,
        save_video: bool = True,
        # sensor noise config
        obs_noise: bool = False,
        image_keys: Optional[List[str]] = None,
        state_keys: Optional[List[str]] = None,
        image_noise_std: float = 0.0,
        state_noise_std: float = 0.0,
        image_range: str = "auto",   # "auto", "zero_one", "zero_255", "none"
    ):
        super().__init__(env)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.max_episode_steps = int(max_episode_steps)
        self.record = bool(record)
        self.record_images_as_uint8 = bool(record_images_as_uint8)

        self._step_count = 0
        self.save_video = save_video
        # sensor noise config
        self.obs_noise = bool(obs_noise)
        self.image_keys = list(image_keys or [])
        self.state_keys = list(state_keys or [])
        self.image_noise_std = float(image_noise_std)
        self.state_noise_std = float(state_noise_std)
        self.image_range = str(image_range)
        # buffers for primitive steps
        self._traj_obs_list: List[Dict[str, Any]] = []
        self._traj_next_obs_list: List[Dict[str, Any]] = []
        self._traj_actions: List[np.ndarray] = []
        self._traj_rewards: List[float] = []
        self._traj_dones: List[int] = []
        self._traj_states: List[np.ndarray] = []

        # obs history for stacked output
        self._obs_hist: List[Dict[str, Any]] = []

    def _add_sensor_noise_to_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add sensor noise to a single-step observation dict.
        Returns a new dict, does not modify input in-place.
        """
        if not self.obs_noise:
            return {k: np.array(v, copy=True) for k, v in obs.items()}

        noisy_obs: Dict[str, Any] = {}

        for k, v in obs.items():
            x = np.array(v, copy=True)

            is_image = k in self.image_keys
            is_state = k in self.state_keys

            # fallback heuristic if keys were not provided
            if not is_image and not is_state:
                if x.ndim == 3:
                    is_image = True
                else:
                    is_state = True

            if is_image and self.image_noise_std > 0:
                x_float = x.astype(np.float32)
                x_float = x_float + np.random.randn(*x.shape).astype(np.float32) * self.image_noise_std

                if self.image_range != "none":
                    if self.image_range == "zero_one":
                        x_float = np.clip(x_float, 0.0, 1.0)
                    elif self.image_range == "zero_255":
                        x_float = np.clip(x_float, 0.0, 255.0)
                    elif self.image_range == "auto":
                        xmax = float(np.max(x)) if x.size > 0 else 0.0
                        if np.issubdtype(x.dtype, np.floating):
                            if xmax <= 1.5:
                                x_float = np.clip(x_float, 0.0, 1.0)
                            else:
                                x_float = np.clip(x_float, 0.0, 255.0)
                        else:
                            x_float = np.clip(x_float, 0.0, 255.0)
                    else:
                        raise ValueError(f"Unsupported image_range: {self.image_range}")

                if np.issubdtype(x.dtype, np.integer):
                    x = np.rint(x_float).astype(x.dtype)
                else:
                    x = x_float.astype(x.dtype)

            elif is_state and self.state_noise_std > 0:
                x_float = x.astype(np.float32)
                x_float = x_float + np.random.randn(*x.shape).astype(np.float32) * self.state_noise_std
                if np.issubdtype(x.dtype, np.floating):
                    x = x_float.astype(x.dtype)
                else:
                    x = np.rint(x_float).astype(x.dtype)

            noisy_obs[k] = x

        return noisy_obs

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        if self.obs_noise:
            obs = self._add_sensor_noise_to_obs(obs)

        self._step_count = 0

        # reset buffers
        self._traj_obs_list.clear()
        self._traj_next_obs_list.clear()
        self._traj_actions.clear()
        self._traj_rewards.clear()
        self._traj_dones.clear()
        self._traj_states.clear()

        # reset obs history with repeated first obs
        self._obs_hist = [obs for _ in range(self.n_obs_steps)]

        return self._get_stacked_obs()

    

    def _get_stacked_obs(self) -> Dict[str, np.ndarray]:
        """
        Return stacked obs dict with shape (n_obs_steps, ...) for each key.
        """
        out = {}
        keys = self._obs_hist[-1].keys()
        for k in keys:
            out[k] = np.stack([np.asarray(o[k]) for o in self._obs_hist], axis=0)
        return out

    def _record_transition(
        self,
        obs_t: Dict[str, Any],
        action_t: np.ndarray,
        reward_t: float,
        done_t: bool,
        obs_tp1: Dict[str, Any],
    ) -> None:
        if not self.record:
            return

        # record obs / next_obs dicts (we store single-step frames, not stacked)
        self._traj_obs_list.append(obs_t)
        self._traj_next_obs_list.append(obs_tp1)

        self._traj_actions.append(_as_float32(action_t))
        self._traj_rewards.append(float(reward_t))
        self._traj_dones.append(1 if bool(done_t) else 0)

        # optional state
        if hasattr(self.env, "get_flattened_state"):
            st = self.env.get_flattened_state()
            self._traj_states.append(_as_float32(st))

    def step(self, action):
        """
        action can be:
          - (n_action_steps, action_dim)  -> execute all primitive steps
          - (action_dim,)                -> execute 1 primitive step (still works)
        """
        # Normalize action format
        a = np.asarray(action)
        if a.ndim == 1:
            action_seq = a[None, ...]  # (1, Da)
        elif a.ndim == 2:
            action_seq = a
        else:
            raise ValueError(f"action must be 1D or 2D, got shape {a.shape}")

        # If user passed fewer than n_action_steps, we execute that many
        # If user passed more, we execute up to n_action_steps
        K = min(action_seq.shape[0], self.n_action_steps)

        total_reward = 0.0
        done = False
        info_out: Dict[str, Any] = {}

        # execute primitive steps
        for j in range(K):
            if self._step_count >= self.max_episode_steps:
                done = True
                break

            obs_t = self._obs_hist[-1]
            a_t = action_seq[j]
            if self.obs_noise:
                obs_tp1_clean, r, d, info = self.env.step(a_t)
                obs_tp1 = self._add_sensor_noise_to_obs(obs_tp1_clean)
            else:
                obs_tp1, r, d, info = self.env.step(a_t)
            total_reward += float(np.mean(np.asarray(r)))
            done = bool(np.all(d)) if isinstance(d, (np.ndarray, list, tuple)) else bool(d)
            info_out = info

            self._record_transition(obs_t, a_t, float(np.mean(np.asarray(r))), done, obs_tp1)

            # update obs history
            self._obs_hist.append(obs_tp1)
            if len(self._obs_hist) > self.n_obs_steps:
                self._obs_hist = self._obs_hist[-self.n_obs_steps :]

            self._step_count += 1
            if done:
                break

        # Return stacked obs like MultiStepWrapper
        stacked_obs = self._get_stacked_obs()
        return stacked_obs, total_reward, done, info_out

    def get_recorded_trajectory(self) -> RecordedTrajectory:
        """
        Convert internal lists into arrays in a robomimic-friendly shape:
          obs/<k>:      (T, ...)
          next_obs/<k>: (T, ...)
          actions:      (T, Da)
          rewards:      (T,)
          dones:        (T,)
          states:       (T, Ds) or None
        """
        T = len(self._traj_actions)
        if T == 0:
            # return empty
            return RecordedTrajectory(
                obs={}, next_obs={}, actions=np.zeros((0, 0), np.float32),
                rewards=np.zeros((0,), np.float32), dones=np.zeros((0,), np.uint8),
                states=None
            )

        # Convert obs dict list -> dict arrays
        keys = self._traj_obs_list[0].keys()
        obs_out: Dict[str, np.ndarray] = {}
        next_obs_out: Dict[str, np.ndarray] = {}

        for k in keys:
            obs_seq = []
            next_seq = []
            for t in range(T):
                o = self._traj_obs_list[t][k]
                no = self._traj_next_obs_list[t][k]

                # The underlying env wrapper might return CHW float images or HWC uint8.
                # Normalize images to HWC uint8 if requested.
                if k.endswith("image") and self.record_images_as_uint8:
                    obs_seq.append(_img_to_hwc_uint8(o))
                    next_seq.append(_img_to_hwc_uint8(no))
                else:
                    obs_seq.append(_as_float32(o))
                    next_seq.append(_as_float32(no))

            obs_out[k] = np.stack(obs_seq, axis=0)
            next_obs_out[k] = np.stack(next_seq, axis=0)

        actions = np.stack(self._traj_actions, axis=0).astype(np.float32)
        rewards = np.asarray(self._traj_rewards, dtype=np.float32)
        dones = np.asarray(self._traj_dones, dtype=np.uint8)
        states = np.stack(self._traj_states, axis=0).astype(np.float32) if len(self._traj_states) == T else None

        return RecordedTrajectory(
            obs=obs_out,
            next_obs=next_obs_out,
            actions=actions,
            rewards=rewards,
            dones=dones,
            states=states,
        )

    def clear_recorded_trajectory(self) -> None:
        self._traj_obs_list.clear()
        self._traj_next_obs_list.clear()
        self._traj_actions.clear()
        self._traj_rewards.clear()
        self._traj_dones.clear()
        self._traj_states.clear()