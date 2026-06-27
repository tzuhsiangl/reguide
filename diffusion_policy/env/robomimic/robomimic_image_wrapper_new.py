# robomimic_image_wrapper_new.py
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
import gym
from gym import spaces

from robomimic.envs.env_robosuite import EnvRobosuite


def to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to HWC uint8 in [0,255].
    Accepts:
      - HWC uint8
      - HWC float in [0,1] or [0,255]
      - CHW float/uint8
    """
    x = np.asarray(img)

    # If someone passes stacked frames (K,H,W,C) by mistake, fail loudly
    if x.ndim == 4:
        raise ValueError(f"to_hwc_uint8 expected a single image, got shape {x.shape} (stacked?).")

    # CHW -> HWC if needed
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.moveaxis(x, 0, -1)

    if np.issubdtype(x.dtype, np.floating):
        vmax = float(np.max(x)) if x.size else 0.0
        if vmax <= 1.5:
            x = x * 255.0
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)

    return x

def to_chw_float01(img: np.ndarray) -> np.ndarray:
        x = np.asarray(img)
        # If HWC -> CHW
        if x.ndim == 3 and x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
            x = np.moveaxis(x, -1, 0)
        # If uint8 -> float [0,1]
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        else:
            x = x.astype(np.float32)
            # if looks like [0,255], scale down
            if x.size and x.max() > 1.5:
                x = x / 255.0
        return x
        
class RobomimicImageWrapper(gym.Env):
    """
    Gym wrapper that:
      - exposes obs/action spaces using shape_meta
      - supports reset_to(init_state) and deterministic seed resets
      - exposes dataset-style state vector via env.get_state()['states'] (matches /states {T,45})
    """
    def __init__(
        self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        render_obs_key: str = "agentview_image",
        save_hdf5: bool = False,  
    ):
        self.env = env
        self.shape_meta = shape_meta
        self.render_obs_key = render_obs_key
        self.init_state = init_state

        self.seed_state_map: dict[int, np.ndarray] = {}
        self._seed: Optional[int] = None
        self.has_reset_before = False
        self.render_cache = None

        action_shape = tuple(shape_meta["action"]["shape"])
        # If dataset stored chunked actions like (K,7), env still expects (7,)
        if len(action_shape) == 2 and action_shape[-1] == 7:
            action_shape = (7,)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=action_shape, dtype=np.float32)

        obs_space = spaces.Dict()
        for key, spec in shape_meta["obs"].items():
            shape = tuple(spec["shape"])
            if key.endswith("image"):
                # dataset stores uint8 HWC
                obs_space[key] = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
            else:
                obs_space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        self.observation_space = obs_space

    def seed(self, seed: Optional[int] = None):
        self._seed = None if seed is None else int(seed)

    def set_init_state(self, state: Optional[np.ndarray]):
        self.init_state = state

    def get_state_vec(self) -> np.ndarray:
        st = self.env.get_state()
        if "states" not in st:
            raise RuntimeError("env.get_state() does not contain key 'states'.")
        return np.asarray(st["states"])

    def get_success_label(self) -> bool:
        """
        Runner expects this. In robomimic robosuite envs, success check is usually _check_success().
        """
        return bool(self.env.env._check_success())

    def get_flattened_state(self) -> np.ndarray:
        """
        Match author wrapper behavior (sim state flatten).
        """
        return self.env.env.sim.get_state().flatten()

    # Optional: only needed if you use Transport / ToolHang specific checks anywhere
    def get_check_tool_on_frame(self) -> bool:
        return bool(self.env.env._check_tool_on_frame())

    def get_check_frame_assembled(self) -> bool:
        return bool(self.env.env._check_frame_assembled())

    def get_trash_in_trash_bin(self) -> bool:
        return bool(self.env.env.transport.trash_in_trash_bin)

    def get_payload_in_target_bin(self) -> bool:
        return bool(self.env.env.transport.payload_in_target_bin)

    def get_observation(self, raw_obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        if self.render_obs_key in raw_obs:
            self.render_cache = raw_obs[self.render_obs_key]

        obs: Dict[str, Any] = {}
        for k in self.observation_space.spaces.keys():
            v = raw_obs[k]
            if k.endswith("image"):
                obs[k] = to_chw_float01(v)   # CHW float32 [0,1]
            else:
                obs[k] = np.asarray(v, dtype=np.float32)
        return obs

    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                self.env.reset()
                self.has_reset_before = True
            raw_obs = self.env.reset_to({"states": self.init_state})
            return self.get_observation(raw_obs)

        if self._seed is not None:
            seed = self._seed
            if seed in self.seed_state_map:
                raw_obs = self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                st = self.env.get_state()["states"]
                self.seed_state_map[seed] = st
                raw_obs = self.env.reset_to({"states": st})
            self._seed = None
            return self.get_observation(raw_obs)

        raw_obs = self.env.reset()
        st = self.env.get_state()["states"]
        raw_obs = self.env.reset_to({"states": st})
        return self.get_observation(raw_obs)

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must call reset() or step() before render().")
        return to_hwc_uint8(self.render_cache)
