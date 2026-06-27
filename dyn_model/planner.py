import os
from pathlib import Path
import json
import copy

import h5py
import hydra
import numpy as np
import torch
import torch.utils.data as data
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from einops import rearrange
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate

import robomimic.utils.file_utils as FileUtils

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.robomimic_replay_image_dataset import (
    _convert_actions,
    undo_transform_action,
)
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

from dyn_model.datasets.robomimic_dset import RobomimicImageDynamicsModelDataset
from dyn_model.datasets.img_transforms import (
    default_transform,
    get_eval_crop_transform,
    get_eval_crop_transform_resnet,
)
from dyn_model.plan import load_model


class Planner:
    def __init__(
        self,
        demo_dataset_config,
        dynamics_model_ckpt,
        action_step=8,
        output_dir="debug/",
        guidance=False,
        latent_dir="/targets/transport/kmeans_prop_time",
        pcg_data_path=None,
        targets_num=None,
        tau=None,
        threshold_perc=None,
        soft_min=None,
        # phase switching knobs
        phase_switch_margin=0.1,
        phase_switch_min_steps=2,
        phase_switch_use_threshold=True,
        # NEW: guidance activation threshold
        guidance_threshold_lower_perc=None,
        guidance_threshold_upper_perc=None,  # NEW
        guidance_disable_when_next_is_near=True,
        # reward softmin temperatures
        proto_softmin_temp_v=0.05,
        proto_softmin_temp_p=0.05,
    ):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.demo_dataset_config = demo_dataset_config

        dynamics_model_dir = os.path.dirname(os.path.dirname(dynamics_model_ckpt))
        with open(os.path.join(dynamics_model_dir, "hydra.yaml"), "r") as f:
            model_cfg = OmegaConf.load(f)
            assert model_cfg.abs_action == self.demo_dataset_config.abs_action

        self.dyn_model = load_model(Path(dynamics_model_ckpt), model_cfg, device=self.device)
        if (not model_cfg.model.train_encoder) and (model_cfg.encoder_ckpt_path is not None):
            encoder_ckpt = torch.load(model_cfg.encoder_ckpt_path, map_location="cuda")
            self.dyn_model.encoder.load_state_dict(encoder_ckpt["encoder"])
            print("loaded encoder from", model_cfg.encoder_ckpt_path)

        self.dyn_model = self.accelerator.prepare(self.dyn_model)
        self.dyn_model.eval()

        self.pcg_data_path = pcg_data_path
        self.subgoal_idx_episodes = []
        self._subgoal_idx_trace = []
        self.subgoal_idx = 0
        self.guidance = guidance
        self.increment = False

        self.latent_dir = latent_dir
        self.targets_num = targets_num
        self.tau = tau
        self.threshold_percent = threshold_perc
        self.soft_min = soft_min

        self.phase_switch_margin = float(phase_switch_margin)
        self.phase_switch_min_steps = int(phase_switch_min_steps)
        self.phase_switch_use_threshold = bool(phase_switch_use_threshold)
        self.phase_switch_counter = 0

        self.guidance_threshold_lower_percent = (
            int(guidance_threshold_lower_perc)
            if guidance_threshold_lower_perc is not None
            else int(threshold_perc)
        )
        self.guidance_threshold_upper_percent = (
            int(guidance_threshold_upper_perc)
            if guidance_threshold_upper_perc is not None
            else None
        )

        self.guidance_disable_when_next_is_near = bool(guidance_disable_when_next_is_near)

        self.proto_softmin_temp_v = float(proto_softmin_temp_v)
        self.proto_softmin_temp_p = float(proto_softmin_temp_p)

        print("tau percent", self.tau)
        print("threshold percent", self.threshold_percent)
        print("guidance threshold percent (lower)", self.guidance_threshold_lower_percent)
        print("guidance threshold percent (upper)", self.guidance_threshold_upper_percent)
        print("use soft threshold", self.soft_min)
        print("phase switch margin", self.phase_switch_margin)
        print("phase switch min steps", self.phase_switch_min_steps)
        print("phase switch use threshold", self.phase_switch_use_threshold)
        print("guidance disable when next is near", self.guidance_disable_when_next_is_near)

        if hasattr(model_cfg, "env") and hasattr(model_cfg.env, "shape_obs"):
            self.shape_obs = model_cfg.env.shape_obs
        else:
            self.shape_obs = None

        wm_normalizer = LinearNormalizer()
        wm_normalizer.load_state_dict(
            torch.load(os.path.join(dynamics_model_dir, "normalizer.pth"))
        )
        self.dyn_model_normalizer = wm_normalizer.to(self.device)
        self.policy_action_normalizer = LinearNormalizer()

        self.abs_action = self.demo_dataset_config.abs_action

        dataset_path = getattr(self.demo_dataset_config, "dataset_path", None)
        if dataset_path is None:
            dataset_path = getattr(self.demo_dataset_config, "expert_dataset_path", None)
        if dataset_path is None:
            raise ValueError(
                "demo_dataset_config must contain dataset_path or expert_dataset_path"
            )

        self.dataset_path = dataset_path
        print("planner dataset_path =", self.dataset_path)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=self.dataset_path)

        self.use_crop = model_cfg.use_crop
        self.original_img_size = model_cfg.original_img_size
        self.cropped_img_size = model_cfg.cropped_img_size
        if self.use_crop:
            self.img_transform = get_eval_crop_transform_resnet(
                original_img_size=self.original_img_size,
                cropped_img_size=self.cropped_img_size,
            )

        self.view_names = model_cfg.view_names
        self.env_name = env_meta["env_name"]
        print("env_name", self.env_name)

        self.frameskip = model_cfg.frameskip
        self.exec_step = action_step
        print("self.demo_dataset_config.horizon", self.demo_dataset_config.horizon)
        self.horizon = self.demo_dataset_config.horizon // 16

        if self.guidance:
            self.get_demo_latents_subgoal_proto()

        self.timestep = 0
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep="rotation_6d"
        )
        self.output_dir = output_dir
        self.idx = 0

        self.corr_log = []
        # self.corr_log_episodes = []
        # self.corr_log_path = Path(self.output_dir) / "subgoal_corr_log.jsonl"
        # self.corr_log_path.parent.mkdir(parents=True, exist_ok=True)
        # open(self.corr_log_path, "a").close()

        # self.corr_summary = []
        # self.corr_summary_path = Path(self.output_dir) / "subgoal_corr_summary.jsonl"
        # open(self.corr_summary_path, "a").close()

        self.rollout_step = 0

    def log_corr(self, rec: dict):
        self.corr_log.append(rec)

    def advance_phase(self):
        # print(f"[advance_phase] {self.subgoal_idx} -> {self.subgoal_idx + 1} at rollout {self.rollout_step}", flush=True)

        n_phases = int(self.phase_proto_visual.shape[0])
        if self.subgoal_idx < n_phases - 1:
            self.subgoal_idx += 1
        self.increment = False
        self.phase_switch_counter = 0

    def reset_phase_state(self):
        self.subgoal_idx = 0
        self.increment = False
        # self.phase_hit_count = 0
        self.phase_switch_counter = 0
        self._subgoal_idx_trace = [0]

    def _softmin_over_set(self, D: torch.Tensor, tau: float, dim: int = 1):
        tau = float(max(tau, 1e-12))
        X = -D / tau
        return (-tau) * torch.logsumexp(X, dim=dim)

    def _flatten_online_latents(self, z_obs):
        v = z_obs["visual"]
        p = z_obs.get("proprio", None)

        if v.ndim > 2:
            v = v.reshape(v.size(0), -1)
            if "Transport" in self.env_name:
                v = v[..., :2048]
            else:
                v = v[..., :1024]

        if p is None:
            p = v.new_zeros((v.size(0), 0))
        elif p.ndim > 2:
            p = p.reshape(p.size(0), -1)

        return v, p

    def _encode_x0_to_vp(self, x0, current_obs):
        a0_norm = x0[:, 1 : 1 + self.horizon * self.frameskip]
        a0_unnorm = self.policy_action_normalizer.unnormalize(a0_norm)
        a0_dyn = self.dyn_model_normalizer["act"].normalize(a0_unnorm)
        action_batch = rearrange(
            a0_dyn, "b (h f) a -> b h (f a)", f=self.frameskip, h=self.horizon
        )

        B = x0.shape[0]
        obs_wm = self.prepare_obs(current_obs, B)

        act_0 = action_batch[:, :1, :]
        z = self.dyn_model.encode(obs_wm, act_0)
        z_pred = self.dyn_model.predict(z)
        z_new = z_pred[:, -1:, ...]
        z_obs, _ = self.dyn_model.separate_emb(z_new)

        v, p = self._flatten_online_latents(z_obs)
        return v, p, z_obs

    def _phase_distance_to_idx(self, v, p, phase_idx):
        device = v.device
        j = int(phase_idx)

        Gv_all = self.phase_proto_visual[j].to(device, non_blocking=True)
        Gp_all = self.phase_proto_proprio[j].to(device, non_blocking=True)
        mask = self.phase_proto_mask[j].to(device, non_blocking=True).bool()

        if mask.any():
            Gv = Gv_all[mask]
            Gp = Gp_all[mask]
        else:
            Gv, Gp = Gv_all, Gp_all

        if v.size(-1) != Gv.size(-1):
            dv = min(v.size(-1), Gv.size(-1))
            v2, Gv2 = v[..., :dv], Gv[..., :dv]
        else:
            v2, Gv2 = v, Gv

        if p.size(-1) != Gp.size(-1):
            dp = min(p.size(-1), Gp.size(-1))
            p2, Gp2 = p[..., :dp], Gp[..., :dp]
        else:
            p2, Gp2 = p, Gp

        if (Gp2.numel() > 0) and (p2.numel() > 0):
            z_full = torch.cat([v2, p2], dim=-1)
            G_full = torch.cat([Gv2, Gp2], dim=-1)
        else:
            z_full, G_full = v2, Gv2

        use_soft = bool(getattr(self, "soft_min", False))

        if not use_soft:
            d = torch.cdist(z_full, G_full, p=2).min(dim=1).values
            thr = float(self.phase_thresholds_raw[j].item())
        else:
            if (
                getattr(self, "softmin_taus_raw", None) is None
                or getattr(self, "phase_thresholds_soft_raw", None) is None
            ):
                raise RuntimeError("soft_min=True but softmin taus/thresholds are missing.")
            tau_phase = float(self.softmin_taus_raw[j].item())
            D = torch.cdist(z_full, G_full, p=2)
            d = self._softmin_over_set(D, tau_phase)
            thr = float(self.phase_thresholds_soft_raw[j].item())

        return float(d.mean().item()), thr

    def _score_vp_against_phase(self, v, p, phase_idx: int):
        """
        Same scoring logic as score_x0_against_phase, but takes precomputed
        (v, p) latents so the dynamics-model forward pass can be shared across
        multiple phase comparisons.
        """
        device = v.device
        n_phases = int(self.phase_proto_visual.shape[0])
        j = int(max(0, min(int(phase_idx), n_phases - 1)))

        Gv_all = self.phase_proto_visual[j].to(device, non_blocking=True)
        Gp_all = self.phase_proto_proprio[j].to(device, non_blocking=True)
        mask = self.phase_proto_mask[j].to(device, non_blocking=True).bool()

        if mask.any():
            Gv = Gv_all[mask]
            Gp = Gp_all[mask]
        else:
            Gv, Gp = Gv_all, Gp_all

        if v.size(-1) != Gv.size(-1):
            dv = min(v.size(-1), Gv.size(-1))
            v2, Gv2 = v[..., :dv], Gv[..., :dv]
        else:
            v2, Gv2 = v, Gv

        if p.size(-1) != Gp.size(-1):
            dp = min(p.size(-1), Gp.size(-1))
            p2, Gp2 = p[..., :dp], Gp[..., :dp]
        else:
            p2, Gp2 = p, Gp

        if (Gp2.numel() > 0) and (p2.numel() > 0):
            z_full = torch.cat([v2, p2], dim=-1)
            G_full = torch.cat([Gv2, Gp2], dim=-1)
        else:
            z_full, G_full = v2, Gv2

        D_hard = torch.cdist(z_full, G_full, p=2)
        d_hard = D_hard.min(dim=1).values

        thr_pair = self.get_phase_threshold_pair(j)
        thr_hard_switch = float(thr_pair["switch_thr"])
        thr_hard_guide_lower = float(thr_pair["guide_thr_lower"])
        thr_hard_guide_upper = thr_pair.get("guide_thr_upper", None)

        use_soft = bool(thr_pair["use_soft"])

        d_soft = None
        thr_soft_switch = None
        thr_soft_guide_lower = None
        thr_soft_guide_upper = None
        if use_soft:
            if (
                getattr(self, "softmin_taus_raw", None) is None
                or getattr(self, "phase_thresholds_soft_raw", None) is None
            ):
                raise RuntimeError("soft_min=True but softmin taus/thresholds are missing.")
            tau_phase = float(self.softmin_taus_raw[j].item())
            d_soft_t = self._softmin_over_set(D_hard, tau_phase)
            d_soft = float(d_soft_t.mean().item())
            thr_soft_switch = float(self.phase_thresholds_soft_raw[j].item())
            thr_soft_guide_lower = float(self.guidance_phase_thresholds_lower_soft_raw[j].item())
            thr_soft_guide_upper = (
                float(self.guidance_phase_thresholds_upper_soft_raw[j].item())
                if getattr(self, "guidance_phase_thresholds_upper_soft_raw", None) is not None
                else None
            )

        return {
            "phase": int(j),
            "d_hard": float(d_hard.mean().item()),
            "switch_thr_hard": float(thr_hard_switch),
            "guide_thr_lower_hard": float(thr_hard_guide_lower),
            "guide_thr_upper_hard": None if thr_hard_guide_upper is None else float(thr_hard_guide_upper),
            "margin_switch_hard": float(thr_hard_switch - float(d_hard.mean().item())),
            "margin_guide_lower_hard": float(thr_hard_guide_lower - float(d_hard.mean().item())),
            "d_soft": None if d_soft is None else float(d_soft),
            "switch_thr_soft": None if d_soft is None else float(thr_soft_switch),
            "guide_thr_lower_soft": None if d_soft is None else float(thr_soft_guide_lower),
            "guide_thr_upper_soft": (
                None if d_soft is None or thr_soft_guide_upper is None
                else float(thr_soft_guide_upper)
            ),
            "margin_switch_soft": None if d_soft is None else float(thr_soft_switch - d_soft),
            "margin_guide_lower_soft": None if d_soft is None else float(thr_soft_guide_lower - d_soft),
            "use_soft": bool(use_soft),
        }

    def _compute_global_phase_diagnostics(self, x0, current_obs):
        """
        Compute distance to every phase's prototypes, return diagnostics
        for three-case analysis.
        """
        v, p, _ = self._encode_x0_to_vp(x0, current_obs)
        n_phases = int(self.phase_proto_visual.shape[0])

        d_per_phase = []
        thr_per_phase = []
        for j in range(n_phases):
            d_j, thr_j = self._phase_distance_to_idx(v, p, j)
            d_per_phase.append(float(d_j))
            thr_per_phase.append(float(thr_j))

        d_arr = torch.tensor(d_per_phase)
        argmin_phase = int(torch.argmin(d_arr).item())
        d_global_min = float(d_arr.min().item())

        return {
            "d_per_phase": d_per_phase,
            "thr_per_phase": thr_per_phase,
            "d_global_min": d_global_min,
            "argmin_phase": argmin_phase,
        }


    def get_phase_threshold_pair(self, phase_idx: int):
        j = int(phase_idx)
        use_soft = bool(getattr(self, "soft_min", False))

        if not use_soft:
            switch_thr = float(self.phase_thresholds_raw[j].item())
            guide_thr_lower = float(self.guidance_phase_thresholds_lower_raw[j].item())
            guide_thr_upper = (
                float(self.guidance_phase_thresholds_upper_raw[j].item())
                if getattr(self, "guidance_phase_thresholds_upper_raw", None) is not None
                else None
            )
        else:
            if (
                self.phase_thresholds_soft_raw is None
                or self.guidance_phase_thresholds_lower_soft_raw is None
            ):
                raise RuntimeError("soft_min=True but soft thresholds are missing.")
            switch_thr = float(self.phase_thresholds_soft_raw[j].item())
            guide_thr_lower = float(self.guidance_phase_thresholds_lower_soft_raw[j].item())
            guide_thr_upper = (
                float(self.guidance_phase_thresholds_upper_soft_raw[j].item())
                if getattr(self, "guidance_phase_thresholds_upper_soft_raw", None) is not None
                else None
            )

        return {
            "switch_thr": switch_thr,
            "guide_thr_lower": guide_thr_lower,
            "guide_thr_upper": guide_thr_upper,
            "use_soft": use_soft,
        }

    def score_x0_against_phase(self, x0, current_obs, phase_idx: int):
        """
        Convenience wrapper kept for backwards compatibility. Encodes x0 then
        delegates to _score_vp_against_phase. Callers that need to score against
        multiple phases should encode once and call _score_vp_against_phase
        directly to avoid redundant dynamics-model forward passes.
        """
        v, p, _ = self._encode_x0_to_vp(x0, current_obs)
        return self._score_vp_against_phase(v, p, phase_idx)

    def compare_current_next_phase(self, x0, current_obs, phase_idx=None):
        if phase_idx is None:
            phase_idx = int(getattr(self, "subgoal_idx", 0))

        n_phases = int(self.phase_proto_visual.shape[0])
        cur = int(max(0, min(phase_idx, n_phases - 1)))
        nxt = int(min(cur + 1, n_phases - 1))

        cur_info = self.score_x0_against_phase(x0, current_obs, cur)
        nxt_info = self.score_x0_against_phase(x0, current_obs, nxt)

        use_soft = bool(cur_info["use_soft"])

        if use_soft:
            d_cur = cur_info["d_soft"]
            d_next = nxt_info["d_soft"]
            margin_cur = cur_info["margin_switch_soft"]
            margin_next = nxt_info["margin_switch_soft"]
            thr_cur = cur_info["switch_thr_soft"]
            thr_next = nxt_info["switch_thr_soft"]
            guide_thr_lower_cur = cur_info["guide_thr_lower_soft"]
            guide_thr_lower_next = nxt_info["guide_thr_lower_soft"]
            guide_thr_upper_cur = cur_info["guide_thr_upper_soft"]
            guide_thr_upper_next = nxt_info["guide_thr_upper_soft"]
            guide_margin_cur = cur_info["margin_guide_lower_soft"]
            guide_margin_next = nxt_info["margin_guide_lower_soft"]
        else:
            d_cur = cur_info["d_hard"]
            d_next = nxt_info["d_hard"]
            margin_cur = cur_info["margin_switch_hard"]
            margin_next = nxt_info["margin_switch_hard"]
            thr_cur = cur_info["switch_thr_hard"]
            thr_next = nxt_info["switch_thr_hard"]
            guide_thr_lower_cur = cur_info["guide_thr_lower_hard"]
            guide_thr_lower_next = nxt_info["guide_thr_lower_hard"]
            guide_thr_upper_cur = cur_info["guide_thr_upper_hard"]
            guide_thr_upper_next = nxt_info["guide_thr_upper_hard"]
            guide_margin_cur = cur_info["margin_guide_lower_hard"]
            guide_margin_next = nxt_info["margin_guide_lower_hard"]

        return {
            "phase_cur": int(cur),
            "phase_next": int(nxt),
            "use_soft": bool(use_soft),
            "d_cur": float(d_cur),
            "thr_cur": float(thr_cur),
            "margin_cur": float(margin_cur),
            "guide_thr_lower_cur": float(guide_thr_lower_cur),
            "guide_thr_upper_cur": None if guide_thr_upper_cur is None else float(guide_thr_upper_cur),
            "guide_margin_cur": float(guide_margin_cur),
            "d_next": float(d_next),
            "thr_next": float(thr_next),
            "margin_next": float(margin_next),
            "guide_thr_lower_next": float(guide_thr_lower_next),
            "guide_thr_upper_next": None if guide_thr_upper_next is None else float(guide_thr_upper_next),
            "guide_margin_next": float(guide_margin_next),
            "next_minus_cur_margin": float(margin_next - margin_cur),
            "cur_minus_next_dist": float(d_cur - d_next),
        }

    def decide_guidance_and_phase(self, x0, current_obs, phase_idx=None, log=False, update_state=True):
        if phase_idx is None:
            phase_idx = int(getattr(self, "subgoal_idx", 0))

        n_phases = int(self.phase_proto_visual.shape[0])
        cur = int(max(0, min(phase_idx, n_phases - 1)))
        nxt = int(min(cur + 1, n_phases - 1))

        # ----- Single dynamics-model forward pass (was 3 before) -----
        v, p, _ = self._encode_x0_to_vp(x0, current_obs)

        # Score against every phase using the same (v, p) — cheap, just cdist per phase
        per_phase_info = [self._score_vp_against_phase(v, p, j) for j in range(n_phases)]
        cur_info = per_phase_info[cur]
        nxt_info = per_phase_info[nxt]

        use_soft = bool(cur_info["use_soft"])

        # Extract current/next phase distances and thresholds (same logic as
        # compare_current_next_phase, just inlined)
        if use_soft:
            d_cur = cur_info["d_soft"]
            d_next = nxt_info["d_soft"]
            thr_cur = cur_info["switch_thr_soft"]
            thr_next = nxt_info["switch_thr_soft"]
            guide_thr_lower_cur = cur_info["guide_thr_lower_soft"]
            guide_thr_upper_cur = cur_info["guide_thr_upper_soft"]
        else:
            d_cur = cur_info["d_hard"]
            d_next = nxt_info["d_hard"]
            thr_cur = cur_info["switch_thr_hard"]
            thr_next = nxt_info["switch_thr_hard"]
            guide_thr_lower_cur = cur_info["guide_thr_lower_hard"]
            guide_thr_upper_cur = cur_info["guide_thr_upper_hard"]

        d_cur = float(d_cur)
        d_next = float(d_next) if d_next is not None else None
        thr_next = float(thr_next) if thr_next is not None else None
        switch_thr_cur = float(thr_cur)
        guide_thr_lower_cur = float(guide_thr_lower_cur)

        margin_cur_switch = switch_thr_cur - d_cur
        margin_cur_guide_lower = guide_thr_lower_cur - d_cur

        # Sign convention: distances are negative; larger d (less negative) = farther from prototype.
        # d_cur > guide_thr_lower_cur means "drifted past the lower bound" -> Case 2 (guidance on)
        # d_cur > guide_thr_upper_cur means "drifted past the upper bound" -> Case 3 (guidance off)
        
        apply_guidance = (d_cur > guide_thr_lower_cur)
        # apply_guidance = True

        if guide_thr_upper_cur is not None:
            apply_guidance = apply_guidance and (d_cur <= float(guide_thr_upper_cur))

        next_is_near = False
        if (d_next is not None) and (thr_next is not None):
            next_is_near = (d_next <= thr_next)

        if self.guidance_disable_when_next_is_near and next_is_near:
            apply_guidance = False

        switch_phase = False
        better_than_current = False
        acceptable_next = False

        if cur < n_phases - 1 and d_next is not None:
            better_than_current = (d_next < (d_cur - self.phase_switch_margin))
            acceptable_next = (d_next < thr_next) if self.phase_switch_use_threshold else True

            if update_state:
                if better_than_current and acceptable_next:
                    self.phase_switch_counter += 1
                else:
                    self.phase_switch_counter = 0

                if self.phase_switch_counter >= self.phase_switch_min_steps:
                    switch_phase = True
            else:
                switch_phase = False
        else:
            if update_state:
                self.phase_switch_counter = 0

        # Build global diagnostics from per_phase_info (no extra encoding needed)
        if use_soft:
            d_per_phase = [info["d_soft"] for info in per_phase_info]
            thr_per_phase = [info["switch_thr_soft"] for info in per_phase_info]
        else:
            d_per_phase = [info["d_hard"] for info in per_phase_info]
            thr_per_phase = [info["switch_thr_hard"] for info in per_phase_info]

        d_arr = np.asarray(d_per_phase, dtype=np.float32)
        argmin_phase = int(d_arr.argmin())
        d_global_min = float(d_arr.min())

        result = {
            "phase_cur": cur,
            "phase_next": nxt,
            "apply_guidance": bool(apply_guidance),
            "switch_phase": bool(switch_phase),
            "d_cur": float(d_cur),
            "guide_thr_lower_cur": float(guide_thr_lower_cur),
            "guide_thr_upper_cur": None if guide_thr_upper_cur is None else float(guide_thr_upper_cur),
            "switch_thr_cur": float(switch_thr_cur),
            "margin_cur_guide_lower": float(margin_cur_guide_lower),
            "margin_cur_switch": float(margin_cur_switch),
            "d_next": None if d_next is None else float(d_next),
            "thr_next": None if thr_next is None else float(thr_next),
            "better_than_current": bool(better_than_current),
            "acceptable_next": bool(acceptable_next),
            "next_is_near": bool(next_is_near),
            "phase_switch_counter": int(self.phase_switch_counter),
            "d_global_min": float(d_global_min),
            "argmin_phase": int(argmin_phase),
            "d_per_phase": [float(x) for x in d_per_phase],
            "thr_per_phase": [float(x) for x in thr_per_phase],
        }

        log_record = dict(result)
        log_record["rollout_step"] = int(getattr(self, "rollout_step", 0))
        log_record["phase"] = int(cur)
        if log:
            self.log_corr(log_record)
        return result

    def score_x0_proto(self, x0, current_obs, w_visual: float = 1.0, w_proprio: float = 0.2):
        cmp_info = self.compare_current_next_phase(x0, current_obs)
        return cmp_info["d_cur"], cmp_info["thr_cur"], cmp_info["phase_cur"]

    def _phase_distance_and_margin(self, x0, current_obs):
        cmp_info = self.compare_current_next_phase(x0, current_obs)
        d = cmp_info["d_cur"]
        tau = cmp_info["thr_cur"]
        phase = cmp_info["phase_cur"]
        margin = float(tau - d)
        return {
            "phase": int(phase),
            "d": float(d),
            "tau": float(tau),
            "margin": float(margin),
            "support_near": int(d <= tau),
            "d_next": cmp_info["d_next"],
            "thr_next": cmp_info["thr_next"],
            "margin_next": cmp_info["margin_next"],
        }

    def compute_subgoal_reward_vp_proto(
        self,
        current_visual_latent,
        current_proprio_latent,
        w_visual: float = 1.0,
        w_proprio: float = 0.2,
    ):
        if len(current_visual_latent.shape) > 2:
            v = current_visual_latent.reshape(current_visual_latent.size(0), -1)
            if "ToolHang" in self.env_name:
                v = v[..., 512:]
            elif "Square" in self.env_name:
                v = v[..., :1024]
            elif "Transport" in self.env_name:
                v = v[..., :2048]
        else:
            v = current_visual_latent

        if len(current_proprio_latent.shape) > 2:
            p = current_proprio_latent.reshape(current_proprio_latent.size(0), -1)
        else:
            p = current_proprio_latent

        device = v.device
        B = v.size(0)

        n_phases = int(self.phase_proto_visual.shape[0])
        j = int(max(0, min(int(self.subgoal_idx), n_phases - 1)))
        self.subgoal_idx = j

        Gv_all = self.phase_proto_visual[j].to(device, non_blocking=True)
        Gp_all = self.phase_proto_proprio[j].to(device, non_blocking=True)
        mask = self.phase_proto_mask[j].to(device, non_blocking=True).bool()

        if mask.any():
            Gv = Gv_all[mask]
            Gp = Gp_all[mask]
        else:
            Gv, Gp = Gv_all, Gp_all

        if v.size(-1) != Gv.size(-1):
            dv = min(v.size(-1), Gv.size(-1))
            v2, Gv2 = v[..., :dv], Gv[..., :dv]
        else:
            v2, Gv2 = v, Gv

        if p.size(-1) != Gp.size(-1):
            dp = min(p.size(-1), Gp.size(-1))
            p2, Gp2 = p[..., :dp], Gp[..., :dp]
        else:
            p2, Gp2 = p, Gp

        use_soft = bool(getattr(self, "soft_min", False))

        if not use_soft:
            dist_v = torch.cdist(v2, Gv2, p=2).min(dim=1).values
            if (Gp2.numel() > 0) and (p2.numel() > 0):
                dist_p = torch.cdist(p2, Gp2, p=2).min(dim=1).values
            else:
                dist_p = v2.new_zeros((B,))
        else:
            temp_v = float(getattr(self, "proto_softmin_temp_v", 0.05))
            temp_p = float(getattr(self, "proto_softmin_temp_p", 0.05))

            Dv_mat = torch.cdist(v2, Gv2, p=2)
            dist_v = self._softmin_over_set(Dv_mat, temp_v)

            if (Gp2.numel() > 0) and (p2.numel() > 0):
                Dp_mat = torch.cdist(p2, Gp2, p=2)
                dist_p = self._softmin_over_set(Dp_mat, temp_p)
            else:
                dist_p = v2.new_zeros((B,))

        dist_total = (w_visual * dist_v) + (w_proprio * dist_p)
        reward = -dist_total
        return reward

    def compute_loss_mpgd_subgoal_vp_proto(
        self,
        x0,
        current_obs,
        w_visual: float = 1.0,
        w_proprio: float = 0.2,
    ):
        a0_norm = x0[:, 1 : 1 + self.horizon * self.frameskip]
        a0_unnorm = self.policy_action_normalizer.unnormalize(a0_norm)
        a0_dyn = self.dyn_model_normalizer["act"].normalize(a0_unnorm)

        action_batch = rearrange(
            a0_dyn, "b (h f) a -> b h (f a)", f=self.frameskip, h=self.horizon
        )

        B = x0.shape[0]
        obs_wm = self.prepare_obs(current_obs, B)

        act_0 = action_batch[:, :1, :]
        z = self.dyn_model.encode(obs_wm, act_0)
        z_pred = self.dyn_model.predict(z)
        z_new = z_pred[:, -1:, ...]
        z_obs, _ = self.dyn_model.separate_emb(z_new)

        reward = self.compute_subgoal_reward_vp_proto(
            z_obs["visual"],
            z_obs["proprio"],
            w_visual=w_visual,
            w_proprio=w_proprio,
        )

        loss = (-reward).mean()
        return loss

    def prepare_obs(self, current_obs, action_shape):
        proprio_arrays = []

        for key, meta in self.shape_obs.items():
            if meta.get("type") == "rgb" or "image" in key:
                continue
            if key in current_obs:
                proprio_arrays.append(current_obs[key])

        proprio = (
            torch.cat(proprio_arrays, dim=-1).to("cuda")
            if proprio_arrays
            else torch.zeros((1, 0)).to("cuda")
        )

        visual = {}
        for view_name in self.view_names:
            visual[view_name] = current_obs[view_name].to("cuda")
            visual[view_name] = self.dyn_model_normalizer[view_name].normalize(
                visual[view_name].to("cuda")
            )
            visual[view_name] = self.img_transform(
                visual[view_name].view(
                    -1, 3, self.original_img_size, self.original_img_size
                )
            )
            visual[view_name] = visual[view_name].view(
                -1, 1, 3, self.cropped_img_size, self.cropped_img_size
            )

        current_obs_wm = {"visual": visual, "proprio": proprio}
        current_obs_wm["proprio"] = self.dyn_model_normalizer["state"].normalize(
            current_obs_wm["proprio"]
        )

        current_obs_wm["proprio"] = current_obs_wm["proprio"].expand(action_shape, -1, -1)
        current_obs_wm["visual"] = {
            key: value.expand(action_shape, -1, -1, -1, -1)
            for key, value in current_obs_wm["visual"].items()
        }
        return current_obs_wm

    def get_demo_latents_subgoal_proto(self):
        if not hasattr(self, "pcg_data_path") or self.pcg_data_path is None:
            raise ValueError("Missing self.pcg_data_path.")
        if not hasattr(self, "device"):
            raise ValueError("Missing self.device.")
        if not hasattr(self, "latent_dir") or self.latent_dir is None:
            raise ValueError("Missing self.latent_dir.")

        Pkeep = int(getattr(self, "targets_num", 40))
        thr_p = int(getattr(self, "threshold_percent", 70))
        tau_p = int(getattr(self, "tau", 10))
        guide_thr_lower_p = int(getattr(self, "guidance_threshold_lower_percent", thr_p))
        guide_thr_upper_p = getattr(self, "guidance_threshold_upper_percent", None)

        if not (0 < tau_p < 100):
            raise ValueError("self.tau must be in (0,100)")
        if Pkeep <= 0:
            raise ValueError("self.targets_num must be > 0")
        if not (0 < thr_p < 100):
            raise ValueError("self.threshold_percent must be in (0,100)")
        if not (0 < guide_thr_lower_p < 100):
            raise ValueError("self.guidance_threshold_lower_percent must be in (0,100)")
        if guide_thr_upper_p is not None and not (0 < int(guide_thr_upper_p) < 100):
            raise ValueError("self.guidance_threshold_upper_percent must be in (0,100)")
        if guide_thr_upper_p is not None and int(guide_thr_upper_p) <= guide_thr_lower_p:
            raise ValueError(
                f"guidance_threshold_upper_perc ({guide_thr_upper_p}) must be > "
                f"guidance_threshold_lower_perc ({guide_thr_lower_p})"
            )

        TARGET_LATENTS_PATH = f"{self.latent_dir}/target_latents"
        TARGET_OFFSETS_PATH = f"{self.latent_dir}/target_offsets"
        TARGET_TIMESTEPS_PATH = f"{self.latent_dir}/target_timesteps"

        THR_PCTS_PATH = f"{self.latent_dir}/threshold_percentiles"
        THR_L2_MAT_PATH = f"{self.latent_dir}/thresholds_l2"
        THR_Z_MAT_PATH = f"{self.latent_dir}/thresholds_zscore"

        MEAN_PATH = f"{self.latent_dir}/latent_mean"
        STD_PATH = f"{self.latent_dir}/latent_std"

        SOFT_TAU_PCTS_PATH = f"{self.latent_dir}/softmin_tau_percentiles"
        SOFT_TAUS_RAW_PATH = f"{self.latent_dir}/softmin_taus_raw_per_phase"
        SOFT_TAUS_Z_PATH = f"{self.latent_dir}/softmin_taus_z_per_phase"

        SOFT_THR_L2_PATH = f"{self.latent_dir}/thresholds_softmin_l2"
        SOFT_THR_Z_PATH = f"{self.latent_dir}/thresholds_softmin_zscore"

        with h5py.File(self.pcg_data_path, "r") as f:
            target_latents_np = f[TARGET_LATENTS_PATH][:]
            target_offsets_np = f[TARGET_OFFSETS_PATH][:].astype(np.int32)
            target_timesteps_np = f[TARGET_TIMESTEPS_PATH][:].astype(np.int32)

            if THR_PCTS_PATH not in f or THR_L2_MAT_PATH not in f:
                raise KeyError(
                    f"Missing {THR_PCTS_PATH} or {THR_L2_MAT_PATH} in {self.pcg_data_path}"
                )

            pcts = f[THR_PCTS_PATH][:].astype(np.int32)
            thr_l2_mat = f[THR_L2_MAT_PATH][:].astype(np.float32)

            where = np.where(pcts == thr_p)[0]
            if where.size == 0:
                raise ValueError(
                    f"threshold_percent={thr_p} not found. Available: {pcts.tolist()}"
                )
            col = int(where[0])
            thr_l2 = thr_l2_mat[:, col].astype(np.float32)

            # where_g = np.where(pcts == guide_thr_p)[0]
            # if where_g.size == 0:
            #     raise ValueError(
            #         f"guidance_threshold_percent={guide_thr_p} not found. Available: {pcts.tolist()}"
            #     )
            # guide_col = int(where_g[0])
            # guide_thr_l2 = thr_l2_mat[:, guide_col].astype(np.float32)
            # Lower guidance threshold (Case 1 -> Case 2 boundary)
            where_g_lower = np.where(pcts == guide_thr_lower_p)[0]
            if where_g_lower.size == 0:
                raise ValueError(
                    f"guidance_threshold_lower_perc={guide_thr_lower_p} not found. "
                    f"Available: {pcts.tolist()}"
                )
            guide_lower_col = int(where_g_lower[0])
            guide_lower_thr_l2 = thr_l2_mat[:, guide_lower_col].astype(np.float32)

            # Upper guidance threshold (Case 2 -> Case 3 boundary), optional
            guide_upper_col = None
            guide_upper_thr_l2 = None
            if guide_thr_upper_p is not None:
                where_g_upper = np.where(pcts == int(guide_thr_upper_p))[0]
                if where_g_upper.size == 0:
                    raise ValueError(
                        f"guidance_threshold_upper_perc={guide_thr_upper_p} not found. "
                        f"Available: {pcts.tolist()}"
                    )
                guide_upper_col = int(where_g_upper[0])
                guide_upper_thr_l2 = thr_l2_mat[:, guide_upper_col].astype(np.float32)

            # thr_z = None
            # guide_thr_z = None
            # if THR_Z_MAT_PATH in f:
            #     thr_z_mat = f[THR_Z_MAT_PATH][:].astype(np.float32)
            #     if thr_z_mat.shape != thr_l2_mat.shape:
            #         raise RuntimeError(
            #             f"thresholds_zscore shape {thr_z_mat.shape} != thresholds_l2 shape {thr_l2_mat.shape}"
            #         )
            #     thr_z = thr_z_mat[:, col].astype(np.float32)
            #     guide_thr_z = thr_z_mat[:, guide_col].astype(np.float32)

            thr_z = None
            guide_lower_thr_z = None
            guide_upper_thr_z = None
            if THR_Z_MAT_PATH in f:
                thr_z_mat = f[THR_Z_MAT_PATH][:].astype(np.float32)
                if thr_z_mat.shape != thr_l2_mat.shape:
                    raise RuntimeError(
                        f"thresholds_zscore shape {thr_z_mat.shape} != thresholds_l2 shape {thr_l2_mat.shape}"
                    )
                thr_z = thr_z_mat[:, col].astype(np.float32)
                guide_lower_thr_z = thr_z_mat[:, guide_lower_col].astype(np.float32)
                if guide_upper_col is not None:
                    guide_upper_thr_z = thr_z_mat[:, guide_upper_col].astype(np.float32)

            soft = None
            if (
                SOFT_TAU_PCTS_PATH in f
                and SOFT_TAUS_RAW_PATH in f
                and SOFT_THR_L2_PATH in f
            ):
                tau_pcts = f[SOFT_TAU_PCTS_PATH][:].astype(np.int32)
                taus_raw_per_phase_np = f[SOFT_TAUS_RAW_PATH][:].astype(np.float32)
                thr_soft_l2_np = f[SOFT_THR_L2_PATH][:].astype(np.float32)

                wtau = np.where(tau_pcts == tau_p)[0]
                if wtau.size == 0:
                    raise ValueError(
                        f"tau={tau_p} not found. Available tau percentiles: {tau_pcts.tolist()}"
                    )
                tau_col = int(wtau[0])

                taus_raw_sel = taus_raw_per_phase_np[:, tau_col].astype(np.float32)
                # thr_soft_raw_sel = thr_soft_l2_np[:, tau_col, col].astype(np.float32)
                # guide_thr_soft_raw_sel = thr_soft_l2_np[:, tau_col, guide_col].astype(
                #     np.float32
                # )
                thr_soft_raw_sel = thr_soft_l2_np[:, tau_col, col].astype(np.float32)
                guide_lower_thr_soft_raw_sel = thr_soft_l2_np[:, tau_col, guide_lower_col].astype(
                    np.float32
                )
                guide_upper_thr_soft_raw_sel = None
                if guide_upper_col is not None:
                    guide_upper_thr_soft_raw_sel = thr_soft_l2_np[:, tau_col, guide_upper_col].astype(
                        np.float32
                    )

                # taus_z_sel = None
                # thr_soft_z_sel = None
                # guide_thr_soft_z_sel = None
                # if (SOFT_TAUS_Z_PATH in f) and (SOFT_THR_Z_PATH in f):
                #     taus_z_per_phase_np = f[SOFT_TAUS_Z_PATH][:].astype(np.float32)
                #     thr_soft_z_np = f[SOFT_THR_Z_PATH][:].astype(np.float32)
                #     taus_z_sel = taus_z_per_phase_np[:, tau_col].astype(np.float32)
                #     thr_soft_z_sel = thr_soft_z_np[:, tau_col, col].astype(np.float32)
                #     guide_thr_soft_z_sel = thr_soft_z_np[:, tau_col, guide_col].astype(
                #         np.float32
                #     )
                taus_z_sel = None
                thr_soft_z_sel = None
                guide_lower_thr_soft_z_sel = None
                guide_upper_thr_soft_z_sel = None
                if (SOFT_TAUS_Z_PATH in f) and (SOFT_THR_Z_PATH in f):
                    taus_z_per_phase_np = f[SOFT_TAUS_Z_PATH][:].astype(np.float32)
                    thr_soft_z_np = f[SOFT_THR_Z_PATH][:].astype(np.float32)
                    taus_z_sel = taus_z_per_phase_np[:, tau_col].astype(np.float32)
                    thr_soft_z_sel = thr_soft_z_np[:, tau_col, col].astype(np.float32)
                    guide_lower_thr_soft_z_sel = thr_soft_z_np[:, tau_col, guide_lower_col].astype(
                        np.float32
                    )
                    if guide_upper_col is not None:
                        guide_upper_thr_soft_z_sel = thr_soft_z_np[:, tau_col, guide_upper_col].astype(
                            np.float32
                        )

                # soft = dict(
                #     tau_percentiles=tau_pcts,
                #     tau=int(tau_p),
                #     tau_col=int(tau_col),
                #     taus_raw=taus_raw_sel,
                #     thr_soft_raw=thr_soft_raw_sel,
                #     guide_thr_soft_raw=guide_thr_soft_raw_sel,
                #     taus_z=taus_z_sel,
                #     thr_soft_z=thr_soft_z_sel,
                #     guide_thr_soft_z=guide_thr_soft_z_sel,
                # )
                soft = dict(
                    tau_percentiles=tau_pcts,
                    tau=int(tau_p),
                    tau_col=int(tau_col),
                    taus_raw=taus_raw_sel,
                    thr_soft_raw=thr_soft_raw_sel,
                    guide_lower_thr_soft_raw=guide_lower_thr_soft_raw_sel,
                    guide_upper_thr_soft_raw=guide_upper_thr_soft_raw_sel,
                    taus_z=taus_z_sel,
                    thr_soft_z=thr_soft_z_sel,
                    guide_lower_thr_soft_z=guide_lower_thr_soft_z_sel,
                    guide_upper_thr_soft_z=guide_upper_thr_soft_z_sel,
                )

            mean_np = f[MEAN_PATH][:].astype(np.float32) if MEAN_PATH in f else None
            std_np = f[STD_PATH][:].astype(np.float32) if STD_PATH in f else None

            n_phases = int(len(target_offsets_np) - 1)

            phase_step_p10 = np.zeros((n_phases,), dtype=np.float32)
            phase_step_p50 = np.zeros((n_phases,), dtype=np.float32)
            phase_step_p90 = np.zeros((n_phases,), dtype=np.float32)
            phase_step_span = np.zeros((n_phases,), dtype=np.float32)
            stall_k_per_phase = np.zeros((n_phases,), dtype=np.int32)

            stall_alpha = 0.35
            stall_k_min = 10
            stall_k_max = 40

            for j in range(n_phases):
                s = int(target_offsets_np[j])
                e = int(target_offsets_np[j + 1])

                if e <= s:
                    stall_k_per_phase[j] = stall_k_min
                    continue

                ts = target_timesteps_np[s:e].astype(np.float32)

                p10 = float(np.percentile(ts, 10))
                p50 = float(np.percentile(ts, 50))
                p90 = float(np.percentile(ts, 90))
                span = max(0.0, p90 - p10)

                phase_step_p10[j] = p10
                phase_step_p50[j] = p50
                phase_step_p90[j] = p90
                phase_step_span[j] = span

                k = int(np.floor(stall_alpha * span))
                k = max(stall_k_min, min(k, stall_k_max))
                stall_k_per_phase[j] = k

            self.target_timesteps = torch.from_numpy(target_timesteps_np).to(self.device)
            self.phase_step_p10 = torch.from_numpy(phase_step_p10).float().to(self.device)
            self.phase_step_p50 = torch.from_numpy(phase_step_p50).float().to(self.device)
            self.phase_step_p90 = torch.from_numpy(phase_step_p90).float().to(self.device)
            self.phase_step_span = torch.from_numpy(phase_step_span).float().to(self.device)
            self.stall_k_per_phase = torch.from_numpy(stall_k_per_phase).to(self.device)

        latents = torch.from_numpy(target_latents_np).float().to(self.device)
        if latents.ndim != 2:
            latents = latents.reshape(latents.shape[0], -1)
        N, D = latents.shape

        if D == 2080:
            visual = latents[:, :2048]
            proprio = latents[:, 2048:]
        elif D == 2048:
            visual = latents
            proprio = latents.new_zeros((N, 0))
        elif D == 1056:
            visual = latents[:, :1024]
            proprio = latents[:, 1024:]
        elif D == 1040:
            visual = latents[:, :1024]
            proprio = latents[:, 1024:]
        elif D == 1024:
            visual = latents
            proprio = latents.new_zeros((N, 0))
        else:
            raise RuntimeError(
                f"Unexpected latent dim D={D}. Expected 2080, 2048, 1056, 1040, or 1024."
            )

        target_offsets = torch.from_numpy(target_offsets_np).to(self.device)
        n_phases = int(target_offsets.numel() - 1)
        if n_phases <= 0:
            raise RuntimeError("target_offsets invalid (n_phases <= 0).")

        self.n_phases = n_phases

        phase_proto_latents = latents.new_zeros((n_phases, Pkeep, D))
        phase_proto_mask = torch.zeros((n_phases, Pkeep), dtype=torch.bool, device=self.device)

        Dv = int(visual.shape[1])
        Dp = int(proprio.shape[1])
        phase_proto_visual = visual.new_zeros((n_phases, Pkeep, Dv))
        phase_proto_proprio = proprio.new_zeros((n_phases, Pkeep, Dp))

        for j in range(n_phases):
            s = int(target_offsets[j].item())
            e = int(target_offsets[j + 1].item())
            if e <= s:
                continue
            count = e - s
            take = min(Pkeep, count)

            phase_proto_latents[j, :take] = latents[s : s + take]
            phase_proto_visual[j, :take] = visual[s : s + take]
            if Dp > 0:
                phase_proto_proprio[j, :take] = proprio[s : s + take]
            phase_proto_mask[j, :take] = True

        self.phase_proto_latents = phase_proto_latents.contiguous()
        self.phase_proto_visual = phase_proto_visual.contiguous()
        self.phase_proto_proprio = phase_proto_proprio.contiguous()
        self.phase_proto_mask = phase_proto_mask.contiguous()

        # self.threshold_percentiles = torch.from_numpy(pcts).to(self.device)
        # self.threshold_col = int(col)
        # self.guidance_threshold_col = int(guide_col)
        self.threshold_percentiles = torch.from_numpy(pcts).to(self.device)
        self.threshold_col = int(col)
        self.guidance_threshold_lower_col = int(guide_lower_col)
        self.guidance_threshold_upper_col = (
            int(guide_upper_col) if guide_upper_col is not None else None
        )

        self.phase_thresholds_raw = (
            torch.from_numpy(thr_l2).float().to(self.device).contiguous()
        )
        self.phase_thresholds_z = (
            torch.from_numpy(thr_z).float().to(self.device).contiguous()
            if thr_z is not None
            else None
        )

        # self.guidance_phase_thresholds_raw = (
        #     torch.from_numpy(guide_thr_l2).float().to(self.device).contiguous()
        # )
        # self.guidance_phase_thresholds_z = (
        #     torch.from_numpy(guide_thr_z).float().to(self.device).contiguous()
        #     if guide_thr_z is not None
        #     else None
        # )
        # Lower bound (Case 1 -> Case 2)
        self.guidance_phase_thresholds_lower_raw = (
            torch.from_numpy(guide_lower_thr_l2).float().to(self.device).contiguous()
        )
        self.guidance_phase_thresholds_lower_z = (
            torch.from_numpy(guide_lower_thr_z).float().to(self.device).contiguous()
            if guide_lower_thr_z is not None
            else None
        )
        # Upper bound (Case 2 -> Case 3), optional
        self.guidance_phase_thresholds_upper_raw = (
            torch.from_numpy(guide_upper_thr_l2).float().to(self.device).contiguous()
            if guide_upper_thr_l2 is not None
            else None
        )
        self.guidance_phase_thresholds_upper_z = (
            torch.from_numpy(guide_upper_thr_z).float().to(self.device).contiguous()
            if guide_upper_thr_z is not None
            else None
        )

        if soft is not None:
            self.tau_percentiles = torch.from_numpy(soft["tau_percentiles"]).to(self.device)
            self.tau = int(soft["tau"])
            self.tau_col = int(soft["tau_col"])

            self.softmin_taus_raw = (
                torch.from_numpy(soft["taus_raw"]).float().to(self.device).contiguous()
            )
            self.phase_thresholds_soft_raw = (
                torch.from_numpy(soft["thr_soft_raw"]).float().to(self.device).contiguous()
            )
            # self.guidance_phase_thresholds_soft_raw = (
            #     torch.from_numpy(soft["guide_thr_soft_raw"]).float().to(self.device).contiguous()
            # )
            self.guidance_phase_thresholds_lower_soft_raw = (
                torch.from_numpy(soft["guide_lower_thr_soft_raw"]).float().to(self.device).contiguous()
            )
            self.guidance_phase_thresholds_upper_soft_raw = (
                torch.from_numpy(soft["guide_upper_thr_soft_raw"]).float().to(self.device).contiguous()
                if soft.get("guide_upper_thr_soft_raw") is not None
                else None
            )

            if soft["taus_z"] is not None and soft["thr_soft_z"] is not None:
                self.softmin_taus_z = (
                    torch.from_numpy(soft["taus_z"]).float().to(self.device).contiguous()
                )
                self.phase_thresholds_soft_z = (
                    torch.from_numpy(soft["thr_soft_z"]).float().to(self.device).contiguous()
                )
                # if soft["guide_thr_soft_z"] is not None:
                #     self.guidance_phase_thresholds_soft_z = (
                #         torch.from_numpy(soft["guide_thr_soft_z"])
                #         .float()
                #         .to(self.device)
                #         .contiguous()
                #     )
                # else:
                #     self.guidance_phase_thresholds_soft_z = None
                if soft.get("guide_lower_thr_soft_z") is not None:
                    self.guidance_phase_thresholds_lower_soft_z = (
                        torch.from_numpy(soft["guide_lower_thr_soft_z"]).float().to(self.device).contiguous()
                    )
                else:
                    self.guidance_phase_thresholds_lower_soft_z = None

                if soft.get("guide_upper_thr_soft_z") is not None:
                    self.guidance_phase_thresholds_upper_soft_z = (
                        torch.from_numpy(soft["guide_upper_thr_soft_z"]).float().to(self.device).contiguous()
                    )
                else:
                    self.guidance_phase_thresholds_upper_soft_z = None
            else:
                self.softmin_taus_z = None
                self.phase_thresholds_soft_z = None
                self.guidance_phase_thresholds_soft_z = None
        else:
            self.tau_percentiles = None
            self.tau_col = None
            self.softmin_taus_raw = None
            self.phase_thresholds_soft_raw = None
            self.softmin_taus_z = None
            self.phase_thresholds_soft_z = None
            # self.guidance_phase_thresholds_soft_raw = None
            # self.guidance_phase_thresholds_soft_z = None
            self.guidance_phase_thresholds_lower_soft_raw = None
            self.guidance_phase_thresholds_lower_soft_z = None
            self.guidance_phase_thresholds_upper_soft_raw = None
            self.guidance_phase_thresholds_upper_soft_z = None

        self.latent_mean = (
            torch.from_numpy(mean_np).float().to(self.device).contiguous()
            if mean_np is not None
            else None
        )
        self.latent_std = (
            torch.from_numpy(std_np).float().to(self.device).contiguous()
            if std_np is not None
            else None
        )

        print("[subgoal prototypes] loaded prototype-per-phase")
        print("  file:", self.pcg_data_path)
        print("  group:", self.latent_dir)
        print("  target_latents shape:", (N, D))
        print("  n_phases:", n_phases, "| Pkeep:", Pkeep)
        print(
            "  switch threshold percent:",
            thr_p,
            "| selected column:",
            self.threshold_col,
            "| available:",
            pcts.tolist(),
        )
        # print(
        #     "  guidance threshold percent:",
        #     guide_thr_p,
        #     "| selected column:",
        #     self.guidance_threshold_col,
        # )
        print(
            "  guidance threshold percent (lower):",
            guide_thr_lower_p,
            "| selected column:",
            self.guidance_threshold_lower_col,
        )
        print(
            "  guidance threshold percent (upper):",
            guide_thr_upper_p,
            "| selected column:",
            self.guidance_threshold_upper_col,
        )
        print(
            "  guidance_phase_thresholds_lower_raw:",
            tuple(self.guidance_phase_thresholds_lower_raw.shape),
        )
        if self.guidance_phase_thresholds_upper_raw is not None:
            print(
                "  guidance_phase_thresholds_upper_raw:",
                tuple(self.guidance_phase_thresholds_upper_raw.shape),
            )
        print("  phase_proto_latents:", tuple(self.phase_proto_latents.shape))
        print("  phase_thresholds_raw:", tuple(self.phase_thresholds_raw.shape))
        # print("  guidance_phase_thresholds_raw:", tuple(self.guidance_phase_thresholds_raw.shape))
        print("  target_timesteps loaded:", tuple(target_timesteps_np.shape))
        print("  stall_k_per_phase:", stall_k_per_phase.tolist())
        print("  phase_step_p10:", phase_step_p10.tolist())
        print("  phase_step_p50:", phase_step_p50.tolist())
        print("  phase_step_p90:", phase_step_p90.tolist())

        if self.latent_mean is not None and self.latent_std is not None:
            print("  latent_mean/std available for zscore distance.")
        else:
            print("  latent_mean/std NOT found (ok if you only use raw L2).")

        if self.phase_thresholds_soft_raw is not None:
            print("  softmin:")
            print(
                "    tau:",
                self.tau,
                "| tau_col:",
                self.tau_col,
                "| available:",
                self.tau_percentiles.tolist(),
            )
            print(
                "    softmin_taus_raw (per phase):",
                self.softmin_taus_raw[:3].tolist(),
                "...",
            )
            print(
                "    phase_thresholds_soft_raw:",
                tuple(self.phase_thresholds_soft_raw.shape),
            )
            # print(
            #     "    guidance_phase_thresholds_soft_raw:",
            #     tuple(self.guidance_phase_thresholds_soft_raw.shape),
            # )
            print(
                "    guidance_phase_thresholds_lower_soft_raw:",
                tuple(self.guidance_phase_thresholds_lower_soft_raw.shape),
            )
            if self.guidance_phase_thresholds_upper_soft_raw is not None:
                print(
                    "    guidance_phase_thresholds_upper_soft_raw:",
                    tuple(self.guidance_phase_thresholds_upper_soft_raw.shape),
                )
            else:
                print("  softmin not present in file.")

    def compute_current_reward(self, current_obs):
        with torch.no_grad():
            current_obs_wm = self.prepare_obs(current_obs, 1)
            encode_obs = self.dyn_model.encode_obs(current_obs_wm)
            current_visual_latent = encode_obs["visual"]
            reward, _ = self.compute_nn_reward(current_visual_latent.squeeze(1))
        return reward

    def decode_nn_index(self, nn_idx: torch.Tensor):
        gi = nn_idx.detach().cpu().long()
        demo = self.demo_id_of_latent[gi]
        t = self.t_of_latent[gi]
        tnorm = self.t_norm_of_latent[gi]
        return demo, t, tnorm

    def _get_num_phases(self) -> int:
        if hasattr(self, "phase_proto_visual"):
            return int(self.phase_proto_visual.shape[0])
        if hasattr(self, "demo_visual_latents"):
            return int(self.demo_visual_latents.shape[0])
        return 1

    def _compute_phase_trace_metrics(self, phase_trace):
        if phase_trace is None or len(phase_trace) == 0:
            return {
                "PhaseAdvanceCount": 0,
                "MaxPhaseReached": None,
                "FinalPhase": None,
                "LongestPhaseStall": 0,
            }

        arr = [int(x) for x in phase_trace]
        final_phase = int(arr[-1])
        max_phase = int(max(arr))

        advance_count = 0
        longest_stall = 1
        cur_stall = 1

        for i in range(1, len(arr)):
            if arr[i] > arr[i - 1]:
                advance_count += 1
                cur_stall = 1
            elif arr[i] == arr[i - 1]:
                cur_stall += 1
                longest_stall = max(longest_stall, cur_stall)
            else:
                cur_stall = 1

        return {
            "PhaseAdvanceCount": int(advance_count),
            "MaxPhaseReached": int(max_phase),
            "FinalPhase": int(final_phase),
            "LongestPhaseStall": int(longest_stall),
        }

    # def _summarize_corr_episode(self, eps_adv: float = 0.0):
    #     if len(self.corr_log) == 0:
    #         return None
    #     near = np.array([r.get("near", 0) for r in self.corr_log], dtype=np.float32)
    #     adv = np.array([r.get("adv", 0.0) for r in self.corr_log], dtype=np.float32)
    #     win = np.array([r.get("win", 0) for r in self.corr_log], dtype=np.float32)

    #     out = {
    #         "N_steps": int(len(self.corr_log)),
    #         "N_near": int(near.sum()),
    #         "Accept": float(((near > 0) & (adv > eps_adv)).mean()),
    #     }
    #     if near.sum() > 0:
    #         out["WinNear"] = float((win * near).sum() / near.sum())
    #         out["AdvNear"] = float((adv * near).sum() / near.sum())
    #     else:
    #         out["WinNear"] = None
    #         out["AdvNear"] = None
    #     return out

    # def get_current_corr_summary(self, eps_adv: float = 0.0):
    #     return self._summarize_corr_episode(eps_adv=eps_adv)

    def compute_episode_selection_score(self, summ: dict):
        if summ is None:
            return None

        n_steps = max(int(summ.get("N_steps", 0)), 1)
        n_near = int(summ.get("N_near", 0))
        near_ratio = float(n_near) / float(n_steps)

        accept = float(summ.get("Accept", 0.0) or 0.0)
        winnear = float(summ.get("WinNear", 0.0) or 0.0)
        advnear = float(summ.get("AdvNear", 0.0) or 0.0)

        adv_bonus = max(0.0, advnear) / 0.02
        adv_bonus = min(adv_bonus, 1.0)

        score = (
            0.40 * accept
            + 0.30 * winnear
            + 0.20 * near_ratio
            + 0.10 * adv_bonus
        )
        return float(score)

    def should_keep_episode(
        self,
        summ: dict,
        score_thresh: float = 0.72,
        min_accept: float = 0.50,
        min_near_ratio: float = 0.50,
        min_steps: int = 10,
    ):
        if summ is None:
            return False

        n_steps = int(summ.get("N_steps", 0))
        n_near = int(summ.get("N_near", 0))
        accept = float(summ.get("Accept", 0.0) or 0.0)

        if n_steps <= 0:
            return False

        near_ratio = float(n_near) / float(n_steps)
        score = self.compute_episode_selection_score(summ)

        if n_steps < min_steps:
            return False
        if accept < min_accept:
            return False
        if near_ratio < min_near_ratio:
            return False
        if score is None or score < score_thresh:
            return False

        return True

    def _summarize_compact_episode(self, weak_margin_eps: float = 0.05, stall_k: int = 3):
        if len(self.corr_log) == 0:
            return None

        rows = self.corr_log
        N = len(rows)

        support_near = np.array([r.get("support_near", 0) for r in rows], dtype=np.float32)
        margin_base = np.array([r.get("margin_base", 0.0) for r in rows], dtype=np.float32)
        margin_corr = np.array([r.get("margin_corr", 0.0) for r in rows], dtype=np.float32)
        margin_gain = margin_corr - margin_base

        weak_near = np.array(
            [1.0 if abs(float(r.get("margin_base", 0.0))) <= weak_margin_eps else 0.0 for r in rows],
            dtype=np.float32,
        )

        proto_win = np.array([r.get("proto_win", 0) for r in rows], dtype=np.float32)
        phase_before = [int(r.get("phase", 0)) for r in rows]

        out = {
            "N_steps": int(N),
            "N_support_near": int(support_near.sum()),
            "SupportNearRate": float(support_near.mean()),
            "WeakNearRate": float(weak_near.mean()),
            "MarginGainAll": float(margin_gain.mean()),
            "ProtoWinRate": float(proto_win.mean()),
        }
        d_gm = [r["d_global_min"] for r in self.corr_log if "d_global_min" in r]
        if d_gm:
            out["d_global_min_mean"]   = float(np.mean(d_gm))
            out["d_global_min_median"] = float(np.median(d_gm))
            out["d_global_min_p90"]    = float(np.percentile(d_gm, 90))
            out["d_global_min_max"]    = float(max(d_gm))

        if support_near.sum() > 0:
            out["MarginGainNear"] = float((margin_gain * support_near).sum() / support_near.sum())
            out["ProtoWinNear"] = float((proto_win * support_near).sum() / support_near.sum())
        else:
            out["MarginGainNear"] = None
            out["ProtoWinNear"] = None

        weak_mask = weak_near > 0
        if weak_mask.sum() > 0:
            out["MarginGainWeak"] = float(margin_gain[weak_mask].mean())
        else:
            out["BoundaryCrossWeakRate"] = None
            out["MarginGainWeak"] = None

        phase_trace = getattr(self, "_subgoal_idx_trace", None)
        out.update(self._compute_phase_trace_metrics(phase_trace))

        stall_hits = 0
        stall_rescues = 0
        cur_stall = 1
        in_stall = False

        for i in range(1, len(phase_before)):
            if phase_before[i] == phase_before[i - 1]:
                cur_stall += 1
            else:
                cur_stall = 1
                in_stall = False

            if cur_stall >= stall_k and not in_stall:
                stall_hits += 1
                if margin_gain[i] > 0:
                    stall_rescues += 1
                in_stall = True

        out["N_stall_windows"] = int(stall_hits)
        out["StallRescueRate"] = float(stall_rescues / stall_hits) if stall_hits > 0 else None

        return out

    def get_current_compact_summary(self, weak_margin_eps: float = 0.05, stall_k=None):
        if len(self.corr_log) == 0:
            return None
        if stall_k is None:
            phase_before = [int(r.get("phase", 0)) for r in self.corr_log]
            phase_idx = max(set(phase_before), key=phase_before.count)

            if hasattr(self, "stall_k_per_phase") and self.stall_k_per_phase is not None:
                if torch.is_tensor(self.stall_k_per_phase):
                    phase_idx = min(int(phase_idx), len(self.stall_k_per_phase) - 1)
                    stall_k = int(self.stall_k_per_phase[phase_idx].item())
                else:
                    phase_idx = min(int(phase_idx), len(self.stall_k_per_phase) - 1)
                    stall_k = int(self.stall_k_per_phase[phase_idx])
            else:
                stall_k = 3

        return self._summarize_compact_episode(
            weak_margin_eps=weak_margin_eps,
            stall_k=stall_k,
        )

    def compute_episode_selection_score_compact(self, summ: dict):
        if summ is None:
            return None

        support_near = float(summ.get("SupportNearRate", 0.0) or 0.0)
        margin_gain_near = float(summ.get("MarginGainNear", 0.0) or 0.0)
        max_phase = float(summ.get("MaxPhaseReached", 0.0) or 0.0)
        stall_rescue = float(summ.get("StallRescueRate", 0.0) or 0.0)

        n_phases = max(self._get_num_phases(), 1)
        max_phase_norm = min(max_phase / max(n_phases - 1, 1), 1.0)

        mg_bonus = max(0.0, margin_gain_near) / 0.05
        mg_bonus = min(mg_bonus, 1.0)

        score = (
            0.30 * support_near
            + 0.35 * mg_bonus
            + 0.10 * max_phase_norm
            + 0.25 * stall_rescue
        )
        return float(score)

    def compute_episode_quality_compact(self, summ: dict, success: bool):
        if summ is None:
            return None

        proto_win = float(summ.get("ProtoWinRate", 0.0) or 0.0)
        margin_gain_all = float(summ.get("MarginGainAll", 0.0) or 0.0)
        margin_gain_near = float(summ.get("MarginGainNear", 0.0) or 0.0)
        support_near = float(summ.get("SupportNearRate", 0.0) or 0.0)
        stall_rescue = float(summ.get("StallRescueRate", 0.0) or 0.0)

        mg_all = min(max(margin_gain_all, 0.0) / 0.10, 1.0)
        mg_near = min(max(margin_gain_near, 0.0) / 0.05, 1.0)

        score = (
            0.25 * proto_win
            + 0.30 * mg_near
            + 0.20 * mg_all
            + 0.15 * support_near
            + 0.10 * stall_rescue
        )

        return float(min(score, 1.0))

    def compute_episode_selection_score_success(self, summ: dict):
        if summ is None:
            return None

        support_near = float(summ.get("SupportNearRate", 0.0) or 0.0)
        proto_win = float(summ.get("ProtoWinRate", 0.0) or 0.0)
        margin_gain_near = float(summ.get("MarginGainNear", 0.0) or 0.0)
        margin_gain_all = float(summ.get("MarginGainAll", 0.0) or 0.0)
        stall_rescue = float(summ.get("StallRescueRate", 0.0) or 0.0)
        max_phase = int(summ.get("MaxPhaseReached", 0) or 0)

        n_phases = max(self._get_num_phases(), 1)
        phase_score = min(max_phase / max(n_phases - 1, 1), 1.0)

        mg_near = min(max(margin_gain_near, 0.0) / 0.10, 1.5)
        mg_all = min(max(margin_gain_all, 0.0) / 0.15, 1.5)

        reliability = (
            0.25 * support_near
            + 0.15 * proto_win
            + 0.15 * phase_score
        )

        correction = (
            0.20 * mg_near
            + 0.15 * mg_all
            + 0.10 * stall_rescue
        )

        easy_penalty = 0.0
        if support_near > 0.95 and proto_win > 0.95 and mg_near < 0.2:
            easy_penalty = 0.10

        raw_score = reliability + correction - easy_penalty
        return float(raw_score)

    def should_keep_episode_compact(
        self,
        summ: dict,
        score_thresh: float = 0.40,
        min_support_near: float = 0.70,
        min_steps: int = 12,
        min_margin_gain_near: float = 0.003,
        require_final_phase: bool = True,
    ):
        if summ is None:
            return False

        n_steps = int(summ.get("N_steps", 0))
        support_near = float(summ.get("SupportNearRate", 0.0) or 0.0)
        margin_gain_near = float(summ.get("MarginGainNear", 0.0) or 0.0)
        final_phase = int(summ.get("FinalPhase", 0) or 0)

        if n_steps < min_steps:
            return False
        if support_near < min_support_near:
            return False
        if margin_gain_near < min_margin_gain_near:
            return False
        if require_final_phase and final_phase < 1:
            return False

        score = self.compute_episode_selection_score_compact(summ)
        if score is None or score < score_thresh:
            return False

        return True

    def should_keep_success_episode_compact(
        self,
        summ: dict,
        score_thresh: float = 0.70,
        min_support_near: float = 0.85,
        min_steps: int = 12,
        min_margin_gain_near: float = 0.01,
        min_proto_win: float = 0.85,
        require_final_phase: bool = True,
    ):
        if summ is None:
            return False

        n_steps = int(summ.get("N_steps", 0) or 0)
        support_near = float(summ.get("SupportNearRate", 0.0) or 0.0)
        margin_gain_near = float(summ.get("MarginGainNear", 0.0) or 0.0)
        proto_win = float(summ.get("ProtoWinRate", 0.0) or 0.0)
        final_phase = int(summ.get("FinalPhase", 0) or 0)

        if n_steps < min_steps:
            return False
        if support_near < min_support_near:
            return False
        if margin_gain_near < min_margin_gain_near:
            return False
        if proto_win < min_proto_win:
            return False
        if require_final_phase and final_phase < 1:
            return False

        score = self.compute_episode_selection_score_success(summ)
        if score is None or score < score_thresh:
            return False

        return True

    # def flush_corr_log_episode(self, episode_id=None, eps_adv: float = 0.0):
    #     if len(self.corr_log) == 0:
    #         self._subgoal_idx_trace = []
    #         self.phase_hit_count = 0
    #         self.phase_switch_counter = 0
    #         return

    #     if len(self.corr_log) > 0:
    #         if episode_id is not None:
    #             for r in self.corr_log:
    #                 r["episode_id"] = int(episode_id)
    #         with open(self.corr_log_path, "a") as f:
    #             for r in self.corr_log:
    #                 f.write(json.dumps(r) + "\n")
    #         self.corr_log_episodes.append(self.corr_log)

    #     phase_before = [int(r.get("phase", 0)) for r in self.corr_log]
    #     phase_idx = max(set(phase_before), key=phase_before.count)

    #     if hasattr(self, "stall_k_per_phase") and self.stall_k_per_phase is not None:
    #         if torch.is_tensor(self.stall_k_per_phase):
    #             phase_idx = min(int(phase_idx), len(self.stall_k_per_phase) - 1)
    #             stall_k = int(self.stall_k_per_phase[phase_idx].item())
    #         else:
    #             phase_idx = min(int(phase_idx), len(self.stall_k_per_phase) - 1)
    #             stall_k = int(self.stall_k_per_phase[phase_idx])
    #     else:
    #         stall_k = 3

    #     summ = self._summarize_compact_episode(
    #         weak_margin_eps=0.05,
    #         stall_k=stall_k,
    #     )

    #     if summ is not None:
    #         if episode_id is not None:
    #             summ["episode_id"] = int(episode_id)
    #         summ["stall_k_used"] = int(stall_k)
    #         with open(self.corr_summary_path, "a") as f:
    #             f.write(json.dumps(summ) + "\n")
    #         self.corr_summary.append(summ)

    #     self.corr_log = []
    #     self._subgoal_idx_trace = []
    #     self.phase_hit_count = 0
    #     self.phase_switch_counter = 0

    def set_policy_action_normalizer(self, policy_action_normalizer):
        self.policy_action_normalizer = policy_action_normalizer