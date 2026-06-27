import os
import json
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from einops import rearrange
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from dyn_model.datasets.img_transforms import get_eval_crop_transform_resnet
from dyn_model.plan import load_model


class LiberoPlanner:
    """
    LIBERO planner with feature parity to the Robomimic `Planner`:
      * three-case guidance (lower / upper guidance thresholds)
      * phase-switch counter / advance_phase / reset_phase_state
      * single-forward-pass `decide_guidance_and_phase`
      * global per-phase diagnostics (d_per_phase, argmin_phase, d_global_min)
      * full compact corr-log summary suite

    LIBERO-specific bits (kept from the original LiberoPlanner):
      * proprio = cat([ee_ori, ee_pos, joint_states])
      * optional image cropping (img_transform only when use_crop=True)
      * optional `language` field in obs
      * supports the 532-dim (512 visual + 20 proprio) prototype split
      * dataset_path comes straight from the task hdf5 (no robomimic env_meta lookup)
    """

    def __init__(
        self,
        demo_dataset_config,
        dynamics_model_ckpt,
        action_step=8,
        output_dir="debug/",
        guidance=False,
        latent_dir="/targets/libero_10",
        pcg_data_path=None,
        targets_num=None,
        tau=None,
        threshold_perc=None,
        soft_min=None,
        # phase switching knobs
        phase_switch_margin=0.1,
        phase_switch_min_steps=2,
        phase_switch_use_threshold=True,
        # guidance activation thresholds (two-bound, three-case scheme)
        guidance_threshold_lower_perc=None,
        guidance_threshold_upper_perc=None,
        guidance_disable_when_next_is_near=True,
        # reward softmin temperatures
        proto_softmin_temp_v=0.05,
        proto_softmin_temp_p=0.05,
    ):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.demo_dataset_config = demo_dataset_config

        # ---------------------------------------------------------------
        # load dynamics-model config
        # ---------------------------------------------------------------
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

        if hasattr(model_cfg, "env") and hasattr(model_cfg.env, "shape_obs"):
            self.shape_obs = model_cfg.env.shape_obs
        else:
            self.shape_obs = None

        # ---------------------------------------------------------------
        # planner state
        # ---------------------------------------------------------------
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
        self.phase_hit_count = 0

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

        # ---------------------------------------------------------------
        # dynamics-model normalizer
        # ---------------------------------------------------------------
        wm_normalizer = LinearNormalizer()
        wm_normalizer.load_state_dict(
            torch.load(os.path.join(dynamics_model_dir, "normalizer.pth"))
        )
        self.dyn_model_normalizer = wm_normalizer.to(self.device)
        self.policy_action_normalizer = LinearNormalizer()

        self.abs_action = self.demo_dataset_config.abs_action

        # LIBERO: pull dataset_path directly from the demo dataset config.
        # (no robomimic env_meta lookup, no env_name)
        dataset_path = getattr(self.demo_dataset_config, "dataset_path", None)
        if dataset_path is None:
            dataset_path = getattr(self.demo_dataset_config, "expert_dataset_path", None)
        if dataset_path is None:
            raise ValueError(
                "demo_dataset_config must contain dataset_path or expert_dataset_path"
            )
        self.dataset_path = dataset_path
        print("planner dataset_path =", self.dataset_path)

        # ---------------------------------------------------------------
        # image preprocessing (LIBERO supports use_crop=False)
        # ---------------------------------------------------------------
        self.use_crop = model_cfg.use_crop
        self.original_img_size = model_cfg.original_img_size
        self.cropped_img_size = model_cfg.cropped_img_size
        if self.use_crop:
            self.img_transform = get_eval_crop_transform_resnet(
                original_img_size=self.original_img_size,
                cropped_img_size=self.cropped_img_size,
            )
        else:
            self.img_transform = None

        self.view_names = model_cfg.view_names
        self.frameskip = model_cfg.frameskip
        self.exec_step = action_step

        print("self.demo_dataset_config.horizon", self.demo_dataset_config.horizon)
        self.horizon = self.demo_dataset_config.horizon // 16

        # ---------------------------------------------------------------
        # load prototypes
        # ---------------------------------------------------------------
        if self.guidance:
            self.get_demo_latents_subgoal_proto()

        self.timestep = 0
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep="rotation_6d"
        )
        self.output_dir = output_dir
        self.idx = 0

        # corr / summary logs
        self.corr_log = []
        self.corr_log_episodes = []
        self.corr_log_path = Path(self.output_dir) / "subgoal_corr_log.jsonl"
        self.corr_log_path.parent.mkdir(parents=True, exist_ok=True)
        open(self.corr_log_path, "a").close()

        self.corr_summary = []
        self.corr_summary_path = Path(self.output_dir) / "subgoal_corr_summary.jsonl"
        open(self.corr_summary_path, "a").close()

        self.rollout_step = 0

        # placeholders for optional NN-reward demo latents
        self.demo_visual_latents = None
        self.demo_proprio_latents = None
        self.demo_latents = None
        self.demo_images = None

    # ===================================================================
    # tiny helpers
    # ===================================================================
    def set_policy_action_normalizer(self, policy_action_normalizer):
        self.policy_action_normalizer = policy_action_normalizer

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
        self.phase_hit_count = 0
        self.phase_switch_counter = 0
        self._subgoal_idx_trace = [0]

    def _softmin_over_set(self, D: torch.Tensor, tau: float, dim: int = 1):
        tau = float(max(tau, 1e-12))
        X = -D / tau
        return (-tau) * torch.logsumexp(X, dim=dim)

    def _get_num_phases(self) -> int:
        if hasattr(self, "phase_proto_visual"):
            return int(self.phase_proto_visual.shape[0])
        if hasattr(self, "demo_visual_latents") and self.demo_visual_latents is not None:
            return int(self.demo_visual_latents.shape[0])
        return 1

    # ===================================================================
    # LIBERO obs preparation
    # ===================================================================
    def prepare_obs(self, current_obs, action_shape):
        ee_ori = current_obs["ee_ori"]
        ee_pos = current_obs["ee_pos"]
        joint_states = current_obs["joint_states"]

        proprio = torch.cat([ee_ori, ee_pos, joint_states], dim=-1).to(self.device)

        visual = {}
        for view_name in self.view_names:
            visual[view_name] = current_obs[view_name].to(self.device)
            visual[view_name] = self.dyn_model_normalizer[view_name].normalize(
                visual[view_name]
            )
            if self.img_transform is not None:
                visual[view_name] = self.img_transform(
                    visual[view_name].view(
                        -1, 3, self.original_img_size, self.original_img_size
                    )
                )
                visual[view_name] = visual[view_name].view(
                    -1, 1, 3, self.cropped_img_size, self.cropped_img_size
                )
            else:
                visual[view_name] = visual[view_name].view(
                    -1, 1, 3, self.original_img_size, self.original_img_size
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
        if "language" in current_obs:
            current_obs_wm["language"] = current_obs["language"].expand(action_shape, -1, -1)

        return current_obs_wm

    # ===================================================================
    # latent flattening / dynamics-model forward
    # ===================================================================
    def _flatten_online_latents(self, z_obs):
        """
        LIBERO version: no env-name-based visual slicing. Just flatten.
        Prototype dimension matching is handled later in _score_vp_against_phase.
        """
        v = z_obs["visual"]
        p = z_obs.get("proprio", None)

        if v.ndim > 2:
            v = v.reshape(v.size(0), -1)

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

    # ===================================================================
    # phase distance / scoring
    # ===================================================================
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

    def _score_vp_against_phase(self, v, p, phase_idx: int):
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

    def score_x0_against_phase(self, x0, current_obs, phase_idx: int):
        v, p, _ = self._encode_x0_to_vp(x0, current_obs)
        return self._score_vp_against_phase(v, p, phase_idx)

    def _compute_global_phase_diagnostics(self, x0, current_obs):
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

    # ===================================================================
    # phase comparison / decision
    # ===================================================================
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

        # single forward pass
        v, p, _ = self._encode_x0_to_vp(x0, current_obs)

        per_phase_info = [self._score_vp_against_phase(v, p, j) for j in range(n_phases)]
        cur_info = per_phase_info[cur]
        nxt_info = per_phase_info[nxt]

        use_soft = bool(cur_info["use_soft"])

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

        # Three-case guidance: same logic as Robomimic planner.
        # apply_guidance = True
        apply_guidance = (d_cur > guide_thr_lower_cur)


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

    # ===================================================================
    # reward & loss
    # ===================================================================
    def compute_subgoal_reward_vp_proto(
        self,
        current_visual_latent,
        current_proprio_latent,
        w_visual: float = 1.0,
        w_proprio: float = 0.2,
    ):
        # LIBERO has no env-name-based slicing: just flatten and trim against
        # the prototype dim later.
        if len(current_visual_latent.shape) > 2:
            v = current_visual_latent.reshape(current_visual_latent.size(0), -1)
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

    # ===================================================================
    # NN-reward path (optional, retained for compatibility)
    # ===================================================================
    def get_demo_latents(self):
        demo_dataset: BaseImageDataset

        self.demo_dataset_config.dataset_path = self.dataset_path
        self.demo_dataset_config.horizon = 1
        self.demo_dataset_config.n_obs_steps = 1
        self.demo_dataset_config.pad_before = 0
        self.demo_dataset_config.pad_after = 0

        demo_dataset = hydra.utils.instantiate(self.demo_dataset_config)
        demo_loader = DataLoader(demo_dataset, batch_size=64, shuffle=False, num_workers=4)

        demo_visual_latents = []
        demo_proprio_latents = []
        demo_images = []

        with torch.no_grad():
            for _, batch in enumerate(demo_loader):
                obs = batch["obs"]
                obs = dict_apply(obs, lambda x: x[:, -1:, ...])

                ee_ori = obs["ee_ori"]
                ee_pos = obs["ee_pos"]
                joint_states = obs["joint_states"]
                proprio = torch.cat([ee_ori, ee_pos, joint_states], dim=-1).to(self.device)

                visual = {}
                for view_name in self.view_names:
                    visual[view_name] = obs[view_name].to(self.device)
                    visual[view_name] = self.dyn_model_normalizer[view_name].normalize(
                        visual[view_name]
                    )
                    if self.img_transform is not None:
                        visual[view_name] = self.img_transform(
                            visual[view_name].view(
                                -1, 3, self.original_img_size, self.original_img_size
                            )
                        )
                        visual[view_name] = visual[view_name].view(
                            -1, 1, 3, self.cropped_img_size, self.cropped_img_size
                        )
                    else:
                        visual[view_name] = visual[view_name].view(
                            -1, 1, 3, self.original_img_size, self.original_img_size
                        )
                demo_images.append({k: v.cpu() for k, v in visual.items()})

                obs_wm = {"visual": visual, "proprio": proprio}
                if "language" in obs:
                    obs_wm["language"] = obs["language"]
                obs_wm["proprio"] = self.dyn_model_normalizer["state"].normalize(
                    obs_wm["proprio"]
                )

                encode_obs = self.dyn_model.encode_obs(obs_wm)
                demo_visual_latents.append(encode_obs["visual"].cpu())
                demo_proprio_latents.append(encode_obs["proprio"].cpu())
                torch.cuda.empty_cache()

        self.demo_visual_latents = torch.cat(demo_visual_latents, dim=0)
        self.demo_proprio_latents = torch.cat(demo_proprio_latents, dim=0)

        if self.demo_visual_latents.ndim > 2:
            self.demo_visual_latents = self.demo_visual_latents.reshape(
                self.demo_visual_latents.size(0), -1
            )
        if self.demo_proprio_latents.ndim > 2:
            self.demo_proprio_latents = self.demo_proprio_latents.reshape(
                self.demo_proprio_latents.size(0), -1
            )

        self.demo_latents = torch.cat(
            [self.demo_visual_latents, self.demo_proprio_latents], dim=-1
        )
        print("demo_visual_latents shape", self.demo_visual_latents.shape)
        print("demo_proprio_latents shape", self.demo_proprio_latents.shape)

        self.demo_images = {
            key: torch.cat([d[key] for d in demo_images], dim=0)
            for key in demo_images[0].keys()
        }

        del demo_dataset

    def compute_current_reward(self, current_obs):
        if self.demo_latents is None:
            raise RuntimeError("compute_current_reward requires demo latents. Call get_demo_latents() first.")
        with torch.no_grad():
            current_obs_wm = self.prepare_obs(current_obs, 1)
            encode_obs = self.dyn_model.encode_obs(current_obs_wm)
            current_visual_latent = encode_obs["visual"]
            current_proprio_latent = encode_obs["proprio"]
            reward, _ = self.compute_nn_reward(
                current_visual_latent.squeeze(1),
                current_proprio_latent.squeeze(1),
            )
        return reward

    def compute_nn_reward(self, current_visual_latent, current_proprio_latent):
        if self.demo_latents is None:
            raise RuntimeError("compute_nn_reward requires demo latents. Call get_demo_latents() first.")

        if len(current_visual_latent.shape) > 2:
            current_visual_latent = current_visual_latent.reshape(current_visual_latent.size(0), -1)
        if len(current_proprio_latent.shape) > 2:
            current_proprio_latent = current_proprio_latent.reshape(current_proprio_latent.size(0), -1)

        current_latent = torch.cat([current_visual_latent, current_proprio_latent], dim=-1)
        weights = torch.cat([
            torch.full((current_visual_latent.shape[-1],), 1, device=current_visual_latent.device),
            torch.full((current_proprio_latent.shape[-1],), 2, device=current_visual_latent.device),
        ])
        current_latent = current_latent * weights.unsqueeze(0)

        device = current_visual_latent.device
        chunk_size = 2048

        global_min_cost = None
        global_min_idx = None

        for start in range(0, self.demo_latents.shape[0], chunk_size):
            demo_chunk = self.demo_latents[start:start + chunk_size].to(device, non_blocking=True)
            demo_chunk = demo_chunk * weights
            dist = torch.cdist(current_latent, demo_chunk, p=2)
            cost, idx = dist.min(dim=-1)

            if global_min_cost is None:
                global_min_cost = cost
                global_min_idx = idx + start
            else:
                mask = cost < global_min_cost
                global_min_cost[mask] = cost[mask]
                global_min_idx[mask] = idx[mask] + start

        reward = -global_min_cost
        return reward, global_min_idx

    # ===================================================================
    # prototype loading (LIBERO version)
    # ===================================================================
    def get_demo_latents_subgoal_proto(self):
        if self.pcg_data_path is None:
            raise ValueError("Missing self.pcg_data_path")
        if self.latent_dir is None:
            raise ValueError("Missing self.latent_dir")

        Pkeep = int(getattr(self, "targets_num", 40))
        tau_p = int(getattr(self, "tau", 50))
        thr_p = int(getattr(self, "threshold_percent", 90))
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

        SOFT_TAU_PCTS_PATH = f"{self.latent_dir}/softmin_tau_percentiles"
        SOFT_TAUS_RAW_PATH = f"{self.latent_dir}/softmin_taus_raw_per_phase"
        SOFT_TAUS_Z_PATH = f"{self.latent_dir}/softmin_taus_z_per_phase"
        SOFT_THR_L2_PATH = f"{self.latent_dir}/thresholds_softmin_l2"
        SOFT_THR_Z_PATH = f"{self.latent_dir}/thresholds_softmin_zscore"

        MEAN_PATH = f"{self.latent_dir}/latent_mean"
        STD_PATH = f"{self.latent_dir}/latent_std"

        with h5py.File(self.pcg_data_path, "r") as f:
            if TARGET_LATENTS_PATH not in f:
                raise KeyError(f"Missing {TARGET_LATENTS_PATH}")
            if TARGET_OFFSETS_PATH not in f:
                raise KeyError(f"Missing {TARGET_OFFSETS_PATH}")
            if THR_PCTS_PATH not in f or THR_L2_MAT_PATH not in f:
                raise KeyError(f"Missing threshold datasets under {self.latent_dir}")

            target_latents_np = f[TARGET_LATENTS_PATH][:].astype(np.float32)
            target_offsets_np = f[TARGET_OFFSETS_PATH][:].astype(np.int32)

            target_timesteps_np = None
            if TARGET_TIMESTEPS_PATH in f:
                target_timesteps_np = f[TARGET_TIMESTEPS_PATH][:].astype(np.int32)

            pcts = f[THR_PCTS_PATH][:].astype(np.int32)
            thr_l2_mat = f[THR_L2_MAT_PATH][:].astype(np.float32)

            # switch threshold column
            where = np.where(pcts == thr_p)[0]
            if where.size == 0:
                raise ValueError(
                    f"threshold_perc={thr_p} not found. available={pcts.tolist()}"
                )
            col = int(where[0])
            thr_l2 = thr_l2_mat[:, col].astype(np.float32)

            # lower guidance threshold column
            where_g_lower = np.where(pcts == guide_thr_lower_p)[0]
            if where_g_lower.size == 0:
                raise ValueError(
                    f"guidance_threshold_lower_perc={guide_thr_lower_p} not found. "
                    f"Available: {pcts.tolist()}"
                )
            guide_lower_col = int(where_g_lower[0])
            guide_lower_thr_l2 = thr_l2_mat[:, guide_lower_col].astype(np.float32)

            # upper guidance threshold column (optional)
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

            # zscore thresholds (optional)
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

            # softmin tables (optional)
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
                        f"tau={tau_p} not found. available={tau_pcts.tolist()}"
                    )
                tau_col = int(wtau[0])

                taus_raw_sel = taus_raw_per_phase_np[:, tau_col].astype(np.float32)
                thr_soft_raw_sel = thr_soft_l2_np[:, tau_col, col].astype(np.float32)
                guide_lower_thr_soft_raw_sel = thr_soft_l2_np[:, tau_col, guide_lower_col].astype(np.float32)
                guide_upper_thr_soft_raw_sel = None
                if guide_upper_col is not None:
                    guide_upper_thr_soft_raw_sel = thr_soft_l2_np[:, tau_col, guide_upper_col].astype(np.float32)

                taus_z_sel = None
                thr_soft_z_sel = None
                guide_lower_thr_soft_z_sel = None
                guide_upper_thr_soft_z_sel = None
                if (SOFT_TAUS_Z_PATH in f) and (SOFT_THR_Z_PATH in f):
                    taus_z_per_phase_np = f[SOFT_TAUS_Z_PATH][:].astype(np.float32)
                    thr_soft_z_np = f[SOFT_THR_Z_PATH][:].astype(np.float32)
                    taus_z_sel = taus_z_per_phase_np[:, tau_col].astype(np.float32)
                    thr_soft_z_sel = thr_soft_z_np[:, tau_col, col].astype(np.float32)
                    guide_lower_thr_soft_z_sel = thr_soft_z_np[:, tau_col, guide_lower_col].astype(np.float32)
                    if guide_upper_col is not None:
                        guide_upper_thr_soft_z_sel = thr_soft_z_np[:, tau_col, guide_upper_col].astype(np.float32)

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

        # ---------------- split prototypes ----------------
        latents = torch.from_numpy(target_latents_np).float().to(self.device)
        if latents.ndim != 2:
            latents = latents.reshape(latents.shape[0], -1)
        N, D = latents.shape

        # LIBERO-specific: support the 532-dim split too
        if D == 532:
            Dv, Dp = 512, 20
            visual = latents[:, :Dv]
            proprio = latents[:, Dv:Dv + Dp]
        elif D == 2080:
            Dv, Dp = 2048, 32
            visual = latents[:, :Dv]
            proprio = latents[:, Dv:Dv + Dp]
        elif D == 2048:
            Dv, Dp = 2048, 0
            visual = latents
            proprio = latents.new_zeros((N, 0))
        elif D == 1056:
            Dv, Dp = 1024, 32
            visual = latents[:, :Dv]
            proprio = latents[:, Dv:Dv + Dp]
        elif D == 1040:
            Dv, Dp = 1024, 16
            visual = latents[:, :Dv]
            proprio = latents[:, Dv:Dv + Dp]
        elif D == 1024:
            Dv, Dp = 1024, 0
            visual = latents
            proprio = latents.new_zeros((N, 0))
        else:
            raise RuntimeError(
                f"Unexpected proto latent dim D={D}. "
                f"Expected one of [532, 2080, 2048, 1056, 1040, 1024]."
            )

        target_offsets = torch.from_numpy(target_offsets_np).to(self.device)
        n_phases = int(target_offsets.numel() - 1)
        if n_phases <= 0:
            raise RuntimeError("target_offsets invalid (n_phases <= 0).")
        self.n_phases = n_phases

        phase_proto_latents = latents.new_zeros((n_phases, Pkeep, D))
        phase_proto_visual = visual.new_zeros((n_phases, Pkeep, Dv))
        phase_proto_proprio = proprio.new_zeros((n_phases, Pkeep, Dp))
        phase_proto_mask = torch.zeros((n_phases, Pkeep), dtype=torch.bool, device=self.device)

        for j in range(n_phases):
            s = int(target_offsets[j].item())
            e = int(target_offsets[j + 1].item())
            if e <= s:
                continue
            take = min(Pkeep, e - s)
            phase_proto_latents[j, :take] = latents[s:s + take]
            phase_proto_visual[j, :take] = visual[s:s + take]
            if Dp > 0:
                phase_proto_proprio[j, :take] = proprio[s:s + take]
            phase_proto_mask[j, :take] = True

        self.phase_proto_latents = phase_proto_latents.contiguous()
        self.phase_proto_visual = phase_proto_visual.contiguous()
        self.phase_proto_proprio = phase_proto_proprio.contiguous()
        self.phase_proto_mask = phase_proto_mask.contiguous()

        # ---------------- thresholds ----------------
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

        self.guidance_phase_thresholds_lower_raw = (
            torch.from_numpy(guide_lower_thr_l2).float().to(self.device).contiguous()
        )
        self.guidance_phase_thresholds_lower_z = (
            torch.from_numpy(guide_lower_thr_z).float().to(self.device).contiguous()
            if guide_lower_thr_z is not None
            else None
        )
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

        # softmin block
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
                self.guidance_phase_thresholds_lower_soft_z = None
                self.guidance_phase_thresholds_upper_soft_z = None
        else:
            self.tau_percentiles = None
            self.tau_col = None
            self.softmin_taus_raw = None
            self.phase_thresholds_soft_raw = None
            self.softmin_taus_z = None
            self.phase_thresholds_soft_z = None
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

        # phase step stats / stall-k per phase
        if target_timesteps_np is not None:
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
        else:
            self.target_timesteps = None
            self.phase_step_p10 = None
            self.phase_step_p50 = None
            self.phase_step_p90 = None
            self.phase_step_span = None
            self.stall_k_per_phase = None

        print("[libero subgoal prototypes] loaded")
        print("  file:", self.pcg_data_path)
        print("  group:", self.latent_dir)
        print("  target_latents:", tuple(latents.shape))
        print("  visual dim:", Dv, "| proprio dim:", Dp)
        print("  phase_proto_visual:", tuple(self.phase_proto_visual.shape))
        print("  phase_proto_proprio:", tuple(self.phase_proto_proprio.shape))
        print(
            "  switch threshold percent:",
            thr_p,
            "| selected column:",
            self.threshold_col,
            "| available:",
            pcts.tolist(),
        )
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
        if self.phase_thresholds_soft_raw is not None:
            print("  softmin tau:", self.tau, "| tau_col:", self.tau_col)

    # ===================================================================
    # corr-log summary suite (mirrors Robomimic Planner)
    # ===================================================================
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

    def _summarize_corr_episode(self, eps_adv: float = 0.0):
        if len(self.corr_log) == 0:
            return None
        near = np.array([r.get("near", 0) for r in self.corr_log], dtype=np.float32)
        adv = np.array([r.get("adv", 0.0) for r in self.corr_log], dtype=np.float32)
        win = np.array([r.get("win", 0) for r in self.corr_log], dtype=np.float32)

        out = {
            "N_steps": int(len(self.corr_log)),
            "N_near": int(near.sum()),
            "Accept": float(((near > 0) & (adv > eps_adv)).mean()),
        }
        if near.sum() > 0:
            out["WinNear"] = float((win * near).sum() / near.sum())
            out["AdvNear"] = float((adv * near).sum() / near.sum())
        else:
            out["WinNear"] = None
            out["AdvNear"] = None
        return out

    def get_current_corr_summary(self, eps_adv: float = 0.0):
        return self._summarize_corr_episode(eps_adv=eps_adv)

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
            out["d_global_min_mean"] = float(np.mean(d_gm))
            out["d_global_min_median"] = float(np.median(d_gm))
            out["d_global_min_p90"] = float(np.percentile(d_gm, 90))
            out["d_global_min_max"] = float(max(d_gm))

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

    def flush_corr_log_episode(self, episode_id=None, eps_adv: float = 0.0):
        if len(self.corr_log) == 0:
            self._subgoal_idx_trace = []
            self.phase_hit_count = 0
            self.phase_switch_counter = 0
            return

        if episode_id is not None:
            for r in self.corr_log:
                r["episode_id"] = int(episode_id)
        with open(self.corr_log_path, "a") as f:
            for r in self.corr_log:
                f.write(json.dumps(r) + "\n")
        self.corr_log_episodes.append(self.corr_log)

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

        summ = self._summarize_compact_episode(
            weak_margin_eps=0.05,
            stall_k=stall_k,
        )

        if summ is not None:
            if episode_id is not None:
                summ["episode_id"] = int(episode_id)
            summ["stall_k_used"] = int(stall_k)
            with open(self.corr_summary_path, "a") as f:
                f.write(json.dumps(summ) + "\n")
            self.corr_summary.append(summ)

        self.corr_log = []
        self._subgoal_idx_trace = []
        self.phase_hit_count = 0
        self.phase_switch_counter = 0

    def diagnose_pipeline_gap(self, demo_path, demo_name="demo_0", max_steps=None):
        """
        Compare extraction-pipeline latents vs planner-pipeline latents on demo data.
        
        For each step t in the demo:
        - Extraction pipeline: encode_obs(obs_t) -> visual_extract, proprio_extract
        - Planner pipeline: encode(obs_{t}, action_t) -> predict -> separate_emb 
                            -> visual_planner, proprio_planner
        
        The dynamics model is trained to make these match. We measure the gap.
        """
        import h5py
        from diffusion_policy.common.pose_util import axisangle2quat_batch
        
        with h5py.File(demo_path, "r") as f:
            if f"data/{demo_name}" not in f:
                raise KeyError(f"Demo {demo_name} not found in {demo_path}")
            
            obs_grp = f[f"data/{demo_name}/obs"]
            action_grp = f[f"data/{demo_name}/actions"]
            
            T = obs_grp[self.view_names[0]].shape[0]
            if max_steps is not None:
                T = min(T, max_steps)
            
            gaps = []
            d_cur_extract = []
            d_cur_planner = []
            
            for t in range(T - 1):
                # ===========================================================
                # STEP 1: Load raw obs and action at step t
                # ===========================================================
                visual_raw = {}
                for v in self.view_names:
                    x = obs_grp[v][t:t+1]  # (1, H, W, 3) uint8
                    x = torch.from_numpy(x).float() / 255.0 if x.dtype == np.uint8 else torch.from_numpy(x).float()
                    if x.ndim == 4 and x.shape[-1] == 3:
                        x = x.permute(0, 3, 1, 2)  # (1, 3, H, W)
                    visual_raw[v] = x.to(self.device)
                
                ee_ori_raw = obs_grp["ee_ori"][t:t+1]
                ee_pos_raw = obs_grp["ee_pos"][t:t+1]
                joint_states_raw = obs_grp["joint_states"][t:t+1]
                
                # Convert ee_ori from axis-angle to quaternion (matches extraction)
                ee_ori_raw = axisangle2quat_batch(ee_ori_raw)
                
                ee_ori_t = torch.from_numpy(ee_ori_raw).float().to(self.device)
                ee_pos_t = torch.from_numpy(ee_pos_raw).float().to(self.device)
                joint_states_t = torch.from_numpy(joint_states_raw).float().to(self.device)
                
                # Read action at step t (for planner pipeline)
                action_t = action_grp[t:t+1]
                action_t = torch.from_numpy(action_t).float().to(self.device)
                
                # ===========================================================
                # STEP 2: Extraction-style encoding (encode_obs only)
                # ===========================================================
                with torch.no_grad():
                    visual_normalized = {}
                    for v in self.view_names:
                        vn = self.dyn_model_normalizer[v].normalize(visual_raw[v])
                        if self.img_transform is not None:
                            vn = self.img_transform(vn)
                        vn = vn.unsqueeze(1)  # (1, 1, 3, H, W)
                        visual_normalized[v] = vn
                    
                    proprio_cat = torch.cat([ee_ori_t, ee_pos_t, joint_states_t], dim=-1)
                    proprio_norm = self.dyn_model_normalizer["state"].normalize(proprio_cat)
                    proprio_norm = proprio_norm.unsqueeze(1)  # (1, 1, D)
                    
                    obs_wm_extract = {"visual": visual_normalized, "proprio": proprio_norm}
                    
                    enc = self.dyn_model.encode_obs(obs_wm_extract)
                    v_extract = enc["visual"].squeeze(1)  # (1, Dv)
                    p_extract = enc["proprio"].squeeze(1)  # (1, Dp)
                    
                    if v_extract.ndim > 2:
                        v_extract = v_extract.reshape(v_extract.size(0), -1)
                    if p_extract.ndim > 2:
                        p_extract = p_extract.reshape(p_extract.size(0), -1)
                
                # ===========================================================
                # STEP 3: Planner-style encoding (encode + predict + separate_emb)
                # ===========================================================
                with torch.no_grad():
                    # Normalize action using dyn model normalizer (matches planner's flow)
                    action_norm = self.dyn_model_normalizer["act"].normalize(action_t)
                    # Reshape action to expected format: (B, H, F*A)
                    # For one step: (1, 1, action_dim*frameskip), but here we just have one action
                    # so this might need adjustment based on action sequence length
                    # For now, replicate action to match dynamics model's expected horizon
                    action_batch = action_norm.unsqueeze(1)  # (1, 1, action_dim)
                    
                    # We use obs at step t-1 + action at step t-1 to predict step t
                    # But for this diagnostic, simpler: use obs at step t + action at step t
                    # to predict step t+1, then compare to encode_obs(obs at step t+1)
                    # 
                    # Actually the simplest test: encode(obs_t, action_t) -> predict -> compare
                    # to encode_obs(obs_t)
                    
                    z = self.dyn_model.encode(obs_wm_extract, action_batch)
                    z_pred = self.dyn_model.predict(z)
                    z_new = z_pred[:, -1:, ...]
                    z_obs_pred, _ = self.dyn_model.separate_emb(z_new)
                    
                    v_planner = z_obs_pred["visual"]
                    p_planner = z_obs_pred["proprio"]
                    
                    if v_planner.ndim > 2:
                        v_planner = v_planner.reshape(v_planner.size(0), -1)
                    if p_planner.ndim > 2:
                        p_planner = p_planner.reshape(p_planner.size(0), -1)
                
                # ===========================================================
                # STEP 4: Compute gap
                # ===========================================================
                v_gap = torch.norm(v_extract - v_planner, dim=-1).item()
                p_gap = torch.norm(p_extract - p_planner, dim=-1).item()
                
                # Also compute distance to current phase's prototypes for context
                phase_idx = self.subgoal_idx  # use current phase
                d_extract, thr = self._phase_distance_to_idx(v_extract, p_extract, phase_idx)
                d_planner, _ = self._phase_distance_to_idx(v_planner, p_planner, phase_idx)
                
                gaps.append({
                    "step": t,
                    "v_gap": v_gap,
                    "p_gap": p_gap,
                    "total_gap": float(torch.norm(
                        torch.cat([v_extract, p_extract], dim=-1) - 
                        torch.cat([v_planner, p_planner], dim=-1), dim=-1).item()),
                    "d_to_proto_extract": d_extract,
                    "d_to_proto_planner": d_planner,
                    "thr_proto": thr,
                })
            
            return gaps