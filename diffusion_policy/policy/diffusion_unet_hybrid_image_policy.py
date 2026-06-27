import os
from typing import Dict, Optional, Sequence

import yaml
import cv2
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.common.language_models import extract_text_features, get_text_model
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_core as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc


def boundary_penalty(action, lower_bound=-1.0, upper_bound=1.0):
    penalty = torch.relu(action - upper_bound) + torch.relu(lower_bound - action)
    return penalty.sum()


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        task_name: str = "square",
        num_inference_steps=None,
        obs_as_global_cond=True,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
        obs_config=None,
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]

        obs_config = {
            "low_dim": [],
            "rgb": [],
            "depth": [],
            "scan": [],
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            if key == "language":
                continue
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                obs_config["rgb"].append(key)
            elif obs_type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        self.obs_config = obs_config

        config = get_robomimic_config(
            algo_name="bc_rnn",
            hdf5_type="image",
            task_name=task_name,
            dataset_type="ph",
        )

        with config.unlocked():
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu",
        )

        obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features,
                ),
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )

        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
            if "language" in shape_meta["obs"]:
                global_cond_dim += 32

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.dynamics_model_normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.correct_num = 0

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

        if "language" in shape_meta["obs"]:
            self.text_model, self.tokenizer, self.max_length = get_text_model(
                "libero_10", "clip"
            )

    def initialize_planner(
        self,
        planner_target,
        demo_dataset_config,
        dynamics_model_ckpt,
        action_step,
        output_dir,
        guidance_start_timestep,
        guidance_scale,
        guidance,
        pcg_data_path=None,
        latent_dir=None,
        targets_num=40,
        tau=10,
        threshold_perc=70,
        soft_min=False,
        phase_switch_margin=0.1,
        phase_switch_min_steps=2,
        phase_switch_use_threshold=True,
        guidance_threshold_lower_perc=None,   # RENAMED
        guidance_threshold_upper_perc=None,   # NEW
        guidance_disable_when_next_is_near=True,
        proto_softmin_temp_v=0.05,
        proto_softmin_temp_p=0.05,
    ):
        planner_cls = hydra.utils.get_class(planner_target)
        self.planner = planner_cls(
            demo_dataset_config=demo_dataset_config,
            dynamics_model_ckpt=dynamics_model_ckpt,
            action_step=action_step,
            output_dir=output_dir,
            guidance=guidance,
            latent_dir=latent_dir,
            pcg_data_path=pcg_data_path,
            targets_num=targets_num,
            tau=tau,
            threshold_perc=threshold_perc,
            soft_min=soft_min,
            phase_switch_margin=phase_switch_margin,
            phase_switch_min_steps=phase_switch_min_steps,
            phase_switch_use_threshold=phase_switch_use_threshold,
            guidance_threshold_lower_perc=guidance_threshold_lower_perc,
            guidance_threshold_upper_perc=guidance_threshold_upper_perc,
            guidance_disable_when_next_is_near=guidance_disable_when_next_is_near,
            proto_softmin_temp_v=proto_softmin_temp_v,
            proto_softmin_temp_p=proto_softmin_temp_p,
        )
        self.guidance_start_timestep = guidance_start_timestep
        self.guidance_scale = guidance_scale
        self.planner.set_policy_action_normalizer(self.normalizer["action"])
        self.guidance = guidance

    def reset(self):
        try:
            super().reset()
        except Exception:
            pass

        planner = getattr(self, "planner", None)
        if planner is None:
            return

        if not hasattr(planner, "subgoal_idx_episodes"):
            planner.subgoal_idx_episodes = []
        if not hasattr(planner, "_subgoal_idx_trace"):
            planner._subgoal_idx_trace = []

        if len(planner._subgoal_idx_trace) > 0:
            planner.subgoal_idx_episodes.append(planner._subgoal_idx_trace.copy())

        # if hasattr(planner, "flush_corr_log_episode"):
        #     planner.flush_corr_log_episode()

        if hasattr(planner, "reset_phase_state"):
            planner.reset_phase_state()
        else:
            planner.subgoal_idx = 0
            planner.increment = False
            # planner.phase_hit_count = 0
            planner.phase_switch_counter = 0
            planner._subgoal_idx_trace = [0]

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        current_obs=None,
        text_latents=None,
        **kwargs,
    ):
        if text_latents is not None:
            current_obs["language"] = text_latents

        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            trajectory = trajectory.detach().requires_grad_()

            model_output = model(
                trajectory,
                t,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def guided_conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        current_obs=None,
        text_latents=None,
        **kwargs,
    ):
        if text_latents is not None:
            current_obs["language"] = text_latents

        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        timesteps = [int(x) for x in scheduler.timesteps.tolist()]
        alphas_cumprod = scheduler.alphas_cumprod.to(trajectory.device)

        last_x0 = None
        last_x0_guided = None
        last_x0_used = None
        last_t = None
        last_decision = None

        for t in timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]

            with torch.no_grad():
                model_output = model(
                    trajectory,
                    t,
                    local_cond=local_cond,
                    global_cond=global_cond,
                )

            alpha_bar_t = alphas_cumprod[t]

            x0 = (trajectory - torch.sqrt(1.0 - alpha_bar_t) * model_output) / torch.sqrt(
                alpha_bar_t
            )

            # decision = self.planner.decide_guidance_and_phase(x0.detach(), current_obs)
            #===========================================================
            # is_last = (t == timesteps[-1])
            decision = self.planner.decide_guidance_and_phase(
                x0.detach(), current_obs, log=False, update_state=False
            )
            #===========================================================

            # if decision["switch_phase"]:
            #     self.planner.advance_phase()
            #     decision = self.planner.decide_guidance_and_phase(x0.detach(), current_obs)

            do_guidance = (t < self.guidance_start_timestep) and bool(decision["apply_guidance"])

            model_output_used = model_output
            x0_used = x0.detach()
            x0_guided = None

            if do_guidance:
                with torch.enable_grad():
                    x0_g = x0.detach().requires_grad_(True)
                    loss = self.planner.compute_loss_mpgd_subgoal_vp_proto(x0_g, current_obs)
                    grad_x0 = torch.autograd.grad(loss, x0_g)[0]

                c_t = self.guidance_scale
                x0_guided = (x0 - c_t * grad_x0).detach()
                x0_used = x0_guided

                eps_guided = (
                    trajectory - torch.sqrt(alpha_bar_t) * x0_guided
                ) / torch.sqrt(1.0 - alpha_bar_t)
                model_output_used = eps_guided

            with torch.no_grad():
                trajectory = scheduler.step(
                    model_output_used,
                    t,
                    trajectory,
                    generator=generator,
                    **kwargs,
                ).prev_sample

            last_x0 = x0.detach()
            last_x0_guided = x0_guided
            last_x0_used = x0_used
            last_t = int(t)
            last_decision = decision

        #====================for ablation study update noise action==================
        # for t in timesteps:
        #     trajectory[condition_mask] = condition_data[condition_mask]

        #     # =======================================================================
        #     # LPB-style: trajectory needs requires_grad BEFORE the model call so that
        #     # gradients can flow back from the loss through model(trajectory) to
        #     # trajectory itself.
        #     # =======================================================================
        #     trajectory = trajectory.detach().requires_grad_(True)

        #     model_output = model(
        #         trajectory,
        #         t,
        #         local_cond=local_cond,
        #         global_cond=global_cond,
        #     )

        #     alpha_bar_t = alphas_cumprod[t]

        #     # Compute x0 estimate (with gradient flow through trajectory).
        #     # We still call this 'x0' for the decision step; we detach it there.
        #     x0 = (trajectory - torch.sqrt(1.0 - alpha_bar_t) * model_output) / torch.sqrt(
        #         alpha_bar_t
        #     )

        #     decision = self.planner.decide_guidance_and_phase(
        #         x0.detach(), current_obs, log=False, update_state=False
        #     )

        #     do_guidance = (t < self.guidance_start_timestep) and bool(decision["apply_guidance"])

        #     model_output_used = model_output
        #     x0_used = x0.detach()
        #     x0_guided = None

        #     if do_guidance:
        #         # ===================================================================
        #         # LPB-style classifier guidance:
        #         # - Compute loss on the x0 estimate (which depends on trajectory)
        #         # - Take gradient w.r.t. the NOISY trajectory
        #         # - Perturb the noisy trajectory with sqrt(1 - alpha_bar_t) scaling
        #         # ===================================================================
        #         loss = self.planner.compute_loss_mpgd_subgoal_vp_proto(x0, current_obs)
        #         cond_grad = -torch.autograd.grad(loss, trajectory)[0]

        #         c_t = self.guidance_scale
        #         grad_scale = c_t * torch.sqrt(1.0 - alpha_bar_t)

        #         # Perturb trajectory; detach to drop the autograd graph.
        #         trajectory = (trajectory.detach() + grad_scale * cond_grad).detach()

        #         # Track effective x0 implied by the new trajectory (for logging).
        #         with torch.no_grad():
        #             x0_used = (
        #                 trajectory - torch.sqrt(1.0 - alpha_bar_t) * model_output.detach()
        #             ) / torch.sqrt(alpha_bar_t)
        #             x0_guided = x0_used.clone()

        #         # model_output_used stays as the original model_output (LPB does not
        #         # recompute it after the perturbation).
        #         model_output_used = model_output.detach()
        #     else:
        #         # No guidance this step: detach so we don't keep the autograd graph.
        #         trajectory = trajectory.detach()
        #         model_output_used = model_output.detach()

        #     with torch.no_grad():
        #         trajectory = scheduler.step(
        #             model_output_used,
        #             t,
        #             trajectory,
        #             generator=generator,
        #             **kwargs,
        #         ).prev_sample

        #     last_x0 = x0.detach()
        #     last_x0_guided = x0_guided
        #     last_x0_used = x0_used
        #     last_t = int(t)
        #     last_decision = decision
        #============================================================================
        # AFTER the denoising loop ends:
        final_decision = self.planner.decide_guidance_and_phase(
            last_x0, current_obs, log=True, update_state=True  # <-- counter advances here
        )
        if final_decision["switch_phase"]:
            self.planner.advance_phase()
        trajectory[condition_mask] = condition_data[condition_mask]

        if last_x0 is not None and last_x0_used is not None:
            base_cmp = self.planner.compare_current_next_phase(last_x0, current_obs)
            used_cmp = self.planner.compare_current_next_phase(last_x0_used, current_obs)

            phase_before = int(base_cmp["phase_cur"])
            phase_after = int(getattr(self.planner, "subgoal_idx", phase_before))

            proto_adv_cur = float(used_cmp["margin_cur"] - base_cmp["margin_cur"])
            proto_win_cur = int(proto_adv_cur > 0.0)

            rec = {
                "rollout_step": int(self.planner.rollout_step),
                "diff_t": int(last_t),
                "phase": phase_before,
                "phase_after": phase_after,
                "next_phase": int(base_cmp["phase_next"]),
                "guidance_applied": int(last_x0_guided is not None),
                "guidance_scale": float(self.guidance_scale),
                "d_cur_base": float(base_cmp["d_cur"]),
                "d_cur_used": float(used_cmp["d_cur"]),
                "thr_cur_switch": float(last_decision["switch_thr_cur"]) if last_decision is not None else float(base_cmp["thr_cur"]),
                "thr_cur_guide_lower": (
                    float(last_decision["guide_thr_lower_cur"])
                    if last_decision is not None
                    else float(base_cmp["guide_thr_lower_cur"])
                ),
                "thr_cur_guide_upper": (
                    (float(last_decision["guide_thr_upper_cur"])
                    if last_decision["guide_thr_upper_cur"] is not None else None)
                    if last_decision is not None
                    else (float(base_cmp["guide_thr_upper_cur"])
                        if base_cmp["guide_thr_upper_cur"] is not None else None)
                ),
                "margin_base": float(base_cmp["margin_cur"]),
                "margin_corr": float(used_cmp["margin_cur"]),
                "margin_gain": float(used_cmp["margin_cur"] - base_cmp["margin_cur"]),
                "d_next_base": float(base_cmp["d_next"]) if base_cmp["d_next"] is not None else None,
                "d_next_used": float(used_cmp["d_next"]) if used_cmp["d_next"] is not None else None,
                "thr_next": float(base_cmp["thr_next"]) if base_cmp["thr_next"] is not None else None,
                "margin_next_base": float(base_cmp["margin_next"]) if base_cmp["margin_next"] is not None else None,
                "margin_next_used": float(used_cmp["margin_next"]) if used_cmp["margin_next"] is not None else None,
                "margin_next_gain": (
                    float(used_cmp["margin_next"] - base_cmp["margin_next"])
                    if (used_cmp["margin_next"] is not None and base_cmp["margin_next"] is not None)
                    else None
                ),
                "phase_switch_counter": int(final_decision["phase_switch_counter"]),
                "switch_phase_decision": int(final_decision["switch_phase"]),
                "better_than_current": int(final_decision["better_than_current"]),
                "acceptable_next": int(final_decision["acceptable_next"]),
                "next_is_near": int(final_decision["next_is_near"]),
                "apply_guidance_decision": int(final_decision["apply_guidance"]),
                "phase_switch_counter": int(last_decision["phase_switch_counter"]) if last_decision is not None else None,
                "support_near": int(base_cmp["margin_cur"] > 0.0),
                "proto_adv": float(proto_adv_cur),
                "proto_win": int(proto_win_cur),
            }
            self.planner.log_corr(rec)

        self.planner._subgoal_idx_trace.append(int(self.planner.subgoal_idx))
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], language_goal=None) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict

        text_latents = None
        if language_goal is not None:
            text_tokens = self.tokenizer(
                language_goal,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            text_latents = extract_text_features(
                self.text_model,
                text_tokens,
                language_emb_model="clip",
            )

        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            if text_latents is not None:
                global_cond = torch.cat([global_cond, text_latents], dim=-1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        with torch.no_grad():
            nsample = self.conditional_sample(
                cond_data,
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                current_obs=dict_apply(obs_dict, lambda x: x[:, -1:, ...]),
                **self.kwargs,
            )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def predict_action_dyn_guided(self, obs_dict: Dict[str, torch.Tensor], language_goal=None) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict

        text_latents = None
        if language_goal is not None:
            text_tokens = self.tokenizer(
                language_goal,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            text_latents = extract_text_features(
                self.text_model,
                text_tokens,
                language_emb_model="clip",
            )

        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            if text_latents is not None:
                global_cond = torch.cat([global_cond, text_latents], dim=-1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        nsample = self.guided_conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            current_obs=dict_apply(obs_dict, lambda x: x[:, -1:, ...]),
            text_latents=text_latents,
            **self.kwargs,
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            "action": action,
            "action_pred": action_pred,
        }

        if hasattr(self, "planner"):
            self.planner.rollout_step += 1
        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        text_latents = None
        if "language" in batch["obs"]:
            if "language" in batch["obs"]:
                language_goal = batch["obs"]["language"]
                del batch["obs"]["language"]
                text_tokens = {
                    "input_ids": language_goal[:, 0].long()[:, 0],
                    "attention_mask": language_goal[:, 0].long()[:, 1],
                }
                text_latents = extract_text_features(
                    self.text_model,
                    text_tokens,
                    language_emb_model="clip",
                )
            elif "language_latents" in batch:
                text_latents = batch["language_latents"]

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
            if text_latents is not None:
                global_cond = torch.cat([global_cond, text_latents], dim=-1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss