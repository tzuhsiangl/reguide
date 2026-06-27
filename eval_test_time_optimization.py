import sys
import os
import copy
import pathlib
import json
import random

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import dill
import wandb

from diffusion_policy.workspace.base_workspace import BaseWorkspace

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


@hydra.main(version_base="1.1", config_path="dyn_model/conf/planner", config_name="eval_transport")
def main(cfg: DictConfig):
    output_dir = cfg.output_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    config_save_path = os.path.join(output_dir, "eval_config.yaml")
    OmegaConf.save(config=cfg, f=config_save_path)
    print(f"Configuration saved to {config_save_path}")

    # -------------------------
    # load checkpoint payload
    # -------------------------
    with open(cfg.policy_checkpoint, "rb") as f:
        payload = torch.load(f, pickle_module=dill)

    # -------------------------
    # update cfg from checkpoint
    # -------------------------
    cfg_task_env_runner = payload["cfg"]

    cfg_task_env_runner.n_action_steps = cfg.n_action_steps
    cfg_task_env_runner.task.env_runner.n_action_steps = cfg.n_action_steps
    cfg_task_env_runner.policy.n_action_steps = cfg.n_action_steps

    cfg_task_env_runner.task.env_runner.n_test = cfg.n_test
    cfg_task_env_runner.task.env_runner.n_test_vis = cfg.n_test
    cfg_task_env_runner.task.env_runner.n_train = 0
    cfg_task_env_runner.task.env_runner.n_train_vis = 0
    cfg_task_env_runner.task.env_runner.test_start_seed = cfg.test_start_seed


    # ---- force dataset path to local scratch ----
    # if cfg.get("demo_dataset_path", None) is not None:
    #     # for planner / dataset metadata
    #     if hasattr(cfg_task_env_runner.task, "dataset"):
    #         cfg_task_env_runner.task.dataset.dataset_path = cfg.demo_dataset_path

    #     # for env runner
    #     cfg_task_env_runner.task.env_runner.dataset_path = cfg.demo_dataset_path

    if cfg.get("demo_dataset_path", None) is not None:
        cfg_task_env_runner.task.env_runner.dataset_path = cfg.demo_dataset_path
    elif "expert_dataset_path" in cfg_task_env_runner.task.dataset:
        cfg_task_env_runner.task.env_runner.dataset_path = cfg_task_env_runner.task.dataset.expert_dataset_path
    elif "dataset_path" in cfg_task_env_runner.task.dataset:
        cfg_task_env_runner.task.env_runner.dataset_path = cfg_task_env_runner.task.dataset.dataset_path

    print("[DEBUG] task.dataset.dataset_path =",
        getattr(cfg_task_env_runner.task.dataset, "dataset_path", None), flush=True)
    print("[DEBUG] task.env_runner.dataset_path =",
        cfg_task_env_runner.task.env_runner.get("dataset_path", None), flush=True)
    # --------------------------------------------


    if "libero" in cfg.policy_checkpoint:
        cfg_task_env_runner.task.env_runner.dataset_path = cfg.dataset_path

    # -------------------------
    # initialize workspace
    # -------------------------
    cls = hydra.utils.get_class(cfg_task_env_runner._target_)
    workspace = cls(cfg_task_env_runner, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # -------------------------
    # choose policy
    # -------------------------
    policy = workspace.model
    if cfg_task_env_runner.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(cfg.device)
    policy.to(device)
    policy.eval()

    # -------------------------
    # load normalizer
    # -------------------------
    normalizer_dir = os.path.dirname(os.path.dirname(cfg.policy_checkpoint))
    normalizer_path = os.path.join(normalizer_dir, "normalizer.pth")
    policy.normalizer.load_state_dict(torch.load(normalizer_path))
    policy.normalizer.to(device)

    # -------------------------
    # planner dataset cfg
    # -------------------------
    # The planner expects a dataset config exposing `dataset_path`. Continual-training
    # checkpoints store it as `expert_dataset_path`, so normalize to `dataset_path` here.
    demo_dataset_cfg = copy.deepcopy(payload["cfg"].task.dataset)
    OmegaConf.set_struct(demo_dataset_cfg, False)

    if cfg.get("demo_dataset_path", None) is not None:
        demo_dataset_cfg.dataset_path = cfg.demo_dataset_path
        if "expert_dataset_path" in demo_dataset_cfg:
            demo_dataset_cfg.expert_dataset_path = cfg.demo_dataset_path
    elif "dataset_path" not in demo_dataset_cfg:
        if "expert_dataset_path" in demo_dataset_cfg:
            demo_dataset_cfg.dataset_path = demo_dataset_cfg.expert_dataset_path
        else:
            raise ValueError("Could not determine dataset_path for planner.")

    print("[DEBUG] planner dataset_path =", demo_dataset_cfg.get("dataset_path", None), flush=True)

    # -------------------------
    # initialize planner
    # -------------------------
    policy.initialize_planner(
        planner_target=cfg.planner_target,
        demo_dataset_config=demo_dataset_cfg,
        dynamics_model_ckpt=cfg.dynamics_model_checkpoint,
        action_step=cfg_task_env_runner.n_action_steps,
        output_dir=cfg.output_dir,
        guidance=cfg.guidance,
        guidance_start_timestep=cfg.guidance_start_timestep,
        guidance_scale=cfg.guidance_scale,
        pcg_data_path=cfg.get("pcg_data_path", None),
        latent_dir=cfg.latent_dir,
        targets_num=cfg.targets_num,
        tau=cfg.tau,
        threshold_perc=cfg.threshold_perc,
        soft_min=cfg.soft_min,
        phase_switch_margin=cfg.phase_switch_margin,
        phase_switch_min_steps=cfg.phase_switch_min_steps,
        phase_switch_use_threshold=cfg.phase_switch_use_threshold,
        guidance_threshold_lower_perc=cfg.get("guidance_threshold_lower_perc", None),
        guidance_threshold_upper_perc=cfg.get("guidance_threshold_upper_perc", None),
        guidance_disable_when_next_is_near=cfg.get("guidance_disable_when_next_is_near", True),
        proto_softmin_temp_v=cfg.get("proto_softmin_temp_v", 0.05),
        proto_softmin_temp_p=cfg.get("proto_softmin_temp_p", 0.05),
    )

    # -------------------------
    # runner config override
    # -------------------------
    cfg_task_env_runner.task.env_runner._target_ = cfg.env_runner_target

    OmegaConf.set_struct(cfg_task_env_runner, False)
    OmegaConf.set_struct(cfg_task_env_runner.task.env_runner, False)

    save_hdf5 = bool(cfg.get("save_hdf5", False))
    save_video = bool(cfg.get("save_video", True))
    guidance = bool(cfg.get("guidance", False))
    obs_noise = bool(cfg.get("obs_noise", False))
    action_noise = bool(cfg.get("action_noise", False))
    rollout = bool(cfg.get("rollout", False))
    obs_image_noise_std = float(cfg.get("obs_image_noise_std", 0.0))
    obs_state_noise_std = float(cfg.get("obs_state_noise_std", 0.0))
    action_noise_std = float(cfg.get("action_noise_std", 0.0))

    cfg_task_env_runner.task.env_runner.save_hdf5 = save_hdf5
    cfg_task_env_runner.task.env_runner.save_video = save_video
    cfg_task_env_runner.task.env_runner.guidance = guidance
    cfg_task_env_runner.task.env_runner.obs_noise = obs_noise
    cfg_task_env_runner.task.env_runner.action_noise = action_noise
    cfg_task_env_runner.task.env_runner.obs_image_noise_std = obs_image_noise_std
    cfg_task_env_runner.task.env_runner.obs_state_noise_std = obs_state_noise_std
    cfg_task_env_runner.task.env_runner.action_noise_std = action_noise_std
    cfg_task_env_runner.task.env_runner.rollout = rollout

    if save_hdf5:
        cfg_task_env_runner.task.env_runner.success_hdf5_path = str(cfg.success_hdf5_path)

    print("[DEBUG] runner save_hdf5 =", cfg_task_env_runner.task.env_runner.save_hdf5, flush=True)
    print("[DEBUG] runner success_hdf5_path =", cfg_task_env_runner.task.env_runner.get("success_hdf5_path", None), flush=True)
    print("[DEBUG] runner save_video =", cfg_task_env_runner.task.env_runner.save_video, flush=True)

    # -------------------------
    # instantiate env runner
    # -------------------------
    dataset_target = payload["cfg"].task.dataset._target_
    if "libero" in dataset_target:
        env_runner = hydra.utils.instantiate(
            cfg_task_env_runner.task.env_runner,
            output_dir=output_dir,
            task_dir=cfg_task_env_runner.task.env_runner.dataset_path,
        )
    else:
        env_runner = hydra.utils.instantiate(
            cfg_task_env_runner.task.env_runner,
            output_dir=output_dir,
        )

    # -------------------------
    # run eval
    # -------------------------
    runner_log = env_runner.run(policy)

    # -------------------------
    # save eval results
    # -------------------------
    results = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            results[key] = value._path
        else:
            results[key] = value

    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Evaluation results saved to {results_path}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()