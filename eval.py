"""
 python eval.py --config-name=eval_pusht
"""
import sys
import os
import pathlib

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# line-buffered stdout / stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


@hydra.main(config_path="dyn_model/conf/planner", config_name="eval_pusht")
def main(cfg: DictConfig):
    """
    Evaluate a diffusion policy checkpoint (no dynamics-model guidance),
    using the newer Hydra-based config style.
    """
    output_dir = cfg.output_dir

    # Handle existing output dir
    if os.path.exists(output_dir):
        confirm = input(f"Output path {output_dir} already exists! Overwrite? (y/N): ")
        if confirm.lower() != 'y':
            sys.exit(1)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save config used for this eval
    config_save_path = os.path.join(output_dir, 'eval_config.yaml')
    OmegaConf.save(config=cfg, f=config_save_path)
    print(f"Configuration saved to {config_save_path}")

    # -------------------------------------------------------------------------
    # Load diffusion policy checkpoint
    # -------------------------------------------------------------------------
    with open(cfg.policy_checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)

    # This is the original training cfg saved in the checkpoint
    cfg_task_env_runner = payload['cfg']

    # -------------------------------------------------------------------------
    # Override some runtime parameters using eval config
    # (same style as your planner script, but without planner stuff)
    # -------------------------------------------------------------------------
    # action horizon
    cfg_task_env_runner.n_action_steps = cfg.n_action_steps
    cfg_task_env_runner.task.env_runner.n_action_steps = cfg.n_action_steps
    cfg_task_env_runner.policy.n_action_steps = cfg.n_action_steps

    # eval counts / seeds
    cfg_task_env_runner.task.env_runner.n_test = cfg.n_test
    cfg_task_env_runner.task.env_runner.n_test_vis = cfg.n_test
    cfg_task_env_runner.task.env_runner.n_train = 0
    cfg_task_env_runner.task.env_runner.n_train_vis = 0
    cfg_task_env_runner.task.env_runner.test_start_seed = cfg.test_start_seed

    # For libero-style tasks, you might want to override dataset_path
    if 'libero' in cfg.policy_checkpoint:
        cfg_task_env_runner.task.env_runner.dataset_path = cfg.dataset_path

    # -------------------------------------------------------------------------
    # Initialize workspace and load model
    # -------------------------------------------------------------------------
    cls = hydra.utils.get_class(cfg_task_env_runner._target_)
    workspace = cls(cfg_task_env_runner, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Get policy (EMA if enabled)
    policy = workspace.model
    if cfg_task_env_runner.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(cfg.device)
    policy.to(device)
    policy.eval()

    # -------------------------------------------------------------------------
    # Load normalizer from normalizer.pth (newer DP behavior)
    # -------------------------------------------------------------------------
    normalizer_dir = os.path.dirname(os.path.dirname(cfg.policy_checkpoint))
    normalizer_path = os.path.join(normalizer_dir, 'normalizer.pth')
    print(f"Loading normalizer from {normalizer_path}")
    policy.normalizer.load_state_dict(torch.load(normalizer_path, map_location=device))
    policy.normalizer.to(device)

    # -------------------------------------------------------------------------
    # Instantiate env runner (no planner / guidance)
    # -------------------------------------------------------------------------
    # Let the planner config choose which env_runner implementation to use
    cfg_task_env_runner.task.env_runner._target_ = cfg.env_runner_target

    dataset_target = payload['cfg'].task.dataset._target_
    if 'libero' in dataset_target:
        env_runner = hydra.utils.instantiate(
            cfg_task_env_runner.task.env_runner,
            output_dir=output_dir,
            task_dir=cfg_task_env_runner.task.env_runner.dataset_path
        )
    else:
        env_runner = hydra.utils.instantiate(
            cfg_task_env_runner.task.env_runner,
            output_dir=output_dir
        )

    # -------------------------------------------------------------------------
    # Run evaluation
    # -------------------------------------------------------------------------
    runner_log = env_runner.run(policy)

    # Save results as JSON
    results = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            results[key] = value._path
        else:
            results[key] = value

    results_path = os.path.join(output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Evaluation results saved to {results_path}")


if __name__ == '__main__':
    main()
