if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.env_runner.load_env import env_rollout, load_env_runner
import dill
from omegaconf import open_dict

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetHybridLiberoContinualWorkspace(BaseWorkspace):
    """
    Identical training loop to TrainDiffusionUnetHybridWorkspace. Lives in its
    own file so we can evolve the continual-learning workspace independently
    (e.g., add per-task validation, EMA carryover from a prior run, mixing-ratio
    schedules) without affecting the original Robomimic pipeline.
    """

    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # if cfg.training.resume:
        #     lastest_ckpt_path = self.get_checkpoint_path()
        #     if lastest_ckpt_path.is_file():
        #         print(f"Resuming from checkpoint {lastest_ckpt_path}")
        #         self.load_checkpoint(path=lastest_ckpt_path)
        #====================for finetuning =======================
        # =========================
        # Optional: initialize model from pretrained checkpoint
        # =========================
        init_ckpt_path = getattr(cfg.training, "init_ckpt_path", None)

        init_ckpt_path_active = (
            init_ckpt_path is not None
            and str(init_ckpt_path).lower() not in ["", "none", "null"]
        )

        if init_ckpt_path_active:
            init_ckpt_path = pathlib.Path(os.path.expanduser(str(init_ckpt_path)))

            if not init_ckpt_path.is_file():
                raise FileNotFoundError(
                    f"training.init_ckpt_path does not exist: {init_ckpt_path}"
                )

            print(f"Initializing model weights from pretrained checkpoint: {init_ckpt_path}")

            payload = torch.load(
                init_ckpt_path,
                map_location="cpu",
                pickle_module=dill,
                weights_only=False,
            )

            if "state_dicts" not in payload:
                raise KeyError(
                    f"Checkpoint does not contain 'state_dicts'. Available keys: {list(payload.keys())}"
                )

            state_dicts = payload["state_dicts"]
            print("Checkpoint state_dict keys:", list(state_dicts.keys()))

            init_from_ema = bool(getattr(cfg.training, "init_from_ema", True))

            if init_from_ema and "ema_model" in state_dicts:
                print("Loading ema_model weights into model.")
                self.model.load_state_dict(state_dicts["ema_model"], strict=True)
            elif "model" in state_dicts:
                print("Loading raw model weights into model.")
                self.model.load_state_dict(state_dicts["model"], strict=True)
            else:
                raise KeyError(
                    f"Checkpoint has neither 'ema_model' nor 'model'. "
                    f"Available state_dict keys: {list(state_dicts.keys())}"
                )

            if cfg.training.use_ema and self.ema_model is not None:
                print("Copying initialized model weights into ema_model.")
                self.ema_model.load_state_dict(self.model.state_dict(), strict=True)

            # New run: reset counters
            self.global_step = 0
            self.epoch = 0
        #=========================================================
        # configure dataset (the wrapper handles expert + rollout mixing)
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            ) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # env_runner: BaseImageRunner = hydra.utils.instantiate(
        #     cfg.task.env_runner, output_dir=self.output_dir
        # )
        # assert isinstance(env_runner, BaseImageRunner)
        # configure env
        if "libero" not in cfg.task.name:
            env_runner: BaseImageRunner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir
            )
            assert isinstance(env_runner, BaseImageRunner)
        else:
            with open_dict(cfg):
                cfg.task.dataset.dataset_path = cfg.task.dataset.expert_dataset_path
            env_runner = load_env_runner(cfg, self.output_dir)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        state_dict = normalizer.state_dict()
        print('saving normalizer to ', os.path.join(self.output_dir,"normalizer.pth"))
        torch.save(state_dict, os.path.join(self.output_dir,"normalizer.pth"))

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     step_log.update(runner_log)
                if (self.epoch % cfg.training.rollout_every) == 0:
                    if "libero" not in cfg.task.name:
                        runner_log = env_runner.run(policy)
                    else:
                        runner_log = env_rollout(cfg, env_runner, policy)
                    step_log.update(runner_log)

                # if (self.epoch % cfg.training.val_every) == 0:
                #     with torch.no_grad():
                #         val_losses = list()
                #         with tqdm.tqdm(
                #             val_dataloader,
                #             desc=f"Validation epoch {self.epoch}",
                #             leave=False,
                #             mininterval=cfg.training.tqdm_interval_sec,
                #         ) as tepoch:
                #             for batch_idx, batch in enumerate(tepoch):
                #                 batch = dict_apply(
                #                     batch, lambda x: x.to(device, non_blocking=True)
                #                 )
                #                 loss = self.model.compute_loss(batch)
                #                 val_losses.append(loss)
                #                 if (cfg.training.max_val_steps is not None) and batch_idx >= (
                #                     cfg.training.max_val_steps - 1
                #                 ):
                #                     break
                #         if len(val_losses) > 0:
                #             val_loss = torch.mean(torch.tensor(val_losses)).item()
                #             step_log["val_loss"] = val_loss

                if (self.epoch % cfg.training.sample_every) == 0 and 'libero' not in cfg.task.name:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(tag=str(self.epoch))
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # metric_dict = dict()
                    # for key, value in step_log.items():
                    #     new_key = key.replace("/", "_")
                    #     metric_dict[new_key] = value

                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    # if topk_ckpt_path is not None:
                    #     self.save_checkpoint(path=topk_ckpt_path)

                # policy.train()
                self.model.train()
                # EMA model stays in eval
                if self.ema_model is not None:
                    self.ema_model.eval()

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionUnetHybridLiberoContinualWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()