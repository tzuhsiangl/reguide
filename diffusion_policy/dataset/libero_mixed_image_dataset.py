import math
import numpy as np
import torch
from typing import Dict

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.libero_replay_image_dataset import LiberoReplayImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class LiberoMixedImageDataset(BaseImageDataset):
    """
    Two-buffer dataset for continual learning with LIBERO. Mixes expert demos
    with rollout demos at a fixed sampling ratio.

    - Each underlying buffer is loaded by LiberoReplayImageDataset, so the
      existing zarr cache / SequenceSampler / language-emb logic is reused.
    - Sampling: at __getitem__ time, draw from expert with probability
      `expert_sample_weight`, otherwise from rollout. The DataLoader's `idx`
      is ignored on purpose; sample identity is decided by the random draw.
    - Normalization: stats computed on EXPERT only. Keeps action/obs scaling
      matched to the clean expert distribution.
    - Validation: defers to expert.get_validation_dataset() so val loss
      reflects expert imitation.

    Epoch length is NOT provided manually. It is computed from
    rollout_reuse_rate so that, in expectation, each rollout window is sampled
    rollout_reuse_rate times per epoch.

        rollout_reuse_rate = r means:
            expected_rollout_draws_per_epoch / n_rollout = r
    """

    def __init__(
        self,
        shape_meta: dict,
        expert_dataset_path: str,
        rollout_dataset_path: str,
        expert_sample_weight: float = 0.7,
        rollout_reuse_rate: float = 1.0,
        rollout_language_goal: str = None,  # rollout-only goal override (single-task file)
        # forwarded to both LiberoReplayImageDataset instances
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int = None,
        abs_action: bool = False,
        rotation_rep: str = "rotation_6d",
        use_legacy_normalizer: bool = False,
        use_cache: bool = True,
        seed: int = 42,
        val_ratio: float = 0.0,
        language_emb_model: str = None,
        data_aug: bool = False,
    ):
        if not (0.0 <= expert_sample_weight < 1.0):
            raise ValueError(
                f"expert_sample_weight must be in [0, 1), got {expert_sample_weight}. "
                f"Use rollout_reuse_rate only when rollout can actually be sampled."
            )
        if rollout_reuse_rate <= 0:
            raise ValueError(
                f"rollout_reuse_rate must be positive, got {rollout_reuse_rate}"
            )

        shared_kwargs = dict(
            shape_meta=shape_meta,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            n_obs_steps=n_obs_steps,
            abs_action=abs_action,
            rotation_rep=rotation_rep,
            use_legacy_normalizer=use_legacy_normalizer,
            use_cache=use_cache,
            seed=seed,
            val_ratio=val_ratio,
            language_emb_model=language_emb_model,
            data_aug=data_aug,
        )

        # Expert is a multi-task folder: each file resolves its own goal from
        # filename/attr, so NO override here.
        self.expert = LiberoReplayImageDataset(
            dataset_path=expert_dataset_path, **shared_kwargs
        )
        # Rollout is a single-task file: stamp the one goal via override.
        self.rollout = LiberoReplayImageDataset(
            dataset_path=rollout_dataset_path,
            language_goal=rollout_language_goal,
            **shared_kwargs,
        )

        self.expert_sample_weight = float(expert_sample_weight)
        self.rollout_reuse_rate = float(rollout_reuse_rate)

        n_expert = len(self.expert)
        n_rollout = len(self.rollout)

        if n_rollout <= 0:
            raise ValueError("rollout dataset has no windows, cannot use rollout_reuse_rate.")

        # expected rollout draws per epoch = epoch_length * (1 - w)
        # want this to equal rollout_reuse_rate * n_rollout
        self._len = int(math.ceil(
            self.rollout_reuse_rate * n_rollout / (1.0 - self.expert_sample_weight)
        ))

        # Mirror a few attrs in case downstream code reaches for them.
        self.shape_meta = shape_meta
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.use_legacy_normalizer = use_legacy_normalizer

        expected_expert_draws = self._len * self.expert_sample_weight
        expected_rollout_draws = self._len * (1.0 - self.expert_sample_weight)

        expert_reuse = expected_expert_draws / n_expert if n_expert > 0 else 0.0
        rollout_reuse = expected_rollout_draws / n_rollout if n_rollout > 0 else 0.0

        print(
            f"[LiberoMixedImageDataset] "
            f"expert={n_expert} samples, "
            f"rollout={n_rollout} samples, "
            f"expert_sample_weight={self.expert_sample_weight}, "
            f"rollout_reuse_rate={self.rollout_reuse_rate}, "
            f"epoch_length={self._len}, "
            f"expected_expert_draws={expected_expert_draws:.1f}, "
            f"expected_rollout_draws={expected_rollout_draws:.1f}, "
            f"expert_reuse~={expert_reuse:.3f}, "
            f"rollout_reuse~={rollout_reuse:.3f}"
        )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # idx intentionally ignored; source and sample are chosen randomly
        if np.random.rand() < self.expert_sample_weight:
            sub_idx = np.random.randint(len(self.expert))
            return self.expert[sub_idx]
        else:
            sub_idx = np.random.randint(len(self.rollout))
            return self.rollout[sub_idx]

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        return self.expert.get_normalizer(**kwargs)

    def get_validation_dataset(self):
        return self.expert.get_validation_dataset()

    def get_all_actions(self) -> torch.Tensor:
        return self.expert.get_all_actions()