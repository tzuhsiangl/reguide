#!/usr/bin/env python3
"""
Extract *encoded observations* (no prediction) episode-by-episode from a RoboMimic-style HDF5 file
using the pretrained encoder of your dynamics model, and save:

  /data/demo_k/visual_latent        (T, ...)   raw visual encoder output
  /data/demo_k/proprio_latent       (T, Dp)    raw proprio encoder output
  /data/demo_k/obs_latent_concat    (T, Dflat+Dp)  flatten(visual_latent) || proprio_latent

This script reads the model interface from the dynamics model's hydra.yaml:
- views      = cfg.view_names
- proprio    = keys in cfg.env.shape_obs with type != 'rgb' (in YAML order)
- crop       = cfg.use_crop, cfg.original_img_size, cfg.cropped_img_size
- normalizer = normalizer.pth in the same folder as hydra.yaml

Expected input HDF5 layout (matches your file):
  /data/demo_9/obs/shouldercamera0_image   (T,140,140,3)
  /data/demo_9/obs/robot0_eef_pos          (T,3)
  ...

Run:
  python extract_latents.py \
    --dynamics_ckpt /path/to/trained/dyn_model \
    --policy_ckpt_path /path/to/trained/diffuison/policy \
    --in_hdf5 /path/to/training_data \
    --out_hdf5 /path/to/output \
    --batch_size 64 \
    --device cuda
"""

import argparse
import os
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from diffusion_policy.model.common.normalizer import LinearNormalizer
from dyn_model.plan import load_model
from dyn_model.datasets.img_transforms import get_eval_crop_transform_resnet
from typing import Optional



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dynamics_ckpt", type=str, required=True, help="Path to dynamics model checkpoint (.ckpt or .pth).")
    p.add_argument(
        "--policy_ckpt_path",
        type=str,
        default=None,
        help="ABSOLUTE path to base policy checkpoint used to init the ResNet encoder (overrides hydra.yaml).",
    )
    p.add_argument("--in_hdf5", type=str, required=True, help="Input RoboMimic HDF5 file.")
    p.add_argument("--out_hdf5", type=str, required=True, help="Output HDF5 to write latents.")
    p.add_argument("--batch_size", type=int, default=64, help="Timesteps per batch for encoding.")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    p.add_argument("--max_demos", type=int, default=-1, help="-1 for all demos; otherwise limit.")
    p.add_argument("--demo_regex", type=str, default=r"demo_\d+", help="Regex for demo group names under /data.")
    return p.parse_args()


def to_image_tensor(x: np.ndarray) -> torch.Tensor:
    """(B,H,W,3) uint8/float -> (B,3,H,W) float; if uint8 scale to [0,1]."""
    if x.ndim != 4:
        raise ValueError(f"Expected image batch rank 4, got {x.shape}")
    if x.shape[-1] == 3:
        x = np.transpose(x, (0, 3, 1, 2))
    if x.shape[1] != 3:
        raise ValueError(f"Expected channel dim=3, got {x.shape}")
    t = torch.from_numpy(x)
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    else:
        t = t.float()
    return t


def resolve_path(p: Optional[str], base_dir: Path) -> Optional[str]:
    """Resolve relative paths against base_dir; keep absolute paths as-is."""
    if p is None:
        return None
    p = str(p)
    if os.path.isabs(p):
        return p
    return str((base_dir / p).resolve())


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt_path = Path(args.dynamics_ckpt)
    model_dir = ckpt_path.parents[1]  # same convention as your training code

    hydra_yaml = model_dir / "hydra.yaml"
    normalizer_pth = model_dir / "normalizer.pth"
    if not hydra_yaml.exists():
        raise FileNotFoundError(f"Cannot find hydra.yaml at: {hydra_yaml}")
    if not normalizer_pth.exists():
        raise FileNotFoundError(f"Cannot find normalizer.pth at: {normalizer_pth}")

    cfg = OmegaConf.load(str(hydra_yaml))

    # --- Fix: resolve relative paths + allow command-line override for policy_ckpt_path ---
    with open_dict(cfg):
        # first resolve any relative path against model_dir
        if hasattr(cfg, "policy_ckpt_path"):
            cfg.policy_ckpt_path = resolve_path(cfg.policy_ckpt_path, model_dir)
        if hasattr(cfg, "env") and hasattr(cfg.env, "policy_ckpt_path"):
            cfg.env.policy_ckpt_path = resolve_path(cfg.env.policy_ckpt_path, model_dir)

        # then override if user provided an absolute path
        if args.policy_ckpt_path is not None:
            if not os.path.isabs(args.policy_ckpt_path):
                raise ValueError("--policy_ckpt_path must be an ABSOLUTE path.")
            if not os.path.exists(args.policy_ckpt_path):
                raise FileNotFoundError(f"--policy_ckpt_path does not exist: {args.policy_ckpt_path}")
            cfg.policy_ckpt_path = args.policy_ckpt_path
            if hasattr(cfg, "env"):
                cfg.env.policy_ckpt_path = args.policy_ckpt_path

    # Authoritative IO spec from hydra.yaml
    view_names = list(cfg.view_names)
   


    if not (hasattr(cfg, "env") and hasattr(cfg.env, "shape_obs")):
        raise ValueError("hydra.yaml missing env.shape_obs; cannot determine proprio keys.")
    shape_obs = cfg.env.shape_obs
    # proprio_keys = [k for k, meta in shape_obs.items() if meta.get("type", None) != "rgb"]
    proprio_keys = [
        k for k, meta in shape_obs.items()
        if meta.get("type", None) != "rgb" and k != "language"
    ]

    use_crop = bool(getattr(cfg, "use_crop", False))
    original_img_size = int(getattr(cfg, "original_img_size", getattr(cfg, "img_size", 140)))
    cropped_img_size = int(getattr(cfg, "cropped_img_size", original_img_size))

    img_transform = None
    if use_crop:
        img_transform = get_eval_crop_transform_resnet(
            original_img_size=original_img_size,
            cropped_img_size=cropped_img_size,
        )

    print("[info] dynamics_ckpt:", ckpt_path)
    print("[info] model_dir:", model_dir)
    print("[info] policy_ckpt_path used:", getattr(cfg, "policy_ckpt_path", None))
    print("[info] views:", view_names)
    print("[info] proprio_keys:", proprio_keys)
    print(f"[info] crop: {use_crop} (orig={original_img_size}, crop={cropped_img_size})")
    print("[info] device:", device)
    # import pdb; pdb.set_trace()
    # Load model (encoder init will use cfg.policy_ckpt_path, now valid)
    dyn_model = load_model(ckpt_path, cfg, device=device)
    dyn_model.eval()

    # Load normalizer
    wm_norm = LinearNormalizer()
    wm_norm.load_state_dict(torch.load(str(normalizer_pth), map_location="cpu"))
    wm_norm = wm_norm.to(device)

    # Sanity: required normalizer keys exist
    _ = wm_norm["state"]
    for v in view_names:
        _ = wm_norm[v]

    in_path = Path(args.in_hdf5)
    out_path = Path(args.out_hdf5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    demo_pat = re.compile(args.demo_regex)

    with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
        if "data" not in fin:
            raise KeyError("Input HDF5 missing top-level group '/data'.")

        demos = sorted([k for k in fin["data"].keys() if demo_pat.fullmatch(k)])
        if args.max_demos > 0:
            demos = demos[: args.max_demos]

        print(f"[info] Found {len(demos)} demos under /data")

        # Global metadata
        fout.attrs["source_hdf5"] = str(in_path)
        fout.attrs["dynamics_ckpt"] = str(ckpt_path)
        fout.attrs["policy_ckpt_path"] = str(getattr(cfg, "policy_ckpt_path", ""))
        fout.attrs["view_names"] = np.string_(str(view_names))
        fout.attrs["proprio_keys"] = np.string_(str(proprio_keys))
        fout.attrs["use_crop"] = int(use_crop)
        fout.attrs["original_img_size"] = original_img_size
        fout.attrs["cropped_img_size"] = cropped_img_size

        for demo in demos:
            obs_grp = fin["data"][demo]["obs"]

            if view_names[0] not in obs_grp:
                raise KeyError(f"Demo {demo}: missing view '{view_names[0]}' in /obs.")
            T = obs_grp[view_names[0]].shape[0]
            print(f"[info] {demo}: T={T}")

            z_vis_chunks = []
            z_pro_chunks = []

            for start in range(0, T, args.batch_size):
                end = min(T, start + args.batch_size)
                B = end - start

                visual = {}
                for v in view_names:
                    x = obs_grp[v][start:end]          # (B,H,W,3)
                    x = to_image_tensor(x).to(device)  # (B,3,H,W)
                    x = wm_norm[v].normalize(x)
                    if use_crop:
                        x = img_transform(x)
                    visual[v] = x.unsqueeze(1)         # (B,1,3,H,W)

                props = []
                for k in proprio_keys:
                    if k not in obs_grp:
                        raise KeyError(f"Demo {demo}: missing proprio key '{k}' in /obs.")
                    a = obs_grp[k][start:end]
                    props.append(torch.from_numpy(a).float().to(device))
                proprio = torch.cat(props, dim=-1)     # (B, D_total)
                proprio = wm_norm["state"].normalize(proprio)
                proprio = proprio.unsqueeze(1)         # (B,1,D)

                obs_wm = {"visual": visual, "proprio": proprio}

                with torch.no_grad():
                    enc = dyn_model.encode_obs(obs_wm)
                    z_vis = enc["visual"].detach().cpu().squeeze(1)   # (B, ...)
                    z_pro = enc["proprio"].detach().cpu().squeeze(1)  # (B, Dp)

                z_vis_chunks.append(z_vis)
                z_pro_chunks.append(z_pro)

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            z_vis_all = torch.cat(z_vis_chunks, dim=0)  # (T, ...)
            z_pro_all = torch.cat(z_pro_chunks, dim=0)  # (T, Dp)

            z_vis_flat = z_vis_all.reshape(z_vis_all.shape[0], -1) if z_vis_all.ndim > 2 else z_vis_all
            z_concat = torch.cat([z_vis_flat, z_pro_all], dim=-1)   # (T, Dflat+Dp)

            out_demo = fout.create_group(f"data/{demo}")
            out_demo.create_dataset(
                "visual_latent", data=z_vis_all.numpy(),
                compression="gzip", compression_opts=4, shuffle=True
            )
            out_demo.create_dataset(
                "proprio_latent", data=z_pro_all.numpy(),
                compression="gzip", compression_opts=4, shuffle=True
            )
            out_demo.create_dataset(
                "obs_latent_concat", data=z_concat.numpy(),
                compression="gzip", compression_opts=4, shuffle=True
            )
            out_demo.attrs["T"] = T

            print(
                f"[done] {demo}: visual {tuple(z_vis_all.shape)} | "
                f"proprio {tuple(z_pro_all.shape)} | concat {tuple(z_concat.shape)}"
            )

    print(f"[done] Wrote encoded observation latents to: {out_path}")


if __name__ == "__main__":
    main()