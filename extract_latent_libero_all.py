#!/usr/bin/env python3
"""
Extract planner-pipeline latents for ALL LIBERO tasks into a single HDF5.

For each task:
  1. Find the demo HDF5 file.
  2. Compute language features for the task.
  3. Run the planner pipeline (encode + predict + separate_emb) for each demo.
  4. Save latents under /data/<task_name>/<demo_name>/.

Output HDF5 structure:
  /data/<task_name>/<demo_name>/visual_latent      (T_eff, Dv)
  /data/<task_name>/<demo_name>/proprio_latent     (T_eff, Dp)
  /data/<task_name>/<demo_name>/obs_latent_concat  (T_eff, Dv + Dp)
  /data/<task_name>/attrs:
    language_goal: str
    n_demos: int

Usage:
  python extract_latent_libero_all.py \
    --dynamics_ckpt /path/to/model_50.pth \
    --demo_dir /path/to/training_data \
    --out_hdf5 /path/to/multitask_planner_latents.hdf5 \
    --batch_size 32 \
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
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.pose_util import axisangle2quat_batch
from diffusion_policy.common.language_models import extract_text_features, get_text_model
from dyn_model.plan import load_model
from dyn_model.datasets.img_transforms import get_eval_crop_transform_resnet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dynamics_ckpt", type=str, required=True)
    p.add_argument("--policy_ckpt_path", type=str, default=None)
    p.add_argument("--demo_dir", type=str, required=True,
                   help="Directory containing all task demo HDF5 files.")
    p.add_argument("--out_hdf5", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_demos_per_task", type=int, default=-1,
                   help="Limit demos per task (-1 for all).")
    p.add_argument("--max_tasks", type=int, default=-1,
                   help="Limit number of tasks (-1 for all).")
    p.add_argument("--task_pattern", type=str, default=r".*_demo\.hdf5$",
                   help="Regex for which demo files to include.")
    p.add_argument("--task_names", type=str, nargs="+", default=None,
                   help="If specified, only process these tasks (basenames "
                        "without _demo.hdf5).")
    p.add_argument("--demo_regex", type=str, default=r"demo_\d+")
    p.add_argument("--language_model_name", type=str, default="libero_10")
    p.add_argument("--language_emb_model", type=str, default="clip")
    p.add_argument("--no_language", action="store_true")
    return p.parse_args()


def resolve_path(p, base_dir):
    if p is None:
        return None
    p = str(p)
    if os.path.isabs(p):
        return p
    return str((base_dir / p).resolve())


def to_image_tensor(x_np):
    """(B, H, W, 3) -> (B, 3, H, W) float."""
    if x_np.ndim != 4:
        raise ValueError(f"Expected rank-4 image batch, got {x_np.shape}")
    if x_np.shape[-1] == 3:
        x_np = np.transpose(x_np, (0, 3, 1, 2))
    t = torch.from_numpy(x_np)
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    else:
        t = t.float()
    return t


def setup_models_and_normalizer(args):
    """Load dynamics model, normalizer, image transform, language tokenizer."""
    ckpt_path = Path(args.dynamics_ckpt)
    model_dir = ckpt_path.parents[1]

    hydra_yaml = model_dir / "hydra.yaml"
    normalizer_pth = model_dir / "normalizer.pth"

    if not hydra_yaml.exists():
        raise FileNotFoundError(f"Missing {hydra_yaml}")
    if not normalizer_pth.exists():
        raise FileNotFoundError(f"Missing {normalizer_pth}")

    cfg = OmegaConf.load(str(hydra_yaml))

    with open_dict(cfg):
        if hasattr(cfg, "policy_ckpt_path"):
            cfg.policy_ckpt_path = resolve_path(cfg.policy_ckpt_path, model_dir)
        if hasattr(cfg, "env") and hasattr(cfg.env, "policy_ckpt_path"):
            cfg.env.policy_ckpt_path = resolve_path(cfg.env.policy_ckpt_path, model_dir)
        if args.policy_ckpt_path is not None:
            cfg.policy_ckpt_path = args.policy_ckpt_path
            if hasattr(cfg, "env"):
                cfg.env.policy_ckpt_path = args.policy_ckpt_path

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    print(f"[setup] dynamics_ckpt: {ckpt_path}")
    print(f"[setup] model_dir: {model_dir}")

    dyn_model = load_model(ckpt_path, cfg, device=device)
    dyn_model.eval()

    wm_norm = LinearNormalizer()
    wm_norm.load_state_dict(torch.load(str(normalizer_pth), map_location="cpu"))
    wm_norm = wm_norm.to(device)

    use_crop = bool(getattr(cfg, "use_crop", False))
    original_img_size = int(getattr(cfg, "original_img_size", getattr(cfg, "img_size", 140)))
    cropped_img_size = int(getattr(cfg, "cropped_img_size", original_img_size))
    img_transform = None
    if use_crop:
        img_transform = get_eval_crop_transform_resnet(
            original_img_size=original_img_size,
            cropped_img_size=cropped_img_size,
        )

    view_names = list(cfg.view_names)
    frameskip = int(getattr(cfg, "frameskip", 1))

    # Pre-load language model if needed
    text_model = None
    tokenizer = None
    max_length = None
    if not args.no_language:
        text_model, tokenizer, max_length = get_text_model(
            args.language_model_name, args.language_emb_model
        )
        text_model = text_model.to(device)
        text_model.eval()

    print(f"[setup] view_names: {view_names}")
    print(f"[setup] use_crop: {use_crop} ({original_img_size} -> {cropped_img_size})")
    print(f"[setup] frameskip: {frameskip}")
    print(f"[setup] device: {device}")
    print(f"[setup] language model: {args.language_model_name} / {args.language_emb_model}"
          f" {'(loaded)' if text_model is not None else '(disabled)'}")

    return {
        "dyn_model": dyn_model,
        "wm_norm": wm_norm,
        "img_transform": img_transform,
        "view_names": view_names,
        "frameskip": frameskip,
        "use_crop": use_crop,
        "original_img_size": original_img_size,
        "cropped_img_size": cropped_img_size,
        "device": device,
        "cfg": cfg,
        "text_model": text_model,
        "tokenizer": tokenizer,
        "max_length": max_length,
        "language_emb_model": args.language_emb_model,
    }


def extract_language_for_task(task_name, setup):
    """Compute language features for a specific task."""
    text_model = setup["text_model"]
    tokenizer = setup["tokenizer"]
    max_length = setup["max_length"]
    device = setup["device"]

    if text_model is None:
        return None, None

    # Convert task_name (with underscores) to natural language goal
    # e.g., "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    #     -> "KITCHEN SCENE3 turn on the stove and put the moka pot on it"
    language_goal = " ".join(task_name.split("_"))

    text_tokens = tokenizer(
        [language_goal],
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        text_latents = extract_text_features(
            text_model, text_tokens,
            language_emb_model=setup["language_emb_model"],
        )

    if text_latents.ndim == 2:
        text_latents = text_latents.unsqueeze(1)

    return text_latents, language_goal


def prepare_obs_batch(obs_grp, t_indices, setup, text_latents=None):
    """Build obs_wm for a batch of timesteps."""
    view_names = setup["view_names"]
    wm_norm = setup["wm_norm"]
    img_transform = setup["img_transform"]
    use_crop = setup["use_crop"]
    original_img_size = setup["original_img_size"]
    cropped_img_size = setup["cropped_img_size"]
    device = setup["device"]

    B = len(t_indices)

    # Visual
    visual = {}
    for v in view_names:
        images = []
        for t in t_indices:
            img = obs_grp[v][t:t + 1]
            images.append(img)
        x_np = np.concatenate(images, axis=0)
        x = to_image_tensor(x_np).to(device)
        x = wm_norm[v].normalize(x)
        if use_crop and img_transform is not None:
            x = img_transform(x.view(-1, 3, original_img_size, original_img_size))
            x = x.view(B, 1, 3, cropped_img_size, cropped_img_size)
        else:
            x = x.view(B, 1, 3, original_img_size, original_img_size)
        visual[v] = x

    # Proprio
    ee_oris = []
    ee_poses = []
    joints = []
    for t in t_indices:
        ee_oris.append(obs_grp["ee_ori"][t:t + 1])
        ee_poses.append(obs_grp["ee_pos"][t:t + 1])
        joints.append(obs_grp["joint_states"][t:t + 1])
    ee_ori = np.concatenate(ee_oris, axis=0)
    ee_pos = np.concatenate(ee_poses, axis=0)
    joint_states = np.concatenate(joints, axis=0)

    ee_ori = axisangle2quat_batch(ee_ori)

    proprio = np.concatenate([ee_ori, ee_pos, joint_states], axis=-1)
    proprio = torch.from_numpy(proprio).float().to(device)
    proprio = wm_norm["state"].normalize(proprio)
    proprio = proprio.unsqueeze(1)

    obs_wm = {"visual": visual, "proprio": proprio}

    if text_latents is not None:
        lang = text_latents.expand(B, -1, -1)
        obs_wm["language"] = lang

    return obs_wm


def prepare_actions_batch(actions_np, t_indices, frameskip, wm_norm,
                          rotation_transformer, device):
    """Build action input for a batch."""
    B = len(t_indices)

    batched_actions = []
    for t in t_indices:
        window = actions_np[t:t + frameskip]
        if window.shape[0] < frameskip:
            pad_count = frameskip - window.shape[0]
            pad = np.tile(window[-1:], (pad_count, 1))
            window = np.concatenate([window, pad], axis=0)
        batched_actions.append(window)
    batched_actions = np.stack(batched_actions, axis=0)

    if batched_actions.shape[-1] == 7:
        pos = batched_actions[..., :3]
        aa = batched_actions[..., 3:6]
        grip = batched_actions[..., 6:7]
        rot6d = rotation_transformer.forward(aa.reshape(-1, 3)).reshape(B, frameskip, 6)
        batched_actions = np.concatenate([pos, rot6d, grip], axis=-1)

    actions_t = torch.from_numpy(batched_actions).float().to(device)

    B_, F_, A_ = actions_t.shape
    actions_flat = actions_t.reshape(B_ * F_, A_)
    actions_norm = wm_norm["act"].normalize(actions_flat).reshape(B_, F_, A_)

    action_batch = actions_norm.reshape(B_, 1, F_ * A_)
    return action_batch


def run_planner_pipeline(obs_wm, action_batch, dyn_model):
    """encode + predict + separate_emb."""
    with torch.no_grad():
        z = dyn_model.encode(obs_wm, action_batch)
        z_pred = dyn_model.predict(z)
        z_new = z_pred[:, -1:, ...]
        z_obs, _ = dyn_model.separate_emb(z_new)

    v = z_obs["visual"]
    p = z_obs["proprio"]

    if v.ndim > 2:
        v = v.reshape(v.size(0), -1)
    if p.ndim > 2:
        p = p.reshape(p.size(0), -1)

    return v, p


def process_demo(demo, fin, fout_task_grp, setup, text_latents, args):
    """Process a single demo within a task."""
    dyn_model = setup["dyn_model"]
    wm_norm = setup["wm_norm"]
    frameskip = setup["frameskip"]
    device = setup["device"]
    view_names = setup["view_names"]
    rotation_transformer = setup["rotation_transformer"]

    obs_grp = fin["data"][demo]["obs"]
    actions_np = fin["data"][demo]["actions"][:]

    T = obs_grp[view_names[0]].shape[0]
    T_eff = T - frameskip

    if T_eff <= 0:
        print(f"      [skip] {demo}: T={T} too short for frameskip={frameskip}")
        return False

    z_vis_chunks = []
    z_pro_chunks = []

    for start in range(0, T_eff, args.batch_size):
        end = min(T_eff, start + args.batch_size)
        t_indices = list(range(start, end))

        obs_wm = prepare_obs_batch(obs_grp, t_indices, setup, text_latents=text_latents)
        action_batch = prepare_actions_batch(
            actions_np, t_indices, frameskip, wm_norm, rotation_transformer, device
        )

        v, p = run_planner_pipeline(obs_wm, action_batch, dyn_model)

        z_vis_chunks.append(v.detach().cpu())
        z_pro_chunks.append(p.detach().cpu())

        if device.type == "cuda":
            torch.cuda.empty_cache()

    z_vis_all = torch.cat(z_vis_chunks, dim=0)
    z_pro_all = torch.cat(z_pro_chunks, dim=0)

    z_vis_flat = z_vis_all.reshape(z_vis_all.shape[0], -1) if z_vis_all.ndim > 2 else z_vis_all
    z_concat = torch.cat([z_vis_flat, z_pro_all], dim=-1)

    demo_grp = fout_task_grp.create_group(demo)
    demo_grp.create_dataset(
        "visual_latent", data=z_vis_all.numpy(),
        compression="gzip", compression_opts=4, shuffle=True,
    )
    demo_grp.create_dataset(
        "proprio_latent", data=z_pro_all.numpy(),
        compression="gzip", compression_opts=4, shuffle=True,
    )
    demo_grp.create_dataset(
        "obs_latent_concat", data=z_concat.numpy(),
        compression="gzip", compression_opts=4, shuffle=True,
    )
    demo_grp.attrs["T_original"] = T
    demo_grp.attrs["T_effective"] = T_eff
    demo_grp.attrs["frameskip"] = frameskip

    return True


def process_task(task_path, task_name, fout, setup, args):
    """Process all demos for one task."""
    demo_pat = re.compile(args.demo_regex)

    text_latents, language_goal = extract_language_for_task(task_name, setup)

    print(f"  Language goal: '{language_goal}'")

    n_processed = 0
    with h5py.File(task_path, "r") as fin:
        if "data" not in fin:
            print(f"  [warn] no /data group, skipping")
            return 0

        demos = sorted([k for k in fin["data"].keys() if demo_pat.fullmatch(k)])
        if args.max_demos_per_task > 0:
            demos = demos[:args.max_demos_per_task]

        if len(demos) == 0:
            print(f"  [warn] no demos matching pattern")
            return 0

        print(f"  {len(demos)} demos to process")

        # Create task group in output
        task_path_in_h5 = f"data/{task_name}"
        if task_path_in_h5 in fout:
            del fout[task_path_in_h5]
        task_grp = fout.create_group(task_path_in_h5)

        for demo in demos:
            print(f"    {demo}...", end="", flush=True)
            ok = process_demo(demo, fin, task_grp, setup, text_latents, args)
            if ok:
                n_processed += 1
                print(" done")
            else:
                print(" skipped")

        task_grp.attrs["language_goal"] = language_goal or ""
        task_grp.attrs["n_demos"] = n_processed
        task_grp.attrs["source_file"] = str(task_path)

    return n_processed


def main():
    args = parse_args()

    print("=" * 72)
    print("Multi-Task Planner-Pipeline Latent Extraction")
    print("=" * 72)

    setup = setup_models_and_normalizer(args)
    setup["rotation_transformer"] = RotationTransformer("axis_angle", "rotation_6d")
    print("[setup] Rotation transformer: axis_angle -> rotation_6d")
    print()

    # Find task files
    demo_dir = Path(args.demo_dir)
    if not demo_dir.is_dir():
        raise NotADirectoryError(f"{demo_dir} is not a directory")

    task_pattern = re.compile(args.task_pattern)
    task_files = sorted([f for f in demo_dir.iterdir()
                         if f.is_file() and task_pattern.search(f.name)])

    if args.task_names is not None:
        # Filter to specified tasks
        wanted = set(args.task_names)
        task_files = [
            f for f in task_files
            if f.name.replace("_demo.hdf5", "") in wanted
        ]

    if args.max_tasks > 0:
        task_files = task_files[:args.max_tasks]

    if len(task_files) == 0:
        raise ValueError(f"No task files found in {demo_dir}")

    print(f"Found {len(task_files)} tasks:")
    for tf in task_files:
        print(f"  {tf.name}")
    print()

    # Output file
    out_path = Path(args.out_hdf5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as fout:
        fout.attrs["source_dir"] = str(demo_dir)
        fout.attrs["dynamics_ckpt"] = str(args.dynamics_ckpt)
        fout.attrs["extraction_pipeline"] = "planner"
        fout.attrs["frameskip"] = setup["frameskip"]
        fout.attrs["view_names"] = np.string_(str(setup["view_names"]))
        fout.attrs["multitask"] = True
        fout.attrs["use_language"] = not args.no_language

        total_demos = 0
        for i, task_file in enumerate(task_files):
            # Strip "_demo.hdf5" to get task name
            task_name = task_file.name
            if task_name.endswith("_demo.hdf5"):
                task_name = task_name[:-len("_demo.hdf5")]
            elif task_name.endswith(".hdf5"):
                task_name = task_name[:-len(".hdf5")]

            print(f"[{i + 1}/{len(task_files)}] Task: {task_name}")
            n = process_task(task_file, task_name, fout, setup, args)
            total_demos += n
            print(f"  -> processed {n} demos")
            print()

        fout.attrs["total_demos"] = total_demos
        fout.attrs["n_tasks"] = len(task_files)

    print("=" * 72)
    print(f"Done. Wrote {total_demos} demos across {len(task_files)} tasks to:")
    print(f"  {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()