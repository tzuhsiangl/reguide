#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import yaml
"""
python merge_by_seed.py \
  --search_roots /path/to/root/of/the/eval/result \
  --seeds 1 2 3 4 5 \
  --rollout_type success \
  --pick_mode best \
  --out /root/to/output/merged/data.hdf5 \
  --overwrite \
  --shuffle \
  --shuffle_seed 42

"""

DEMO_RE = re.compile(r"^demo_(\d+)$")


# =========================
# Eval discovery utilities
# =========================
def find_eval_files(category_path: str) -> List[Tuple[Path, Path]]:
    """
    Recursively search under category_path for run folders that contain
    both eval_config.yaml and eval_results.json (or eval_result.json).
    """
    eval_pairs = []
    category_path = Path(category_path)

    if not category_path.exists():
        return eval_pairs

    result_files = list(category_path.rglob("eval_results.json"))
    result_files += list(category_path.rglob("eval_result.json"))

    seen = set()
    for result_file in sorted(set(result_files)):
        run_dir = result_file.parent
        config_file = run_dir / "eval_config.yaml"

        if config_file.exists():
            key = (str(config_file), str(result_file))
            if key not in seen:
                seen.add(key)
                eval_pairs.append((config_file, result_file))
                print(f"[FOUND EVAL] {run_dir}")
        else:
            print(f"[MISSING] eval_config.yaml in {run_dir}")

    return eval_pairs


def extract_mean_score(results: dict, result_path: Path) -> float:
    """
    Supports:
      - old format: test/mean_score
      - LIBERO format: test/<task_name>_mean_score
    """
    if "test/mean_score" in results:
        return float(results["test/mean_score"])

    mean_score_keys = [
        k for k in results.keys()
        if k.startswith("test/") and k.endswith("_mean_score")
    ]

    if len(mean_score_keys) == 1:
        return float(results[mean_score_keys[0]])

    if len(mean_score_keys) > 1:
        raise ValueError(
            f"Multiple *_mean_score keys found in {result_path}: {mean_score_keys}"
        )

    raise ValueError(
        f"No mean score key found in {result_path}. "
        f"Example keys: {list(results.keys())[:5]}"
    )


def extract_start_seed_and_mean(config_path: Path, result_path: Path) -> Tuple[int, float]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    start_seed = config.get("test_start_seed", None)
    if start_seed is None:
        raise ValueError(f"'test_start_seed' not found in {config_path}")

    with open(result_path, "r") as f:
        results = json.load(f)

    mean_score = extract_mean_score(results, result_path)
    return int(start_seed), float(mean_score)


def collect_matching_rollout_files(
    search_roots: List[str],
    target_seeds: List[int],
    rollout_filename: str = "rollouts_success.hdf5",
    pick_mode: str = "all",
) -> Tuple[List[str], Dict[int, List[Tuple[str, float]]]]:
    """
    Search all roots, find eval runs, match seeds, and collect rollout HDF5 paths.

    Args:
        search_roots: root folders to search under
        target_seeds: seeds to keep
        rollout_filename: 'rollouts_success.hdf5' or 'rollouts_failed.hdf5'
        pick_mode:
            - 'all'      : keep all matching runs for each seed
            - 'best'     : keep only highest mean_score run per seed
            - 'first'    : keep first matching run per seed

    Returns:
        matched_files: flat list of rollout file paths
        seed_to_runs: {seed: [(rollout_path, mean_score), ...]}
    """
    target_seed_set = set(int(s) for s in target_seeds)
    seed_to_runs: Dict[int, List[Tuple[str, float]]] = {s: [] for s in target_seeds}

    for root in search_roots:
        print(f"\n[SEARCH ROOT] {root}")
        eval_pairs = find_eval_files(root)

        for config_path, result_path in eval_pairs:
            run_dir = config_path.parent
            try:
                start_seed, mean_score = extract_start_seed_and_mean(config_path, result_path)
            except Exception as e:
                print(f"[SKIP] Failed to parse {run_dir}: {e}")
                continue

            if start_seed not in target_seed_set:
                continue

            rollout_path = run_dir / rollout_filename
            if not rollout_path.exists():
                print(f"[SKIP] Seed {start_seed} matched, but missing {rollout_filename} in {run_dir}")
                continue

            print(
                f"[MATCH] seed={start_seed} mean_score={mean_score:.6f} "
                f"file={rollout_path}"
            )
            seed_to_runs[start_seed].append((str(rollout_path), mean_score))

    # Apply pick mode
    matched_files: List[str] = []

    for seed in target_seeds:
        runs = seed_to_runs.get(seed, [])
        if not runs:
            continue

        if pick_mode == "all":
            chosen = runs
        elif pick_mode == "best":
            chosen = [max(runs, key=lambda x: x[1])]
        elif pick_mode == "first":
            chosen = [runs[0]]
        else:
            raise ValueError(f"Unknown pick_mode: {pick_mode}")

        for path, score in chosen:
            matched_files.append(path)
            print(f"[USE] seed={seed} mean_score={score:.6f} path={path}")

    return matched_files, seed_to_runs


# =========================
# HDF5 merge utilities
# =========================
def list_demos(fin: h5py.File) -> List[str]:
    if "data" not in fin:
        return []
    demos = []
    for k in fin["data"].keys():
        m = DEMO_RE.match(k)
        if m:
            demos.append(k)
    demos.sort(key=lambda x: int(x.split("_")[1]))
    return demos


def first_demo_name(fin: h5py.File) -> str:
    demos = list_demos(fin)
    if not demos:
        raise RuntimeError("No /data/demo_* groups found in template file.")
    return demos[0]


def copy_attrs(src_obj, dst_obj) -> None:
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v


def read_template_obs_schema(template_path: str) -> List[str]:
    with h5py.File(template_path, "r") as ftmp:
        dn = first_demo_name(ftmp)
        demo = ftmp["data"][dn]
        if "obs" not in demo:
            raise RuntimeError(f"Template file {template_path} demo {dn} missing /obs group.")
        keys = sorted(list(demo["obs"].keys()))
        if not keys:
            raise RuntimeError(f"Template file {template_path} demo {dn} has empty /obs.")
        return keys


def collect_demo_refs(input_paths: List[str]) -> List[Tuple[str, str]]:
    demo_refs: List[Tuple[str, str]] = []
    for in_path in input_paths:
        print(f"[SCAN HDF5] {in_path}")
        with h5py.File(in_path, "r") as fin:
            if "data" not in fin:
                print(f"[WARN] {in_path} has no /data. Skipping.")
                continue
            demos = list_demos(fin)
            print(f"[INFO] Found {len(demos)} demos in {in_path}")
            for demo_name in demos:
                demo_refs.append((in_path, demo_name))
    return demo_refs


def merge_files(
    out_path: str,
    template_path: str,
    input_paths: List[str],
    require_all_top_level: bool = False,
    copy_file_attrs: bool = True,
    copy_data_attrs: bool = True,
    shuffle: bool = False,
    shuffle_seed: int = 0,
) -> int:
    keep_obs_keys = read_template_obs_schema(template_path)
    print(f"[INFO] Using template obs keys from: {template_path}")
    print(f"[INFO] Obs keys ({len(keep_obs_keys)}): {keep_obs_keys}")

    TOP_LEVEL_KEYS = ["actions", "abs_actions", "rewards", "dones", "states"]
    demo_refs = collect_demo_refs(input_paths)

    if len(demo_refs) == 0:
        raise RuntimeError("No demos found in any input files.")

    if shuffle:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(demo_refs)
        print(f"[INFO] Shuffled {len(demo_refs)} demos with seed={shuffle_seed}")
    else:
        print(f"[INFO] Merging {len(demo_refs)} demos without shuffling")

    with h5py.File(out_path, "w") as fout:
        gdata_out = fout.create_group("data")
        next_id = 0

        first_input_for_attrs = None
        for in_path in input_paths:
            with h5py.File(in_path, "r") as fin:
                if "data" in fin:
                    first_input_for_attrs = in_path
                    if copy_file_attrs:
                        copy_attrs(fin, fout)
                    if copy_data_attrs:
                        copy_attrs(fin["data"], gdata_out)
                    break

        if first_input_for_attrs is None:
            raise RuntimeError("Could not find any valid input file with /data group.")

        for in_path, demo_name in demo_refs:
            with h5py.File(in_path, "r") as fin:
                src_demo = fin["data"][demo_name]
                if "obs" not in src_demo:
                    raise RuntimeError(f"{in_path}:{demo_name} missing /obs group.")

                src_obs = src_demo["obs"]
                missing = [k for k in keep_obs_keys if k not in src_obs]
                if missing:
                    raise RuntimeError(
                        f"{in_path}:{demo_name} missing required obs keys: {missing}\n"
                        f"Fix: choose a different template or regenerate this file."
                    )

                dst_demo_name = f"demo_{next_id}"
                dst_demo = gdata_out.create_group(dst_demo_name)
                next_id += 1

                copy_attrs(src_demo, dst_demo)

                for k in TOP_LEVEL_KEYS:
                    if k in src_demo:
                        fin.copy(src_demo[k], dst_demo, name=k)
                    else:
                        if require_all_top_level:
                            raise RuntimeError(f"{in_path}:{demo_name} missing required dataset '{k}'.")

                dst_obs = dst_demo.create_group("obs")
                copy_attrs(src_obs, dst_obs)
                for k in keep_obs_keys:
                    fin.copy(src_obs[k], dst_obs, name=k)

        gdata_out.attrs["num_demos"] = next_id
        gdata_out.attrs["shuffled"] = bool(shuffle)
        gdata_out.attrs["shuffle_seed"] = int(shuffle_seed) if shuffle else -1

        print(f"[INFO] Wrote {next_id} demos into {out_path}")
        return next_id


def write_masks(
    fout: h5py.File,
    num_demos: int,
    seed: int = 0,
    train_frac: float = 0.9,
    perc_list: Tuple[float, ...] = (0.2, 0.5),
) -> None:
    if num_demos <= 0:
        raise ValueError("num_demos must be > 0")

    rng = np.random.default_rng(seed)
    all_idx = np.arange(num_demos, dtype=np.int64)
    rng.shuffle(all_idx)

    def n_round(frac: float, n: int) -> int:
        return int(round(frac * n))

    n_train = n_round(train_frac, num_demos)
    n_train = max(0, min(num_demos, n_train))

    train_idx = np.sort(all_idx[:n_train])
    valid_idx = np.sort(all_idx[n_train:])

    gmask = fout.require_group("mask")

    def _write(name: str, arr: np.ndarray) -> None:
        if name in gmask:
            del gmask[name]
        gmask.create_dataset(name, data=np.asarray(arr, dtype=np.int64))

    _write("train", train_idx)
    _write("valid", valid_idx)

    for p in perc_list:
        if p <= 0 or p > 1:
            raise ValueError(f"Invalid percentage {p}. Must be in (0, 1].")

        n_sub = n_round(p, num_demos)
        n_sub = max(1, min(num_demos, n_sub))

        sub = np.sort(all_idx[:n_sub])

        n_sub_train = n_round(train_frac, n_sub)
        n_sub_train = max(0, min(n_sub, n_sub_train))
        sub_train = np.sort(all_idx[:n_sub_train])
        sub_valid = np.sort(all_idx[n_sub_train:n_sub])

        tag = f"{int(round(p * 100))}_percent"
        _write(tag, sub)
        _write(f"{tag}_train", sub_train)
        _write(f"{tag}_valid", sub_valid)

    gmask.attrs["seed"] = int(seed)
    gmask.attrs["train_frac"] = float(train_frac)
    gmask.attrs["num_demos"] = int(num_demos)


# =========================
# Main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find rollout HDF5 files by test_start_seed and merge them."
    )

    # Search / selection
    ap.add_argument(
        "--search_roots",
        nargs="+",
        required=True,
        help="Root folders to recursively search for eval_config.yaml + eval_results.json",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Target test_start_seed values to collect",
    )
    ap.add_argument(
        "--rollout_type",
        choices=["success", "failed"],
        default="success",
        help="Which rollout file to collect from matching runs",
    )
    ap.add_argument(
        "--pick_mode",
        choices=["all", "best", "first"],
        default="all",
        help="For duplicate runs with same seed: all, best mean_score, or first",
    )

    # Merge
    ap.add_argument("--template", default=None, help="Template HDF5 path. If omitted, first matched file is used.")
    ap.add_argument("--out", required=True, help="Output merged HDF5 path.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    ap.add_argument("--require_all_top_level", action="store_true")

    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--shuffle_seed", type=int, default=0)

    ap.add_argument("--remake_masks", action="store_true")
    ap.add_argument("--mask_seed", type=int, default=0)
    ap.add_argument("--mask_train_frac", type=float, default=0.9)
    ap.add_argument("--mask_perc", type=float, nargs="*", default=[0.2, 0.5])

    ap.add_argument(
        "--write_matched_list",
        default=None,
        help="Optional text file to save all matched HDF5 paths",
    )

    args = ap.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"[ERROR] Output exists: {args.out}. Use --overwrite to replace it.")

    rollout_filename = f"rollouts_{args.rollout_type}.hdf5"

    matched_files, seed_to_runs = collect_matching_rollout_files(
        search_roots=args.search_roots,
        target_seeds=args.seeds,
        rollout_filename=rollout_filename,
        pick_mode=args.pick_mode,
    )

    print("\n========== SUMMARY ==========")
    for seed in args.seeds:
        runs = seed_to_runs.get(seed, [])
        print(f"seed={seed}: {len(runs)} matched run(s)")

    if not matched_files:
        raise RuntimeError("No matched rollout files found for the given seeds.")

    matched_files = list(dict.fromkeys(matched_files))  # de-duplicate while preserving order

    if args.write_matched_list is not None:
        with open(args.write_matched_list, "w") as f:
            for p in matched_files:
                f.write(p + "\n")
        print(f"[INFO] Wrote matched file list to {args.write_matched_list}")

    template_path = args.template if args.template is not None else matched_files[0]
    print(f"[INFO] Template path: {template_path}")

    num_demos = merge_files(
        out_path=args.out,
        template_path=template_path,
        input_paths=matched_files,
        require_all_top_level=args.require_all_top_level,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
    )

    if args.remake_masks:
        with h5py.File(args.out, "a") as fout:
            write_masks(
                fout=fout,
                num_demos=num_demos,
                seed=args.mask_seed,
                train_frac=args.mask_train_frac,
                perc_list=tuple(args.mask_perc),
            )
        print("[INFO] Wrote /mask datasets into output.")

    print("[DONE]")


if __name__ == "__main__":
    main()