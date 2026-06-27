#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple, Optional

import h5py
import numpy as np


DEMO_RE = re.compile(r"^demo_(\d+)$")
"""
python first_n_data.py \
  --template /path/to/merged/rollout/data \
  --inputs /path/to/training/data \
  --out /output/file/path/and/name \
  --first_n 30 \
  --overwrite
"""

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
    """
    Return a flat list of (input_path, demo_name) across all files.
    """
    demo_refs: List[Tuple[str, str]] = []
    for in_path in input_paths:
        print(f"[INFO] Scanning input: {in_path}")
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
    first_n: Optional[int] = None,
) -> int:
    """
    Returns number of demos written.
    """
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
        print(f"[INFO] Found {len(demo_refs)} demos without shuffling")

    if first_n is not None:
        if first_n <= 0:
            raise RuntimeError(f"--first_n must be > 0, got {first_n}")
        original_count = len(demo_refs)
        demo_refs = demo_refs[:first_n]
        print(f"[INFO] Keeping first {len(demo_refs)} / {original_count} demos")

    with h5py.File(out_path, "w") as fout:
        gdata_out = fout.create_group("data")
        next_id = 0

        # Copy attrs once from the first valid input
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
                        f"Fix: either (1) choose a different template, or (2) regenerate/convert this file."
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

                # Intentionally do NOT copy next_obs

        gdata_out.attrs["num_demos"] = next_id
        gdata_out.attrs["shuffled"] = bool(shuffle)
        gdata_out.attrs["shuffle_seed"] = int(shuffle_seed) if shuffle else -1
        gdata_out.attrs["first_n"] = int(first_n) if first_n is not None else -1

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Template HDF5 defining obs schema.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input HDF5 files to merge.")
    ap.add_argument("--out", required=True, help="Output merged/subset HDF5 path.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    ap.add_argument(
        "--require_all_top_level",
        action="store_true",
        help="If set, require actions/abs_actions/rewards/dones/states exist in every demo.",
    )

    ap.add_argument("--shuffle", action="store_true", help="Shuffle all demos across all input files before writing.")
    ap.add_argument("--shuffle_seed", type=int, default=0, help="Random seed for shuffling.")

    ap.add_argument(
        "--first_n",
        type=int,
        default=None,
        help="Keep only the first N demos after optional shuffling.",
    )

    ap.add_argument("--remake_masks", action="store_true", help="Create /mask datasets in the output.")
    ap.add_argument("--mask_seed", type=int, default=0, help="Seed for mask splitting.")
    ap.add_argument("--mask_train_frac", type=float, default=0.9, help="Train fraction for /mask/train vs /mask/valid.")
    ap.add_argument(
        "--mask_perc",
        type=float,
        nargs="*",
        default=[0.2, 0.5],
        help="Percent subsets to create (e.g., 0.2 0.5 for 20%% and 50%%).",
    )
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"[ERROR] Output exists: {args.out}. Use --overwrite to replace it.")

    num_demos = merge_files(
        out_path=args.out,
        template_path=args.template,
        input_paths=args.inputs,
        require_all_top_level=args.require_all_top_level,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        first_n=args.first_n,
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