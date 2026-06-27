from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple, Optional

import h5py
import numpy as np


DEMO_RE = re.compile(r"^demo_(\d+)$")

"""
Merge expert training demos with successful policy rollouts into a single
robomimic-style HDF5 dataset, optionally writing train/valid masks.

Example:
python merge_rollout_and_training_data.py \
  --template /path/to/rollouts_success.hdf5 \
  --train    /path/to/expert_demos.hdf5 \
  --rollout_success /path/to/rollouts_success.hdf5 \
  --num_success 30 \
  --out /path/to/merged_success_30.hdf5 \
  --overwrite \
  --remake_masks

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


def choose_demo_names(
    fin: h5py.File,
    max_count: Optional[int],
    rng: np.random.Generator,
    shuffle: bool,
) -> List[str]:
    demos = list_demos(fin)

    if max_count is None or max_count < 0:
        selected = demos
    else:
        if max_count > len(demos):
            raise RuntimeError(
                f"Requested {max_count} demos, but file only has {len(demos)} demos."
            )
        if shuffle:
            idx = np.arange(len(demos))
            rng.shuffle(idx)
            selected = [demos[i] for i in idx[:max_count]]
        else:
            selected = demos[:max_count]

    return selected


def copy_one_demo(
    fin: h5py.File,
    src_demo_name: str,
    gdata_out: h5py.Group,
    next_id: int,
    keep_obs_keys: List[str],
    require_all_top_level: bool = False,
) -> int:
    TOP_LEVEL_KEYS = ["actions", "abs_actions", "rewards", "dones", "states"]

    src_demo = fin["data"][src_demo_name]
    if "obs" not in src_demo:
        raise RuntimeError(f"Missing /obs group in {src_demo_name}")

    src_obs = src_demo["obs"]
    missing = [k for k in keep_obs_keys if k not in src_obs]
    if missing:
        raise RuntimeError(
            f"{fin.filename}:{src_demo_name} missing required obs keys: {missing}\n"
            f"Fix: either (1) choose a different template, or (2) regenerate/convert this file."
        )

    dst_demo_name = f"demo_{next_id}"
    dst_demo = gdata_out.create_group(dst_demo_name)

    copy_attrs(src_demo, dst_demo)

    for k in TOP_LEVEL_KEYS:
        if k in src_demo:
            fin.copy(src_demo[k], dst_demo, name=k)
        else:
            if require_all_top_level:
                raise RuntimeError(f"{fin.filename}:{src_demo_name} missing required dataset '{k}'.")

    dst_obs = dst_demo.create_group("obs")
    copy_attrs(src_obs, dst_obs)
    for k in keep_obs_keys:
        fin.copy(src_obs[k], dst_obs, name=k)

    # Intentionally do NOT copy next_obs
    return next_id + 1


def merge_selected_files(
    out_path: str,
    template_path: str,
    train_path: str,
    success_path: Optional[str],
    num_success: int,
    shuffle_rollouts: bool = True,
    selection_seed: int = 0,
    require_all_top_level: bool = False,
    copy_file_attrs: bool = True,
    copy_data_attrs: bool = True,
) -> int:
    keep_obs_keys = read_template_obs_schema(template_path)
    print(f"[INFO] Using template obs keys from: {template_path}")
    print(f"[INFO] Obs keys ({len(keep_obs_keys)}): {keep_obs_keys}")

    rng = np.random.default_rng(selection_seed)

    with h5py.File(out_path, "w") as fout:
        gdata_out = fout.create_group("data")
        next_id = 0
        copied_attrs = False

        def maybe_copy_root_attrs(fin: h5py.File) -> None:
            nonlocal copied_attrs
            if copied_attrs:
                return
            if copy_file_attrs:
                copy_attrs(fin, fout)
            if copy_data_attrs and "data" in fin:
                copy_attrs(fin["data"], gdata_out)
            copied_attrs = True

        # 1) copy all training demos
        print(f"[INFO] Reading training file: {train_path}")
        with h5py.File(train_path, "r") as ftrain:
            if "data" not in ftrain:
                raise RuntimeError(f"{train_path} has no /data group.")
            maybe_copy_root_attrs(ftrain)

            train_demos = list_demos(ftrain)
            print(f"[INFO] Found {len(train_demos)} training demos")
            for demo_name in train_demos:
                next_id = copy_one_demo(
                    fin=ftrain,
                    src_demo_name=demo_name,
                    gdata_out=gdata_out,
                    next_id=next_id,
                    keep_obs_keys=keep_obs_keys,
                    require_all_top_level=require_all_top_level,
                )

        # 2) copy selected rollout success demos
        if success_path is not None and num_success != 0:
            print(f"[INFO] Reading rollout success file: {success_path}")
            with h5py.File(success_path, "r") as fsucc:
                if "data" not in fsucc:
                    raise RuntimeError(f"{success_path} has no /data group.")

                selected_success = choose_demo_names(
                    fin=fsucc,
                    max_count=num_success,
                    rng=rng,
                    shuffle=shuffle_rollouts,
                )
                print(f"[INFO] Selected {len(selected_success)} rollout success demos")

                for demo_name in selected_success:
                    next_id = copy_one_demo(
                        fin=fsucc,
                        src_demo_name=demo_name,
                        gdata_out=gdata_out,
                        next_id=next_id,
                        keep_obs_keys=keep_obs_keys,
                        require_all_top_level=require_all_top_level,
                    )

        gdata_out.attrs["num_demos"] = next_id
        gdata_out.attrs["num_train_demos"] = len(list_demos(h5py.File(train_path, "r")))
        gdata_out.attrs["num_selected_success"] = int(num_success)

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
    ap.add_argument("--train", required=True, help="Training HDF5 file (all demos will be kept).")
    ap.add_argument("--rollout_success", default=None, help="Rollout success HDF5 file.")
    ap.add_argument("--num_success", type=int, default=0, help="How many rollout success demos to merge.")
    ap.add_argument("--out", required=True, help="Output merged HDF5 path.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")

    ap.add_argument(
        "--shuffle_rollouts",
        action="store_true",
        help="Randomly sample success demos instead of taking the first N.",
    )
    ap.add_argument(
        "--selection_seed",
        type=int,
        default=0,
        help="Seed used when sampling rollout success demos.",
    )

    ap.add_argument(
        "--require_all_top_level",
        action="store_true",
        help="If set, require actions/abs_actions/rewards/dones/states exist in every demo.",
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

    if args.num_success < 0:
        raise SystemExit("[ERROR] --num_success must be >= 0")

    num_demos = merge_selected_files(
        out_path=args.out,
        template_path=args.template,
        train_path=args.train,
        success_path=args.rollout_success,
        num_success=args.num_success,
        shuffle_rollouts=args.shuffle_rollouts,
        selection_seed=args.selection_seed,
        require_all_top_level=args.require_all_top_level,
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
