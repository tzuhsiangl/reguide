#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


DEMO_RE = re.compile(r"^demo_(\d+)$")
"""
python convert_libero_rollout_format.py \
  --input path/to/data/libero/guidance_multitask/LIVING_ROOM_SCENE5/LIVING_ROOM_SCENE5_rollout_success_first_50_old.hdf5 \
  --output path/to/data/libero/guidance_multitask/LIVING_ROOM_SCENE5/LIVING_ROOM_SCENE5_rollout_success_first_50.hdf5 \
  --overwrite

python convert_libero_rollout_format.py \
  --input path/to/data/libero/guidance_multitask/LIVING_ROOM_SCENE6/LIVING_ROOM_SCENE6_rollout_success_first_50_old.hdf5 \
  --output path/to/data/libero/guidance_multitask/LIVING_ROOM_SCENE6/LIVING_ROOM_SCENE6_rollout_success_first_50.hdf5 \
  --overwrite
"""

def list_demos(fin: h5py.File) -> List[str]:
    if "data" not in fin:
        return []
    demos = [k for k in fin["data"].keys() if DEMO_RE.match(k)]
    demos.sort(key=lambda x: int(x.split("_")[1]))
    return demos


def copy_attrs(src_obj, dst_obj) -> None:
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v


def quat_to_rotvec(quat: np.ndarray, quat_order: str = "xyzw") -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dim 4, got shape {q.shape}")

    q_flat = q.reshape(-1, 4)

    if quat_order == "wxyz":
        q_flat = q_flat[:, [1, 2, 3, 0]]
    elif quat_order != "xyzw":
        raise ValueError(f"Unsupported quat_order: {quat_order}")

    rotvec = R.from_quat(q_flat).as_rotvec().astype(np.float32)
    return rotvec.reshape(q.shape[:-1] + (3,))


def convert_obs_dict(
    src_obs: h5py.Group,
    quat_order: str = "xyzw",
) -> Dict[str, np.ndarray]:
    required_src = [
        "agentview_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_joint_pos",
    ]
    missing = [k for k in required_src if k not in src_obs]
    if missing:
        raise RuntimeError(
            f"Missing required rollout obs keys: {missing}. "
            f"Available keys: {sorted(list(src_obs.keys()))}"
        )

    out: Dict[str, np.ndarray] = {}

    out["agentview_rgb"] = src_obs["agentview_image"][:]
    out["ee_pos"] = src_obs["robot0_eef_pos"][:].astype(np.float32)
    out["joint_states"] = src_obs["robot0_joint_pos"][:].astype(np.float32)
    out["ee_ori"] = quat_to_rotvec(src_obs["robot0_eef_quat"][:], quat_order=quat_order)

    return out


def write_obs_group(
    dst_parent: h5py.Group,
    group_name: str,
    obs_data: Dict[str, np.ndarray],
    compress: Optional[str] = "gzip",
    compress_lvl: int = 4,
) -> None:
    g = dst_parent.create_group(group_name)

    for k, arr in obs_data.items():
        is_image = (k.endswith("_rgb") or k.endswith("_image")) and arr.ndim >= 3
        if is_image and compress is not None:
            g.create_dataset(
                k,
                data=arr,
                compression=compress,
                compression_opts=compress_lvl,
            )
        else:
            g.create_dataset(k, data=arr)


def convert_demo(
    src_demo: h5py.Group,
    dst_data: h5py.Group,
    dst_demo_name: str,
    quat_order: str = "xyzw",
    compress: Optional[str] = "gzip",
    compress_lvl: int = 4,
    require_next_obs: bool = False,
) -> None:
    dst_demo = dst_data.create_group(dst_demo_name)
    copy_attrs(src_demo, dst_demo)

    top_level_keys = ["actions", "abs_actions", "rewards", "dones", "states"]
    for k in top_level_keys:
        if k in src_demo:
            src_demo.file.copy(src_demo[k], dst_demo, name=k)

    if "obs" not in src_demo:
        raise RuntimeError(f"{src_demo.name} missing /obs")
    obs_data = convert_obs_dict(src_demo["obs"], quat_order=quat_order)
    write_obs_group(
        dst_parent=dst_demo,
        group_name="obs",
        obs_data=obs_data,
        compress=compress,
        compress_lvl=compress_lvl,
    )

    if "next_obs" in src_demo:
        next_obs_data = convert_obs_dict(src_demo["next_obs"], quat_order=quat_order)
        write_obs_group(
            dst_parent=dst_demo,
            group_name="next_obs",
            obs_data=next_obs_data,
            compress=compress,
            compress_lvl=compress_lvl,
        )
    elif require_next_obs:
        raise RuntimeError(f"{src_demo.name} missing /next_obs")


def convert_file(
    in_path: str,
    out_path: str,
    quat_order: str = "xyzw",
    compress: Optional[str] = "gzip",
    compress_lvl: int = 4,
    overwrite: bool = False,
    require_next_obs: bool = False,
) -> None:
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"Output exists: {out_path}. Use --overwrite to replace it.")

    with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
        copy_attrs(fin, fout)

        if "data" not in fin:
            raise RuntimeError(f"{in_path} has no /data group")

        src_data = fin["data"]
        dst_data = fout.create_group("data")
        copy_attrs(src_data, dst_data)

        demos = list_demos(fin)
        print(f"[INFO] Found {len(demos)} demos in {in_path}")

        for i, demo_name in enumerate(demos):
            print(f"[INFO] Converting {demo_name} -> demo_{i}")
            convert_demo(
                src_demo=src_data[demo_name],
                dst_data=dst_data,
                dst_demo_name=f"demo_{i}",
                quat_order=quat_order,
                compress=compress,
                compress_lvl=compress_lvl,
                require_next_obs=require_next_obs,
            )

        dst_data.attrs["num_demos"] = len(demos)

    print(f"[DONE] Wrote converted file to: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert LIBERO rollout HDF5 from rollout schema to training-file schema."
    )
    ap.add_argument("--input", required=True, help="Input rollout HDF5")
    ap.add_argument("--output", required=True, help="Output converted HDF5")
    ap.add_argument(
        "--quat_order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order in input file",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_compress", action="store_true")
    ap.add_argument("--compress_lvl", type=int, default=4)
    ap.add_argument("--require_next_obs", action="store_true")

    args = ap.parse_args()

    compress = None if args.no_compress else "gzip"

    convert_file(
        in_path=args.input,
        out_path=args.output,
        quat_order=args.quat_order,
        compress=compress,
        compress_lvl=args.compress_lvl,
        overwrite=args.overwrite,
        require_next_obs=args.require_next_obs,
    )


if __name__ == "__main__":
    main()