#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np


"""
Examples:

# Cap each input at 25 demos and stamp explicit language goals per input.
python merge_libero_data.py \
  --template path/to/data/libero/rollout/KITCHEN_SCENE3_rollouts_base_policy_success.hdf5 \
  --inputs \
      path/to/data/libero/rollout/KITCHEN_SCENE3_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE4_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE6_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE8_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE1_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE2_cheese_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE2_soup_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE5_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE6_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/STUDY_SCENE1_rollouts_base_policy_success.hdf5 \
  --input_language_goals \
      "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" \
      "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" \
      "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it" \
      "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" \
      "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket" \
      "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket" \
      "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" \
      "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate" \
      "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate" \
      "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" \
  --out path/to/data/libero/augmentation/libero_base_policy_evenly_250.hdf5 \
  --demos_per_file 25 \
  --overwrite

python merge_libero_data.py \
  --template path/to/data/libero/rollout/KITCHEN_SCENE3_rollouts_success.hdf5 \
  --inputs \
      path/to/data/libero/rollout/KITCHEN_SCENE3_rollouts_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE4_rollouts_success.hdf5\
      path/to/data/libero/rollout/KITCHEN_SCENE6_rollouts_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE8_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE1_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE2_cheese_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE2_soup_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE5_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE6_rollouts_success.hdf5 \
      path/to/data/libero/rollout/STUDY_SCENE1_rollouts_success.hdf5 \
  --input_language_goals \
      "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" \
      "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" \
      "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it" \
      "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" \
      "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket" \
      "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket" \
      "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" \
      "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate" \
      "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate" \
      "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" \
  --out path/to/data/libero/augmentation/libero_rollouts_evenly_250.hdf5 \
  --demos_per_file 25 \
  --overwrite


python merge_libero_data.py \
  --template path/to/data/libero/rollout/KITCHEN_SCENE3_rollouts_base_policy_success.hdf5 \
  --inputs \
      path/to/data/libero/rollout/KITCHEN_SCENE3_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE4_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE6_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE8_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE1_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE2_cheese_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE2_soup_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE5_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE6_rollouts_base_policy_success.hdf5 \
      path/to/data/libero/rollout/STUDY_SCENE1_rollouts_base_policy_success.hdf5 \
  --input_language_goals \
      "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" \
      "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" \
      "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it" \
      "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" \
      "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket" \
      "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket" \
      "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" \
      "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate" \
      "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate" \
      "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" \
  --out path/to/data/libero/augmentation/libero_base_policy_evenly_500.hdf5 \
  --demos_per_file 50 \
  --overwrite

python merge_libero_data.py \
  --template path/to/data/libero/rollout/KITCHEN_SCENE3_rollouts_success.hdf5 \
  --inputs \
      path/to/data/libero/rollout/KITCHEN_SCENE3_rollouts_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE4_rollouts_success.hdf5\
      path/to/data/libero/rollout/KITCHEN_SCENE6_rollouts_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE8_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE1_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE2_cheese_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE2_soup_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE5_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE6_rollouts_success.hdf5 \
      path/to/data/libero/rollout/STUDY_SCENE1_rollouts_success.hdf5 \
  --input_language_goals \
     "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" \
      "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" \
      "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it" \
      "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" \
      "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket" \
      "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket" \
      "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" \
      "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate" \
      "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate" \
      "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" \
  --out path/to/data/libero/augmentation/libero_rollouts_evenly_500.hdf5 \
  --demos_per_file 50 \
  --overwrite


python merge_libero_data.py \
  --template path/to/data/libero/rollout/KITCHEN_SCENE3_rollouts_success.hdf5 \
  --inputs \
      path/to/data/libero/rollout/KITCHEN_SCENE3_rollouts_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE4_rollouts_success.hdf5\
      path/to/data/libero/rollout/KITCHEN_SCENE6_rollouts_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE8_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE1_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE2_cheese_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE2_soup_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE5_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE6_rollouts_success.hdf5 \
      path/to/data/libero/rollout/STUDY_SCENE1_rollouts_success.hdf5 \
  --input_language_goals \
      "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" \
      "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" \
      "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it" \
      "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" \
      "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket" \
      "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket" \
      "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" \
      "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate" \
      "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate" \
      "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" \
  --out path/to/data/libero/augmentation/libero_rollouts_weighted_250.hdf5 \
  --demos_per_file 15 15 25 40 15 25 15 40 45 15 \
  --overwrite


python merge_libero_data.py \
  --template path/to/data/libero/rollout/KITCHEN_SCENE3_rollouts_success.hdf5 \
  --inputs \
      path/to/data/libero/rollout/KITCHEN_SCENE3_rollouts_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE4_rollouts_success.hdf5\
      path/to/data/libero/rollout/KITCHEN_SCENE6_rollouts_success.hdf5 \
      path/to/data/libero/rollout/KITCHEN_SCENE8_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE1_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE2_cheese_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE2_soup_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE5_rollouts_success.hdf5 \
      path/to/data/libero/rollout/LIVING_ROOM_SCENE6_rollouts_success.hdf5 \
      path/to/data/libero/rollout/STUDY_SCENE1_rollouts_success.hdf5 \
  --input_language_goals \
      "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" \
      "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" \
      "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it" \
      "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" \
      "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket" \
      "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket" \
      "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" \
      "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate" \
      "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate" \
      "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" \
  --out path/to/data/libero/augmentation/libero_rollouts_weighted_500.hdf5 \
  --demos_per_file 35 35 50 70 35 50 35 70 85 35 \
  --overwrite




# Same idea but derive goals from filenames stripped of '_demo.hdf5'.
python merge_hdf5.py \
  --template /path/to/STUDY_SCENE1_..._demo.hdf5 \
  --inputs \
      /path/to/STUDY_SCENE1_..._demo.hdf5 \
      /path/to/LIVING_ROOM_SCENE2_..._demo.hdf5 \
      ... \
  --out /path/to/libero_10_rollouts_merged.hdf5 \
  --demos_per_file 25 \
  --language_goals_from_filenames \
  --shuffle --shuffle_seed 42 --overwrite
"""


DEMO_RE = re.compile(r"^demo_(\d+)$")


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


def resolve_per_file_limits(
    input_paths: Sequence[str],
    demos_per_file: Optional[Sequence[int]],
) -> List[Optional[int]]:
    """
    Turn the --demos_per_file CLI value into a per-input-file integer limit
    (or None for "take all").

      None              -> all files: no cap
      [N]               -> all files: cap N
      [N1, N2, ..., Nk] -> must match len(input_paths); per-file caps
    """
    if demos_per_file is None or len(demos_per_file) == 0:
        return [None] * len(input_paths)
    if len(demos_per_file) == 1:
        n = int(demos_per_file[0])
        if n <= 0:
            raise ValueError(f"--demos_per_file must be positive; got {n}")
        return [n] * len(input_paths)
    if len(demos_per_file) != len(input_paths):
        raise ValueError(
            f"--demos_per_file has {len(demos_per_file)} values but there are "
            f"{len(input_paths)} input files. Pass either 1 value (applied to all) "
            f"or {len(input_paths)} values (one per input)."
        )
    out = []
    for n in demos_per_file:
        n = int(n)
        if n <= 0:
            raise ValueError(f"--demos_per_file values must be positive; got {n}")
        out.append(n)
    return out


def resolve_input_language_goals(
    input_paths: Sequence[str],
    input_language_goals: Optional[Sequence[str]],
    language_goals_from_filenames: bool,
) -> List[Optional[str]]:
    """
    Returns one language goal per input file, or None for "don't write the attr."
    Precedence:
      1. Explicit --input_language_goals (must match input count).
      2. --language_goals_from_filenames (derive from each filename, stripping '_demo.hdf5').
      3. None for every input (no attribute written; legacy behavior).
    """
    if input_language_goals is not None:
        if len(input_language_goals) != len(input_paths):
            raise ValueError(
                f"--input_language_goals has {len(input_language_goals)} values "
                f"but there are {len(input_paths)} inputs."
            )
        return [str(g) for g in input_language_goals]

    if language_goals_from_filenames:
        out = []
        for p in input_paths:
            fname = os.path.basename(p)
            if not fname.endswith("_demo.hdf5"):
                raise ValueError(
                    f"--language_goals_from_filenames expects filenames ending "
                    f"in '_demo.hdf5'; got: {fname}"
                )
            out.append(fname[: -len("_demo.hdf5")])
        return out

    return [None] * len(input_paths)


def collect_demo_refs(
    input_paths: List[str],
    per_file_limits: List[Optional[int]],
) -> List[Tuple[str, str]]:
    """
    Return a flat list of (input_path, demo_name) across all files.
    If per_file_limits[i] is set, take only the first N demos from input_paths[i]
    (in demo_0, demo_1, ... order).
    """
    demo_refs: List[Tuple[str, str]] = []
    for in_path, limit in zip(input_paths, per_file_limits):
        print(f"[INFO] Scanning input: {in_path}")
        with h5py.File(in_path, "r") as fin:
            if "data" not in fin:
                print(f"[WARN] {in_path} has no /data. Skipping.")
                continue
            demos = list_demos(fin)
            n_avail = len(demos)
            if limit is not None and n_avail < limit:
                print(
                    f"[WARN] {in_path}: only {n_avail} demos available, "
                    f"requested {limit}. Taking all {n_avail}."
                )
                kept = demos
            elif limit is not None:
                kept = demos[:limit]
            else:
                kept = demos
            print(f"[INFO] Keeping {len(kept)}/{n_avail} demos from {in_path}")
            for demo_name in kept:
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
    demos_per_file: Optional[Sequence[int]] = None,
    input_language_goals: Optional[Sequence[str]] = None,
    language_goals_from_filenames: bool = False,
) -> int:
    """
    Returns number of demos written.
    """
    keep_obs_keys = read_template_obs_schema(template_path)
    print(f"[INFO] Using template obs keys from: {template_path}")
    print(f"[INFO] Obs keys ({len(keep_obs_keys)}): {keep_obs_keys}")

    TOP_LEVEL_KEYS = ["actions", "abs_actions", "rewards", "dones", "states"]

    per_file_limits = resolve_per_file_limits(input_paths, demos_per_file)
    if any(l is not None for l in per_file_limits):
        print(f"[INFO] Per-file demo limits: {per_file_limits}")

    input_goals = resolve_input_language_goals(
        input_paths, input_language_goals, language_goals_from_filenames
    )
    path_to_goal = {p: g for p, g in zip(input_paths, input_goals)}
    if any(g is not None for g in input_goals):
        print("[INFO] Per-input language goals:")
        for p, g in zip(input_paths, input_goals):
            print(f"  {os.path.basename(p):<60} -> {g}")

    demo_refs = collect_demo_refs(input_paths, per_file_limits)

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

                # Stamp per-demo language_goal if a goal is configured for this input file.
                # The dataset reader (libero_replay_image_dataset.py) prefers this attribute
                # when present, falling back to filename inference when absent.
                goal_for_this_input = path_to_goal.get(in_path)
                if goal_for_this_input is not None:
                    dst_demo.attrs["language_goal"] = goal_for_this_input

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
    ap.add_argument("--out", required=True, help="Output merged HDF5 path.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    ap.add_argument(
        "--require_all_top_level",
        action="store_true",
        help="If set, require actions/abs_actions/rewards/dones/states exist in every demo.",
    )

    ap.add_argument("--shuffle", action="store_true", help="Shuffle all demos across all input files before writing.")
    ap.add_argument("--shuffle_seed", type=int, default=0, help="Random seed for shuffling.")

    ap.add_argument(
        "--demos_per_file",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Take only the first N demos from each input file (in demo_0, demo_1, ... order). "
            "Pass one value to apply the same cap to all inputs (e.g., --demos_per_file 30), "
            "or one value per input matching --inputs order (e.g., --demos_per_file 30 50 25). "
            "If a file has fewer demos than requested, all available are taken. "
            "Subsampling happens BEFORE --shuffle."
        ),
    )

    ap.add_argument(
        "--input_language_goals",
        nargs="+",
        default=None,
        help=(
            "One language goal per input file (must match --inputs length, in order). "
            "Each demo copied from that input gets a `language_goal` attribute "
            "set to this value. Use the LIBERO TASK_ID format, e.g., "
            "'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy'. "
            "Mutually exclusive with --language_goals_from_filenames."
        ),
    )
    ap.add_argument(
        "--language_goals_from_filenames",
        action="store_true",
        help=(
            "If set, derive each input's language goal from its filename by "
            "stripping the trailing '_demo.hdf5'. Use this when each input "
            "file is named after its task. Mutually exclusive with "
            "--input_language_goals."
        ),
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

    if args.input_language_goals is not None and args.language_goals_from_filenames:
        raise SystemExit(
            "[ERROR] --input_language_goals and --language_goals_from_filenames "
            "are mutually exclusive. Pick one."
        )

    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"[ERROR] Output exists: {args.out}. Use --overwrite to replace it.")

    num_demos = merge_files(
        out_path=args.out,
        template_path=args.template,
        input_paths=args.inputs,
        require_all_top_level=args.require_all_top_level,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        demos_per_file=args.demos_per_file,
        input_language_goals=args.input_language_goals,
        language_goals_from_filenames=args.language_goals_from_filenames,
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