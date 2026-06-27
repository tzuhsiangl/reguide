#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd
import yaml


def find_eval_files(category_path):
    """
    Recursively search under category_path for run folders that contain
    both eval_config.yaml and eval_results.json (or eval_result.json).

    Works for both:
      - baseline: category/date/time/
      - guidance: category/method/date/time/
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
                print(f"  [Found] {run_dir}")
        else:
            print(f"  [Missing] eval_config.yaml in {run_dir}")

    return eval_pairs


def extract_mean_score(results, result_path):
    """
    Extract mean score from eval_results.json.

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


def extract_start_seed_and_mean(config_path, result_path):
    """
    Extract test_start_seed from eval_config.yaml and mean score
    from eval_results.json.

    Returns:
        start_seed : int
        mean_score : float
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    start_seed = config.get("test_start_seed", None)
    if start_seed is None:
        raise ValueError(f"'test_start_seed' not found in {config_path}")

    with open(result_path, "r") as f:
        results = json.load(f)

    mean_score = extract_mean_score(results, result_path)

    return int(start_seed), float(mean_score)


def build_raw_table(category_paths, category_names):
    """
    Build a table where:
        - index  : test_start_seed
        - columns: category names
        - values : mean_score
    """
    all_data = {}

    for cat_path, cat_name in zip(category_paths, category_names):
        print(f"\n[Category] {cat_name}  →  {cat_path}")
        cat_path = Path(cat_path)

        if not cat_path.exists():
            print(f"  [Error] Path does not exist: {cat_path}")
            all_data[cat_name] = {}
            continue

        eval_pairs = find_eval_files(cat_path)

        if not eval_pairs:
            print(f"  [Warning] No eval files found under {cat_path}")
            all_data[cat_name] = {}
            continue

        cat_data = {}
        for config_path, result_path in eval_pairs:
            try:
                start_seed, mean_score = extract_start_seed_and_mean(
                    config_path, result_path
                )

                if start_seed in cat_data:
                    print(
                        f"  [Warning] Duplicate seed {start_seed} in {cat_name}, "
                        f"overwriting old value {cat_data[start_seed]} with {mean_score}"
                    )

                print(f"    start_seed={start_seed}, mean_score={mean_score}")
                cat_data[start_seed] = mean_score
            except Exception as e:
                print(f"  [Error] Could not parse files: {e}")

        all_data[cat_name] = cat_data

    all_seeds = set()
    for cat_data in all_data.values():
        all_seeds.update(cat_data.keys())
    all_seeds = sorted(all_seeds)

    if not all_seeds:
        raise RuntimeError(
            "No valid evaluation rows were parsed. "
            "Check file discovery and metric key extraction."
        )

    rows = []
    for seed in all_seeds:
        row = {"seed": seed}
        for cat_name in category_names:
            row[cat_name] = all_data[cat_name].get(seed, None)
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty or "seed" not in df.columns:
        raise RuntimeError(
            f"Failed to build DataFrame correctly. Columns found: {list(df.columns)}"
        )

    df = df.set_index("seed").sort_index()
    return df


if __name__ == "__main__":

    # CATEGORY_PATHS = [
    #     "path/to/outputs/inference/libero/Baseline/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo",
    #     "path/to/outputs/inference/libero/Baseline/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo",
    #     "path/to/outputs/inference/libero/Baseline/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo",
    #     "path/to/outputs/inference/libero/Baseline/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo",
    #     "path/to/outputs/inference/libero/Baseline/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo",
    #     # "path/to/outputs/inference/libero/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo",
    #     # "path/to/outputs/inference/libero/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo",
    #     # "path/to/outputs/inference/libero/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo",
    #     # "path/to/outputs/inference/libero/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo",
    #     # "path/to/outputs/inference/libero/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo",
    #     # "path/to/outputs/inference/libero/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo",
    #     # "path/to/outputs/inference/libero/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo",
    #     # "path/to/outputs/inference/libero/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo",
    #     # "path/to/outputs/inference/libero/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo",
    #     # "path/to/outputs/inference/libero/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo",
    # ]

    # CATEGORY_NAMES = [
    #     "KITCHEN_SCENE3",
    #     "KITCHEN_SCENE4",
    #     "KITCHEN_SCENE6",
    #     "KITCHEN_SCENE8",
    #     "LIVING_ROOM_SCENE1",
    #     "LIVING_ROOM_SCENE2_1",
    #     "LIVING_ROOM_SCENE2_2",
    #     "LIVING_ROOM_SCENE5",
    #     "LIVING_ROOM_SCENE6",
    #     "STUDY_SCENE1",
    #     # "KITCHEN_SCENE3_guidance",
    #     # "KITCHEN_SCENE4_guidance",
    #     # "KITCHEN_SCENE6_guidance",
    #     # "KITCHEN_SCENE8_guidance",
    #     # "LIVING_ROOM_SCENE1_guidance",
    #     # "LIVING_ROOM_SCENE2_1_guidance",
    #     # "LIVING_ROOM_SCENE2_2_guidance",
    #     # "LIVING_ROOM_SCENE5_guidance",
    #     # "LIVING_ROOM_SCENE6_guidance",
    #     # "STUDY_SCENE1_guidance",
    # ]
    # CATEGORY_PATHS = [
    #     "path/to/train_DAgger/lpb/outputs/inference/libero/Baseline/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo",
    #     "path/to/train_DAgger/lpb/outputs/inference/libero/Baseline/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo",
    #     "path/to/train_DAgger/lpb/outputs/inference/libero/Baseline/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo",
    #     "path/to/train_DAgger/lpb/outputs/inference/libero/Baseline/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo",
    #     "path/to/train_DAgger/lpb/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo",
    #     "path/to/train_DAgger/lpb/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo",
    #     "path/to/train_DAgger/lpb/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo",
    #     "path/to/train_DAgger/lpb/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo",
    #     "path/to/train_DAgger/lpb/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo",
    #     "path/to/train_DAgger/lpb/outputs/inference/libero/Baseline/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo",
        
    # ]

    # CATEGORY_NAMES = [
    #     "KITCHEN_SCENE3",
    #     "KITCHEN_SCENE4",
    #     "KITCHEN_SCENE6",
    #     "KITCHEN_SCENE8",
    #     "LIVING_ROOM_SCENE1",
    #     "LIVING_ROOM_SCENE2_1",
    #     "LIVING_ROOM_SCENE2_2",
    #     "LIVING_ROOM_SCENE5",
    #     "LIVING_ROOM_SCENE6",
    #     "STUDY_SCENE1",
  
    # ]

    # CATEGORY_PATHS = [
    #     "path/to/outputs/inference/libero/Baseline/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/Baseline/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/Baseline/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/Baseline/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate/one_phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/Baseline/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
        
    # ]

    # CATEGORY_NAMES = [
    #     "KITCHEN_SCENE3",
    #     "KITCHEN_SCENE3_guidance_0.005",
    #     "KITCHEN_SCENE3_guidance_0.01",
    #     "KITCHEN_SCENE4",
    #     "KITCHEN_SCENE4_guidance_0.005",
    #     "KITCHEN_SCENE4_guidance_0.01",
    #     "KITCHEN_SCENE6",
    #     "KITCHEN_SCENE6_guidance_0.005",
    #     "KITCHEN_SCENE6_guidance_0.01",
    #     "KITCHEN_SCENE8",
    #     "KITCHEN_SCENE8_guidance_0.005",
    #     "KITCHEN_SCENE8_guidance_0.01",
    #     "LIVING_ROOM_SCENE1",
    #     "LIVING_ROOM_SCENE1_guidance_0.005",
    #     "LIVING_ROOM_SCENE1_guidance_0.01",
    #     "LIVING_ROOM_SCENE2_1",
    #     "LIVING_ROOM_SCENE2-1_guidance_0.005",
    #     "LIVING_ROOM_SCENE2-1_guidance_0.01",
    #     "LIVING_ROOM_SCENE2_2",
    #     "LIVING_ROOM_SCENE2-2_guidance_0.005",
    #     "LIVING_ROOM_SCENE2-2_guidance_0.01",
    #     "LIVING_ROOM_SCENE5",
    #     "LIVING_ROOM_SCENE5_guidance_0.005",
    #     "LIVING_ROOM_SCENE5_guidance_0.01",
    #     "LIVING_ROOM_SCENE6",
    #     "LIVING_ROOM_SCENE6_guidance_0.005",
    #     "LIVING_ROOM_SCENE6_guidance_0.01",
    #     "LIVING_ROOM_SCENE6_single_guidance",
    #     "STUDY_SCENE1",
    #     "STUDY_SCENE1_guidance_0.005",
    #     "STUDY_SCENE1_guidance_0.01",
        
    # ]

    # CATEGORY_PATHS = [
    #     # "path/to/outputs/inference/libero/Baseline/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo",
    #     # "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/phase3_k30_step20_scale0.01_50_90_tau70_threshold70_margin0.1_step2",
    #     # "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/phase3_k30_step20_scale0.01_70_90_tau70_threshold70_margin0.1_step2",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate/phase4_k40_step20_scale0.005_50_null_tau70_threshold70_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate/phase4_k40_step20_scale0.005_50_null_tau70_threshold80_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate/phase4_k40_step20_scale0.05_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    # ]

    # CATEGORY_NAMES = [
    #     # "KITCHEN_SCENE3",
    #     # "phase3_k30_step20_scale0.01_50_90_tau70_threshold70_margin0.1_step2",
    #     # "phase3_k30_step20_scale0.01_70_90_tau70_threshold70_margin0.1_step2",
    #     "LIVING_ROOM_SCENE6_baseline",
    #     "phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "phase4_k40_step20_scale0.005_50_null_tau70_threshold70_margin0.1_step2",
    #     "phase4_k40_step20_scale0.005_50_null_tau70_threshold80_margin0.1_step2",
    #     "phase4_k40_step20_scale0.05_50_null_tau70_threshold90_margin0.1_step2",
    #     "phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    # ]

    # CATEGORY_PATHS = [
    #     "path/to/outputs/inference/libero/Baseline/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/one_phase3_k30_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    # ]

    # CATEGORY_NAMES = [
    #     "KITCHEN_SCENE3",
    #     "phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "one_phase3_k30_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
       
    # ]

    # CATEGORY_PATHS = [
    #     "path/to/outputs/inference/libero/Baseline/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/phase4_k40_step20_scale0.001_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/phase4_k40_step20_scale0.05_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate/phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate/phase4_k40_step20_scale0.01_50_null_tau70_threshold90_margin0.1_step2",
    #     "path/to/outputs/inference/libero/threshold_guidance/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate/one_phase4_k40_step20_scale0.005_50_null_tau70_threshold90_margin0.1_step2",
        
    # ]

    # CATEGORY_NAMES = [
    #     "KITCHEN_SCENE8",
    #     "KITCHEN_SCENE8_guidance_0.005",
    #     "KITCHEN_SCENE8_guidance_0.01",
    #     "KITCHEN_SCENE8_guidance_0.001",
    #     "KITCHEN_SCENE8_guidance_0.05",
    #     "LIVING_ROOM_SCENE5",
    #     "LIVING_ROOM_SCENE5_guidance_0.005",
    #     "LIVING_ROOM_SCENE5_guidance_0.01",
    #     "LIVING_ROOM_SCENE6",
    #     "LIVING_ROOM_SCENE6_guidance_0.005",
    #     "LIVING_ROOM_SCENE6_guidance_0.01",
    #     "LIVING_ROOM_SCENE6_single_guidance",        
    # ]

    CATEGORY_PATHS = [
        "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo",
        "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo_rollouts_success_13_share_80",
        "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo_rollouts_success_25_share_80",
        "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo_rollouts_success_38_share_80",
        "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo_rollouts_success_50_share_80",
        "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo",
        "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo_rollouts_success_13_share_80",
        "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo_rollouts_success_25_share_80",
        "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo_rollouts_success_38_share_80",
        "path/to/outputs/inference/libero/Baseline/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo_rollouts_success_50_share_80",
    ]

    CATEGORY_NAMES = [
        "LIVING_ROOM_SCENE5",
        "LIVING_ROOM_SCENE5_rollouts_success_13_share_80",
        "LIVING_ROOM_SCENE5_rollouts_success_25_share_80",
        "LIVING_ROOM_SCENE5_rollouts_success_38_share_80",
        "LIVING_ROOM_SCENE5_rollouts_success_50_share_80",
        "LIVING_ROOM_SCENE6",
        "LIVING_ROOM_SCENE6_rollouts_success_13_share_80",
        "LIVING_ROOM_SCENE6_rollouts_success_25_share_80",
        "LIVING_ROOM_SCENE6_rollouts_success_38_share_80",
        "LIVING_ROOM_SCENE6_rollouts_success_50_share_80",
    ]
    

    # OUTPUT_CSV = "libero_base_policy_guidance.csv"
    # OUTPUT_CSV = "libero_lpb.csv"
    OUTPUT_CSV = "test.csv"



    df = build_raw_table(CATEGORY_PATHS, CATEGORY_NAMES)
    df.to_csv(OUTPUT_CSV)

    print(f"\n[Done] Raw results saved to: {OUTPUT_CSV}")
    # print("\n--- Preview ---")
    # print(df.to_string())