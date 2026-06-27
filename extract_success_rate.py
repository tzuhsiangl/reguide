import os
import json
import yaml
import pandas as pd
from pathlib import Path


def find_eval_files(category_path):
    """
    Search through category/date/time/ structure to find
    eval_config.yaml and eval_results.json files.
    """
    eval_pairs = []
    category_path = Path(category_path)

    for date_dir in sorted(category_path.iterdir()):
        if not date_dir.is_dir():
            continue
        for time_dir in sorted(date_dir.iterdir()):
            if not time_dir.is_dir():
                continue

            config_file = time_dir / "eval_config.yaml"
            result_file = time_dir / "eval_results.json"

            if config_file.exists() and result_file.exists():
                eval_pairs.append((config_file, result_file))
                print(f"  [Found] {time_dir}")
            else:
                if not config_file.exists():
                    print(f"  [Missing] eval_config.yaml in {time_dir}")
                if not result_file.exists():
                    print(f"  [Missing] eval_result.json in {time_dir}")

    return eval_pairs


def extract_start_seed_and_mean(config_path, result_path):
    """
    Extract test_start_seed from eval_config.yaml and
    test/mean_score from eval_result.json.

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

    mean_score = results.get("test/mean_score", None)
    if mean_score is None:
        raise ValueError(f"'test/mean_score' not found in {result_path}")

    return start_seed, mean_score


def build_raw_table(category_paths, category_names):
    """
    Build a table where:
        - index  : test_start_seed
        - columns: category names
        - values : test/mean_score

    Args:
        category_paths : list of folder paths (one per category)
        category_names : list of category names

    Returns:
        DataFrame with seed as index
    """
    # {cat_name: {start_seed: mean_score}}
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
                print(f"    start_seed={start_seed}, mean_score={mean_score}")
                cat_data[start_seed] = mean_score
            except Exception as e:
                print(f"  [Error] Could not parse files: {e}")

        all_data[cat_name] = cat_data

    # Collect all unique start seeds across categories
    all_seeds = set()
    for cat_data in all_data.values():
        all_seeds.update(cat_data.keys())
    all_seeds = sorted(all_seeds)

    # Build DataFrame
    rows = []
    for seed in all_seeds:
        row = {"seed": seed}
        for cat_name in category_names:
            row[cat_name] = all_data[cat_name].get(seed, None)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("seed")
    return df


# ======================================================================= #
#  MAIN
# ======================================================================= #
if __name__ == "__main__":

    # ------------------------------------------------------------------ #
    #  Configure your category paths and names here
    # ------------------------------------------------------------------ #

    # CATEGORY_PATHS = [
    #     "/path/to/eval/root/of/category1",
    #     "/path/to/eval/root/of/category2",
    #     "/path/to/eval/root/of/category3",
    # ]
    # CATEGORY_NAMES = [
    #     "category name 1",
    #     "category name 2",
    #     "category name 3",
    # ]

    CATEGORY_PATHS = [
        "path/to/outputs/inference/can/threshold_guidance/15_demos/phase3_k30_p100_step10_scale0.01_30_90_tau10_threshold80_margin0.2_step2",
        "path/to/reguide/lpb/test",

    ]
    CATEGORY_NAMES = [
        "category name 1",
        "category name 2",

    ]




    OUTPUT_CSV = "output_file_name.csv"
   

    

    
    df = build_raw_table(CATEGORY_PATHS, CATEGORY_NAMES)

    df.to_csv(OUTPUT_CSV)
    print(f"\n[Done] Raw results saved to: {OUTPUT_CSV}")
