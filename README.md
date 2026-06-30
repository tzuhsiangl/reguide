# ReGuide: From Test-Time Guidance to Self-Improving Diffusion Policies

[Tzu-Hsiang Lin](https://tzuhsiangl.github.io/),
[Srinivas Shakkottai](https://cesg.tamu.edu/faculty/srinivas-shakkottai/),
[Dileep Kalathil](https://engineering.tamu.edu/electrical/profiles/kalathil-dileep.html),
[P. R. Kumar](https://cesg.tamu.edu/faculty/p-r-kumar/)

<p align="center">
  <a href="https://reguide-project.github.io/"><img src="https://img.shields.io/badge/Project-Page-1f6feb"></a>
  <a href="https://arxiv.org/abs/2606.28939"><img src="https://img.shields.io/badge/arXiv-2606.28939-b31b1b?logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/thl1246/reguide-training-data"><img src="https://img.shields.io/badge/Data-HF-FFD21E?logo=huggingface&logoColor=black"></a>
  <a href="https://huggingface.co/thl1246/reguide-checkpoints"><img src="https://img.shields.io/badge/Models-HF-FFD21E?logo=huggingface&logoColor=black"></a>
</p>


## Installation
Install conda environment with
```console
$ conda env create -f conda_environment.yaml
```
Activate conda env with
```console
$ conda activate reguide
```

## Quick Start: Evaluate with Pretrained Models

If you just want to run ReGuide without training anything, download our
pretrained checkpoints and reference data, point the eval config at them, and
run inference.

**1. Download checkpoints and data** from Hugging Face (requires `git-lfs`;
alternatively use `huggingface-cli download`):

```console
# pretrained policy + dynamics-model checkpoints (and PCG guidance targets)
$ git clone https://huggingface.co/thl1246/reguide-checkpoints

# reference / expert demonstration datasets
$ git clone https://huggingface.co/datasets/thl1246/reguide-training-data
```

**2. Set the data path.** Only `demo_dataset_path` needs to be set in
`dyn_model/conf/planner/eval_<task>.yaml`:

```yaml
demo_dataset_path: 'path/to/reguide-training-data/<task>/expert_demos.hdf5'
```

The dynamics model, policy, and PCG guidance targets are set directly in the run
scripts (Step 3) via the `DYN_MODEL`, `POLICY`, and `PCG_DATA_PATH` variables at
the top of the script.

**3. Run inference.** Two modes are available (`<task>` ∈ `can`, `square`,
`transport`, `tool_hang`):

```console
# ReGuide test-time guidance (Phase-Conditioned Guidance)
$ bash pcg/pcg_can.sh            # or pcg_square.sh, pcg_transport.sh, ...

# Base policy without guidance (baseline)
$ bash eval/eval_can.sh          # or eval/eval_<task>.sh
```

Besides the checkpoint paths, each script exposes `NUM_TEST`, `SEEDS`, and
`SAVE_DATA` knobs at the top.

> **Reproducibility / seeds.** The eval scripts include the exact `SEEDS` used in
> the paper. These seeds fix the *evaluation scenarios* (environment initial
> conditions / object placements), not the outcome: the diffusion sampling is
> stochastic and the rollouts are also subject to GPU/CUDA nondeterminism, so
> absolute success rates may vary by a few percent across runs and hardware.
> Report the mean (± std / 95% CI) over the seed set rather than a single number.

To aggregate success rates across runs/seeds, use `extract_success_rate.py`. The
eval/pcg scripts write results to `outputs/inference/<task>/<CATEGORY>/<date>/<time>_seed<seed>/`,
so set `CATEGORY_PATHS` to the **`<CATEGORY>` level** (i.e.
`outputs/inference/<task>/<CATEGORY>`) — the script recurses into every
`<date>/<time>` run beneath it. Set `CATEGORY_NAMES` / `OUTPUT_CSV` at the bottom
of the script too, then run it; it reads each run's `eval_results.json` and
writes a CSV of per-category success rates.

```console
$ python extract_success_rate.py
```

## Train from Scratch (Full Pipeline)

The full ReGuide pipeline runs in six stages. Each stage is driven by a shell
script whose task name and file paths are set in variables at the top of the
file — **edit those (e.g. `TASK`, dataset / checkpoint paths) before running**.
Run each stage directly with `bash script.sh` (single GPU). Supported tasks:
`can`, `square`, `transport`, and `tool_hang`.

```
1. Generate Data            →  generate_data.sh
2. Train Diffusion Policy   →  train/train_<task>.sh
3. Collect Rollouts         →  eval/eval_<task>.sh (ROLLOUT=true, SAVE_DATA=true)
4. Train Dynamics Model     →  train_dyn_model.sh
5. Phase-Conditioned        →  pcg_data_generation/{extract_latent,generate_pcg}.sh
   Guidance (PCG)              pcg/pcg_<task>.sh
6. ReGuide self-improvement →  merge_by_seed.py
   (FT / FS)                   FS: merge_rollout_and_training_data.py + train/train_<task>.sh
                               FT: first_n_data.py + train_reguide-ft/train_<task>_ft.sh
```

#### Step 1 — Generate Data

Convert the raw robomimic demonstrations into image observations (with absolute
actions).

```console
$ bash generate_data.sh
```

#### Step 2 — Train Diffusion Policy

Train the base diffusion policy on the expert demonstrations. This wraps
`train.py` with `image_<task>_diffusion_policy_cnn.yaml`.

```console
$ bash train/train_can.sh          # or train/train_square.sh, train/train_transport.sh, ...
```

Outputs a checkpoint under the configured `hydra.run.dir`. Use it as the `POLICY`
in later stages.

#### Step 3 — Collect Base-Policy Rollouts

Roll out the trained base policy to collect data for dynamics-model training (and
later guidance). In the eval script set `POLICY`, and `ROLLOUT=true`,
`SAVE_DATA=true`.

```console
$ bash eval/eval_can.sh          # or eval/eval_<task>.sh
```

#### Step 4 — Train Dynamics Model

Train the dynamics ("world") model on the rollout data, reusing the base
policy's encoder. This wraps `dyn_model/train.py`.

```console
$ bash train_dyn_model.sh
```

Set `TRAIN_DATASET` / `VAL_DATASET` to the rollouts from Step 3 and `POLICY_CKPT`
to the base policy from Step 2. Produces a dynamics checkpoint (`model_*.pth`)
used as `DYN_MODEL` below.

#### Step 5 — Phase-Conditioned Guidance (PCG)

Encode observations into latents, cluster them into per-phase guidance targets,
then run test-time guided rollouts.

```console
# (a) encode observations into latents  ->  extract_latents.py
$ bash pcg_data_generation/extract_latent.sh

# (b) cluster latents into phase targets  ->  pcg_data_generation/latent_state_clustering.py
$ bash pcg_data_generation/generate_pcg.sh

# (c) run phase-conditioned guided rollouts  ->  eval_test_time_optimization.py
#     set DYN_MODEL + POLICY; set SAVE_DATA=true to save the success rollouts
$ bash pcg/pcg_can.sh        # or pcg_square.sh, pcg_transport.sh, ...
```

#### Step 6 — ReGuide Self-Improvement (FT / FS)

The guided rollouts from Step 5 save successful episodes per seed. First merge
them into one success dataset, then retrain in one of two modes:

```console
# merge the per-seed guided-success rollouts into a single dataset
$ python merge_by_seed.py \
    --search_roots /path/to/pcg/output/root \
    --seeds 5831 143870 4762 1517 2873 ... \
    --rollout_type success --pick_mode best \
    --out /path/to/all_success.hdf5 \
    --overwrite --shuffle --shuffle_seed 42
```

- **ReGuide FS (From-Scratch):** merge the success rollouts with the expert demos
  into one dataset, then train a fresh policy on it (no init checkpoint).
  ```console
  $ python merge_rollout_and_training_data.py \
      --template /path/to/all_success.hdf5 \
      --train    /path/to/expert_demos.hdf5 \
      --rollout_success /path/to/all_success.hdf5 \
      --num_success 30 \
      --out /path/to/merged_success_30.hdf5 \
      --overwrite --remake_masks
  $ bash train/train_can.sh                  # DATASET = merged dataset above
  ```
- **ReGuide FT (Fine-Tune):** extract the first N success rollouts, then continue
  training from the base checkpoint. The FT script takes the expert demos and the
  rollouts separately (`image_<task>_diffusion_policy_cnn_ft.yaml` +
  `training.init_ckpt_path`).
  ```console
  $ python first_n_data.py \
      --template /path/to/all_success.hdf5 \
      --inputs   /path/to/all_success.hdf5 \
      --out /path/to/rollouts_first_30.hdf5 \
      --first_n 30 --overwrite
  $ bash train_reguide-ft/train_can_ft.sh    # set EXPERT_DATASET + ROLLOUT_DATASET inside
  ```

Evaluate any resulting policy (no guidance) with the eval scripts:

```console
$ bash eval/eval_can.sh        # or eval/eval_<task>.sh
```

## Code

 - [LPB (Latent Policy Barrier)](https://github.com/zhanyisun/lpb/tree/main): This codebase is built on top of the LPB codebase.
 - [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/): The base diffusion policy was built on top of the Diffusion Policy codebase.
 - [DINO-WM](https://github.com/gaoyuezhou/dino_wm): Part of the dynamics model training code was adapted from the DINO-WM codebase.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{lin2026reguide,
  title={ReGuide: From Test-Time Guidance to Self-Improving Diffusion Policies},
  author={Lin, Tzu-Hsiang and Shakkottai, Srinivas and Kalathil, Dileep and Kumar, P. R.},
  journal={arXiv preprint arXiv:2606.28939},
  year={2026}
}
```
