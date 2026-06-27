#!/usr/bin/env python3
"""
Cluster robot latent trajectories into macro phases using current-state + temporal-delta features.

GPU version:
  - Keeps sklearn PCA / KMeans on CPU by default.
  - Uses PyTorch GPU for expensive pairwise distance and softmin computations.
  - Falls back to CPU automatically if CUDA is unavailable or --device cpu is used.

Input HDF5 expected structure:
  /data/demo_k/visual_latent   (T, 1, Dv) or (T, Dv)
  /data/demo_k/proprio_latent  (T, Dp)
Optional:
  /data/demo_k/obs_latent_concat  (unused here except for debugging if needed)

Main idea:
  1. Load per-demo visual + proprio latents
  2. Build clustering features using:
       [visual_t, alpha_p * proprio_t, alpha_dv * (visual_t - visual_{t-1}),
        alpha_dp * (proprio_t - proprio_{t-1}), optional alpha_t * t_norm]
  3. Standardize + PCA + KMeans with many clusters
  4. Order clusters by median normalized demo time
  5. Merge ordered clusters into macro phases
  6. Pick multiple prototypes per phase
  7. Save per-phase prototype sets and thresholds

Example:
python cluster_robot_latents_gpu.py \
  --latents_h5 latents.hdf5 \
  --save_targets_h5 targets.hdf5 \
  --task square \
  --cluster_k 20 \
  --n_phases 3 \
  --pca_dim 64 \
  --n_prototypes_per_phase 100 \
  --prototype_pick per_cluster_score \
  --device cuda \
  --distance_chunk 4096 \
  --overwrite
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    import torch
except ImportError:
    torch = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_h5", type=str, required=True)
    p.add_argument("--demo", type=str, default=None)
    p.add_argument("--demos", type=str, nargs="+", default=None)
    p.add_argument("--max_demos", type=int, default=-1)
    p.add_argument("--max_points_per_demo", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--alpha_proprio", type=float, default=1.0)
    p.add_argument("--alpha_dvisual", type=float, default=1.0)
    p.add_argument("--alpha_dproprio", type=float, default=1.0)
    p.add_argument("--append_time", action="store_true")
    p.add_argument("--alpha_time", type=float, default=3.0)

    p.add_argument("--pca_dim", type=int, default=64)
    p.add_argument("--cluster_k", type=int, required=True)
    p.add_argument("--n_init", type=int, default=20)
    p.add_argument("--max_iter", type=int, default=300)
    p.add_argument("--n_phases", type=int, required=True)
    p.add_argument(
        "--phase_merge",
        type=str,
        default="time_bins",
        choices=["time_bins", "equal_clusters"],
    )

    p.add_argument("--n_prototypes_per_phase", type=int, default=40)
    p.add_argument(
        "--prototype_pick",
        type=str,
        default="score",
        choices=[
            "random",
            "score",
            "farthest",
            "per_cluster_score",
            "per_cluster_farthest",
            "per_cluster_random",
        ],
    )
    p.add_argument("--farthest_pool", type=int, default=200)
    p.add_argument("--lambda_time", type=float, default=1.0)

    p.add_argument("--threshold_quantile", type=float, default=0.90)
    p.add_argument(
        "--save_threshold_percentiles",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    )
    p.add_argument(
        "--softmin_tau_percentiles",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    )
    p.add_argument("--softmin_chunk", type=int, default=1024)

    # GPU / distance options
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for pairwise distance and softmin computations. Uses CPU fallback if CUDA unavailable.",
    )
    p.add_argument(
        "--distance_chunk",
        type=int,
        default=4096,
        help="Chunk size for torch.cdist distance computations.",
    )
    p.add_argument(
        "--torch_float64",
        action="store_true",
        help="Use float64 for torch distance computation. Slower but sometimes useful for checking numeric differences.",
    )

    p.add_argument("--save_targets_h5", type=str, required=True)
    p.add_argument("--task", type=str, default="task")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--do_plot", action="store_true")
    p.add_argument("--color_by", type=str, default="phase", choices=["cluster", "time", "phase"])
    p.add_argument("--alpha_all", type=float, default=0.6)
    p.add_argument("--s_all", type=float, default=10.0)
    p.add_argument("--s_target", type=float, default=70.0)
    p.add_argument("--cmap", type=str, default="tab20")
    p.add_argument("--save_plot", type=str, default="targets_plot.png")

    p.add_argument(
        "--diag_tau_percents",
        type=int,
        nargs="+",
        default=[10, 20, 30, 50, 70, 90],
        help="Tau percentiles to compare in cross-phase soft confusion diagnostics.",
    )
    p.add_argument(
        "--diag_threshold_percents",
        type=int,
        nargs="+",
        default=[50, 60, 70, 80, 90],
        help="Threshold percentiles to compare in cross-phase soft confusion diagnostics.",
    )
    p.add_argument(
        "--diag_use_runtime_proto_count",
        action="store_true",
        help="If set, cross-phase diagnostics only use the first n_prototypes_per_phase prototypes per phase.",
    )
    p.add_argument(
        "--skip_cross_phase_diag",
        action="store_true",
        help="Skip cross-phase hard/soft confusion diagnostics to save time.",
    )

    return p.parse_args()


def get_torch_device(args) -> str:
    if args.device == "cpu":
        return "cpu"
    if torch is None:
        print("[warn] PyTorch is not installed. Falling back to CPU NumPy distances.")
        return "numpy"
    if not torch.cuda.is_available():
        print("[warn] CUDA is not available. Falling back to CPU torch distances.")
        return "cpu"
    return "cuda"


def list_demos(f: h5py.File):
    if "data" not in f:
        raise KeyError("Expected top-level group '/data' in the HDF5.")
    return sorted([k for k in f["data"].keys() if k.startswith("demo_")])


def load_demo_visual_proprio(f: h5py.File, demo: str):
    v_path = f"data/{demo}/visual_latent"
    p_path = f"data/{demo}/proprio_latent"
    if v_path not in f:
        raise KeyError(f"Missing dataset '{v_path}'.")
    if p_path not in f:
        raise KeyError(f"Missing dataset '{p_path}'.")

    visual = f[v_path][:].astype(np.float32)
    proprio = f[p_path][:].astype(np.float32)

    if visual.ndim > 2:
        visual = visual.reshape(visual.shape[0], -1)
    if proprio.ndim > 2:
        proprio = proprio.reshape(proprio.shape[0], -1)

    if visual.shape[0] != proprio.shape[0]:
        raise ValueError(
            f"Mismatched lengths in demo {demo}: visual T={visual.shape[0]} vs proprio T={proprio.shape[0]}"
        )
    return visual, proprio


def uniform_subsample_indices(T: int, max_points: int):
    if max_points <= 0 or T <= max_points:
        return np.arange(T, dtype=np.int32)
    return np.linspace(0, T - 1, max_points).astype(np.int32)


def safe_zscore_stats(Z: np.ndarray):
    mu = Z.mean(axis=0).astype(np.float32)
    sigma = Z.std(axis=0).astype(np.float32)
    sigma = np.maximum(sigma, 1e-6).astype(np.float32)
    return mu, sigma


def softmin_from_dmat(D: np.ndarray, tau: float, axis: int = 1):
    tau = float(max(tau, 1e-12))
    X = -D / tau
    m = np.max(X, axis=axis, keepdims=True)
    lse = m + np.log(np.sum(np.exp(X - m), axis=axis, keepdims=True) + 1e-12)
    return (-tau * lse).squeeze(axis)


def build_temporal_features(
    visual: np.ndarray,
    proprio: np.ndarray,
    alpha_proprio: float,
    alpha_dvisual: float,
    alpha_dproprio: float,
    t_norm: Optional[np.ndarray] = None,
    append_time: bool = False,
    alpha_time: float = 3.0,
):
    feats = [visual.astype(np.float32)]

    if proprio.shape[1] > 0 and alpha_proprio != 0.0:
        feats.append((alpha_proprio * proprio).astype(np.float32))

    d_visual = np.zeros_like(visual, dtype=np.float32)
    d_visual[1:] = visual[1:] - visual[:-1]
    if alpha_dvisual != 0.0:
        feats.append((alpha_dvisual * d_visual).astype(np.float32))

    d_proprio = np.zeros_like(proprio, dtype=np.float32)
    if proprio.shape[1] > 0:
        d_proprio[1:] = proprio[1:] - proprio[:-1]
        if alpha_dproprio != 0.0:
            feats.append((alpha_dproprio * d_proprio).astype(np.float32))

    if append_time:
        if t_norm is None:
            raise ValueError("append_time=True but t_norm is None")
        feats.append((alpha_time * t_norm.reshape(-1, 1)).astype(np.float32))

    X = np.concatenate(feats, axis=1).astype(np.float32)
    Z_save = np.concatenate([visual, proprio], axis=1).astype(np.float32)
    return X, Z_save


# ---------------------------------------------------------------------------
# GPU / CPU distance helpers
# ---------------------------------------------------------------------------

def _torch_dtype(args):
    return torch.float64 if args.torch_float64 else torch.float32


def pairwise_l2_min(Z: np.ndarray, G: np.ndarray, device: str, chunk: int, args) -> np.ndarray:
    """Return min_j ||Z_i - G_j||_2 for each row Z_i."""
    if Z.shape[0] == 0 or G.shape[0] == 0:
        return np.empty((Z.shape[0],), dtype=np.float32)

    if device == "numpy" or torch is None:
        outs = []
        for st in range(0, Z.shape[0], chunk):
            ed = min(Z.shape[0], st + chunk)
            D = np.linalg.norm(Z[st:ed, None, :] - G[None, :, :], axis=2)
            outs.append(D.min(axis=1).astype(np.float32))
        return np.concatenate(outs, axis=0)

    dtype = _torch_dtype(args)
    Z_t = torch.as_tensor(Z, dtype=dtype, device=device)
    G_t = torch.as_tensor(G, dtype=dtype, device=device)

    outs = []
    with torch.no_grad():
        for st in range(0, Z_t.shape[0], chunk):
            ed = min(Z_t.shape[0], st + chunk)
            D = torch.cdist(Z_t[st:ed], G_t, p=2)
            outs.append(D.min(dim=1).values.detach().cpu())
    return torch.cat(outs, dim=0).numpy().astype(np.float32)


def pairwise_l2_matrix(Z: np.ndarray, G: np.ndarray, device: str, chunk: int, args) -> np.ndarray:
    """Return full pairwise distance matrix of shape [len(Z), len(G)], chunked over Z."""
    if Z.shape[0] == 0 or G.shape[0] == 0:
        return np.empty((Z.shape[0], G.shape[0]), dtype=np.float32)

    if device == "numpy" or torch is None:
        outs = []
        for st in range(0, Z.shape[0], chunk):
            ed = min(Z.shape[0], st + chunk)
            D = np.linalg.norm(Z[st:ed, None, :] - G[None, :, :], axis=2).astype(np.float32)
            outs.append(D)
        return np.concatenate(outs, axis=0)

    dtype = _torch_dtype(args)
    Z_t = torch.as_tensor(Z, dtype=dtype, device=device)
    G_t = torch.as_tensor(G, dtype=dtype, device=device)

    outs = []
    with torch.no_grad():
        for st in range(0, Z_t.shape[0], chunk):
            ed = min(Z_t.shape[0], st + chunk)
            D = torch.cdist(Z_t[st:ed], G_t, p=2)
            outs.append(D.detach().cpu())
    return torch.cat(outs, dim=0).numpy().astype(np.float32)


def softmin_distances(
    Z: np.ndarray,
    G: np.ndarray,
    taus: Sequence[float],
    device: str,
    chunk: int,
    args,
) -> List[np.ndarray]:
    """Return list where output[t][i] = -tau_t * logsumexp_j(-||Z_i-G_j||/tau_t)."""
    taus = [float(max(t, 1e-12)) for t in taus]
    results = [np.empty((Z.shape[0],), dtype=np.float32) for _ in taus]

    if Z.shape[0] == 0 or G.shape[0] == 0:
        return results

    if device == "numpy" or torch is None:
        for st in range(0, Z.shape[0], chunk):
            ed = min(Z.shape[0], st + chunk)
            D = np.linalg.norm(Z[st:ed, None, :] - G[None, :, :], axis=2).astype(np.float32)
            for ti, tau in enumerate(taus):
                results[ti][st:ed] = softmin_from_dmat(D, tau, axis=1).astype(np.float32)
        return results

    dtype = _torch_dtype(args)
    Z_t = torch.as_tensor(Z, dtype=dtype, device=device)
    G_t = torch.as_tensor(G, dtype=dtype, device=device)
    taus_t = torch.as_tensor(taus, dtype=dtype, device=device)

    with torch.no_grad():
        for st in range(0, Z_t.shape[0], chunk):
            ed = min(Z_t.shape[0], st + chunk)
            D = torch.cdist(Z_t[st:ed], G_t, p=2)
            for ti, tau in enumerate(taus_t):
                val = -tau * torch.logsumexp(-D / tau, dim=1)
                results[ti][st:ed] = val.detach().cpu().numpy().astype(np.float32)

    return results


# ---------------------------------------------------------------------------
# Prototype-picking helpers
# ---------------------------------------------------------------------------

def farthest_point_sampling(X: np.ndarray, m: int, seed: int = 0):
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32)
    if n <= m:
        return np.arange(n, dtype=np.int32)

    rng = np.random.RandomState(seed)
    first = int(rng.randint(0, n))
    sel = [first]
    dmin = np.linalg.norm(X - X[first:first + 1], axis=1)

    for _ in range(1, m):
        i = int(np.argmax(dmin))
        sel.append(i)
        di = np.linalg.norm(X - X[i:i + 1], axis=1)
        dmin = np.minimum(dmin, di)

    return np.array(sel, dtype=np.int32)


def _allocate_per_cluster_quotas(cluster_sizes_in_phase: np.ndarray, total_target: int) -> np.ndarray:
    sizes = np.asarray(cluster_sizes_in_phase, dtype=np.int64)
    n_clusters = len(sizes)
    if n_clusters == 0 or total_target <= 0:
        return np.zeros(n_clusters, dtype=np.int64)

    nonempty = sizes > 0
    n_nonempty = int(nonempty.sum())
    if n_nonempty == 0:
        return np.zeros(n_clusters, dtype=np.int64)

    if n_nonempty >= total_target:
        order = np.argsort(-sizes)
        alloc = np.zeros(n_clusters, dtype=np.int64)
        for i in order[:total_target]:
            alloc[i] = 1
        return alloc

    alloc = np.where(nonempty, 1, 0).astype(np.int64)
    alloc = np.minimum(alloc, sizes)
    remaining = total_target - int(alloc.sum())

    if remaining > 0:
        eligible = np.maximum(sizes - alloc, 0)
        if eligible.sum() > 0:
            shares = eligible / eligible.sum()
            extra = np.floor(shares * remaining).astype(np.int64)
            leftover = remaining - int(extra.sum())
            if leftover > 0:
                fractional = shares * remaining - extra
                order = np.argsort(-fractional)
                for i in order:
                    if leftover <= 0:
                        break
                    if extra[i] + alloc[i] < sizes[i]:
                        extra[i] += 1
                        leftover -= 1
            alloc = np.minimum(alloc + extra, sizes)

    return alloc


def _pick_per_cluster(
    phase_idx: int,
    labels: np.ndarray,
    phase_clusters: list,
    tnorm_of_global: np.ndarray,
    cluster_tmed: np.ndarray,
    Y: np.ndarray,
    Zsave_all_z: np.ndarray,
    P_per: int,
    mode: str,
    lambda_time: float,
    farthest_pool: int,
    seed: int,
) -> np.ndarray:
    clist = phase_clusters[phase_idx]
    if len(clist) == 0:
        return np.array([], dtype=np.int32)

    cluster_global_idx = []
    cluster_sizes_local = []
    for c in clist:
        idx = np.where(labels == int(c))[0]
        cluster_global_idx.append(idx)
        cluster_sizes_local.append(int(idx.size))
    cluster_sizes_local = np.array(cluster_sizes_local, dtype=np.int64)

    quotas = _allocate_per_cluster_quotas(cluster_sizes_local, P_per)

    chosen_all = []
    for ci, (c, idx, q) in enumerate(zip(clist, cluster_global_idx, quotas)):
        q = int(q)
        if q <= 0 or idx.size == 0:
            continue

        if mode == "random":
            sel = np.random.choice(idx, size=min(q, idx.size), replace=False)
            chosen_all.extend(sel.tolist())
            continue

        Y_c = Y[idx]
        muY = Y_c.mean(axis=0, keepdims=True)
        d_pca = np.linalg.norm(Y_c - muY, axis=1)
        t_med_c = float(cluster_tmed[int(c)])
        d_time = np.abs(tnorm_of_global[idx] - t_med_c)
        score = d_pca + float(lambda_time) * d_time
        order = np.argsort(score)
        idx_sorted = idx[order]

        if mode == "score":
            sel = idx_sorted[:min(q, idx_sorted.size)]
            chosen_all.extend(sel.tolist())
        elif mode == "farthest":
            pool = min(int(farthest_pool), idx_sorted.size)
            pool_idx = idx_sorted[:pool].astype(np.int32)
            Z_pool = Zsave_all_z[pool_idx]
            sel_local = farthest_point_sampling(
                Z_pool,
                m=min(q, pool),
                seed=seed + 997 * phase_idx + 31 * ci,
            )
            sel = pool_idx[sel_local]
            chosen_all.extend(sel.tolist())
        else:
            raise ValueError(f"Unknown per-cluster mode: {mode}")

    return np.array(chosen_all, dtype=np.int32)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def compute_cross_phase_hard_confusion(
    phase_point_indices,
    Zsave_all,
    target_latents,
    target_offsets,
    thr_l2,
    thr_idx,
    n_ph,
    device,
    chunk,
    args,
    runtime_proto_count=None,
):
    med_hard = np.full((n_ph, n_ph), np.nan, dtype=np.float32)
    acc_hard = np.full((n_ph, n_ph), np.nan, dtype=np.float32)

    for src in range(n_ph):
        idx_src = phase_point_indices[src]
        if idx_src.size == 0:
            continue
        Z_src = Zsave_all[idx_src].astype(np.float32)

        for tgt in range(n_ph):
            s = int(target_offsets[tgt])
            e = int(target_offsets[tgt + 1])
            if runtime_proto_count is not None:
                e = min(e, s + int(runtime_proto_count))
            if e <= s:
                continue

            G = target_latents[s:e].astype(np.float32)
            dmin = pairwise_l2_min(Z_src, G, device=device, chunk=chunk, args=args)

            med_hard[src, tgt] = float(np.median(dmin))
            thr_tgt = float(thr_l2[tgt, thr_idx])
            acc_hard[src, tgt] = float((dmin <= thr_tgt).mean())

    return med_hard, acc_hard


def compute_cross_phase_soft_confusion(
    phase_point_indices,
    Zsave_all,
    target_latents,
    target_offsets,
    taus_raw_per_phase,
    thr_soft_l2,
    tau_idx,
    thr_idx,
    n_ph,
    device,
    chunk,
    args,
    runtime_proto_count=None,
):
    med_soft = np.full((n_ph, n_ph), np.nan, dtype=np.float32)
    acc_soft = np.full((n_ph, n_ph), np.nan, dtype=np.float32)

    for src in range(n_ph):
        idx_src = phase_point_indices[src]
        if idx_src.size == 0:
            continue
        Z_src = Zsave_all[idx_src].astype(np.float32)

        for tgt in range(n_ph):
            s = int(target_offsets[tgt])
            e = int(target_offsets[tgt + 1])
            if runtime_proto_count is not None:
                e = min(e, s + int(runtime_proto_count))
            if e <= s:
                continue

            G = target_latents[s:e].astype(np.float32)
            tau_tgt = float(taus_raw_per_phase[tgt, tau_idx])
            thr_tgt = float(thr_soft_l2[tgt, tau_idx, thr_idx])

            d_soft = softmin_distances(
                Z_src,
                G,
                [tau_tgt],
                device=device,
                chunk=chunk,
                args=args,
            )[0]

            med_soft[src, tgt] = float(np.median(d_soft))
            acc_soft[src, tgt] = float((d_soft <= thr_tgt).mean())

    return med_soft, acc_soft


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.n_phases <= 0:
        raise ValueError("--n_phases must be > 0")
    if args.cluster_k <= 0:
        raise ValueError("--cluster_k must be > 0")
    if not (0.0 < args.threshold_quantile < 1.0):
        raise ValueError("--threshold_quantile must be in (0,1)")
    if args.distance_chunk <= 0:
        raise ValueError("--distance_chunk must be > 0")
    if args.softmin_chunk <= 0:
        raise ValueError("--softmin_chunk must be > 0")

    device = get_torch_device(args)
    if device == "cuda":
        print(f"[device] using CUDA: {torch.cuda.get_device_name(0)}")
    elif device == "cpu":
        print("[device] using torch CPU for distance computations")
    else:
        print("[device] using NumPy CPU for distance computations")

    # ---- load demos ----
    demo_data = []
    with h5py.File(args.latents_h5, "r") as f:
        all_demos = list_demos(f)
        if args.demo is not None:
            demos = [args.demo]
        elif args.demos is not None:
            demos = list(args.demos)
        else:
            demos = all_demos if args.max_demos <= 0 else all_demos[:args.max_demos]

        for demo in demos:
            visual_full, proprio_full = load_demo_visual_proprio(f, demo)
            T = visual_full.shape[0]
            idx = uniform_subsample_indices(T, args.max_points_per_demo)
            visual = visual_full[idx]
            proprio = proprio_full[idx]
            t_sub = idx.astype(np.int32)
            denom = max(1, T - 1)
            t_norm_sub = t_sub.astype(np.float32) / float(denom)
            X_sub, Z_save = build_temporal_features(
                visual=visual,
                proprio=proprio,
                alpha_proprio=args.alpha_proprio,
                alpha_dvisual=args.alpha_dvisual,
                alpha_dproprio=args.alpha_dproprio,
                t_norm=t_norm_sub,
                append_time=args.append_time,
                alpha_time=args.alpha_time,
            )
            demo_data.append((demo, X_sub, t_sub, t_norm_sub, Z_save))

    if len(demo_data) == 0:
        raise RuntimeError("No demos loaded.")

    X_all = np.concatenate([x for (_d, x, _t, _tn, _zs) in demo_data], axis=0)
    Zsave_all = np.concatenate([zs for (_d, _x, _t, _tn, zs) in demo_data], axis=0)
    N_total = X_all.shape[0]
    D_feat = X_all.shape[1]
    D_save = Zsave_all.shape[1]
    demo_names = [d for (d, *_rest) in demo_data]

    demo_of_global = np.empty(N_total, dtype=np.int32)
    t_of_global = np.empty(N_total, dtype=np.int32)
    tnorm_of_global = np.empty(N_total, dtype=np.float32)
    off = 0
    for di, (_demo, X_sub, t_sub, t_norm_sub, _Zsave) in enumerate(demo_data):
        n = X_sub.shape[0]
        demo_of_global[off:off + n] = di
        t_of_global[off:off + n] = t_sub
        tnorm_of_global[off:off + n] = t_norm_sub
        off += n

    mu_z, std_z = safe_zscore_stats(Zsave_all)

    print("=== Loaded ===")
    print(f"demos={len(demo_data)} N_total={N_total} D_feat={D_feat} D_save={D_save}")
    print(f"cluster_k={args.cluster_k} n_phases={args.n_phases} phase_merge={args.phase_merge}")
    print(f"prototypes/phase={args.n_prototypes_per_phase} pick={args.prototype_pick}")

    # ---- standardize -> PCA -> KMeans ----
    # This part remains sklearn CPU. Use RAPIDS cuML separately if you want GPU PCA/KMeans.
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X_all)

    pca_dim = min(int(args.pca_dim), X_std.shape[0], X_std.shape[1])
    if pca_dim != int(args.pca_dim):
        print(f"[warn] requested pca_dim={args.pca_dim}, using pca_dim={pca_dim}")

    pca = PCA(n_components=pca_dim, random_state=args.seed)
    Y = pca.fit_transform(X_std).astype(np.float32)
    evr = pca.explained_variance_ratio_.astype(np.float32)

    print("=== PCA ===")
    print(f"pca_dim={pca_dim} explained_sum={evr.sum() * 100:.2f}%")

    kmeans = KMeans(
        n_clusters=args.cluster_k,
        n_init=args.n_init,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    labels = kmeans.fit_predict(Y).astype(np.int32)

    if N_total > args.cluster_k:
        sil = silhouette_score(Y, labels, sample_size=min(5000, N_total), random_state=args.seed)
        print(f"silhouette_score: {sil:.4f}")
    else:
        print("silhouette_score: skipped because N_total <= cluster_k")

    # ---- cluster median time ----
    cluster_tmed = np.full((args.cluster_k,), np.nan, dtype=np.float32)
    cluster_sizes = np.zeros((args.cluster_k,), dtype=np.int32)
    cluster_nonempty = []

    for c in range(args.cluster_k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        cluster_nonempty.append(c)
        cluster_sizes[c] = int(idx.size)
        cluster_tmed[c] = float(np.median(tnorm_of_global[idx]))

    cluster_nonempty = np.array(cluster_nonempty, dtype=np.int32)
    if cluster_nonempty.size < args.n_phases:
        raise RuntimeError(
            f"Too few non-empty clusters ({cluster_nonempty.size}) for n_phases={args.n_phases}."
        )

    ordered_clusters = cluster_nonempty[np.argsort(cluster_tmed[cluster_nonempty])]
    ordered_tmed = cluster_tmed[ordered_clusters]

    # ---- merge clusters into phases ----
    n_ph = args.n_phases
    phase_clusters = [[] for _ in range(n_ph)]

    if args.phase_merge == "equal_clusters":
        chunks = np.array_split(ordered_clusters, n_ph)
        for j in range(n_ph):
            phase_clusters[j] = [int(x) for x in chunks[j].tolist()]
    else:
        edges = np.linspace(0.0, 1.0, n_ph + 1)
        for c in ordered_clusters.tolist():
            t = float(cluster_tmed[int(c)])
            j = int(np.searchsorted(edges, t, side="right") - 1)
            j = max(0, min(n_ph - 1, j))
            phase_clusters[j].append(int(c))

        empty = [j for j in range(n_ph) if len(phase_clusters[j]) == 0]
        if len(empty) > 0:
            print(f"[warn] empty phases {empty} under time_bins; falling back to equal_clusters.")
            chunks = np.array_split(ordered_clusters, n_ph)
            for j in range(n_ph):
                phase_clusters[j] = [int(x) for x in chunks[j].tolist()]

    phase_num_clusters = np.array([len(x) for x in phase_clusters], dtype=np.int32)

    phase_point_indices = []
    phase_sizes = np.zeros((n_ph,), dtype=np.int32)
    phase_tmed = np.zeros((n_ph,), dtype=np.float32)

    for j in range(n_ph):
        clist = phase_clusters[j]
        idx = np.where(np.isin(labels, np.array(clist, dtype=np.int32)))[0]
        phase_point_indices.append(idx.astype(np.int32))
        phase_sizes[j] = int(idx.size)
        phase_tmed[j] = float(np.median(tnorm_of_global[idx])) if idx.size > 0 else np.nan

    print("=== Macro phases ===")
    for j in range(n_ph):
        print(
            f"phase[{j}] clusters={phase_num_clusters[j]:>3} "
            f"points={phase_sizes[j]:>7} t_med={phase_tmed[j]:.3f}"
        )

    # ---- pick prototypes per phase ----
    P_per = int(args.n_prototypes_per_phase)
    if P_per <= 0:
        raise ValueError("--n_prototypes_per_phase must be > 0")

    target_offsets = np.zeros((n_ph + 1,), dtype=np.int32)
    target_global_indices_all = []
    target_cluster_ids_all = []

    Zsave_all_z = ((Zsave_all - mu_z[None, :]) / std_z[None, :]).astype(np.float32)
    proto_cluster_coverage = []

    for j in range(n_ph):
        idx = phase_point_indices[j]
        if idx.size == 0:
            target_offsets[j + 1] = target_offsets[j]
            proto_cluster_coverage.append({})
            continue

        if args.prototype_pick == "random":
            take = min(P_per, idx.size)
            chosen = np.random.choice(idx, size=take, replace=False).astype(np.int32)

        elif args.prototype_pick in ("score", "farthest"):
            Y_phase = Y[idx]
            muY = Y_phase.mean(axis=0, keepdims=True)
            d_pca = np.linalg.norm(Y_phase - muY, axis=1)
            d_time = np.abs(tnorm_of_global[idx] - float(phase_tmed[j]))
            score = d_pca + float(args.lambda_time) * d_time
            order = np.argsort(score)
            idx_sorted = idx[order]

            if args.prototype_pick == "score":
                take = min(P_per, idx_sorted.size)
                chosen = idx_sorted[:take].astype(np.int32)
            else:
                pool = min(int(args.farthest_pool), idx_sorted.size)
                pool_idx = idx_sorted[:pool].astype(np.int32)
                Z_pool = Zsave_all_z[pool_idx]
                sel_local = farthest_point_sampling(
                    Z_pool,
                    m=min(P_per, pool),
                    seed=args.seed + 997 * j,
                )
                chosen = pool_idx[sel_local].astype(np.int32)

        elif args.prototype_pick.startswith("per_cluster_"):
            mode = args.prototype_pick.replace("per_cluster_", "")
            chosen = _pick_per_cluster(
                phase_idx=j,
                labels=labels,
                phase_clusters=phase_clusters,
                tnorm_of_global=tnorm_of_global,
                cluster_tmed=cluster_tmed,
                Y=Y,
                Zsave_all_z=Zsave_all_z,
                P_per=P_per,
                mode=mode,
                lambda_time=args.lambda_time,
                farthest_pool=args.farthest_pool,
                seed=args.seed,
            )
        else:
            raise ValueError(f"Unknown --prototype_pick {args.prototype_pick}")

        target_offsets[j] = len(target_global_indices_all)
        target_global_indices_all.extend(chosen.tolist())
        target_cluster_ids_all.extend(labels[chosen].tolist())
        target_offsets[j + 1] = len(target_global_indices_all)

        cov = {}
        for cid in labels[chosen].tolist():
            cov[int(cid)] = cov.get(int(cid), 0) + 1
        proto_cluster_coverage.append(cov)

    target_global_indices = np.array(target_global_indices_all, dtype=np.int32)
    target_cluster_ids = np.array(target_cluster_ids_all, dtype=np.int32)
    total_proto = int(target_global_indices.size)

    if total_proto == 0:
        raise RuntimeError("No prototypes were selected. Check n_phases, cluster_k, and prototype settings.")

    target_latents = Zsave_all[target_global_indices].astype(np.float32)
    target_latents_z = ((target_latents - mu_z[None, :]) / std_z[None, :]).astype(np.float32)
    target_tnorm = tnorm_of_global[target_global_indices].astype(np.float32)
    target_timesteps = t_of_global[target_global_indices].astype(np.int32)
    target_demo_names = [demo_names[int(demo_of_global[g])] for g in target_global_indices.tolist()]

    print("=== Prototype cluster coverage per phase ===")
    for j in range(n_ph):
        n_clusters_in_phase = phase_num_clusters[j]
        cov = proto_cluster_coverage[j]
        n_clusters_used = len(cov)
        n_protos = sum(cov.values())
        cov_pct = 100.0 * n_clusters_used / n_clusters_in_phase if n_clusters_in_phase > 0 else 0.0
        print(
            f"phase[{j}] protos={n_protos} "
            f"covers {n_clusters_used}/{n_clusters_in_phase} clusters ({cov_pct:.0f}%)"
        )

    # ---- thresholds on distance-to-set ----
    ps = sorted(set(int(x) for x in args.save_threshold_percentiles))
    for p in ps:
        if p <= 0 or p >= 100:
            raise ValueError(f"Invalid percentile {p}; must be in (0,100).")
    ps = np.array(ps, dtype=np.int32)
    qs = ps.astype(np.float32) / 100.0

    thr_l2 = np.full((n_ph, ps.size), np.nan, dtype=np.float32)
    thr_z = np.full((n_ph, ps.size), np.nan, dtype=np.float32)
    dmin_cache_raw = [None for _ in range(n_ph)]
    dmin_cache_z = [None for _ in range(n_ph)]

    print("=== Computing hard thresholds ===")
    for j in range(n_ph):
        idx = phase_point_indices[j]
        s = int(target_offsets[j])
        e = int(target_offsets[j + 1])
        if idx.size == 0 or e <= s:
            continue

        dmin = pairwise_l2_min(
            Zsave_all[idx].astype(np.float32),
            target_latents[s:e].astype(np.float32),
            device=device,
            chunk=args.distance_chunk,
            args=args,
        )
        dminz = pairwise_l2_min(
            Zsave_all_z[idx].astype(np.float32),
            target_latents_z[s:e].astype(np.float32),
            device=device,
            chunk=args.distance_chunk,
            args=args,
        )

        dmin_cache_raw[j] = dmin.astype(np.float32)
        dmin_cache_z[j] = dminz.astype(np.float32)
        thr_l2[j, :] = np.quantile(dmin, qs).astype(np.float32)
        thr_z[j, :] = np.quantile(dminz, qs).astype(np.float32)

    print("=== Thresholds (distance-to-set) saved percentiles ===")
    print("percentiles:", ps.tolist())
    p_list = ps.tolist()
    p50_i = p_list.index(50) if 50 in p_list else None
    p90_i = p_list.index(90) if 90 in p_list else None

    for j in range(n_ph):
        if np.any(np.isfinite(thr_l2[j])):
            msg = f"phase[{j}] protos={target_offsets[j+1] - target_offsets[j]}"
            if p50_i is not None:
                msg += f" raw(p50)={thr_l2[j, p50_i]:.4f} z(p50)={thr_z[j, p50_i]:.4f}"
            if p90_i is not None:
                msg += f" raw(p90)={thr_l2[j, p90_i]:.4f} z(p90)={thr_z[j, p90_i]:.4f}"
            print(msg)

    # ---- softmin taus per phase from dmin percentiles ----
    tau_ps = sorted(set(int(x) for x in args.softmin_tau_percentiles))
    for p in tau_ps:
        if p <= 0 or p >= 100:
            raise ValueError(f"Invalid softmin_tau_percentile {p}; must be in (0,100).")
    tau_ps = np.array(tau_ps, dtype=np.int32)
    tau_qs = tau_ps.astype(np.float32) / 100.0

    taus_raw_per_phase = np.full((n_ph, tau_ps.size), np.nan, dtype=np.float32)
    taus_z_per_phase = np.full((n_ph, tau_ps.size), np.nan, dtype=np.float32)

    for j in range(n_ph):
        if dmin_cache_raw[j] is None:
            continue
        taus_raw_per_phase[j, :] = np.quantile(dmin_cache_raw[j], tau_qs).astype(np.float32)
        taus_z_per_phase[j, :] = np.quantile(dmin_cache_z[j], tau_qs).astype(np.float32)

    # ---- softmin thresholds ----
    T_tau = tau_ps.size
    thr_soft_l2 = np.full((n_ph, T_tau, ps.size), np.nan, dtype=np.float32)
    thr_soft_z = np.full((n_ph, T_tau, ps.size), np.nan, dtype=np.float32)

    print("=== Computing softmin thresholds ===")
    for j in range(n_ph):
        idx = phase_point_indices[j]
        s = int(target_offsets[j])
        e = int(target_offsets[j + 1])
        if idx.size == 0 or e <= s:
            continue

        taus_raw = taus_raw_per_phase[j]
        taus_zj = taus_z_per_phase[j]
        if (not np.all(np.isfinite(taus_raw))) or (not np.all(np.isfinite(taus_zj))):
            continue

        G_raw = target_latents[s:e].astype(np.float32)
        G_z = target_latents_z[s:e].astype(np.float32)

        d_soft_raw_all = softmin_distances(
            Zsave_all[idx].astype(np.float32),
            G_raw,
            taus_raw.tolist(),
            device=device,
            chunk=args.softmin_chunk,
            args=args,
        )
        d_soft_z_all = softmin_distances(
            Zsave_all_z[idx].astype(np.float32),
            G_z,
            taus_zj.tolist(),
            device=device,
            chunk=args.softmin_chunk,
            args=args,
        )

        for ti in range(T_tau):
            thr_soft_l2[j, ti, :] = np.quantile(d_soft_raw_all[ti], qs).astype(np.float32)
            thr_soft_z[j, ti, :] = np.quantile(d_soft_z_all[ti], qs).astype(np.float32)

    print("=== Softmin Thresholds saved ===")
    print("tau_percentiles:", tau_ps.tolist())
    print("threshold_percentiles:", ps.tolist())

    tau_list = tau_ps.tolist()
    tau50_i = tau_list.index(50) if 50 in tau_list else None
    for j in range(n_ph):
        if np.any(np.isfinite(thr_soft_l2[j])):
            msg = f"phase[{j}] protos={target_offsets[j+1] - target_offsets[j]}"
            if tau50_i is not None and p50_i is not None:
                msg += (
                    f" soft_raw(tauP50={taus_raw_per_phase[j, tau50_i]:.4f},p50)="
                    f"{thr_soft_l2[j, tau50_i, p50_i]:.4f}"
                )
            if tau50_i is not None and p90_i is not None:
                msg += (
                    f" soft_raw(tauP50={taus_raw_per_phase[j, tau50_i]:.4f},p90)="
                    f"{thr_soft_l2[j, tau50_i, p90_i]:.4f}"
                )
            print(msg)

    # ---- save phase cluster ids with offsets ----
    phase_cluster_ids_flat = []
    phase_cluster_offsets = np.zeros((n_ph + 1,), dtype=np.int32)
    cur = 0
    for j in range(n_ph):
        phase_cluster_offsets[j] = cur
        phase_cluster_ids_flat.extend(phase_clusters[j])
        cur += len(phase_clusters[j])
    phase_cluster_offsets[n_ph] = cur
    phase_cluster_ids_flat = np.array(phase_cluster_ids_flat, dtype=np.int32)

    print("=== Phase sequence (clusters in each phase, ordered) ===")
    for j in range(n_ph):
        clist = phase_clusters[j]
        if len(clist) == 0:
            print(f"phase[{j}] EMPTY")
            continue
        tlist = cluster_tmed[np.array(clist, dtype=np.int32)]
        tmin = float(np.nanmin(tlist))
        tmax = float(np.nanmax(tlist))
        head = clist[:12]
        tail = "" if len(clist) <= 12 else f" ... (+{len(clist) - 12} more)"
        print(
            f"phase[{j}] "
            f"n_clusters={len(clist):>3} points={int(phase_sizes[j]):>7} "
            f"tmed_phase={float(phase_tmed[j]):.3f} "
            f"cluster_tmed_range=[{tmin:.3f},{tmax:.3f}] "
            f"clusters={head}{tail}"
        )

    # ==========================================================
    # ---- cross-phase confusion diagnostics ----
    # ==========================================================
    runtime_proto_count = P_per if args.diag_use_runtime_proto_count else None

    if args.skip_cross_phase_diag:
        print("=== Cross-phase diagnostics skipped ===")
        hard_diag_thr_percent = 90 if 90 in ps.tolist() else int(ps[-1])
        diag_tau_percents = [x for x in args.diag_tau_percents if x in tau_ps.tolist()]
        diag_threshold_percents = [x for x in args.diag_threshold_percents if x in ps.tolist()]
        med_hard = np.full((n_ph, n_ph), np.nan, dtype=np.float32)
        acc_hard = np.full((n_ph, n_ph), np.nan, dtype=np.float32)
        soft_diag_acc = np.full(
            (len(diag_tau_percents), len(diag_threshold_percents), n_ph, n_ph),
            np.nan,
            dtype=np.float32,
        )
        soft_diag_med = np.full_like(soft_diag_acc, np.nan, dtype=np.float32)
    else:
        print("=== Cross-phase soft/hard distance confusion ===")
        np.set_printoptions(precision=3, suppress=True)

        if runtime_proto_count is not None:
            print(f"[diag] Using only first {runtime_proto_count} prototypes/phase to match runtime behavior.")
        else:
            print("[diag] Using all saved prototypes in each phase for diagnostics.")

        hard_diag_thr_percent = 90 if 90 in ps.tolist() else int(ps[-1])
        hard_thr_idx = ps.tolist().index(hard_diag_thr_percent)

        med_hard, acc_hard = compute_cross_phase_hard_confusion(
            phase_point_indices=phase_point_indices,
            Zsave_all=Zsave_all,
            target_latents=target_latents,
            target_offsets=target_offsets,
            thr_l2=thr_l2,
            thr_idx=hard_thr_idx,
            n_ph=n_ph,
            device=device,
            chunk=args.distance_chunk,
            args=args,
            runtime_proto_count=runtime_proto_count,
        )

        print("--- Hard min median distance matrix (rows=src phase, cols=tgt phase) ---")
        print(med_hard)
        print(f"--- Hard min acceptance matrix using threshold_percent={hard_diag_thr_percent} ---")
        print(acc_hard)

        diag_tau_percents = [x for x in args.diag_tau_percents if x in tau_ps.tolist()]
        diag_threshold_percents = [x for x in args.diag_threshold_percents if x in ps.tolist()]

        if len(diag_tau_percents) == 0:
            raise ValueError(f"No diag_tau_percents found in saved tau percentiles: {tau_ps.tolist()}")
        if len(diag_threshold_percents) == 0:
            raise ValueError(f"No diag_threshold_percents found in saved threshold percentiles: {ps.tolist()}")

        soft_diag_acc = np.full(
            (len(diag_tau_percents), len(diag_threshold_percents), n_ph, n_ph),
            np.nan,
            dtype=np.float32,
        )
        soft_diag_med = np.full_like(soft_diag_acc, np.nan, dtype=np.float32)

        for a, diag_tau_percent in enumerate(diag_tau_percents):
            tau_idx = tau_ps.tolist().index(diag_tau_percent)

            for b, diag_thr_percent in enumerate(diag_threshold_percents):
                thr_idx = ps.tolist().index(diag_thr_percent)

                med_soft, acc_soft = compute_cross_phase_soft_confusion(
                    phase_point_indices=phase_point_indices,
                    Zsave_all=Zsave_all,
                    target_latents=target_latents,
                    target_offsets=target_offsets,
                    taus_raw_per_phase=taus_raw_per_phase,
                    thr_soft_l2=thr_soft_l2,
                    tau_idx=tau_idx,
                    thr_idx=thr_idx,
                    n_ph=n_ph,
                    device=device,
                    chunk=args.softmin_chunk,
                    args=args,
                    runtime_proto_count=runtime_proto_count,
                )

                soft_diag_med[a, b] = med_soft
                soft_diag_acc[a, b] = acc_soft

                print(
                    f"--- Softmin median distance matrix @ tau_percent={diag_tau_percent}, "
                    f"threshold_percent={diag_thr_percent} ---"
                )
                print(med_soft)
                print(
                    f"--- Softmin acceptance matrix @ tau_percent={diag_tau_percent}, "
                    f"threshold_percent={diag_thr_percent} ---"
                )
                print(acc_soft)

                for src in range(n_ph):
                    parts = []
                    for tgt in range(n_ph):
                        if np.isfinite(acc_soft[src, tgt]):
                            parts.append(
                                f"to phase[{tgt}]: med_soft={med_soft[src, tgt]:.4f}, "
                                f"acc_soft={acc_soft[src, tgt]:.3f}"
                            )
                    print(f"src phase[{src}] " + " | ".join(parts))

    # ---- save HDF5 ----
    base = f"targets/{args.task}"
    out_dir = os.path.dirname(os.path.abspath(args.save_targets_h5))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(args.save_targets_h5, "a") as f:
        if base in f:
            if args.overwrite:
                del f[base]
            else:
                raise RuntimeError(f"Group '{base}' exists. Use --overwrite.")

        g = f.create_group(base)
        str_dt = h5py.string_dtype(encoding="utf-8")

        g.create_dataset("phase_cluster_offsets", data=phase_cluster_offsets)
        g.create_dataset("phase_cluster_ids", data=phase_cluster_ids_flat)
        g.create_dataset("phase_num_clusters", data=phase_num_clusters)
        g.create_dataset("phase_sizes", data=phase_sizes)
        g.create_dataset("phase_tmed", data=phase_tmed)

        g.create_dataset("cluster_tmed", data=cluster_tmed)
        g.create_dataset("cluster_sizes", data=cluster_sizes)
        g.create_dataset("ordered_clusters", data=ordered_clusters)
        g.create_dataset("ordered_cluster_tmed", data=ordered_tmed)

        g.create_dataset("target_offsets", data=target_offsets)
        g.create_dataset("target_global_indices", data=target_global_indices)
        g.create_dataset("target_cluster_ids", data=target_cluster_ids)
        g.create_dataset("target_tnorm", data=target_tnorm)
        g.create_dataset("target_timesteps", data=target_timesteps)
        g.create_dataset("target_demo_names", data=np.array(target_demo_names, dtype=object), dtype=str_dt)
        g.create_dataset("target_latents", data=target_latents)

        g.create_dataset("threshold_percentiles", data=ps)
        g.create_dataset("thresholds_l2", data=thr_l2)
        g.create_dataset("thresholds_zscore", data=thr_z)

        g.create_dataset("softmin_tau_percentiles", data=tau_ps)
        g.create_dataset("softmin_taus_raw_per_phase", data=taus_raw_per_phase)
        g.create_dataset("softmin_taus_z_per_phase", data=taus_z_per_phase)
        g.create_dataset("thresholds_softmin_l2", data=thr_soft_l2)
        g.create_dataset("thresholds_softmin_zscore", data=thr_soft_z)

        g.create_dataset("latent_mean", data=mu_z)
        g.create_dataset("latent_std", data=std_z)
        g.create_dataset("explained_variance_ratio", data=evr)

        g.create_dataset("cross_phase_median_hard", data=med_hard)
        g.create_dataset("cross_phase_accept_hard", data=acc_hard)

        g.create_dataset("diag_tau_percents", data=np.array(diag_tau_percents, dtype=np.int32))
        g.create_dataset("diag_threshold_percents", data=np.array(diag_threshold_percents, dtype=np.int32))
        g.create_dataset("cross_phase_median_soft_grid", data=soft_diag_med)
        g.create_dataset("cross_phase_accept_soft_grid", data=soft_diag_acc)

        g.attrs["feature_type"] = "visual+proprio+delta_visual+delta_proprio(+time)"
        g.attrs["alpha_proprio"] = float(args.alpha_proprio)
        g.attrs["alpha_dvisual"] = float(args.alpha_dvisual)
        g.attrs["alpha_dproprio"] = float(args.alpha_dproprio)
        g.attrs["append_time"] = bool(args.append_time)
        g.attrs["alpha_time"] = float(args.alpha_time)
        g.attrs["pca_dim"] = int(pca_dim)
        g.attrs["cluster_k"] = int(args.cluster_k)
        g.attrs["n_phases"] = int(args.n_phases)
        g.attrs["phase_merge"] = str(args.phase_merge)
        g.attrs["n_prototypes_per_phase"] = int(args.n_prototypes_per_phase)
        g.attrs["prototype_pick"] = str(args.prototype_pick)
        g.attrs["farthest_pool"] = int(args.farthest_pool)
        g.attrs["lambda_time"] = float(args.lambda_time)
        g.attrs["threshold_quantile"] = float(args.threshold_quantile)
        g.attrs["total_points"] = int(N_total)
        g.attrs["total_prototypes"] = int(total_proto)
        g.attrs["D_feat"] = int(D_feat)
        g.attrs["D_save"] = int(D_save)
        g.attrs["seed"] = int(args.seed)
        g.attrs["softmin_tau_mode"] = "per_phase_from_dmin_percentiles"
        g.attrs["softmin_tau_percentiles"] = str(tau_ps.tolist())
        g.attrs["softmin_chunk"] = int(args.softmin_chunk)
        g.attrs["distance_backend"] = str(device)
        g.attrs["distance_chunk"] = int(args.distance_chunk)
        g.attrs["torch_float64"] = bool(args.torch_float64)

        g.attrs["cross_phase_hard_threshold_percent"] = int(hard_diag_thr_percent)
        g.attrs["diag_use_runtime_proto_count"] = bool(args.diag_use_runtime_proto_count)
        g.attrs["skip_cross_phase_diag"] = bool(args.skip_cross_phase_diag)

    print(f"[saved] {args.save_targets_h5} -> /{base}/")

    # ---- plot ----
    if args.do_plot:
        pca2 = PCA(n_components=2, random_state=args.seed)
        Y2 = pca2.fit_transform(X_std)
        plt.figure()

        if args.color_by == "cluster":
            cval = labels
            sc = plt.scatter(
                Y2[:, 0],
                Y2[:, 1],
                c=cval,
                cmap=args.cmap,
                s=args.s_all,
                alpha=args.alpha_all,
            )
            cb = plt.colorbar(sc)
            cb.set_label("cluster id")
        elif args.color_by == "time":
            cval = tnorm_of_global
            sc = plt.scatter(
                Y2[:, 0],
                Y2[:, 1],
                c=cval,
                s=args.s_all,
                alpha=args.alpha_all,
            )
            cb = plt.colorbar(sc)
            cb.set_label("normalized time")
        else:
            phase_id = np.full((N_total,), -1, dtype=np.int32)
            for j in range(n_ph):
                phase_id[phase_point_indices[j]] = j
            sc = plt.scatter(
                Y2[:, 0],
                Y2[:, 1],
                c=phase_id,
                cmap="tab10",
                s=args.s_all,
                alpha=args.alpha_all,
            )
            cb = plt.colorbar(sc)
            cb.set_label("macro phase id")

        proto2 = Y2[target_global_indices]
        for j in range(n_ph):
            s = int(target_offsets[j])
            e = int(target_offsets[j + 1])
            if e <= s:
                continue
            cxy = proto2[s:e].mean(axis=0)
            plt.scatter(
                cxy[0],
                cxy[1],
                s=180,
                marker="o",
                edgecolors="black",
                linewidths=0.8,
                alpha=1.0,
            )
            plt.text(
                cxy[0],
                cxy[1],
                f"P{j}",
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
            )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA2 | cluster_k={args.cluster_k} phases={args.n_phases} protos={total_proto}")
        plt.tight_layout()

        if args.save_plot:
            plot_dir = os.path.dirname(os.path.abspath(args.save_plot))
            if plot_dir:
                os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(args.save_plot, dpi=200)
            print(f"[saved plot] {args.save_plot}")
        plt.close()


if __name__ == "__main__":
    main()
