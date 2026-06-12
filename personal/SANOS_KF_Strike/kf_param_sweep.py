"""
KF hyperparameter sweep — q_scale / ratio / ell / stencil.

Requires a phase1 cache produced by phase1_cache.py (data loading + DLV
fitting done once and serialised to disk).  All KF combinations run in a
multiprocessing pool so the slow phase-1 cost is paid exactly once.

Usage
-----
    python phase1_cache.py --out phase1_cache.pkl          # run once
    python kf_param_sweep.py [--cache phase1_cache.pkl] [--out results/kf_sweep_results.csv] [--workers 4]

Edit PARAM_GRID below to define the search space.
Results are sorted by held-out RMSE (prior RMSE, one-step-ahead).
"""

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import pickle
import sys
import warnings

import numpy as np

project_root = "/Users/elie/Documents/GitHub/quant_data/"
sanos_kf_dir = os.path.join(project_root, "personal", "SANOS_KF_Strike")
for _p in [project_root, sanos_kf_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dlv_kalman_filter import KFConfig, run_dlv_kalman_filter_v2
from dlv_pricing_absolute import option_prices

# ── Parameter grid — edit this ────────────────────────────────────────────────
# ratio   = r_stale / r_diag  (inf → ignore stale nodes; selection-H on fresh only)
# stencil = 1 (permissive) or 3 (conservative butterfly support)
PARAM_GRID = {
    "q_scale": [1e3, 5e3, 1e4, 1e5],
    "ratio": [1, 3, 10, 30, 100, float("inf")],
    "ell": [1.0, 5.0, 10.0],
    "stencil": [1, 3],
}

# Fixed KF parameters (not swept)
FIXED_KF = dict(
    r_diag=1e-15,
    p0_scale=1e-6,
    p0_correlated=False,
    soft_floor=0.0,
    ignore_at_dlv_max=True,
)

# ── Worker globals (set once per process by _init_worker) ─────────────────────
_cache = None
_grid_K = None
_is_call = None
_snap_lookup = None
_kf_N = None
_kf_box = None
_kf_pay = None
_dlv_min = None


def _init_worker(cache_path: str) -> None:
    global _cache, _grid_K, _is_call, _snap_lookup
    global _kf_N, _kf_box, _kf_pay, _dlv_min
    with open(cache_path, "rb") as f:
        _cache = pickle.load(f)
    _grid_K = _cache["grid_K"]
    _is_call = _cache["is_call"]
    _snap_lookup = _cache["snap_lookup"]
    xstrikes_row = _cache["xstrikes_row"]
    tte_geom = _cache["tte_geom"]
    _kf_N = len(_grid_K)
    _kf_box = _grid_K**2 / (xstrikes_row[2:] - xstrikes_row[:-2]) * tte_geom
    _kf_box[0] = 0.0
    _kf_box[-1] = 0.0
    _kf_pay = np.maximum(_grid_K[np.newaxis, :] - _grid_K[:, np.newaxis], 0.0)
    _dlv_min = _cache["phase1_config"]["dlv_min"]


# ── Density reconstruction from KF DLV² ──────────────────────────────────────


def _density_from_kf_dlvsq(dlvsq_net: np.ndarray, fwd: float) -> np.ndarray | None:
    N = _kf_N
    v = dlvsq_net + _dlv_min**2
    A = np.zeros((N, N))
    b = np.zeros(N)
    A[0] = 1.0
    b[0] = 1.0
    A[1] = _grid_K
    b[1] = fwd
    for j in range(1, N - 1):
        A[j + 1] = _kf_pay[j]
        A[j + 1, j] -= v[j - 1] * _kf_box[j]
        b[j + 1] = max(fwd - _grid_K[j], 0.0)
    try:
        d = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    if (-np.minimum(d, 0.0)).sum() > 0.02:
        return None
    d = np.maximum(d, 0.0)
    s = d.sum()
    return d / s if s > 1e-12 else None


# ── Pricing metrics (posterior + prior) ───────────────────────────────────────


def _pricing_metrics(
    results: list, x_hist: np.ndarray, x_pred: np.ndarray
) -> tuple[float, float, float, float, int, int]:
    """
    Return (rmse_held_out, nse_held_out, nse_held_out_med, rmse_in_sample,
            nse_in_sample, nse_in_sample_med, n_eval_prior, n_eval_post).
    held-out = prior (x_pred), in-sample = posterior (x_hist).
    NSE is aggregated across steps both ways: mean (outlier-sensitive) and
    median (robust); the per-step NSE itself stays a mean over the snapshot's
    options.
    n_eval_* counts steps where density reconstruction succeeded — if this
    differs across stencil values, the averages are not comparable.
    """
    T = len(results)
    rmse_post = np.full(T, np.nan)
    nse_post = np.full(T, np.nan)
    rmse_prior = np.full(T, np.nan)
    nse_prior = np.full(T, np.nan)
    is_call_cache: dict = {}

    for k in range(T):
        r = results[k]
        if r["failed"]:
            continue
        snap = _snap_lookup.get(r["ts"])
        if snap is None:
            continue
        strikes_k, bid_k, ask_k = snap
        if len(strikes_k) == 0:
            continue
        mid_k = 0.5 * (bid_k + ask_k)
        hs_k = np.maximum(0.5 * (ask_k - bid_k), 1e-8)
        m = len(strikes_k)
        if m not in is_call_cache:
            is_call_cache[m] = np.full(m, _is_call, dtype=bool)
        is_call_k = is_call_cache[m]
        fwd_k = r["fwd"]

        for vec, rmse_arr, nse_arr in (
            (x_hist[k], rmse_post, nse_post),
            (x_pred[k], rmse_prior, nse_prior),
        ):
            d = _density_from_kf_dlvsq(vec, fwd_k)
            if d is None:
                continue
            px = option_prices(d, _grid_K, strikes_k, is_call_k, fwd_k)
            resid = px - mid_k
            rmse_arr[k] = float(np.sqrt(np.mean(resid**2)))
            nse_arr[k] = float(np.mean(np.abs(resid) / hs_k))

    return (
        float(np.nanmean(rmse_prior)),
        float(np.nanmean(nse_prior)),
        float(np.nanmedian(nse_prior)),
        float(np.nanmean(rmse_post)),
        float(np.nanmean(nse_post)),
        float(np.nanmedian(nse_post)),
        int(np.sum(~np.isnan(rmse_prior))),
        int(np.sum(~np.isnan(rmse_post))),
    )


# ── Worker ────────────────────────────────────────────────────────────────────


def _run_one(task: dict) -> dict:
    results = _cache["results"]
    N_dlv = _cache["N_dlv"]
    dlv_anchor_pure = _cache["dlv_anchor_pure"]

    p = task
    r_diag = FIXED_KF["r_diag"]
    r_stale = r_diag * p["ratio"]  # inf * r_diag = inf ✓

    cfg = KFConfig(
        q_scale=p["q_scale"],
        ell=p["ell"],
        r_diag=r_diag,
        r_stale=r_stale,
        stencil=p["stencil"],
        p0_scale=FIXED_KF["p0_scale"],
        p0_correlated=FIXED_KF["p0_correlated"],
        soft_floor=FIXED_KF["soft_floor"],
        trading_day_s=_cache["trading_day_s"],
        ignore_at_dlv_max=FIXED_KF["ignore_at_dlv_max"],
        dlv_max_sq_net=_cache["dlv_max_sq_net"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_hist, x_pred, *_ = run_dlv_kalman_filter_v2(
            results, N_dlv=N_dlv, grid_K_nodes=dlv_anchor_pure, cfg=cfg
        )

    rmse_ho, nse_ho, nse_ho_med, rmse_is, nse_is, nse_is_med, n_eval_prior, n_eval_post = (
        _pricing_metrics(results, x_hist, x_pred)
    )

    def _label(v):
        return "inf" if v == float("inf") else v

    return {
        "q_scale": p["q_scale"],
        "ratio": _label(p["ratio"]),
        "ell": p["ell"],
        "stencil": p["stencil"],
        "rmse_held_out": rmse_ho,
        "nse_held_out": nse_ho,
        "nse_held_out_med": nse_ho_med,
        "rmse_in_sample": rmse_is,
        "nse_in_sample": nse_is,
        "nse_in_sample_med": nse_is_med,
        "n_eval_prior": n_eval_prior,
        "n_eval_post": n_eval_post,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main(cache_path: str, out_path: str, n_workers: int) -> None:
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    tasks = [dict(zip(keys, c)) for c in combos]
    total = len(tasks)
    sizes = "  ".join(f"{k}×{len(PARAM_GRID[k])}" for k in keys)
    print(f"Sweeping {total} combinations ({sizes}) across {n_workers} workers ...")

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(cache_path,),
    ) as pool:
        records = list(pool.imap_unordered(_run_one, tasks, chunksize=4))

    records.sort(key=lambda r: r["rmse_held_out"])
    for i, r in enumerate(records, 1):
        r["rank"] = i

    fieldnames = [
        "rank",
        "q_scale",
        "ratio",
        "ell",
        "stencil",
        "rmse_held_out",
        "nse_held_out",
        "nse_held_out_med",
        "rmse_in_sample",
        "nse_in_sample",
        "nse_in_sample_med",
        "n_eval_prior",
        "n_eval_post",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {total} rows → {out_path}")
    print("\nTop 10 by held-out RMSE:")
    header = (
        f"{'rank':>4}  {'q_scale':>8}  {'ratio':>6}  {'ell':>6}  {'stencil':>7}"
        f"  {'rmse_held_out':>13}  {'nse_ho_mean':>11}  {'nse_ho_med':>10}  {'n_eval_prior':>12}"
    )
    print(header)
    for r in records[:10]:
        print(
            f"{r['rank']:>4}  {r['q_scale']:>8.0e}  {str(r['ratio']):>6}"
            f"  {r['ell']:>6.1f}  {r['stencil']:>7}"
            f"  {r['rmse_held_out']:>13.5f}  {r['nse_held_out']:>11.3f}  {r['nse_held_out_med']:>10.3f}"
            f"  {r['n_eval_prior']:>12}"
        )
    best = records[0]
    print(
        f"\nBest: q_scale={best['q_scale']:.0e}  ratio={best['ratio']}"
        f"  ell={best['ell']}  stencil={best['stencil']}"
        f"  → RMSE={best['rmse_held_out']:.5f}"
        f"  NSE mean={best['nse_held_out']:.3f} med={best['nse_held_out_med']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="phase1_cache.pkl")
    parser.add_argument("--out", default="results/kf_sweep_results.csv")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()
    main(args.cache, args.out, args.workers)
