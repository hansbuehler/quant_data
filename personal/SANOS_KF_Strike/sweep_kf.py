"""
KF hyperparameter sweep over pre-computed Phase 1 results.

Each worker process loads the Phase 1 cache once via Pool initializer,
then processes multiple parameter combinations without re-loading.

Usage:
    python phase1_cache.py --out phase1_cache.pkl   # run once
    python sweep_kf.py --cache phase1_cache.pkl --out sweep_results.csv [--workers 4]

Edit PARAM_GRID below to define the search space.
Results are sorted by held-out RMSE (NIS-passing rows first).
"""

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
from scipy.stats import chi2 as _chi2

project_root = "/Users/elie/Documents/GitHub/quant_data/"
sanos_kf_dir = os.path.join(project_root, "personal", "SANOS_KF_Strike")
for _p in [project_root, sanos_kf_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dlv_kalman_filter import KFConfig, make_fresh_node_mask, run_dlv_kalman_filter_v2
from dlv_pricing_absolute import option_prices

# ── Parameter grid — edit this ────────────────────────────────────────────
PARAM_GRID = {
    "q_scale": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "ell": [1.0],
    "r_diag": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "stencil": [1],
}

# KF params that are fixed across all runs
FIXED_KF = dict(
    p0_scale=1e-3,
    p0_correlated=True,
    soft_floor=0.0,  # DLV² ≥ 0 is a hard physical constraint; prevents negative drift at large q_scale
    ignore_at_dlv_max=True,
)

# ── Worker globals (set once per process in _init_worker) ─────────────────
_cache = None
_grid_K = None  # (N,) all interior anchors
_is_call = None  # bool — True for calls, False for puts
_snap_lookup = None  # dict: ts → (strikes, bids, asks) numpy arrays
_anchor_pure = None  # (N,) anchor strikes (kept for stale-mask logic)
# density reconstruction from KF DLV² state
_kf_N = None  # int — number of interior anchors
_kf_box = None  # (N,) box weights: K²/(K_{j+1,full}-K_{j-1,full})*tte_geom
_kf_pay = None  # (N,N) payoff matrix: max(K_i - K_j, 0)
_dlv_min = None  # float — DLV_MIN from phase1 config
# raw anchor fresh-masks from results (stencil-independent)
_raw_fresh_anchor = None  # list[ndarray|None] length T


def _init_worker(cache_path: str) -> None:
    global _cache, _grid_K, _is_call, _snap_lookup, _anchor_pure
    global _kf_N, _kf_box, _kf_pay, _dlv_min, _raw_fresh_anchor
    with open(cache_path, "rb") as f:
        _cache = pickle.load(f)
    _grid_K = _cache["grid_K"]
    _is_call = _cache["is_call"]
    _snap_lookup = _cache["snap_lookup"]
    _anchor_pure = _cache["anchor_pure"]

    xstrikes_row = _cache["xstrikes_row"]
    tte_geom = _cache["tte_geom"]
    _kf_N = len(_grid_K)
    _kf_box = _grid_K**2 / (xstrikes_row[2:] - xstrikes_row[:-2]) * tte_geom
    _kf_box[0] = 0.0
    _kf_box[-1] = 0.0
    _kf_pay = np.maximum(_grid_K[np.newaxis, :] - _grid_K[:, np.newaxis], 0.0)
    _dlv_min = _cache["phase1_config"]["dlv_min"]

    _raw_fresh_anchor = [r.get("fresh_anchor_mask") for r in _cache["results"]]


# ── Density reconstruction from KF DLV² ───────────────────────────────────


def _density_from_kf_dlvsq(dlvsq_net: np.ndarray, fwd: float) -> np.ndarray | None:
    """
    Reconstruct density from (N-2,) net-DLV² via the N×N exactly-determined system.
    Returns None if the solve fails or >2% of probability mass is negative.
    """
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


# ── KF-prior pricing metric (DLV²-reconstruction path) ────────────────────


def compute_pricing_metrics_kf_prior(
    results_in: list,
    x_pred_hist_arr: np.ndarray,
    stencil: int,
) -> tuple[float, float, float, int]:
    """
    NSE using density reconstructed from x_pred_hist via _density_from_kf_dlvsq.

    Splits steps by whether ALL DLV nodes were fresh at k-1 (i.e. x_pred[k]
    exactly equals z[k-1]) vs at least one non-fresh node.

    Residuals are pooled across all qualifying steps (not averaged per-step then
    averaged again), so varying batch sizes are handled correctly.

    Returns
    -------
    nse_overall      : pooled NSE over all valid steps
    nse_allfresh     : pooled NSE for steps where prev step was fully fresh
    nse_hasnonfresh  : pooled NSE for steps where prev step had ≥1 non-fresh node
    n_recon_fail     : number of steps where density reconstruction returned None
    """
    T = len(results_in)
    N_dlv = x_pred_hist_arr.shape[1]

    # Build fresh-node matrix for this stencil
    fresh_matrix = np.zeros((T, N_dlv), dtype=bool)
    for k, (r, raw_mask) in enumerate(zip(results_in, _raw_fresh_anchor)):
        if r.get("failed"):
            continue
        if raw_mask is not None:
            fresh_matrix[k] = make_fresh_node_mask(raw_mask, stencil)
        elif r.get("dlvsqs") is not None:
            fresh_matrix[k] = True

    abs_resid_all = []
    abs_resid_allfresh = []
    abs_resid_hasnonfresh = []
    n_recon_fail = 0
    is_call_cache: dict = {}

    for k in range(1, T):
        r = results_in[k]
        if r.get("failed"):
            continue

        snap = _snap_lookup.get(r["ts"])
        if snap is None:
            continue
        strikes_k, bid_k, ask_k = snap
        if len(strikes_k) == 0:
            continue

        d = _density_from_kf_dlvsq(x_pred_hist_arr[k], r["fwd"])
        if d is None:
            n_recon_fail += 1
            continue

        mid_k = 0.5 * (bid_k + ask_k)
        hs_k = np.maximum(0.5 * (ask_k - bid_k), 1e-8)
        m = len(strikes_k)
        if m not in is_call_cache:
            is_call_cache[m] = np.full(m, _is_call, dtype=bool)

        px = option_prices(d, _grid_K, strikes_k, is_call_cache[m], r["fwd"])
        nse_quotes = np.abs(px - mid_k) / hs_k  # per-quote NSE contributions

        abs_resid_all.extend(nse_quotes.tolist())

        if fresh_matrix[k - 1].all():
            abs_resid_allfresh.extend(nse_quotes.tolist())
        else:
            abs_resid_hasnonfresh.extend(nse_quotes.tolist())

    def _mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    return (
        _mean(abs_resid_all),
        _mean(abs_resid_allfresh),
        _mean(abs_resid_hasnonfresh),
        n_recon_fail,
    )


# ── Section 5b metric ──────────────────────────────────────────────────────


def compute_pricing_metrics(
    results_in: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Held-out one-step-ahead pricing RMSE and NSE.

    At step k, prices fresh quotes with results_in[k-1]["densities"][0]
    (posterior from the previous step, before step k is assimilated).

    Returns
    -------
    step_rmse        : (T,) per-step RMSE, NaN if skipped
    step_nse         : (T,) per-step half-spread-normalised error, NaN if skipped
    step_rmse_stale  : (T,) RMSE restricted to quotes stale at k-1, NaN if none
    """
    T = len(results_in)
    step_rmse = np.full(T, np.nan)
    step_nse = np.full(T, np.nan)
    step_rmse_stale = np.full(T, np.nan)

    is_call_arr_cache: dict = {}
    prev_density = None
    prev_fresh_anchor = None

    for k in range(T):
        r = results_in[k]
        densities_k = r.get("densities")
        d_post = densities_k[0] if densities_k is not None else None

        if r["failed"]:
            prev_density = None
            prev_fresh_anchor = None
            continue

        fresh_anchor = r.get("fresh_anchor_mask")

        if prev_density is None:
            prev_density = d_post
            prev_fresh_anchor = fresh_anchor
            continue

        snap_data = _snap_lookup.get(r["ts"])
        if snap_data is None:
            prev_density = d_post
            prev_fresh_anchor = fresh_anchor
            continue

        strikes_k, bid_k, ask_k = snap_data
        if len(strikes_k) == 0:
            prev_density = d_post
            prev_fresh_anchor = fresh_anchor
            continue

        mid_k = 0.5 * (bid_k + ask_k)
        half_spread_k = np.maximum(0.5 * (ask_k - bid_k), 1e-8)
        m = len(strikes_k)
        if m not in is_call_arr_cache:
            is_call_arr_cache[m] = np.full(m, _is_call, dtype=bool)

        model_px = option_prices(prev_density, _grid_K, strikes_k, is_call_arr_cache[m], r["fwd"])
        resid = model_px - mid_k

        step_rmse[k] = float(np.sqrt(np.mean(resid**2)))
        step_nse[k] = float(np.mean(np.abs(resid) / half_spread_k))

        if prev_fresh_anchor is not None and fresh_anchor is not None:
            newly_fresh = fresh_anchor & ~prev_fresh_anchor
            if newly_fresh.any():
                nf_strikes = set(np.round(_anchor_pure[newly_fresh]).astype(int))
                stale_mask = np.array([int(round(s)) in nf_strikes for s in strikes_k], dtype=bool)
                if stale_mask.sum() > 0:
                    step_rmse_stale[k] = float(np.sqrt(np.mean(resid[stale_mask] ** 2)))

        prev_density = d_post
        prev_fresh_anchor = fresh_anchor

    return step_rmse, step_nse, step_rmse_stale


# ── Section 5c metric ──────────────────────────────────────────────────────


def compute_guardrail_metrics(
    x_hist_in: np.ndarray,
    P_diag_hist_in: np.ndarray,
    innov_hist_in: np.ndarray,
    ts_hist_in: np.ndarray,
    cfg_in: KFConfig,
) -> dict:
    """
    Compute NIS, lag-1 ACF, and stability guardrail metrics.

    Returns dict with: mean_nis_norm, nis_chi2_lo, nis_chi2_hi,
                       lag1_acf, n_blowup, n_neg_mean
    """
    T = len(ts_hist_in)
    dt_days = np.zeros(T)
    dt_days[1:] = np.diff(ts_hist_in) / cfg_in.trading_day_s

    # Approximate prior diagonal: Ppred[k,j] = P_post[k-1,j] + q_scale * Δt_k
    P_diag_pred = np.empty_like(P_diag_hist_in)
    P_diag_pred[0] = P_diag_hist_in[0]
    P_diag_pred[1:] = P_diag_hist_in[:-1] + cfg_in.q_scale * dt_days[1:, np.newaxis]

    nis_per_step = []
    m_per_step = []
    for k in range(T):
        fresh = ~np.isnan(innov_hist_in[k])
        m_k = int(fresh.sum())
        if m_k == 0:
            continue
        nu = innov_hist_in[k, fresh]
        S_diag = P_diag_pred[k, fresh] + cfg_in.r_diag
        nis_per_step.append(float(np.sum(nu**2 / S_diag)) / m_k)
        m_per_step.append(m_k)

    if nis_per_step:
        mean_nis_norm = float(np.mean(nis_per_step))
        total_m = int(sum(m_per_step))
        nis_chi2_lo = float(_chi2.ppf(0.025, df=total_m)) / total_m
        nis_chi2_hi = float(_chi2.ppf(0.975, df=total_m)) / total_m
    else:
        mean_nis_norm = nis_chi2_lo = nis_chi2_hi = float("nan")

    # Lag-1 autocorrelation of innovations, averaged over nodes
    n_dim = innov_hist_in.shape[1]
    acfs = []
    for j in range(n_dim):
        col = innov_hist_in[:, j]
        valid_k = np.where(~np.isnan(col))[0]
        if len(valid_k) < 3:
            continue
        pairs = valid_k[:-1][np.diff(valid_k) == 1]
        if len(pairs) < 2:
            continue
        a, b_vec = col[pairs], col[pairs + 1]
        if a.std() > 0 and b_vec.std() > 0:
            acfs.append(float(np.corrcoef(a, b_vec)[0, 1]))
    lag1_acf = float(np.nanmean(acfs)) if acfs else float("nan")

    # Wing posterior-std blow-ups
    n_wings = max(1, n_dim // 5)
    wing_idx = list(range(n_wings)) + list(range(n_dim - n_wings, n_dim))
    std_wing = np.sqrt(np.maximum(P_diag_hist_in[:, wing_idx], 0.0))
    baseline_steps = max(1, T // 20)
    std_baseline = std_wing[:baseline_steps].mean()
    std_threshold = 5.0 * std_baseline if std_baseline > 0 else 1.0
    n_blowup = int((std_wing > std_threshold).sum())

    n_neg_mean = 0
    if cfg_in.soft_floor is None:
        n_neg_mean = int((x_hist_in < 0).sum())

    return dict(
        mean_nis_norm=mean_nis_norm,
        nis_chi2_lo=nis_chi2_lo,
        nis_chi2_hi=nis_chi2_hi,
        lag1_acf=lag1_acf,
        n_blowup=n_blowup,
        n_neg_mean=n_neg_mean,
    )


# ── Worker ────────────────────────────────────────────────────────────────


def _run_one(task: dict) -> dict:
    global _cache
    results = _cache["results"]
    N_dlv = _cache["N_dlv"]
    grid_K_nodes = _cache["dlv_anchor_pure"]

    kp = task["kf_params"]
    cfg = KFConfig(
        q_scale=kp["q_scale"],
        ell=kp["ell"],
        r_diag=kp["r_diag"],
        stencil=kp["stencil"],
        p0_scale=kp.get("p0_scale", FIXED_KF["p0_scale"]),
        trading_day_s=_cache["trading_day_s"],
        dlv_max_sq_net=_cache["dlv_max_sq_net"],
        p0_correlated=FIXED_KF["p0_correlated"],
        soft_floor=FIXED_KF["soft_floor"],
        ignore_at_dlv_max=FIXED_KF["ignore_at_dlv_max"],
    )

    x_hist, x_pred_hist, P_diag_hist, innov_hist, ts_hist, failed_hist, n_fresh_hist = (
        run_dlv_kalman_filter_v2(results, N_dlv=N_dlv, grid_K_nodes=grid_K_nodes, cfg=cfg)
    )

    valid = innov_hist[~np.isnan(innov_hist)]
    post_std = np.sqrt(np.maximum(P_diag_hist, 0.0))
    atm_idx = N_dlv // 2

    # Section 5b: held-out one-step-ahead pricing metrics (batch density path)
    s_rmse, s_nse, s_rmse_stale = compute_pricing_metrics(results)

    # KF-prior pricing metrics (DLV²-reconstruction path, fresh/non-fresh split)
    nse_kf, nse_kf_allfresh, nse_kf_hasnonfresh, n_recon_fail = compute_pricing_metrics_kf_prior(
        results, x_pred_hist, kp["stencil"]
    )

    # Section 5c: guardrail metrics (NIS, ACF, stability)
    gm = compute_guardrail_metrics(x_hist, P_diag_hist, innov_hist, ts_hist, cfg)
    nis_in_band = gm["nis_chi2_lo"] <= gm["mean_nis_norm"] <= gm["nis_chi2_hi"]

    return {
        **kp,
        "q_r_ratio": kp["q_scale"] / kp["r_diag"],
        # innovation diagnostics
        "innov_mae": float(np.abs(valid).mean()) if len(valid) else float("nan"),
        "innov_std": float(valid.std()) if len(valid) else float("nan"),
        "innov_mean": float(valid.mean()) if len(valid) else float("nan"),
        # pricing metrics — batch density path (5b)
        "rmse_mean": float(np.nanmean(s_rmse)),
        "nse_mean": float(np.nanmean(s_nse)),
        "rmse_stale": float(np.nanmean(s_rmse_stale)),
        # pricing metrics — KF DLV²-reconstruction path
        "nse_kf": nse_kf,
        "nse_kf_allfresh": nse_kf_allfresh,
        "nse_kf_hasnonfresh": nse_kf_hasnonfresh,
        "n_recon_fail": n_recon_fail,
        # guardrail metrics (5c)
        "nis_norm": gm["mean_nis_norm"],
        "nis_chi2_lo": gm["nis_chi2_lo"],
        "nis_chi2_hi": gm["nis_chi2_hi"],
        "nis_in_band": nis_in_band,
        "lag1_acf": gm["lag1_acf"],
        "n_blowup": gm["n_blowup"],
        "n_neg_mean": gm["n_neg_mean"],
        # posterior and coverage stats
        "post_std_mean": float(post_std.mean()),
        "post_std_atm_mean": float(post_std[:, atm_idx].mean()),
        "n_fresh_mean": float(n_fresh_hist.mean()),
        "n_failed": int(failed_hist.sum()),
    }


def main(cache_path: str, out_path: str, n_workers: int) -> None:
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    tasks = [{"kf_params": dict(zip(keys, combo))} for combo in combos]
    print(f"Sweeping {len(tasks)} combinations across {n_workers} workers")

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(cache_path,),
    ) as pool:
        records = pool.map(_run_one, tasks)

    # NIS-passing rows ranked by held-out RMSE; NIS-failing rows appended at bottom
    pass_rows = [r for r in records if r.get("nis_in_band")]
    fail_rows = [r for r in records if not r.get("nis_in_band")]
    pass_rows.sort(key=lambda r: r.get("rmse_mean") or float("inf"))
    fail_rows.sort(key=lambda r: r.get("rmse_mean") or float("inf"))
    records = pass_rows + fail_rows

    fieldnames = list(records[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved to {out_path}")
    print(f"\nCells passing NIS band: {len(pass_rows)} / {len(records)}")
    if pass_rows:
        display_cols = [
            "stencil",
            "ell",
            "q_r_ratio",
            "nse_kf",
            "nse_kf_allfresh",
            "nse_kf_hasnonfresh",
            "nse_mean",
            "rmse_stale",
            "nis_norm",
            "lag1_acf",
            "n_blowup",
        ]
        present = [c for c in display_cols if c in fieldnames]
        header = "  ".join(f"{k:>12}" for k in present)
        print(f"\nTop 5 by held-out RMSE (NIS-passing):\n{header}")
        for r in pass_rows[:5]:
            row = "  ".join(
                f"{r[k]:>12.4g}" if isinstance(r[k], float) else f"{r[k]:>12}" for k in present
            )
            print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="phase1_cache.pkl")
    parser.add_argument("--out", default="sweep_results.csv")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()
    main(args.cache, args.out, args.workers)
