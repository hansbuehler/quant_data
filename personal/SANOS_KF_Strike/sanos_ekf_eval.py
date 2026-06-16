"""
Direct-observation SANOS EKF — evaluation harness ("Option 2").

Filters the DLV² state by assimilating *raw market quotes* (price-space
observations), instead of fitting SANOS per snapshot and filtering the fit.
This removes the "garbage batch" problem structurally: every quote enters the
update weighted by its own R = (half-spread)², so a thin snapshot simply
contributes little information rather than a toxic full-curve fit.

Built-in improvements (per design discussion):
  #1 forward re-centering  — the density-solve imposes mean=fwd_k each step,
                             so the martingale mean tracks the live forward.
  #2 correlated process / prior noise — Q = q_scale·Δt·C, P0 = p0·C, with C an
                             RBF correlation over the DLV nodes (ties sparse
                             wings to quoted neighbours).
  #3 warm start            — x0 = first snapshot's SANOS DLV² fit (not a flat
                             curve), removing the convergence transient.
  #5 comparable evaluation — held-out / in-sample mean & median NSE computed on
                             the SAME full-market metric as the baselines
                             (persist-all ≈ 4.36 mean, gate ≈ 0.67).

Observation map and its Jacobian are closed-form numpy (the density solve is an
N×N linear system; dd/dx is one matrix column per node), so no autodiff is
needed and the whole day runs in numpy.
"""

from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

# Anchor sibling imports and cache lookups to this file's location, so the
# module works regardless of the notebook / process working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from dlv_pricing_absolute import option_prices

# Directories searched (in order) for phase1_cache_*.pkl.
_CACHE_DIRS = [_HERE, os.path.dirname(_HERE), os.getcwd()]


def _resolve_cache(trade_date) -> str:
    name = f"phase1_cache_{trade_date}.pkl"
    for d in _CACHE_DIRS:
        path = os.path.join(d, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} not found in {_CACHE_DIRS}")


TRADE_DATE = "2024-01-03"


# ── Load cache ────────────────────────────────────────────────────────────────
def load_cache(trade_date=TRADE_DATE):
    """(Re)load phase1_cache_{trade_date}.pkl into module globals.
    Call before run()/persist_baseline() to switch dates within a notebook."""
    global TRADE_DATE, results, snap_lookup, grid_K, dlv_anchor, xstrikes_row
    global tte, dlv_min, is_call_flag, trading_day_s, N, n_state, box, pay_full, _I
    TRADE_DATE = str(trade_date)
    with open(_resolve_cache(TRADE_DATE), "rb") as _f:
        _c = pickle.load(_f)
    results = _c["results"]
    snap_lookup = _c["snap_lookup"]
    grid_K = _c["grid_K"].astype(float)
    dlv_anchor = _c["dlv_anchor_pure"].astype(float)  # interior DLV nodes (n_state)
    xstrikes_row = _c["xstrikes_row"].astype(float)
    tte = float(_c["tte_geom"])
    dlv_min = _c["phase1_config"]["dlv_min"]
    is_call_flag = bool(_c["is_call"])
    trading_day_s = _c["trading_day_s"]
    N = len(grid_K)
    n_state = N - 2
    box = grid_K**2 / (xstrikes_row[2:] - xstrikes_row[:-2]) * tte
    box[0] = 0.0
    box[-1] = 0.0
    pay_full = np.maximum(grid_K[np.newaxis, :] - grid_K[:, np.newaxis], 0.0)  # (N,N)
    _I = np.eye(n_state)


load_cache()  # default date at import so __main__/_selftest still work


# ── Differentiable density + observation (closed form) ────────────────────────
def _dens_and_ddx(x: np.ndarray, fwd: float):
    """
    Solve the exactly-constrained SANOS density system A(x) d = b(fwd):
      row 0 : sum d = 1           (normalisation)
      row 1 : sum d·K = fwd       (martingale → forward re-centering, #1)
      rows  : DLV² equalities
    Return (d, dd_dx) with dd_dx[:, s] = ∂d/∂x_s, derived from
    ∂d/∂x_s = -A⁻¹ (∂A/∂x_s) d and the single-entry structure of ∂A/∂x_s.
    """
    v = x + dlv_min**2
    A = np.zeros((N, N))
    b = np.zeros(N)
    A[0] = 1.0
    b[0] = 1.0
    A[1] = grid_K
    b[1] = fwd
    A[2:N] = pay_full[1 : N - 1]
    A[np.arange(2, N), np.arange(1, N - 1)] -= v * box[1 : N - 1]
    b[2:N] = np.maximum(fwd - grid_K[1 : N - 1], 0.0)
    Ainv = np.linalg.inv(A)
    d = Ainv @ b
    dd_dx = Ainv[:, 2:N] * (box[1 : N - 1] * d[1 : N - 1])[np.newaxis, :]
    return d, dd_dx


def _density_eval(x: np.ndarray, fwd: float):
    """Baseline-identical reconstruction (clip + reject) for fair NSE scoring."""
    v = x + dlv_min**2
    A = np.zeros((N, N))
    b = np.zeros(N)
    A[0] = 1.0
    b[0] = 1.0
    A[1] = grid_K
    b[1] = fwd
    A[2:N] = pay_full[1 : N - 1]
    A[np.arange(2, N), np.arange(1, N - 1)] -= v * box[1 : N - 1]
    b[2:N] = np.maximum(fwd - grid_K[1 : N - 1], 0.0)
    try:
        d = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    if (-np.minimum(d, 0.0)).sum() > 0.02:
        return None
    d = np.maximum(d, 0.0)
    s = d.sum()
    return d / s if s > 1e-12 else None


def _corr(nodes: np.ndarray, ell: float) -> np.ndarray:
    dist = np.abs(nodes[:, None] - nodes[None, :])
    return np.exp(-0.5 * (dist / ell) ** 2)


def _tsec(r) -> float:
    ts = r["ts"]
    return ts.timestamp() if hasattr(ts, "timestamp") else float(ts) / 1e9


# ── EKF run over snapshots ────────────────────────────────────────────────────
def run(
    q_scale, ell, p0_scale=1e-6, warm=True, correlated=True, verbose=False, return_arrays=False
):
    C = _corr(dlv_anchor, ell) if correlated else _I
    P = p0_scale * C

    # warm start (#3): first valid snapshot's SANOS DLV² fit, else flat
    idx0 = next(
        (k for k, r in enumerate(results) if not r["failed"] and r.get("dlvsqs") is not None),
        None,
    )
    if warm and idx0 is not None:
        x = results[idx0]["dlvsqs"][0].copy()
    else:
        x = np.full(n_state, 0.04)

    t_prev = _tsec(results[idx0]) if idx0 is not None else 0.0

    T = len(results)
    nse_ho = np.full(T, np.nan)
    nse_is = np.full(T, np.nan)
    ts_hist = np.full(T, np.nan)
    x_prior_hist = np.full((T, n_state), np.nan)
    x_post_hist = np.full((T, n_state), np.nan)

    for k in range(T):
        r = results[k]
        t_k = _tsec(r)
        dt_days = max(t_k - t_prev, 0.0) / trading_day_s
        t_prev = t_k
        ts_hist[k] = t_k

        # ── predict (correlated process noise, #2) ────────────────────────────
        P_pred = P + q_scale * dt_days * C
        P_pred = 0.5 * (P_pred + P_pred.T)
        x_prior = x.copy()
        x_prior_hist[k] = x_prior

        snap = snap_lookup.get(r["ts"]) if not r["failed"] else None
        has_mkt = snap is not None and len(snap[0]) > 0

        # ── held-out: price snapshot k with prior (state propagated from k-1) ──
        if has_mkt:
            sk, bk, ak = snap
            mid = 0.5 * (bk + ak)
            hs = np.maximum(0.5 * (ak - bk), 1e-8)
            d_ho = _density_eval(x_prior, r["fwd"])
            if d_ho is not None:
                px = option_prices(d_ho, grid_K, sk, np.full(len(sk), is_call_flag), r["fwd"])
                nse_ho[k] = float(np.mean(np.abs(px - mid) / hs))

        # ── update: assimilate snapshot k's raw quotes (batch EKF) ────────────
        if has_mkt and not r["failed"]:
            fwd = r["fwd"]
            d, dd_dx = _dens_and_ddx(x_prior, fwd)
            pay_q = np.maximum(grid_K[np.newaxis, :] - sk[:, np.newaxis], 0.0)  # (m,N)
            h = pay_q @ d  # model call price C/df
            H = pay_q @ dd_dx  # (m, n_state)
            # market call price (PCP for puts) and R from half-spread
            z = mid + (0.0 if is_call_flag else (fwd - sk))
            R = np.maximum(hs**2, 1e-12)
            S = H @ P_pred @ H.T + np.diag(R)
            try:
                K_gain = P_pred @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                x = x_prior
                P = P_pred
            else:
                x = np.maximum(x_prior + K_gain @ (z - h), 0.0)
                IKH = _I - K_gain @ H
                P = IKH @ P_pred @ IKH.T + K_gain @ np.diag(R) @ K_gain.T
                P = 0.5 * (P + P.T)
        else:
            x = x_prior
            P = P_pred
        x_post_hist[k] = x

        # ── in-sample: price snapshot k with posterior ────────────────────────
        if has_mkt:
            d_is = _density_eval(x, r["fwd"])
            if d_is is not None:
                px = option_prices(d_is, grid_K, sk, np.full(len(sk), is_call_flag), r["fwd"])
                nse_is[k] = float(np.mean(np.abs(px - mid) / hs))

    out = dict(
        q_scale=q_scale,
        ell=ell,
        warm=warm,
        correlated=correlated,
        ho_mean=float(np.nanmean(nse_ho)),
        ho_med=float(np.nanmedian(nse_ho)),
        is_mean=float(np.nanmean(nse_is)),
        is_med=float(np.nanmedian(nse_is)),
        n_eval=int(np.sum(~np.isnan(nse_ho))),
    )
    if return_arrays:
        out.update(
            nse_ho_arr=nse_ho,
            nse_is_arr=nse_is,
            ts=ts_hist,
            x_prior=x_prior_hist,
            x_post=x_post_hist,
        )
    if verbose:
        print({k: v for k, v in out.items() if not isinstance(v, np.ndarray)})
    return out


def eval_table(q_scale, ell, p0_scale=1e-2, warm=False, correlated=True, model_name="SANOS EKF"):
    """Run the EKF and emit the unified-metrics standard evaluation table.

    One row per (snapshot, strike) market quote, all in CALL price space (C/df):
      prior_call = model call price from the prior state (held-out / one-step ahead)
      post_call  = model call price from the posterior state (in-sample)
      mid_call   = market mid converted to a call via put-call parity
    Pass the returned DataFrame to unified_metrics.evaluate(...).
    """
    import pandas as pd
    from dlv_pricing_absolute import call_prices

    o = run(q_scale, ell, p0_scale=p0_scale, warm=warm, correlated=correlated, return_arrays=True)
    x_prior, x_post = o["x_prior"], o["x_post"]
    rows = []
    for k in range(len(results)):
        r = results[k]
        snap = snap_lookup.get(r["ts"]) if not r["failed"] else None
        if snap is None or len(snap[0]) == 0:
            continue
        sk, bk, ak = snap
        fwd = r["fwd"]
        mid = 0.5 * (bk + ak)
        hs = np.maximum(0.5 * (ak - bk), 1e-8)
        # market mid -> CALL price (PCP: C = P + F - K for puts)
        mid_call = mid if is_call_flag else mid + (fwd - sk)
        d_ho = _density_eval(x_prior[k], fwd)
        d_is = _density_eval(x_post[k], fwd)
        prior_call = call_prices(d_ho, grid_K, sk) if d_ho is not None else np.full(len(sk), np.nan)
        post_call = call_prices(d_is, grid_K, sk) if d_is is not None else np.full(len(sk), np.nan)
        for j in range(len(sk)):
            rows.append(
                (
                    k,
                    float(sk[j]),
                    float(fwd),
                    float(mid_call[j]),
                    float(hs[j]),
                    float(prior_call[j]),
                    float(post_call[j]),
                )
            )
    return pd.DataFrame(
        rows,
        columns=["ts", "strike", "forward", "mid_call", "half_spread", "prior_call", "post_call"],
    )


def persist_baseline(return_arrays=False):
    """Persist-all baseline: price snapshot k with snapshot k-1's SANOS DLV² fit
    (re-centered on the current forward), on the identical held-out metric."""
    T = len(results)
    nse = np.full(T, np.nan)
    ts_hist = np.full(T, np.nan)
    prev_dl = None
    for k in range(T):
        r = results[k]
        ts_hist[k] = _tsec(r)
        if r["failed"]:
            prev_dl = None
            continue
        snap = snap_lookup.get(r["ts"])
        if snap is not None and len(snap[0]) > 0 and prev_dl is not None:
            sk, bk, ak = snap
            mid = 0.5 * (bk + ak)
            hs = np.maximum(0.5 * (ak - bk), 1e-8)
            d = _density_eval(prev_dl, r["fwd"])
            if d is not None:
                px = option_prices(d, grid_K, sk, np.full(len(sk), is_call_flag), r["fwd"])
                nse[k] = float(np.mean(np.abs(px - mid) / hs))
        if r.get("dlvsqs") is not None:
            prev_dl = r["dlvsqs"][0]
    out = dict(
        ho_mean=float(np.nanmean(nse)),
        ho_med=float(np.nanmedian(nse)),
        n_eval=int(np.sum(~np.isnan(nse))),
    )
    if return_arrays:
        out.update(nse_ho_arr=nse, ts=ts_hist)
    return out


# ── self-test: analytic Jacobian vs finite differences ────────────────────────
def _selftest():
    k = next(i for i, r in enumerate(results) if not r["failed"] and r.get("dlvsqs") is not None)
    x = results[k]["dlvsqs"][0].copy()
    fwd = results[k]["fwd"]
    d0, ddx = _dens_and_ddx(x, fwd)
    eps = 1e-6
    fd = np.zeros_like(ddx)
    for s in range(n_state):
        xp, xm = x.copy(), x.copy()
        xp[s] += eps
        xm[s] -= eps
        dp, _ = _dens_and_ddx(xp, fwd)
        dm, _ = _dens_and_ddx(xm, fwd)
        fd[:, s] = (dp - dm) / (2 * eps)
    err = np.max(np.abs(fd - ddx))
    print(f"[selftest] max|analytic dd/dx - finite-diff| = {err:.2e}")
    print(f"[selftest] density sum={d0.sum():.6f}  mean={d0 @ grid_K:.4f}  fwd={fwd:.4f}")
    assert err < 1e-3, "Jacobian mismatch"


if __name__ == "__main__":
    _selftest()
    print("\nbaselines (fit-then-filter, same metric): persist-all NSE mean≈4.36 | gate≈0.67\n")
    print(f"{'cfg':<46}{'HO mean':>9}{'HO med':>8}{'IS mean':>9}{'IS med':>8}{'n_eval':>8}")
    # q_scale is in DLV²² per trading day. DLV² ~ O(0.01–4) and moves little
    # intraday, so q_scale is small — unlike the fit-then-filter pipeline (where
    # the observation WAS DLV² with R≈0, so q_scale≈1e3–1e5).
    # Optimum region: flat init (warm-start hurts — seeds capped wing values),
    # tight ell, q≈1 (DLV²²/day). Plus ablations isolating each knob at q=1,ell=3.
    sweep = [
        dict(q_scale=q, ell=ell, p0_scale=1e-2, warm=False, correlated=True)
        for ell in (2.0, 3.0, 5.0)
        for q in (3e-1, 1.0, 3.0)
    ]
    ablations = [
        dict(q_scale=1.0, ell=3.0, p0_scale=1e-2, warm=True, correlated=True),  # +warm (hurts)
        dict(q_scale=1.0, ell=3.0, p0_scale=1e-2, warm=False, correlated=False),  # -corr (hurts)
    ]
    recs = []
    for cfg in sweep + ablations:
        o = run(**cfg)
        recs.append(o)
        lbl = (
            f"q={cfg['q_scale']:.0e} ell={cfg['ell']:.0f} p0={cfg['p0_scale']:.0e}"
            f" warm={int(cfg['warm'])} corr={int(cfg['correlated'])}"
        )
        print(
            f"{lbl:<46}{o['ho_mean']:>9.3f}{o['ho_med']:>8.3f}"
            f"{o['is_mean']:>9.3f}{o['is_med']:>8.3f}{o['n_eval']:>8}"
        )

    best_mean = min(recs, key=lambda o: o["ho_mean"])
    best_med = min(recs, key=lambda o: o["ho_med"])
    print(
        f"\nbest held-out MEAN  : {best_mean['ho_mean']:.3f} "
        f"(q={best_mean['q_scale']:.0e} ell={best_mean['ell']:.0f} "
        f"warm={int(best_mean['warm'])} corr={int(best_mean['correlated'])})"
    )
    print(
        f"best held-out MEDIAN: {best_med['ho_med']:.3f} "
        f"(q={best_med['q_scale']:.0e} ell={best_med['ell']:.0f} "
        f"warm={int(best_med['warm'])} corr={int(best_med['correlated'])})"
    )
    print("vs baselines — persist-all: mean 4.36 / med 0.76   gate: mean 0.67 / med 0.57")
