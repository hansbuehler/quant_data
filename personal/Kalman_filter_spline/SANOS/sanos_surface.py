"""
sanos_surface.py

Batch LP calibration of the SANOS (Section 4.1 homogeneous prototype) arbitrage-free
call price surface.

Reference
---------
SANOS paper, Section 4.1 — single-expiry LP with uniform anchor strike grid.

The surface is:
    C(K) = Σ_j q_j · BS_call(spot=K_j^anchor, strike=K, var=η·V_atm)

where q is a discrete probability on the anchor strikes satisfying:
    q ≥ 0,  Σ q_j = 1,  Σ q_j · K_j^anchor = F  (martingale)

Performance notes
-----------------
- Design matrix built via numpy broadcast — no Python loop.
- IV inversion: cdxcore.bs.implied() — hybrid bisection + Newton-Raphson, vectorized.
- LP solves run in parallel across expiries via ThreadPoolExecutor.
- Evaluation groups all quotes/trades by expiry: one call per expiry, not per row.

Public API
----------
SanosConfig          — configuration dataclass
fit_sanos_slice()    — fit one expiry, returns (anchor_ks, q, V_atm)
eval_sanos_iv()      — evaluate fitted surface in IV space
run_sanos_batch()    — whole-day batch pipeline (matches run_spline_surface_kalman API)
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import polars as pl
from cdxcore.bs import bs
from scipy.optimize import brentq
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SanosConfig:
    N: int = 30  # number of anchor strikes
    # smoothness: 0=kinky exact fit, 0.25=paper default (good for T≥0.05), 0.9+=short-dated
    # For SPY daily options (T≈0.008) use eta≥0.8 to avoid near-degenerate LP basis.
    eta: float = 0.25
    k_min: float = -0.25  # anchor grid log-moneyness lower bound
    k_max: float = 0.15  # anchor grid log-moneyness upper bound
    eps_obj: float = 1e-8  # L1 coefficient in objective (numerical regularizer)
    solver: str = "CLARABEL"


# ---------------------------------------------------------------------------
# Scalar BS primitives — kept for backward compatibility / notebook cells
# ---------------------------------------------------------------------------


def _bs_call_norm(anchor_ks: np.ndarray, market_k: float, v: float) -> np.ndarray:
    """
    Undiscounted BS call price normalized by forward (F=1).

    Vectorized over anchor_ks, scalar market_k and v.

        d1 = (anchor_ks - market_k + v/2) / sqrt(v)
        d2 = d1 - sqrt(v)
        Call_norm = exp(anchor_ks) * N(d1) - exp(market_k) * N(d2)

    Returns (N,) array.
    """
    v = max(float(v), 1e-10)
    sqrt_v = np.sqrt(v)
    d1 = (anchor_ks - market_k + v / 2.0) / sqrt_v
    d2 = d1 - sqrt_v
    return np.exp(anchor_ks) * norm.cdf(d1) - np.exp(market_k) * norm.cdf(d2)


def _bs_call_norm_single(market_k: float, T: float, iv: float) -> float:
    """
    BS call price normalized by F=1 at a single market strike, given IV.

        d1 = (-market_k + 0.5*iv^2*T) / (iv*sqrt(T))
        Call_norm = N(d1) - exp(market_k) * N(d2)
    """
    iv = max(float(iv), 1e-8)
    T = max(float(T), 1e-8)
    sqrtT = np.sqrt(T)
    v = iv * iv * T
    d1 = (-market_k + 0.5 * v) / (iv * sqrtT)
    d2 = d1 - iv * sqrtT
    return float(norm.cdf(d1) - np.exp(market_k) * norm.cdf(d2))


def _bs_put_norm_single(market_k: float, T: float, iv: float) -> float:
    """Normalized put price via put-call parity: P = C - 1 + exp(k)."""
    return _bs_call_norm_single(market_k, T, iv) - 1.0 + np.exp(market_k)


def _iv_from_norm_price(price: float, market_k: float, T: float) -> float:
    """
    Invert BS to recover implied vol from a normalized call price.

    Returns nan if price is outside no-arbitrage bounds
    [max(0, 1 - exp(market_k)), 1.0].
    """
    intrinsic = max(0.0, 1.0 - np.exp(market_k))
    if price <= intrinsic or price >= 1.0:
        return float("nan")
    try:
        return brentq(
            lambda iv: _bs_call_norm_single(market_k, T, iv) - price, 1e-6, 5.0, xtol=1e-8
        )
    except ValueError:
        return float("nan")


def _iv_from_norm_price_put(price: float, market_k: float, T: float) -> float:
    """
    Invert BS to recover implied vol from a normalized put price.

    Numerically preferable for OTM puts (k < 0) and short T.
    Returns nan when price is too close to zero (option has negligible time value).
    """
    intrinsic = max(0.0, np.exp(market_k) - 1.0)
    if price <= intrinsic or price < 5e-5 or price >= np.exp(market_k):
        return float("nan")
    try:
        return brentq(lambda iv: _bs_put_norm_single(market_k, T, iv) - price, 1e-6, 5.0, xtol=1e-8)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Batch BS primitives — vectorized, no Python loops
# ---------------------------------------------------------------------------


def _bs_call_norm_matrix(anchor_ks: np.ndarray, eval_ks: np.ndarray, v: float) -> np.ndarray:
    """
    Full (M, N) design matrix via numpy broadcasting.

        A[i, j] = BS_call(anchor=anchor_ks[j], strike=eval_ks[i], var=v)

    Replaces the Python list-comprehension [_bs_call_norm(anchor_ks, mk, v) for mk in eval_ks].
    Returns (M, N) ndarray.
    """
    v = max(float(v), 1e-10)
    sqrt_v = np.sqrt(v)
    a = anchor_ks[np.newaxis, :]  # (1, N)
    m = eval_ks[:, np.newaxis]  # (M, 1)
    d1 = (a - m + v / 2.0) / sqrt_v
    d2 = d1 - sqrt_v
    return np.exp(a) * norm.cdf(d1) - np.exp(m) * norm.cdf(d2)  # (M, N)


def _bs_call_norm_batch(eval_ks: np.ndarray, T: float, ivs: np.ndarray) -> np.ndarray:
    """
    BS call price for each (eval_ks[i], ivs[i]) pair.

    Vectorized over (M,) arrays — replaces per-element calls in fit_sanos_slice.
    Returns (M,) ndarray.
    """
    ivs = np.maximum(np.asarray(ivs, dtype=float), 1e-8)
    eval_ks = np.asarray(eval_ks, dtype=float)
    sqrtT = np.sqrt(max(float(T), 1e-8))
    v = ivs * ivs * float(T)
    d1 = (-eval_ks + 0.5 * v) / (ivs * sqrtT)
    d2 = d1 - ivs * sqrtT
    return norm.cdf(d1) - np.exp(eval_ks) * norm.cdf(d2)


# ---------------------------------------------------------------------------
# Vectorized IV inversion — cdxcore.bs.implied (hybrid bisection + NR)
# ---------------------------------------------------------------------------


def _invert_iv_batch(c_model_vec: np.ndarray, eval_ks: np.ndarray, T: float) -> np.ndarray:
    """
    Invert BS call prices to implied vols for all strikes simultaneously.

    Uses cdxcore.bs.implied (hybrid bisection + Newton-Raphson, vectorized).
    OTM calls (k≥0) are inverted directly; OTM puts (k<0) use put price via PCP.
    Returns (M,) IVs — nan where price is degenerate (near intrinsic or < 5e-5 for puts).
    """
    eval_ks = np.asarray(eval_ks, dtype=float)
    c_model_vec = np.asarray(c_model_vec, dtype=float)

    k_ratio = np.exp(eval_ks)
    call_mask = eval_ks >= 0.0

    # OTM calls: use call price; OTM puts: convert call → put via PCP
    prices = np.where(call_mask, c_model_vec, c_model_vec + k_ratio - 1.0)

    # Pre-filter degenerate prices that bs.implied cannot bracket:
    #   calls: must be strictly above intrinsic and below 1
    #   puts:  must be ≥ 5e-5 (below this IV is indeterminate at short T)
    intr = np.maximum(0.0, np.where(call_mask, 1.0 - k_ratio, k_ratio - 1.0))
    valid = (prices > intr + 1e-8) & np.where(call_mask, prices < 1.0, prices >= 5e-5)

    ivs = np.full(len(eval_ks), float("nan"))
    if valid.any():
        ivs[valid] = bs.implied(
            k=k_ratio[valid],
            prices=prices[valid],
            is_call=call_mask[valid],
            sqrtT=np.full(int(valid.sum()), np.sqrt(max(float(T), 1e-8))),
            price_tol=1e-7,
            on_exceed_bounds="quiet",
        )
    return ivs


# ---------------------------------------------------------------------------
# Single-expiry LP fit (Section 4.1 — homogeneous anchor grid)
# ---------------------------------------------------------------------------


def fit_sanos_slice(
    market_ks: np.ndarray,
    bid_ivs: np.ndarray,
    ask_ivs: np.ndarray,
    T: float,
    cfg: SanosConfig,
    is_calls: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """
    Fit the SANOS homogeneous LP for one expiry slice.

    Parameters
    ----------
    market_ks : (M,) log-moneyness of market quotes (OTM preferred)
    bid_ivs   : (M,) bid implied volatilities
    ask_ivs   : (M,) ask implied volatilities
    T         : time-to-expiry in years
    cfg       : SanosConfig
    is_calls  : (M,) bool, True=call, False=put.  If None, infer from sign of k.

    Returns
    -------
    anchor_ks : (N,) log-moneyness of anchor strikes
    q_star    : (N,) fitted probability weights, or None if LP failed
    V_atm     : ATM total variance used for the η-scaling
    """
    market_ks = np.asarray(market_ks, dtype=float)
    bid_ivs = np.asarray(bid_ivs, dtype=float)
    ask_ivs = np.asarray(ask_ivs, dtype=float)

    if is_calls is None:
        is_calls = market_ks >= 0.0

    # ── ATM total variance ────────────────────────────────────────────────────
    atm_idx = int(np.argmin(np.abs(market_ks)))
    atm_iv = np.sqrt(bid_ivs[atm_idx] * ask_ivs[atm_idx])  # geometric mean
    V_atm = atm_iv**2 * max(T, 1e-8)

    # ── Market call prices (vectorized, no loop) ──────────────────────────────
    # _bs_call_norm_batch returns the call price for any IV regardless of call/put:
    # put-call parity with the same IV gives the same call price.
    bid_calls = _bs_call_norm_batch(market_ks, T, bid_ivs)
    ask_calls = _bs_call_norm_batch(market_ks, T, ask_ivs)
    mid_calls = (bid_calls + ask_calls) / 2.0

    valid = np.isfinite(bid_calls) & np.isfinite(ask_calls) & (ask_calls > bid_calls)
    if valid.sum() < 3:
        warnings.warn(f"SANOS: too few valid quotes ({valid.sum()}); LP skipped.")
        return np.linspace(cfg.k_min, cfg.k_max, cfg.N), None, V_atm

    M_v = int(valid.sum())
    bid_v = bid_calls[valid]
    ask_v = ask_calls[valid]
    mid_v = mid_calls[valid]
    w = 1.0 / np.maximum((ask_v - bid_v) / 2.0, 1e-6)

    # ── Anchor grid — N capped at the number of valid quotes ─────────────────
    # More anchors than data points leaves the LP underdetermined: q becomes
    # arbitrarily sparse.  N_eff ≤ M_v ensures each anchor is constrained by
    # at least one market quote on average.
    N_eff = max(5, min(cfg.N, M_v))
    anchor_ks = np.linspace(cfg.k_min, cfg.k_max, N_eff)
    anchor_K = np.exp(anchor_ks)
    v_smooth = max(cfg.eta * V_atm, 1e-12)
    A = _bs_call_norm_matrix(anchor_ks, market_ks[valid], v_smooth)  # (M_v, N_eff)

    # ── CVXPY LP (MDL objective, eq. 33 of paper) ────────────────────────────
    q = cp.Variable(N_eff, nonneg=True)
    over = cp.Variable(M_v, nonneg=True)  # max(c_model - ask, 0)
    under = cp.Variable(M_v, nonneg=True)  # max(bid - c_model, 0)
    c_mod = A @ q

    prob = cp.Problem(
        cp.Minimize(w @ (cfg.eps_obj * cp.abs(c_mod - mid_v) + over + under)),
        [
            cp.sum(q) == 1.0,  # unit probability
            anchor_K @ q == 1.0,  # martingale: E[K_anchor] = F = 1
            over >= c_mod - ask_v,
            under >= bid_v - c_mod,
        ],
    )
    try:
        prob.solve(solver=cfg.solver, verbose=False)
    except cp.SolverError as e:
        warnings.warn(f"SANOS: solver error — {e}")
        return anchor_ks, None, V_atm

    if prob.status not in ("optimal", "optimal_inaccurate"):
        warnings.warn(f"SANOS: LP status '{prob.status}'; returning None.")
        return anchor_ks, None, V_atm
    if prob.status == "optimal_inaccurate":
        warnings.warn("SANOS: LP returned 'optimal_inaccurate'; results may be unreliable.")

    q_star = np.maximum(q.value, 0.0)  # clip tiny negatives from solver tolerance
    q_star /= q_star.sum()  # re-normalize
    return anchor_ks, q_star, V_atm


# ---------------------------------------------------------------------------
# Surface evaluation — fully vectorized
# ---------------------------------------------------------------------------


def eval_sanos_iv(
    q: np.ndarray,
    anchor_ks: np.ndarray,
    eval_ks: np.ndarray,
    V_atm: float,
    T: float,
    eta: float,
) -> np.ndarray:
    """
    Evaluate the fitted SANOS surface in IV space.

    Fully vectorized: one matrix multiply + batch IV inversion, no per-strike loop.

    Parameters
    ----------
    q         : (N,) fitted density from fit_sanos_slice
    anchor_ks : (N,) anchor log-moneyness grid
    eval_ks   : (M,) evaluation log-moneyness values
    V_atm     : ATM total variance used during fitting
    T         : time-to-expiry in years
    eta       : smoothness parameter (same as used during fitting)

    Returns
    -------
    (M,) implied volatilities (nan where BS inversion fails)
    """
    v_smooth = max(eta * V_atm, 1e-12)
    eval_ks = np.asarray(eval_ks, dtype=float)
    A_eval = _bs_call_norm_matrix(anchor_ks, eval_ks, v_smooth)  # (M, N)
    c_model = A_eval @ q  # (M,)
    return _invert_iv_batch(c_model, eval_ks, T)


# ---------------------------------------------------------------------------
# Batch pipeline — parallel LP solves + batch evaluation
# ---------------------------------------------------------------------------


def run_sanos_batch(
    df_quotes_iv: pl.DataFrame,
    df_trades_iv: pl.DataFrame,
    cfg: SanosConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Whole-day batch SANOS calibration.

    LP solves run in parallel across expiries (ThreadPoolExecutor).
    Evaluation groups all quotes/trades by expiry — one eval call per expiry,
    not per row.

    NOTE: This is a batch fit — the surface uses all quotes simultaneously,
    so it has access to "future" quotes relative to any given trade timestamp.
    Residuals reflect surface quality, not online prediction quality.

    Parameters
    ----------
    df_quotes_iv : output of build_iv_observations() (quotes side)
    df_trades_iv : output of build_iv_observations() (trades side)
    cfg          : SanosConfig

    Returns
    -------
    df_quotes_result : df_quotes_iv + column sanos_iv
    df_trades_result : df_trades_iv + columns sanos_matched, trade_residual
    """
    # OTM filter for LP fitting (calls k≥0, puts k<0)
    df_otm = df_quotes_iv.filter(
        ((pl.col("is_call")) & (pl.col("log_moneyness") >= 0.0))
        | ((~pl.col("is_call")) & (pl.col("log_moneyness") < 0.0))
    )

    # Unique expiries (round T to 6 decimal places to merge float noise)
    T_col = (pl.col("T") * 1e6).round(0) / 1e6
    expiry_groups = (
        df_otm.with_columns(T_col.alias("_T_round"))
        .group_by("_T_round")
        .agg(pl.col("T").first())
        .sort("_T_round")
    )
    expiry_vals = expiry_groups["T"].to_numpy()

    # ── Build LP task list ────────────────────────────────────────────────────
    task_args = []
    for T_exp in expiry_vals:
        T_round = round(T_exp * 1e6) / 1e6
        grp = df_otm.filter(((pl.col("T") * 1e6).round(0) / 1e6) == T_round)
        mks = grp["log_moneyness"].to_numpy().astype(float)
        bids = grp["bid_iv"].to_numpy().astype(float)
        asks = grp["ask_iv"].to_numpy().astype(float)
        ics = grp["is_call"].to_numpy().astype(bool)
        ok = np.isfinite(mks) & np.isfinite(bids) & np.isfinite(asks)
        task_args.append((T_exp, T_round, mks[ok], bids[ok], asks[ok], ics[ok], int(ok.sum())))

    def _fit_one(args):
        T_exp, T_round, mks, bids, asks, ics, n_ok = args
        if n_ok < 3:
            return T_round, (np.linspace(cfg.k_min, cfg.k_max, cfg.N), None, 0.0), T_exp, n_ok
        anchor_ks, q_star, V_atm = fit_sanos_slice(mks, bids, asks, T_exp, cfg, ics)
        return T_round, (anchor_ks, q_star, V_atm), T_exp, n_ok

    # ── Parallel LP solves (CLARABEL releases GIL during solve) ──────────────
    n_workers = min(len(expiry_vals), os.cpu_count() or 1)
    slice_results: dict[float, tuple[np.ndarray, np.ndarray | None, float]] = {}

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for T_round, result, T_exp, n_ok in pool.map(_fit_one, task_args):
            anchor_ks, q_star, V_atm = result
            martingale_err = (
                abs(np.exp(anchor_ks) @ q_star - 1.0) if q_star is not None else float("nan")
            )
            status = "OK" if q_star is not None else "FAILED"
            print(
                f"SANOS: T={T_exp:.4f}  status={status}  n_quotes={n_ok}"
                f"  V_atm={V_atm:.5f}  martingale_err={martingale_err:.2e}"
            )
            slice_results[T_round] = result

    # ── Nearest-expiry mapping (vectorized, no per-row loop) ──────────────────
    slice_keys = np.array(sorted(slice_results.keys()))

    def _nearest_keys(T_rounds: np.ndarray) -> np.ndarray:
        """For each T_round, return the nearest slice key (vectorized)."""
        diffs = np.abs(T_rounds[:, np.newaxis] - slice_keys[np.newaxis, :])  # (n, n_exp)
        return slice_keys[np.argmin(diffs, axis=1)]

    # ── Evaluate at all quotes — one call per expiry, not per row ─────────────
    q_ks = df_quotes_iv["log_moneyness"].to_numpy().astype(float)
    q_Ts = df_quotes_iv["T"].to_numpy().astype(float)
    q_T_nearest = _nearest_keys(np.round(q_Ts * 1e6) / 1e6)
    sanos_iv_q = np.full(len(q_ks), float("nan"))

    for T_key in slice_keys:
        anchor_ks_e, q_star_e, V_atm_e = slice_results[T_key]
        if q_star_e is None:
            continue
        qm = q_T_nearest == T_key
        if not qm.any():
            continue
        batch_ks = q_ks[qm]
        ok_ks = np.isfinite(batch_ks)
        if ok_ks.any():
            idx = np.where(qm)[0][ok_ks]
            sanos_iv_q[idx] = eval_sanos_iv(
                q_star_e, anchor_ks_e, batch_ks[ok_ks], V_atm_e, float(T_key), cfg.eta
            )

    df_quotes_result = df_quotes_iv.with_columns(
        pl.Series("sanos_iv", sanos_iv_q.tolist()),
    )

    # ── Evaluate at trades — one call per expiry ──────────────────────────────
    if df_trades_iv.is_empty():
        return df_quotes_result, df_trades_iv.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("sanos_matched"),
                pl.lit(None, dtype=pl.Float64).alias("trade_residual"),
            ]
        )

    t_ks = df_trades_iv["log_moneyness"].to_numpy().astype(float)
    t_Ts = df_trades_iv["T"].to_numpy().astype(float)
    t_T_nearest = _nearest_keys(np.round(t_Ts * 1e6) / 1e6)
    sanos_matched = np.full(len(t_ks), float("nan"))

    for T_key in slice_keys:
        anchor_ks_e, q_star_e, V_atm_e = slice_results[T_key]
        if q_star_e is None:
            continue
        tm = t_T_nearest == T_key
        if not tm.any():
            continue
        batch_ks = t_ks[tm]
        ok_ks = np.isfinite(batch_ks)
        if ok_ks.any():
            idx = np.where(tm)[0][ok_ks]
            sanos_matched[idx] = eval_sanos_iv(
                q_star_e, anchor_ks_e, batch_ks[ok_ks], V_atm_e, float(T_key), cfg.eta
            )

    obs_iv = df_trades_iv["observed_iv"].to_numpy().astype(float)
    trade_residual = obs_iv - sanos_matched

    df_trades_result = df_trades_iv.with_columns(
        [
            pl.Series("sanos_matched", sanos_matched.tolist()),
            pl.Series("trade_residual", trade_residual.tolist()),
        ]
    )

    return df_quotes_result, df_trades_result
