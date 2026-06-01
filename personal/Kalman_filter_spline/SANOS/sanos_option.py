"""
sanos_option.py

SANOS LP calibration directly on option prices — no IV roundtrip.

Calls and puts are fitted by independent LP solves per expiry.
Forward and discount factor come from the 2×2 put-call parity system
(build_implied_forward).  The only IV inversion in the pipeline is one
scalar BS inversion at the ATM mid-price to set V_atm (the η-smoothing
bandwidth) — all bid/ask bounds feed into the LP as normalized prices.

Public API
----------
SanosOptionConfig         — configuration dataclass
fit_sanos_call_slice()    — LP fit on normalized call prices (one expiry)
fit_sanos_put_slice()     — LP fit on normalized put prices (one expiry)
eval_sanos_call_prices()  — evaluate fitted call prices
eval_sanos_put_prices()   — evaluate fitted put prices
run_sanos_option_batch()  — whole-day batch pipeline (price space)
build_option_observations() — fetch → match → forward → tag (data pipeline)

Key differences from sanos_surface.py
--------------------------------------
| Aspect             | sanos_surface.py              | sanos_option.py                 |
|--------------------|-------------------------------|---------------------------------|
| LP input bounds    | bid_iv/ask_iv → price via BS  | bid_price/ask_price directly    |
| V_atm              | geometric mean of ATM IVs     | single BS inversion at ATM mid  |
| Call/put handling  | OTM convention, one LP        | separate LP per type            |
| Put design matrix  | not needed                    | A_put = A_call − exp(a) + exp(k)|
| Output             | sanos_iv                      | sanos_price (dollars)           |
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import polars as pl
from scipy.optimize import brentq
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SanosOptionConfig:
    N: int = 30
    eta: float = 0.25
    k_min: float = -0.25
    k_max: float = 0.15
    eps_obj: float = 1e-8
    solver: str = "CLARABEL"
    discount_factor_clip: tuple[float, float] | None = (0.998, 1.002)
    max_leg_staleness_s: float = 60.0
    otm_calls_only: bool = True  # restrict call LP to k >= 0
    otm_puts_only: bool = True  # restrict put LP to k < 0


# ---------------------------------------------------------------------------
# BS primitives (duplicated from sanos_surface for self-containedness)
# ---------------------------------------------------------------------------


def _bs_call_norm_matrix(anchor_ks: np.ndarray, eval_ks: np.ndarray, v: float) -> np.ndarray:
    """
    (M, N) design matrix.  A[i, j] = BS_call(anchor=anchor_ks[j], strike=eval_ks[i], var=v).
    Vectorized via numpy broadcasting.
    """
    v = max(float(v), 1e-10)
    sqrt_v = np.sqrt(v)
    a = anchor_ks[np.newaxis, :]  # (1, N)
    m = eval_ks[:, np.newaxis]  # (M, 1)
    d1 = (a - m + v / 2.0) / sqrt_v
    d2 = d1 - sqrt_v
    return np.exp(a) * norm.cdf(d1) - np.exp(m) * norm.cdf(d2)


def _bs_call_norm_single(market_k: float, T: float, iv: float) -> float:
    iv = max(float(iv), 1e-8)
    T = max(float(T), 1e-8)
    v = iv * iv * T
    d1 = (-market_k + 0.5 * v) / (iv * np.sqrt(T))
    return float(norm.cdf(d1) - np.exp(market_k) * norm.cdf(d1 - iv * np.sqrt(T)))


def _iv_from_norm_call(price: float, market_k: float, T: float) -> float:
    """Invert BS to recover implied vol from a normalized call price."""
    intrinsic = max(0.0, 1.0 - np.exp(market_k))
    if price <= intrinsic or price >= 1.0:
        return float("nan")
    try:
        return brentq(
            lambda iv: _bs_call_norm_single(market_k, T, iv) - price, 1e-6, 5.0, xtol=1e-8
        )
    except ValueError:
        return float("nan")


def _iv_from_norm_put(price: float, market_k: float, T: float) -> float:
    """Invert BS to recover implied vol from a normalized put price."""
    c = price + 1.0 - np.exp(market_k)  # call price via put-call parity
    return _iv_from_norm_call(c, market_k, T)


# ---------------------------------------------------------------------------
# Put design matrix
# ---------------------------------------------------------------------------


def _bs_put_norm_matrix(anchor_ks: np.ndarray, eval_ks: np.ndarray, v: float) -> np.ndarray:
    """
    (M, N) design matrix for put prices.

        A_put[i, j] = BS_put(anchor=anchor_ks[j], strike=eval_ks[i], var=v)
                    = A_call[i, j] − exp(anchor_ks[j]) + exp(eval_ks[i])

    At the LP solution the martingale constraint (Σ q_j exp(anchor_j) = 1) collapses
    this to p_i = c_i − 1 + exp(k_i), but the full matrix is required during solving.
    """
    A_call = _bs_call_norm_matrix(anchor_ks, eval_ks, v)
    return A_call - np.exp(anchor_ks)[np.newaxis, :] + np.exp(eval_ks)[:, np.newaxis]


# ---------------------------------------------------------------------------
# V_atm from a single near-ATM price
# ---------------------------------------------------------------------------


def _v_atm_from_price(mid_norm: float, k_atm: float, T: float, is_call: bool) -> float:
    """
    Single BS inversion at the near-ATM mid-price to estimate V_atm = σ²T.
    Falls back to (20% vol)²T if the inversion fails.
    """
    T = max(float(T), 1e-8)
    iv = (
        _iv_from_norm_call(mid_norm, k_atm, T) if is_call else _iv_from_norm_put(mid_norm, k_atm, T)
    )
    if not np.isfinite(iv) or iv <= 0:
        return 0.04 * T  # 20% vol fallback
    return iv * iv * T


# ---------------------------------------------------------------------------
# Single-expiry LP: calls
# ---------------------------------------------------------------------------


def fit_sanos_call_slice(
    market_ks: np.ndarray,
    bid_norm: np.ndarray,
    ask_norm: np.ndarray,
    T: float,
    cfg: SanosOptionConfig,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """
    Fit SANOS LP for call prices (one expiry).

    Parameters
    ----------
    market_ks : (M,) log-moneyness of market quotes
    bid_norm  : (M,) normalized bid call prices  (= bid_price / (F * df))
    ask_norm  : (M,) normalized ask call prices
    T         : time-to-expiry in years
    cfg       : SanosOptionConfig

    Returns
    -------
    anchor_ks : (N,) log-moneyness of anchor strikes
    q_star    : (N,) fitted probability weights, or None if LP failed
    V_atm     : ATM total variance used for η-scaling
    """
    market_ks = np.asarray(market_ks, dtype=float)
    bid_norm = np.asarray(bid_norm, dtype=float)
    ask_norm = np.asarray(ask_norm, dtype=float)

    atm_idx = int(np.argmin(np.abs(market_ks)))
    mid_atm = (bid_norm[atm_idx] + ask_norm[atm_idx]) / 2.0
    V_atm = _v_atm_from_price(mid_atm, float(market_ks[atm_idx]), T, is_call=True)

    valid = np.isfinite(bid_norm) & np.isfinite(ask_norm) & (ask_norm > bid_norm)
    if valid.sum() < 3:
        warnings.warn(f"SANOS calls: too few valid quotes ({valid.sum()}); LP skipped.")
        return np.linspace(cfg.k_min, cfg.k_max, cfg.N), None, V_atm

    bid_v = bid_norm[valid]
    ask_v = ask_norm[valid]
    mid_v = (bid_v + ask_v) / 2.0
    w = 1.0 / np.maximum((ask_v - bid_v) / 2.0, 1e-6)
    M_v = int(valid.sum())

    N_eff = max(5, min(cfg.N, M_v))
    anchor_ks = np.linspace(cfg.k_min, cfg.k_max, N_eff)
    anchor_K = np.exp(anchor_ks)
    v_smooth = max(cfg.eta * V_atm, 1e-12)
    A = _bs_call_norm_matrix(anchor_ks, market_ks[valid], v_smooth)  # (M_v, N_eff)

    q = cp.Variable(N_eff, nonneg=True)
    over = cp.Variable(M_v, nonneg=True)
    under = cp.Variable(M_v, nonneg=True)
    c_mod = A @ q

    prob = cp.Problem(
        cp.Minimize(w @ (cfg.eps_obj * cp.abs(c_mod - mid_v) + over + under)),
        [
            cp.sum(q) == 1.0,
            anchor_K @ q == 1.0,
            over >= c_mod - ask_v,
            under >= bid_v - c_mod,
        ],
    )
    try:
        prob.solve(solver=cfg.solver, verbose=False)
    except cp.SolverError as e:
        warnings.warn(f"SANOS calls: solver error — {e}")
        return anchor_ks, None, V_atm

    if prob.status not in ("optimal", "optimal_inaccurate"):
        warnings.warn(f"SANOS calls: LP status '{prob.status}'")
        return anchor_ks, None, V_atm
    if prob.status == "optimal_inaccurate":
        warnings.warn("SANOS calls: 'optimal_inaccurate'; results may be unreliable.")

    q_star = np.maximum(q.value, 0.0)
    q_star /= q_star.sum()
    return anchor_ks, q_star, V_atm


# ---------------------------------------------------------------------------
# Single-expiry LP: puts
# ---------------------------------------------------------------------------


def fit_sanos_put_slice(
    market_ks: np.ndarray,
    bid_norm: np.ndarray,
    ask_norm: np.ndarray,
    T: float,
    cfg: SanosOptionConfig,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """
    Fit SANOS LP for put prices (one expiry).

    Uses the put design matrix A_put = A_call − exp(anchor) + exp(k).
    The martingale constraint acts on q via anchor_K @ q = 1 (unchanged).

    Parameters
    ----------
    market_ks : (M,) log-moneyness (typically k < 0 for OTM puts)
    bid_norm  : (M,) normalized bid put prices  (= bid_price / (F * df))
    ask_norm  : (M,) normalized ask put prices
    T         : time-to-expiry in years
    cfg       : SanosOptionConfig

    Returns
    -------
    anchor_ks : (N,) log-moneyness of anchor strikes
    q_star    : (N,) fitted probability weights, or None if LP failed
    V_atm     : ATM total variance
    """
    market_ks = np.asarray(market_ks, dtype=float)
    bid_norm = np.asarray(bid_norm, dtype=float)
    ask_norm = np.asarray(ask_norm, dtype=float)

    atm_idx = int(np.argmin(np.abs(market_ks)))
    mid_atm = (bid_norm[atm_idx] + ask_norm[atm_idx]) / 2.0
    V_atm = _v_atm_from_price(mid_atm, float(market_ks[atm_idx]), T, is_call=False)

    valid = np.isfinite(bid_norm) & np.isfinite(ask_norm) & (ask_norm > bid_norm)
    if valid.sum() < 3:
        warnings.warn(f"SANOS puts: too few valid quotes ({valid.sum()}); LP skipped.")
        return np.linspace(cfg.k_min, cfg.k_max, cfg.N), None, V_atm

    bid_v = bid_norm[valid]
    ask_v = ask_norm[valid]
    mid_v = (bid_v + ask_v) / 2.0
    w = 1.0 / np.maximum((ask_v - bid_v) / 2.0, 1e-6)
    M_v = int(valid.sum())
    mks_v = market_ks[valid]

    N_eff = max(5, min(cfg.N, M_v))
    anchor_ks = np.linspace(cfg.k_min, cfg.k_max, N_eff)
    anchor_K = np.exp(anchor_ks)
    v_smooth = max(cfg.eta * V_atm, 1e-12)
    A = _bs_put_norm_matrix(anchor_ks, mks_v, v_smooth)  # (M_v, N_eff)

    q = cp.Variable(N_eff, nonneg=True)
    over = cp.Variable(M_v, nonneg=True)
    under = cp.Variable(M_v, nonneg=True)
    p_mod = A @ q

    prob = cp.Problem(
        cp.Minimize(w @ (cfg.eps_obj * cp.abs(p_mod - mid_v) + over + under)),
        [
            cp.sum(q) == 1.0,
            anchor_K @ q == 1.0,
            over >= p_mod - ask_v,
            under >= bid_v - p_mod,
        ],
    )
    try:
        prob.solve(solver=cfg.solver, verbose=False)
    except cp.SolverError as e:
        warnings.warn(f"SANOS puts: solver error — {e}")
        return anchor_ks, None, V_atm

    if prob.status not in ("optimal", "optimal_inaccurate"):
        warnings.warn(f"SANOS puts: LP status '{prob.status}'")
        return anchor_ks, None, V_atm
    if prob.status == "optimal_inaccurate":
        warnings.warn("SANOS puts: 'optimal_inaccurate'; results may be unreliable.")

    q_star = np.maximum(q.value, 0.0)
    q_star /= q_star.sum()
    return anchor_ks, q_star, V_atm


# ---------------------------------------------------------------------------
# Surface evaluation
# ---------------------------------------------------------------------------


def eval_sanos_call_prices(
    q: np.ndarray,
    anchor_ks: np.ndarray,
    eval_ks: np.ndarray,
    V_atm: float,
    eta: float,
) -> np.ndarray:
    """(M,) normalized call prices: A_call @ q."""
    v_smooth = max(eta * V_atm, 1e-12)
    A = _bs_call_norm_matrix(anchor_ks, np.asarray(eval_ks, dtype=float), v_smooth)
    return A @ q


def eval_sanos_put_prices(
    q: np.ndarray,
    anchor_ks: np.ndarray,
    eval_ks: np.ndarray,
    V_atm: float,
    eta: float,
) -> np.ndarray:
    """(M,) normalized put prices: A_put @ q."""
    v_smooth = max(eta * V_atm, 1e-12)
    A = _bs_put_norm_matrix(anchor_ks, np.asarray(eval_ks, dtype=float), v_smooth)
    return A @ q


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------


def run_sanos_option_batch(
    df_quotes: pl.DataFrame,
    df_trades: pl.DataFrame,
    cfg: SanosOptionConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Batch SANOS calibration directly on option prices.

    Calls and puts are fitted by separate LP solves per expiry, in parallel.
    LP bounds come from normalized bid/ask prices — no IV conversion.

    Parameters
    ----------
    df_quotes : columns ts_ns, is_call, log_moneyness, T, bid_price, ask_price,
                observed_price, forward, discount_factor (optional, defaults to 1)
    df_trades : same schema (observed_price = trade price)
    cfg       : SanosOptionConfig

    Returns
    -------
    df_quotes_result : df_quotes + column sanos_price
    df_trades_result : df_trades + columns sanos_matched, trade_residual
    """
    has_df_q = "discount_factor" in df_quotes.columns
    has_df_t = "discount_factor" in df_trades.columns

    # ── Normalize prices by F*df ──────────────────────────────────────────────
    def _norm_scale(df: pl.DataFrame, has_df: bool) -> pl.Expr:
        return pl.col("forward") * pl.col("discount_factor") if has_df else pl.col("forward")

    df_q = df_quotes.with_columns(
        (pl.col("bid_price") / _norm_scale(df_quotes, has_df_q)).alias("bid_norm"),
        (pl.col("ask_price") / _norm_scale(df_quotes, has_df_q)).alias("ask_norm"),
        (pl.col("observed_price") / _norm_scale(df_quotes, has_df_q)).alias("mid_norm"),
    )

    # ── OTM filter for LP fitting ─────────────────────────────────────────────
    df_otm_calls = df_q.filter(
        pl.col("is_call") & (pl.col("log_moneyness") >= 0.0 if cfg.otm_calls_only else pl.lit(True))
    )
    df_otm_puts = df_q.filter(
        (~pl.col("is_call"))
        & (pl.col("log_moneyness") < 0.0 if cfg.otm_puts_only else pl.lit(True))
    )

    # ── Unique intraday snapshots ─────────────────────────────────────────────
    # 1e6 precision: one LP per unique timestamp snapshot (~N_strikes quotes each).
    # This mirrors sanos_surface and gives the intraday evolution of the fit.
    T_round_expr = (pl.col("T") * 1e6).round(0) / 1e6

    def _expiry_T_vals(df: pl.DataFrame) -> list[float]:
        return (
            df.with_columns(T_round_expr.alias("_T_round"))
            .group_by("_T_round")
            .agg(pl.col("T").first())
            .sort("_T_round")["T"]
            .to_list()
        )

    call_expiries = _expiry_T_vals(df_otm_calls)
    put_expiries = _expiry_T_vals(df_otm_puts)

    # ── Build LP task lists ───────────────────────────────────────────────────
    def _make_tasks(df_otm: pl.DataFrame, expiry_vals: list[float]) -> list[tuple]:
        tasks = []
        for T_exp in expiry_vals:
            T_r = round(T_exp * 1e6) / 1e6
            grp = df_otm.filter(((pl.col("T") * 1e6).round(0) / 1e6) == T_r)
            mks = grp["log_moneyness"].to_numpy().astype(float)
            bids = grp["bid_norm"].to_numpy().astype(float)
            asks = grp["ask_norm"].to_numpy().astype(float)
            ok = np.isfinite(mks) & np.isfinite(bids) & np.isfinite(asks)
            tasks.append((T_exp, T_r, mks[ok], bids[ok], asks[ok], int(ok.sum())))
        return tasks

    call_tasks = _make_tasks(df_otm_calls, call_expiries)
    put_tasks = _make_tasks(df_otm_puts, put_expiries)

    # ── Parallel LP solves — calls and puts interleaved ───────────────────────
    call_results: dict[float, tuple] = {}
    put_results: dict[float, tuple] = {}

    def _fit_call(args: tuple) -> tuple:
        T_exp, T_r, mks, bids, asks, n_ok = args
        if n_ok < 3:
            return "call", T_r, (np.linspace(cfg.k_min, cfg.k_max, cfg.N), None, 0.0), T_exp, n_ok
        return "call", T_r, fit_sanos_call_slice(mks, bids, asks, T_exp, cfg), T_exp, n_ok

    def _fit_put(args: tuple) -> tuple:
        T_exp, T_r, mks, bids, asks, n_ok = args
        if n_ok < 3:
            return "put", T_r, (np.linspace(cfg.k_min, cfg.k_max, cfg.N), None, 0.0), T_exp, n_ok
        return "put", T_r, fit_sanos_put_slice(mks, bids, asks, T_exp, cfg), T_exp, n_ok

    n_workers = min(len(call_tasks) + len(put_tasks), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        call_futs = [pool.submit(_fit_call, a) for a in call_tasks]
        put_futs = [pool.submit(_fit_put, a) for a in put_tasks]

        for fut in call_futs:
            _, T_r, res, T_exp, n_ok = fut.result()
            anchor_ks, q_star, V_atm = res
            mart_err = abs(np.exp(anchor_ks) @ q_star - 1.0) if q_star is not None else float("nan")
            status = "OK" if q_star is not None else "FAILED"
            print(
                f"SANOS calls: T={T_exp:.4f}  {status}  n={n_ok}  V_atm={V_atm:.5f}  mart_err={mart_err:.2e}"
            )
            call_results[T_r] = res

        for fut in put_futs:
            _, T_r, res, T_exp, n_ok = fut.result()
            anchor_ks, q_star, V_atm = res
            mart_err = abs(np.exp(anchor_ks) @ q_star - 1.0) if q_star is not None else float("nan")
            status = "OK" if q_star is not None else "FAILED"
            print(
                f"SANOS puts:  T={T_exp:.4f}  {status}  n={n_ok}  V_atm={V_atm:.5f}  mart_err={mart_err:.2e}"
            )
            put_results[T_r] = res

    # ── Nearest-expiry lookup (vectorized) ────────────────────────────────────
    def _nearest_key(T_rounds: np.ndarray, keys: np.ndarray) -> np.ndarray:
        if len(keys) == 0:
            return np.full(len(T_rounds), np.nan)
        diffs = np.abs(T_rounds[:, np.newaxis] - keys[np.newaxis, :])
        return keys[np.argmin(diffs, axis=1)]

    call_keys = np.array(sorted(call_results))
    put_keys = np.array(sorted(put_results))

    # ── Evaluate at quotes — one call per expiry ──────────────────────────────
    q_ks = df_q["log_moneyness"].to_numpy().astype(float)
    q_Ts = df_q["T"].to_numpy().astype(float)
    q_fwd = df_quotes["forward"].to_numpy().astype(float)
    q_df_arr = (
        df_quotes["discount_factor"].to_numpy().astype(float)
        if has_df_q
        else np.ones(len(df_quotes))
    )
    q_is_call = df_q["is_call"].to_numpy().astype(bool)
    q_Tr = np.round(q_Ts * 1e6) / 1e6

    sanos_norm_q = np.full(len(q_ks), np.nan)

    if len(call_keys) > 0:
        q_call_idx = np.where(q_is_call)[0]
        q_call_Tr = _nearest_key(q_Tr[q_call_idx], call_keys)
        for T_key in call_keys:
            a_ks, q_star, V_atm = call_results[T_key]
            if q_star is None:
                continue
            mask = q_call_Tr == T_key
            if not mask.any():
                continue
            idx = q_call_idx[mask]
            ok = np.isfinite(q_ks[idx])
            if ok.any():
                sanos_norm_q[idx[ok]] = eval_sanos_call_prices(
                    q_star, a_ks, q_ks[idx[ok]], V_atm, cfg.eta
                )

    if len(put_keys) > 0:
        q_put_idx = np.where(~q_is_call)[0]
        q_put_Tr = _nearest_key(q_Tr[q_put_idx], put_keys)
        for T_key in put_keys:
            a_ks, q_star, V_atm = put_results[T_key]
            if q_star is None:
                continue
            mask = q_put_Tr == T_key
            if not mask.any():
                continue
            idx = q_put_idx[mask]
            ok = np.isfinite(q_ks[idx])
            if ok.any():
                sanos_norm_q[idx[ok]] = eval_sanos_put_prices(
                    q_star, a_ks, q_ks[idx[ok]], V_atm, cfg.eta
                )

    scale_q = q_fwd * q_df_arr
    sanos_price_q = np.where(np.isfinite(sanos_norm_q), sanos_norm_q * scale_q, np.nan)

    df_quotes_result = df_quotes.with_columns(pl.Series("sanos_price", sanos_price_q.tolist()))

    # ── Evaluate at trades ────────────────────────────────────────────────────
    if df_trades.is_empty():
        return df_quotes_result, df_trades.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("sanos_matched"),
                pl.lit(None, dtype=pl.Float64).alias("trade_residual"),
            ]
        )

    df_t_norm = df_trades.with_columns(
        (pl.col("observed_price") / _norm_scale(df_trades, has_df_t)).alias("mid_norm"),
    )

    t_ks = df_t_norm["log_moneyness"].to_numpy().astype(float)
    t_Ts = df_t_norm["T"].to_numpy().astype(float)
    t_fwd = df_trades["forward"].to_numpy().astype(float)
    t_df_arr = (
        df_trades["discount_factor"].to_numpy().astype(float)
        if has_df_t
        else np.ones(len(df_trades))
    )
    t_is_call = df_t_norm["is_call"].to_numpy().astype(bool)
    t_Tr = np.round(t_Ts * 1e6) / 1e6

    sanos_norm_t = np.full(len(t_ks), np.nan)

    if len(call_keys) > 0:
        t_call_idx = np.where(t_is_call)[0]
        t_call_Tr = _nearest_key(t_Tr[t_call_idx], call_keys)
        for T_key in call_keys:
            a_ks, q_star, V_atm = call_results[T_key]
            if q_star is None:
                continue
            mask = t_call_Tr == T_key
            if not mask.any():
                continue
            idx = t_call_idx[mask]
            ok = np.isfinite(t_ks[idx])
            if ok.any():
                sanos_norm_t[idx[ok]] = eval_sanos_call_prices(
                    q_star, a_ks, t_ks[idx[ok]], V_atm, cfg.eta
                )

    if len(put_keys) > 0:
        t_put_idx = np.where(~t_is_call)[0]
        t_put_Tr = _nearest_key(t_Tr[t_put_idx], put_keys)
        for T_key in put_keys:
            a_ks, q_star, V_atm = put_results[T_key]
            if q_star is None:
                continue
            mask = t_put_Tr == T_key
            if not mask.any():
                continue
            idx = t_put_idx[mask]
            ok = np.isfinite(t_ks[idx])
            if ok.any():
                sanos_norm_t[idx[ok]] = eval_sanos_put_prices(
                    q_star, a_ks, t_ks[idx[ok]], V_atm, cfg.eta
                )

    scale_t = t_fwd * t_df_arr
    sanos_matched = np.where(np.isfinite(sanos_norm_t), sanos_norm_t * scale_t, np.nan)
    obs_price = df_trades["observed_price"].to_numpy().astype(float)
    trade_residual = obs_price - sanos_matched

    df_trades_result = df_trades.with_columns(
        [
            pl.Series("sanos_matched", sanos_matched.tolist()),
            pl.Series("trade_residual", trade_residual.tolist()),
        ]
    )

    return df_quotes_result, df_trades_result


# ---------------------------------------------------------------------------
# Data pipeline entry point
# ---------------------------------------------------------------------------


def build_option_observations(
    df_tickers: pl.DataFrame,
    trade_date,
    df_spy: pl.DataFrame,
    min_trades: int = 1,
    taq_config=None,
    simple: bool = False,
    cfg: SanosOptionConfig | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Pipeline phases 1–4: fetch → match → forward estimation → tag with T and forward.

    Parameters
    ----------
    df_tickers : option tickers DataFrame with columns ticker, trade_count
    trade_date : date to fetch
    df_spy     : SPY trades DataFrame (ts_ns, price)
    min_trades : minimum trades required per ticker
    taq_config : TAQMatchConfig (uses defaults if None)
    simple     : if True, use spot as forward with discount_factor=1 (skip 2×2 PCP system)
    cfg        : SanosOptionConfig (uses defaults if None)

    Returns
    -------
    (df_quotes, df_trades) with columns:
        ts_ns, strike, is_call, log_moneyness, T,
        observed_price, bid_price, ask_price, forward, discount_factor
    """
    import personal.shared_functions.events as events
    import personal.shared_functions.utils as utils
    from personal.shared_functions.kalman_filters import _add_spot_T, build_implied_forward
    from quantdata.massive.client import MassiveClient
    from quantdata.massive.taq import TAQMatchConfig, match_taq

    if cfg is None:
        cfg = SanosOptionConfig()
    if taq_config is None:
        taq_config = TAQMatchConfig()

    client = MassiveClient()
    spy_ts = events.filter_df(df_spy).select(["ts_ns", "price"]).rename({"price": "spot"})
    candidates = df_tickers.filter(pl.col("trade_count") >= min_trades)

    _bad_expr = (
        (pl.col("bid_price") >= pl.col("ask_price"))
        | (pl.col("bid_price") < 0.011)
        | (pl.col("bid_size") <= 0)
        | pl.col("bid_size").is_null()
        | (pl.col("ask_size") <= 0)
        | pl.col("ask_size").is_null()
    )

    ticker_meta: list[tuple] = []
    quote_frames: list[pl.DataFrame] = []
    trade_frames: list[pl.DataFrame] = []
    raw_calls_q: dict[tuple, pl.DataFrame] = {}
    raw_puts_q: dict[tuple, pl.DataFrame] = {}

    for row in candidates.iter_rows(named=True):
        ticker = row["ticker"]
        try:
            strike, expiry, otype = utils.parse_option_ticker(ticker)
            is_call = otype == "C"

            df_quotes_raw = client.fetch_quotes(ticker, trade_date)
            df_matched = match_taq(ticker, trade_date, config=taq_config)

            if df_quotes_raw.is_empty():
                print(f"Skipping {ticker}: no quotes")
                continue

            bad = df_quotes_raw.filter(_bad_expr)
            if bad.height >= 10:
                print(f"Skipping {ticker}: {bad.height} bad quotes")
                continue
            if bad.height > 0:
                df_quotes_raw = df_quotes_raw.filter(~_bad_expr)

            df_matched_valid = df_matched.filter(pl.col("quote_ts_ns").is_not_null())
            if df_matched_valid.is_empty():
                print(f"Skipping {ticker}: no matched trades")
                continue

            quote_frames.append(events.process_quotes(df_quotes_raw))
            trade_frames.append(df_matched_valid)
            ticker_meta.append((strike, is_call, expiry))

            key = (expiry, strike)
            if is_call:
                raw_calls_q[key] = df_quotes_raw
            else:
                raw_puts_q[key] = df_quotes_raw

        except Exception as exc:
            print(f"Skipping {ticker}: {exc}")

    if not ticker_meta:
        raise RuntimeError("No options produced valid data")

    # ── Implied forward per expiry ────────────────────────────────────────────
    expiries_seen = sorted({exp for _, _, exp in ticker_meta})
    forward_by_expiry: dict = {}
    _fallback_fwd = spy_ts.rename({"spot": "forward"}).with_columns(
        pl.lit(1.0).alias("discount_factor")
    )

    for exp in expiries_seen:
        calls_at = {k: df for (e, k), df in raw_calls_q.items() if e == exp}
        puts_at = {k: df for (e, k), df in raw_puts_q.items() if e == exp}
        if set(calls_at) & set(puts_at):
            try:
                forward_by_expiry[exp] = build_implied_forward(
                    calls_at,
                    puts_at,
                    spy_ts=spy_ts,
                    simple=simple,
                    discount_factor_clip=cfg.discount_factor_clip,
                    max_leg_staleness_s=cfg.max_leg_staleness_s,
                )
                n_ovlp = len(set(calls_at) & set(puts_at))
                print(f"Expiry {exp.date()}: forward from {n_ovlp} overlapping strike(s)")
            except Exception as exc:
                print(f"Expiry {exp.date()}: forward failed ({exc}), using spot")
                forward_by_expiry[exp] = _fallback_fwd
        else:
            print(f"Expiry {exp.date()}: no overlapping strikes, using spot as forward")
            forward_by_expiry[exp] = _fallback_fwd

    # ── Tag with spot, T, forward, discount_factor ────────────────────────────
    tagged_quotes: list[pl.DataFrame] = []
    tagged_trades: list[pl.DataFrame] = []
    for i, (strike, is_call, expiry) in enumerate(ticker_meta):
        tags = [pl.lit(strike).alias("_strike"), pl.lit(is_call).alias("is_call")]
        fwd_ts = forward_by_expiry[expiry]
        tagged_quotes.append(
            _add_spot_T(quote_frames[i], spy_ts, expiry, fwd_ts).with_columns(tags)
        )
        tagged_trades.append(
            _add_spot_T(trade_frames[i], spy_ts, expiry, fwd_ts).with_columns(tags)
        )

    all_q = pl.concat(tagged_quotes)
    all_t = pl.concat(tagged_trades)

    # ── Ensure discount_factor is present (fill null / add if missing) ────────
    for df_name, df_ref in [("quotes", all_q), ("trades", all_t)]:
        if "discount_factor" not in df_ref.columns:
            if df_name == "quotes":
                all_q = all_q.with_columns(pl.lit(1.0).alias("discount_factor"))
            else:
                all_t = all_t.with_columns(pl.lit(1.0).alias("discount_factor"))
    all_q = all_q.with_columns(pl.col("discount_factor").fill_null(1.0))
    all_t = all_t.with_columns(pl.col("discount_factor").fill_null(1.0))

    # ── Add log_moneyness = log(K / F) ────────────────────────────────────────
    def _add_log_moneyness(df: pl.DataFrame) -> pl.DataFrame:
        fwd = df["forward"].to_numpy().astype(float)
        k = df["_strike"].to_numpy().astype(float)
        log_m = np.where((fwd > 0) & np.isfinite(fwd) & (k > 0), np.log(k / fwd), np.nan)
        return df.with_columns(pl.Series("log_moneyness", log_m.tolist()))

    all_q = _add_log_moneyness(all_q).rename({"_strike": "strike"})
    all_t = _add_log_moneyness(all_t).rename({"_strike": "strike"})

    # ── Normalize trade price column name ─────────────────────────────────────
    if "price" in all_t.columns and "observed_price" not in all_t.columns:
        all_t = all_t.rename({"price": "observed_price"})

    # ── Select output columns ─────────────────────────────────────────────────
    out_cols = [
        "ts_ns",
        "strike",
        "is_call",
        "log_moneyness",
        "T",
        "observed_price",
        "bid_price",
        "ask_price",
        "forward",
        "discount_factor",
    ]
    all_q = all_q.select([c for c in out_cols if c in all_q.columns]).sort("ts_ns")
    all_t = all_t.select([c for c in out_cols if c in all_t.columns]).sort("ts_ns")

    return all_q, all_t
