"""
Unified evaluation metrics for all volatility models
=====================================================

One metric set, computed identically for every model:

    * held-out  RMSE   (price space)
    * held-out  NSE    = mean over strikes of |model - mid| / half-spread
    * in-sample RMSE   (price space)
    * in-sample NSE
    * implied negative mass  (Breeden-Litzenberger, butterfly-arbitrage proxy)

All metrics live in **price space** (undiscounted call price C/df, dollar units),
so the IV-based Kalman models must convert their IV predictions to BS call prices
before evaluation (see :func:`bs_call_price` and :func:`to_call_price`).

Two predictions are scored per market observation:

    * ``prior_call``  — model call price from the state *before* assimilating the
                        observation (temporal hold-out / one-step-ahead).
    * ``post_call``   — model call price *after* the update (in-sample fit).

This is the SANOS-EKF "temporal prior/posterior" definition, applied at snapshot
granularity for the SANOS models and at quote granularity for the streaming KF
models (their pre-update prediction is the prior, the post-update is the posterior).

The contract between a model and this module is a single long-format table — see
:func:`evaluate`.  Per-model adapters that produce this table live alongside each
model; :func:`comparison_table` stacks the resulting metric dicts into one frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Black-Scholes price space helpers (for IV-based models)
# ---------------------------------------------------------------------------


def bs_call_price(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Undiscounted Black-Scholes *call* price  C/df = F·N(d1) - K·N(d2).

    Vectorised; returns NaN where inputs are invalid (sigma<=0, T<=0, ...).
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sqrtT = np.sqrt(np.maximum(T, 0.0))
    out = np.full(np.broadcast(F, K, T, sigma).shape, np.nan)
    valid = (
        np.isfinite(sigma)
        & (sigma > 0)
        & np.isfinite(F)
        & (F > 0)
        & np.isfinite(K)
        & (K > 0)
        & (sqrtT > 0)
    )
    if valid.any():
        Fv, Kv, Tv, sv = np.broadcast_arrays(F, K, T, sigma)
        Fv, Kv, Tv, sv = Fv[valid], Kv[valid], Tv[valid], sv[valid]
        sv_sqrtT = sv * np.sqrt(Tv)
        d1 = (np.log(Fv / Kv) + 0.5 * sv**2 * Tv) / sv_sqrtT
        d2 = d1 - sv_sqrtT
        out[valid] = Fv * norm.cdf(d1) - Kv * norm.cdf(d2)
    return out


def to_call_price(
    price: np.ndarray, K: np.ndarray, F: np.ndarray, is_call: np.ndarray
) -> np.ndarray:
    """Convert mixed undiscounted call/put prices to call prices via put-call parity.

    C/df = P/df + F - K  for the rows where ``is_call`` is False.
    """
    price = np.asarray(price, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    return np.where(is_call, price, price + np.asarray(F, float) - np.asarray(K, float))


# ---------------------------------------------------------------------------
# Implied negative mass (butterfly-arbitrage proxy) via Breeden-Litzenberger
# ---------------------------------------------------------------------------


def negative_mass_bl(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    forward: float,
    lo_mult: float = 0.8,
    hi_mult: float = 1.2,
) -> tuple[float, float]:
    """
    Implied negative mass from a *call-price curve* via Breeden-Litzenberger.

        density  p(K) = d²C/dK²   (>= 0 iff no butterfly arbitrage)
        negative mass = sum of the negative part of p   (<= 0; 0 = arb-free)

    Works in pure space (kappa = K/F, C_pure = C/F) with the boundary
    convention used across the project: deep-ITM call slope = -1, deep-OTM
    call slope = 0, far strikes at ``lo_mult``·K_min and ``hi_mult``·K_max.

    Parameters
    ----------
    strikes      : (n,) dollar strikes (need not be sorted; deduplicated here)
    call_prices  : (n,) undiscounted call prices C/df at those strikes
    forward      : forward F (>0); used to map to pure space

    Returns
    -------
    (negative_mass, total_mass) : floats.  total_mass should be ~1.
    """
    strikes = np.asarray(strikes, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)
    F = float(forward)

    ok = np.isfinite(strikes) & np.isfinite(call_prices) & (strikes > 0)
    strikes, call_prices = strikes[ok], call_prices[ok]
    if F <= 0 or len(strikes) < 3:
        return (np.nan, np.nan)

    order = np.argsort(strikes)
    strikes, call_prices = strikes[order], call_prices[order]
    uniq, idx = np.unique(strikes, return_index=True)
    if len(uniq) < 3:
        return (np.nan, np.nan)
    strikes, call_prices = uniq, call_prices[idx]

    kappa = strikes / F
    c_pure = call_prices / F

    ext_k = np.concatenate([[kappa[0] * lo_mult], kappa, [kappa[-1] * hi_mult]])
    ext_c = np.concatenate([[1.0 - kappa[0] * lo_mult], c_pure, [0.0]])

    dCdK = np.diff(ext_c) / np.diff(ext_k)
    dCdK_full = np.concatenate([[-1.0], dCdK, [0.0]])
    p = np.diff(dCdK_full)

    return float(np.sum(np.minimum(p, 0.0))), float(np.sum(p))


def negative_mass_surface(
    strikes: np.ndarray,
    call_prices_2d: np.ndarray,
    forwards: np.ndarray,
    lo_mult: float = 0.8,
    hi_mult: float = 1.2,
) -> np.ndarray:
    """Vectorised :func:`negative_mass_bl` over a reconstructed wide surface.

    For streaming models (single-strike / spline KF) the smile at a sampled
    snapshot time is reconstructed across a *common* sorted strike grid, giving
    a (n_snapshots, n_strikes) call-price matrix.  This returns the per-snapshot
    negative mass (signed, <= 0).  NaNs in a row are not supported — fill the
    grid first (every snapshot must price the whole strike set).
    """
    K = np.asarray(strikes, dtype=float)
    C = np.asarray(call_prices_2d, dtype=float)
    F = np.asarray(forwards, dtype=float).reshape(-1, 1)
    order = np.argsort(K)
    K, C = K[order], C[:, order]

    kappa = K / F  # (n_snap, n_strike)
    c_pure = C / F
    n = C.shape[0]
    ext_k = np.concatenate([kappa[:, :1] * lo_mult, kappa, kappa[:, -1:] * hi_mult], axis=1)
    ext_c = np.concatenate([1.0 - kappa[:, :1] * lo_mult, c_pure, np.zeros((n, 1))], axis=1)
    dCdK = np.diff(ext_c, axis=1) / np.diff(ext_k, axis=1)
    dCdK_full = np.concatenate([-np.ones((n, 1)), dCdK, np.zeros((n, 1))], axis=1)
    p = np.diff(dCdK_full, axis=1)
    return np.sum(np.minimum(p, 0.0), axis=1)


# ---------------------------------------------------------------------------
# Core evaluation: long-format table -> unified metric dict
# ---------------------------------------------------------------------------

# The standard evaluation table has one row per market observation with columns:
#   ts          : snapshot identifier (any hashable; defines the NSE grouping)
#   strike      : dollar strike K
#   forward     : forward F at that snapshot
#   mid_call    : market mid as an undiscounted CALL price (PCP-converted)
#   half_spread : market half-spread in the SAME price units (> 0)
#   prior_call  : model call price from the pre-update (held-out) state
#   post_call   : model call price from the post-update (in-sample) state
STANDARD_COLUMNS = ["ts", "strike", "forward", "mid_call", "half_spread", "prior_call", "post_call"]


def evaluate(
    df: pd.DataFrame,
    *,
    model_name: str = "model",
    ts_col: str = "ts",
    strike_col: str = "strike",
    forward_col: str = "forward",
    mid_col: str = "mid_call",
    hs_col: str = "half_spread",
    prior_col: str = "prior_call",
    post_col: str = "post_call",
    arb_tol: float = -1e-4,
    neg_mass_values: np.ndarray | None = None,
) -> dict:
    """
    Compute the unified metric set from a standard evaluation table.

    NSE and RMSE are aggregated the SANOS way: computed per snapshot (mean /
    RMS over that snapshot's strikes), then averaged across snapshots — reported
    as both the mean and the median over snapshots.

    Negative mass (signed, <= 0; 0 = arbitrage-free):
      * default — computed per snapshot via Breeden-Litzenberger on the
        *posterior* call-price curve (needs multi-strike snapshots, e.g. SANOS EKF).
      * ``neg_mass_values`` — pass a precomputed per-row array to use instead
        (e.g. the gate's native density mass, or a reconstructed surface for the
        streaming KF models, where snapshots are one strike each).

    Returns a flat dict of scalars keyed by ``model`` plus the metrics.
    """
    df = df.copy()
    hs = df[hs_col].to_numpy(dtype=float)
    df["_hs"] = np.where(np.isfinite(hs) & (hs > 0), hs, np.nan)

    if neg_mass_values is not None:
        nmv = np.asarray(neg_mass_values, dtype=float)
        neg_mass = list(nmv[np.isfinite(nmv)])
    else:
        neg_mass = []

    ho_nse, is_nse, ho_rmse, is_rmse = [], [], [], []

    for _, g in df.groupby(ts_col, sort=True):
        mid = g[mid_col].to_numpy(float)
        hsg = g["_hs"].to_numpy(float)
        prior = g[prior_col].to_numpy(float)
        post = g[post_col].to_numpy(float)

        m_prior = np.isfinite(prior) & np.isfinite(mid) & np.isfinite(hsg)
        m_post = np.isfinite(post) & np.isfinite(mid) & np.isfinite(hsg)
        if m_prior.any():
            ho_nse.append(np.mean(np.abs(prior[m_prior] - mid[m_prior]) / hsg[m_prior]))
            ho_rmse.append(np.sqrt(np.mean((prior[m_prior] - mid[m_prior]) ** 2)))
        if m_post.any():
            is_nse.append(np.mean(np.abs(post[m_post] - mid[m_post]) / hsg[m_post]))
            is_rmse.append(np.sqrt(np.mean((post[m_post] - mid[m_post]) ** 2)))

        if neg_mass_values is None:
            fwd = float(g[forward_col].iloc[0])
            nm, _ = negative_mass_bl(g[strike_col].to_numpy(float), post, fwd)
            if np.isfinite(nm):
                neg_mass.append(nm)

    def _agg(x):
        x = np.asarray(x, float)
        if not len(x):
            return (np.nan, np.nan, np.nan)
        return (float(np.nanmean(x)), float(np.nanmedian(x)), float(np.nanstd(x)))

    ho_nse_mean, ho_nse_med, ho_nse_std = _agg(ho_nse)
    is_nse_mean, is_nse_med, is_nse_std = _agg(is_nse)
    ho_rmse_mean, ho_rmse_med, ho_rmse_std = _agg(ho_rmse)
    is_rmse_mean, is_rmse_med, is_rmse_std = _agg(is_rmse)
    nm_mean, nm_med, _ = _agg(neg_mass)
    arb_frac = float(np.mean(np.asarray(neg_mass) < arb_tol)) if neg_mass else np.nan

    return {
        "model": model_name,
        "ho_nse_mean": ho_nse_mean,
        "ho_nse_med": ho_nse_med,
        "ho_nse_std": ho_nse_std,
        "is_nse_mean": is_nse_mean,
        "is_nse_med": is_nse_med,
        "is_nse_std": is_nse_std,
        "ho_rmse_mean": ho_rmse_mean,
        "ho_rmse_med": ho_rmse_med,
        "ho_rmse_std": ho_rmse_std,
        "is_rmse_mean": is_rmse_mean,
        "is_rmse_med": is_rmse_med,
        "is_rmse_std": is_rmse_std,
        "neg_mass_mean": nm_mean,
        "neg_mass_med": nm_med,
        "arb_frac": arb_frac,
        "n_snapshots": int(len(ho_nse)),
    }


def from_option_prices(
    *,
    ts,
    strike,
    forward,
    is_call,
    bid,
    ask,
    prior_price,
    post_price,
    discount_factor=None,
) -> pd.DataFrame:
    """Assemble a standard eval table from raw option (call OR put) prices.

    Everything is converted to undiscounted CALL price space (C/df) via put-call
    parity, so call and put quotes mix freely.  Use for the streaming IV Kalman
    models, whose ``kalman_price`` columns are native-type BS prices:

      * spline KF      — prior_price = kalman_price_prior, post_price = kalman_price
      * single-strike  — post_price = kalman_price, prior_price = kalman_price
                         shifted one quote back *within each strike* (random-walk
                         EKF => one-step-ahead prior == previous posterior).

    ``bid``/``ask`` are raw market option prices; pass ``discount_factor`` to
    undiscount them (defaults to 1).  Negative mass is NOT computed here (these
    snapshots are one strike each) — reconstruct a surface and use
    :func:`negative_mass_surface`, then pass it to :func:`evaluate` as
    ``neg_mass_values``.
    """
    ts = np.asarray(ts)
    K = np.asarray(strike, float)
    F = np.asarray(forward, float)
    ic = np.asarray(is_call, bool)
    df_ = np.asarray(discount_factor, float) if discount_factor is not None else np.ones(len(K))
    df_ = np.where(np.isfinite(df_) & (df_ > 0), df_, 1.0)

    bid_c = to_call_price(np.asarray(bid, float) / df_, K, F, ic)
    ask_c = to_call_price(np.asarray(ask, float) / df_, K, F, ic)
    return pd.DataFrame(
        {
            "ts": ts,
            "strike": K,
            "forward": F,
            "mid_call": 0.5 * (bid_c + ask_c),
            "half_spread": np.maximum(0.5 * (ask_c - bid_c), 1e-10),
            "prior_call": to_call_price(np.asarray(prior_price, float), K, F, ic),
            "post_call": to_call_price(np.asarray(post_price, float), K, F, ic),
        }
    )


def comparison_table(metric_dicts: list[dict]) -> pd.DataFrame:
    """Stack per-model metric dicts (from :func:`evaluate`) into one comparison frame."""
    cols = [
        "model",
        "ho_nse_mean",
        "ho_nse_med",
        "ho_nse_std",
        "is_nse_mean",
        "is_nse_med",
        "is_nse_std",
        "ho_rmse_mean",
        "ho_rmse_med",
        "ho_rmse_std",
        "is_rmse_mean",
        "is_rmse_med",
        "is_rmse_std",
        "neg_mass_mean",
        "neg_mass_med",
        "arb_frac",
        "n_snapshots",
    ]
    df = pd.DataFrame(metric_dicts)
    return df[[c for c in cols if c in df.columns]].set_index("model")
