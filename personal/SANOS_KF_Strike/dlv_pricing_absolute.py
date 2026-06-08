"""
Option pricing and IV inversion for single-expiry SANOS.
Absolute-strike variant: prices are C/df (not C/(F*df)); strikes are dollars.

Put-call parity: C/df = P/df + F - K
BS inversion: normalize price by fwd before calling the BS formula.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# ---------------------------------------------------------------------------
# SANOS call / put pricing from density
# ---------------------------------------------------------------------------


def call_prices(
    density: np.ndarray,
    grid_K: np.ndarray,
    query_K: np.ndarray,
) -> np.ndarray:
    """
    Undiscounted call prices C/df under the fitted density.

        C/df(K) = sum_i max(K_i - K, 0) * d_i

    Parameters
    ----------
    density : (N,) risk-neutral density over grid_K
    grid_K  : (N,) interior anchor strikes (dollars)
    query_K : (M,) query strikes (dollars)

    Returns
    -------
    (M,) undiscounted call prices C/df
    """
    payoffs = np.maximum(grid_K[np.newaxis, :] - query_K[:, np.newaxis], 0.0)  # (M, N)
    return payoffs @ density


def put_prices(
    density: np.ndarray,
    grid_K: np.ndarray,
    query_K: np.ndarray,
    fwd: float,
) -> np.ndarray:
    """
    Undiscounted put prices P/df via put-call parity: P/df = C/df + K - F.
    """
    return call_prices(density, grid_K, query_K) + query_K - fwd


def option_prices(
    density: np.ndarray,
    grid_K: np.ndarray,
    query_K: np.ndarray,
    is_call: np.ndarray,
    fwd: float,
) -> np.ndarray:
    """
    Mixed undiscounted call/put prices: calls where is_call=True, puts elsewhere.
    """
    c = call_prices(density, grid_K, query_K)
    p = c + query_K - fwd
    return np.where(is_call, c, p)


# ---------------------------------------------------------------------------
# Black-Scholes primitives (normalized, log-moneyness space)
# ---------------------------------------------------------------------------


def _bs_call_norm(log_moneyness: float, T: float, iv: float) -> float:
    """
    Normalized BS call price C/(F*df) as a function of log(K/F).
    """
    iv = max(float(iv), 1e-8)
    T = max(float(T), 1e-8)
    sqrt_T = np.sqrt(T)
    d1 = (-log_moneyness + 0.5 * iv**2 * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T
    return float(norm.cdf(d1) - np.exp(log_moneyness) * norm.cdf(d2))


# ---------------------------------------------------------------------------
# IV inversion
# ---------------------------------------------------------------------------


def iv_from_norm_call(price: float, log_moneyness: float, T: float, fwd: float) -> float:
    """
    Implied vol from an undiscounted call price C/df.

    Internally normalizes by fwd to match the BS formula which expects C/(F*df).
    """
    price_norm = float(price) / fwd
    intrinsic = max(0.0, 1.0 - np.exp(log_moneyness))
    if price_norm <= intrinsic or price_norm >= 1.0:
        return float("nan")
    try:
        return brentq(
            lambda iv: _bs_call_norm(log_moneyness, T, iv) - price_norm,
            1e-6,
            5.0,
            xtol=1e-8,
        )
    except ValueError:
        return float("nan")


def iv_from_norm_put(price: float, log_moneyness: float, T: float, fwd: float) -> float:
    """
    Implied vol from an undiscounted put price P/df (converts to call via PCP).
    """
    call_price = price + fwd * (
        1.0 - np.exp(log_moneyness)
    )  # C/df = P/df + F - K = P/df + F*(1 - K/F)
    return iv_from_norm_call(call_price, log_moneyness, T, fwd)


def v_atm_from_price(
    mid_norm: float,
    log_moneyness: float,
    T: float,
    is_call: bool,
    fwd: float,
) -> float:
    """
    Single BS inversion near ATM to estimate V_atm = sigma^2 * T.
    mid_norm is the undiscounted price C/df or P/df.
    """
    T = max(float(T), 1e-8)
    iv = (
        iv_from_norm_call(mid_norm, log_moneyness, T, fwd)
        if is_call
        else iv_from_norm_put(mid_norm, log_moneyness, T, fwd)
    )
    if not np.isfinite(iv) or iv <= 0:
        return 0.04 * T
    return iv * iv * T


def iv_smile(
    density: np.ndarray,
    grid_K: np.ndarray,
    query_lk: np.ndarray,
    T: float,
    fwd: float,
    is_call: bool = True,
) -> np.ndarray:
    """
    Convert a SANOS density to an IV smile at the given log-moneyness grid.

    Parameters
    ----------
    density   : (N,) fitted density
    grid_K    : (N,) interior anchor dollar strikes
    query_lk  : (M,) log-moneyness values log(K/F)
    T         : time-to-expiry
    fwd       : forward price (used to convert log-moneyness to absolute K)
    is_call   : if False, invert put prices instead

    Returns
    -------
    (M,) implied vols, nan where inversion fails
    """
    query_K = fwd * np.exp(query_lk)  # absolute K from log-moneyness
    if is_call:
        prices = call_prices(density, grid_K, query_K)
        fn = iv_from_norm_call
    else:
        prices = put_prices(density, grid_K, query_K, fwd)
        fn = iv_from_norm_put
    return np.array([fn(float(p), float(lk), T, fwd) for p, lk in zip(prices, query_lk)])
