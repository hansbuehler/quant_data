"""
Option pricing and IV inversion for single-expiry SANOS.

All prices are normalized (divided by F * df). Strikes are K/F values.
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
    Normalized call prices under the fitted density.

        C(K) = sum_i max(K_i - K, 0) * d_i

    Parameters
    ----------
    density : (N,) risk-neutral density over grid_K
    grid_K  : (N,) interior anchor strikes (K/F)
    query_K : (M,) query strikes (K/F)

    Returns
    -------
    (M,) normalized call prices
    """
    payoffs = np.maximum(grid_K[np.newaxis, :] - query_K[:, np.newaxis], 0.0)  # (M, N)
    return payoffs @ density


def put_prices(
    density: np.ndarray,
    grid_K: np.ndarray,
    query_K: np.ndarray,
) -> np.ndarray:
    """
    Normalized put prices via put-call parity: P(K) = C(K) + K - 1.
    """
    return call_prices(density, grid_K, query_K) + query_K - 1.0


def option_prices(
    density: np.ndarray,
    grid_K: np.ndarray,
    query_K: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """
    Mixed call/put prices: calls where is_call=True, puts elsewhere.
    """
    c = call_prices(density, grid_K, query_K)
    p = c + query_K - 1.0
    return np.where(is_call, c, p)


# ---------------------------------------------------------------------------
# Black-Scholes primitives (normalized, log-moneyness space)
# ---------------------------------------------------------------------------


def _bs_call_norm(log_moneyness: float, T: float, iv: float) -> float:
    """
    Normalized BS call price: C / (F * df) as a function of log(K/F).

        C_norm = N(d1) - exp(log_moneyness) * N(d2)
        d1 = (-log_moneyness + 0.5 * iv^2 * T) / (iv * sqrt(T))
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


def iv_from_norm_call(price: float, log_moneyness: float, T: float) -> float:
    """
    Implied vol from a normalized call price via Brent root-finding.
    Returns nan if price is outside the no-arbitrage bounds.
    """
    intrinsic = max(0.0, 1.0 - np.exp(log_moneyness))
    if price <= intrinsic or price >= 1.0:
        return float("nan")
    try:
        return brentq(
            lambda iv: _bs_call_norm(log_moneyness, T, iv) - price,
            1e-6,
            5.0,
            xtol=1e-8,
        )
    except ValueError:
        return float("nan")


def iv_from_norm_put(price: float, log_moneyness: float, T: float) -> float:
    """
    Implied vol from a normalized put price (converts to call via PCP).
    """
    call_price = price + 1.0 - np.exp(log_moneyness)
    return iv_from_norm_call(call_price, log_moneyness, T)


def v_atm_from_price(
    mid_norm: float,
    log_moneyness: float,
    T: float,
    is_call: bool,
) -> float:
    """
    Single BS inversion near ATM to estimate V_atm = sigma^2 * T.
    Falls back to (20% vol)^2 * T if inversion fails.
    """
    T = max(float(T), 1e-8)
    iv = (
        iv_from_norm_call(mid_norm, log_moneyness, T)
        if is_call
        else iv_from_norm_put(mid_norm, log_moneyness, T)
    )
    if not np.isfinite(iv) or iv <= 0:
        return 0.04 * T
    return iv * iv * T


def iv_smile(
    density: np.ndarray,
    grid_K: np.ndarray,
    query_lk: np.ndarray,
    T: float,
    is_call: bool = True,
) -> np.ndarray:
    """
    Convert a SANOS density to an IV smile at the given log-moneyness grid.

    Parameters
    ----------
    density   : (N,) fitted density
    grid_K    : (N,) interior anchor K/F values
    query_lk  : (M,) log-moneyness values log(K/F)
    T         : time-to-expiry
    is_call   : if False, invert put prices instead

    Returns
    -------
    (M,) implied vols, nan where inversion fails
    """
    query_K = np.exp(query_lk)
    if is_call:
        prices = call_prices(density, grid_K, query_K)
        fn = iv_from_norm_call
    else:
        prices = put_prices(density, grid_K, query_K)
        fn = iv_from_norm_put
    return np.array([fn(float(p), float(lk), T) for p, lk in zip(prices, query_lk)])
