"""Coordinate transformations for normalized zero-carry option surfaces."""

from __future__ import annotations

import numpy as np
from cdxcore.bs import bs
from cdxcore.err import verify_inp


def strike_from_log_moneyness(log_moneyness: np.ndarray | float) -> np.ndarray:
    """Return forward-relative strike ``K/F`` from ``log(K/F)``."""

    value = np.asarray(log_moneyness, dtype=np.float64)
    verify_inp(
        np.all(np.isfinite(value)),
        lambda: "log moneyness must contain only finite values",
    )
    return np.exp(value)


def log_moneyness_from_strike(strike: np.ndarray | float) -> np.ndarray:
    """Return ``log(K/F)`` from positive forward-relative strikes."""

    value = np.asarray(strike, dtype=np.float64)
    verify_inp(
        np.all(np.isfinite(value)) and np.all(value > 0.0),
        lambda: "forward-relative strikes must be finite and positive",
    )
    return np.log(value)


def normalized_call_intrinsic(strike: np.ndarray | float) -> np.ndarray:
    """Return normalized zero-carry call intrinsic value ``max(1-K/F, 0)``."""

    value = np.asarray(strike, dtype=np.float64)
    verify_inp(
        np.all(np.isfinite(value)) and np.all(value > 0.0),
        lambda: "forward-relative strikes must be finite and positive",
    )
    return np.maximum(1.0 - value, 0.0)


def normalized_time_value(
    strike: np.ndarray | float,
    call_price: np.ndarray | float,
) -> np.ndarray:
    """Return normalized call time value."""

    price = np.asarray(call_price, dtype=np.float64)
    verify_inp(
        np.all(np.isfinite(price)),
        lambda: "call prices must contain only finite values",
    )
    return price - normalized_call_intrinsic(strike)


def implied_volatility(
    *,
    tau: np.ndarray | float,
    strike: np.ndarray | float,
    call_price: np.ndarray | float,
) -> np.ndarray:
    """Invert normalized zero-carry calls to Black-Scholes implied volatility."""

    tau_array = np.asarray(tau, dtype=np.float64)
    strike_array = np.asarray(strike, dtype=np.float64)
    price_array = np.asarray(call_price, dtype=np.float64)
    verify_inp(
        np.all(np.isfinite(tau_array)) and np.all(tau_array > 0.0),
        lambda: "time to maturity must be finite and positive",
    )
    lower = normalized_call_intrinsic(strike_array)
    upper = np.ones(np.broadcast_shapes(strike_array.shape, price_array.shape))
    verify_inp(
        np.all(np.isfinite(price_array))
        and np.all(price_array >= lower - 1.0e-10)
        and np.all(price_array <= upper + 1.0e-10),
        lambda: "call prices violate normalized no-arbitrage bounds",
    )
    return np.asarray(
        bs.implied(
            k=strike_array,
            prices=np.clip(price_array, lower, upper),
            sqrtT=np.sqrt(tau_array),
            on_exceed_bounds="quiet",
        ),
        dtype=np.float64,
    )


def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    """Return integration weights for a finite, strictly increasing grid."""

    value = np.asarray(grid, dtype=np.float64)
    verify_inp(
        value.ndim == 1
        and value.size >= 2
        and np.all(np.isfinite(value))
        and np.all(np.diff(value) > 0.0),
        lambda: f"grid must be finite and strictly increasing, got shape {value.shape}",
    )
    weights = np.empty_like(value)
    weights[0] = 0.5 * (value[1] - value[0])
    weights[-1] = 0.5 * (value[-1] - value[-2])
    weights[1:-1] = 0.5 * (value[2:] - value[:-2])
    return weights
