"""
Numerical helpers for option prices and implied-vol state transforms.
"""

from __future__ import annotations

import numpy as np
from cdxcore.bs import bs

from quantdata.biz_calendar import DAYS_PER_YEAR

from .constants import MAX_VOL, MIN_SPOT, MIN_TTE_Y, MIN_VOL, TRADING_SECONDS_PER_DAY


def to_years(days):
    return np.maximum(np.asarray(days, dtype=float) / DAYS_PER_YEAR, MIN_TTE_Y)


def to_sqrt_years(days):
    return np.sqrt(to_years(days))


def clip_log_vol(log_vol):
    return np.clip(np.asarray(log_vol, dtype=float), np.log(MIN_VOL), np.log(MAX_VOL))


def unit_price_from_logvol(log_vol, strike_over_spot, sqrt_t, is_call):
    return bs.price(
        k=np.asarray(strike_over_spot, dtype=float),
        vol=np.exp(clip_log_vol(log_vol)),
        sqrtT=np.asarray(sqrt_t, dtype=float),
        is_call=np.asarray(is_call),
    )


def option_price_from_logvol(log_vol, spot, strike, sqrt_t, is_call):
    spot = np.maximum(np.asarray(spot, dtype=float), MIN_SPOT)
    return spot * unit_price_from_logvol(log_vol, np.asarray(strike, dtype=float) / spot, sqrt_t, is_call)


def direct_mid_vol(mid_price, spot, strike, sqrt_t, is_call):
    mid_price = np.asarray(mid_price, dtype=float)
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    sqrt_t = np.asarray(sqrt_t, dtype=float)
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), mid_price.shape)

    out = np.full(mid_price.shape, np.nan, dtype=float)
    mask = np.isfinite(mid_price) & np.isfinite(spot) & np.isfinite(strike) & np.isfinite(sqrt_t) & (spot > MIN_SPOT)
    if np.any(mask):
        out[mask] = np.asarray(
            bs.implied(
                k=strike[mask] / spot[mask],
                prices=np.clip(mid_price[mask] / spot[mask], 0.0, None),
                is_call=is_call[mask],
                sqrtT=sqrt_t[mask],
                default_vol=MIN_VOL,
                on_exceed_bounds="quiet",
            ),
            dtype=float,
        )
    return out


def finite_difference_jacobian(fn, x, eps=1e-4):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    base = np.atleast_1d(np.asarray(fn(x), dtype=float))
    jac = np.zeros((base.size, x.size), dtype=float)
    for j in range(x.size):
        step = np.zeros_like(x)
        step[j] = eps
        fp = np.atleast_1d(np.asarray(fn(x + step), dtype=float))
        fm = np.atleast_1d(np.asarray(fn(x - step), dtype=float))
        jac[:, j] = (fp - fm) / (2.0 * eps)
    return base, jac


def realized_scale(levels, dt_days, fallback):
    levels = np.asarray(levels, dtype=float)
    dt_y = np.asarray(dt_days, dtype=float) / DAYS_PER_YEAR
    valid = np.isfinite(levels) & np.isfinite(dt_y) & (dt_y > 0.0)
    levels = levels[valid]
    dt_y = dt_y[valid]
    if levels.size < 3:
        return float(fallback)

    dx = np.diff(levels)
    step_dt = dt_y[1:]
    inst_var = dx * dx / np.maximum(step_dt, 1e-12)
    inst_var = inst_var[np.isfinite(inst_var)]
    if inst_var.size == 0:
        return float(fallback)
    return float(np.sqrt(np.nanmedian(inst_var)))


def sanitize_interval_days(values, fallback=1e-12):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values > 0.0), values, fallback)


def weighted_lstsq(x, y, w):
    sqrt_w = np.sqrt(np.asarray(w, dtype=float))[:, None]
    beta, *_ = np.linalg.lstsq(sqrt_w * x, sqrt_w[:, 0] * y, rcond=None)
    return beta


def surface_z(strike, spot):
    strike = np.asarray(strike, dtype=float)
    spot = np.maximum(np.asarray(spot, dtype=float), MIN_SPOT)
    return strike / spot - 1.0


def surface_log_vol(state, strike, spot):
    log_atm, vol_skew, vol_cvxy = np.asarray(state, dtype=float)
    z = surface_z(strike, spot)
    return log_atm + vol_skew * z + 0.5 * vol_cvxy * z * z


def bin_width_days(bin_seconds):
    return float(bin_seconds) / TRADING_SECONDS_PER_DAY
