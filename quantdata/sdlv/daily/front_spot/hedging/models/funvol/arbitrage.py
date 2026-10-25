"""Empirical fixed-strike static-arbitrage screen for generated IV paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cdxcore.bs import bs
from numpy.typing import NDArray

from quantdata.biz_calendar import NMS
from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class StaticArbitrageScreen:
    """Violation fractions on an explicitly limited common-strike grid."""

    paths: int
    interior_paths: int
    strike_nodes: int
    strike_min: float
    strike_max: float
    tolerance: float
    strike_monotonicity_fraction: float
    strike_convexity_fraction: float
    calendar_fraction: float
    any_violation_fraction: float


def screen_generated_iv_surfaces(
    implied_volatilities: NDArray[np.floating],
    *,
    grid: GenerativeSurfaceGrid,
    log_strikes: FloatArray | None = None,
    tolerance: float = 1.0e-8,
) -> StaticArbitrageScreen:
    """Check Black call monotonicity, convexity and calendar order.

    This is an empirical grid screen, not a certificate of static no-arbitrage.
    Only paths on which every tested strike lies inside every expiry-specific
    floating-NMS grid enter the violation fractions.
    """
    surfaces = np.asarray(implied_volatilities, dtype=np.float64)
    if (
        grid.nms_nodes is None
        or surfaces.ndim != 3
        or surfaces.shape[1:] != (len(grid.expiries_y), grid.strike_count)
        or not np.all(np.isfinite(surfaces) & (surfaces > 0.0))
    ):
        raise ValueError("FuNVol static screen requires positive floating-NMS IV paths")
    strike_logs = (
        np.linspace(-0.02, 0.02, 21, dtype=np.float64)
        if log_strikes is None
        else np.asarray(log_strikes, dtype=np.float64)
    )
    if (
        strike_logs.ndim != 1
        or len(strike_logs) < 3
        or np.any(~np.isfinite(strike_logs))
        or np.any(np.diff(strike_logs) <= 0.0)
        or tolerance < 0.0
    ):
        raise ValueError("FuNVol static screen strike grid or tolerance is invalid")
    expiries = np.asarray(grid.expiries_y, dtype=np.float64)
    nodes = np.asarray(grid.nms_nodes, dtype=np.float64)
    atm_volatility = surfaces[:, :, grid.nms_nodes.index(0.0)]
    scale = np.sqrt(
        (NMS.SCALE**2 * NMS.EPS + atm_volatility**2 * expiries[None, :]) / (NMS.EPS + 1.0)
    )
    interior = np.all(
        (strike_logs[0] >= nodes[0] * scale) & (strike_logs[-1] <= nodes[-1] * scale),
        axis=1,
    )
    interior_count = int(np.count_nonzero(interior))
    if interior_count == 0:
        raise ValueError("FuNVol static screen has no paths inside the tested strike grid")
    selected = surfaces[interior]
    selected_scale = scale[interior]
    query_nms = strike_logs[None, None, :] / selected_scale[:, :, None]
    lower = np.clip(np.searchsorted(nodes, query_nms, side="right") - 1, 0, len(nodes) - 2)
    upper = lower + 1
    weight = (query_nms - nodes[lower]) / (nodes[upper] - nodes[lower])
    log_volatility = np.log(selected)
    interpolated_log_volatility = (
        np.take_along_axis(log_volatility, lower, axis=2) * (1.0 - weight)
        + np.take_along_axis(log_volatility, upper, axis=2) * weight
    )
    strikes = np.exp(strike_logs)
    calls = np.asarray(
        bs.price(
            k=strikes[None, None, :],
            vol=np.exp(interpolated_log_volatility),
            sqrtT=np.sqrt(expiries)[None, :, None],
        ),
        dtype=np.float64,
    )
    strike_slopes = np.diff(calls, axis=2) / np.diff(strikes)[None, None, :]
    monotonic = np.any(strike_slopes > tolerance, axis=(1, 2))
    convex = np.any(np.diff(strike_slopes, axis=2) < -tolerance, axis=(1, 2))
    calendar = np.any(np.diff(calls, axis=1) < -tolerance, axis=(1, 2))
    return StaticArbitrageScreen(
        paths=len(surfaces),
        interior_paths=interior_count,
        strike_nodes=len(strikes),
        strike_min=float(strikes[0]),
        strike_max=float(strikes[-1]),
        tolerance=tolerance,
        strike_monotonicity_fraction=float(np.mean(monotonic)),
        strike_convexity_fraction=float(np.mean(convex)),
        calendar_fraction=float(np.mean(calendar)),
        any_violation_fraction=float(np.mean(monotonic | convex | calendar)),
    )
