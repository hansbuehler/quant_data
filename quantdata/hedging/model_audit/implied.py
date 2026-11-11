"""Vectorized cdxcore Black inversion without an additional price-interiority veto."""

from __future__ import annotations

import numpy as np
from cdxcore.bs import bs


def implied_volatilities_with_validity(*, expiries_y: np.ndarray, normalized_strikes: np.ndarray,
                                       normalized_prices: np.ndarray, is_call: np.ndarray,
                                       diagnostics: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Use cdxcore's returned estimates, vector repricing and boundary conventions.

    Inputs are never clipped by this adapter. Library boundary estimates are
    counted, not relabelled as failed paths merely for small time value.
    """
    expiries, strikes, prices = np.broadcast_arrays(expiries_y, normalized_strikes, normalized_prices)
    calls = np.broadcast_to(np.asarray(is_call, dtype=bool), prices.shape)
    if not np.isfinite(expiries).all() or not np.isfinite(strikes).all() or not np.isfinite(prices).all():
        raise ValueError("Black inversion inputs must be finite")
    vols = np.asarray(bs.implied(k=strikes, prices=prices, is_call=calls, sqrtT=np.sqrt(expiries),
        price_tol=1e-8, eps=1e-10, on_exceed_bounds="warn"), dtype=np.float64)
    fitted = bs.price(k=strikes, vol=vols, sqrtT=np.sqrt(expiries), is_call=calls, eps=1e-10)
    intrinsic = np.where(calls, np.maximum(1. - strikes, 0.), np.maximum(strikes - 1., 0.))
    upper = np.where(calls, 1., strikes)
    outside = (prices < intrinsic - 1e-10) | (prices > upper + 1e-10)
    residuals = np.abs(fitted - prices)
    failed = residuals > 1e-8
    valid = ~failed & ~outside & np.isfinite(vols) & (vols > 0.) & (residuals <= 1e-8)
    if diagnostics is not None:
        diagnostics.append({"inversion": "cdxcore.bs.implied vector mode, unchanged library boundary conventions",
            "count": int(prices.size), "failed": int(failed.sum()), "outside_price_bounds": int(outside.sum()),
            "library_intrinsic_cases": int(np.sum(np.abs(prices - intrinsic) <= 1e-8)),
            "library_min_vol_cases": int(np.sum(vols == .01)), "library_max_vol_cases": int(np.sum(vols == 5.)),
            "unusable": int(np.sum(~valid)), "max_repricing_error": float(np.max(residuals))})
    return vols, valid


def summarize_inversions(records: list[dict]) -> dict:
    """Aggregate inversion calls, including repeated maturity-interpolation queries."""
    names = ("count", "failed", "outside_price_bounds", "library_intrinsic_cases",
             "library_min_vol_cases", "library_max_vol_cases", "unusable")
    return {**{name: sum(record[name] for record in records) for name in names},
            "max_repricing_error": max((record["max_repricing_error"] for record in records), default=0.),
            "policy": "direct vector cdxcore.bs.implied; no custom intrinsic-distance exclusion or input clipping"}
