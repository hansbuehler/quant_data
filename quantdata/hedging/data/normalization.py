"""Shared normalized-curve and retained-observation timing helpers."""

from __future__ import annotations

import numpy as np


def interpolate_positive_curve_at_business_dte(
    *,
    business_dtes: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
    target_business_dte: int,
    name: str,
) -> float:
    """Log-linearly interpolate a positive curve anchored at one at DTE zero."""
    dtes = np.asarray(business_dtes, dtype=np.float64)
    curve = np.asarray(values, dtype=np.float64)
    if (
        dtes.ndim != 1
        or curve.shape != dtes.shape
        or len(dtes) == 0
        or np.any(~np.isfinite(dtes))
        or np.any(~np.isfinite(curve))
        or np.any(dtes <= 0.0)
        or np.any(np.diff(dtes) <= 0.0)
        or np.any(curve <= 0.0)
    ):
        raise ValueError(f"Invalid {name} curve")
    target = float(target_business_dte)
    if target <= 0.0 or target > dtes[-1] + 1.0e-8:
        raise ValueError(
            f"Cannot interpolate {name} at {target} business days from horizon {dtes[-1]}"
        )
    anchored_dtes = np.concatenate(([0.0], dtes))
    anchored_logs = np.concatenate(([0.0], np.log(curve)))
    return float(np.exp(np.interp(target, anchored_dtes, anchored_logs)))
