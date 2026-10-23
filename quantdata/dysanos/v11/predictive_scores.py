"""Proper next-session ATM-IV forecast scores for historical and 2025 audits."""

from __future__ import annotations

import numpy as np


def empirical_crps(observed: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Compute CRPS per origin and tenor without an O(paths²) pairwise array."""
    values = np.asarray(observed, dtype=np.float64)
    draws = np.asarray(samples, dtype=np.float64)
    if (
        values.ndim != 2
        or draws.ndim != 3
        or draws.shape[0] != values.shape[0]
        or draws.shape[2] != values.shape[1]
        or draws.shape[1] < 2
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(draws))
    ):
        raise ValueError("observed must be (origins, tenors) and samples (origins, paths, tenors)")
    ordered = np.sort(draws, axis=1)
    count = draws.shape[1]
    coefficients = 2.0 * np.arange(1, count + 1) - count - 1.0
    absolute_error = np.mean(np.abs(draws - values[:, None, :]), axis=1)
    pair_term = np.sum(ordered * coefficients[None, :, None], axis=1) / count**2
    return absolute_error - pair_term


def predictive_panel(observed: np.ndarray, samples: np.ndarray) -> dict[str, object]:
    """Summarize forecasts in the same units as observed ATM log-IV changes."""
    values = np.asarray(observed, dtype=np.float64)
    draws = np.asarray(samples, dtype=np.float64)
    crps = empirical_crps(values, draws)
    predicted_mean = np.mean(draws, axis=1)
    predicted_variance = np.var(draws, axis=1, ddof=1)
    innovation = values - predicted_mean
    if np.any(predicted_variance <= 0.0):
        raise ValueError("a predictive distribution has zero path variance")
    quantile_coverage: dict[str, list[float]] = {}
    for nominal in (0.50, 0.80, 0.95):
        tail = (1.0 - nominal) / 2.0
        lower, upper = np.quantile(draws, [tail, 1.0 - tail], axis=1)
        quantile_coverage[f"{nominal:.2f}"] = np.mean(
            (values >= lower) & (values <= upper), axis=0
        ).tolist()
    return {
        "origins": len(values),
        "paths_per_origin": draws.shape[1],
        "innovation_bias": np.mean(innovation, axis=0).tolist(),
        "innovation_rmse": np.sqrt(np.mean(innovation**2, axis=0)).tolist(),
        "predictive_variance_ratio": (
            np.var(innovation, axis=0, ddof=1)
            / np.mean(predicted_variance, axis=0)
        ).tolist(),
        "mean_crps": np.mean(crps, axis=0).tolist(),
        "central_interval_coverage": quantile_coverage,
    }
