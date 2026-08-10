"""Streaming diagnostics for state-independent, single-option arbitrage.

The routines in this module deliberately do not consume ``z``.  They count
the signs of delta-hedged option returns, and of the long/short net returns
after costs, while retaining the simulated path as the statistical cluster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
from cdxcore.err import verify_inp

FLOAT32_UNIT_ROUNDOFF = float(np.finfo(np.float32).eps)
DEFAULT_ROUNDOFF_MULTIPLIER = 32.0


def delta_hedged_option_returns(
    *, dC: np.ndarray, delta: np.ndarray, dS: np.ndarray
) -> np.ndarray:
    """Return ``dC - delta*dS`` without including the spot instrument."""

    verify_inp(
        dC.shape == delta.shape and dC.ndim == 4,
        lambda: f"dC/delta must share shape (S, TAU, T, N), got {dC.shape}/{delta.shape}",
    )
    verify_inp(
        dS.shape == dC.shape[:2],
        lambda: f"dS shape {dS.shape} != {dC.shape[:2]}",
    )
    values = dC - delta * dS[..., None, None]
    if not np.isfinite(values).all():
        raise ValueError("delta-hedged option returns contain non-finite values")
    return values.reshape(dC.shape[0], dC.shape[1], -1)


def delta_hedged_option_cost_notionals(
    *, C: np.ndarray, delta: np.ndarray, spot: np.ndarray
) -> np.ndarray:
    """Return gross traded notionals ``|C| + |delta*spot|`` for options."""

    verify_inp(
        C.shape == delta.shape and C.ndim == 4,
        lambda: f"C/delta must share shape (S, TAU, T, N), got {C.shape}/{delta.shape}",
    )
    verify_inp(
        spot.shape == C.shape[:2],
        lambda: f"spot shape {spot.shape} != {C.shape[:2]}",
    )
    values = np.abs(C) + np.abs(delta * spot[..., None, None])
    if not np.isfinite(values).all():
        raise ValueError("delta-hedged option cost notionals contain non-finite values")
    return values.reshape(C.shape[0], C.shape[1], -1)


def delta_hedged_roundoff_band(
    *,
    C: np.ndarray,
    dC: np.ndarray,
    delta: np.ndarray,
    dS: np.ndarray,
    multiplier: float = DEFAULT_ROUNDOFF_MULTIPLIER,
) -> np.ndarray:
    """Return a conservative absolute error band for float32 hedged returns.

    ``dC`` was itself formed by subtracting two float32 option values.  The
    scale therefore includes both reconstructed endpoint values and the hedge
    term, rather than the possibly cancellation-dominated final ``dH``.
    """

    verify_inp(
        math.isfinite(multiplier) and multiplier >= 0.0,
        lambda: f"'multiplier' must be finite and nonnegative, got {multiplier}",
    )
    verify_inp(
        C.shape == dC.shape == delta.shape and C.ndim == 4,
        lambda: f"C/dC/delta shapes differ: {C.shape}/{dC.shape}/{delta.shape}",
    )
    verify_inp(
        dS.shape == C.shape[:2],
        lambda: f"dS shape {dS.shape} != {C.shape[:2]}",
    )
    hedge = delta * dS[..., None, None]
    scale = np.maximum(
        np.abs(C + dC) + np.abs(C) + np.abs(hedge),
        np.finfo(np.float32).tiny,
    )
    band = multiplier * FLOAT32_UNIT_ROUNDOFF * scale
    return band.reshape(C.shape[0], C.shape[1], -1)


def net_return_roundoff_band(
    values: np.ndarray,
    cost: np.ndarray,
    *,
    multiplier: float = DEFAULT_ROUNDOFF_MULTIPLIER,
) -> np.ndarray:
    """Return a float32 arithmetic band for ``+/- values - cost``."""

    verify_inp(
        values.shape == cost.shape,
        lambda: f"return/cost shapes differ: {values.shape}/{cost.shape}",
    )
    verify_inp(
        math.isfinite(multiplier) and multiplier >= 0.0,
        lambda: f"'multiplier' must be finite and nonnegative, got {multiplier}",
    )
    return multiplier * FLOAT32_UNIT_ROUNDOFF * np.maximum(
        np.abs(values) + np.abs(cost), np.finfo(np.float32).tiny
    )


@dataclass
class SignAccumulator:
    """Streaming per-instrument signs and path-clustered moments."""

    instruments: int

    def __post_init__(self) -> None:
        verify_inp(
            self.instruments > 0,
            lambda: f"'instruments' must be positive, got {self.instruments}",
        )
        J = self.instruments
        self.positive = np.zeros(J, dtype=np.int64)
        self.negative = np.zeros(J, dtype=np.int64)
        self.neutral = np.zeros(J, dtype=np.int64)
        self.positive_paths = np.zeros(J, dtype=np.int64)
        self.negative_paths = np.zeros(J, dtype=np.int64)
        self.total = np.zeros(J, dtype=np.float64)
        self.total_sqr = np.zeros(J, dtype=np.float64)
        self.path_mean_total = np.zeros(J, dtype=np.float64)
        self.path_mean_sqr = np.zeros(J, dtype=np.float64)
        self.minimum = np.full(J, np.inf, dtype=np.float64)
        self.maximum = np.full(J, -np.inf, dtype=np.float64)
        self.observations = 0
        self.paths = 0
        self.steps: int | None = None

    def update(self, values: np.ndarray, tolerance: float | np.ndarray) -> None:
        """Accumulate one path batch with shape ``(S, TAU, J)``."""

        verify_inp(
            values.ndim == 3 and values.shape[-1] == self.instruments,
            lambda: f"values must have shape (S, TAU, {self.instruments}), got {values.shape}",
        )
        if not np.isfinite(values).all():
            raise ValueError("sign values contain non-finite values")
        tolerance_array = np.asarray(tolerance)
        verify_inp(
            tolerance_array.ndim == 0 or tolerance_array.shape == values.shape,
            lambda: f"tolerance shape {tolerance_array.shape} is incompatible with {values.shape}",
        )
        if not np.isfinite(tolerance_array).all() or np.any(tolerance_array < 0.0):
            raise ValueError("sign tolerance must be finite and nonnegative")
        if self.steps is None:
            self.steps = values.shape[1]
        verify_inp(
            values.shape[1] == self.steps,
            lambda: f"step count changed from {self.steps} to {values.shape[1]}",
        )

        positive = values > tolerance_array
        negative = values < -tolerance_array
        self.positive += positive.sum(axis=(0, 1), dtype=np.int64)
        self.negative += negative.sum(axis=(0, 1), dtype=np.int64)
        self.neutral += (~positive & ~negative).sum(axis=(0, 1), dtype=np.int64)
        self.positive_paths += positive.any(axis=1).sum(axis=0, dtype=np.int64)
        self.negative_paths += negative.any(axis=1).sum(axis=0, dtype=np.int64)
        values64 = values.astype(np.float64, copy=False)
        self.total += values64.sum(axis=(0, 1), dtype=np.float64)
        self.total_sqr += np.square(values64).sum(axis=(0, 1), dtype=np.float64)
        path_mean = values64.mean(axis=1)
        self.path_mean_total += path_mean.sum(axis=0, dtype=np.float64)
        self.path_mean_sqr += np.square(path_mean).sum(axis=0, dtype=np.float64)
        self.minimum = np.minimum(self.minimum, values64.min(axis=(0, 1)))
        self.maximum = np.maximum(self.maximum, values64.max(axis=(0, 1)))
        self.observations += values.shape[0] * values.shape[1]
        self.paths += values.shape[0]

    def arrays(self) -> dict[str, np.ndarray | int | str]:
        """Return complete per-instrument arrays using population moments."""

        if self.observations == 0 or self.paths == 0 or self.steps is None:
            raise RuntimeError("cannot summarize an empty sign accumulator")
        mean = self.total / float(self.observations)
        variance = (
            self.total_sqr / float(self.observations) - np.square(mean)
        ).clip(min=0.0)
        path_mean = self.path_mean_total / float(self.paths)
        path_variance = (
            self.path_mean_sqr / float(self.paths) - np.square(path_mean)
        ).clip(min=0.0)
        signed = self.positive - self.negative
        nonneutral = self.positive + self.negative
        return {
            "positive": self.positive.copy(),
            "negative": self.negative.copy(),
            "neutral": self.neutral.copy(),
            "positive_paths": self.positive_paths.copy(),
            "negative_paths": self.negative_paths.copy(),
            "imbalance": np.abs(signed),
            "signed_imbalance": signed,
            "imbalance_fraction_all": np.abs(signed) / float(self.observations),
            "imbalance_fraction_nonneutral": np.divide(
                np.abs(signed),
                nonneutral,
                out=np.zeros_like(mean),
                where=nonneutral > 0,
            ),
            "minority_fraction_all": np.minimum(self.positive, self.negative)
            / float(self.observations),
            "mean": mean,
            "std_population": np.sqrt(variance),
            "path_mean_std_population": np.sqrt(path_variance),
            "path_clustered_mean_error": np.sqrt(path_variance / float(self.paths)),
            "minimum": self.minimum.copy(),
            "maximum": self.maximum.copy(),
            "observations": self.observations,
            "paths": self.paths,
            "steps": self.steps,
            "std_convention": "population",
        }


def _instrument_item(
    arrays: dict[str, np.ndarray | int | str],
    option: int,
    *,
    strikes: int,
    direction: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "option": option,
        "instrument": option + 1,
        "expiry_index": option // strikes,
        "strike_index": option % strikes,
    }
    if direction is not None:
        item["direction"] = direction
    for name in (
        "positive",
        "negative",
        "neutral",
        "positive_paths",
        "negative_paths",
        "imbalance",
        "signed_imbalance",
    ):
        item[name] = int(np.asarray(arrays[name])[option])
    for name in (
        "imbalance_fraction_all",
        "imbalance_fraction_nonneutral",
        "minority_fraction_all",
        "mean",
        "std_population",
        "path_mean_std_population",
        "path_clustered_mean_error",
        "minimum",
        "maximum",
    ):
        item[name] = float(np.asarray(arrays[name])[option])
    error = item["path_clustered_mean_error"]
    item["mean_lower_95"] = item["mean"] - NormalDist().inv_cdf(0.95) * error
    return item


def rank_imbalance(
    accumulator: SignAccumulator,
    *,
    strikes: int,
    count: int = 20,
) -> list[dict[str, Any]]:
    """Return options with the largest requested raw imbalance count."""

    arrays = accumulator.arrays()
    imbalance = np.asarray(arrays["imbalance"])
    order = np.argsort(-imbalance, kind="stable")[:count]
    return [
        _instrument_item(arrays, int(option), strikes=strikes) for option in order
    ]


def rank_no_arbitrage_candidates(
    long: SignAccumulator,
    short: SignAccumulator,
    *,
    strikes: int,
    count: int = 20,
) -> list[dict[str, Any]]:
    """Rank directions by fewest losses, then largest lower mean bound."""

    verify_inp(
        long.instruments == short.instruments,
        lambda: f"long/short instrument counts differ: {long.instruments}/{short.instruments}",
    )
    long_arrays, short_arrays = long.arrays(), short.arrays()
    items: list[dict[str, Any]] = []
    for direction, arrays in (("long", long_arrays), ("short", short_arrays)):
        items.extend(
            _instrument_item(arrays, option, strikes=strikes, direction=direction)
            for option in range(long.instruments)
        )
    items.sort(
        key=lambda item: (
            item["negative"],
            item["negative_paths"],
            -item["mean_lower_95"],
        )
    )
    return items[:count]


def population_standard_error(
    total: np.ndarray,
    total_sqr: np.ndarray,
    observations: int,
) -> np.ndarray:
    """Return population standard deviation divided by ``sqrt(observations)``."""

    verify_inp(observations > 0, lambda: f"'observations' must be positive, got {observations}")
    mean = total / float(observations)
    variance = (total_sqr / float(observations) - np.square(mean)).clip(min=0.0)
    return np.sqrt(variance / float(observations))


__all__ = [
    "DEFAULT_ROUNDOFF_MULTIPLIER",
    "FLOAT32_UNIT_ROUNDOFF",
    "SignAccumulator",
    "delta_hedged_option_returns",
    "delta_hedged_option_cost_notionals",
    "delta_hedged_roundoff_band",
    "net_return_roundoff_band",
    "population_standard_error",
    "rank_imbalance",
    "rank_no_arbitrage_candidates",
]
