"""Episode-level inference for daily and overlapping weekly hedge losses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

_NORMAL_95 = 1.959963984540054
_NORMAL_99 = 2.5758293035489004


class HacKernel(StrEnum):
    """Supported lag windows for episode-mean long-run variance."""

    HANSEN_HODRICK = "hansen-hodrick"
    BARTLETT = "bartlett"


@version("1.1")
@dataclass(frozen=True, kw_only=True)
class EpisodeInferenceConfig:
    """Precommitted dependence and bootstrap controls for episode losses."""

    hac_lags: int = 4
    hac_kernel: HacKernel = HacKernel.HANSEN_HODRICK
    bootstrap_block_length: int = 50
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 20_261_012

    def __post_init__(self) -> None:
        verify_inp(self.hac_lags >= 0, lambda: "hac_lags must be non-negative")
        verify_inp(
            isinstance(self.hac_kernel, HacKernel),
            lambda: "hac_kernel must be a HacKernel",
        )
        verify_inp(
            self.bootstrap_block_length > 0,
            lambda: "bootstrap_block_length must be positive",
        )
        verify_inp(
            self.bootstrap_samples >= 100,
            lambda: "bootstrap_samples must be at least 100",
        )
        verify_inp(
            isinstance(self.bootstrap_seed, int)
            and not isinstance(self.bootstrap_seed, bool),
            lambda: "bootstrap_seed must be an integer",
        )


@dataclass(frozen=True, kw_only=True)
class MeanConfidenceInterval:
    """HAC and moving-block-bootstrap uncertainty for a sample mean."""

    hac_standard_error: float
    raw_hac_long_run_variance: float
    hac_long_run_variance: float
    hac_variance_was_floored: bool
    hac_kernel: HacKernel
    hac_lags: int
    hac_95: tuple[float, float]
    hac_99: tuple[float, float]
    bootstrap_95: tuple[float, float]
    bootstrap_99: tuple[float, float]


@dataclass(frozen=True, kw_only=True)
class EpisodeLossSummary:
    """Distribution and dependence-aware mean statistics for episode losses."""

    observations: int
    mean: float
    median: float
    sample_standard_deviation: float
    root_mean_square: float
    mean_absolute_loss: float
    quantile_95: float
    quantile_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    inference: MeanConfidenceInterval


def summarize_episode_losses(
    losses: ArrayLike,
    *,
    episode_dates: Sequence[date] | None = None,
    episode_indices: Sequence[int] | None = None,
    full_episode_count: int | None = None,
    config: EpisodeInferenceConfig = EpisodeInferenceConfig(),
) -> EpisodeLossSummary:
    """Summarize ordered complete-episode losses with overlap-aware inference."""
    values = _finite_vector(losses, name="losses")
    if episode_dates is not None:
        _validate_chronological_dates(episode_dates, observations=len(values))
    indices, full_count = _validated_episode_indices(
        episode_indices,
        observations=len(values),
        full_episode_count=full_episode_count,
    )
    if len(values) < 2:
        raise ValueError("at least two episode losses are required")
    effective_hac_lags = min(config.hac_lags, full_count - 1)
    effective_block_length = min(config.bootstrap_block_length, full_count)

    mean = float(np.mean(values))
    raw_long_run_variance = irregular_hac_long_run_variance(
        values,
        episode_indices=indices,
        full_episode_count=full_count,
        lags=effective_hac_lags,
        kernel=config.hac_kernel,
    )
    long_run_variance = max(raw_long_run_variance, 0.0)
    standard_error = math.sqrt(long_run_variance / len(values))
    bootstrap_95 = masked_moving_block_bootstrap_mean_interval(
        values,
        episode_indices=indices,
        full_episode_count=full_count,
        confidence_level=0.95,
        block_length=effective_block_length,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )
    bootstrap_99 = masked_moving_block_bootstrap_mean_interval(
        values,
        episode_indices=indices,
        full_episode_count=full_count,
        confidence_level=0.99,
        block_length=effective_block_length,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )
    quantile_95 = float(np.quantile(values, 0.95, method="linear"))
    quantile_99 = float(np.quantile(values, 0.99, method="linear"))
    return EpisodeLossSummary(
        observations=len(values),
        mean=mean,
        median=float(np.median(values)),
        sample_standard_deviation=float(np.std(values, ddof=1)),
        root_mean_square=float(np.sqrt(np.mean(np.square(values)))),
        mean_absolute_loss=float(np.mean(np.abs(values))),
        quantile_95=quantile_95,
        quantile_99=quantile_99,
        expected_shortfall_95=float(np.mean(values[values >= quantile_95])),
        expected_shortfall_99=float(np.mean(values[values >= quantile_99])),
        inference=MeanConfidenceInterval(
            hac_standard_error=standard_error,
            raw_hac_long_run_variance=raw_long_run_variance,
            hac_long_run_variance=long_run_variance,
            hac_variance_was_floored=raw_long_run_variance < 0.0,
            hac_kernel=config.hac_kernel,
            hac_lags=effective_hac_lags,
            hac_95=(
                mean - _NORMAL_95 * standard_error,
                mean + _NORMAL_95 * standard_error,
            ),
            hac_99=(
                mean - _NORMAL_99 * standard_error,
                mean + _NORMAL_99 * standard_error,
            ),
            bootstrap_95=bootstrap_95,
            bootstrap_99=bootstrap_99,
        ),
    )


def irregular_hac_long_run_variance(
    values: ArrayLike,
    *,
    episode_indices: Sequence[int],
    full_episode_count: int,
    lags: int,
    kernel: HacKernel = HacKernel.HANSEN_HODRICK,
) -> float:
    """Return HAC variance while retaining gaps in the full episode sequence."""
    sample = _finite_vector(values, name="values")
    indices, full_count = _validated_episode_indices(
        episode_indices,
        observations=len(sample),
        full_episode_count=full_episode_count,
    )
    if len(sample) < 2:
        raise ValueError("at least two observations are required")
    if not 0 <= lags < full_count:
        raise ValueError("lags must lie in [0, full_episode_count)")
    if not isinstance(kernel, HacKernel):
        raise ValueError("kernel must be a HacKernel")
    centered = sample - np.mean(sample)
    full_centered = np.zeros(full_count, dtype=np.float64)
    selected = np.zeros(full_count, dtype=bool)
    full_centered[indices] = centered
    selected[indices] = True
    numerator = float(centered @ centered)
    for lag in range(1, lags + 1):
        paired = selected[lag:] & selected[:-lag]
        covariance_sum = float(
            full_centered[lag:][paired] @ full_centered[:-lag][paired]
        )
        weight = (
            1.0
            if kernel is HacKernel.HANSEN_HODRICK
            else 1.0 - lag / (lags + 1.0)
        )
        numerator += 2.0 * weight * covariance_sum
    return numerator / len(sample)


def masked_moving_block_bootstrap_mean_interval(
    values: ArrayLike,
    *,
    episode_indices: Sequence[int],
    full_episode_count: int,
    confidence_level: float,
    block_length: int,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap full chronological blocks while retaining a slice's gaps."""
    sample = _finite_vector(values, name="values")
    indices, full_count = _validated_episode_indices(
        episode_indices,
        observations=len(sample),
        full_episode_count=full_episode_count,
    )
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not 1 <= block_length <= full_count:
        raise ValueError("block_length must lie in [1, full_episode_count]")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    full_values = np.full(full_count, np.nan, dtype=np.float64)
    full_values[indices] = sample
    generator = np.random.default_rng(seed)
    blocks_per_sample = math.ceil(full_count / block_length)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    completed = 0
    while completed < samples:
        requested = samples - completed
        starts = generator.integers(
            0,
            full_count,
            size=(requested, blocks_per_sample),
        )
        offsets = np.arange(block_length)
        sampled_indices = (
            starts[:, :, None] + offsets[None, None, :]
        ) % full_count
        sampled_indices = sampled_indices.reshape(requested, -1)[:, :full_count]
        sampled = full_values[sampled_indices]
        counts = np.sum(np.isfinite(sampled), axis=1)
        usable = counts > 0
        if not np.any(usable):
            continue
        sums = np.nansum(sampled[usable], axis=1)
        current = sums / counts[usable]
        take = min(len(current), samples - completed)
        bootstrap_means[completed : completed + take] = current[:take]
        completed += take
    tail = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(
        bootstrap_means,
        (tail, 1.0 - tail),
        method="linear",
    )
    return float(lower), float(upper)


def hac_standard_error_of_mean(
    values: ArrayLike,
    *,
    lags: int,
    kernel: HacKernel = HacKernel.HANSEN_HODRICK,
) -> float:
    """Return a HAC standard error for an ordered complete-episode mean.

    The Hansen--Hodrick kernel is the production default because an episode of
    length ``H`` started every session has mechanically nonzero covariance
    through lag ``H-1``. Bartlett weights would attenuate those known overlap
    covariances and understate the long-run variance even asymptotically.
    """
    sample = _finite_vector(values, name="values")
    long_run_variance = hac_long_run_variance(
        sample,
        lags=lags,
        kernel=kernel,
    )
    return math.sqrt(max(long_run_variance, 0.0) / len(sample))


def hac_long_run_variance(
    values: ArrayLike,
    *,
    lags: int,
    kernel: HacKernel = HacKernel.HANSEN_HODRICK,
) -> float:
    """Return the raw, potentially negative finite-sample HAC estimate."""
    sample = _finite_vector(values, name="values")
    if len(sample) < 2:
        raise ValueError("at least two observations are required")
    if not 0 <= lags < len(sample):
        raise ValueError("lags must lie in [0, observations)")
    if not isinstance(kernel, HacKernel):
        raise ValueError("kernel must be a HacKernel")
    centered = sample - np.mean(sample)
    long_run_variance = float(centered @ centered / len(sample))
    for lag in range(1, lags + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / len(sample))
        weight = (
            1.0
            if kernel is HacKernel.HANSEN_HODRICK
            else 1.0 - lag / (lags + 1.0)
        )
        long_run_variance += 2.0 * weight * covariance
    return long_run_variance


def moving_block_bootstrap_mean_interval(
    values: ArrayLike,
    *,
    confidence_level: float,
    block_length: int,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Return a circular moving-block percentile interval for the sample mean."""
    sample = _finite_vector(values, name="values")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not 1 <= block_length <= len(sample):
        raise ValueError("block_length must lie in [1, observations]")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    generator = np.random.default_rng(seed)
    blocks_per_sample = math.ceil(len(sample) / block_length)
    starts = generator.integers(
        0,
        len(sample),
        size=(samples, blocks_per_sample),
    )
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(sample)
    indices = indices.reshape(samples, -1)[:, : len(sample)]
    bootstrap_means = np.mean(sample[indices], axis=1)
    tail = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(
        bootstrap_means,
        (tail, 1.0 - tail),
        method="linear",
    )
    return float(lower), float(upper)


def paired_episode_differences(
    left_ids: Sequence[str],
    left_losses: ArrayLike,
    right_ids: Sequence[str],
    right_losses: ArrayLike,
) -> FloatArray:
    """Align complete episodes and return left-minus-right loss differences."""
    left = _finite_vector(left_losses, name="left_losses")
    right = _finite_vector(right_losses, name="right_losses")
    if len(left_ids) != len(left) or len(right_ids) != len(right):
        raise ValueError("episode identifiers must align with loss vectors")
    if len(set(left_ids)) != len(left_ids) or len(set(right_ids)) != len(right_ids):
        raise ValueError("episode identifiers must be unique within each model")
    right_by_id = dict(zip(right_ids, right, strict=True))
    if set(left_ids) != set(right_ids):
        raise ValueError("paired model comparisons require identical episode sets")
    return np.asarray(
        [left_value - right_by_id[episode_id] for episode_id, left_value in zip(left_ids, left, strict=True)],
        dtype=np.float64,
    )


def _finite_vector(values: ArrayLike, *, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return vector


def _validated_episode_indices(
    episode_indices: Sequence[int] | None,
    *,
    observations: int,
    full_episode_count: int | None,
) -> tuple[NDArray[np.int64], int]:
    if episode_indices is None:
        if full_episode_count not in {None, observations}:
            raise ValueError(
                "full_episode_count requires explicit episode_indices when it "
                "differs from the observation count"
            )
        return np.arange(observations, dtype=np.int64), observations
    indices = np.asarray(tuple(episode_indices), dtype=np.int64)
    if indices.shape != (observations,):
        raise ValueError("episode_indices must align with episode losses")
    if full_episode_count is None or full_episode_count <= 0:
        raise ValueError("full_episode_count must be positive with episode_indices")
    if (
        np.any(indices < 0)
        or np.any(indices >= full_episode_count)
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(
            "episode_indices must be unique, increasing, and inside the full sequence"
        )
    return indices, full_episode_count


def _validate_chronological_dates(
    episode_dates: Sequence[date],
    *,
    observations: int,
) -> None:
    dates = tuple(episode_dates)
    if len(dates) != observations:
        raise ValueError("episode_dates must align with episode losses")
    if any(not isinstance(value, date) for value in dates):
        raise TypeError("episode_dates must contain date values")
    if tuple(sorted(set(dates))) != dates:
        raise ValueError("episode_dates must be unique and chronological")
