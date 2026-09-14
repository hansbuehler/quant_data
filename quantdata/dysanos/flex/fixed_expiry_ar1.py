"""Coordinatewise AR(1) innovations for fixed-expiry DLV tower rolls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Sequence

import numpy as np
import torch
from cdxcore.err import verify_inp

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs


@dataclass(frozen=True)
class FixedExpiryAr1Diagnostics:
    """Training-only diagnostics for fixed-expiry rollover innovations."""

    training_end: str
    sample_count: int
    residual_dimension: int
    rank: int
    calendar_time_increment_y: float
    explained_variance_fraction: float
    residual_covariance_eigenvalues: tuple[float, ...]
    num_zero_variance_predictors: int
    maximum_absolute_centered_residual_mean: float
    maximum_absolute_residual_predictor_covariance: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible diagnostics."""

        return asdict(self)


@dataclass(frozen=True)
class FixedExpiryAr1Fit:
    r"""Coordinatewise rollover means and a joint innovation factorization.

    ``conditional_intercept`` and ``conditional_slope`` have shape
    ``(3, inner)`` and describe

    ``endpoint[1:4] = intercept + slope * origin[2:5] + residual``

    in one-based expiry notation. ``weekly_innovation_loadings`` factor the
    rank-truncated weekly residual covariance. ``raw_log_loadings`` have shape
    ``(4, inner, rank)`` and are annualized for direct use by the analytic
    fixed-contract LVsq drift; their front-expiry row is exactly zero.
    """

    conditional_intercept: np.ndarray
    conditional_slope: np.ndarray
    centered_residuals: np.ndarray
    residual_covariance: np.ndarray
    residual_covariance_eigenvalues: np.ndarray
    weekly_innovation_loadings: np.ndarray
    raw_log_loadings: np.ndarray
    calendar_time_increment_y: float
    diagnostics: FixedExpiryAr1Diagnostics


def _extract_raw_log_dlvsq_tower(
    raw_states: np.ndarray,
    *,
    num_inner_strikes: int | None,
) -> np.ndarray:
    """Return four expiry-major raw log-DLV slabs from native raw states."""

    values = np.asarray(raw_states, dtype=np.float64)
    if values.ndim == 2:
        verify_inp(
            isinstance(num_inner_strikes, int) and num_inner_strikes > 0,
            lambda: (
                "'num_inner_strikes' must be supplied for flattened native "
                f"raw states, found {num_inner_strikes!r}"
            ),
        )
        verify_inp(
            values.shape[1] % 4 == 0,
            lambda: (
                "flattened raw-state width must contain four equal expiry "
                f"blocks, found shape {values.shape}"
            ),
        )
        block_width = values.shape[1] // 4
        verify_inp(
            block_width >= num_inner_strikes,
            lambda: (
                f"expiry block width {block_width} is smaller than the "
                f"requested {num_inner_strikes} raw DLV coordinates"
            ),
        )
        tower = values.reshape(len(values), 4, block_width)[:, :, :num_inner_strikes]
    elif values.ndim == 3:
        verify_inp(
            values.shape[1] == 4,
            lambda: (
                f"blocked raw states must have shape (dates, 4, block_width), found {values.shape}"
            ),
        )
        inner = values.shape[2] if num_inner_strikes is None else num_inner_strikes
        verify_inp(
            isinstance(inner, int) and 0 < inner <= values.shape[2],
            lambda: (
                "'num_inner_strikes' must select a positive prefix of each "
                f"raw expiry block, found {num_inner_strikes!r} for {values.shape}"
            ),
        )
        tower = values[:, :, :inner]
    else:
        raise ValueError(
            "raw_states must have shape (dates, native_state_size) or (dates, 4, block_width)"
        )
    verify_inp(
        len(tower) > 0 and np.all(np.isfinite(tower)),
        lambda: "raw log-DLV states must be non-empty and finite",
    )
    return np.array(tower, dtype=np.float64, copy=True)


def _orient_eigenvectors(eigenvectors: np.ndarray) -> np.ndarray:
    """Choose a deterministic sign for each covariance eigenvector."""

    output = eigenvectors.copy()
    pivot = np.argmax(np.abs(output), axis=0)
    signs = np.sign(output[pivot, np.arange(output.shape[1])])
    signs[signs == 0.0] = 1.0
    output *= signs[None]
    return output


def fit_fixed_expiry_ar1(
    raw_states: np.ndarray,
    dates: Sequence[date],
    *,
    num_inner_strikes: int | None = None,
    rank: int = 25,
    training_end: date = date(2024, 12, 31),
    calendar_time_increment_y: float = 5.0 / DAYS_PER_YEAR,
) -> FixedExpiryAr1Fit:
    r"""Fit fixed-expiry rollover means and joint raw log-DLV loadings.

    Weekly pairs are exact five-session NYSE pairs. For each pair, the target
    consists of endpoint expiry slabs 1--3 and the predictor consists of origin
    expiry slabs 2--4. Each coordinate receives its own affine AR(1)
    conditional mean. PCA is then applied jointly to the centered residual
    vectors, in expiry-major order, using population covariance normalization.

    Parameters
    ----------
    raw_states:
        Either blocked raw states with shape ``(dates, 4, block_width)`` or
        flattened native states with four equally sized expiry blocks. DLV
        coordinates must be the leading coordinates in every block.
    dates:
        Unique increasing observation dates.
    num_inner_strikes:
        Number of leading raw DLV coordinates per expiry. This is required for
        flattened native states and optional for DLV-only blocked states.
    rank:
        Number of joint residual covariance factors to retain.
    training_end:
        Last permitted endpoint date. No later state enters either side of the
        fitted weekly pairs.
    calendar_time_increment_y:
        Year fraction of one weekly roll. The returned ``raw_log_loadings`` are
        divided by its square root for use as instantaneous SDE loadings.
    """

    date_values = tuple(dates)
    verify_inp(
        bool(date_values)
        and all(isinstance(value, date) for value in date_values)
        and date_values == tuple(sorted(set(date_values))),
        lambda: "dates must be non-empty, unique, increasing date values",
    )
    verify_inp(
        isinstance(training_end, date),
        lambda: f"'training_end' must be a date, found {training_end!r}",
    )
    verify_inp(
        isinstance(rank, int) and not isinstance(rank, bool) and rank > 0,
        lambda: f"'rank' must be a positive integer, found {rank!r}",
    )
    verify_inp(
        np.isfinite(calendar_time_increment_y) and calendar_time_increment_y > 0.0,
        lambda: (
            "'calendar_time_increment_y' must be finite and positive, found "
            f"{calendar_time_increment_y!r}"
        ),
    )
    tower = _extract_raw_log_dlvsq_tower(
        raw_states,
        num_inner_strikes=num_inner_strikes,
    )
    verify_inp(
        len(tower) == len(date_values),
        lambda: (
            f"raw states contain {len(tower)} dates but dates contains {len(date_values)} entries"
        ),
    )

    try:
        origin, endpoint = exact_weekly_pairs(date_values, weeks=1)
    except KeyError as error:
        raise ValueError("dates must all be NYSE business sessions") from error
    training_pair = np.asarray(
        [date_values[int(index)] <= training_end for index in endpoint],
        dtype=bool,
    )
    origin = origin[training_pair]
    endpoint = endpoint[training_pair]
    sample_count = len(origin)
    residual_dimension = 3 * tower.shape[2]
    verify_inp(
        sample_count > rank and rank <= residual_dimension,
        lambda: (
            "fixed-expiry AR fit needs more training pairs than retained "
            f"factors and rank <= {residual_dimension}; found "
            f"{sample_count} pairs and rank {rank}"
        ),
    )

    predictor = tower[origin, 1:]
    target = tower[endpoint, :3]
    predictor_mean = np.mean(predictor, axis=0)
    target_mean = np.mean(target, axis=0)
    centered_predictor = predictor - predictor_mean
    centered_target = target - target_mean
    predictor_sum_squares = np.sum(np.square(centered_predictor), axis=0)
    predictor_scale = np.maximum(
        np.sum(np.square(predictor), axis=0),
        1.0,
    )
    variable_predictor = predictor_sum_squares > (
        np.finfo(np.float64).eps * predictor_scale * sample_count
    )
    slope = np.zeros_like(predictor_sum_squares)
    slope[variable_predictor] = (
        np.sum(centered_predictor * centered_target, axis=0)[variable_predictor]
        / predictor_sum_squares[variable_predictor]
    )
    intercept = target_mean - slope * predictor_mean
    residual = target - (intercept + slope * predictor)
    centered_residual = residual - np.mean(residual, axis=0)
    residual_vectors = centered_residual.reshape(sample_count, residual_dimension)

    covariance = residual_vectors.T @ residual_vectors / sample_count
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = _orient_eigenvectors(eigenvectors[:, order])
    tolerance = (
        np.finfo(np.float64).eps * max(float(np.max(np.abs(eigenvalues))), 1.0) * residual_dimension
    )
    if float(np.min(eigenvalues)) < -tolerance:
        raise FloatingPointError("residual covariance is not positive semidefinite")
    nonnegative_eigenvalues = np.maximum(eigenvalues, 0.0)
    total_variance = float(np.sum(nonnegative_eigenvalues))
    verify_inp(
        np.isfinite(total_variance) and total_variance > 0.0,
        lambda: "fixed-expiry rollover residuals have zero or invalid variance",
    )
    weekly_loadings = eigenvectors[:, :rank] * np.sqrt(nonnegative_eigenvalues[:rank])[None]
    raw_log_loadings = np.zeros(
        (4, tower.shape[2], rank),
        dtype=np.float64,
    )
    raw_log_loadings[1:] = weekly_loadings.reshape(3, tower.shape[2], rank) / np.sqrt(
        calendar_time_increment_y
    )

    predictor_covariance = np.mean(
        centered_residual * centered_predictor,
        axis=0,
    )
    explained_variance = float(np.sum(nonnegative_eigenvalues[:rank]) / total_variance)
    diagnostics = FixedExpiryAr1Diagnostics(
        training_end=training_end.isoformat(),
        sample_count=sample_count,
        residual_dimension=residual_dimension,
        rank=rank,
        calendar_time_increment_y=float(calendar_time_increment_y),
        explained_variance_fraction=explained_variance,
        residual_covariance_eigenvalues=tuple(float(value) for value in eigenvalues),
        num_zero_variance_predictors=int(np.sum(~variable_predictor)),
        maximum_absolute_centered_residual_mean=float(
            np.max(np.abs(np.mean(residual_vectors, axis=0)))
        ),
        maximum_absolute_residual_predictor_covariance=float(np.max(np.abs(predictor_covariance))),
    )
    return FixedExpiryAr1Fit(
        conditional_intercept=intercept,
        conditional_slope=slope,
        centered_residuals=residual_vectors,
        residual_covariance=covariance,
        residual_covariance_eigenvalues=eigenvalues,
        weekly_innovation_loadings=weekly_loadings,
        raw_log_loadings=raw_log_loadings,
        calendar_time_increment_y=float(calendar_time_increment_y),
        diagnostics=diagnostics,
    )


def torch_fixed_expiry_raw_log_loadings(
    fit: FixedExpiryAr1Fit,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Move fitted analytic-drift loadings to one Torch dtype and device."""

    return torch.as_tensor(fit.raw_log_loadings, dtype=dtype, device=device)


__all__ = [
    "FixedExpiryAr1Diagnostics",
    "FixedExpiryAr1Fit",
    "fit_fixed_expiry_ar1",
    "torch_fixed_expiry_raw_log_loadings",
]
