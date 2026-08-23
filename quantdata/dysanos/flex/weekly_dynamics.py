"""Weekly hierarchical physical-measure dynamics for the FLEX state."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8, unique_label48_8
from cdxcore.util import is_filename
from cdxcore.verbose import Context
from cdxcore.version import version

from quantdata.biz_calendar import get_business_days
from quantdata.dysanos.flex.config import FlexMarketConfig, ForwardMuFlexGeometryConfig
from quantdata.dysanos.market import smooth_interpolated_market
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.widb.options_filter import OptionsFilter

STRONG_STATE_SIZES: tuple[int, ...] = (
    16,
    19,
    17,
    15,
    13,
    10,
    10,
    10,
    5,
    5,
    5,
    5,
    5,
    5,
)
PHASE3_TRAIN_START = date(2022, 1, 3)
PHASE3_TRAIN_END = date(2024, 12, 31)
PHASE3_VALIDATION_START = date(2025, 1, 2)
PHASE3_VALIDATION_END = date(2025, 8, 29)


def _validate_identity(label: str, group: str) -> None:
    group_path = Path(group)
    verify_inp(
        isinstance(label, str)
        and bool(label)
        and (label.startswith("*") or is_filename(label)),
        lambda: f"invalid model label {label!r}",
    )
    verify_inp(
        not group_path.is_absolute()
        and all(
            part not in {"", ".", ".."} and is_filename(part)
            for part in group_path.parts
        ),
        lambda: f"invalid relative model group {group!r}",
    )


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class WeeklyDynamicsConfig:
    """Logical geometry and regularization of the weekly state model.

    The PCA threshold, ridge penalty, and numerical gates are explicit ad-hoc
    choices. They must be reported and sensitivity-tested rather than treated
    as financial constants.
    """

    label: str
    group: str = "flex/weekly-dynamics"
    user_version: str = "0.1"
    state_sizes_by_expiry: tuple[int, ...] = STRONG_STATE_SIZES
    weekly_session_stride: int = 5
    pca_mode: str = "reduced"
    pca_variance_threshold: float = 0.995
    ridge_penalty: float = 0.0
    covariance_eigenvalue_floor: float = 1.0e-10
    maximum_transition_condition_number: float = 1.0e8
    generator_roundtrip_relative_tolerance: float = 1.0e-8
    maximum_standardized_transition_power_norm: float = 10.0
    minimum_stationary_variance_trace_ratio: float = 0.01
    maximum_stationary_variance_trace_ratio: float = 100.0

    def __post_init__(self) -> None:
        _validate_identity(self.label, self.group)
        verify_inp(
            self.pca_mode in {"full", "reduced"},
            lambda: f"pca_mode must be 'full' or 'reduced', got {self.pca_mode!r}",
        )
        verify_inp(
            len(self.state_sizes_by_expiry) > 0
            and all(size > 0 for size in self.state_sizes_by_expiry),
            lambda: f"invalid expiry state allocation {self.state_sizes_by_expiry}",
        )
        verify_inp(
            self.weekly_session_stride > 0,
            lambda: f"weekly_session_stride must be positive, got {self.weekly_session_stride}",
        )
        verify_inp(
            0.0 < self.pca_variance_threshold <= 1.0,
            lambda: f"invalid PCA threshold {self.pca_variance_threshold}",
        )
        verify_inp(
            self.ridge_penalty >= 0.0,
            lambda: f"ridge_penalty must be non-negative, got {self.ridge_penalty}",
        )
        verify_inp(
            self.covariance_eigenvalue_floor >= 0.0,
            lambda: (
                "covariance_eigenvalue_floor must be non-negative, got "
                f"{self.covariance_eigenvalue_floor}"
            ),
        )
        verify_inp(
            self.maximum_transition_condition_number > 1.0,
            lambda: (
                "maximum_transition_condition_number must exceed one, got "
                f"{self.maximum_transition_condition_number}"
            ),
        )
        verify_inp(
            0.0 < self.generator_roundtrip_relative_tolerance < 1.0,
            lambda: (
                "invalid generator round-trip tolerance "
                f"{self.generator_roundtrip_relative_tolerance}"
            ),
        )
        verify_inp(
            self.maximum_standardized_transition_power_norm >= 1.0,
            lambda: (
                "maximum standardized transition-power norm must be at least "
                f"one, got {self.maximum_standardized_transition_power_norm}"
            ),
        )
        verify_inp(
            0.0 < self.minimum_stationary_variance_trace_ratio
            < self.maximum_stationary_variance_trace_ratio,
            lambda: (
                "stationary variance trace-ratio bounds must be positive and "
                "ordered, got "
                f"{self.minimum_stationary_variance_trace_ratio} and "
                f"{self.maximum_stationary_variance_trace_ratio}"
            ),
        )

    @property
    def state_size(self) -> int:
        return sum(self.state_sizes_by_expiry)

    @property
    def filename(self) -> str:
        if self.label.startswith("*"):
            return unique_label48_8(self.label[1:], as_file_name=True)
        return named_unique_filename48_8(self.label, self)


@dataclass(frozen=True)
class ExpiryPcaFit:
    """Full spectrum and selected level-PCA decoder for one expiry block."""

    mean: np.ndarray
    full_components: np.ndarray
    explained_variance: np.ndarray
    reduced_rank: int

    @property
    def reduced_components(self) -> np.ndarray:
        return self.full_components[: self.reduced_rank]


@dataclass(frozen=True)
class HierarchicalBlockFit:
    """One expiry block of the lag-only one-session affine transition."""

    intercept: np.ndarray
    own_lag: np.ndarray
    previous_lag: np.ndarray | None
    observations: int
    design_columns: int
    design_rank: int
    design_condition: float
    underidentified: bool
    r2: float
    generator_roundtrip_relative_error: float
    mean_reversion_per_week: np.ndarray


@dataclass(frozen=True)
class WeeklyDynamicsFit:
    """Fitted stationary surface dynamics and joint spot innovations."""

    config: WeeklyDynamicsConfig
    pca_by_expiry: tuple[ExpiryPcaFit, ...]
    blocks: tuple[HierarchicalBlockFit, ...]
    reduced_transition: np.ndarray
    mean_reversion_generator: np.ndarray
    equilibrium_mean: np.ndarray
    weekly_joint_innovation_covariance: np.ndarray
    stationary_surface_covariance: np.ndarray
    weekly_log_spot_drift: float
    weekly_spot_front_beta: np.ndarray
    weekly_spot_orthogonal_std: float
    weekly_transition_observations: int
    training_rows: int
    diagnostics: dict[str, Any]

    @property
    def reduced_state_size(self) -> int:
        return sum(pca.reduced_rank for pca in self.pca_by_expiry)

    @property
    def reduced_sizes_by_expiry(self) -> tuple[int, ...]:
        return tuple(pca.reduced_rank for pca in self.pca_by_expiry)


@dataclass(frozen=True)
class WeeklyTrainingHistory:
    """Date-aligned strong states and observed reference spot levels."""

    dates: tuple[date, ...]
    states: np.ndarray
    log_spot: np.ndarray

    def __post_init__(self) -> None:
        verify_inp(
            len(self.dates) > 0
            and self.dates == tuple(sorted(set(self.dates)))
            and self.states.ndim == 2
            and self.states.shape[0] == len(self.dates)
            and self.log_spot.shape == (len(self.dates),)
            and np.all(np.isfinite(self.states))
            and np.all(np.isfinite(self.log_spot)),
            lambda: (
                "weekly history must contain finite date-aligned arrays; got "
                f"{len(self.dates)}/{self.states.shape}/{self.log_spot.shape}"
            ),
        )

    def subset(self, start: date, end: date) -> WeeklyTrainingHistory:
        """Return a strict calendar interval without mixing train/validation."""
        if end < start:
            raise ValueError(f"invalid history interval {start}..{end}")
        selected = np.asarray(
            [start <= value <= end for value in self.dates], dtype=bool
        )
        return WeeklyTrainingHistory(
            dates=tuple(value for value, keep in zip(self.dates, selected) if keep),
            states=self.states[selected],
            log_spot=self.log_spot[selected],
        )


def _orient_components(components: np.ndarray) -> np.ndarray:
    output = components.copy()
    for index, component in enumerate(output):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0.0:
            output[index] *= -1.0
    return output


def _fit_level_pca(
    block: np.ndarray,
    threshold: float,
    *,
    mode: str,
) -> tuple[ExpiryPcaFit, np.ndarray]:
    mean = np.mean(block, axis=0)
    centered = block - mean
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    total = float(np.sum(np.square(singular_values)))
    cumulative = (
        np.cumsum(np.square(singular_values)) / total
        if total > 0.0
        else np.ones_like(singular_values)
    )
    if mode == "full" and len(singular_values) < block.shape[1]:
        raise ValueError(
            "full per-expiry PCA requires at least as many level observations "
            f"as state coordinates; got {block.shape}"
        )
    rank = (
        block.shape[1]
        if mode == "full"
        else min(int(np.searchsorted(cumulative, threshold) + 1), block.shape[1])
    )
    full = _orient_components(right)
    scores = centered @ full[:rank].T
    assert scores.shape == (block.shape[0], rank)
    return (
        ExpiryPcaFit(
            mean=mean,
            full_components=full,
            explained_variance=cumulative,
            reduced_rank=rank,
        ),
        scores,
    )


def _ridge_coefficients(design: np.ndarray, target: np.ndarray, penalty: float) -> np.ndarray:
    if penalty == 0.0:
        return np.linalg.lstsq(design, target, rcond=None)[0]
    gram = design.T @ design
    regularizer = penalty * np.eye(design.shape[1], dtype=design.dtype)
    regularizer[0, 0] = 0.0
    return np.linalg.solve(gram + regularizer, design.T @ target)


def _verified_generator(
    transition: np.ndarray,
    *,
    maximum_condition: float,
    roundtrip_tolerance: float,
) -> tuple[np.ndarray, float, float, float] | None:
    """Return ``K=-log(A)`` only when its exponential reliably recovers ``A``."""
    from scipy.linalg import expm, logm

    condition = float(np.linalg.cond(transition))
    if not np.isfinite(condition) or condition > maximum_condition:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        # SciPy 1.18 removed ``disp`` and the optional error-estimate return.
        # Compute the same one-norm round-trip diagnostic explicitly so this
        # correctness gate has identical semantics across supported versions.
        logarithm = logm(transition)
    imaginary = float(np.max(np.abs(logarithm.imag)))
    scale = max(1.0, float(np.max(np.abs(logarithm.real))))
    if imaginary > 1.0e-8 * scale:
        return None
    generator = -logarithm.real
    eigenvalues = np.linalg.eigvals(generator)
    if not np.all(np.isfinite(generator)) or np.min(eigenvalues.real) <= 0.0:
        return None
    recovered = expm(-generator)
    estimated_error = float(
        np.linalg.norm(recovered - transition, ord=1)
        / max(np.linalg.norm(transition, ord=1), np.finfo(float).eps)
    )
    roundtrip_error = float(
        np.linalg.norm(recovered - transition)
        / max(np.linalg.norm(transition), np.finfo(float).eps)
    )
    if not np.isfinite(roundtrip_error) or roundtrip_error > roundtrip_tolerance:
        return None
    return generator, condition, roundtrip_error, float(estimated_error)


def _transition_power_diagnostics(
    transition: np.ndarray,
    *,
    maximum_weeks: int = 1040,
) -> tuple[float, int, dict[int, float]]:
    """Measure non-normal amplification in the fitted standardized metric."""
    requested = {1, 4, 52, 260, 520, maximum_weeks}
    power = np.eye(transition.shape[0], dtype=np.float64)
    maximum_norm = 0.0
    maximum_index = 0
    selected: dict[int, float] = {}
    for week in range(1, maximum_weeks + 1):
        power = power @ transition
        norm = float(np.linalg.norm(power, ord=2))
        if norm > maximum_norm:
            maximum_norm = norm
            maximum_index = week
        if week in requested:
            selected[week] = norm
    return maximum_norm, maximum_index, selected


def fit_weekly_hierarchical_dynamics(
    states: np.ndarray,
    log_spot: np.ndarray,
    *,
    config: WeeklyDynamicsConfig,
    dates: tuple[date, ...] | None = None,
) -> WeeklyDynamicsFit:
    """Fit a DYSANOS-style lag-only matrix-exponential model.

    The one-session affine transition for expiry ``j`` uses only its own lag
    and expiry ``j-1``'s lag. The complete non-symmetric transition is fitted
    in standardized PCA coordinates, and its real stable matrix logarithm is
    compounded to the exact five-session Torch step. No contemporaneous
    expiry regressor, blockwise symmetrization, eigenvalue clipping, or silent
    stability fallback is used. A single state block is the direct/global
    full-K special case; multiple blocks impose the expiry hierarchy.
    """
    from scipy.linalg import expm, logm, solve_discrete_lyapunov

    states = np.asarray(states, dtype=np.float64)
    log_spot = np.asarray(log_spot, dtype=np.float64)
    verify_inp(
        states.ndim == 2
        and states.shape[1] == config.state_size
        and states.shape[0] >= config.weekly_session_stride + 3
        and log_spot.shape == (states.shape[0],)
        and np.all(np.isfinite(states))
        and np.all(np.isfinite(log_spot)),
        lambda: (
            "states/log_spot must be finite aligned arrays with state width "
            f"{config.state_size}; got {states.shape}/{log_spot.shape}"
        ),
    )

    pca_fits: list[ExpiryPcaFit] = []
    score_blocks: list[np.ndarray] = []
    start = 0
    for width in config.state_sizes_by_expiry:
        pca, scores = _fit_level_pca(
            states[:, start : start + width],
            config.pca_variance_threshold,
            mode=config.pca_mode,
        )
        pca_fits.append(pca)
        score_blocks.append(scores)
        start += width

    scores = np.concatenate(score_blocks, axis=1)
    score_center = np.mean(scores, axis=0)
    score_scale = np.std(scores, axis=0)
    scale_floor = math.sqrt(np.finfo(np.float64).eps) * max(
        1.0, float(np.max(score_scale))
    )
    if np.any(score_scale <= scale_floor):
        raise ValueError(
            "training PCA score has effectively zero variance; minimum/"
            f"floor={float(np.min(score_scale)):.6g}/{scale_floor:.6g}"
        )
    standardized = (scores - score_center) / score_scale

    stride = config.weekly_session_stride
    if dates is None:
        daily_origin = np.arange(states.shape[0] - 1, dtype=np.int64)
        daily_next = daily_origin + 1
        weekly_origin = np.arange(states.shape[0] - stride, dtype=np.int64)
        weekly_next = weekly_origin + stride
        daily_selection = "adjacent row; caller supplied no dates"
        weekly_selection = "row shift; caller supplied no dates"
    else:
        verify_inp(
            len(dates) == states.shape[0] and dates == tuple(sorted(set(dates))),
            lambda: "dates must align with states and be unique/increasing",
        )
        date_index = {value: index for index, value in enumerate(dates)}
        business_days = get_business_days(dates[0], dates[-1])
        business_index = {value: index for index, value in enumerate(business_days)}
        verify_inp(
            all(value in business_index for value in dates),
            lambda: "every fitted date must be an NYSE business session",
        )
        daily_pairs = [
            (index, date_index[business_days[position + 1]])
            for index, value in enumerate(dates)
            if (position := business_index[value]) + 1 < len(business_days)
            and business_days[position + 1] in date_index
        ]
        weekly_pairs = [
            (index, date_index[business_days[position + stride]])
            for index, value in enumerate(dates)
            if (position := business_index[value]) + stride < len(business_days)
            and business_days[position + stride] in date_index
        ]
        verify_inp(
            len(daily_pairs) >= 3 and len(weekly_pairs) >= 3,
            lambda: (
                "fewer than three exact daily/weekly transitions remain after "
                f"calendar matching: {len(daily_pairs)}/{len(weekly_pairs)}"
            ),
        )
        daily_origin = np.asarray(
            [pair[0] for pair in daily_pairs], dtype=np.int64
        )
        daily_next = np.asarray([pair[1] for pair in daily_pairs], dtype=np.int64)
        weekly_origin = np.asarray(
            [pair[0] for pair in weekly_pairs], dtype=np.int64
        )
        weekly_next = np.asarray(
            [pair[1] for pair in weekly_pairs], dtype=np.int64
        )
        daily_selection = "exact adjacent NYSE business-session endpoints"
        weekly_selection = "exact five-NYSE-session endpoints"

    sizes = [block.shape[1] for block in score_blocks]
    offsets = np.cumsum((0, *sizes))
    reduced_size = int(offsets[-1])
    daily_transition = np.zeros((reduced_size, reduced_size), dtype=np.float64)
    daily_intercept = np.zeros(reduced_size, dtype=np.float64)
    block_payloads: list[dict[str, Any]] = []
    for expiry_index in range(len(sizes)):
        current_slice = slice(offsets[expiry_index], offsets[expiry_index + 1])
        design_parts = [
            np.ones((len(daily_origin), 1), dtype=np.float64),
            standardized[daily_origin, current_slice],
        ]
        source_slices = [current_slice]
        if expiry_index > 0:
            previous_slice = slice(offsets[expiry_index - 1], offsets[expiry_index])
            design_parts.append(standardized[daily_origin, previous_slice])
            source_slices.append(previous_slice)
        design = np.concatenate(design_parts, axis=1)
        target = standardized[daily_next, current_slice]
        coefficients = _ridge_coefficients(
            design, target, config.ridge_penalty
        )
        prediction = design @ coefficients
        residual = target - prediction
        singular = np.linalg.svd(design, compute_uv=False)
        tolerance = np.finfo(float).eps * max(design.shape) * singular[0]
        design_rank = int(np.sum(singular > tolerance))
        if design_rank < design.shape[1]:
            raise ValueError(
                "lag-only daily hierarchy is underidentified at expiry "
                f"{expiry_index}: rank/columns={design_rank}/{design.shape[1]}"
            )
        daily_intercept[current_slice] = coefficients[0]
        cursor = 1
        fitted_sources: list[np.ndarray] = []
        for source_slice in source_slices:
            source_width = source_slice.stop - source_slice.start
            coefficient = coefficients[cursor : cursor + source_width]
            daily_transition[source_slice, current_slice] = coefficient
            fitted_sources.append(coefficient)
            cursor += source_width
        centered_target = target - np.mean(target, axis=0, keepdims=True)
        denominator = float(np.sum(np.square(centered_target)))
        block_payloads.append(
            {
                "intercept": coefficients[0],
                "own_lag": fitted_sources[0],
                "previous_lag": (
                    fitted_sources[1] if expiry_index > 0 else None
                ),
                "observations": len(daily_origin),
                "design_columns": design.shape[1],
                "design_rank": design_rank,
                "design_condition": float(singular[0] / singular[-1]),
                "r2": (
                    1.0 - float(np.sum(np.square(residual))) / denominator
                    if denominator > 0.0
                    else math.nan
                ),
            }
        )

    verified = _verified_generator(
        daily_transition,
        maximum_condition=config.maximum_transition_condition_number,
        roundtrip_tolerance=config.generator_roundtrip_relative_tolerance,
    )
    if verified is None:
        attempted = -logm(daily_transition)
        raise ValueError(
            "lag-only daily hierarchy has no reliable real stable matrix "
            "logarithm; no fallback was applied. condition/max-imag/min-real="
            f"{float(np.linalg.cond(daily_transition)):.6g}/"
            f"{float(np.max(np.abs(attempted.imag))):.6g}/"
            f"{float(np.min(np.linalg.eigvals(attempted).real)):.6g}"
        )
    (
        daily_generator_standardized,
        daily_transition_condition,
        generator_roundtrip_error,
        logm_error,
    ) = verified
    weekly_generator_standardized = stride * daily_generator_standardized
    weekly_transition_standardized = expm(-weekly_generator_standardized)
    weekly_discrete_roundtrip_error = float(
        np.linalg.norm(
            weekly_transition_standardized
            - np.linalg.matrix_power(daily_transition, stride)
        )
        / max(
            np.linalg.norm(weekly_transition_standardized),
            np.finfo(float).eps,
        )
    )
    transition_spectral_radius = float(
        np.max(np.abs(np.linalg.eigvals(weekly_transition_standardized)))
    )
    if transition_spectral_radius >= 1.0:
        raise ValueError(
            "lag-only hierarchical weekly transition is not stationary: "
            f"spectral radius={transition_spectral_radius:.12g}"
        )
    equilibrium_standardized = np.linalg.solve(
        (np.eye(reduced_size) - daily_transition).T,
        daily_intercept,
    )
    weekly_intercept_standardized = equilibrium_standardized @ (
        np.eye(reduced_size) - weekly_transition_standardized
    )

    # Map the fitted standardized dynamics back to the raw PCA scores consumed
    # by the existing Torch decoder. Similarity preserves eigenvalues and the
    # matrix-exponential identity; diagnostics remain in the fitted metric.
    scale_ratio = score_scale[None, :] / score_scale[:, None]
    transition = weekly_transition_standardized * scale_ratio
    generator = weekly_generator_standardized * scale_ratio
    equilibrium = score_center + equilibrium_standardized * score_scale
    expected_raw_intercept = equilibrium @ (np.eye(reduced_size) - transition)
    mapped_raw_intercept = weekly_intercept_standardized * score_scale + (
        score_center - score_center @ transition
    )
    if not np.allclose(
        expected_raw_intercept,
        mapped_raw_intercept,
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise RuntimeError("standardized affine transition mapping is inconsistent")

    structural_blocks: list[HierarchicalBlockFit] = []
    for expiry_index, payload in enumerate(block_payloads):
        block_slice = slice(offsets[expiry_index], offsets[expiry_index + 1])
        block_generator = daily_generator_standardized[block_slice, block_slice]
        block_roundtrip = float(
            np.linalg.norm(expm(-block_generator) - payload["own_lag"])
            / max(np.linalg.norm(payload["own_lag"]), np.finfo(float).eps)
        )
        structural_blocks.append(
            HierarchicalBlockFit(
                intercept=payload["intercept"],
                own_lag=payload["own_lag"],
                previous_lag=payload["previous_lag"],
                observations=payload["observations"],
                design_columns=payload["design_columns"],
                design_rank=payload["design_rank"],
                design_condition=payload["design_condition"],
                underidentified=False,
                r2=payload["r2"],
                generator_roundtrip_relative_error=block_roundtrip,
                mean_reversion_per_week=(
                    stride * np.linalg.eigvals(block_generator).real
                ),
            )
        )

    prediction = equilibrium + (scores[weekly_origin] - equilibrium) @ transition
    reduced_residual = scores[weekly_next] - prediction
    standardized_residual = reduced_residual / score_scale
    training_score_mean = np.mean(scores, axis=0)
    pair_endpoint_drift = np.mean(
        scores[weekly_next] - scores[weekly_origin], axis=0
    )
    reduced_residual_mean = np.mean(reduced_residual, axis=0)
    training_score_rms = math.sqrt(float(np.mean(np.square(scores))))
    standardized_training_rms = math.sqrt(
        float(np.mean(np.square(standardized)))
    )
    eigenvalues = np.linalg.eigvals(generator)
    off_diagonal = generator - np.diag(np.diag(generator))
    if float(np.max(np.abs(off_diagonal))) <= 1.0e-12:
        raise ValueError(
            "hierarchical fit produced a diagonal full mean-reversion generator; "
            "the expiry coupling is not identified"
        )
    # Estimate the complete weekly innovation covariance. The old independent
    # block assumption was contradicted by cross-expiry residual correlations
    # and is not part of the DYSANOS/grid construction.
    structural_correlations = np.corrcoef(
        standardized_residual, rowvar=False
    )
    standardized_covariance = np.atleast_2d(
        np.cov(standardized_residual, rowvar=False, bias=True)
    )
    standardized_covariance = 0.5 * (
        standardized_covariance + standardized_covariance.T
    )
    covariance_values, covariance_vectors = np.linalg.eigh(
        standardized_covariance
    )
    standardized_covariance = (
        covariance_vectors
        * np.maximum(
            covariance_values, config.covariance_eigenvalue_floor
        )[None, :]
    ) @ covariance_vectors.T
    reduced_covariance = standardized_covariance * (
        score_scale[:, None] * score_scale[None, :]
    )

    # Spot is a unit-root output driven only by the whitened front-expiry
    # structural innovation plus an always-retained orthogonal scalar shock.
    # This preserves autonomous stationary surface dynamics and the DYSANOS
    # surface-first timing convention.
    front_width = sizes[0]
    front_covariance = reduced_covariance[:front_width, :front_width]
    front_values, front_vectors = np.linalg.eigh(front_covariance)
    front_factor = front_vectors * np.sqrt(
        np.maximum(front_values, config.covariance_eigenvalue_floor)
    )[None, :]
    whitened_front = np.linalg.solve(
        front_factor, reduced_residual[:, :front_width].T
    ).T
    spot_return = log_spot[weekly_next] - log_spot[weekly_origin]
    spot_design = np.concatenate(
        (
            np.ones((len(weekly_origin), 1), dtype=np.float64),
            whitened_front,
        ),
        axis=1,
    )
    spot_coefficients = _ridge_coefficients(
        spot_design, spot_return[:, None], config.ridge_penalty
    )[:, 0]
    spot_drift = float(spot_coefficients[0])
    spot_beta = spot_coefficients[1:]
    spot_residual = spot_return - spot_design @ spot_coefficients
    spot_orthogonal_variance = max(
        float(np.mean(np.square(spot_residual))),
        config.covariance_eigenvalue_floor,
    )
    # The full surface covariance retains cross-expiry residual dependence.
    # A spot shock driven by the whitened front marginal therefore covaries
    # with every surface block through its covariance with that marginal.
    spot_cross_reduced = reduced_covariance[:, :front_width] @ np.linalg.solve(
        front_factor.T, spot_beta
    )
    covariance = np.zeros((reduced_size + 1, reduced_size + 1), dtype=np.float64)
    covariance[0, 0] = float(spot_beta @ spot_beta) + spot_orthogonal_variance
    covariance[0, 1:] = spot_cross_reduced
    covariance[1:, 0] = spot_cross_reduced
    covariance[1:, 1:] = reduced_covariance
    covariance = 0.5 * (covariance + covariance.T)
    stationary_standardized_covariance = solve_discrete_lyapunov(
        weekly_transition_standardized.T, standardized_covariance
    )
    stationary_covariance = stationary_standardized_covariance * (
        score_scale[:, None] * score_scale[None, :]
    )
    stationary_covariance = 0.5 * (
        stationary_covariance + stationary_covariance.T
    )
    if np.min(np.linalg.eigvalsh(stationary_covariance)) < -1.0e-10:
        raise ValueError("fitted stationary surface covariance is not positive")

    maximum_power_norm, maximum_power_week, power_norms = (
        _transition_power_diagnostics(weekly_transition_standardized)
    )
    stationary_trace_ratio = float(
        np.trace(stationary_standardized_covariance)
        / np.trace(np.cov(standardized, rowvar=False, bias=True))
    )
    if maximum_power_norm > config.maximum_standardized_transition_power_norm:
        raise ValueError(
            "standardized hierarchical transition fails the transient-power "
            f"gate: {maximum_power_norm:.6g} > "
            f"{config.maximum_standardized_transition_power_norm:.6g}"
        )
    if not (
        config.minimum_stationary_variance_trace_ratio
        <= stationary_trace_ratio
        <= config.maximum_stationary_variance_trace_ratio
    ):
        raise ValueError(
            "standardized stationary/training variance trace ratio fails its "
            f"gate: {stationary_trace_ratio:.6g}"
        )

    diagnostics: dict[str, Any] = {
        "training_rows": int(states.shape[0]),
        "daily_transition_observations": len(daily_origin),
        "weekly_transition_observations": len(weekly_origin),
        "full_state_size": config.state_size,
        "reduced_state_size": reduced_size,
        "reduced_sizes_by_expiry": sizes,
        "pca_variance_threshold": config.pca_variance_threshold,
        "pca_mode": config.pca_mode,
        "ridge_penalty": config.ridge_penalty,
        "covariance_eigenvalue_floor": config.covariance_eigenvalue_floor,
        "maximum_transition_condition_number": (
            config.maximum_transition_condition_number
        ),
        "generator_roundtrip_relative_tolerance": (
            config.generator_roundtrip_relative_tolerance
        ),
        "maximum_standardized_transition_power_norm": (
            config.maximum_standardized_transition_power_norm
        ),
        "minimum_stationary_variance_trace_ratio": (
            config.minimum_stationary_variance_trace_ratio
        ),
        "maximum_stationary_variance_trace_ratio": (
            config.maximum_stationary_variance_trace_ratio
        ),
        "estimator": "DYSANOS/grid lag-only daily affine matrix logarithm",
        "daily_transition_condition_number": daily_transition_condition,
        "transition_condition_number": float(np.linalg.cond(transition)),
        "generator_roundtrip_relative_error": generator_roundtrip_error,
        "weekly_discrete_roundtrip_relative_error": (
            weekly_discrete_roundtrip_error
        ),
        "scipy_logm_estimated_error": logm_error,
        "score_standardization_minimum_scale": float(np.min(score_scale)),
        "score_standardization_maximum_scale": float(np.max(score_scale)),
        "standardized_transition_norm": float(
            np.linalg.norm(weekly_transition_standardized, ord=2)
        ),
        "standardized_transition_maximum_power_norm": maximum_power_norm,
        "standardized_transition_maximum_power_week": maximum_power_week,
        "standardized_transition_power_norms": {
            str(week): norm for week, norm in power_norms.items()
        },
        "standardized_stationary_variance_trace_ratio": (
            stationary_trace_ratio
        ),
        "pca_by_expiry": [
            {
                "expiry_index": index,
                "full_rank": len(pca.mean),
                "reduced_rank": pca.reduced_rank,
                "retained_variance": float(
                    pca.explained_variance[pca.reduced_rank - 1]
                ),
            }
            for index, pca in enumerate(pca_fits)
        ],
        "hierarchy_by_expiry": [
            {
                "expiry_index": index,
                "r2": block.r2,
                "design_columns": block.design_columns,
                "design_rank": block.design_rank,
                "observations_per_design_column": (
                    block.observations / block.design_columns
                ),
                "design_condition": block.design_condition,
                "underidentified": block.underidentified,
                "generator_roundtrip_relative_error": (
                    block.generator_roundtrip_relative_error
                ),
                "minimum_mean_reversion_per_week": float(
                    np.min(block.mean_reversion_per_week)
                ),
                "maximum_mean_reversion_per_week": float(
                    np.max(block.mean_reversion_per_week)
                ),
                "maximum_half_life_weeks": float(
                    math.log(2.0)
                    / np.min(block.mean_reversion_per_week)
                ),
                "structural_residual_mean_l2": float(
                    np.linalg.norm(
                        np.mean(
                            standardized_residual[
                                :, offsets[index] : offsets[index + 1]
                            ],
                            axis=0,
                        )
                    )
                ),
                "structural_residual_mean_max_abs": float(
                    np.max(
                        np.abs(
                            np.mean(
                                standardized_residual[
                                    :, offsets[index] : offsets[index + 1]
                                ],
                                axis=0,
                            )
                        )
                    )
                ),
            }
            for index, block in enumerate(structural_blocks)
        ],
        "underidentified_expiries": [
            index for index, block in enumerate(structural_blocks) if block.underidentified
        ],
        "minimum_generator_real_eigenvalue": float(np.min(eigenvalues.real)),
        "maximum_transition_spectral_radius": float(
            transition_spectral_radius
        ),
        "maximum_generator_off_diagonal": float(np.max(np.abs(off_diagonal))),
        "equilibrium_constraint": (
            "daily affine equilibrium estimated in training-standardized PCA coordinates"
        ),
        "equilibrium_l2": float(np.linalg.norm(equilibrium)),
        "training_score_mean_max_abs": float(
            np.max(np.abs(training_score_mean))
        ),
        "training_score_rms": training_score_rms,
        "standardized_training_score_rms": standardized_training_rms,
        "equilibrium_distance_over_training_rms": float(
            np.linalg.norm(equilibrium_standardized)
            / max(standardized_training_rms, np.finfo(float).eps)
        ),
        "pair_endpoint_drift_l2": float(np.linalg.norm(pair_endpoint_drift)),
        "pair_endpoint_drift_max_abs": float(
            np.max(np.abs(pair_endpoint_drift))
        ),
        "reduced_residual_mean_l2": float(
            np.linalg.norm(reduced_residual_mean)
        ),
        "reduced_residual_mean_max_abs": float(
            np.max(np.abs(reduced_residual_mean))
        ),
        "spot_front_explained_fraction": float(
            (spot_beta @ spot_beta) / covariance[0, 0]
        ),
        "spot_orthogonal_std": math.sqrt(spot_orthogonal_variance),
        "maximum_structural_cross_expiry_residual_correlation": float(
            max(
                (
                    np.max(np.abs(structural_correlations[
                        offsets[left] : offsets[left + 1],
                        offsets[right] : offsets[right + 1],
                    ]))
                    for left in range(len(sizes))
                    for right in range(left + 1, len(sizes))
                ),
                default=0.0,
            )
        ),
        "origin_weighting": (
            "equal weight per exact daily origin for drift and per exact weekly "
            "origin for innovations"
        ),
        "daily_transition_selection": daily_selection,
        "transition_selection": weekly_selection,
        "innovation_covariance": "unrestricted full weekly residual covariance",
    }
    return WeeklyDynamicsFit(
        config=config,
        pca_by_expiry=tuple(pca_fits),
        blocks=tuple(structural_blocks),
        reduced_transition=transition,
        mean_reversion_generator=generator,
        equilibrium_mean=equilibrium,
        weekly_joint_innovation_covariance=covariance,
        stationary_surface_covariance=stationary_covariance,
        weekly_log_spot_drift=spot_drift,
        weekly_spot_front_beta=spot_beta,
        weekly_spot_orthogonal_std=math.sqrt(spot_orthogonal_variance),
        weekly_transition_observations=len(weekly_origin),
        training_rows=states.shape[0],
        diagnostics=diagnostics,
    )


def fit_phase3_training_history(
    history: WeeklyTrainingHistory,
    *,
    config: WeeklyDynamicsConfig,
) -> WeeklyDynamicsFit:
    """Fit only the locked 2022--2024 training interval.

    Validation observations from 2025 are rejected rather than silently used
    to estimate PCA bases, means, regressions, or innovation covariances.
    """
    if not history.dates or any(
        value < PHASE3_TRAIN_START or value > PHASE3_TRAIN_END
        for value in history.dates
    ):
        raise ValueError(
            "Phase 3 fitting accepts only training dates in "
            f"{PHASE3_TRAIN_START}..{PHASE3_TRAIN_END}"
        )
    return fit_weekly_hierarchical_dynamics(
        history.states,
        history.log_spot,
        dates=history.dates,
        config=config,
    )


@dataclass(frozen=True)
class WeeklyStep:
    """One differentiable weekly transition and its Gaussian spot quantile."""

    reduced_state: torch.Tensor
    normal_log_spot: torch.Tensor
    spot_standard_normal: torch.Tensor


class TorchWeeklyDynamics(BaseModel):
    """Torch-native sampler for a fitted weekly hierarchical state model."""

    def __init__(self, env: BaseEnvironment, *, fit: WeeklyDynamicsFit):
        super().__init__(env)
        self.config = fit.config
        self.fit_diagnostics = fit.diagnostics
        self.reduced_sizes_by_expiry = fit.reduced_sizes_by_expiry
        self.register_buffer(
            "mean_reversion_generator",
            torch.as_tensor(fit.mean_reversion_generator, dtype=env.fdtype, device=env.device),
        )
        self.register_buffer(
            "equilibrium_mean",
            torch.as_tensor(fit.equilibrium_mean, dtype=env.fdtype, device=env.device),
        )
        self.register_buffer(
            "weekly_joint_covariance",
            torch.as_tensor(
                fit.weekly_joint_innovation_covariance,
                dtype=env.fdtype,
                device=env.device,
            ),
        )
        self.register_buffer(
            "weekly_log_spot_drift",
            torch.tensor(fit.weekly_log_spot_drift, dtype=env.fdtype, device=env.device),
        )
        weekly_surface_transition = torch.as_tensor(
            fit.reduced_transition, dtype=env.fdtype, device=env.device
        )
        self.register_buffer("weekly_surface_transition", weekly_surface_transition)
        joint_transition = torch.eye(
            fit.reduced_state_size + 1, dtype=env.fdtype, device=env.device
        )
        joint_transition[1:, 1:] = weekly_surface_transition
        self.register_buffer("weekly_joint_transition", joint_transition)
        weekly_covariance = torch.as_tensor(
            fit.weekly_joint_innovation_covariance,
            dtype=env.fdtype,
            device=env.device,
        )
        covariance_values, covariance_vectors = torch.linalg.eigh(weekly_covariance)
        self.register_buffer(
            "weekly_joint_factor",
            covariance_vectors
            * torch.sqrt(torch.clamp(covariance_values, min=0.0)).unsqueeze(0),
        )
        # Validation repeatedly uses 2W and 4W endpoints. Precompute their
        # exact propagated moments and covariance factors once; the simulation
        # step itself remains only matrix multiplies and a Gaussian factor.
        for weeks, prefix in ((2, "two_week"), (4, "four_week")):
            propagated_covariance = torch.zeros_like(weekly_covariance)
            power = torch.eye(
                fit.reduced_state_size + 1,
                dtype=env.fdtype,
                device=env.device,
            )
            for _ in range(weeks):
                propagated_covariance = (
                    propagated_covariance
                    + power.T @ weekly_covariance @ power
                )
                power = power @ joint_transition
            propagated_covariance = 0.5 * (
                propagated_covariance + propagated_covariance.T
            )
            values, vectors = torch.linalg.eigh(propagated_covariance)
            self.register_buffer(f"{prefix}_joint_transition", power)
            self.register_buffer(f"{prefix}_joint_covariance", propagated_covariance)
            self.register_buffer(
                f"{prefix}_joint_factor",
                vectors * torch.sqrt(torch.clamp(values, min=0.0)).unsqueeze(0),
            )
        self.register_buffer(
            "weekly_spot_innovation_std",
            torch.sqrt(
                torch.clamp(
                    weekly_covariance[0, 0], min=torch.finfo(env.fdtype).tiny
                )
            ),
        )
        stationary_covariance = torch.as_tensor(
            fit.stationary_surface_covariance,
            dtype=env.fdtype,
            device=env.device,
        )
        stationary_values, stationary_vectors = torch.linalg.eigh(
            stationary_covariance
        )
        self.register_buffer("stationary_surface_covariance", stationary_covariance)
        self.register_buffer(
            "stationary_surface_factor",
            stationary_vectors
            * torch.sqrt(torch.clamp(stationary_values, min=0.0)).unsqueeze(0),
        )
        full_center = np.concatenate([pca.mean for pca in fit.pca_by_expiry])
        decode = np.zeros((fit.reduced_state_size, fit.config.state_size))
        reduced_start = 0
        full_start = 0
        for pca, full_width in zip(fit.pca_by_expiry, fit.config.state_sizes_by_expiry):
            rank = pca.reduced_rank
            decode[
                reduced_start : reduced_start + rank,
                full_start : full_start + full_width,
            ] = pca.reduced_components
            reduced_start += rank
            full_start += full_width
        self.register_buffer(
            "full_state_center",
            torch.as_tensor(full_center, dtype=env.fdtype, device=env.device),
        )
        self.register_buffer(
            "full_state_decoder",
            torch.as_tensor(decode, dtype=env.fdtype, device=env.device),
        )

    @property
    def reduced_state_size(self) -> int:
        return self.mean_reversion_generator.shape[0]

    def decode_full_state(self, reduced_state: torch.Tensor) -> torch.Tensor:
        if reduced_state.shape[-1] != self.reduced_state_size:
            raise ValueError(
                f"expected reduced state width {self.reduced_state_size}, "
                f"got {tuple(reduced_state.shape)}"
            )
        return self.full_state_center + reduced_state @ self.full_state_decoder

    def sample_stationary_reduced_states(
        self,
        num_samples: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw the invariant surface state directly in Torch."""
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        normal = torch.randn(
            (num_samples, self.reduced_state_size),
            dtype=self.fdtype,
            device=self.device,
            generator=generator,
        )
        return self.equilibrium_mean + normal @ self.stationary_surface_factor.T

    def transition_matrix(self, *, step_weeks: int = 1) -> torch.Tensor:
        if not isinstance(step_weeks, int) or step_weeks <= 0:
            raise ValueError(f"step_weeks must be a positive integer, got {step_weeks!r}")
        if step_weeks == 1:
            return self.weekly_surface_transition
        if step_weeks == 2:
            return self.two_week_joint_transition[1:, 1:]
        if step_weeks == 4:
            return self.four_week_joint_transition[1:, 1:]
        return torch.linalg.matrix_power(self.weekly_surface_transition, step_weeks)

    def _joint_moments(self, step_weeks: int) -> tuple[torch.Tensor, torch.Tensor]:
        if step_weeks == 1:
            return self.weekly_joint_transition, self.weekly_joint_covariance
        if step_weeks == 2:
            return self.two_week_joint_transition, self.two_week_joint_covariance
        if step_weeks == 4:
            return self.four_week_joint_transition, self.four_week_joint_covariance
        joint_transition = self.weekly_joint_transition
        covariance = torch.zeros_like(self.weekly_joint_covariance)
        power = torch.eye(
            self.reduced_state_size + 1,
            dtype=self.fdtype,
            device=self.device,
        )
        for _ in range(step_weeks):
            covariance = covariance + power.T @ self.weekly_joint_covariance @ power
            power = power @ joint_transition
        return power, 0.5 * (covariance + covariance.T)

    def _joint_factor(self, step_weeks: int, covariance: torch.Tensor) -> torch.Tensor:
        if step_weeks == 1:
            return self.weekly_joint_factor
        if step_weeks == 2:
            return self.two_week_joint_factor
        if step_weeks == 4:
            return self.four_week_joint_factor
        values, vectors = torch.linalg.eigh(covariance)
        return vectors * torch.sqrt(torch.clamp(values, min=0.0)).unsqueeze(0)

    def step(
        self,
        reduced_state: torch.Tensor,
        log_spot: torch.Tensor,
        *,
        step_weeks: int = 1,
        standard_normals: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> WeeklyStep:
        if not isinstance(step_weeks, int) or step_weeks <= 0:
            raise ValueError(
                f"step_weeks must be a positive integer, got {step_weeks!r}"
            )
        if reduced_state.shape[:-1] != log_spot.shape:
            raise ValueError(
                "state batch dimensions must match log_spot: "
                f"{tuple(reduced_state.shape)}/{tuple(log_spot.shape)}"
            )
        transition, covariance = self._joint_moments(step_weeks)
        factor = self._joint_factor(step_weeks, covariance)
        expected_noise_shape = reduced_state.shape[:-1] + (self.reduced_state_size + 1,)
        if standard_normals is None:
            standard_normals = torch.randn(
                expected_noise_shape,
                dtype=self.fdtype,
                device=self.device,
                generator=generator,
            )
        elif standard_normals.shape != expected_noise_shape:
            raise ValueError(
                f"expected normal draws {expected_noise_shape}, got "
                f"{tuple(standard_normals.shape)}"
            )
        random_noise = standard_normals @ factor.T
        spot_std = torch.linalg.vector_norm(factor[0])
        surface_transition = transition[1:, 1:]
        next_state = (
            self.equilibrium_mean
            + (reduced_state - self.equilibrium_mean) @ surface_transition
            + random_noise[..., 1:]
        )
        normal_log_spot = (
            log_spot
            + float(step_weeks) * self.weekly_log_spot_drift
            + random_noise[..., 0]
        )
        return WeeklyStep(
            reduced_state=next_state,
            normal_log_spot=normal_log_spot,
            # Implied-spot mode replaces the fitted P spot marginal while
            # retaining its front-surface Gaussian copula coordinate.
            spot_standard_normal=random_noise[..., 0]
            / torch.clamp(spot_std, min=torch.finfo(self.fdtype).tiny),
        )


def load_strong_weekly_history(
    checkpoint_path: Path,
    report_path: Path,
    *,
    ticker: str = "SPX",
    start: date = PHASE3_TRAIN_START,
    end: date = PHASE3_VALIDATION_END,
    geometry: ForwardMuFlexGeometryConfig = ForwardMuFlexGeometryConfig(),
    market_config: FlexMarketConfig = FlexMarketConfig(),
    verbose: Context | None = Context.quiet,
) -> WeeklyTrainingHistory:
    """Load date-aligned strong states and ``market.py`` reference spots.

    This is the production bridge from Phase 2. It validates the completed
    checkpoint and allocation, but deliberately does not choose a split. Use
    ``history.subset(date(2022,1,3), date(2024,12,31))`` for training and the
    disjoint 2025 interval for validation.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw_dates = report.get("market_dates")
    if not isinstance(raw_dates, list) or not raw_dates:
        raise ValueError(f"report has no market_dates: {report_path}")
    dates = tuple(date.fromisoformat(value) for value in raw_dates)
    if dates != tuple(sorted(set(dates))):
        raise ValueError("report market_dates must be unique and increasing")
    if end < start or end > PHASE3_VALIDATION_END:
        raise ValueError(
            "Phase 3 history range must be ordered and cannot extend past "
            f"{PHASE3_VALIDATION_END}; got {start}..{end}"
        )
    latent_results = report.get("latent")
    selected_results = (
        [
            result
            for result in latent_results
            if isinstance(result, Mapping)
            and tuple(result.get("state_sizes_by_expiry", ())) == STRONG_STATE_SIZES
        ]
        if isinstance(latent_results, list)
        else []
    )
    if len(selected_results) != 1 or not isinstance(
        selected_results[0].get("checkpoint_proof"), Mapping
    ):
        raise ValueError("report does not identify exactly one completed Strong result")
    from quantdata.dysanos.flex.training import latent_checkpoint_proof

    actual_proof = latent_checkpoint_proof(checkpoint_path)
    expected_proof = selected_results[0]["checkpoint_proof"]
    for proof_key in (
        "checkpoint_sha256",
        "market_fingerprints_sha256",
        "latent_state_sha256",
        "num_market_days",
    ):
        if actual_proof.get(proof_key) != expected_proof.get(proof_key):
            raise ValueError(
                f"Strong checkpoint/report provenance mismatch for {proof_key}"
            )
    checkpoint: Mapping[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("phase") != "complete":
        raise ValueError(f"strong checkpoint is not complete: {checkpoint_path}")
    audits = checkpoint.get("training_audits")
    if not isinstance(audits, Mapping) or not all(
        isinstance(audits.get(name), Mapping)
        and audits[name].get("converged") is True
        for name in ("supervised", "market")
    ):
        raise ValueError("strong checkpoint has no successful convergence proof")
    identity = checkpoint.get("model_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("checkpoint has no model identity")
    model_config = identity.get("model_config")
    allocation = (
        model_config.get("state_sizes_by_expiry")
        if isinstance(model_config, Mapping)
        else None
    )
    if tuple(allocation or ()) != STRONG_STATE_SIZES:
        raise ValueError(f"checkpoint is not the selected strong model: {allocation}")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint has no model_state_dict")
    state_tensor = state_dict.get("daily_state_embedding.weight")
    if not isinstance(state_tensor, torch.Tensor):
        raise ValueError("checkpoint has no daily state embedding")
    states = state_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    if states.shape != (len(dates), sum(STRONG_STATE_SIZES)):
        raise ValueError(
            f"checkpoint/report history mismatch: {states.shape} versus {len(dates)}"
        )

    import math as _math

    selected = np.asarray([start <= value <= end for value in dates], dtype=bool)
    selected_dates = tuple(value for value, keep in zip(dates, selected) if keep)
    if not selected_dates:
        raise ValueError(f"checkpoint has no dates in requested interval {start}..{end}")
    spot_refs: list[float] = []
    for trading_date in selected_dates:
        payload = smooth_interpolated_market(
            ticker,
            trading_date,
            smoothness=market_config.smoothness,
            options_filter=OptionsFilter(
                max_num_options=market_config.max_num_options,
                max_business_days_to_expiry=_math.ceil(geometry.max_business_dte),
                max_nms=(-100.0, 100.0),
            ),
            verbose=verbose,
        )
        spot_ref = payload.get("spot_ref") if payload else None
        if not isinstance(spot_ref, (int, float)) or not math.isfinite(spot_ref) or spot_ref <= 0.0:
            raise ValueError(f"market.py has no valid spot_ref for {trading_date}")
        spot_refs.append(float(spot_ref))
    return WeeklyTrainingHistory(
        dates=selected_dates,
        states=states[selected],
        log_spot=np.log(np.asarray(spot_refs, dtype=np.float64)),
    )


def weekly_dynamics_checkpoint_payload(
    fit: WeeklyDynamicsFit,
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a versioned, reconstruction-complete weekly dynamics checkpoint."""
    return {
        "schema_version": 2,
        "phase": "complete",
        "config_version": fit.config.version.unique_id48,
        "config": asdict(fit.config),
        "source": dict(source),
        "pca_by_expiry": [
            {
                "mean": pca.mean,
                "full_components": pca.full_components,
                "explained_variance": pca.explained_variance,
                "reduced_rank": pca.reduced_rank,
            }
            for pca in fit.pca_by_expiry
        ],
        "blocks": [
            {
                "intercept": block.intercept,
                "own_lag": block.own_lag,
                "previous_lag": block.previous_lag,
                "observations": block.observations,
                "design_columns": block.design_columns,
                "design_rank": block.design_rank,
                "design_condition": block.design_condition,
                "underidentified": block.underidentified,
                "r2": block.r2,
                "generator_roundtrip_relative_error": (
                    block.generator_roundtrip_relative_error
                ),
                "mean_reversion_per_week": block.mean_reversion_per_week,
            }
            for block in fit.blocks
        ],
        "reduced_transition": fit.reduced_transition,
        "mean_reversion_generator": fit.mean_reversion_generator,
        "equilibrium_mean": fit.equilibrium_mean,
        "weekly_joint_innovation_covariance": (
            fit.weekly_joint_innovation_covariance
        ),
        "stationary_surface_covariance": fit.stationary_surface_covariance,
        "weekly_log_spot_drift": fit.weekly_log_spot_drift,
        "weekly_spot_front_beta": fit.weekly_spot_front_beta,
        "weekly_spot_orthogonal_std": fit.weekly_spot_orthogonal_std,
        "weekly_transition_observations": fit.weekly_transition_observations,
        "training_rows": fit.training_rows,
        "diagnostics": fit.diagnostics,
    }


def weekly_dynamics_fit_from_payload(payload: Mapping[str, Any]) -> WeeklyDynamicsFit:
    """Reconstruct a completed fit after strict schema/code-version checks."""
    if payload.get("schema_version") != 2 or payload.get("phase") != "complete":
        raise ValueError("Weekly dynamics checkpoint is not complete")
    raw_config = payload.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Weekly dynamics checkpoint has no model configuration")
    config_values = dict(raw_config)
    config_values["state_sizes_by_expiry"] = tuple(
        config_values["state_sizes_by_expiry"]
    )
    config = WeeklyDynamicsConfig(**config_values)
    if payload.get("config_version") != config.version.unique_id48:
        raise ValueError(
            "Weekly dynamics checkpoint code version is stale; cold refit required"
        )
    raw_pca = payload.get("pca_by_expiry")
    raw_blocks = payload.get("blocks")
    if not isinstance(raw_pca, list) or not isinstance(raw_blocks, list):
        raise ValueError("Weekly dynamics checkpoint has no PCA/hierarchy payload")
    pca_by_expiry = tuple(
        ExpiryPcaFit(
            mean=np.asarray(item["mean"], dtype=np.float64),
            full_components=np.asarray(item["full_components"], dtype=np.float64),
            explained_variance=np.asarray(
                item["explained_variance"], dtype=np.float64
            ),
            reduced_rank=int(item["reduced_rank"]),
        )
        for item in raw_pca
    )
    blocks = tuple(
        HierarchicalBlockFit(
            intercept=np.asarray(item["intercept"], dtype=np.float64),
            own_lag=np.asarray(item["own_lag"], dtype=np.float64),
            previous_lag=(
                None
                if item["previous_lag"] is None
                else np.asarray(item["previous_lag"], dtype=np.float64)
            ),
            observations=int(item["observations"]),
            design_columns=int(item["design_columns"]),
            design_rank=int(item["design_rank"]),
            design_condition=float(item["design_condition"]),
            underidentified=bool(item["underidentified"]),
            r2=float(item["r2"]),
            generator_roundtrip_relative_error=float(
                item["generator_roundtrip_relative_error"]
            ),
            mean_reversion_per_week=np.asarray(
                item["mean_reversion_per_week"], dtype=np.float64
            ),
        )
        for item in raw_blocks
    )
    return WeeklyDynamicsFit(
        config=config,
        pca_by_expiry=pca_by_expiry,
        blocks=blocks,
        reduced_transition=np.asarray(
            payload["reduced_transition"], dtype=np.float64
        ),
        mean_reversion_generator=np.asarray(
            payload["mean_reversion_generator"], dtype=np.float64
        ),
        equilibrium_mean=np.asarray(payload["equilibrium_mean"], dtype=np.float64),
        weekly_joint_innovation_covariance=np.asarray(
            payload["weekly_joint_innovation_covariance"], dtype=np.float64
        ),
        stationary_surface_covariance=np.asarray(
            payload["stationary_surface_covariance"], dtype=np.float64
        ),
        weekly_log_spot_drift=float(payload["weekly_log_spot_drift"]),
        weekly_spot_front_beta=np.asarray(
            payload["weekly_spot_front_beta"], dtype=np.float64
        ),
        weekly_spot_orthogonal_std=float(payload["weekly_spot_orthogonal_std"]),
        weekly_transition_observations=int(
            payload["weekly_transition_observations"]
        ),
        training_rows=int(payload["training_rows"]),
        diagnostics=dict(payload["diagnostics"]),
    )


def save_weekly_dynamics_fit(
    fit: WeeklyDynamicsFit,
    path: Path,
    *,
    source: Mapping[str, Any],
) -> None:
    """Atomically save a completed fit; the caller owns any training lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(weekly_dynamics_checkpoint_payload(fit, source=source), temporary)
    temporary.replace(path)


def load_weekly_dynamics_fit(path: Path) -> tuple[WeeklyDynamicsFit, dict[str, Any]]:
    """Load a completed weekly fit and return its immutable source metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid weekly dynamics checkpoint: {path}")
    fit = weekly_dynamics_fit_from_payload(payload)
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Weekly dynamics checkpoint has no source provenance")
    return fit, dict(source)


__all__ = [
    "ExpiryPcaFit",
    "HierarchicalBlockFit",
    "PHASE3_TRAIN_END",
    "PHASE3_TRAIN_START",
    "PHASE3_VALIDATION_END",
    "PHASE3_VALIDATION_START",
    "STRONG_STATE_SIZES",
    "TorchWeeklyDynamics",
    "WeeklyDynamicsConfig",
    "WeeklyDynamicsFit",
    "WeeklyStep",
    "WeeklyTrainingHistory",
    "fit_phase3_training_history",
    "fit_weekly_hierarchical_dynamics",
    "load_strong_weekly_history",
    "load_weekly_dynamics_fit",
    "save_weekly_dynamics_fit",
    "weekly_dynamics_checkpoint_payload",
    "weekly_dynamics_fit_from_payload",
]
