"""Out-of-sample statistical diagnostics for weekly FLEX dynamics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
from cdxcore.err import verify_inp

from quantdata.biz_calendar import get_business_days
from quantdata.dysanos.flex.weekly_dynamics import (
    PHASE3_VALIDATION_END,
    PHASE3_VALIDATION_START,
    WeeklyDynamicsFit,
    WeeklyTrainingHistory,
)


@dataclass(frozen=True)
class HorizonValidation:
    """Date-balanced Gaussian forecast diagnostics for one exact horizon."""

    weeks: int
    num_pairs: int
    joint_dimension: int
    negative_log_likelihood: float
    negative_log_likelihood_per_dimension: float
    unconditional_negative_log_likelihood: float
    unconditional_negative_log_likelihood_per_dimension: float
    log_score_improvement_per_dimension: float
    log_score_improvement_hac_standard_error_per_dimension: float
    log_score_improvement_hac_ci95_lower_per_dimension: float
    log_score_improvement_hac_ci95_upper_per_dimension: float
    random_walk_negative_log_likelihood_per_dimension: float
    random_walk_log_score_improvement_per_dimension: float
    random_walk_log_score_hac_standard_error_per_dimension: float
    diagonal_ar_negative_log_likelihood_per_dimension: float
    diagonal_ar_log_score_improvement_per_dimension: float
    diagonal_ar_log_score_hac_standard_error_per_dimension: float
    coordinate_negative_log_likelihood: tuple[float, ...]
    expiry_conditional_nll_per_dimension: tuple[float, ...]
    joint_energy_score: float
    mahalanobis_mean_per_dimension: float
    mahalanobis_median_per_dimension: float
    mahalanobis_coverage_50: float
    mahalanobis_coverage_90: float
    mahalanobis_coverage_95: float
    mahalanobis_ks_iid_naive_pvalue: float
    mahalanobis_nonoverlap_ks_pvalue: float
    surface_rmse: float
    surface_mae: float
    persistence_surface_rmse: float
    unconditional_surface_rmse: float
    covariance_calibration_frobenius_per_dimension: float
    spot_log_density: float
    spot_conditional_front_log_density: float
    spot_conditional_front_standardized_std: float
    spot_pit_mean: float
    spot_pit_ks_iid_naive_pvalue: float
    spot_pit_nonoverlap_ks_pvalue: float
    spot_standardized_mean: float
    spot_standardized_std: float
    spot_coverage_50: float
    spot_coverage_80: float
    spot_coverage_90: float
    spot_coverage_95: float
    spot_coverage_99: float
    hac_lag_origins: int
    nonoverlapping_pairs: int
    origin_endpoint_dates: tuple[tuple[str, str], ...]
    excluded_origin_dates: tuple[str, ...]
    finite: bool
    nondegenerate: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def project_strong_states(
    fit: WeeklyDynamicsFit,
    states: np.ndarray,
) -> np.ndarray:
    """Project 140-dimensional Strong states through training-only PCA maps."""
    values = np.asarray(states, dtype=np.float64)
    verify_inp(
        values.ndim == 2
        and values.shape[1] == fit.config.state_size
        and np.all(np.isfinite(values)),
        lambda: (
            "Strong states must be a finite two-dimensional array of width "
            f"{fit.config.state_size}; got {values.shape}"
        ),
    )
    blocks: list[np.ndarray] = []
    start = 0
    for width, pca in zip(
        fit.config.state_sizes_by_expiry,
        fit.pca_by_expiry,
        strict=True,
    ):
        block = values[:, start : start + width]
        blocks.append((block - pca.mean) @ pca.reduced_components.T)
        start += width
    output = np.concatenate(blocks, axis=1)
    if output.shape != (len(values), fit.reduced_state_size):
        raise RuntimeError(
            "Internal reduced-state projection mismatch: "
            f"{output.shape} versus {(len(values), fit.reduced_state_size)}"
        )
    return output


def exact_weekly_pairs(
    dates: tuple[date, ...],
    *,
    weeks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoints separated by exactly ``5 * weeks`` NYSE sessions."""
    verify_inp(
        isinstance(weeks, int) and weeks > 0,
        lambda: f"'weeks' must be a positive integer, found {weeks!r}",
    )
    verify_inp(
        bool(dates) and dates == tuple(sorted(set(dates))),
        lambda: "dates must be non-empty, unique, and increasing",
    )
    date_index = {value: index for index, value in enumerate(dates)}
    calendar = get_business_days(dates[0], dates[-1])
    calendar_index = {value: index for index, value in enumerate(calendar)}
    stride = 5 * weeks
    pairs = [
        (index, date_index[calendar[position + stride]])
        for index, value in enumerate(dates)
        if (position := calendar_index[value]) + stride < len(calendar)
        and calendar[position + stride] in date_index
    ]
    return (
        np.asarray([pair[0] for pair in pairs], dtype=np.int64),
        np.asarray([pair[1] for pair in pairs], dtype=np.int64),
    )


def propagated_joint_moments(
    fit: WeeklyDynamicsFit,
    *,
    weeks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose exact row-vector transition moments over whole weekly steps."""
    verify_inp(
        isinstance(weeks, int) and weeks > 0,
        lambda: f"'weeks' must be a positive integer, found {weeks!r}",
    )
    dimension = fit.reduced_state_size + 1
    transition = np.eye(dimension, dtype=np.float64)
    transition[1:, 1:] = fit.reduced_transition
    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    power = np.eye(dimension, dtype=np.float64)
    for _ in range(weeks):
        covariance += power.T @ fit.weekly_joint_innovation_covariance @ power
        power = power @ transition
    return power, 0.5 * (covariance + covariance.T)


def _regularized_covariance(values: np.ndarray) -> np.ndarray:
    """Return a positive-definite population covariance without changing units."""
    covariance = np.atleast_2d(np.cov(values, rowvar=False, bias=True))
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(eigenvalues)), 1.0)
    floor = np.finfo(np.float64).eps * scale * covariance.shape[0]
    return (eigenvectors * np.maximum(eigenvalues, floor)[None, :]) @ eigenvectors.T


@dataclass(frozen=True)
class _BaselineFit:
    """Training-only one-week affine Gaussian benchmark."""

    surface_transition: np.ndarray
    surface_intercept: np.ndarray
    spot_drift: float
    innovation_covariance: np.ndarray


def _fit_baselines(
    fit: WeeklyDynamicsFit,
    history: WeeklyTrainingHistory,
) -> tuple[_BaselineFit, _BaselineFit]:
    """Fit random-walk and coordinatewise AR controls on training dates only."""
    if any(value >= PHASE3_VALIDATION_START for value in history.dates):
        raise ValueError("Baseline fitting accepts training dates only")
    scores = project_strong_states(fit, history.states)
    origin, endpoint = exact_weekly_pairs(history.dates, weeks=1)
    if len(origin) == 0:
        raise ValueError("No exact weekly training pairs are available for baselines")
    spot_return = history.log_spot[endpoint] - history.log_spot[origin]
    delta = scores[endpoint] - scores[origin]
    random_intercept = np.mean(delta, axis=0)
    random_spot_drift = float(np.mean(spot_return))
    random_residual = np.concatenate(
        (
            (spot_return - random_spot_drift)[:, None],
            delta - random_intercept,
        ),
        axis=1,
    )
    random_walk = _BaselineFit(
        surface_transition=np.eye(fit.reduced_state_size, dtype=np.float64),
        surface_intercept=random_intercept,
        spot_drift=random_spot_drift,
        innovation_covariance=_regularized_covariance(random_residual),
    )

    lag = scores[origin]
    target = scores[endpoint]
    lag_mean = np.mean(lag, axis=0)
    target_mean = np.mean(target, axis=0)
    centered_lag = lag - lag_mean
    centered_target = target - target_mean
    denominator = np.sum(np.square(centered_lag), axis=0)
    slope = np.divide(
        np.sum(centered_lag * centered_target, axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > np.finfo(np.float64).eps,
    )
    slope = np.clip(slope, -0.995, 0.995)
    diagonal_transition = np.diag(slope)
    diagonal_intercept = target_mean - slope * lag_mean
    diagonal_spot_drift = float(np.mean(spot_return))
    diagonal_residual = np.concatenate(
        (
            (spot_return - diagonal_spot_drift)[:, None],
            target - (diagonal_intercept + lag * slope),
        ),
        axis=1,
    )
    diagonal_ar = _BaselineFit(
        surface_transition=diagonal_transition,
        surface_intercept=diagonal_intercept,
        spot_drift=diagonal_spot_drift,
        innovation_covariance=_regularized_covariance(diagonal_residual),
    )
    return random_walk, diagonal_ar


def _baseline_forecast(
    baseline: _BaselineFit,
    origins: np.ndarray,
    *,
    weeks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate one training-only affine baseline for an integer week horizon."""
    dimension = origins.shape[1] + 1
    surface = origins.copy()
    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    joint_transition = np.eye(dimension, dtype=np.float64)
    joint_transition[1:, 1:] = baseline.surface_transition
    power = np.eye(dimension, dtype=np.float64)
    for _ in range(weeks):
        surface = (
            baseline.surface_intercept
            + surface @ baseline.surface_transition
        )
        covariance += power.T @ baseline.innovation_covariance @ power
        power = power @ joint_transition
    mean = np.concatenate(
        (
            np.full((len(origins), 1), weeks * baseline.spot_drift),
            surface,
        ),
        axis=1,
    )
    return mean, 0.5 * (covariance + covariance.T)


def _gaussian_scores(
    residuals: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observation NLL and squared Mahalanobis distance."""
    values, vectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(values)), 1.0)
    floor = np.finfo(np.float64).eps * scale * covariance.shape[0]
    values = np.maximum(values, floor)
    rotated = residuals @ vectors
    mahalanobis = np.sum(np.square(rotated) / values, axis=1)
    log_determinant = float(np.sum(np.log(values)))
    nll = 0.5 * (
        covariance.shape[0] * math.log(2.0 * math.pi)
        + log_determinant
        + mahalanobis
    )
    return nll, mahalanobis


def _conditional_block_nll(
    residuals: np.ndarray,
    covariance: np.ndarray,
    block_sizes: tuple[int, ...],
) -> tuple[float, ...]:
    """Score each expiry conditional only on preceding surface blocks."""
    output: list[float] = []
    offset = 1
    for width in block_sizes:
        target = slice(offset, offset + width)
        context = slice(1, offset)
        if offset == 1:
            nll, _ = _gaussian_scores(
                residuals[:, target],
                covariance[target, target],
            )
            output.append(float(np.mean(nll) / width))
            offset += width
            continue
        context_covariance = covariance[context, context]
        cross = covariance[target, context]
        conditional_loading = np.linalg.solve(
            context_covariance,
            cross.T,
        ).T
        conditional_residual = (
            residuals[:, target]
            - residuals[:, context] @ conditional_loading.T
        )
        conditional_covariance = (
            covariance[target, target]
            - conditional_loading @ covariance[context, target]
        )
        nll, _ = _gaussian_scores(
            conditional_residual,
            0.5 * (conditional_covariance + conditional_covariance.T),
        )
        output.append(float(np.mean(nll) / width))
        offset += width
    return tuple(output)


def _energy_score(
    residuals: np.ndarray,
    covariance: np.ndarray,
    *,
    seed: int,
    num_draws: int = 512,
) -> float:
    """Deterministically approximate the joint Gaussian energy score."""
    values, vectors = np.linalg.eigh(covariance)
    factor = vectors * np.sqrt(np.maximum(values, 0.0))[None, :]
    generator = np.random.default_rng(seed)
    first = generator.standard_normal((num_draws, covariance.shape[0])) @ factor.T
    second = generator.standard_normal((num_draws, covariance.shape[0])) @ factor.T
    observation_term = np.mean(
        np.linalg.norm(first[None, :, :] - residuals[:, None, :], axis=2),
        axis=1,
    )
    sample_term = 0.5 * float(np.mean(np.linalg.norm(first - second, axis=1)))
    return float(np.mean(observation_term - sample_term))


def _hac_mean_standard_error(values: np.ndarray, *, max_lag: int) -> float:
    """Newey-West standard error of a mean over overlapping forecast origins."""
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    count = len(centered)
    if count < 2:
        return 0.0
    lag_cap = min(max_lag, count - 1)
    long_run_variance = float(centered @ centered / count)
    for lag in range(1, lag_cap + 1):
        weight = 1.0 - lag / (lag_cap + 1.0)
        covariance = float(centered[lag:] @ centered[:-lag] / count)
        long_run_variance += 2.0 * weight * covariance
    return math.sqrt(max(long_run_variance, 0.0) / count)


def _nonoverlapping_pair_indices(
    origin: np.ndarray,
    endpoint: np.ndarray,
) -> np.ndarray:
    """Return greedy validation-row indices with disjoint forecast intervals."""
    selected: list[int] = []
    last_endpoint = -1
    for index, (start, end) in enumerate(zip(origin, endpoint, strict=True)):
        if int(start) >= last_endpoint:
            selected.append(index)
            last_endpoint = int(end)
    return np.asarray(selected, dtype=np.int64)


def _empirical_unconditional_forecast(
    fit: WeeklyDynamicsFit,
    history: WeeklyTrainingHistory,
    *,
    weeks: int,
    num_forecasts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an honest horizon-specific unconditional Gaussian on training pairs."""
    scores = project_strong_states(fit, history.states)
    origin, endpoint = exact_weekly_pairs(history.dates, weeks=weeks)
    if len(origin) == 0:
        raise ValueError(
            f"No exact {weeks}-week training pairs for unconditional baseline"
        )
    joint = np.concatenate(
        (
            (history.log_spot[endpoint] - history.log_spot[origin])[:, None],
            scores[endpoint],
        ),
        axis=1,
    )
    mean = np.broadcast_to(np.mean(joint, axis=0), (num_forecasts, joint.shape[1]))
    return mean, _regularized_covariance(joint)


def evaluate_horizon(
    fit: WeeklyDynamicsFit,
    history: WeeklyTrainingHistory,
    training_history: WeeklyTrainingHistory,
    *,
    weeks: int,
) -> HorizonValidation:
    """Score one exact 1W/2W/4W forecast on untouched 2025 observations."""
    from scipy.stats import chi2, kstest, norm

    if any(
        value < PHASE3_VALIDATION_START or value > PHASE3_VALIDATION_END
        for value in history.dates
    ):
        raise ValueError(
            "OOS dynamics scoring accepts only dates in the locked 2025 "
            f"interval {PHASE3_VALIDATION_START}..{PHASE3_VALIDATION_END}"
        )
    reduced = project_strong_states(fit, history.states)
    origin, endpoint = exact_weekly_pairs(history.dates, weeks=weeks)
    if len(origin) == 0:
        raise ValueError(f"No exact {weeks}-week validation pairs are available")
    transition, covariance = propagated_joint_moments(fit, weeks=weeks)
    surface_transition = transition[1:, 1:]
    predicted_surface = fit.equilibrium_mean + (
        reduced[origin] - fit.equilibrium_mean
    ) @ surface_transition
    realized_spot = history.log_spot[endpoint] - history.log_spot[origin]
    residuals = np.concatenate(
        (
            (realized_spot - weeks * fit.weekly_log_spot_drift)[:, None],
            reduced[endpoint] - predicted_surface,
        ),
        axis=1,
    )
    nll, mahalanobis = _gaussian_scores(residuals, covariance)

    random_walk, diagonal_ar = _fit_baselines(fit, training_history)
    realized_joint = np.concatenate(
        (realized_spot[:, None], reduced[endpoint]),
        axis=1,
    )
    random_mean, random_covariance = _baseline_forecast(
        random_walk,
        reduced[origin],
        weeks=weeks,
    )
    diagonal_mean, diagonal_covariance = _baseline_forecast(
        diagonal_ar,
        reduced[origin],
        weeks=weeks,
    )
    random_nll, _ = _gaussian_scores(
        realized_joint - random_mean,
        random_covariance,
    )
    diagonal_nll, _ = _gaussian_scores(
        realized_joint - diagonal_mean,
        diagonal_covariance,
    )

    # This is an independently estimated, horizon-specific unconditional
    # Gaussian. It retains empirical spot/surface covariance and uses no 2025
    # observation to estimate either its mean or covariance.
    baseline_mean, baseline_covariance = _empirical_unconditional_forecast(
        fit,
        training_history,
        weeks=weeks,
        num_forecasts=len(origin),
    )
    baseline_nll, _ = _gaussian_scores(
        realized_joint - baseline_mean,
        baseline_covariance,
    )

    dimension = residuals.shape[1]
    chi2_cdf = chi2.cdf(mahalanobis, df=dimension)
    ks_pvalue = float(kstest(chi2_cdf, "uniform").pvalue)
    spot_standard_deviation = math.sqrt(
        max(float(covariance[0, 0]), np.finfo(np.float64).tiny)
    )
    standardized_spot = residuals[:, 0] / spot_standard_deviation
    spot_pit = norm.cdf(standardized_spot)
    spot_log_density = (
        -0.5 * math.log(2.0 * math.pi * spot_standard_deviation**2)
        - 0.5 * np.square(standardized_spot)
    )
    coordinate_variance = np.diag(covariance)
    coordinate_nll = 0.5 * (
        math.log(2.0 * math.pi)
        + np.log(coordinate_variance)[None, :]
        + np.square(residuals) / coordinate_variance[None, :]
    )
    front_width = fit.reduced_sizes_by_expiry[0]
    front_slice = slice(1, 1 + front_width)
    front_covariance = covariance[front_slice, front_slice]
    spot_front_loading = np.linalg.solve(
        front_covariance,
        covariance[front_slice, 0],
    )
    conditional_spot_residual = (
        residuals[:, 0] - residuals[:, front_slice] @ spot_front_loading
    )
    conditional_spot_variance = max(
        float(
            covariance[0, 0]
            - covariance[0, front_slice] @ spot_front_loading
        ),
        np.finfo(np.float64).tiny,
    )
    conditional_spot_standardized = (
        conditional_spot_residual / math.sqrt(conditional_spot_variance)
    )
    conditional_spot_log_density = (
        -0.5 * math.log(2.0 * math.pi * conditional_spot_variance)
        - 0.5 * np.square(conditional_spot_standardized)
    )
    covariance_values, covariance_vectors = np.linalg.eigh(covariance)
    whitened = (
        residuals @ covariance_vectors
        / np.sqrt(np.maximum(covariance_values, np.finfo(np.float64).tiny))[None, :]
    )
    whitened_covariance = np.cov(whitened, rowvar=False, bias=True)
    covariance_calibration = float(
        np.linalg.norm(whitened_covariance - np.eye(dimension), ord="fro")
        / dimension
    )
    improvement = baseline_nll - nll
    random_improvement = random_nll - nll
    diagonal_improvement = diagonal_nll - nll
    hac_lag = 5 * weeks - 1
    improvement_hac_se = _hac_mean_standard_error(improvement, max_lag=hac_lag)
    random_hac_se = _hac_mean_standard_error(
        random_improvement,
        max_lag=hac_lag,
    )
    diagonal_hac_se = _hac_mean_standard_error(
        diagonal_improvement,
        max_lag=hac_lag,
    )
    mean_improvement = float(np.mean(improvement) / dimension)
    improvement_hac_se_per_dimension = improvement_hac_se / dimension
    nonoverlapping = _nonoverlapping_pair_indices(origin, endpoint)
    origin_set = set(int(item) for item in origin)
    conditional_block_nll = _conditional_block_nll(
        residuals,
        covariance,
        fit.reduced_sizes_by_expiry,
    )
    energy_score = _energy_score(
        residuals,
        covariance,
        seed=18_000 + weeks,
    )
    model_rmse = float(
        np.sqrt(np.mean(np.square(reduced[endpoint] - predicted_surface)))
    )
    model_mae = float(
        np.mean(np.abs(reduced[endpoint] - predicted_surface))
    )
    persistence_rmse = float(
        np.sqrt(np.mean(np.square(reduced[endpoint] - reduced[origin])))
    )
    unconditional_rmse = float(
        np.sqrt(np.mean(np.square(reduced[endpoint] - fit.equilibrium_mean)))
    )
    statistics: list[float] = [
        float(np.mean(nll)),
        float(np.mean(baseline_nll)),
        float(np.mean(mahalanobis)),
        model_rmse,
        model_mae,
        persistence_rmse,
        unconditional_rmse,
        covariance_calibration,
        float(np.mean(spot_log_density)),
        float(np.mean(conditional_spot_log_density)),
        float(np.std(conditional_spot_standardized, ddof=0)),
        float(np.mean(standardized_spot)),
        float(np.std(standardized_spot, ddof=0)),
        float(np.mean(random_nll)),
        float(np.mean(diagonal_nll)),
        energy_score,
        *conditional_block_nll,
    ]
    finite = bool(np.all(np.isfinite(statistics)))
    minimum_covariance_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    stochastic_floor = max(fit.config.covariance_eigenvalue_floor, 1.0e-14)
    spot_variance = float(covariance[0, 0])
    surface_variance_per_coordinate = float(
        np.trace(covariance[1:, 1:]) / fit.reduced_state_size
    )
    stationary_variance_per_coordinate = float(
        np.trace(fit.stationary_surface_covariance) / fit.reduced_state_size
    )
    # Research gate: the process need not beat every baseline yet, but it must
    # have genuine spot/surface variation rather than merely a covariance floor.
    nondegenerate = bool(
        finite
        and minimum_covariance_eigenvalue > 0.0
        and spot_variance > 100.0 * stochastic_floor
        and surface_variance_per_coordinate > 100.0 * stochastic_floor
        and stationary_variance_per_coordinate > 100.0 * stochastic_floor
    )
    return HorizonValidation(
        weeks=weeks,
        num_pairs=len(origin),
        joint_dimension=dimension,
        negative_log_likelihood=float(np.mean(nll)),
        negative_log_likelihood_per_dimension=float(np.mean(nll) / dimension),
        unconditional_negative_log_likelihood=float(np.mean(baseline_nll)),
        unconditional_negative_log_likelihood_per_dimension=float(
            np.mean(baseline_nll) / dimension
        ),
        log_score_improvement_per_dimension=mean_improvement,
        log_score_improvement_hac_standard_error_per_dimension=(
            improvement_hac_se_per_dimension
        ),
        log_score_improvement_hac_ci95_lower_per_dimension=(
            mean_improvement - 1.96 * improvement_hac_se_per_dimension
        ),
        log_score_improvement_hac_ci95_upper_per_dimension=(
            mean_improvement + 1.96 * improvement_hac_se_per_dimension
        ),
        random_walk_negative_log_likelihood_per_dimension=float(
            np.mean(random_nll) / dimension
        ),
        random_walk_log_score_improvement_per_dimension=float(
            np.mean(random_improvement) / dimension
        ),
        random_walk_log_score_hac_standard_error_per_dimension=float(
            random_hac_se / dimension
        ),
        diagonal_ar_negative_log_likelihood_per_dimension=float(
            np.mean(diagonal_nll) / dimension
        ),
        diagonal_ar_log_score_improvement_per_dimension=float(
            np.mean(diagonal_improvement) / dimension
        ),
        diagonal_ar_log_score_hac_standard_error_per_dimension=float(
            diagonal_hac_se / dimension
        ),
        coordinate_negative_log_likelihood=tuple(
            float(value) for value in np.mean(coordinate_nll, axis=0)
        ),
        expiry_conditional_nll_per_dimension=conditional_block_nll,
        joint_energy_score=energy_score,
        mahalanobis_mean_per_dimension=float(np.mean(mahalanobis) / dimension),
        mahalanobis_median_per_dimension=float(np.median(mahalanobis) / dimension),
        mahalanobis_coverage_50=float(
            np.mean(mahalanobis <= chi2.ppf(0.50, df=dimension))
        ),
        mahalanobis_coverage_90=float(
            np.mean(mahalanobis <= chi2.ppf(0.90, df=dimension))
        ),
        mahalanobis_coverage_95=float(
            np.mean(mahalanobis <= chi2.ppf(0.95, df=dimension))
        ),
        mahalanobis_ks_iid_naive_pvalue=ks_pvalue,
        mahalanobis_nonoverlap_ks_pvalue=float(
            kstest(chi2_cdf[nonoverlapping], "uniform").pvalue
        ),
        surface_rmse=model_rmse,
        surface_mae=model_mae,
        persistence_surface_rmse=persistence_rmse,
        unconditional_surface_rmse=unconditional_rmse,
        covariance_calibration_frobenius_per_dimension=covariance_calibration,
        spot_log_density=float(np.mean(spot_log_density)),
        spot_conditional_front_log_density=float(
            np.mean(conditional_spot_log_density)
        ),
        spot_conditional_front_standardized_std=float(
            np.std(conditional_spot_standardized, ddof=0)
        ),
        spot_pit_mean=float(np.mean(spot_pit)),
        spot_pit_ks_iid_naive_pvalue=float(kstest(spot_pit, "uniform").pvalue),
        spot_pit_nonoverlap_ks_pvalue=float(
            kstest(spot_pit[nonoverlapping], "uniform").pvalue
        ),
        spot_standardized_mean=float(np.mean(standardized_spot)),
        spot_standardized_std=float(np.std(standardized_spot, ddof=0)),
        spot_coverage_50=float(
            np.mean(np.abs(standardized_spot) <= norm.ppf(0.75))
        ),
        spot_coverage_80=float(
            np.mean(np.abs(standardized_spot) <= norm.ppf(0.90))
        ),
        spot_coverage_90=float(
            np.mean(np.abs(standardized_spot) <= norm.ppf(0.95))
        ),
        spot_coverage_95=float(
            np.mean(np.abs(standardized_spot) <= norm.ppf(0.975))
        ),
        spot_coverage_99=float(
            np.mean(np.abs(standardized_spot) <= norm.ppf(0.995))
        ),
        hac_lag_origins=hac_lag,
        nonoverlapping_pairs=len(nonoverlapping),
        origin_endpoint_dates=tuple(
            (history.dates[int(start)].isoformat(), history.dates[int(end)].isoformat())
            for start, end in zip(origin, endpoint, strict=True)
        ),
        excluded_origin_dates=tuple(
            value.isoformat()
            for index, value in enumerate(history.dates)
            if index not in origin_set
        ),
        finite=finite,
        nondegenerate=nondegenerate,
    )


def evaluate_oos_dynamics(
    fit: WeeklyDynamicsFit,
    history: WeeklyTrainingHistory,
    training_history: WeeklyTrainingHistory,
) -> dict[str, Any]:
    """Evaluate the requested exact 1W, 2W, and 4W 2025 horizons."""
    horizons = {
        f"{weeks}w": evaluate_horizon(
            fit,
            history,
            training_history,
            weeks=weeks,
        ).to_dict()
        for weeks in (1, 2, 4)
    }
    return {
        "validation_start": history.dates[0].isoformat(),
        "validation_end": history.dates[-1].isoformat(),
        "num_validation_dates": len(history.dates),
        "date_weighting": "one equal observation per exact horizon origin date",
        "uncertainty": (
            "Newey-West standard errors use 5*weeks-1 overlapping-origin lags; "
            "nonoverlapping_pairs is a greedy disjoint-interval count"
        ),
        "baseline_training_end": training_history.dates[-1].isoformat(),
        "horizons": horizons,
        "all_horizons_finite": all(
            bool(value["finite"]) for value in horizons.values()
        ),
        "all_horizons_nondegenerate": all(
            bool(value["nondegenerate"]) for value in horizons.values()
        ),
    }


def evaluate_one_week_interval(
    fit: WeeklyDynamicsFit,
    history: WeeklyTrainingHistory,
) -> dict[str, int | float | bool]:
    """Score a generic later interval for train-only hyperparameter sensitivity."""
    reduced = project_strong_states(fit, history.states)
    origin, endpoint = exact_weekly_pairs(history.dates, weeks=1)
    if len(origin) == 0:
        raise ValueError("No exact weekly pairs are available in the score interval")
    transition, covariance = propagated_joint_moments(fit, weeks=1)
    predicted_surface = fit.equilibrium_mean + (
        reduced[origin] - fit.equilibrium_mean
    ) @ transition[1:, 1:]
    realized_spot = history.log_spot[endpoint] - history.log_spot[origin]
    residuals = np.concatenate(
        (
            (realized_spot - fit.weekly_log_spot_drift)[:, None],
            reduced[endpoint] - predicted_surface,
        ),
        axis=1,
    )
    nll, mahalanobis = _gaussian_scores(residuals, covariance)
    statistics = np.asarray(
        [
            np.mean(nll) / residuals.shape[1],
            np.sqrt(np.mean(np.square(residuals[:, 1:]))),
            np.mean(mahalanobis) / residuals.shape[1],
        ],
        dtype=np.float64,
    )
    return {
        "num_pairs": len(origin),
        "negative_log_likelihood_per_dimension": float(statistics[0]),
        "surface_rmse": float(statistics[1]),
        "mahalanobis_mean_per_dimension": float(statistics[2]),
        "finite": bool(np.all(np.isfinite(statistics))),
    }


__all__ = [
    "HorizonValidation",
    "evaluate_horizon",
    "evaluate_oos_dynamics",
    "evaluate_one_week_interval",
    "exact_weekly_pairs",
    "project_strong_states",
    "propagated_joint_moments",
]
