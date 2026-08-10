"""Compare autonomous and log-spot-driven ML-SANOS surface drifts.

The comparison is deliberately based on chronological one-step forecasts.  An
in-sample comparison is not decisive because the log-spot-driven model nests
the autonomous model and adds one coefficient to every surface-state equation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.verbose import Context
from cdxcore.version import version
from scipy.stats import norm

from quantdata.biz_calendar import DAYS_PER_YEAR

from .torch_prep import GenDysanosExperiment, _load_generator_components


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class SurfaceSpotDriftComparisonConfig:
    """Configuration for chronological surface-drift comparisons."""

    training_fraction: float = 0.7
    minimum_rolling_training: int = 600
    rolling_validation_size: int = 200
    hac_lag: int = 20
    covariance_eigenvalue_floor: float = 1.0e-8

    def __post_init__(self) -> None:
        verify_inp(
            0.5 <= self.training_fraction < 1.0,
            lambda: (
                "'training_fraction' must lie in [0.5, 1), got "
                f"{self.training_fraction}"
            ),
        )
        verify_inp(
            self.minimum_rolling_training > 0,
            lambda: (
                "'minimum_rolling_training' must be positive, got "
                f"{self.minimum_rolling_training}"
            ),
        )
        verify_inp(
            self.rolling_validation_size > 0,
            lambda: (
                "'rolling_validation_size' must be positive, got "
                f"{self.rolling_validation_size}"
            ),
        )
        verify_inp(
            self.hac_lag >= 0,
            lambda: f"'hac_lag' must be non-negative, got {self.hac_lag}",
        )
        verify_inp(
            math.isfinite(self.covariance_eigenvalue_floor)
            and self.covariance_eigenvalue_floor > 0.0,
            lambda: (
                "'covariance_eigenvalue_floor' must be finite and positive, "
                f"got {self.covariance_eigenvalue_floor}"
            ),
        )


@dataclass(frozen=True)
class _SurfaceDriftFit:
    surface_center: np.ndarray
    surface_scale: np.ndarray
    driver_center: float | None
    driver_scale: float | None
    coefficients: np.ndarray
    covariance: np.ndarray
    covariance_inverse: np.ndarray
    covariance_log_determinant: float
    innovation_scale: np.ndarray
    spot_drift_loading: np.ndarray | None
    surface_generator: np.ndarray


def _validate_inputs(
    states: np.ndarray,
    time_y: np.ndarray,
    config: SurfaceSpotDriftComparisonConfig,
) -> None:
    verify_inp(
        states.ndim == 2 and states.shape[0] >= 100 and states.shape[1] >= 2,
        lambda: (
            "'states' must have shape (observations>=100, states>=2), got "
            f"{states.shape}"
        ),
    )
    verify_inp(
        time_y.shape == (states.shape[0],),
        lambda: (
            f"'time_y' must have shape {(states.shape[0],)}, got {time_y.shape}"
        ),
    )
    verify_inp(
        np.all(np.isfinite(states)) and np.all(np.isfinite(time_y)),
        "states and times must be finite",
    )
    dt = np.diff(time_y)
    verify_inp(np.all(dt > 0.0), "observation times must be strictly increasing")
    transitions = states.shape[0] - 1
    verify_inp(
        config.minimum_rolling_training < transitions,
        lambda: (
            "'minimum_rolling_training' must be below the number of transitions "
            f"{transitions}, got {config.minimum_rolling_training}"
        ),
    )


def _training_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    floor = math.sqrt(np.finfo(values.dtype).eps) * max(
        1.0,
        float(np.max(np.abs(scale))),
    )
    verify_inp(
        np.all(scale > floor),
        lambda: f"one or more predictors are constant on training data: {scale}",
    )
    return center, scale


def _fit_surface_drift(
    *,
    states: np.ndarray,
    normalized_dt: np.ndarray,
    training_indices: np.ndarray,
    driver: np.ndarray | None,
    covariance_eigenvalue_floor: float,
) -> _SurfaceDriftFit:
    surface = states[:, 1:]
    surface_start = surface[:-1]
    surface_change = np.diff(surface, axis=0)
    training_surface = surface_start[training_indices]
    surface_center, surface_scale = _training_location_scale(training_surface)
    standardized_surface = (
        surface_start - surface_center[None, :]
    ) / surface_scale[None, :]

    design_parts = [
        np.ones((surface_start.shape[0], 1), dtype=np.float64),
        standardized_surface,
    ]
    driver_center: float | None = None
    driver_scale: float | None = None
    if driver is not None:
        verify_inp(
            driver.shape == (surface_start.shape[0],),
            lambda: (
                f"'driver' must have shape {(surface_start.shape[0],)}, got "
                f"{driver.shape}"
            ),
        )
        driver_center_array, driver_scale_array = _training_location_scale(
            driver[training_indices, None]
        )
        driver_center = float(driver_center_array[0])
        driver_scale = float(driver_scale_array[0])
        design_parts.append(
            ((driver - driver_center) / driver_scale)[:, None]
        )

    base_design = np.concatenate(design_parts, axis=1)
    sqrt_dt = np.sqrt(normalized_dt)
    design = sqrt_dt[:, None] * base_design
    target = surface_change / sqrt_dt[:, None]
    coefficients = np.linalg.lstsq(
        design[training_indices],
        target[training_indices],
        rcond=None,
    )[0]
    training_error = (
        target[training_indices]
        - design[training_indices] @ coefficients
    )
    covariance = training_error.T @ training_error / len(training_indices)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(eigenvalues[-1]), np.finfo(float).tiny)
    eigenvalue_floor = covariance_eigenvalue_floor * scale
    eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
    covariance = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    covariance_inverse = (
        eigenvectors * (1.0 / eigenvalues)[None, :]
    ) @ eigenvectors.T

    surface_derivative = coefficients[
        1 : 1 + surface.shape[1]
    ] / surface_scale[:, None]
    surface_generator = -surface_derivative
    spot_drift_loading = None
    if driver is not None:
        assert driver_scale is not None
        # Column-vector paper convention: dh = (... - a s - kappa_h h)dt.
        spot_drift_loading = -coefficients[-1] / driver_scale

    return _SurfaceDriftFit(
        surface_center=surface_center,
        surface_scale=surface_scale,
        driver_center=driver_center,
        driver_scale=driver_scale,
        coefficients=coefficients,
        covariance=covariance,
        covariance_inverse=covariance_inverse,
        covariance_log_determinant=float(np.sum(np.log(eigenvalues))),
        innovation_scale=np.sqrt(np.diag(covariance)),
        spot_drift_loading=spot_drift_loading,
        surface_generator=surface_generator,
    )


def _innovation_errors(
    fit: _SurfaceDriftFit,
    *,
    states: np.ndarray,
    normalized_dt: np.ndarray,
    driver: np.ndarray | None,
) -> np.ndarray:
    surface = states[:, 1:]
    surface_start = surface[:-1]
    standardized_surface = (
        surface_start - fit.surface_center[None, :]
    ) / fit.surface_scale[None, :]
    design_parts = [
        np.ones((surface_start.shape[0], 1), dtype=np.float64),
        standardized_surface,
    ]
    if fit.driver_center is not None:
        assert fit.driver_scale is not None and driver is not None
        design_parts.append(
            ((driver - fit.driver_center) / fit.driver_scale)[:, None]
        )
    else:
        assert driver is None
    design = np.sqrt(normalized_dt)[:, None] * np.concatenate(
        design_parts,
        axis=1,
    )
    target = np.diff(surface, axis=0) / np.sqrt(normalized_dt)[:, None]
    error = target - design @ fit.coefficients
    if not np.all(np.isfinite(error)):
        raise FloatingPointError("surface drift prediction produced non-finite errors")
    return error


def _gaussian_log_scores(fit: _SurfaceDriftFit, error: np.ndarray) -> np.ndarray:
    dimension = error.shape[1]
    mahalanobis = np.einsum(
        "ti,ij,tj->t",
        error,
        fit.covariance_inverse,
        error,
        optimize=True,
    )
    return -0.5 * (
        dimension * math.log(2.0 * math.pi)
        + fit.covariance_log_determinant
        + mahalanobis
    )


def _hac_mean_test(values: np.ndarray, maximum_lag: int) -> dict[str, float]:
    verify_inp(
        values.ndim == 1 and values.size >= 2 and np.all(np.isfinite(values)),
        lambda: f"HAC observations must be a finite vector, got {values.shape}",
    )
    centered = values - np.mean(values)
    observations = len(values)
    maximum_lag = min(maximum_lag, observations - 1)
    long_run_variance = float(centered @ centered / observations)
    for lag in range(1, maximum_lag + 1):
        covariance = float(
            centered[lag:] @ centered[:-lag] / observations
        )
        weight = 1.0 - lag / (maximum_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = math.sqrt(long_run_variance / observations)
    mean = float(np.mean(values))
    if standard_error > 0.0:
        statistic = mean / standard_error
        p_value = float(2.0 * norm.sf(abs(statistic)))
    else:
        statistic = math.copysign(math.inf, mean) if mean != 0.0 else 0.0
        p_value = 0.0 if mean != 0.0 else 1.0
    return {
        "mean": mean,
        "hac_standard_error": standard_error,
        "hac_t_statistic": statistic,
        "two_sided_normal_p_value": p_value,
        "hac_maximum_lag": int(maximum_lag),
    }


def _model_metrics(
    fit: _SurfaceDriftFit,
    *,
    error: np.ndarray,
    common_innovation_scale: np.ndarray,
    indices: np.ndarray,
) -> dict[str, object]:
    selected_error = error[indices]
    standardized = selected_error / common_innovation_scale[None, :]
    scores = _gaussian_log_scores(fit, selected_error)
    eigenvalues = np.linalg.eigvals(fit.surface_generator)
    return {
        "observations": int(len(indices)),
        "mean_gaussian_log_score_per_date": float(np.mean(scores)),
        "standardized_rmse": float(np.sqrt(np.mean(np.square(standardized)))),
        "mean_standardized_squared_error_per_date": float(
            np.mean(np.sum(np.square(standardized), axis=1))
        ),
        "surface_generator_minimum_real_eigenvalue": float(
            np.min(eigenvalues.real)
        ),
        "surface_generator_maximum_real_eigenvalue": float(
            np.max(eigenvalues.real)
        ),
        "surface_generator_stable": bool(np.all(eigenvalues.real > 0.0)),
    }


def _pair_metrics(
    *,
    base_fit: _SurfaceDriftFit,
    alternative_fit: _SurfaceDriftFit,
    base_error: np.ndarray,
    alternative_error: np.ndarray,
    common_innovation_scale: np.ndarray,
    indices: np.ndarray,
    hac_lag: int,
) -> dict[str, object]:
    base_selected = base_error[indices]
    alternative_selected = alternative_error[indices]
    log_score_gain = (
        _gaussian_log_scores(alternative_fit, alternative_selected)
        - _gaussian_log_scores(base_fit, base_selected)
    )
    base_squared = np.sum(
        np.square(base_selected / common_innovation_scale[None, :]),
        axis=1,
    )
    alternative_squared = np.sum(
        np.square(alternative_selected / common_innovation_scale[None, :]),
        axis=1,
    )
    squared_error_reduction = base_squared - alternative_squared
    return {
        "gaussian_log_score_gain_alternative_minus_autonomous": _hac_mean_test(
            log_score_gain,
            hac_lag,
        ),
        "standardized_squared_error_reduction_autonomous_minus_alternative": (
            _hac_mean_test(squared_error_reduction, hac_lag)
        ),
        "fraction_of_dates_with_better_log_score": float(
            np.mean(log_score_gain > 0.0)
        ),
        "fraction_of_dates_with_lower_squared_error": float(
            np.mean(squared_error_reduction > 0.0)
        ),
    }


def _loading_summary(
    fit: _SurfaceDriftFit,
    common_innovation_scale: np.ndarray,
) -> dict[str, object]:
    assert fit.spot_drift_loading is not None
    assert fit.driver_scale is not None
    effect_per_driver_std = (
        -fit.spot_drift_loading
        * fit.driver_scale
        / common_innovation_scale
    )
    return {
        "paper_convention_a": fit.spot_drift_loading,
        "driver_training_standard_deviation": fit.driver_scale,
        "one_day_drift_effect_of_one_driver_std_in_innovation_std": (
            effect_per_driver_std
        ),
        "effect_rms": float(np.sqrt(np.mean(np.square(effect_per_driver_std)))),
        "effect_max_abs": float(np.max(np.abs(effect_per_driver_std))),
    }


def _coefficient_stability(loadings: list[np.ndarray]) -> dict[str, float]:
    if len(loadings) < 2:
        return {
            "folds": float(len(loadings)),
            "adjacent_cosine_mean": math.nan,
            "adjacent_sign_agreement_mean": math.nan,
        }
    cosines = []
    sign_agreements = []
    for left, right in zip(loadings[:-1], loadings[1:], strict=True):
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        cosines.append(float(left @ right / denominator) if denominator > 0.0 else math.nan)
        sign_agreements.append(float(np.mean(np.sign(left) == np.sign(right))))
    return {
        "folds": float(len(loadings)),
        "adjacent_cosine_mean": float(np.nanmean(cosines)),
        "adjacent_cosine_minimum": float(np.nanmin(cosines)),
        "adjacent_sign_agreement_mean": float(np.mean(sign_agreements)),
        "adjacent_sign_agreement_minimum": float(np.min(sign_agreements)),
    }


def _driver_distribution_shift(
    driver: np.ndarray,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> dict[str, float]:
    training = driver[training_indices]
    validation = driver[validation_indices]
    training_mean = float(np.mean(training))
    training_std = float(np.std(training))
    verify_inp(training_std > 0.0, "training driver must not be constant")
    return {
        "training_mean": training_mean,
        "training_std": training_std,
        "training_minimum": float(np.min(training)),
        "training_maximum": float(np.max(training)),
        "validation_mean": float(np.mean(validation)),
        "validation_std": float(np.std(validation)),
        "validation_minimum": float(np.min(validation)),
        "validation_maximum": float(np.max(validation)),
        "validation_mean_in_training_std": float(
            (np.mean(validation) - training_mean) / training_std
        ),
        "validation_fraction_outside_training_range": float(
            np.mean(
                (validation < np.min(training))
                | (validation > np.max(training))
            )
        ),
    }


def compare_surface_spot_drifts(
    states: np.ndarray,
    time_y: np.ndarray,
    *,
    config: SurfaceSpotDriftComparisonConfig = SurfaceSpotDriftComparisonConfig(),
) -> dict[str, object]:
    """Compare surface-only, log-spot-driven, and time-trend drift forecasts."""

    states = np.asarray(states, dtype=np.float64)
    time_y = np.asarray(time_y, dtype=np.float64)
    _validate_inputs(states, time_y, config)
    transitions = states.shape[0] - 1
    normalized_dt = np.diff(time_y) * DAYS_PER_YEAR
    verify_inp(
        np.all(normalized_dt > 0.0),
        "normalized time increments must be positive",
    )
    spot_driver = states[:-1, 0]
    time_driver = time_y[:-1]
    training_count = int(math.floor(config.training_fraction * transitions))
    training_indices = np.arange(training_count)
    validation_indices = np.arange(training_count, transitions)

    autonomous_fit = _fit_surface_drift(
        states=states,
        normalized_dt=normalized_dt,
        training_indices=training_indices,
        driver=None,
        covariance_eigenvalue_floor=config.covariance_eigenvalue_floor,
    )
    spot_fit = _fit_surface_drift(
        states=states,
        normalized_dt=normalized_dt,
        training_indices=training_indices,
        driver=spot_driver,
        covariance_eigenvalue_floor=config.covariance_eigenvalue_floor,
    )
    time_fit = _fit_surface_drift(
        states=states,
        normalized_dt=normalized_dt,
        training_indices=training_indices,
        driver=time_driver,
        covariance_eigenvalue_floor=config.covariance_eigenvalue_floor,
    )
    autonomous_error = _innovation_errors(
        autonomous_fit,
        states=states,
        normalized_dt=normalized_dt,
        driver=None,
    )
    spot_error = _innovation_errors(
        spot_fit,
        states=states,
        normalized_dt=normalized_dt,
        driver=spot_driver,
    )
    time_error = _innovation_errors(
        time_fit,
        states=states,
        normalized_dt=normalized_dt,
        driver=time_driver,
    )
    common_scale = autonomous_fit.innovation_scale

    def split_report(indices: np.ndarray) -> dict[str, object]:
        return {
            "autonomous_surface": _model_metrics(
                autonomous_fit,
                error=autonomous_error,
                common_innovation_scale=common_scale,
                indices=indices,
            ),
            "log_spot_driven_surface": _model_metrics(
                spot_fit,
                error=spot_error,
                common_innovation_scale=common_scale,
                indices=indices,
            ),
            "calendar_time_control": _model_metrics(
                time_fit,
                error=time_error,
                common_innovation_scale=common_scale,
                indices=indices,
            ),
            "log_spot_vs_autonomous": _pair_metrics(
                base_fit=autonomous_fit,
                alternative_fit=spot_fit,
                base_error=autonomous_error,
                alternative_error=spot_error,
                common_innovation_scale=common_scale,
                indices=indices,
                hac_lag=config.hac_lag,
            ),
            "calendar_time_vs_autonomous": _pair_metrics(
                base_fit=autonomous_fit,
                alternative_fit=time_fit,
                base_error=autonomous_error,
                alternative_error=time_error,
                common_innovation_scale=common_scale,
                indices=indices,
                hac_lag=config.hac_lag,
            ),
        }

    rolling_folds = []
    rolling_log_score_gains = []
    rolling_squared_error_reductions = []
    rolling_loadings = []
    for validation_start in range(
        config.minimum_rolling_training,
        transitions,
        config.rolling_validation_size,
    ):
        validation_end = min(
            validation_start + config.rolling_validation_size,
            transitions,
        )
        fold_training = np.arange(validation_start)
        fold_validation = np.arange(validation_start, validation_end)
        fold_autonomous = _fit_surface_drift(
            states=states,
            normalized_dt=normalized_dt,
            training_indices=fold_training,
            driver=None,
            covariance_eigenvalue_floor=config.covariance_eigenvalue_floor,
        )
        fold_spot = _fit_surface_drift(
            states=states,
            normalized_dt=normalized_dt,
            training_indices=fold_training,
            driver=spot_driver,
            covariance_eigenvalue_floor=config.covariance_eigenvalue_floor,
        )
        fold_autonomous_error = _innovation_errors(
            fold_autonomous,
            states=states,
            normalized_dt=normalized_dt,
            driver=None,
        )
        fold_spot_error = _innovation_errors(
            fold_spot,
            states=states,
            normalized_dt=normalized_dt,
            driver=spot_driver,
        )
        fold_common_scale = fold_autonomous.innovation_scale
        base_selected = fold_autonomous_error[fold_validation]
        spot_selected = fold_spot_error[fold_validation]
        log_score_gain = (
            _gaussian_log_scores(fold_spot, spot_selected)
            - _gaussian_log_scores(fold_autonomous, base_selected)
        )
        squared_error_reduction = np.sum(
            np.square(base_selected / fold_common_scale[None, :]),
            axis=1,
        ) - np.sum(
            np.square(spot_selected / fold_common_scale[None, :]),
            axis=1,
        )
        rolling_log_score_gains.append(log_score_gain)
        rolling_squared_error_reductions.append(squared_error_reduction)
        assert fold_spot.spot_drift_loading is not None
        rolling_loadings.append(fold_spot.spot_drift_loading)
        rolling_folds.append(
            {
                "training_transitions": int(validation_start),
                "validation_start_transition": int(validation_start),
                "validation_end_transition_exclusive": int(validation_end),
                "validation_transitions": int(len(fold_validation)),
                "mean_log_score_gain": float(np.mean(log_score_gain)),
                "mean_standardized_squared_error_reduction": float(
                    np.mean(squared_error_reduction)
                ),
            }
        )

    rolling_log_score_gain = np.concatenate(rolling_log_score_gains)
    rolling_squared_error_reduction = np.concatenate(
        rolling_squared_error_reductions
    )
    regression_parameters_autonomous = (states.shape[1]) * (states.shape[1] - 1)
    regression_parameters_spot = regression_parameters_autonomous + states.shape[1] - 1
    covariance_parameters = (states.shape[1] - 1) * states.shape[1] // 2
    training_autonomous_log_likelihood = float(
        np.sum(_gaussian_log_scores(autonomous_fit, autonomous_error[training_indices]))
    )
    training_spot_log_likelihood = float(
        np.sum(_gaussian_log_scores(spot_fit, spot_error[training_indices]))
    )

    return {
        "definition": {
            "autonomous": "surface drift depends on an intercept and lagged surface state",
            "log_spot_driven": (
                "log-spot has constant drift; every surface drift additionally "
                "depends on lagged log-spot"
            ),
            "calendar_time_control": (
                "replace lagged log-spot with calendar time to test whether "
                "log-spot merely proxies a trend"
            ),
            "innovation_error": (
                "(observed surface change - fitted drift) / sqrt(normalized dt)"
            ),
        },
        "observations": int(states.shape[0]),
        "transitions": int(transitions),
        "surface_dimension": int(states.shape[1] - 1),
        "training_transitions": int(training_count),
        "validation_transitions": int(len(validation_indices)),
        "configuration": {
            "training_fraction": config.training_fraction,
            "minimum_rolling_training": config.minimum_rolling_training,
            "rolling_validation_size": config.rolling_validation_size,
            "hac_lag": config.hac_lag,
            "covariance_eigenvalue_floor": config.covariance_eigenvalue_floor,
        },
        "driver_distribution_shift": {
            "log_spot": _driver_distribution_shift(
                spot_driver,
                training_indices,
                validation_indices,
            ),
            "calendar_time": _driver_distribution_shift(
                time_driver,
                training_indices,
                validation_indices,
            ),
        },
        "training": split_report(training_indices),
        "validation": split_report(validation_indices),
        "log_spot_loading": _loading_summary(spot_fit, common_scale),
        "information_criteria_training": {
            "autonomous_aic": float(
                -2.0 * training_autonomous_log_likelihood
                + 2.0 * (regression_parameters_autonomous + covariance_parameters)
            ),
            "log_spot_aic": float(
                -2.0 * training_spot_log_likelihood
                + 2.0 * (regression_parameters_spot + covariance_parameters)
            ),
            "autonomous_bic": float(
                -2.0 * training_autonomous_log_likelihood
                + math.log(training_count)
                * (regression_parameters_autonomous + covariance_parameters)
            ),
            "log_spot_bic": float(
                -2.0 * training_spot_log_likelihood
                + math.log(training_count)
                * (regression_parameters_spot + covariance_parameters)
            ),
            "additional_log_spot_coefficients": int(states.shape[1] - 1),
        },
        "expanding_window_validation": {
            "folds": rolling_folds,
            "aggregate_log_score_gain": _hac_mean_test(
                rolling_log_score_gain,
                config.hac_lag,
            ),
            "aggregate_standardized_squared_error_reduction": _hac_mean_test(
                rolling_squared_error_reduction,
                config.hac_lag,
            ),
            "coefficient_stability": _coefficient_stability(rolling_loadings),
        },
    }


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-fraction", type=float, default=0.7)
    parser.add_argument("--minimum-rolling-training", type=int, default=600)
    parser.add_argument("--rolling-validation-size", type=int, default=200)
    parser.add_argument("--hac-lag", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "tex"
        / "generated"
        / "surface_spot_drift_comparison.json",
    )
    arguments = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quantdata-matplotlib")
    experiment = GenDysanosExperiment(
        num_samples=1,
        num_alpha=None,
        use_matrix_exp=True,
        bandwidth=1.0,
    )
    _, market, _, sampler = _load_generator_components(
        experiment,
        verbose=Context.quiet,
    )
    config = SurfaceSpotDriftComparisonConfig(
        training_fraction=arguments.training_fraction,
        minimum_rolling_training=arguments.minimum_rolling_training,
        rolling_validation_size=arguments.rolling_validation_size,
        hac_lag=arguments.hac_lag,
    )
    report = compare_surface_spot_drifts(
        sampler.xstate,
        sampler.time_y,
        config=config,
    )
    sensitivity = []
    for fraction in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        sensitivity_report = compare_surface_spot_drifts(
            sampler.xstate,
            sampler.time_y,
            config=SurfaceSpotDriftComparisonConfig(
                training_fraction=fraction,
                minimum_rolling_training=arguments.minimum_rolling_training,
                rolling_validation_size=arguments.rolling_validation_size,
                hac_lag=arguments.hac_lag,
            ),
        )
        validation = sensitivity_report["validation"]
        pair = validation["log_spot_vs_autonomous"]
        sensitivity.append(
            {
                "training_fraction": fraction,
                "validation_transitions": validation["autonomous_surface"][
                    "observations"
                ],
                "log_score_gain": pair[
                    "gaussian_log_score_gain_alternative_minus_autonomous"
                ],
                "standardized_squared_error_reduction": pair[
                    "standardized_squared_error_reduction_autonomous_minus_alternative"
                ],
                "autonomous_standardized_rmse": validation[
                    "autonomous_surface"
                ]["standardized_rmse"],
                "log_spot_standardized_rmse": validation[
                    "log_spot_driven_surface"
                ]["standardized_rmse"],
            }
        )
    report["training_fraction_sensitivity"] = sensitivity
    report["data"] = {
        "first_date": str(market.dates[0]),
        "last_date": str(market.dates[-1]),
        "historical_dates": int(len(market.dates)),
        "main_training_last_date": str(
            market.dates[report["training_transitions"]]
        ),
        "decoder_state_note": (
            "The ML-SANOS encoder/decoder was trained on all surfaces; only the "
            "time-series drift regressions use chronological holdout data."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote surface-drift comparison to {arguments.output}")


if __name__ == "__main__":
    main()
