"""Test whether surface mean reversion is a genuine spot-return channel."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify, verify_inp
from scipy.linalg import expm

from ..surface_dynamics import (
    MatrixExponentialAR1,
    PCAResult,
    SurfacePanel,
    compute_surface_pca,
    fit_matrix_exponential_ar1,
    load_surface_panel,
)

STATE_LABELS = {
    "log_forward_volatility": "Log forward volatility",
    "log_dlvsq": "Log DLV squared",
}


@dataclass(frozen=True, kw_only=True)
class SurfaceReversionAttributionConfig:
    """Chronological estimation and forecast conventions."""

    num_factors: int = 5
    minimum_training_dates: int = 60
    horizons_sessions: tuple[int, ...] = (1, 5)
    ar_max_nfev: int = 4_000

    def __post_init__(self) -> None:
        verify_inp(
            self.num_factors >= 1,
            lambda: "num_factors must be positive",
        )
        verify_inp(
            self.minimum_training_dates >= self.num_factors * 4,
            lambda: (
                "minimum_training_dates must be at least four times the "
                "number of factors"
            ),
        )
        verify_inp(
            self.horizons_sessions
            and all(horizon >= 1 for horizon in self.horizons_sessions)
            and len(set(self.horizons_sessions)) == len(self.horizons_sessions),
            lambda: "horizons_sessions must contain unique positive integers",
        )
        verify_inp(
            self.ar_max_nfev >= 100,
            lambda: "ar_max_nfev must be at least 100",
        )


@dataclass(frozen=True)
class ExpandingSurfaceFit:
    """One expanding-window PCA and matrix-exponential AR fit."""

    pca: PCAResult
    ar: MatrixExponentialAR1
    standardized_states: np.ndarray
    current_residual_surface: np.ndarray


@dataclass(frozen=True)
class SpotStateFit:
    """Historical spot regression conditional on state and surface shocks."""

    intercept: float
    state_coefficients: np.ndarray
    shock_coefficients: np.ndarray
    expanding_mean_log_return: float


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _newey_west_mean_t(values: np.ndarray, *, max_lag: int) -> float:
    """Return a HAC t-statistic for a nonzero mean."""

    verify_inp(
        values.ndim == 1 and len(values) >= 2 and np.all(np.isfinite(values)),
        lambda: "HAC mean input must be a finite vector with at least two values",
    )
    centered = values - np.mean(values)
    num_values = len(values)
    lag = min(max_lag, num_values - 1)
    long_run_variance = float(centered @ centered) / num_values
    for offset in range(1, lag + 1):
        covariance = float(centered[offset:] @ centered[:-offset]) / num_values
        long_run_variance += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = math.sqrt(long_run_variance / num_values)
    if standard_error == 0.0:
        return float("nan")
    return float(np.mean(values) / standard_error)


def _fit_expanding_surface(
    training_surfaces: np.ndarray,
    training_dates: list[date],
    *,
    config: SurfaceReversionAttributionConfig,
) -> ExpandingSurfaceFit:
    pca = compute_surface_pca(
        training_surfaces,
        num_factors=config.num_factors,
    )
    ar = fit_matrix_exponential_ar1(
        pca.scores,
        training_dates,
        max_nfev=config.ar_max_nfev,
    )
    standardized = pca.scores / ar.factor_scales[None, :]
    loadings = pca.loadings.reshape(config.num_factors, -1)
    reconstructed = pca.mean_surface + (
        pca.scores[-1] @ loadings
    ).reshape(pca.mean_surface.shape)
    residual = training_surfaces[-1] - reconstructed
    return ExpandingSurfaceFit(
        pca=pca,
        ar=ar,
        standardized_states=standardized,
        current_residual_surface=residual,
    )


def _project_surface(surface: np.ndarray, fit: ExpandingSurfaceFit) -> np.ndarray:
    centered = (surface - fit.pca.mean_surface).reshape(-1)
    scores = centered @ fit.pca.loadings.reshape(len(fit.ar.factor_scales), -1).T
    return scores / fit.ar.factor_scales


def _transition(ar: MatrixExponentialAR1, gap_days: int) -> np.ndarray:
    return expm(ar.generator * (gap_days / 365.25))


def forecast_surface(
    fit: ExpandingSurfaceFit,
    *,
    gap_days: int,
) -> np.ndarray:
    """Forecast a log surface, retaining today's unmodeled PCA residual."""

    verify_inp(gap_days >= 1, lambda: "gap_days must be positive")
    future_state = _transition(fit.ar, gap_days) @ fit.standardized_states[-1]
    future_scores = future_state * fit.ar.factor_scales
    loadings = fit.pca.loadings.reshape(len(future_scores), -1)
    forecast = fit.pca.mean_surface + (future_scores @ loadings).reshape(
        fit.pca.mean_surface.shape
    )
    return forecast + fit.current_residual_surface


def _surface_training_shocks(
    fit: ExpandingSurfaceFit,
    training_dates: list[date],
) -> np.ndarray:
    previous = fit.standardized_states[:-1]
    current = fit.standardized_states[1:]
    predictions = np.empty_like(current)
    gaps = np.asarray(
        [
            (right - left).days
            for left, right in zip(
                training_dates[:-1], training_dates[1:], strict=True
            )
        ],
        dtype=np.int64,
    )
    for gap in np.unique(gaps):
        mask = gaps == gap
        predictions[mask] = previous[mask] @ _transition(fit.ar, int(gap)).T
    return current - predictions


def _fit_spot_state_model(
    fit: ExpandingSurfaceFit,
    training_dates: list[date],
    training_spots: np.ndarray,
) -> SpotStateFit:
    verify_inp(
        training_spots.shape == (len(training_dates),)
        and np.all(np.isfinite(training_spots))
        and np.all(training_spots > 0.0),
        lambda: "training spots must be positive and aligned with dates",
    )
    shocks = _surface_training_shocks(fit, training_dates)
    log_returns = np.diff(np.log(training_spots))
    design = np.column_stack(
        (
            np.ones(len(shocks), dtype=np.float64),
            fit.standardized_states[:-1],
            shocks,
        )
    )
    coefficients = np.linalg.lstsq(design, log_returns, rcond=None)[0]
    num_factors = fit.standardized_states.shape[1]
    return SpotStateFit(
        intercept=float(coefficients[0]),
        state_coefficients=np.asarray(coefficients[1 : 1 + num_factors]),
        shock_coefficients=np.asarray(coefficients[1 + num_factors :]),
        expanding_mean_log_return=float(np.mean(log_returns)),
    )


def forecast_spot_log_return(
    current_state: np.ndarray,
    calendar_gaps_days: np.ndarray,
    ar: MatrixExponentialAR1,
    spot_fit: SpotStateFit,
    *,
    revert_state: bool,
) -> float:
    """Return expected cumulative spot log return with zero future shocks."""

    verify_inp(
        current_state.ndim == 1
        and calendar_gaps_days.ndim == 1
        and len(calendar_gaps_days) >= 1
        and np.all(calendar_gaps_days >= 1),
        lambda: "state and positive calendar gaps are required",
    )
    state = current_state.copy()
    expected_return = 0.0
    for gap_days in calendar_gaps_days:
        expected_return += spot_fit.intercept + float(
            spot_fit.state_coefficients @ state
        )
        if revert_state:
            state = _transition(ar, int(gap_days)) @ state
    return expected_return


def _explain_realized_spot_return(
    surfaces: np.ndarray,
    dates: list[date],
    origin_index: int,
    horizon_sessions: int,
    fit: ExpandingSurfaceFit,
    spot_fit: SpotStateFit,
) -> tuple[float, float]:
    """Use realized future surface states to separate state and shock components."""

    state_component = 0.0
    shock_component = 0.0
    previous_state = fit.standardized_states[-1]
    for index in range(origin_index + 1, origin_index + horizon_sessions + 1):
        current_state = _project_surface(surfaces[index], fit)
        gap_days = (dates[index] - dates[index - 1]).days
        shock = current_state - _transition(fit.ar, gap_days) @ previous_state
        state_component += spot_fit.intercept + float(
            spot_fit.state_coefficients @ previous_state
        )
        shock_component += float(spot_fit.shock_coefficients @ shock)
        previous_state = current_state
    return state_component, shock_component


def _load_spots(daily_summary_file: Path, panel: SurfacePanel) -> np.ndarray:
    daily = pl.read_parquet(daily_summary_file).select("date", "spot_ref").sort("date")
    verify(
        daily.height == len(panel.dates)
        and daily["date"].to_list() == panel.dates,
        lambda: "daily spot dates do not align exactly with the surface panel",
    )
    spots = daily["spot_ref"].to_numpy()
    verify(
        np.all(np.isfinite(spots)) and np.all(spots > 0.0),
        lambda: "daily spots contain a non-finite or nonpositive observation",
    )
    return spots


def build_expanding_diagnostics(
    panel: SurfacePanel,
    spots: np.ndarray,
    *,
    config: SurfaceReversionAttributionConfig,
) -> pl.DataFrame:
    """Build genuinely out-of-sample surface and spot forecast observations."""

    verify_inp(
        spots.shape == (len(panel.dates),),
        lambda: "spots must align with panel dates",
    )
    verify_inp(
        len(panel.dates) > config.minimum_training_dates + min(config.horizons_sessions),
        lambda: "the panel is too short for the requested expanding diagnostic",
    )
    atm_expiry_index = int(np.argmin(np.abs(panel.expiry_days - 5)))
    atm_nms_index = int(np.argmin(np.abs(panel.effective_nms)))
    rows: list[dict[str, object]] = []
    maximum_horizon = max(config.horizons_sessions)
    for origin_index in range(config.minimum_training_dates - 1, len(panel.dates) - 1):
        training_dates = panel.dates[: origin_index + 1]
        fits = {
            state: _fit_expanding_surface(
                panel.states[state][: origin_index + 1],
                training_dates,
                config=config,
            )
            for state in STATE_LABELS
        }
        forward_fit = fits["log_forward_volatility"]
        spot_fit = _fit_spot_state_model(
            forward_fit,
            training_dates,
            spots[: origin_index + 1],
        )
        for horizon in config.horizons_sessions:
            target_index = origin_index + horizon
            if target_index >= len(panel.dates):
                continue
            calendar_gaps = np.asarray(
                [
                    (panel.dates[index] - panel.dates[index - 1]).days
                    for index in range(origin_index + 1, target_index + 1)
                ],
                dtype=np.int64,
            )
            realized_spot_return = float(
                math.log(spots[target_index] / spots[origin_index])
            )
            predicted_spot_ar = forecast_spot_log_return(
                forward_fit.standardized_states[-1],
                calendar_gaps,
                forward_fit.ar,
                spot_fit,
                revert_state=True,
            )
            predicted_spot_frozen = forecast_spot_log_return(
                forward_fit.standardized_states[-1],
                calendar_gaps,
                forward_fit.ar,
                spot_fit,
                revert_state=False,
            )
            state_component, shock_component = _explain_realized_spot_return(
                panel.states["log_forward_volatility"],
                panel.dates,
                origin_index,
                horizon,
                forward_fit,
                spot_fit,
            )
            for state, fit in fits.items():
                current_surface = panel.states[state][origin_index]
                realized_surface = panel.states[state][target_index]
                gap_days = (panel.dates[target_index] - panel.dates[origin_index]).days
                predicted_surface = forecast_surface(fit, gap_days=gap_days)
                predicted_change = predicted_surface - current_surface
                realized_change = realized_surface - current_surface
                ar_error = realized_surface - predicted_surface
                no_change_error = realized_change
                predicted_norm = float(np.linalg.norm(predicted_change))
                realized_norm = float(np.linalg.norm(realized_change))
                direction_cosine = (
                    float(predicted_change.reshape(-1) @ realized_change.reshape(-1))
                    / (predicted_norm * realized_norm)
                    if predicted_norm > 0.0 and realized_norm > 0.0
                    else float("nan")
                )
                rows.append(
                    {
                        "date": panel.dates[origin_index],
                        "target_date": panel.dates[target_index],
                        "origin_index": origin_index,
                        "num_training_dates": origin_index + 1,
                        "state": state,
                        "horizon_sessions": horizon,
                        "calendar_horizon_days": gap_days,
                        "ar_mean_reverting": fit.ar.is_mean_reverting,
                        "ar_spectral_radius_one_day": fit.ar.spectral_radius_one_day,
                        "ar_training_r_squared": fit.ar.r_squared,
                        "ar_surface_mse": float(np.mean(np.square(ar_error))),
                        "no_change_surface_mse": float(
                            np.mean(np.square(no_change_error))
                        ),
                        "surface_loss_improvement": float(
                            np.mean(np.square(no_change_error))
                            - np.mean(np.square(ar_error))
                        ),
                        "surface_direction_cosine": direction_cosine,
                        "surface_direction_correct": bool(direction_cosine > 0.0),
                        "predicted_atm_log_change": float(
                            predicted_change[atm_expiry_index, atm_nms_index]
                        ),
                        "realized_atm_log_change": float(
                            realized_change[atm_expiry_index, atm_nms_index]
                        ),
                        "predicted_change_rms": float(
                            np.sqrt(np.mean(np.square(predicted_change)))
                        ),
                        "realized_spot_log_return": realized_spot_return,
                        "predicted_spot_log_return_ar": predicted_spot_ar,
                        "predicted_spot_log_return_frozen": predicted_spot_frozen,
                        "predicted_spot_log_return_expanding_mean": (
                            spot_fit.expanding_mean_log_return * horizon
                        ),
                        "realized_state_spot_component": state_component,
                        "realized_surface_shock_spot_component": shock_component,
                        "surface_explanatory_spot_fit": (
                            state_component + shock_component
                        ),
                    }
                )
        if origin_index + maximum_horizon >= len(panel.dates):
            continue
    diagnostics = pl.DataFrame(rows).sort("state", "horizon_sessions", "date")
    numeric = diagnostics.select(pl.selectors.numeric()).to_numpy()
    verify(
        np.all(np.isfinite(numeric)),
        lambda: "expanding surface-reversion diagnostics contain non-finite values",
    )
    return diagnostics


def summarize_surface_forecasts(diagnostics: pl.DataFrame) -> pl.DataFrame:
    """Summarize AR surface forecasts against a no-change benchmark."""

    rows: list[dict[str, object]] = []
    for key, group in diagnostics.partition_by(
        "state", "horizon_sessions", as_dict=True
    ).items():
        ar_mse = group["ar_surface_mse"].to_numpy()
        no_change_mse = group["no_change_surface_mse"].to_numpy()
        improvement = group["surface_loss_improvement"].to_numpy()
        predicted_atm = group["predicted_atm_log_change"].to_numpy()
        realized_atm = group["realized_atm_log_change"].to_numpy()
        horizon = int(key[1])
        rows.append(
            {
                "state": key[0],
                "horizon_sessions": horizon,
                "num_forecasts": len(group),
                "mean_reverting_fit_fraction": float(
                    group["ar_mean_reverting"].mean()
                ),
                "mean_training_r_squared": float(
                    group["ar_training_r_squared"].mean()
                ),
                "ar_rmse_log_units": float(np.sqrt(np.mean(ar_mse))),
                "no_change_rmse_log_units": float(
                    np.sqrt(np.mean(no_change_mse))
                ),
                "mse_skill_vs_no_change": float(
                    1.0 - np.mean(ar_mse) / np.mean(no_change_mse)
                ),
                "mean_loss_improvement": float(np.mean(improvement)),
                "loss_improvement_hac_t": _newey_west_mean_t(
                    improvement,
                    max_lag=horizon - 1,
                ),
                "mean_direction_cosine": float(
                    group["surface_direction_cosine"].mean()
                ),
                "direction_correct_fraction": float(
                    group["surface_direction_correct"].mean()
                ),
                "predicted_realized_atm_change_correlation": _safe_correlation(
                    predicted_atm,
                    realized_atm,
                ),
            }
        )
    return pl.DataFrame(rows).sort("state", "horizon_sessions")


def summarize_spot_channel(diagnostics: pl.DataFrame) -> pl.DataFrame:
    """Summarize feasible spot forecasts and the realized-shock explanation."""

    forward_rows = diagnostics.filter(pl.col("state") == "log_forward_volatility")
    rows: list[dict[str, object]] = []
    for key, group in forward_rows.partition_by("horizon_sessions", as_dict=True).items():
        horizon = int(key[0])
        actual = group["realized_spot_log_return"].to_numpy()
        baseline = group["predicted_spot_log_return_expanding_mean"].to_numpy()
        baseline_mse = float(np.mean(np.square(actual - baseline)))
        state_path = group["realized_state_spot_component"].to_numpy()
        shock_fit = group["surface_explanatory_spot_fit"].to_numpy()
        shock_component = group["realized_surface_shock_spot_component"].to_numpy()
        for forecast_name, column in (
            ("AR_reverting_state", "predicted_spot_log_return_ar"),
            ("frozen_state", "predicted_spot_log_return_frozen"),
            ("expanding_mean", "predicted_spot_log_return_expanding_mean"),
            ("realized_state_path", "realized_state_spot_component"),
            ("state_plus_realized_surface_shock", "surface_explanatory_spot_fit"),
        ):
            prediction = group[column].to_numpy()
            mse = float(np.mean(np.square(actual - prediction)))
            rows.append(
                {
                    "horizon_sessions": horizon,
                    "forecast": forecast_name,
                    "uses_future_surface": forecast_name
                    in {
                        "realized_state_path",
                        "state_plus_realized_surface_shock",
                    },
                    "num_forecasts": len(group),
                    "mean_predicted_log_return": float(np.mean(prediction)),
                    "mean_realized_log_return": float(np.mean(actual)),
                    "rmse_log_return": float(np.sqrt(mse)),
                    "mse_skill_vs_expanding_mean": float(
                        1.0 - mse / baseline_mse
                    ),
                    "prediction_correlation": _safe_correlation(prediction, actual),
                    "sign_correct_fraction": float(
                        np.mean(np.sign(prediction) == np.sign(actual))
                    ),
                    "shock_incremental_mse_skill": (
                        float(
                            1.0
                            - np.mean(np.square(actual - shock_fit))
                            / np.mean(np.square(actual - state_path))
                        )
                        if forecast_name == "state_plus_realized_surface_shock"
                        else None
                    ),
                    "shock_component_spot_residual_correlation": (
                        _safe_correlation(shock_component, actual - state_path)
                        if forecast_name == "state_plus_realized_surface_shock"
                        else None
                    ),
                }
            )
    return pl.DataFrame(rows).sort("horizon_sessions", "forecast")


def summarize_reversion_counterfactual(diagnostics: pl.DataFrame) -> pl.DataFrame:
    """Compare expected spot returns with reverting and frozen state paths."""

    forward_rows = diagnostics.filter(pl.col("state") == "log_forward_volatility")
    rows: list[dict[str, object]] = []
    for key, group in forward_rows.partition_by("horizon_sessions", as_dict=True).items():
        horizon = int(key[0])
        actual = group["realized_spot_log_return"].to_numpy()
        ar_prediction = group["predicted_spot_log_return_ar"].to_numpy()
        frozen_prediction = group["predicted_spot_log_return_frozen"].to_numpy()
        ar_loss = np.square(actual - ar_prediction)
        frozen_loss = np.square(actual - frozen_prediction)
        reversion_benefit = frozen_loss - ar_loss
        atm_reversion = group["predicted_atm_log_change"].to_numpy()
        rows.append(
            {
                "horizon_sessions": horizon,
                "num_forecasts": len(group),
                "ar_spot_mse": float(np.mean(ar_loss)),
                "frozen_state_spot_mse": float(np.mean(frozen_loss)),
                "ar_mse_skill_vs_frozen_state": float(
                    1.0 - np.mean(ar_loss) / np.mean(frozen_loss)
                ),
                "mean_ar_loss_improvement": float(np.mean(reversion_benefit)),
                "ar_loss_improvement_hac_t": _newey_west_mean_t(
                    reversion_benefit,
                    max_lag=horizon - 1,
                ),
                "ar_vs_frozen_prediction_correlation": _safe_correlation(
                    ar_prediction,
                    frozen_prediction,
                ),
                "mean_abs_ar_frozen_forecast_difference": float(
                    np.mean(np.abs(ar_prediction - frozen_prediction))
                ),
                "predicted_atm_reversion_spot_return_correlation": (
                    _safe_correlation(atm_reversion, actual)
                ),
            }
        )
    return pl.DataFrame(rows).sort("horizon_sessions")


def build_signal_links(
    diagnostics: pl.DataFrame,
    comparison_files: dict[int, Path],
) -> pl.DataFrame:
    """Relate existing OCE deltas to chronological state forecasts."""

    rows: list[dict[str, object]] = []
    forward_rows = diagnostics.filter(pl.col("state") == "log_forward_volatility")
    for horizon, comparison_file in comparison_files.items():
        if not comparison_file.exists():
            continue
        state = forward_rows.filter(pl.col("horizon_sessions") == horizon).select(
            "date",
            "predicted_spot_log_return_ar",
            "predicted_spot_log_return_frozen",
            "predicted_atm_log_change",
        )
        comparison = (
            pl.read_parquet(comparison_file)
            .filter(pl.col("spot_complete"))
            .join(state, on="date", how="inner")
        )
        for key, group in comparison.partition_by(
            "universe", "utility", "risk_aversion", as_dict=True
        ).items():
            delta = group["net_entry_delta"].to_numpy()
            predicted_spot = group["predicted_spot_log_return_ar"].to_numpy()
            predicted_atm = group["predicted_atm_log_change"].to_numpy()
            rows.append(
                {
                    "horizon_sessions": horizon,
                    "universe": key[0],
                    "utility": key[1],
                    "risk_aversion": float(key[2]),
                    "num_dates": len(group),
                    "delta_predicted_spot_correlation": _safe_correlation(
                        delta, predicted_spot
                    ),
                    "delta_predicted_atm_reversion_correlation": _safe_correlation(
                        delta, predicted_atm
                    ),
                    "delta_predicted_spot_sign_agreement": float(
                        np.mean(np.sign(delta) == np.sign(predicted_spot))
                    ),
                    "delta_positive_fraction": float(np.mean(delta > 0.0)),
                    "predicted_spot_positive_fraction": float(
                        np.mean(predicted_spot > 0.0)
                    ),
                }
            )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(
        "horizon_sessions", "universe", "utility", "risk_aversion"
    )


def _render_figure(diagnostics: pl.DataFrame, output_file: Path) -> None:
    import matplotlib.pyplot as plt

    forward = diagnostics.filter(
        (pl.col("state") == "log_forward_volatility")
        & (pl.col("horizon_sessions") == 5)
    ).sort("date")
    one_day = diagnostics.filter(
        (pl.col("state") == "log_forward_volatility")
        & (pl.col("horizon_sessions") == 1)
    ).sort("date")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    dates = forward["date"].to_list()
    axes[0, 0].plot(
        dates,
        np.cumsum(forward["ar_surface_mse"].to_numpy()),
        label="AR reversion",
    )
    axes[0, 0].plot(
        dates,
        np.cumsum(forward["no_change_surface_mse"].to_numpy()),
        label="No surface change",
    )
    axes[0, 0].set_title("Five-session cumulative surface squared error")
    axes[0, 0].set_ylabel("Cumulative mean grid-point squared error")
    axes[0, 0].legend()

    axes[0, 1].scatter(
        forward["predicted_atm_log_change"].to_numpy(),
        forward["realized_atm_log_change"].to_numpy(),
        alpha=0.65,
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[0, 1].axvline(0.0, color="black", linewidth=0.7)
    axes[0, 1].set_title("Five-session front-week ATM reversion")
    axes[0, 1].set_xlabel("Predicted log-vol change")
    axes[0, 1].set_ylabel("Realized log-vol change")

    axes[1, 0].scatter(
        forward["predicted_spot_log_return_ar"].to_numpy(),
        forward["realized_spot_log_return"].to_numpy(),
        alpha=0.65,
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.7)
    axes[1, 0].axvline(0.0, color="black", linewidth=0.7)
    axes[1, 0].set_title("Feasible five-session spot forecast")
    axes[1, 0].set_xlabel("Predicted spot log return")
    axes[1, 0].set_ylabel("Realized spot log return")

    state_residual = (
        one_day["realized_spot_log_return"].to_numpy()
        - one_day["realized_state_spot_component"].to_numpy()
    )
    axes[1, 1].scatter(
        one_day["realized_surface_shock_spot_component"].to_numpy(),
        state_residual,
        alpha=0.65,
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[1, 1].axvline(0.0, color="black", linewidth=0.7)
    axes[1, 1].set_title("One-session realized surface-shock explanation")
    axes[1, 1].set_xlabel("Surface-shock spot component")
    axes[1, 1].set_ylabel("Spot return after state component")

    figure.suptitle("Surface reversion and SPX spot: expanding-window diagnostics")
    figure.savefig(output_file, dpi=160)
    plt.close(figure)


def run_surface_reversion_attribution(
    panel_file: Path,
    daily_summary_file: Path,
    output_dir: Path,
    *,
    comparison_files: dict[int, Path] | None = None,
    config: SurfaceReversionAttributionConfig = SurfaceReversionAttributionConfig(),
) -> dict[str, object]:
    """Run and persist the complete chronological hypothesis test."""

    panel = load_surface_panel(panel_file)
    spots = _load_spots(daily_summary_file, panel)
    diagnostics = build_expanding_diagnostics(panel, spots, config=config)
    surface_summary = summarize_surface_forecasts(diagnostics)
    spot_summary = summarize_spot_channel(diagnostics)
    counterfactual = summarize_reversion_counterfactual(diagnostics)
    signal_links = build_signal_links(diagnostics, comparison_files or {})

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.write_parquet(output_dir / "surface_reversion_observations.parquet")
    surface_summary.write_parquet(output_dir / "surface_forecast_summary.parquet")
    spot_summary.write_parquet(output_dir / "spot_channel_summary.parquet")
    counterfactual.write_parquet(output_dir / "reversion_counterfactual.parquet")
    if not signal_links.is_empty():
        signal_links.write_parquet(output_dir / "oce_delta_signal_links.parquet")
    _render_figure(diagnostics, output_dir / "surface_reversion_spot.png")

    report: dict[str, object] = {
        "panel_file": str(panel_file),
        "daily_summary_file": str(daily_summary_file),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "date_range": [panel.dates[0].isoformat(), panel.dates[-1].isoformat()],
        "methodology": (
            "Expanding-window PCA, matrix-exponential AR, and spot regression; "
            "every forecast uses only observations available at its origin."
        ),
        "surface_benchmark": "unchanged current log surface",
        "spot_benchmark": "expanding historical mean log return",
        "reversion_counterfactual": (
            "hold the current standardized surface state fixed throughout the "
            "forecast horizon while retaining the same fitted spot equation"
        ),
        "realized_shock_caveat": (
            "state_plus_realized_surface_shock is explanatory, not tradable, "
            "because it uses the future realized surface innovation"
        ),
        "oce_signal_link_caveat": (
            "the joined OCE deltas came from the existing full-sample diagnostic; "
            "their link is descriptive and is not a counterfactual optimizer rerun"
        ),
        "surface_forecast_summary": surface_summary.to_dicts(),
        "spot_channel_summary": spot_summary.to_dicts(),
        "reversion_counterfactual_summary": counterfactual.to_dicts(),
        "num_oce_signal_link_rows": signal_links.height,
    }
    (output_dir / "surface_reversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    project = Path("aae/ideas/localvolbysanos")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-file",
        type=Path,
        default=project / "results/2025_full_reference_grid/balanced_panel.parquet",
    )
    parser.add_argument(
        "--daily-summary-file",
        type=Path,
        default=project / "results/2025_full_reference_grid/daily_summary.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project
        / "real_trading/results/2025/surface_reversion_attribution",
    )
    parser.add_argument("--minimum-training-dates", type=int, default=60)
    arguments = parser.parse_args()
    comparison_base = project / "real_trading/results/2025"
    run_surface_reversion_attribution(
        arguments.panel_file,
        arguments.daily_summary_file,
        arguments.output_dir,
        comparison_files={
            1: comparison_base / "one_day/spot_only_comparison.parquet",
            5: comparison_base / "spot_only_comparison.parquet",
        },
        config=SurfaceReversionAttributionConfig(
            minimum_training_dates=arguments.minimum_training_dates
        ),
    )


if __name__ == "__main__":
    main()
