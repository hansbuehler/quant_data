"""Correlation-aware decomposition of Heston and residual-DLV innovations."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from quantdata.gromm2.background import (
    cir_conditional_moments,
    fit_cir_quasi_likelihood,
    kernel_annual_volatilities,
)
from quantdata.gromm2.estimation import load_estimation_panel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class CorrelationAwarePilotConfig:
    """Configuration for the training/holdout decomposition gate."""

    train_fraction: float = 0.8
    background_annual_volatility: float = 0.05
    residual_spot_volatility_floor: float = 0.01
    factor_count: int = 16
    smoothing_expiry_sigma: float = 1.0
    smoothing_strike_sigma: float = 2.0
    dt: float = 1.0 / 252.0

    def __post_init__(self) -> None:
        if not 0.5 <= self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie in [0.5, 1)")
        if self.background_annual_volatility <= 0.0:
            raise ValueError("background_annual_volatility must be positive")
        if self.residual_spot_volatility_floor <= 0.0:
            raise ValueError("residual_spot_volatility_floor must be positive")
        if self.factor_count < 2:
            raise ValueError("factor_count must be at least two")
        if self.smoothing_expiry_sigma < 0.0 or self.smoothing_strike_sigma < 0.0:
            raise ValueError("smoothing sigmas must be non-negative")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")


@dataclass(frozen=True, kw_only=True)
class ResidualSpotPosterior:
    """Conditional moments of the residual-front Gaussian innovation."""

    mean: FloatArray
    variance: FloatArray
    conditional_return_score: FloatArray
    residual_annual_variance: FloatArray
    heston_idiosyncratic_annual_variance: FloatArray


def residual_spot_posterior(
    *,
    annualized_spot_return: FloatArray,
    total_annual_variance: FloatArray,
    heston_variance: FloatArray,
    heston_variance_score: FloatArray,
    heston_correlation: float,
    residual_variance_floor: float,
) -> ResidualSpotPosterior:
    """Condition the latent front shock on return and Heston variance shock.

    The observation equation is

    ``r = sqrt(v) rho z_v + sqrt(v(1-rho^2)) z_perp + sigma_x z_x``.

    Both unobserved shocks are standard normal and independent. The returned
    mean and variance are the exact Gaussian posterior moments of ``z_x``.
    """
    annualized = np.asarray(annualized_spot_return, dtype=np.float64)
    total_variance = np.asarray(total_annual_variance, dtype=np.float64)
    variance = np.asarray(heston_variance, dtype=np.float64)
    variance_score = np.asarray(heston_variance_score, dtype=np.float64)
    if not (
        annualized.shape
        == total_variance.shape
        == variance.shape
        == variance_score.shape
    ):
        raise ValueError("spot and variance arrays must have identical shapes")
    if (
        annualized.ndim != 1
        or np.any(~np.isfinite(annualized))
        or np.any(~np.isfinite(total_variance))
        or np.any(~np.isfinite(variance))
        or np.any(~np.isfinite(variance_score))
        or np.any(total_variance <= 0.0)
        or np.any(variance <= 0.0)
        or not -1.0 < heston_correlation < 1.0
        or residual_variance_floor <= 0.0
    ):
        raise ValueError("invalid residual-spot posterior inputs")

    residual_variance = np.maximum(
        total_variance - variance,
        residual_variance_floor,
    )
    heston_idiosyncratic_variance = variance * (1.0 - heston_correlation**2)
    observation_variance = residual_variance + heston_idiosyncratic_variance
    centered_observation = (
        annualized
        - heston_correlation * np.sqrt(variance) * variance_score
    )
    residual_volatility = np.sqrt(residual_variance)
    posterior_mean = residual_volatility * centered_observation / observation_variance
    posterior_variance = heston_idiosyncratic_variance / observation_variance
    conditional_return_score = centered_observation / np.sqrt(observation_variance)
    return ResidualSpotPosterior(
        mean=posterior_mean,
        variance=posterior_variance,
        conditional_return_score=conditional_return_score,
        residual_annual_variance=residual_variance,
        heston_idiosyncratic_annual_variance=heston_idiosyncratic_variance,
    )


def latent_loading(
    innovations: FloatArray,
    posterior_mean: FloatArray,
    posterior_variance: FloatArray,
) -> FloatArray:
    """Estimate a loading using posterior first and second moments."""
    values = np.asarray(innovations, dtype=np.float64)
    mean = np.asarray(posterior_mean, dtype=np.float64)
    variance = np.asarray(posterior_variance, dtype=np.float64)
    if (
        values.ndim != 2
        or mean.shape != (len(values),)
        or variance.shape != mean.shape
        or np.any(variance < 0.0)
    ):
        raise ValueError("invalid latent-loading inputs")
    denominator = float(np.sum(mean**2 + variance))
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("latent score has zero posterior second moment")
    return mean @ values / denominator


def _kernel_volatility_panel(
    *,
    dataset_path: Path,
    target_business_dtes: FloatArray,
) -> tuple[NDArray[np.str_], FloatArray]:
    geometry = json.loads((dataset_path / "geometry.json").read_text())
    smoothness = float(geometry["sanos_smoothness"])
    geometry_rows = pl.read_parquet(
        dataset_path / "sanos_geometry.parquet",
        columns=["date", "model_expiries_y"],
    )
    state_rows = pl.read_parquet(
        dataset_path / "sanos_state.parquet",
        columns=["date", "kernel_total_variances"],
    )
    if geometry_rows.height != state_rows.height:
        raise ValueError("SANOS geometry and state panels do not align")
    target_expiries = target_business_dtes / 252.0
    dates: list[str] = []
    volatilities: list[FloatArray] = []
    for geometry_row, state_row in zip(
        geometry_rows.iter_rows(named=True),
        state_rows.iter_rows(named=True),
        strict=True,
    ):
        if geometry_row["date"] != state_row["date"]:
            raise ValueError("SANOS geometry and state dates do not align")
        dates.append(geometry_row["date"].isoformat())
        volatilities.append(
            kernel_annual_volatilities(
                np.asarray(geometry_row["model_expiries_y"], dtype=np.float64),
                np.asarray(state_row["kernel_total_variances"], dtype=np.float64),
                target_expiries,
                smoothness=smoothness,
            )
        )
    return np.asarray(dates), np.stack(volatilities)


def _fit_ar_innovations(
    *,
    origins: FloatArray,
    endpoints: FloatArray,
    train_count: int,
    nms_nodes: FloatArray,
    config: CorrelationAwarePilotConfig,
) -> FloatArray:
    train_origins = origins[:train_count]
    train_endpoints = endpoints[:train_count]
    origin_mean = np.mean(train_origins, axis=0)
    endpoint_mean = np.mean(train_endpoints, axis=0)
    centered_origins = train_origins - origin_mean
    centered_endpoints = train_endpoints - endpoint_mean
    denominator = np.sum(centered_origins**2, axis=0)
    scale = np.maximum(np.sum(train_origins**2, axis=0), 1.0)
    variable = denominator > np.finfo(np.float64).eps * scale * train_count
    slope = np.zeros_like(denominator)
    slope[variable] = (
        np.sum(centered_origins * centered_endpoints, axis=0)[variable]
        / denominator[variable]
    )
    intercept = endpoint_mean - slope * origin_mean
    innovations = endpoints - (intercept + slope * origins)
    innovations -= np.mean(innovations[:train_count], axis=0, keepdims=True)
    smoothed = gaussian_filter(
        innovations,
        sigma=(0.0, config.smoothing_expiry_sigma, config.smoothing_strike_sigma),
        mode="nearest",
    )
    smoothed -= np.mean(smoothed[:train_count], axis=0, keepdims=True)
    core = (nms_nodes >= -2.5) & (nms_nodes <= 1.5)
    smoothed[..., ~core] = 0.0
    return smoothed.reshape(len(smoothed), -1) / math.sqrt(config.dt)


def _fit_atm_loadings(
    *,
    log_atm_volatility_changes: FloatArray,
    variance_scores: FloatArray,
    residual_posterior: ResidualSpotPosterior,
    train_count: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    changes = np.asarray(log_atm_volatility_changes, dtype=np.float64)
    variance_scores = np.asarray(variance_scores, dtype=np.float64)
    posterior_mean = residual_posterior.mean
    posterior_variance = residual_posterior.variance
    variance_score = variance_scores[:train_count]
    latent_mean = posterior_mean[:train_count]
    latent_variance = posterior_variance[:train_count]
    targets = changes[:train_count]
    gram = np.asarray(
        [
            [
                variance_score @ variance_score,
                variance_score @ latent_mean,
            ],
            [
                variance_score @ latent_mean,
                np.sum(latent_mean**2 + latent_variance),
            ],
        ]
    )
    right = np.stack(
        (
            variance_score @ targets,
            latent_mean @ targets,
        )
    )
    loadings = np.linalg.solve(gram, right)
    fitted_mean = (
        variance_scores[:, None] * loadings[0][None]
        + posterior_mean[:, None] * loadings[1][None]
    )
    residual_second_moment = (
        (changes - fitted_mean) ** 2
        + posterior_variance[:, None] * loadings[1][None] ** 2
    )
    residual_variance = np.mean(residual_second_moment[:train_count], axis=0)
    return loadings[0], loadings[1], residual_variance


def _model_atm_correlations(
    *,
    heston_loading: FloatArray,
    residual_loading: FloatArray,
    residual_noise_variance: FloatArray,
    heston_variance: FloatArray,
    residual_spot_variance: FloatArray,
    heston_correlation: float,
    dt: float,
) -> FloatArray:
    covariance = math.sqrt(dt) * (
        np.mean(np.sqrt(heston_variance))
        * heston_correlation
        * heston_loading
        + np.mean(np.sqrt(residual_spot_variance)) * residual_loading
    )
    spot_variance = dt * np.mean(heston_variance + residual_spot_variance)
    volatility_variance = (
        heston_loading**2 + residual_loading**2 + residual_noise_variance
    )
    return covariance / np.sqrt(spot_variance * volatility_variance)


def _selected_term_structure(
    business_dtes: FloatArray,
    values: FloatArray,
) -> dict[str, float]:
    selected: dict[str, float] = {}
    for business_dte in (5.0, 20.0, 60.0, 120.0):
        matches = np.flatnonzero(business_dtes == business_dte)
        if len(matches) == 1:
            selected[f"{int(business_dte)}D"] = float(values[int(matches[0])])
    return selected


def run_correlation_aware_pilot(
    *,
    history_path: Path,
    source_history_path: Path,
    dataset_path: Path,
    heston_pilot_path: Path,
    output_path: Path,
    config: CorrelationAwarePilotConfig,
) -> dict[str, object]:
    """Run a chronological pilot without altering the existing DLV history."""
    panel = load_estimation_panel(
        history_path=history_path,
        dataset=dataset_path,
        backbone_annual_volatility=config.background_annual_volatility,
    )
    with h5py.File(source_history_path, "r") as stream:
        dates = np.asarray(stream["dates"]).astype("U10")
        business_dtes = np.asarray(stream["model_business_dtes"], dtype=np.float64)
        expiries = np.asarray(stream["model_expiries_y"], dtype=np.float64)
        total_variances = np.asarray(
            stream["projected_atm_total_variances"], dtype=np.float64
        )
    history_dates = np.asarray(panel["dates"]).astype("U10")
    if not np.array_equal(history_dates, dates):
        raise ValueError("source and fitted history dates do not align")
    kernel_dates, kernel_volatilities = _kernel_volatility_panel(
        dataset_path=dataset_path,
        target_business_dtes=business_dtes,
    )
    if not np.array_equal(kernel_dates, dates):
        raise ValueError("kernel and fitted history dates do not align")
    observations = pl.read_parquet(
        dataset_path / "observations.parquet",
        columns=["date", "forward_adjusted_spot_return_into_date"],
    )
    observation_dates = np.asarray(
        [value.isoformat() for value in observations["date"].to_list()]
    )
    if not np.array_equal(observation_dates, dates):
        raise ValueError("spot-return and fitted history dates do not align")
    spot_returns = np.asarray(
        observations["forward_adjusted_spot_return_into_date"].fill_null(0.0),
        dtype=np.float64,
    )
    with heston_pilot_path.open() as stream:
        heston_pilot = json.load(stream)
    heston_correlation = float(
        heston_pilot["dynamics_adjusted_background"]["correlation"]
    )

    origin_indices = np.asarray(panel["origin_indices"], dtype=np.int64)
    endpoint_indices = origin_indices + 1
    observations_count = len(origin_indices)
    train_count = int(math.floor(config.train_fraction * observations_count))
    if train_count <= config.factor_count + 2 or observations_count - train_count < 20:
        raise ValueError("insufficient history for chronological pilot")

    scale = config.background_annual_volatility / kernel_volatilities[-1, 0]
    heston_variance_path = (scale * kernel_volatilities[:, 0]) ** 2
    cir = fit_cir_quasi_likelihood(
        heston_variance_path[origin_indices[:train_count]],
        heston_variance_path[endpoint_indices[:train_count]],
        step_years=config.dt,
    )
    conditional_mean, conditional_variance = cir_conditional_moments(
        heston_variance_path[origin_indices],
        mean_reversion=cir.mean_reversion,
        long_run_variance=cir.long_run_variance,
        volatility_of_variance=cir.volatility_of_variance,
        step_years=config.dt,
    )
    variance_scores = (
        heston_variance_path[endpoint_indices] - conditional_mean
    ) / np.sqrt(conditional_variance)

    training_spot_mean = float(np.mean(spot_returns[endpoint_indices[:train_count]]))
    annualized_spot_returns = (
        spot_returns[endpoint_indices] - training_spot_mean
    ) / math.sqrt(config.dt)
    total_annual_variance = (
        total_variances[origin_indices, 0] / expiries[0]
    )
    residual_posterior = residual_spot_posterior(
        annualized_spot_return=annualized_spot_returns,
        total_annual_variance=total_annual_variance,
        heston_variance=heston_variance_path[origin_indices],
        heston_variance_score=variance_scores,
        heston_correlation=heston_correlation,
        residual_variance_floor=config.residual_spot_volatility_floor**2,
    )

    origins = np.asarray(panel["log_dlvsq_origins"], dtype=np.float64)
    endpoints = origins + np.asarray(panel["log_dlvsq_changes"], dtype=np.float64)
    innovations = _fit_ar_innovations(
        origins=origins,
        endpoints=endpoints,
        train_count=train_count,
        nms_nodes=np.asarray(panel["nms_nodes"], dtype=np.float64),
        config=config,
    )
    front_loading = latent_loading(
        innovations[:train_count],
        residual_posterior.mean[:train_count],
        residual_posterior.variance[:train_count],
    )
    residual_mean = innovations - residual_posterior.mean[:, None] * front_loading
    heston_loading = (
        variance_scores[:train_count] @ residual_mean[:train_count]
        / (variance_scores[:train_count] @ variance_scores[:train_count])
    )
    train_residual_energy = float(np.sum(residual_mean[:train_count] ** 2))
    heston_explained_fraction = float(
        np.sum(
            (
                variance_scores[:train_count, None]
                * heston_loading[None]
            )
            ** 2
        )
        / train_residual_energy
    )

    augmented_residuals = np.vstack(
        (
            residual_mean[:train_count],
            math.sqrt(float(np.sum(residual_posterior.variance[:train_count])))
            * front_loading[None],
        )
    )
    _, singular_values, right_vectors = np.linalg.svd(
        augmented_residuals,
        full_matrices=False,
    )
    residual_factor_count = config.factor_count - 1
    residual_basis = right_vectors[:residual_factor_count]
    residual_scores = residual_mean @ residual_basis.T
    train_factor_correlations = np.asarray(
        [
            np.corrcoef(
                variance_scores[:train_count], residual_scores[:train_count, factor]
            )[0, 1]
            for factor in range(residual_factor_count)
        ]
    )
    holdout_factor_correlations = np.asarray(
        [
            np.corrcoef(
                variance_scores[train_count:], residual_scores[train_count:, factor]
            )[0, 1]
            for factor in range(residual_factor_count)
        ]
    )
    explained_variance = singular_values**2 / np.sum(singular_values**2)

    annualized_volatilities = np.sqrt(total_variances / expiries[None])
    log_atm_changes = (
        np.log(annualized_volatilities[endpoint_indices])
        - np.log(annualized_volatilities[origin_indices])
    )
    atm_heston_loading, atm_residual_loading, atm_noise_variance = _fit_atm_loadings(
        log_atm_volatility_changes=log_atm_changes,
        variance_scores=variance_scores,
        residual_posterior=residual_posterior,
        train_count=train_count,
    )
    model_holdout_correlations = _model_atm_correlations(
        heston_loading=atm_heston_loading,
        residual_loading=atm_residual_loading,
        residual_noise_variance=atm_noise_variance,
        heston_variance=heston_variance_path[
            origin_indices[train_count:]
        ],
        residual_spot_variance=residual_posterior.residual_annual_variance[
            train_count:
        ],
        heston_correlation=heston_correlation,
        dt=config.dt,
    )
    empirical_holdout_correlations = np.asarray(
        [
            np.corrcoef(
                spot_returns[endpoint_indices[train_count:]],
                log_atm_changes[train_count:, expiry],
            )[0, 1]
            for expiry in range(len(business_dtes))
        ]
    )
    selected = np.isin(business_dtes, np.asarray([5.0, 20.0, 60.0, 120.0]))
    selected_rmse = float(
        np.sqrt(
            np.mean(
                (
                    model_holdout_correlations[selected]
                    - empirical_holdout_correlations[selected]
                )
                ** 2
            )
        )
    )

    train_conditional_score = residual_posterior.conditional_return_score[:train_count]
    holdout_conditional_score = residual_posterior.conditional_return_score[train_count:]
    report: dict[str, object] = {
        "schema": "gromm2-correlation-aware-pilot-v1",
        "history_path": str(history_path.resolve()),
        "source_history_path": str(source_history_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "heston_pilot_path": str(heston_pilot_path.resolve()),
        "config": asdict(config),
        "pair_count": observations_count,
        "train_count": train_count,
        "holdout_count": observations_count - train_count,
        "train_end_date": str(dates[endpoint_indices[train_count - 1]]),
        "holdout_start_date": str(dates[endpoint_indices[train_count]]),
        "heston": {
            "correlation": heston_correlation,
            "variance_proxy": (
                "scaled one-business-day released SANOS kernel variance"
            ),
            "physical_mean_reversion": cir.mean_reversion,
            "physical_long_run_variance": cir.long_run_variance,
            "physical_volatility_of_variance": cir.volatility_of_variance,
            "variance_score_train_mean": float(
                np.mean(variance_scores[:train_count])
            ),
            "variance_score_train_standard_deviation": float(
                np.std(variance_scores[:train_count], ddof=1)
            ),
            "variance_score_holdout_mean": float(
                np.mean(variance_scores[train_count:])
            ),
            "variance_score_holdout_standard_deviation": float(
                np.std(variance_scores[train_count:], ddof=1)
            ),
        },
        "conditional_return_filter": {
            "train_mean": float(np.mean(train_conditional_score)),
            "train_standard_deviation": float(
                np.std(train_conditional_score, ddof=1)
            ),
            "holdout_mean": float(np.mean(holdout_conditional_score)),
            "holdout_standard_deviation": float(
                np.std(holdout_conditional_score, ddof=1)
            ),
            "posterior_residual_score_second_moment_train": float(
                np.mean(
                    residual_posterior.mean[:train_count] ** 2
                    + residual_posterior.variance[:train_count]
                )
            ),
            "posterior_residual_score_second_moment_holdout": float(
                np.mean(
                    residual_posterior.mean[train_count:] ** 2
                    + residual_posterior.variance[train_count:]
                )
            ),
        },
        "dlv_orthogonality": {
            "heston_variance_explained_fraction_after_front_factor": (
                heston_explained_fraction
            ),
            "maximum_absolute_train_factor_correlation": float(
                np.max(np.abs(train_factor_correlations))
            ),
            "maximum_absolute_holdout_factor_correlation": float(
                np.max(np.abs(holdout_factor_correlations))
            ),
            "train_factor_correlations": train_factor_correlations.tolist(),
            "holdout_factor_correlations": holdout_factor_correlations.tolist(),
            "residual_pca_cumulative_explained_variance": np.cumsum(
                explained_variance[:residual_factor_count]
            ).tolist(),
        },
        "atm_correlation_holdout": {
            "selected_term_rmse": selected_rmse,
            "model": _selected_term_structure(
                business_dtes, model_holdout_correlations
            ),
            "empirical": _selected_term_structure(
                business_dtes, empirical_holdout_correlations
            ),
            "business_dtes": business_dtes.tolist(),
            "model_correlations": model_holdout_correlations.tolist(),
            "empirical_correlations": empirical_holdout_correlations.tolist(),
            "heston_atm_loadings": atm_heston_loading.tolist(),
            "residual_atm_loadings": atm_residual_loading.tolist(),
        },
        "interpretation": (
            "This stage tests latent-shock identification on the existing fixed-kernel "
            "DLV history. A non-negligible residual Heston loading is evidence that "
            "date-specific Heston kernels must be fitted before claiming subsystem "
            "orthogonality or strict product-model arbitrage freedom."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--source-history", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--heston-pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--factor-count", type=int, default=16)
    arguments = parser.parse_args()
    report = run_correlation_aware_pilot(
        history_path=arguments.history,
        source_history_path=arguments.source_history,
        dataset_path=arguments.dataset,
        heston_pilot_path=arguments.heston_pilot,
        output_path=arguments.output,
        config=CorrelationAwarePilotConfig(
            train_fraction=arguments.train_fraction,
            factor_count=arguments.factor_count,
        ),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
