"""Estimate observable and canonically identified GROMM2 background laws."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from quantdata.gromm2.background import (
    fit_cir_quasi_likelihood,
    kernel_annual_volatilities,
)


def _kernel_panel(
    *,
    dataset_path: Path,
    target_business_dtes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    target_expiries = np.asarray(target_business_dtes, dtype=np.float64) / 252.0
    dates: list[str] = []
    volatilities: list[np.ndarray] = []
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


def _moving_block_correlation_interval(
    first: np.ndarray,
    second: np.ndarray,
    *,
    origin_indices: np.ndarray,
    block_length: int,
    replications: int,
    seed: int,
) -> tuple[float, float]:
    origins = np.asarray(origin_indices, dtype=np.int64)
    if (
        first.shape != second.shape
        or first.ndim != 1
        or origins.shape != first.shape
    ):
        raise ValueError("bootstrap series must be aligned vectors")
    generator = np.random.default_rng(seed)
    observations = len(first)
    candidate_starts = np.arange(observations - block_length + 1)
    starts = candidate_starts[
        np.asarray(
            [
                np.all(np.diff(origins[start : start + block_length]) == 1)
                for start in candidate_starts
            ]
        )
    ]
    if len(starts) == 0:
        raise ValueError("no contiguous moving block is available")
    blocks = int(np.ceil(observations / block_length))
    correlations = np.empty(replications)
    for replication in range(replications):
        selected = generator.choice(starts, size=blocks, replace=True)
        indices = np.concatenate(
            [np.arange(start, start + block_length) for start in selected]
        )[:observations]
        correlations[replication] = np.corrcoef(first[indices], second[indices])[0, 1]
    return tuple(np.quantile(correlations, [0.025, 0.975]).tolist())


def _write_figure(
    *,
    output: Path,
    business_dtes: np.ndarray,
    kernel_volatilities: np.ndarray,
    spot_correlations: np.ndarray,
    explained_variance: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7.4, 6.1))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.0))
    axes = (
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    )
    quantiles = np.quantile(kernel_volatilities, [0.1, 0.5, 0.9], axis=0)
    axes[0].fill_between(
        business_dtes,
        100.0 * quantiles[0],
        100.0 * quantiles[2],
        color="#56b4e9",
        alpha=0.28,
        label="10--90%",
    )
    axes[0].plot(
        business_dtes,
        100.0 * quantiles[1],
        color="#0072b2",
        linewidth=2.0,
        label="Median",
    )
    axes[0].axhline(5.0, color="#d55e00", linestyle="--", linewidth=1.5, label="5% fixed")
    axes[0].set_xlabel("Business days to expiry")
    axes[0].set_ylabel("Annualized kernel volatility (%)")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        business_dtes,
        spot_correlations,
        color="#009e73",
        linewidth=2.0,
    )
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].set_xlabel("Business days to expiry")
    axes[1].set_ylabel(r"Corr$(R_S,\Delta\log\sigma_Y)$")
    axes[1].set_ylim(-1.0, 0.1)

    component_count = min(12, len(explained_variance))
    axes[2].plot(
        np.arange(1, component_count + 1),
        np.cumsum(explained_variance[:component_count]),
        color="#cc79a7",
        marker="o",
        linewidth=2.0,
    )
    axes[2].axhline(0.9, color="#d55e00", linestyle="--", linewidth=1.2)
    axes[2].set_xlabel("Kernel-volatility PCA factors")
    axes[2].set_ylabel("Cumulative explained variance")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_xticks(np.arange(1, component_count + 1))
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def estimate_background_processes(
    *,
    source_history_path: Path,
    dataset_path: Path,
    factor_estimate_path: Path,
    output_json: Path,
    output_figure: Path,
    fixed_background_annual_volatility: float = 0.05,
    bootstrap_replications: int = 20_000,
    seed: int = 20261014,
) -> dict[str, object]:
    """Estimate every background candidate supported by the released panel."""
    with h5py.File(source_history_path, "r") as stream:
        dates = np.asarray(stream["dates"]).astype("U10")
        business_dtes = np.asarray(stream["model_business_dtes"], dtype=np.float64)
    kernel_dates, kernel_volatilities = _kernel_panel(
        dataset_path=dataset_path,
        target_business_dtes=business_dtes,
    )
    if not np.array_equal(kernel_dates, dates):
        raise ValueError("SANOS kernel and GROMM2 history dates do not align")
    observations = pl.read_parquet(
        dataset_path / "observations.parquet",
        columns=["date", "forward_adjusted_spot_return_into_date"],
    )
    observation_dates = np.asarray(
        [value.isoformat() for value in observations["date"].to_list()]
    )
    if not np.array_equal(observation_dates, dates):
        raise ValueError("spot-return and GROMM2 history dates do not align")
    spot_returns = np.asarray(
        observations["forward_adjusted_spot_return_into_date"].fill_null(0.0),
        dtype=np.float64,
    )
    with np.load(factor_estimate_path, allow_pickle=False) as values:
        pair_origin_indices = np.asarray(values["pair_origin_indices"], dtype=np.int64)
        factor_scores = np.asarray(values["factor_scores"], dtype=np.float64)
    if factor_scores.shape[0] != len(pair_origin_indices):
        raise ValueError("factor scores and pair indices do not align")
    endpoints = pair_origin_indices + 1

    log_kernel_changes = (
        np.log(kernel_volatilities[endpoints])
        - np.log(kernel_volatilities[pair_origin_indices])
    )
    centered_changes = log_kernel_changes - np.mean(
        log_kernel_changes, axis=0, keepdims=True
    )
    _, singular_values, right_vectors = np.linalg.svd(
        centered_changes,
        full_matrices=False,
    )
    explained_variance = singular_values**2 / np.sum(singular_values**2)
    kernel_factor_scores = centered_changes @ right_vectors.T
    if np.corrcoef(kernel_factor_scores[:, 0], factor_scores[:, 0])[0, 1] > 0.0:
        kernel_factor_scores[:, 0] *= -1.0
        right_vectors[0] *= -1.0

    spot_correlations = np.asarray(
        [
            np.corrcoef(spot_returns[endpoints], log_kernel_changes[:, expiry])[0, 1]
            for expiry in range(len(business_dtes))
        ]
    )
    kernel_variance = kernel_volatilities[:, 0] ** 2
    cir = fit_cir_quasi_likelihood(
        kernel_variance[pair_origin_indices],
        kernel_variance[endpoints],
    )
    heston_observed_spot_correlation = float(
        np.corrcoef(cir.standardized_innovations, factor_scores[:, 0])[0, 1]
    )
    heston_correlation_interval = _moving_block_correlation_interval(
        cir.standardized_innovations,
        factor_scores[:, 0],
        origin_indices=pair_origin_indices,
        block_length=10,
        replications=bootstrap_replications,
        seed=seed,
    )

    _write_figure(
        output=output_figure,
        business_dtes=business_dtes,
        kernel_volatilities=kernel_volatilities,
        spot_correlations=spot_correlations,
        explained_variance=explained_variance,
    )

    report: dict[str, object] = {
        "schema": "gromm2-background-estimates-v1",
        "source_history_path": str(source_history_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "factor_estimate_path": str(factor_estimate_path.resolve()),
        "observation_count": len(dates),
        "one_day_pair_count": len(pair_origin_indices),
        "fixed_black": {
            "annual_volatility": fixed_background_annual_volatility,
            "status": "fixed modeling convention; not statistically estimated",
        },
        "floating_sanos_kernel": {
            "business_dtes": business_dtes.tolist(),
            "annual_volatility_mean": np.mean(kernel_volatilities, axis=0).tolist(),
            "annual_volatility_median": np.median(kernel_volatilities, axis=0).tolist(),
            "annual_volatility_q10": np.quantile(
                kernel_volatilities, 0.1, axis=0
            ).tolist(),
            "annual_volatility_q90": np.quantile(
                kernel_volatilities, 0.9, axis=0
            ).tolist(),
            "spot_return_log_volatility_change_correlation": spot_correlations.tolist(),
            "pca_explained_variance": explained_variance.tolist(),
            "pca_first_loading": right_vectors[0].tolist(),
        },
        "heston_proxy": {
            "variance_proxy": "released one-business-day SANOS Black-kernel annual variance",
            "measure": "P",
            "mean_reversion_per_year": cir.mean_reversion,
            "long_run_variance": cir.long_run_variance,
            "long_run_volatility": cir.long_run_volatility,
            "volatility_of_variance": cir.volatility_of_variance,
            "feller_ratio": cir.feller_ratio,
            "standardized_variance_innovation_product_spot_score_correlation": heston_observed_spot_correlation,
            "correlation_moving_block_95_interval": list(heston_correlation_interval),
            "identification": "variance diffusion and reduced-form correlation are estimable under P; the background-return shock and Q variance drift are not separately identified",
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-history", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--factor-estimate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--bootstrap-replications", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261014)
    arguments = parser.parse_args()
    if arguments.bootstrap_replications <= 0:
        raise ValueError("--bootstrap-replications must be positive")
    report = estimate_background_processes(
        source_history_path=arguments.source_history,
        dataset_path=arguments.dataset,
        factor_estimate_path=arguments.factor_estimate,
        output_json=arguments.output_json,
        output_figure=arguments.output_figure,
        bootstrap_replications=arguments.bootstrap_replications,
        seed=arguments.seed,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
