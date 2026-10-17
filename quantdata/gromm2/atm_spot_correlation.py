"""Measure spot/rolling-ATM-volatility correlation across GROMM2 expiries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.special import ndtr, ndtri

from quantdata.gromm2.dlv import decode_density_tower
from quantdata.gromm2.dynamics import (
    load_estimated_dynamics,
    selective_factor_scales,
)
from quantdata.gromm2.pricing import price_active_calls
from quantdata.gromm2.simulation import SimulationState, one_day_risk_neutral_step

_AUDIT_BUSINESS_DTES = np.asarray([5.0, 10.0, 20.0, 60.0, 120.0])
_AUDIT_NMS = np.asarray([-1.0, 0.0, 1.0])


def _historical_correlations(
    *,
    history_path: Path,
    dataset_path: Path,
    recent_start_date: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    with h5py.File(history_path, "r") as stream:
        dates = np.asarray(stream["dates"]).astype("U10")
        business_dtes = np.asarray(stream["model_business_dtes"], dtype=np.float64)
        total_variances = np.asarray(
            stream["projected_atm_total_variances"], dtype=np.float64
        )
    observations = pl.read_parquet(dataset_path / "observations.parquet")
    spot_returns = np.asarray(
        observations["forward_adjusted_spot_return_into_date"].to_list(),
        dtype=np.float64,
    )
    if len(spot_returns) != len(dates):
        raise ValueError("history and observation dates do not align")
    annualized_volatilities = np.sqrt(
        total_variances / (business_dtes[None, :] / 252.0)
    )
    volatility_returns = np.diff(np.log(annualized_volatilities), axis=0)
    aligned_spot_returns = spot_returns[1:]
    recent = dates[1:] >= recent_start_date
    full_correlations = np.asarray(
        [
            np.corrcoef(aligned_spot_returns, volatility_returns[:, expiry])[0, 1]
            for expiry in range(len(business_dtes))
        ]
    )
    recent_correlations = np.asarray(
        [
            np.corrcoef(
                aligned_spot_returns[recent], volatility_returns[recent, expiry]
            )[0, 1]
            for expiry in range(len(business_dtes))
        ]
    )
    return (
        business_dtes,
        full_correlations,
        recent_correlations,
        len(aligned_spot_returns),
        int(np.sum(recent)),
    )


def _antithetic_normals(
    *,
    generator: np.random.Generator,
    substeps: int,
    half_batch_size: int,
    factor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    factor_half = generator.standard_normal(
        (substeps, half_batch_size, factor_count)
    )
    background_half = generator.standard_normal(half_batch_size)
    return (
        np.concatenate((factor_half, -factor_half), axis=1),
        np.concatenate((background_half, -background_half)),
    )


def _available_atm_volatilities(
    *,
    supports: np.ndarray,
    densities: np.ndarray,
    background_levels: np.ndarray,
    spots: np.ndarray,
    remaining_business_dtes: np.ndarray,
    background_annual_volatility: float,
) -> np.ndarray:
    remaining_years = remaining_business_dtes / 252.0
    latent = background_levels[:, None, None] * supports[None, :, :]
    strikes = spots[:, None, None]
    standard_deviations = (
        background_annual_volatility * np.sqrt(remaining_years)[None, :, None]
    )
    log_ratio = np.log(latent / strikes)
    d1 = log_ratio / standard_deviations + 0.5 * standard_deviations
    d2 = d1 - standard_deviations
    kernels = latent * ndtr(d1) - strikes * ndtr(d2)
    calls_over_forward = np.sum(kernels * densities, axis=-1) / spots[:, None]
    probabilities = np.clip(
        0.5 * (calls_over_forward + 1.0),
        0.5000000001,
        0.9999999999,
    )
    return 2.0 * ndtri(probabilities) / np.sqrt(remaining_years)[None, :]


def _interpolate_total_variances(
    *,
    available_business_dtes: np.ndarray,
    available_volatilities: np.ndarray,
    target_business_dtes: np.ndarray,
) -> np.ndarray:
    upper = np.searchsorted(available_business_dtes, target_business_dtes)
    exact = available_business_dtes[upper] == target_business_dtes
    lower = np.where(exact, upper, upper - 1)
    upper = np.where(exact, upper, upper)
    denominator = available_business_dtes[upper] - available_business_dtes[lower]
    weight = np.divide(
        target_business_dtes - available_business_dtes[lower],
        denominator,
        out=np.zeros_like(target_business_dtes, dtype=np.float64),
        where=denominator > 0.0,
    )
    available_total_variances = (
        available_volatilities**2
        * (available_business_dtes[None, :] / 252.0)
    )
    target_total_variances = (
        (1.0 - weight[None, :]) * available_total_variances[:, lower]
        + weight[None, :] * available_total_variances[:, upper]
    )
    return np.sqrt(
        target_total_variances / (target_business_dtes[None, :] / 252.0)
    )


def _simulation_correlations(
    *,
    history_path: Path,
    estimate_path: Path,
    path_count: int,
    batch_size: int,
    substeps: int,
    seed: int,
    initial_projected_total_variances: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[str, float]]]:
    if path_count <= 0 or path_count % batch_size or batch_size % 2:
        raise ValueError("path_count must be divisible by an even batch_size")
    if not jax.config.x64_enabled:
        raise RuntimeError("set JAX_ENABLE_X64=1 for this diagnostic")
    with h5py.File(history_path, "r") as stream:
        supports = np.asarray(stream["supports"], dtype=np.float64)
        transports = np.asarray(stream["transports"], dtype=np.float64)
        expiries = np.asarray(stream["model_expiries_y"], dtype=np.float64)
        business_dtes = np.asarray(stream["model_business_dtes"], dtype=np.float64)
        initial_dlvsq = np.asarray(stream["dlvsq"][-1], dtype=np.float64)
    target_business_dtes = business_dtes[business_dtes <= business_dtes[-1] - 1.0]
    available_business_dtes = business_dtes[1:] - 1.0
    intervals = np.diff(expiries, prepend=0.0)
    dynamics = load_estimated_dynamics(estimate_path)
    initial_tower = np.asarray(
        decode_density_tower(
            jnp.asarray(supports),
            jnp.asarray(transports),
            jnp.asarray(initial_dlvsq),
            jnp.asarray(intervals),
        )
    )
    base_loadings = jnp.asarray(dynamics.log_dlvsq_loadings)
    _, selective_scales = selective_factor_scales(
        dynamics=dynamics,
        hidden=jnp.asarray(dynamics.ssm_terminal_hidden),
        floating_log_dlvsq=jnp.log(jnp.asarray(initial_dlvsq[1:])),
        previous_factor_score=jnp.zeros(dynamics.factor_count),
    )
    loadings_by_model = {
        "constant": base_loadings,
        "selective": base_loadings * selective_scales[None, None, :],
    }
    audit_indices = np.searchsorted(business_dtes, _AUDIT_BUSINESS_DTES)
    if np.any(audit_indices == len(business_dtes)) or np.any(
        business_dtes[audit_indices] != _AUDIT_BUSINESS_DTES
    ):
        raise ValueError("history does not contain every background audit expiry")
    audit_strikes = np.exp(
        np.sqrt(initial_projected_total_variances[audit_indices])[:, None]
        * _AUDIT_NMS[None, :]
    )
    initial_audit_values = np.asarray(
        price_active_calls(
            jnp.asarray(supports[audit_indices]),
            jnp.asarray(initial_tower[audit_indices])[None],
            jnp.ones(1),
            jnp.asarray(_AUDIT_BUSINESS_DTES / 252.0),
            jnp.asarray(audit_strikes),
        )
    )[0]
    daily_step = jax.jit(one_day_risk_neutral_step, static_argnames=("substeps",))
    generator = np.random.default_rng(seed)
    spot_samples = {name: [] for name in loadings_by_model}
    volatility_samples = {name: [] for name in loadings_by_model}
    audit_pair_gains = {name: [] for name in loadings_by_model}
    started = perf_counter()
    for batch_index in range(path_count // batch_size):
        factor_normals, independent_background_normals = _antithetic_normals(
            generator=generator,
            substeps=substeps,
            half_batch_size=batch_size // 2,
            factor_count=dynamics.factor_count,
        )
        state = SimulationState(
            front_density=jnp.broadcast_to(
                jnp.asarray(initial_tower[0]), (batch_size, supports.shape[1])
            ),
            dlvsq=jnp.broadcast_to(
                jnp.asarray(initial_dlvsq),
                (batch_size,) + initial_dlvsq.shape,
            ),
            background_level=jnp.ones(batch_size),
        )
        for name, loadings in loadings_by_model.items():
            print(
                f"batch={batch_index + 1}/{path_count // batch_size} "
                f"model={name} elapsed_s={perf_counter() - started:.1f}",
                flush=True,
            )
            result = daily_step(
                jnp.asarray(supports),
                jnp.asarray(transports),
                jnp.asarray(intervals[1:]),
                state,
                loadings,
                jnp.asarray(factor_normals),
                jnp.asarray(independent_background_normals),
                remaining_front_days=jnp.asarray(1),
                substeps=substeps,
            )
            spots = np.asarray(result.spot)
            available_volatilities = _available_atm_volatilities(
                supports=supports[1:],
                densities=np.asarray(result.density_tower[:, 1:]),
                background_levels=np.asarray(result.state.background_level),
                spots=spots,
                remaining_business_dtes=available_business_dtes,
                background_annual_volatility=0.05,
            )
            target_volatilities = _interpolate_total_variances(
                available_business_dtes=available_business_dtes,
                available_volatilities=available_volatilities,
                target_business_dtes=target_business_dtes,
            )
            spot_samples[name].append(spots - 1.0)
            volatility_samples[name].append(np.log(target_volatilities))
            next_audit_values = np.asarray(
                price_active_calls(
                    jnp.asarray(supports[audit_indices]),
                    result.density_tower[..., audit_indices, :],
                    result.state.background_level,
                    jnp.asarray((_AUDIT_BUSINESS_DTES - 1.0) / 252.0),
                    jnp.asarray(audit_strikes),
                )
            )
            gains = next_audit_values - initial_audit_values[None]
            half = batch_size // 2
            audit_pair_gains[name].append(
                0.5 * (gains[:half] + gains[half:])
            )
    correlations: dict[str, np.ndarray] = {}
    audits: dict[str, dict[str, float]] = {}
    for name in loadings_by_model:
        spots = np.concatenate(spot_samples[name])
        log_volatilities = np.concatenate(volatility_samples[name])
        correlations[name] = np.asarray(
            [
                np.corrcoef(spots, log_volatilities[:, expiry])[0, 1]
                for expiry in range(len(target_business_dtes))
            ]
        )
        pair_gains = np.concatenate(audit_pair_gains[name])
        means = np.mean(pair_gains, axis=0)
        standard_errors = np.std(pair_gains, axis=0, ddof=1) / np.sqrt(
            len(pair_gains)
        )
        z_scores = np.divide(
            means,
            standard_errors,
            out=np.zeros_like(means),
            where=standard_errors > 0.0,
        )
        audits[name] = {
            "maximum_absolute_mean_call_gain_over_spot": float(
                np.max(np.abs(means))
            ),
            "root_mean_square_call_gain_over_spot": float(np.sqrt(np.mean(means**2))),
            "maximum_absolute_z": float(np.max(np.abs(z_scores))),
            "coordinates_above_2_576": int(np.count_nonzero(np.abs(z_scores) > 2.576)),
            "coordinate_count": int(z_scores.size),
        }
    return target_business_dtes, correlations, audits


def _write_figure(
    *,
    output: Path,
    business_dtes: np.ndarray,
    historical_full: np.ndarray,
    historical_recent: np.ndarray,
    simulation_business_dtes: np.ndarray,
    simulated: dict[str, np.ndarray],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(
        business_dtes,
        historical_full,
        color="#202020",
        linewidth=2.0,
        label="Historical, full sample",
    )
    axis.plot(
        business_dtes,
        historical_recent,
        color="#d55e00",
        linewidth=2.0,
        label="Historical, recent period",
    )
    axis.plot(
        simulation_business_dtes,
        simulated["constant"],
        color="#0072b2",
        linewidth=2.0,
        linestyle="--",
        label=r"Constant $\mathbb{Q}$ model",
    )
    axis.plot(
        simulation_business_dtes,
        simulated["selective"],
        color="#009e73",
        linewidth=2.0,
        linestyle="-.",
        label=r"Selective SSM $\mathbb{Q}$ model",
    )
    axis.axhline(0.0, color="#808080", linewidth=0.8)
    axis.set_xlabel("Business days to expiry")
    axis.set_ylabel(r"Corr$(\Delta S/S,\,\Delta\log\sigma_{\rm ATM})$")
    axis.set_xlim(float(business_dtes[0]), float(business_dtes[-1]))
    axis.set_ylim(-1.0, 0.05)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=9, ncol=2)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def run_diagnostic(
    *,
    history_path: Path,
    source_history_path: Path,
    dataset_path: Path,
    estimate_path: Path,
    output_json: Path,
    output_figure: Path,
    recent_start_date: str,
    path_count: int,
    batch_size: int,
    substeps: int,
    seed: int,
) -> dict[str, object]:
    (
        business_dtes,
        historical_full,
        historical_recent,
        historical_count,
        recent_count,
    ) = _historical_correlations(
        history_path=source_history_path,
        dataset_path=dataset_path,
        recent_start_date=recent_start_date,
    )
    with h5py.File(source_history_path, "r") as stream:
        initial_projected_total_variances = np.asarray(
            stream["projected_atm_total_variances"][-1], dtype=np.float64
        )
    simulation_business_dtes, simulated, audits = _simulation_correlations(
        history_path=history_path,
        estimate_path=estimate_path,
        path_count=path_count,
        batch_size=batch_size,
        substeps=substeps,
        seed=seed,
        initial_projected_total_variances=initial_projected_total_variances,
    )
    _write_figure(
        output=output_figure,
        business_dtes=business_dtes,
        historical_full=historical_full,
        historical_recent=historical_recent,
        simulation_business_dtes=simulation_business_dtes,
        simulated=simulated,
    )
    report: dict[str, object] = {
        "schema": "gromm2-atm-vol-spot-correlation-v2",
        "definition": "correlation of daily forward-adjusted spot return with the daily log change in rolling-tenor ATM Black implied volatility",
        "history_path": str(history_path.resolve()),
        "source_history_path": str(source_history_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "estimate_path": str(estimate_path.resolve()),
        "historical_observation_count": historical_count,
        "recent_start_date": recent_start_date,
        "recent_observation_count": recent_count,
        "simulation_initial_date": "2025-08-29",
        "simulation_measure": "Q",
        "simulation_path_count": path_count,
        "simulation_substeps_per_day": substeps,
        "simulation_seed": seed,
        "background_annual_volatility": 0.05,
        "historical_business_dtes": business_dtes.tolist(),
        "historical_full_correlations": historical_full.tolist(),
        "historical_recent_correlations": historical_recent.tolist(),
        "simulation_business_dtes": simulation_business_dtes.tolist(),
        "constant_correlations": simulated["constant"].tolist(),
        "selective_correlations": simulated["selective"].tolist(),
        "one_day_call_martingale_audit": audits,
        "one_day_call_martingale_audit_business_dtes": _AUDIT_BUSINESS_DTES.tolist(),
        "one_day_call_martingale_audit_nms": _AUDIT_NMS.tolist(),
        "one_day_call_martingale_audit_sampling_unit": "antithetic pair",
        "terminal_expiry_note": "The simulated 125D point is omitted because a one-day rolling 125D statistic requires a newly born expiry beyond the terminal model date.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--source-history", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--recent-start-date", default="2024-08-13")
    parser.add_argument("--path-count", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261014)
    arguments = parser.parse_args()
    report = run_diagnostic(
        history_path=arguments.history,
        source_history_path=arguments.source_history,
        dataset_path=arguments.dataset,
        estimate_path=arguments.estimate,
        output_json=arguments.output_json,
        output_figure=arguments.output_figure,
        recent_start_date=arguments.recent_start_date,
        path_count=arguments.path_count,
        batch_size=arguments.batch_size,
        substeps=arguments.substeps,
        seed=arguments.seed,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
