"""Audit rolling daily DYSANOS dynamics across historical origins."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from quantdata.dysanos.flex.front_spot_copula import _front_quantile_spline
from quantdata.dysanos.flex.presentation_hedging import (
    _decode_surface,
    load_presentation_model,
)
from quantdata.dysanos.v11.atm_spot_correlation import (
    _antithetic_correlation_standard_errors,
    _correlations,
    _decode_atm_volatilities,
    _direct_sanos_history,
)
from quantdata.dysanos.v11.daily_joint_ar_spot import (
    fit_daily_joint_ar_spot,
    sample_daily_joint_ar_spot,
)
from quantdata.dysanos.v11.daily_joint_dynamics import (
    _historical_spot_scores,
    fit_daily_joint_gaussian,
)


@dataclass(frozen=True)
class RollingOriginSimulation:
    """Antithetic simulated transitions aligned across rolling origins."""

    next_states: np.ndarray
    log_spot_returns: np.ndarray
    origin_rows: np.ndarray
    retained_variance_fractions: np.ndarray
    condition_numbers: np.ndarray
    spectral_radii: np.ndarray | None = None
    gamma_norms: np.ndarray | None = None
    spot_direction_fractions: np.ndarray | None = None

    def __post_init__(self) -> None:
        count = len(self.next_states)
        if (
            self.next_states.ndim != 2
            or count == 0
            or count % 2
            or self.log_spot_returns.shape != (count,)
            or self.origin_rows.shape != (count,)
            or np.any(~np.isfinite(self.next_states))
            or np.any(~np.isfinite(self.log_spot_returns))
            or np.any(self.origin_rows < 0)
            or self.retained_variance_fractions.ndim != 1
            or self.condition_numbers.shape != self.retained_variance_fractions.shape
            or (
                self.spectral_radii is not None
                and self.spectral_radii.shape != self.retained_variance_fractions.shape
            )
            or (
                self.gamma_norms is not None
                and self.gamma_norms.shape != self.retained_variance_fractions.shape
            )
            or (
                self.spot_direction_fractions is not None
                and self.spot_direction_fractions.shape != self.retained_variance_fractions.shape
            )
        ):
            raise ValueError("invalid rolling-origin simulation")


def rolling_origin_indices(
    observation_count: int,
    *,
    lookback: int,
    stride: int,
) -> np.ndarray:
    """Return state rows with a full trailing window and one next observation."""
    if observation_count <= 0 or lookback <= 0 or stride <= 0:
        raise ValueError("rolling-origin counts must be positive")
    if lookback >= observation_count:
        raise ValueError("lookback must leave at least one out-of-sample step")
    return np.arange(lookback, observation_count, stride, dtype=np.int64)


def _front_mixtures(
    model: object,
    states: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    supports: list[np.ndarray] = []
    densities: list[np.ndarray] = []
    sqrt_variances: list[np.ndarray] = []
    with torch.inference_mode():
        for first in range(0, len(states), batch_size):
            last = min(first + batch_size, len(states))
            batch = torch.as_tensor(
                states[first:last], dtype=model.fdtype, device=model.device
            )
            surface = _decode_surface(model, batch)
            supports.append(surface.support_strikes[:, 0].cpu().double().numpy())
            densities.append(surface.densities[:, 0].cpu().double().numpy())
            sqrt_variances.append(
                surface.backbone_total_vols[:, 0].cpu().double().numpy()
            )
    return (
        np.concatenate(supports),
        np.concatenate(densities),
        np.concatenate(sqrt_variances),
    )


def simulate_rolling_origins(
    model: object,
    *,
    states: np.ndarray,
    joint_observations: np.ndarray,
    spot_scores: np.ndarray,
    origins: np.ndarray,
    lookback: int,
    factor_rank: int,
    dynamics: str,
    paths_per_origin: int,
    seed: int,
    batch_size: int,
) -> RollingOriginSimulation:
    """Fit each trailing window and simulate one next session per origin."""
    if paths_per_origin <= 0 or paths_per_origin % 2:
        raise ValueError("paths_per_origin must be positive and even")
    if dynamics not in {"increment", "ar-spot"}:
        raise ValueError("dynamics must be increment or ar-spot")
    if np.any(origins < lookback) or np.any(origins >= len(joint_observations)):
        raise ValueError("rolling origins are not compatible with the observations")
    supports, densities, sqrt_variances = _front_mixtures(
        model, states[origins], batch_size=batch_size
    )
    generator = np.random.default_rng(seed)
    half = paths_per_origin // 2
    positive_states: list[np.ndarray] = []
    negative_states: list[np.ndarray] = []
    positive_spot: list[np.ndarray] = []
    negative_spot: list[np.ndarray] = []
    retained = np.empty(len(origins), dtype=np.float64)
    condition = np.empty(len(origins), dtype=np.float64)
    spectral_radius = np.full(len(origins), np.nan, dtype=np.float64)
    gamma_norm = np.full(len(origins), np.nan, dtype=np.float64)
    spot_fraction = np.full(len(origins), np.nan, dtype=np.float64)
    started = perf_counter()
    last_report = started
    for local_index, origin in enumerate(origins):
        if dynamics == "increment":
            fit = fit_daily_joint_gaussian(
                joint_observations[origin - lookback : origin],
                factor_rank=factor_rank,
            )
            normals = generator.standard_normal((half, len(fit.mean)))
            positive = fit.mean[None, :] + normals @ fit.factor.T
            negative = fit.mean[None, :] - normals @ fit.factor.T
            positive_states.append(states[origin][None, :] + positive[:, :-1])
            negative_states.append(states[origin][None, :] + negative[:, :-1])
            positive_scores = positive[:, -1]
            negative_scores = negative[:, -1]
            retained[local_index] = fit.retained_variance_fraction
            condition[local_index] = fit.condition_number
        else:
            fit = fit_daily_joint_ar_spot(
                states[origin - lookback : origin + 1],
                spot_scores[origin - lookback : origin],
                rank=factor_rank,
            )
            next_states, next_scores = sample_daily_joint_ar_spot(
                fit, states[origin], count=paths_per_origin, seed=int(generator.integers(2**32))
            )
            positive_states.append(next_states[0, :half])
            negative_states.append(next_states[0, half:])
            positive_scores = next_scores[0, :half]
            negative_scores = next_scores[0, half:]
            spectral_radius[local_index] = fit.spectral_radius
            gamma_norm[local_index] = np.linalg.norm(fit.gamma)
            spot_fraction[local_index] = fit.spot_direction_fraction
        positive_spot.append(
            np.log(
                _front_quantile_spline(
                    support=supports[local_index],
                    density=densities[local_index],
                    sqrt_variance=float(sqrt_variances[local_index]),
                    gaussian_score=positive_scores,
                )
            )
        )
        negative_spot.append(
            np.log(
                _front_quantile_spline(
                    support=supports[local_index],
                    density=densities[local_index],
                    sqrt_variance=float(sqrt_variances[local_index]),
                    gaussian_score=negative_scores,
                )
            )
        )
        now = perf_counter()
        if (
            local_index + 1 == len(origins)
            or now - last_report >= 300.0
            or (local_index + 1) % 25 == 0
        ):
            completed = local_index + 1
            elapsed = now - started
            eta = elapsed * (len(origins) - completed) / completed
            print(
                f"rolling_fit={completed}/{len(origins)} elapsed_s={elapsed:.1f} "
                f"eta_s={eta:.1f}",
                flush=True,
            )
            last_report = now
    origin_half = np.repeat(origins, half)
    return RollingOriginSimulation(
        next_states=np.concatenate(
            (np.concatenate(positive_states), np.concatenate(negative_states))
        ),
        log_spot_returns=np.concatenate(
            (np.concatenate(positive_spot), np.concatenate(negative_spot))
        ),
        origin_rows=np.concatenate((origin_half, origin_half)),
        retained_variance_fractions=retained,
        condition_numbers=condition,
        spectral_radii=spectral_radius,
        gamma_norms=gamma_norm,
        spot_direction_fractions=spot_fraction,
    )


def _write_figure(
    output: Path,
    *,
    business_dtes: np.ndarray,
    direct: np.ndarray,
    affine: np.ndarray,
    simulated: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    axis.plot(business_dtes, direct, color="#202020", label="Direct SANOS OOS")
    axis.plot(business_dtes, affine, color="#d55e00", label="Affine history OOS")
    axis.plot(
        business_dtes,
        simulated,
        color="#0072b2",
        linestyle="--",
        label="Rolling daily DYSANOS",
    )
    axis.axhline(0.0, color="#808080", linewidth=0.8)
    axis.set_xlabel("Business days to expiry")
    axis.set_ylabel(r"Corr$(\Delta\log S,\,\Delta\log\sigma_{\rm ATM})$")
    axis.set_ylim(-1.0, 0.05)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def run_experiment(
    *,
    checkpoint: Path,
    target_history: Path,
    historical_sanos_history: Path,
    historical_dataset: Path,
    output_json: Path,
    output_figure: Path,
    lookback: int,
    stride: int,
    factor_rank: int,
    dynamics: str,
    paths_per_origin: int,
    batch_size: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    """Run a rolling, no-future-data fit of the daily joint innovation law."""
    started = perf_counter()
    _, model = load_presentation_model(checkpoint, device=device, precision=32)
    history = torch.load(target_history, map_location="cpu", weights_only=False)
    dates = history.get("dates")
    states_value = history.get("states")
    spots_value = history.get("spot_refs")
    if (
        not isinstance(dates, list)
        or not isinstance(states_value, torch.Tensor)
        or not isinstance(spots_value, torch.Tensor)
    ):
        raise ValueError("invalid Direct-200 target history")
    states = states_value.double().numpy()
    spots = spots_value.double().numpy()
    state_increments = np.diff(states, axis=0)
    gross_returns = spots[1:] / spots[:-1]
    spot_scores = _historical_spot_scores(
        model, states[:-1], gross_returns, batch_size=batch_size
    )
    joint_observations = np.column_stack((state_increments, spot_scores))
    origins = rolling_origin_indices(
        len(joint_observations), lookback=lookback, stride=stride
    )
    simulation = simulate_rolling_origins(
        model,
        states=states,
        joint_observations=joint_observations,
        spot_scores=spot_scores,
        origins=origins,
        lookback=lookback,
        factor_rank=factor_rank,
        dynamics=dynamics,
        paths_per_origin=paths_per_origin,
        seed=seed,
        batch_size=batch_size,
    )

    reconstructed_atm = _decode_atm_volatilities(
        model, states, batch_size=batch_size
    )
    next_atm = _decode_atm_volatilities(
        model, simulation.next_states, batch_size=batch_size
    )
    simulated_changes = np.log(
        next_atm / reconstructed_atm[simulation.origin_rows]
    )
    simulated_correlations = _correlations(
        simulation.log_spot_returns, simulated_changes
    )
    simulated_errors = _antithetic_correlation_standard_errors(
        simulation.log_spot_returns, simulated_changes
    )

    direct_dates, direct_dtes, direct_atm, direct_spot_returns = (
        _direct_sanos_history(
            history_path=historical_sanos_history,
            dataset_path=historical_dataset,
        )
    )
    if not np.array_equal(direct_dates, np.asarray(dates, dtype="U10")):
        raise ValueError("direct SANOS and DYSANOS histories have different dates")
    business_dtes = (
        np.asarray(model.config.geometry.model_expiries_y, dtype=np.float64) * 252.0
    )
    direct_indices = np.searchsorted(direct_dtes, business_dtes)
    if np.any(direct_indices == len(direct_dtes)) or np.any(
        direct_dtes[direct_indices] != business_dtes
    ):
        raise ValueError("direct SANOS history lacks a DYSANOS audit expiry")
    direct_changes = np.diff(np.log(direct_atm[:, direct_indices]), axis=0)
    affine_changes = np.diff(np.log(reconstructed_atm), axis=0)
    direct_correlations = _correlations(
        direct_spot_returns[origins], direct_changes[origins]
    )
    affine_correlations = _correlations(
        direct_spot_returns[origins], affine_changes[origins]
    )

    report: dict[str, object] = {
        "schema": "dysanos-v11-rolling-daily-joint-dynamics-v2",
        "status": "complete",
        "scope": "rolling dynamics fit with the existing all-history affine decoder",
        "strict_origin_timing": True,
        "decoder_is_rolling_refit": False,
        "method": (
            "at every origin fit the daily joint state-increment/front-spot-score law "
            "using only the preceding lookback sessions"
            if dynamics == "increment"
            else "at every origin fit a stable low-rank daily state AR law jointly with "
            "the current-front-law spot score using only the preceding lookback sessions"
        ),
        "checkpoint": str(checkpoint.resolve()),
        "target_history": str(target_history.resolve()),
        "historical_sanos_history": str(historical_sanos_history.resolve()),
        "historical_dataset": str(historical_dataset.resolve()),
        "lookback": lookback,
        "stride": stride,
        "factor_rank": factor_rank,
        "dynamics": dynamics,
        "origin_count": int(len(origins)),
        "first_origin_date": dates[int(origins[0])],
        "last_origin_date": dates[int(origins[-1])],
        "first_oos_date": dates[int(origins[0]) + 1],
        "last_oos_date": dates[int(origins[-1]) + 1],
        "paths_per_origin": paths_per_origin,
        "simulation_path_count": int(len(simulation.next_states)),
        "simulation_independent_units": int(len(simulation.next_states) // 2),
        "business_dtes": business_dtes.tolist(),
        "direct_sanos_oos_correlations": direct_correlations.tolist(),
        "affine_history_oos_correlations": affine_correlations.tolist(),
        "simulated_correlations": simulated_correlations.tolist(),
        "simulated_correlation_standard_errors": simulated_errors.tolist(),
        "retained_variance_fraction": (
            {
                "minimum": float(np.min(simulation.retained_variance_fractions)),
                "median": float(np.median(simulation.retained_variance_fractions)),
                "maximum": float(np.max(simulation.retained_variance_fractions)),
            }
            if dynamics == "increment"
            else None
        ),
        "condition_number": (
            {
                "minimum": float(np.min(simulation.condition_numbers)),
                "median": float(np.median(simulation.condition_numbers)),
                "maximum": float(np.max(simulation.condition_numbers)),
            }
            if dynamics == "increment"
            else None
        ),
        "spectral_radius": (
            None
            if dynamics == "increment"
            else {
                "minimum": float(np.nanmin(simulation.spectral_radii)),
                "median": float(np.nanmedian(simulation.spectral_radii)),
                "maximum": float(np.nanmax(simulation.spectral_radii)),
            }
        ),
        "gamma_norm": (
            None
            if dynamics == "increment"
            else {
                "minimum": float(np.nanmin(simulation.gamma_norms)),
                "median": float(np.nanmedian(simulation.gamma_norms)),
                "maximum": float(np.nanmax(simulation.gamma_norms)),
            }
        ),
        "spot_direction_fraction": (
            None
            if dynamics == "increment"
            else {
                "minimum": float(np.nanmin(simulation.spot_direction_fractions)),
                "median": float(np.nanmedian(simulation.spot_direction_fractions)),
                "maximum": float(np.nanmax(simulation.spot_direction_fractions)),
            }
        ),
        "elapsed_seconds": perf_counter() - started,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    _write_figure(
        output_figure,
        business_dtes=business_dtes,
        direct=direct_correlations,
        affine=affine_correlations,
        simulated=simulated_correlations,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--historical-sanos-history", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--factor-rank", type=int, default=201)
    parser.add_argument("--dynamics", choices=("increment", "ar-spot"), default="increment")
    parser.add_argument("--paths-per-origin", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    report = run_experiment(
        checkpoint=arguments.checkpoint,
        target_history=arguments.target_history,
        historical_sanos_history=arguments.historical_sanos_history,
        historical_dataset=arguments.historical_dataset,
        output_json=arguments.output_json,
        output_figure=arguments.output_figure,
        lookback=arguments.lookback,
        stride=arguments.stride,
        factor_rank=arguments.factor_rank,
        dynamics=arguments.dynamics,
        paths_per_origin=arguments.paths_per_origin,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        device=arguments.device,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
