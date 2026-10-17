"""Fit a daily joint Gaussian law for DYSANOS state and spot innovations."""

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
from quantdata.dysanos.flex.weekly_simulation import front_week_sanos_gaussian_score
from quantdata.dysanos.v11.atm_spot_correlation import (
    _antithetic_correlation_standard_errors,
    _correlations,
    _decode_atm_volatilities,
    _direct_sanos_history,
)


@dataclass(frozen=True)
class DailyJointGaussianFit:
    """Positive-definite approximation to the daily joint innovation law."""

    mean: np.ndarray
    factor: np.ndarray
    requested_rank: int
    retained_variance_fraction: float
    minimum_eigenvalue: float
    condition_number: float

    def __post_init__(self) -> None:
        dimension = len(self.mean)
        if (
            self.mean.shape != (dimension,)
            or self.factor.shape != (dimension, dimension)
            or np.any(~np.isfinite(self.mean))
            or np.any(~np.isfinite(self.factor))
            or self.requested_rank <= 0
            or not 0.0 < self.retained_variance_fraction <= 1.0
            or self.minimum_eigenvalue <= 0.0
            or self.condition_number < 1.0
        ):
            raise ValueError("invalid daily joint Gaussian fit")

    @property
    def covariance(self) -> np.ndarray:
        return self.factor @ self.factor.T


def fit_daily_joint_gaussian(
    observations: np.ndarray,
    *,
    factor_rank: int,
    relative_eigenvalue_floor: float = 1.0e-8,
) -> DailyJointGaussianFit:
    """Fit a spot-conditional low-rank-plus-diagonal daily covariance.

    The final coordinate is the front-marginal Gaussian spot score. Keeping it
    inside the covariance fit preserves the complete empirical spot/state
    cross-covariance. Low-rank compression is applied only to the residual
    state innovation after regressing it on the spot score.
    """
    values = np.asarray(observations, dtype=np.float64)
    if (
        values.ndim != 2
        or len(values) <= values.shape[1] + 5
        or np.any(~np.isfinite(values))
        or factor_rank <= 0
        or factor_rank > values.shape[1]
        or not 0.0 < relative_eigenvalue_floor < 1.0
    ):
        raise ValueError("invalid daily joint observations or covariance rank")
    mean = np.mean(values, axis=0)
    centered = values - mean[None, :]
    state = centered[:, :-1]
    spot = centered[:, -1]
    spot_sum_squares = float(spot @ spot)
    if spot_sum_squares <= np.finfo(np.float64).tiny:
        raise ValueError("daily spot score has zero variance")
    spot_variance = spot_sum_squares / (len(values) - 1)
    beta = (spot @ state) / spot_sum_squares
    residual = state - spot[:, None] * beta[None, :]
    residual_covariance = residual.T @ residual / (len(values) - 1)
    residual_covariance = 0.5 * (
        residual_covariance + residual_covariance.T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(residual_covariance)
    order = np.argsort(eigenvalues)[::-1]
    descending = np.maximum(eigenvalues[order], 0.0)
    vectors = eigenvectors[:, order]
    retained = min(factor_rank, state.shape[1])
    low_rank = (
        vectors[:, :retained] * descending[:retained][None, :]
    ) @ vectors[:, :retained].T
    if retained < state.shape[1]:
        residual_diagonal = np.maximum(
            np.diag(residual_covariance - low_rank), 0.0
        )
        residual_approximation = low_rank + np.diag(residual_diagonal)
    else:
        residual_approximation = residual_covariance
    state_spot_covariance = beta * spot_variance
    state_covariance = residual_approximation + (
        np.outer(beta, beta) * spot_variance
    )
    approximation = np.empty(
        (values.shape[1], values.shape[1]), dtype=np.float64
    )
    approximation[:-1, :-1] = state_covariance
    approximation[:-1, -1] = state_spot_covariance
    approximation[-1, :-1] = state_spot_covariance
    approximation[-1, -1] = spot_variance
    approximation = 0.5 * (approximation + approximation.T)
    approximate_values, approximate_vectors = np.linalg.eigh(approximation)
    scale = max(float(np.max(approximate_values)), np.finfo(np.float64).tiny)
    floor = relative_eigenvalue_floor * scale
    clipped = np.maximum(approximate_values, floor)
    factor = approximate_vectors * np.sqrt(clipped)[None, :]
    total_variance = float(np.sum(descending))
    retained_fraction = float(np.sum(descending[:retained]) / total_variance)
    return DailyJointGaussianFit(
        mean=mean,
        factor=factor,
        requested_rank=factor_rank,
        retained_variance_fraction=retained_fraction,
        minimum_eigenvalue=float(np.min(clipped)),
        condition_number=float(np.max(clipped) / np.min(clipped)),
    )


def sample_antithetic_joint_innovations(
    fit: DailyJointGaussianFit,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Sample innovations while retaining antithetic pairs as audit units."""
    if count <= 0 or count % 2:
        raise ValueError("count must be positive and even")
    generator = np.random.default_rng(seed)
    half = generator.standard_normal((count // 2, len(fit.mean)))
    normals = np.concatenate((half, -half), axis=0)
    return fit.mean[None, :] + normals @ fit.factor.T


def _historical_spot_scores(
    model: object,
    states: np.ndarray,
    gross_returns: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(len(gross_returns), dtype=np.float64)
    with torch.inference_mode():
        for first in range(0, len(gross_returns), batch_size):
            last = min(first + batch_size, len(gross_returns))
            state_batch = torch.as_tensor(
                states[first:last], dtype=model.fdtype, device=model.device
            )
            gross_batch = torch.as_tensor(
                gross_returns[first:last], dtype=model.fdtype, device=model.device
            )
            scores[first:last] = (
                front_week_sanos_gaussian_score(
                    _decode_surface(model, state_batch), gross_batch
                )
                .cpu()
                .double()
                .numpy()
            )
    return scores


def _terminal_front_gross_returns(
    model: object,
    current_state: np.ndarray,
    gaussian_scores: np.ndarray,
) -> np.ndarray:
    with torch.inference_mode():
        surface = _decode_surface(
            model,
            torch.as_tensor(
                current_state[None], dtype=model.fdtype, device=model.device
            ),
        )
        support = surface.support_strikes[0, 0].cpu().double().numpy()
        density = surface.densities[0, 0].cpu().double().numpy()
        sqrt_variance = float(surface.backbone_total_vols[0, 0].cpu())
    return _front_quantile_spline(
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        gaussian_score=gaussian_scores,
    )


def _write_figure(
    output: Path,
    *,
    business_dtes: np.ndarray,
    historical_full: np.ndarray,
    historical_recent: np.ndarray,
    simulated: dict[str, np.ndarray],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    axis.plot(business_dtes, historical_full, color="#202020", label="Historical")
    axis.plot(
        business_dtes,
        historical_recent,
        color="#d55e00",
        label="Historical, 2025",
    )
    colors = ("#0072b2", "#009e73", "#cc79a7", "#9467bd", "#56b4e9")
    for index, (name, correlations) in enumerate(simulated.items()):
        axis.plot(
            business_dtes,
            correlations,
            color=colors[index % len(colors)],
            linestyle="--",
            label=name,
        )
    axis.axhline(0.0, color="#808080", linewidth=0.8)
    axis.set_xlabel("Business days to expiry")
    axis.set_ylabel(r"Corr$(\Delta\log S,\,\Delta\log\sigma_{\rm ATM})$")
    axis.set_ylim(-1.0, 0.05)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2, fontsize=8)
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
    ranks: tuple[int, ...],
    lookback: int | None,
    path_count: int,
    batch_size: int,
    recent_start_date: str,
    seed: int,
    device: str,
) -> dict[str, object]:
    """Fit daily joint laws and audit their terminal-state ATM correlations."""
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("ranks must be non-empty and unique")
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
        model,
        states[:-1],
        gross_returns,
        batch_size=batch_size,
    )
    joint_observations = np.column_stack((state_increments, spot_scores))
    if lookback is not None:
        if lookback <= joint_observations.shape[1] + 5 or lookback > len(
            joint_observations
        ):
            raise ValueError("lookback lacks covariance headroom or exceeds history")
        fitting_observations = joint_observations[-lookback:]
        fitting_start_date = dates[-lookback - 1]
    else:
        fitting_observations = joint_observations
        fitting_start_date = dates[0]

    reconstructed_atm = _decode_atm_volatilities(
        model, states, batch_size=batch_size
    )
    direct_dates, direct_dtes, direct_atm, direct_spot_returns = (
        _direct_sanos_history(
            history_path=historical_sanos_history,
            dataset_path=historical_dataset,
        )
    )
    if not np.array_equal(direct_dates, np.asarray(dates, dtype="U10")):
        raise ValueError("direct SANOS and DYSANOS histories have different dates")
    recent = direct_dates[1:] >= recent_start_date
    direct_changes = np.diff(np.log(direct_atm), axis=0)
    direct_full_all = _correlations(direct_spot_returns, direct_changes)
    direct_recent_all = _correlations(
        direct_spot_returns[recent], direct_changes[recent]
    )
    business_dtes = (
        np.asarray(model.config.geometry.model_expiries_y, dtype=np.float64) * 252.0
    )
    direct_indices = np.searchsorted(direct_dtes, business_dtes)
    if np.any(direct_indices == len(direct_dtes)) or np.any(
        direct_dtes[direct_indices] != business_dtes
    ):
        raise ValueError("direct SANOS history lacks a DYSANOS audit expiry")
    historical_full = direct_full_all[direct_indices]
    historical_recent = direct_recent_all[direct_indices]
    reconstructed_changes = np.diff(np.log(reconstructed_atm), axis=0)
    reconstructed_full = _correlations(
        direct_spot_returns, reconstructed_changes
    )

    fits: dict[str, DailyJointGaussianFit] = {}
    simulated_states: list[np.ndarray] = []
    simulated_spot_returns: dict[str, np.ndarray] = {}
    started = perf_counter()
    for rank_index, rank in enumerate(ranks):
        name = f"daily rank {rank}"
        fit = fit_daily_joint_gaussian(fitting_observations, factor_rank=rank)
        fits[name] = fit
        innovations = sample_antithetic_joint_innovations(
            fit,
            count=path_count,
            seed=seed + rank_index,
        )
        simulated_states.append(states[-1][None, :] + innovations[:, :-1])
        gross = _terminal_front_gross_returns(
            model,
            states[-1],
            innovations[:, -1],
        )
        simulated_spot_returns[name] = np.log(gross)
        print(
            f"prepared={name} retained_variance={fit.retained_variance_fraction:.6f} "
            f"elapsed_s={perf_counter() - started:.1f}",
            flush=True,
        )

    all_next_atm = _decode_atm_volatilities(
        model,
        np.concatenate(simulated_states, axis=0),
        batch_size=batch_size,
    )
    simulated_correlations: dict[str, np.ndarray] = {}
    simulated_errors: dict[str, np.ndarray] = {}
    for rank_index, name in enumerate(fits):
        first = rank_index * path_count
        next_atm = all_next_atm[first : first + path_count]
        log_vol_change = np.log(next_atm / reconstructed_atm[-1][None, :])
        spot_return = simulated_spot_returns[name]
        simulated_correlations[name] = _correlations(spot_return, log_vol_change)
        simulated_errors[name] = _antithetic_correlation_standard_errors(
            spot_return, log_vol_change
        )

    report: dict[str, object] = {
        "schema": "dysanos-v11-daily-joint-dynamics-v2",
        "status": "complete",
        "method": "daily conditional Gaussian state increment given the front-SANOS spot score; low-rank-plus-diagonal covariance for the residual state innovation",
        "drift": "sample mean daily increment; AR(1) deliberately omitted in this covariance pilot",
        "checkpoint": str(checkpoint.resolve()),
        "target_history": str(target_history.resolve()),
        "historical_sanos_history": str(historical_sanos_history.resolve()),
        "historical_dataset": str(historical_dataset.resolve()),
        "available_daily_observations": int(len(joint_observations)),
        "fitting_daily_observations": int(len(fitting_observations)),
        "fitting_start_date": fitting_start_date,
        "fitting_end_date": dates[-1],
        "recent_start_date": recent_start_date,
        "recent_observations": int(np.sum(recent)),
        "simulation_initial_date": dates[-1],
        "simulation_path_count": path_count,
        "simulation_independent_units": path_count // 2,
        "business_dtes": business_dtes.tolist(),
        "historical_full_correlations": historical_full.tolist(),
        "historical_recent_correlations": historical_recent.tolist(),
        "affine_reconstruction_full_correlations": reconstructed_full.tolist(),
        "spot_score_mean": float(np.mean(spot_scores)),
        "spot_score_standard_deviation": float(np.std(spot_scores, ddof=1)),
        "fits": {
            name: {
                "requested_rank": fit.requested_rank,
                "retained_variance_fraction": fit.retained_variance_fraction,
                "minimum_eigenvalue": fit.minimum_eigenvalue,
                "condition_number": fit.condition_number,
                "simulated_correlations": simulated_correlations[name].tolist(),
                "simulated_correlation_standard_errors": simulated_errors[name].tolist(),
            }
            for name, fit in fits.items()
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    _write_figure(
        output_figure,
        business_dtes=business_dtes,
        historical_full=historical_full,
        historical_recent=historical_recent,
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
    parser.add_argument("--ranks", type=int, nargs="+", default=(8, 16, 32, 64, 201))
    parser.add_argument(
        "--lookback",
        type=int,
        default=0,
        help="Trailing daily observations used by the covariance fit; zero uses all.",
    )
    parser.add_argument("--path-count", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--recent-start-date", default="2025-01-01")
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
        ranks=tuple(arguments.ranks),
        lookback=arguments.lookback or None,
        path_count=arguments.path_count,
        batch_size=arguments.batch_size,
        recent_start_date=arguments.recent_start_date,
        seed=arguments.seed,
        device=arguments.device,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
