"""Audit the exact traded exponential density against direct analytical Q paths."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.fixed_expiry_ar1 import (
    fit_fixed_expiry_ar1,
    torch_fixed_expiry_raw_log_loadings,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.run_direct_utility_frontier import (
    PhysicalTowerDataset,
    _correlation_aware_multiplier_test,
    _physical_tower_dataset,
    _weighted_martingale_audit,
)
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-index", type=int, default=44)
    parser.add_argument("--num-paths", type=int, default=100_000)
    parser.add_argument("--path-batch-size", type=int, default=1_000)
    parser.add_argument("--antithetic-pairs-per-block", type=int, default=50)
    parser.add_argument("--innovation-rank", type=int, default=25)
    parser.add_argument("--num-substeps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2_026_092_4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--target-ess-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-action-l1", type=float, default=256.0)
    parser.add_argument("--traded-ridge-ratio", type=float, default=1.0e-6)
    return parser


def _validate(arguments: argparse.Namespace) -> None:
    block = 2 * arguments.antithetic_pairs_per_block
    if min(
        arguments.num_paths,
        arguments.path_batch_size,
        arguments.antithetic_pairs_per_block,
        arguments.innovation_rank,
        arguments.num_substeps,
        arguments.bootstrap_samples,
    ) <= 0:
        raise ValueError("path, block, rank, substep, and bootstrap counts must be positive")
    if arguments.num_paths % block or arguments.path_batch_size % block:
        raise ValueError("path counts and generation batches must contain complete blocks")
    if arguments.num_paths % arguments.path_batch_size:
        raise ValueError("num-paths must be divisible by path-batch-size")
    if not 0.0 < arguments.target_ess_fraction <= 1.0:
        raise ValueError("target-ess-fraction must lie in (0, 1]")
    if not math.isfinite(arguments.maximum_action_l1) or arguments.maximum_action_l1 <= 0.0:
        raise ValueError("maximum-action-l1 must be finite and positive")
    if not 0.0 <= arguments.traded_ridge_ratio < 1.0:
        raise ValueError("traded-ridge-ratio must lie in [0, 1)")


def _normalized_density(log_density: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    """Normalize each density-process time slice and report overlap diagnostics."""

    if log_density.ndim != 2 or log_density.shape[1] != 4:
        raise ValueError("log_density must have shape (paths, four trade weeks)")
    log_density = log_density.double()
    log_paths = math.log(len(log_density))
    log_mass = torch.logsumexp(log_density, dim=0) - log_paths
    log_weights = log_density - torch.logsumexp(log_density, dim=0)[None] + log_paths
    weights = torch.exp(log_weights)
    diagnostics: list[dict[str, float | int]] = []
    for time_index in range(4):
        time_weights = weights[:, time_index]
        effective_sample_size = len(weights) ** 2 / torch.sum(time_weights.square())
        diagnostics.append(
            {
                "trade_week": time_index,
                "unnormalized_log_mean_density": float(log_mass[time_index]),
                "minimum_log_density": float(torch.min(log_density[:, time_index])),
                "maximum_log_density": float(torch.max(log_density[:, time_index])),
                "effective_sample_size": float(effective_sample_size),
                "effective_sample_fraction": float(effective_sample_size / len(weights)),
                "maximum_normalized_density": float(torch.max(time_weights)),
                "numerically_zero_density_count": int(torch.sum(time_weights == 0.0)),
            }
        )
    return weights, diagnostics


def _time_dependent_martingale_audit(
    dataset: PhysicalTowerDataset,
    weights_by_time: torch.Tensor,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Test each one-step return with the density known at its endpoint."""

    if weights_by_time.shape != (dataset.num_paths, 4):
        raise ValueError("weights_by_time must have shape (paths, four trade weeks)")
    if not bool(torch.all(torch.isfinite(weights_by_time) & (weights_by_time >= 0.0))):
        raise ValueError("weights must be finite and nonnegative")
    if not bool(torch.all(torch.abs(torch.mean(weights_by_time, dim=0) - 1.0) < 1.0e-10)):
        raise ValueError("each endpoint density must have sample mean one")

    block_size = dataset.paths_per_independent_block
    num_blocks = dataset.num_paths // block_size
    option_blocks: list[np.ndarray] = []
    option_by_week: list[dict[str, float | int]] = []
    spot_blocks: list[torch.Tensor] = []
    with torch.no_grad():
        for time_index, returns in enumerate(dataset.returns_by_time):
            weights = weights_by_time[:, time_index].to(returns.device)
            blocks = torch.mean(
                weights.reshape(num_blocks, block_size, 1)
                * returns.reshape(num_blocks, block_size, -1),
                dim=1,
            ).double()
            mean = torch.mean(blocks, dim=0)
            standard_error = torch.std(blocks, dim=0, correction=1) / math.sqrt(num_blocks)
            active = standard_error > 1.0e-14
            t_value = torch.where(active, mean / standard_error.clamp_min(1.0e-14), torch.nan)
            option_by_week.append(
                {
                    "trade_week": time_index,
                    "num_coordinates": returns.shape[1],
                    "maximum_absolute_weighted_mean": float(torch.max(torch.abs(mean))),
                    "maximum_absolute_t": (
                        float(torch.max(torch.abs(t_value[active]))) if bool(torch.any(active)) else 0.0
                    ),
                    "root_mean_square_t": (
                        float(torch.sqrt(torch.mean(t_value[active].square())))
                        if bool(torch.any(active))
                        else 0.0
                    ),
                    "num_absolute_t_above_2": int(torch.sum(torch.abs(t_value[active]) > 2.0)),
                }
            )
            option_blocks.append(blocks.cpu().numpy())
            spot = dataset.spot_returns[:, time_index]
            spot_blocks.append(
                torch.mean(
                    weights.reshape(num_blocks, block_size)
                    * spot.reshape(num_blocks, block_size),
                    dim=1,
                )[:, None].double()
            )
    option_test = _correlation_aware_multiplier_test(
        np.concatenate(option_blocks, axis=1),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    spot_test = _correlation_aware_multiplier_test(
        torch.cat(spot_blocks, dim=1).cpu().numpy(),
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    return {
        "density_timing": "the cumulative endpoint density Z[t+1] weights the return t->t+1",
        "weighted_option_zero_drift": option_test,
        "weighted_option_by_trade_week": option_by_week,
        "weighted_spot_zero_drift": spot_test,
    }


def _factor_shift_diagnostics(values: torch.Tensor) -> list[dict[str, float | int]]:
    """Summarize cumulative integrated squared market prices of risk."""

    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("integrated factor shifts must have shape (paths, four weeks)")
    values = values.double()
    result: list[dict[str, float | int]] = []
    for time_index in range(4):
        current = values[:, time_index]
        result.append(
            {
                "trade_week": time_index,
                "mean_integrated_squared_factor_shift": float(torch.mean(current)),
                "median_integrated_squared_factor_shift": float(torch.median(current)),
                "maximum_integrated_squared_factor_shift": float(torch.max(current)),
                "minimum_lognormal_ess_fraction_from_maximum": float(
                    torch.exp(-torch.max(current))
                ),
            }
        )
    return result


def _uniform_audit(dataset: PhysicalTowerDataset, *, samples: int, seed: int) -> dict[str, Any]:
    return _weighted_martingale_audit(
        dataset,
        torch.ones((1, dataset.num_paths), dtype=dataset.returns_by_time[0].dtype),
        bootstrap_samples=samples,
        bootstrap_seed=seed,
    )["heads"][0]


def run(arguments: argparse.Namespace) -> None:
    """Generate matched P/Q paths and audit the theoretical traded density."""

    _validate(arguments)
    env = BaseEnvironment(precision=64, device=arguments.device, debug=False)
    model, dates, latent_states, sources = load_static_q_inputs(
        arguments.static_decoder_checkpoint,
        arguments.static_report,
        env=env,
    )
    geometry = model.config.geometry
    if not isinstance(geometry, FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("theoretical traded density requires the fitted fixed cash grid")
    if not 0 <= arguments.state_index < len(latent_states):
        raise ValueError("state-index lies outside the static state table")

    with torch.no_grad():
        all_raw = model.raw_states_from_states(latent_states)
        fixed_expiry_fit = fit_fixed_expiry_ar1(
            all_raw.cpu().double().numpy(),
            dates,
            num_inner_strikes=geometry.num_inner_strikes,
            rank=arguments.innovation_rank,
            training_end=date(2024, 12, 31),
        )
        raw_log_loadings = torch_fixed_expiry_raw_log_loadings(
            fixed_expiry_fit,
            dtype=env.fdtype,
            device=env.device,
        )
        intercept = torch.as_tensor(
            fixed_expiry_fit.conditional_intercept,
            dtype=env.fdtype,
            device=env.device,
        )
        slope = torch.as_tensor(
            fixed_expiry_fit.conditional_slope,
            dtype=env.fdtype,
            device=env.device,
        )
        parameters = bound_native_raw(
            all_raw[arguments.state_index : arguments.state_index + 1],
            geometry=geometry,
        )
        surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        strikes = torch.as_tensor(
            geometry.inner_relative_strikes,
            dtype=env.fdtype,
            device=env.device,
        )
        horizon_y = 4.0 * 5.0 / DAYS_PER_YEAR
        entropy_budget = -math.log(arguments.target_ess_fraction)
        maximum_factor_shift_norm = math.sqrt(entropy_budget / horizon_y)
        common = {
            "model": model,
            "geometry": geometry,
            "surface": surface,
            "parameters": parameters,
            "raw_log_loadings": raw_log_loadings,
            "fixed_cash_strikes": strikes,
            "num_paths": arguments.num_paths,
            "path_batch_size": arguments.path_batch_size,
            "pairs_per_block": arguments.antithetic_pairs_per_block,
            "seed": arguments.seed,
            "state_feature_mode": "density",
            "analytic_num_substeps": arguments.num_substeps,
            "dataset_storage": "cpu",
            "ar1_conditional_intercept": intercept,
            "ar1_conditional_slope": slope,
            "bs_y_vol": 0.0,
            "traded_density_ridge_ratio": arguments.traded_ridge_ratio,
            "traded_density_maximum_factor_shift_norm": maximum_factor_shift_norm,
            "traded_density_maximum_action_l1": arguments.maximum_action_l1,
        }
        physical = _physical_tower_dataset(
            **common,
            path_law="achievable-traded-density",
        )
        analytical_q = _physical_tower_dataset(
            **common,
            path_law="analytic",
        )

    if physical.theoretical_log_density_by_time is None:
        raise RuntimeError("physical simulation did not return its theoretical density")
    if physical.theoretical_integrated_squared_factor_shift_by_time is None:
        raise RuntimeError("physical simulation did not return its integrated factor shifts")
    weights, density_diagnostics = _normalized_density(
        physical.theoretical_log_density_by_time
    )
    physical_uniform = _uniform_audit(
        physical,
        samples=arguments.bootstrap_samples,
        seed=arguments.seed + 100_000,
    )
    theoretical_q = _time_dependent_martingale_audit(
        physical,
        weights,
        bootstrap_samples=arguments.bootstrap_samples,
        bootstrap_seed=arguments.seed + 200_000,
    )
    direct_q = _uniform_audit(
        analytical_q,
        samples=arguments.bootstrap_samples,
        seed=arguments.seed + 300_000,
    )
    report = {
        "schema_version": 1,
        "experiment_kind": "theoretical_traded_exponential_density",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": {
            "action": "minimum-norm a solving B' a = lambda, equivalently a=Sigma_R^+ mu_R",
            "density": "u'(sum a'dR)=exp(-sum a'dR), with the exact Gaussian normalizer",
            "learning": "none",
            "information": "density evaluated only from traded option returns; driving normals excluded",
            "viability_constraints": {
                "target_ess_fraction": arguments.target_ess_fraction,
                "four_week_entropy_budget": entropy_budget,
                "maximum_factor_shift_norm": maximum_factor_shift_norm,
                "maximum_action_l1": arguments.maximum_action_l1,
                "traded_ridge_ratio": arguments.traded_ridge_ratio,
            },
        },
        "state_index": arguments.state_index,
        "state_date": dates[arguments.state_index].isoformat(),
        "sources": sources,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "fixed_expiry_ar1": fixed_expiry_fit.diagnostics.to_dict(),
        "physical_generation": physical.diagnostics,
        "direct_q_generation": analytical_q.diagnostics,
        "density_by_endpoint": density_diagnostics,
        "factor_shift_by_endpoint": _factor_shift_diagnostics(
            physical.theoretical_integrated_squared_factor_shift_by_time
        ),
        "physical_uniform_audit": physical_uniform,
        "theoretical_traded_q_audit": theoretical_q,
        "direct_q_uniform_audit": direct_q,
    }
    write_json_atomic(report, arguments.output)
    for label, audit in (
        ("P", physical_uniform),
        ("traded Q", theoretical_q),
        ("direct Q", direct_q),
    ):
        option = audit["weighted_option_zero_drift"]
        spot = audit["weighted_spot_zero_drift"]
        print(
            f"{label}: option p={option['combined_p_value']:.6g}, "
            f"max|t|={option['maximum_absolute_t']:.6g}; "
            f"spot p={spot['combined_p_value']:.6g}, max|t|={spot['maximum_absolute_t']:.6g}",
            flush=True,
        )
    print(json.dumps(density_diagnostics, indent=2), flush=True)
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
