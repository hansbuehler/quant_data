"""Run and audit four exact-density weekly DLV rolls with pathwise re-grids."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from scipy.stats import chi2

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.fixed_grid_dlv import (
    FixedGridDlvRandoms,
    build_fixed_grid_dlv_kernel,
    exact_fixed_grid_dlv_diagnostics,
    make_fixed_grid_dlv_randoms,
    sample_fixed_grid_dlv_paths,
    set_constant_backbone_volatility,
)
from quantdata.dysanos.flex.floating_grid_dlv import (
    build_floating_grid_dlv_kernel,
    floating_surface_from_regrid,
    price_floating_surface_options,
    roll_floating_grid_dlv_kernel,
    smooth_floating_log_loadings,
)
from quantdata.dysanos.flex.lvsq_drift import smooth_lvsq_log_loadings
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class MultiStepRegridDlvConfig:
    """Four-week exact-density simulation and audit specification."""

    num_states: int = 21
    num_paths_per_state: int = 10_000
    path_batch_size: int = 256
    num_steps: int = 4
    calendar_business_days: int = 5
    dlvsq_annualized_log_volatility: float = 0.10
    bs_backbone_annualized_volatility: float = 0.10
    num_option_strikes: int = 11
    correlation_eigenvalue_cutoff: float = 0.01
    antithetic: bool = True
    seed: int = 2_026_090_9

    def __post_init__(self) -> None:
        verify_inp(self.num_states > 0, lambda: "num_states must be positive")
        verify_inp(
            self.num_paths_per_state >= 4,
            lambda: "num_paths_per_state must be at least four",
        )
        verify_inp(
            self.path_batch_size > 0,
            lambda: "path_batch_size must be positive",
        )
        verify_inp(
            self.num_steps == 4,
            lambda: "the current fixed-grid experiment has exactly four steps",
        )
        verify_inp(
            self.num_paths_per_state % 2 == 0
            and self.path_batch_size % 2 == 0,
            lambda: "antithetic path counts and batches must be even",
        )
        verify_inp(
            min(
                self.dlvsq_annualized_log_volatility,
                self.bs_backbone_annualized_volatility,
                self.correlation_eigenvalue_cutoff,
            )
            > 0.0,
            lambda: "volatilities and covariance cutoff must be positive",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=64, choices=(32, 64))
    parser.add_argument("--num-states", type=int, default=21)
    parser.add_argument("--num-paths-per-state", type=int, default=10_000)
    parser.add_argument("--path-batch-size", type=int, default=256)
    return parser


def _synchronize(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _wald_test(
    observations: np.ndarray,
    *,
    correlation_eigenvalue_cutoff: float,
) -> dict[str, float | int]:
    """Test a zero vector after discarding unresolved correlation modes."""

    standard_deviation = np.std(observations, axis=0, ddof=1)
    active = standard_deviation > max(
        1.0e-14,
        1.0e-10 * float(np.max(standard_deviation)),
    )
    standardized = observations[:, active] / standard_deviation[active]
    correlation = np.atleast_2d(np.cov(standardized, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    retained = eigenvalues > correlation_eigenvalue_cutoff
    if not bool(np.any(retained)):
        raise RuntimeError("martingale audit retained no covariance directions")
    mean = np.mean(standardized, axis=0)
    projected_mean = eigenvectors[:, retained].T @ mean
    statistic = float(
        len(standardized)
        * np.sum(np.square(projected_mean) / eigenvalues[retained])
    )
    degrees_of_freedom = int(np.sum(retained))
    z = mean * np.sqrt(len(standardized))
    return {
        "num_moments": int(observations.shape[1]),
        "num_active_moments": int(np.sum(active)),
        "degrees_of_freedom": degrees_of_freedom,
        "discarded_eigenvalues": int(len(eigenvalues) - degrees_of_freedom),
        "statistic": statistic,
        "p_value": float(chi2.sf(statistic, degrees_of_freedom)),
        "maximum_absolute_z": float(np.max(np.abs(z))),
        "root_mean_square_z": float(np.sqrt(np.mean(np.square(z)))),
        "num_absolute_z_above_2": int(np.sum(np.abs(z) > 2.0)),
        "minimum_retained_correlation_eigenvalue": float(
            np.min(eigenvalues[retained])
        ),
    }


def _audit_horizon(
    *,
    horizon: int,
    normalized_spot: torch.Tensor,
    option_values: torch.Tensor,
    initial_option_values: torch.Tensor,
    state_indices: np.ndarray,
    state_dates: tuple[object, ...],
    correlation_eigenvalue_cutoff: float,
) -> dict[str, object]:
    """Audit spot and every original contract still alive at one horizon."""

    spot_change = normalized_spot.numpy() - 1.0
    option_change = option_values.numpy() - initial_option_values.numpy()[:, None]
    states, paths = spot_change.shape
    flattened_options = option_change.reshape(states, paths, -1)
    observations = np.concatenate((spot_change[..., None], flattened_options), axis=-1)
    observations = observations.reshape(states, paths // 2, 2, -1).mean(axis=2)
    pair_spot_change = spot_change.reshape(states, paths // 2, 2).mean(axis=2)
    pair_option_change = flattened_options.reshape(
        states,
        paths // 2,
        2,
        -1,
    ).mean(axis=2)
    state_results: list[dict[str, object]] = []
    p_values: list[float] = []
    all_z: list[np.ndarray] = []
    for state_position in range(states):
        test = _wald_test(
            observations[state_position],
            correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
        )
        standard_deviation = np.std(observations[state_position], axis=0, ddof=1)
        active = standard_deviation > max(
            1.0e-14,
            1.0e-10 * float(np.max(standard_deviation)),
        )
        z = np.full(observations.shape[-1], np.nan)
        z[active] = (
            np.mean(observations[state_position], axis=0)[active]
            / standard_deviation[active]
            * np.sqrt(observations.shape[1])
        )
        all_z.append(z)
        test.update(
            {
                "state_position": state_position,
                "source_state_index": int(state_indices[state_position]),
                "source_state_date": state_dates[state_position].isoformat(),
                "spot_mean_bps": float(
                    10_000.0 * np.mean(pair_spot_change[state_position])
                ),
                "maximum_absolute_option_mean_bps_spot": float(
                    10_000.0
                    * np.max(
                        np.abs(np.mean(pair_option_change[state_position], axis=0))
                    )
                ),
            }
        )
        state_results.append(test)
        p_values.append(float(test["p_value"]))
    clipped_p = np.clip(np.asarray(p_values), np.finfo(float).tiny, 1.0)
    fisher_statistic = float(-2.0 * np.sum(np.log(clipped_p)))
    stacked_z = np.stack(all_z)
    return {
        "horizon_weeks": horizon,
        "num_states": states,
        "num_paths_per_state": paths,
        "num_independent_antithetic_pairs_per_state": paths // 2,
        "num_live_original_expiries": option_values.shape[2],
        "num_moments_per_state": observations.shape[-1],
        "maximum_absolute_spot_mean_bps": float(
            10_000.0 * np.max(np.abs(np.mean(pair_spot_change, axis=1)))
        ),
        "maximum_absolute_option_mean_bps_spot": float(
            10_000.0 * np.max(np.abs(np.mean(pair_option_change, axis=1)))
        ),
        "maximum_absolute_coordinate_z": float(np.nanmax(np.abs(stacked_z))),
        "root_mean_square_coordinate_z": float(
            np.sqrt(np.nanmean(np.square(stacked_z)))
        ),
        "num_absolute_coordinate_z_above_2": int(
            np.sum(np.abs(stacked_z) > 2.0)
        ),
        "minimum_state_omnibus_p_value": float(np.min(p_values)),
        "median_state_omnibus_p_value": float(np.median(p_values)),
        "fisher_combined_statistic": fisher_statistic,
        "fisher_combined_degrees_of_freedom": 2 * states,
        "fisher_combined_p_value": float(chi2.sf(fisher_statistic, 2 * states)),
        "correlation_eigenvalue_cutoff": correlation_eigenvalue_cutoff,
        "states": state_results,
    }


def _factor_scores(
    *,
    initial_dlvsqs: torch.Tensor,
    final_dlvsqs: torch.Tensor,
    raw_log_loadings: torch.Tensor,
    minimum_dlvsq: float,
    calendar_time_increment_y: float,
    coordinate_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project realized log-excess-DLV increments onto their four inputs."""

    response = (
        torch.log(final_dlvsqs[:, 1:] - minimum_dlvsq)
        - torch.log(initial_dlvsqs[:, 1:] - minimum_dlvsq)
    ).flatten(1) / calendar_time_increment_y**0.5
    design = raw_log_loadings[:, 1:].flatten(1, 2)
    ridge_fraction = 64.0 * torch.finfo(design.dtype).eps
    if coordinate_weights is not None:
        expected_shape = initial_dlvsqs.shape
        if coordinate_weights.shape != expected_shape:
            raise ValueError("coordinate_weights have an incompatible DLV shape")
        nonnegative_weight = torch.clamp_min(coordinate_weights[:, 1:], 0.0)
        maximum_weight = torch.amax(
            nonnegative_weight,
            dim=(-2, -1),
            keepdim=True,
        )
        weight_floor = torch.clamp_min(
            1.0e-8 * maximum_weight,
            1.0e-12,
        )
        square_root_weight = torch.sqrt(
            torch.maximum(nonnegative_weight, weight_floor)
        ).flatten(1)
        response = response * square_root_weight
        design = design * square_root_weight[..., None]
        ridge_fraction = 1.0e-10
    gram = torch.einsum("bmf,bmg->bfg", design, design)
    rhs = torch.einsum("bmf,bm->bf", design, response)
    scale = torch.diagonal(gram, dim1=-2, dim2=-1).mean(dim=-1)
    ridge = ridge_fraction * torch.clamp_min(
        scale,
        torch.finfo(design.dtype).tiny,
    )
    identity = torch.eye(
        design.shape[-1],
        dtype=design.dtype,
        device=design.device,
    )
    result = torch.linalg.solve(gram + ridge[:, None, None] * identity, rhs)
    if not bool(torch.all(torch.isfinite(result))):
        raise RuntimeError("DLV factor projection produced a non-finite score")
    return result


def _factor_covariance_audit(
    realized: torch.Tensor,
    intended: torch.Tensor,
) -> dict[str, object]:
    """Compare projected realized innovations with input unit covariance."""

    realized_np = realized.numpy()
    intended_np = intended.numpy()
    centered = realized_np - np.mean(realized_np, axis=2, keepdims=True)
    covariance = np.einsum("wsnf,wsng->wsfg", centered, centered) / (
        realized_np.shape[2] - 1
    )
    target = np.eye(realized_np.shape[-1])
    error = covariance - target
    intended_centered = intended_np - np.mean(intended_np, axis=2, keepdims=True)
    intended_covariance = np.einsum(
        "wsnf,wsng->wsfg",
        intended_centered,
        intended_centered,
    ) / (intended_np.shape[2] - 1)
    empirical_input_error = covariance - intended_covariance
    score_error_norm = np.linalg.vector_norm(realized_np - intended_np, axis=-1)
    return {
        "target_covariance": target.tolist(),
        "maximum_absolute_covariance_error": float(np.max(np.abs(error))),
        "root_mean_square_covariance_error": float(
            np.sqrt(np.mean(np.square(error)))
        ),
        "maximum_absolute_intended_covariance_error": float(
            np.max(np.abs(intended_covariance - target))
        ),
        "maximum_absolute_realized_vs_empirical_input_covariance_error": float(
            np.max(np.abs(empirical_input_error))
        ),
        "root_mean_square_realized_vs_empirical_input_covariance_error": float(
            np.sqrt(np.mean(np.square(empirical_input_error)))
        ),
        "realized_vs_intended_factor_score_rmse": float(
            np.sqrt(np.mean(np.square(realized_np - intended_np)))
        ),
        "realized_vs_intended_factor_score_error_norm_quantiles": {
            "median": float(np.quantile(score_error_norm, 0.50)),
            "p95": float(np.quantile(score_error_norm, 0.95)),
            "p99": float(np.quantile(score_error_norm, 0.99)),
            "maximum": float(np.max(score_error_norm)),
        },
        "by_week": [
            {
                "week": week + 1,
                "mean_covariance": np.mean(covariance[week], axis=0).tolist(),
                "mean_intended_covariance": np.mean(
                    intended_covariance[week], axis=0
                ).tolist(),
                "maximum_absolute_covariance_error": float(
                    np.max(np.abs(error[week]))
                ),
                "maximum_absolute_realized_vs_empirical_input_covariance_error": float(
                    np.max(np.abs(empirical_input_error[week]))
                ),
                "factor_score_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                realized_np[week] - intended_np[week]
                            )
                        )
                    )
                ),
                "factor_score_error_norm_quantiles": {
                    "median": float(np.quantile(score_error_norm[week], 0.50)),
                    "p95": float(np.quantile(score_error_norm[week], 0.95)),
                    "p99": float(np.quantile(score_error_norm[week], 0.99)),
                    "maximum": float(np.max(score_error_norm[week])),
                },
            }
            for week in range(realized_np.shape[0])
        ],
    }


def _intrinsic_values(
    spot: torch.Tensor,
    strikes: torch.Tensor,
) -> torch.Tensor:
    is_call = (strikes <= 1.0).reshape(1, -1)
    calls = torch.relu(spot[:, None] - strikes[None])
    puts = torch.relu(strikes[None] - spot[:, None])
    return torch.where(is_call, calls, puts)


def run(arguments: argparse.Namespace) -> None:
    """Run four exact conditional transitions and write a full audit."""

    config = MultiStepRegridDlvConfig(
        num_states=arguments.num_states,
        num_paths_per_state=arguments.num_paths_per_state,
        path_batch_size=arguments.path_batch_size,
    )
    env = BaseEnvironment(
        precision=arguments.precision,
        device=arguments.device,
        debug=False,
    )
    model, dates, latent_states, sources = load_static_q_inputs(
        arguments.static_decoder_checkpoint,
        arguments.static_report,
        env=env,
    )
    geometry = model.config.geometry
    if not isinstance(
        geometry,
        FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    ):
        raise ValueError("multi-step DLV simulation requires the fixed grid")
    if config.num_states > len(latent_states):
        raise ValueError("num_states exceeds the static state table")
    calendar_dt = config.calendar_business_days / DAYS_PER_YEAR

    with torch.no_grad():
        all_raw = model.raw_states_from_states(latent_states)
        state_indices = np.linspace(
            0,
            len(all_raw) - 1,
            config.num_states,
            dtype=np.int64,
        )
        index = torch.as_tensor(state_indices, dtype=torch.long, device=env.device)
        initial_raw, forward_parameters = set_constant_backbone_volatility(
            all_raw[index],
            decoder=model.sanos_decoder,
            annualized_volatility=config.bs_backbone_annualized_volatility,
        )
        initial_parameters = bound_native_raw(initial_raw, geometry=geometry)
        initial_surface = model.sanos_decoder.forward_batch(
            dlvsqs=initial_parameters.dlvsqs,
            forward_vols=initial_parameters.forward_vols,
            eta=initial_parameters.eta,
        )
        expected_backbone = (
            config.bs_backbone_annualized_volatility
            * torch.sqrt(model.sanos_decoder.model_expiries_y)
        )
        backbone_residual = torch.max(
            torch.abs(initial_surface.backbone_total_vols - expected_backbone)
        )
        fixed_loadings = smooth_lvsq_log_loadings(
            initial_surface.support_strikes[0, 0],
            num_model_expiries=geometry.num_model_expiries,
            annualized_log_volatility=config.dlvsq_annualized_log_volatility,
        )
        fixed_cash_strikes = torch.exp(
            torch.linspace(
                -0.30,
                0.30,
                config.num_option_strikes,
                dtype=env.fdtype,
                device=env.device,
            )
        )

        _synchronize(env.device)
        fixed_kernel_started = perf_counter()
        fixed_kernel = build_fixed_grid_dlv_kernel(
            model.sanos_decoder,
            initial_raw,
            raw_log_loadings=fixed_loadings,
            calendar_time_increment_y=calendar_dt,
        )
        _synchronize(env.device)
        fixed_kernel_seconds = perf_counter() - fixed_kernel_started
        exact = exact_fixed_grid_dlv_diagnostics(
            fixed_kernel,
            fixed_cash_strikes=fixed_cash_strikes,
        )

        generator = torch.Generator(device=torch.device(env.device)).manual_seed(
            config.seed
        )
        weekly_randoms = [
            make_fixed_grid_dlv_randoms(
                batch_size=config.num_states,
                num_paths=config.num_paths_per_state,
                dtype=env.fdtype,
                device=env.device,
                generator=generator,
                antithetic=True,
            )
            for _ in range(config.num_steps)
        ]
        spot_by_week = torch.empty(
            (config.num_steps, config.num_states, config.num_paths_per_state),
            dtype=torch.float64,
            device="cpu",
        )
        option_by_week = [
            torch.empty(
                (
                    config.num_states,
                    config.num_paths_per_state,
                    geometry.num_model_expiries - week,
                    config.num_option_strikes,
                ),
                dtype=torch.float64,
                device="cpu",
            )
            for week in range(config.num_steps)
        ]
        factor_scores = torch.empty(
            (
                config.num_steps,
                config.num_states,
                config.num_paths_per_state,
                fixed_loadings.shape[-1],
            ),
            dtype=torch.float64,
            device="cpu",
        )
        intended_factors = torch.empty_like(factor_scores)
        num_factors = fixed_loadings.shape[-1]
        maximum_conditional_mass_residual = 0.0
        maximum_conditional_mean_residual = 0.0
        maximum_regrid_cash_price_residual = 0.0
        maximum_floating_density_tower_residual = 0.0
        maximum_floating_decoder_residual = 0.0
        maximum_floating_mass_residual = 0.0
        maximum_floating_mean_residual = 0.0
        maximum_floating_fallback_fraction = 0.0
        minimum_floating_density = float("inf")
        minimum_floating_dlvsq = float("inf")
        maximum_floating_dlvsq = 0.0
        initial_options = None
        _synchronize(env.device)
        path_started = perf_counter()
        for start in range(0, config.num_paths_per_state, config.path_batch_size):
            stop = min(start + config.path_batch_size, config.num_paths_per_state)
            if (stop - start) % 2 != 0:
                raise ValueError("every path batch must preserve complete pairs")
            first_randoms = FixedGridDlvRandoms(
                surface_branch_uniforms=weekly_randoms[0].surface_branch_uniforms[
                    :, start:stop
                ],
                front_node_uniforms=weekly_randoms[0].front_node_uniforms[:, start:stop],
                front_backbone_normals=weekly_randoms[0].front_backbone_normals[
                    :, start:stop
                ],
            )
            sampled = sample_fixed_grid_dlv_paths(
                model.sanos_decoder,
                fixed_kernel,
                fixed_cash_strikes=fixed_cash_strikes,
                randoms=first_randoms,
            )
            paths_in_batch = stop - start
            spot_by_week[0, :, start:stop] = sampled.normalized_spot.cpu().double()
            option_by_week[0][:, start:stop] = (
                sampled.fixed_cash_option_values.cpu().double()
            )
            initial_options = sampled.initial_fixed_cash_option_values.cpu().double()
            maximum_conditional_mass_residual = max(
                maximum_conditional_mass_residual,
                float(sampled.maximum_conditional_mass_residual.cpu()),
            )
            maximum_conditional_mean_residual = max(
                maximum_conditional_mean_residual,
                float(sampled.maximum_conditional_mean_residual.cpu()),
            )
            maximum_regrid_cash_price_residual = max(
                maximum_regrid_cash_price_residual,
                float(sampled.regridded_state.maximum_cash_price_residual.cpu()),
            )
            state_index = torch.arange(config.num_states, device=env.device)[:, None]
            first_selected_dlvsqs = fixed_kernel.finite_step.dlvsqs_by_branch[
                state_index,
                sampled.surface_branch_indices,
            ]
            first_design = fixed_loadings.reshape(
                1,
                1,
                geometry.num_model_expiries,
                geometry.num_inner_strikes,
                -1,
            ).expand(config.num_states, paths_in_batch, -1, -1, -1)
            first_scores = _factor_scores(
                initial_dlvsqs=initial_surface.dlvsqs[:, None]
                .expand(-1, paths_in_batch, -1, -1)
                .reshape(-1, geometry.num_model_expiries, geometry.num_inner_strikes),
                final_dlvsqs=first_selected_dlvsqs.reshape(
                    -1,
                    geometry.num_model_expiries,
                    geometry.num_inner_strikes,
                ),
                raw_log_loadings=first_design.reshape(
                    -1,
                    geometry.num_model_expiries,
                    geometry.num_inner_strikes,
                    num_factors,
                ),
                minimum_dlvsq=geometry.min_dlvsq,
                calendar_time_increment_y=calendar_dt,
            ).reshape(config.num_states, paths_in_batch, num_factors)
            factor_scores[0, :, start:stop] = first_scores.cpu().double()
            intended_factors[0, :, start:stop] = fixed_kernel.finite_step.standardized_innovations[
                sampled.surface_branch_indices
            ].cpu().double()

            current_spot = sampled.normalized_spot.reshape(-1)
            surface = floating_surface_from_regrid(
                model.sanos_decoder,
                sampled.regridded_state,
            )
            for step_index in range(1, config.num_steps):
                loadings = smooth_floating_log_loadings(
                    surface.support_strikes[:, 0],
                    num_model_expiries=geometry.num_model_expiries,
                    annualized_log_volatility=config.dlvsq_annualized_log_volatility,
                )
                floating_kernel = build_floating_grid_dlv_kernel(
                    model.sanos_decoder,
                    surface,
                    raw_log_loadings=loadings,
                    calendar_time_increment_y=calendar_dt,
                )
                randoms = weekly_randoms[step_index]
                rolled = roll_floating_grid_dlv_kernel(
                    model.sanos_decoder,
                    floating_kernel,
                    surface_branch_uniforms=randoms.surface_branch_uniforms[
                        :, start:stop
                    ].reshape(-1),
                    front_node_uniforms=randoms.front_node_uniforms[
                        :, start:stop
                    ].reshape(-1),
                    front_backbone_normals=randoms.front_backbone_normals[
                        :, start:stop
                    ].reshape(-1),
                )
                flat_index = torch.arange(len(surface.densities), device=env.device)
                selected_dlvsqs = floating_kernel.dlvsqs_by_branch[
                    flat_index,
                    rolled.surface_branch_indices,
                ]
                scores = _factor_scores(
                    initial_dlvsqs=surface.dlvsqs,
                    final_dlvsqs=selected_dlvsqs,
                    raw_log_loadings=loadings,
                    minimum_dlvsq=geometry.min_dlvsq,
                    calendar_time_increment_y=calendar_dt,
                ).reshape(config.num_states, paths_in_batch, -1)
                factor_scores[step_index, :, start:stop] = scores.cpu().double()
                intended_factors[step_index, :, start:stop] = (
                    floating_kernel.standardized_innovations[
                        rolled.surface_branch_indices
                    ]
                    .reshape(config.num_states, paths_in_batch, -1)
                    .cpu()
                    .double()
                )
                current_spot = current_spot * rolled.gross_spot_return
                spot_by_week[step_index, :, start:stop] = (
                    current_spot.reshape(config.num_states, paths_in_batch)
                    .cpu()
                    .double()
                )
                payoff = _intrinsic_values(current_spot, fixed_cash_strikes)
                future_count = geometry.num_model_expiries - step_index - 1
                if future_count:
                    future = price_floating_surface_options(
                        rolled.surface,
                        current_spot=current_spot,
                        fixed_cash_strikes=fixed_cash_strikes,
                    )[:, :future_count]
                    live_values = torch.cat((payoff[:, None], future), dim=1)
                else:
                    live_values = payoff[:, None]
                option_by_week[step_index][:, start:stop] = (
                    live_values.reshape(
                        config.num_states,
                        paths_in_batch,
                        geometry.num_model_expiries - step_index,
                        config.num_option_strikes,
                    )
                    .cpu()
                    .double()
                )
                maximum_conditional_mass_residual = max(
                    maximum_conditional_mass_residual,
                    float(rolled.maximum_conditional_mass_residual.cpu()),
                )
                maximum_conditional_mean_residual = max(
                    maximum_conditional_mean_residual,
                    float(rolled.maximum_conditional_mean_residual.cpu()),
                )
                maximum_floating_density_tower_residual = max(
                    maximum_floating_density_tower_residual,
                    float(floating_kernel.maximum_density_mean_residual.cpu()),
                )
                maximum_floating_decoder_residual = max(
                    maximum_floating_decoder_residual,
                    float(floating_kernel.maximum_decoder_density_residual.cpu()),
                )
                maximum_floating_mass_residual = max(
                    maximum_floating_mass_residual,
                    float(floating_kernel.maximum_mass_residual.cpu()),
                )
                maximum_floating_mean_residual = max(
                    maximum_floating_mean_residual,
                    float(floating_kernel.maximum_mean_residual.cpu()),
                )
                maximum_floating_fallback_fraction = max(
                    maximum_floating_fallback_fraction,
                    float(floating_kernel.fallback_coordinate_fraction.cpu()),
                )
                minimum_floating_density = min(
                    minimum_floating_density,
                    float(floating_kernel.minimum_density.cpu()),
                )
                minimum_floating_dlvsq = min(
                    minimum_floating_dlvsq,
                    float(floating_kernel.minimum_dlvsq.cpu()),
                )
                maximum_floating_dlvsq = max(
                    maximum_floating_dlvsq,
                    float(floating_kernel.maximum_dlvsq.cpu()),
                )
                surface = rolled.surface
        _synchronize(env.device)
        path_seconds = perf_counter() - path_started

    assert initial_options is not None
    selected_dates = tuple(dates[int(value)] for value in state_indices)
    horizon_audits = [
        _audit_horizon(
            horizon=week + 1,
            normalized_spot=spot_by_week[week],
            option_values=option_by_week[week],
            initial_option_values=initial_options[:, week:],
            state_indices=state_indices,
            state_dates=selected_dates,
            correlation_eigenvalue_cutoff=config.correlation_eigenvalue_cutoff,
        )
        for week in range(config.num_steps)
    ]
    covariance_audit = _factor_covariance_audit(factor_scores, intended_factors)
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_kind": "multistep_price_preserving_regrid_dlv_simulator",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "config": asdict(config),
        "config_version": config.version.unique_id48,
        "device": str(env.device),
        "precision": arguments.precision,
        "sources": sources,
        "source_state_indices": state_indices.tolist(),
        "source_state_dates": [value.isoformat() for value in selected_dates],
        "backbone": {
            "annualized_volatility": config.bs_backbone_annualized_volatility,
            "maximum_total_volatility_residual": float(backbone_residual.cpu()),
            "forward_parameters_minimum": float(forward_parameters.min().cpu()),
            "forward_parameters_maximum": float(forward_parameters.max().cpu()),
            "eta_is_fixed_per_path": True,
        },
        "exact_fixed_grid_kernel": {
            "kernel_seconds": fixed_kernel_seconds,
            "num_branches": len(fixed_kernel.finite_step.branch_probabilities),
            "maximum_density_tower_residual": float(
                exact.maximum_density_tower_residual.cpu()
            ),
            "maximum_option_tower_residual": float(
                exact.maximum_option_tower_residual.cpu()
            ),
            "maximum_spot_mean_residual": float(
                exact.maximum_spot_mean_residual.cpu()
            ),
        },
        "floating_grid_kernels": {
            "maximum_density_tower_residual": (
                maximum_floating_density_tower_residual
            ),
            "maximum_decoder_density_residual": maximum_floating_decoder_residual,
            "maximum_mass_residual": maximum_floating_mass_residual,
            "maximum_mean_residual": maximum_floating_mean_residual,
            "maximum_fallback_coordinate_fraction_per_batch": (
                maximum_floating_fallback_fraction
            ),
            "minimum_density": minimum_floating_density,
            "minimum_dlvsq": minimum_floating_dlvsq,
            "maximum_dlvsq": maximum_floating_dlvsq,
        },
        "paths": {
            "path_seconds": path_seconds,
            "path_state_week_steps_per_second": (
                config.num_states
                * config.num_paths_per_state
                * config.num_steps
                / path_seconds
            ),
            "maximum_conditional_mass_residual": (
                maximum_conditional_mass_residual
            ),
            "maximum_conditional_mean_residual": (
                maximum_conditional_mean_residual
            ),
            "maximum_first_regrid_cash_price_residual": (
                maximum_regrid_cash_price_residual
            ),
            "antithetic_sources": {
                "surface_branch": "u and 1-u select opposite cubature branches",
                "discrete_spot_node": "inverse-CDF u and 1-u",
                "black_spot_factor": "Z and -Z",
                "pair_layout": "adjacent and preserved through all weekly rolls",
            },
        },
        "martingale_audits_by_horizon": horizon_audits,
        "realized_dlvsq_covariance_audit": covariance_audit,
        "interpretation": (
            "At every week the DLV branch is selected, the front node and Black "
            "factor determine spot, future densities are conditioned on that "
            "node, and the density is represented on the new relative grid. "
            "Every audit compares the value of each still-live original cash "
            "contract with its time-zero value."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, arguments.output)
    summary = " ".join(
        f"w{audit['horizon_weeks']}:z={audit['maximum_absolute_coordinate_z']:.3f},"
        f"p={audit['fisher_combined_p_value']:.4g}"
        for audit in horizon_audits
    )
    print(summary, flush=True)
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    run(_parser().parse_args())
