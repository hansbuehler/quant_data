"""Build and audit the finite-step fixed-grid DLV spot/surface simulator."""

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
from quantdata.dysanos.flex.lvsq_drift import smooth_lvsq_log_loadings
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


@version("1.1")
@dataclass(frozen=True, kw_only=True)
class FixedGridDlvSimulatorConfig:
    """Logical one-week kernel and independent-path audit specification."""

    num_states: int = 21
    num_paths_per_state: int = 10_000
    path_batch_size: int = 512
    calendar_business_days: int = 5
    dlvsq_annualized_log_volatility: float = 0.10
    bs_backbone_annualized_volatility: float = 0.10
    num_option_strikes: int = 11
    kernel_method: str = "density_martingale"
    moment_tolerance: float = 1.0e-12
    correlation_eigenvalue_cutoff: float = 0.01
    antithetic: bool = True
    seed: int = 2_026_090_8

    def __post_init__(self) -> None:
        verify_inp(self.num_states > 0, lambda: "num_states must be positive")
        verify_inp(
            self.num_paths_per_state > 1,
            lambda: "num_paths_per_state must exceed one",
        )
        verify_inp(
            self.path_batch_size > 0,
            lambda: "path_batch_size must be positive",
        )
        verify_inp(
            self.calendar_business_days > 0,
            lambda: "calendar_business_days must be positive",
        )
        verify_inp(
            min(
                self.dlvsq_annualized_log_volatility,
                self.bs_backbone_annualized_volatility,
                self.moment_tolerance,
                self.correlation_eigenvalue_cutoff,
            )
            > 0.0,
            lambda: "volatilities and numerical tolerances must be positive",
        )
        verify_inp(
            self.correlation_eigenvalue_cutoff < 1.0,
            lambda: "correlation_eigenvalue_cutoff must be below one",
        )
        verify_inp(
            self.kernel_method in {"density_martingale", "exponential_newton"},
            lambda: f"unknown kernel_method {self.kernel_method!r}",
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
    parser.add_argument("--path-batch-size", type=int, default=512)
    parser.add_argument(
        "--kernel-method",
        choices=("density_martingale", "exponential_newton"),
        default="density_martingale",
    )
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
    """Test a zero mean after removing unresolved covariance modes."""

    standard_deviation = np.std(observations, axis=0, ddof=1)
    active = standard_deviation > max(
        1.0e-14,
        1.0e-10 * float(np.max(standard_deviation)),
    )
    values = observations[:, active] / standard_deviation[active]
    correlation = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    retained = eigenvalues > correlation_eigenvalue_cutoff
    mean = np.mean(values, axis=0)
    projected_mean = eigenvectors[:, retained].T @ mean
    statistic = float(
        len(values) * np.sum(np.square(projected_mean) / eigenvalues[retained])
    )
    degrees_of_freedom = int(np.sum(retained))
    z = mean * np.sqrt(len(values))
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
        "correlation_eigenvalue_cutoff": correlation_eigenvalue_cutoff,
        "minimum_retained_correlation_eigenvalue": float(
            np.min(eigenvalues[retained])
        ),
    }


def _audit_paths(
    *,
    normalized_spot: torch.Tensor,
    option_values: torch.Tensor,
    initial_option_values: torch.Tensor,
    state_indices: np.ndarray,
    state_dates: tuple[object, ...],
    correlation_eigenvalue_cutoff: float,
    antithetic: bool,
) -> dict[str, object]:
    spot_change = normalized_spot.numpy() - 1.0
    option_change = (
        option_values.numpy() - initial_option_values.numpy()[:, None]
    )
    batch, paths = spot_change.shape
    flattened_options = option_change.reshape(batch, paths, -1)
    observations = np.concatenate((spot_change[..., None], flattened_options), axis=-1)
    if antithetic:
        if paths % 2 != 0:
            raise ValueError("antithetic audit requires an even path count")
        observations = observations.reshape(batch, paths // 2, 2, -1).mean(axis=2)
        spot_change = spot_change.reshape(batch, paths // 2, 2).mean(axis=2)
        flattened_options = flattened_options.reshape(
            batch,
            paths // 2,
            2,
            -1,
        ).mean(axis=2)
    independent_observations = observations.shape[1]
    state_results: list[dict[str, object]] = []
    p_values: list[float] = []
    for state_position in range(batch):
        test = _wald_test(
            observations[state_position],
            correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
        )
        test.update(
            {
                "state_position": state_position,
                "source_state_index": int(state_indices[state_position]),
                "source_state_date": state_dates[state_position].isoformat(),
                "spot_mean_bps": float(
                    10_000.0 * np.mean(spot_change[state_position])
                ),
                "maximum_absolute_option_mean_bps_spot": float(
                    10_000.0
                    * np.max(np.abs(np.mean(flattened_options[state_position], axis=0)))
                ),
            }
        )
        state_results.append(test)
        p_values.append(float(test["p_value"]))
    clipped_p = np.clip(np.asarray(p_values), np.finfo(float).tiny, 1.0)
    fisher_statistic = float(-2.0 * np.sum(np.log(clipped_p)))
    all_standard_deviation = np.std(observations, axis=1, ddof=1)
    active = all_standard_deviation > np.maximum(
        1.0e-14,
        1.0e-10 * np.max(all_standard_deviation, axis=1, keepdims=True),
    )
    all_mean = np.mean(observations, axis=1)
    all_z = np.full_like(all_mean, np.nan)
    all_z[active] = (
        all_mean[active]
        / all_standard_deviation[active]
        * np.sqrt(independent_observations)
    )
    return {
        "num_states": batch,
        "num_paths_per_state": paths,
        "num_independent_observations_per_state": independent_observations,
        "antithetic": antithetic,
        "num_moments_per_state": observations.shape[-1],
        "maximum_absolute_spot_mean_bps": float(
            10_000.0 * np.max(np.abs(np.mean(spot_change, axis=1)))
        ),
        "maximum_absolute_option_mean_bps_spot": float(
            10_000.0 * np.max(np.abs(np.mean(flattened_options, axis=1)))
        ),
        "maximum_absolute_coordinate_z": float(np.nanmax(np.abs(all_z))),
        "root_mean_square_coordinate_z": float(
            np.sqrt(np.nanmean(np.square(all_z)))
        ),
        "num_absolute_coordinate_z_above_2": int(
            np.sum(np.abs(all_z) > 2.0)
        ),
        "minimum_state_omnibus_p_value": float(np.min(p_values)),
        "median_state_omnibus_p_value": float(np.median(p_values)),
        "fisher_combined_statistic": fisher_statistic,
        "fisher_combined_degrees_of_freedom": 2 * batch,
        "fisher_combined_p_value": float(chi2.sf(fisher_statistic, 2 * batch)),
        "states": state_results,
    }


def run(arguments: argparse.Namespace) -> None:
    """Build the exact branch kernel and run a fresh independent-path audit."""

    config = FixedGridDlvSimulatorConfig(
        num_states=arguments.num_states,
        num_paths_per_state=arguments.num_paths_per_state,
        path_batch_size=arguments.path_batch_size,
        kernel_method=arguments.kernel_method,
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
        raise ValueError("fixed-grid DLV simulation requires the fixed grid")
    if config.num_states > len(latent_states):
        raise ValueError("num_states exceeds the static state table")

    with torch.no_grad():
        all_raw = model.raw_states_from_states(latent_states)
        state_indices = np.linspace(
            0,
            len(all_raw) - 1,
            config.num_states,
            dtype=np.int64,
        )
        raw_blocks = all_raw.reshape(
            len(all_raw),
            geometry.num_model_expiries,
            geometry.expiry_block_state_size,
        )
        cap_margin = geometry.maximum_log_dlvsq - torch.amax(
            raw_blocks[:, 1:, : geometry.num_inner_strikes],
            dim=(1, 2),
        )
        worst_cap_index = int(torch.argmin(cap_margin).cpu())
        if worst_cap_index not in state_indices:
            state_indices[len(state_indices) // 2] = worst_cap_index
            state_indices.sort()
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
        loadings = smooth_lvsq_log_loadings(
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
        kernel_started = perf_counter()
        kernel = build_fixed_grid_dlv_kernel(
            model.sanos_decoder,
            initial_raw,
            raw_log_loadings=loadings,
            calendar_time_increment_y=config.calendar_business_days / DAYS_PER_YEAR,
            moment_tolerance=config.moment_tolerance,
            method=config.kernel_method,
        )
        _synchronize(env.device)
        kernel_seconds = perf_counter() - kernel_started
        exact = exact_fixed_grid_dlv_diagnostics(
            kernel,
            fixed_cash_strikes=fixed_cash_strikes,
        )

        generator = torch.Generator(device=torch.device(env.device)).manual_seed(
            config.seed
        )
        randoms = make_fixed_grid_dlv_randoms(
            batch_size=config.num_states,
            num_paths=config.num_paths_per_state,
            dtype=env.fdtype,
            device=env.device,
            generator=generator,
            antithetic=config.antithetic,
        )
        option_values = torch.empty(
            (
                config.num_states,
                config.num_paths_per_state,
                geometry.num_model_expiries,
                config.num_option_strikes,
            ),
            dtype=torch.float64,
            device="cpu",
        )
        normalized_spot = torch.empty(
            (config.num_states, config.num_paths_per_state),
            dtype=torch.float64,
            device="cpu",
        )
        maximum_mass_residual = 0.0
        maximum_mean_residual = 0.0
        maximum_regrid_cash_price_residual = 0.0
        initial_options = None
        _synchronize(env.device)
        path_started = perf_counter()
        for start in range(0, config.num_paths_per_state, config.path_batch_size):
            stop = min(start + config.path_batch_size, config.num_paths_per_state)
            batch_randoms = FixedGridDlvRandoms(
                surface_branch_uniforms=randoms.surface_branch_uniforms[:, start:stop],
                front_node_uniforms=randoms.front_node_uniforms[:, start:stop],
                front_backbone_normals=randoms.front_backbone_normals[:, start:stop],
            )
            sampled = sample_fixed_grid_dlv_paths(
                model.sanos_decoder,
                kernel,
                fixed_cash_strikes=fixed_cash_strikes,
                randoms=batch_randoms,
            )
            option_values[:, start:stop] = (
                sampled.fixed_cash_option_values.detach().cpu().double()
            )
            normalized_spot[:, start:stop] = (
                sampled.normalized_spot.detach().cpu().double()
            )
            initial_options = sampled.initial_fixed_cash_option_values.detach().cpu().double()
            maximum_mass_residual = max(
                maximum_mass_residual,
                float(sampled.maximum_conditional_mass_residual.cpu()),
            )
            maximum_mean_residual = max(
                maximum_mean_residual,
                float(sampled.maximum_conditional_mean_residual.cpu()),
            )
            maximum_regrid_cash_price_residual = max(
                maximum_regrid_cash_price_residual,
                float(sampled.regridded_state.maximum_cash_price_residual.cpu()),
            )
        _synchronize(env.device)
        path_seconds = perf_counter() - path_started
    assert initial_options is not None
    selected_dates = tuple(dates[int(value)] for value in state_indices)
    audit = _audit_paths(
        normalized_spot=normalized_spot,
        option_values=option_values,
        initial_option_values=initial_options,
        state_indices=state_indices,
        state_dates=selected_dates,
        correlation_eigenvalue_cutoff=config.correlation_eigenvalue_cutoff,
        antithetic=config.antithetic,
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_kind": "finite_step_fixed_grid_dlv_simulator",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "config": asdict(config),
        "config_version": config.version.unique_id48,
        "device": str(env.device),
        "precision": arguments.precision,
        "sources": sources,
        "source_state_indices": state_indices.tolist(),
        "source_state_dates": [value.isoformat() for value in selected_dates],
        "worst_cap_source_state_index": worst_cap_index,
        "backbone_forward_parameters": forward_parameters.detach().cpu().tolist(),
        "kernel": {
            "num_branches": len(kernel.finite_step.branch_probabilities),
            "method": kernel.method,
            "kernel_seconds": kernel_seconds,
            "newton_iterations_by_expiry": list(
                getattr(kernel.finite_step, "newton_iterations_by_expiry", ())
            ),
            "maximum_density_tower_residual": float(
                exact.maximum_density_tower_residual.cpu()
            ),
            "maximum_option_tower_residual": float(
                exact.maximum_option_tower_residual.cpu()
            ),
            "maximum_spot_mean_residual": float(
                exact.maximum_spot_mean_residual.cpu()
            ),
            "maximum_density_dlvsq_mean_residual": (
                float(kernel.finite_step.maximum_density_dlvsq_mean_residual.cpu())
                if hasattr(
                    kernel.finite_step,
                    "maximum_density_dlvsq_mean_residual",
                )
                else None
            ),
            "maximum_decoder_density_residual": float(
                kernel.finite_step.maximum_decoder_density_residual.cpu()
            ),
            "maximum_mass_residual": (
                float(kernel.finite_step.maximum_mass_residual.cpu())
                if hasattr(kernel.finite_step, "maximum_mass_residual")
                else None
            ),
            "maximum_forward_residual": (
                float(kernel.finite_step.maximum_forward_residual.cpu())
                if hasattr(kernel.finite_step, "maximum_forward_residual")
                else None
            ),
            "maximum_first_order_dlvsq_diffusion_residual": (
                float(
                    kernel.finite_step.maximum_first_order_dlvsq_diffusion_residual.cpu()
                )
                if hasattr(
                    kernel.finite_step,
                    "maximum_first_order_dlvsq_diffusion_residual",
                )
                else None
            ),
            "root_mean_square_first_order_dlvsq_diffusion_residual": (
                float(
                    kernel.finite_step.root_mean_square_first_order_dlvsq_diffusion_residual.cpu()
                )
                if hasattr(
                    kernel.finite_step,
                    "root_mean_square_first_order_dlvsq_diffusion_residual",
                )
                else None
            ),
            "relative_root_mean_square_dlvsq_diffusion_residual": (
                float(
                    kernel.finite_step.relative_root_mean_square_dlvsq_diffusion_residual.cpu()
                )
                if hasattr(
                    kernel.finite_step,
                    "relative_root_mean_square_dlvsq_diffusion_residual",
                )
                else None
            ),
            "minimum_density": (
                float(kernel.finite_step.minimum_density.cpu())
                if hasattr(kernel.finite_step, "minimum_density")
                else None
            ),
            "minimum_dlvsq": (
                float(kernel.finite_step.minimum_dlvsq.cpu())
                if hasattr(kernel.finite_step, "minimum_dlvsq")
                else None
            ),
            "maximum_dlvsq": (
                float(kernel.finite_step.maximum_dlvsq.cpu())
                if hasattr(kernel.finite_step, "maximum_dlvsq")
                else None
            ),
            "fallback_coordinate_fraction": (
                float(kernel.finite_step.fallback_coordinate_fraction.cpu())
                if hasattr(kernel.finite_step, "fallback_coordinate_fraction")
                else None
            ),
            "minimum_raw_cap_margin": float(
                kernel.finite_step.minimum_raw_cap_margin.cpu()
            ),
        },
        "paths": {
            "path_seconds": path_seconds,
            "path_state_steps_per_second": (
                config.num_states * config.num_paths_per_state / path_seconds
            ),
            "maximum_conditional_mass_residual": maximum_mass_residual,
            "maximum_conditional_mean_residual": maximum_mean_residual,
            "maximum_regrid_cash_price_residual": (
                maximum_regrid_cash_price_residual
            ),
        },
        "martingale_audit": audit,
        "interpretation": (
            "The finite DLV branch kernel is revealed independently of the front "
            "spot innovation. Remaining cash option values condition on both the "
            "revealed surface branch and the realized front spot state. No re-grid "
            "or empirical P transition is used."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, arguments.output)
    print(
        f"exact_option={report['kernel']['maximum_option_tower_residual']:.3e} "
        f"max_z={audit['maximum_absolute_coordinate_z']:.3f} "
        f"fisher_p={audit['fisher_combined_p_value']:.6g}",
        flush=True,
    )
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    run(_parser().parse_args())
