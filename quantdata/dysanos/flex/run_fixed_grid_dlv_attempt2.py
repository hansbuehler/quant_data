"""Run Attempt 2.0: a one-week rolling fixed-grid DLVsq SDE audit."""

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
from quantdata.dysanos.flex.fixed_grid_dlv_attempt2 import (
    FixedGridAttempt2Randoms,
    fixed_grid_attempt2_conditional_densities,
    fixed_grid_attempt2_sde_roll,
    make_fixed_grid_attempt2_randoms,
)
from quantdata.dysanos.flex.lvsq_drift import smooth_lvsq_log_loadings
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.run_fixed_grid_dlv_simulator import _wald_test
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


@version("1.1")
@dataclass(frozen=True, kw_only=True)
class FixedGridDlvAttempt2Config:
    """Numerical and statistical specification for the first 2.0 audit."""

    num_states: int = 21
    num_paths_per_state: int = 10_000
    path_batch_size: int = 500
    num_substeps: int = 10
    calendar_business_days: int = 5
    dlvsq_annualized_log_volatility: float = 0.10
    correlation_eigenvalue_cutoff: float = 0.01
    z_resolution_standard_error_floor_bps_spot: float = 1.0e-8
    antithetic: bool = True
    seed: int = 2_026_091_1

    def __post_init__(self) -> None:
        verify_inp(self.num_states > 0, lambda: "num_states must be positive")
        verify_inp(
            self.num_paths_per_state >= 4 and self.num_paths_per_state % 2 == 0,
            lambda: "num_paths_per_state must be even and at least four",
        )
        verify_inp(
            self.path_batch_size > 0 and self.path_batch_size % 2 == 0,
            lambda: "path_batch_size must be positive and even",
        )
        verify_inp(self.num_substeps > 0, lambda: "num_substeps must be positive")
        verify_inp(
            self.calendar_business_days > 0,
            lambda: "calendar_business_days must be positive",
        )
        verify_inp(
            self.dlvsq_annualized_log_volatility > 0.0,
            lambda: "DLVsq annualized log volatility must be positive",
        )
        verify_inp(
            0.0 < self.correlation_eigenvalue_cutoff < 1.0,
            lambda: "correlation eigenvalue cutoff must lie between zero and one",
        )
        verify_inp(
            self.z_resolution_standard_error_floor_bps_spot > 0.0,
            lambda: "z-resolution standard-error floor must be positive",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=64, choices=(32, 64))
    parser.add_argument("--num-states", type=int, default=21)
    parser.add_argument(
        "--state-indices",
        type=str,
        help="Optional comma-separated source-state indices; overrides --num-states.",
    )
    parser.add_argument("--num-paths-per-state", type=int, default=10_000)
    parser.add_argument("--path-batch-size", type=int, default=500)
    parser.add_argument("--num-substeps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2_026_091_1)
    return parser


def _synchronize(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _pair_mean_observations(values: np.ndarray, *, antithetic: bool) -> np.ndarray:
    """Turn adjacent path pairs into the independent observations being tested."""

    if not antithetic:
        return values
    if values.shape[1] % 2:
        raise ValueError("antithetic observations require an even path count")
    return values.reshape(values.shape[0], values.shape[1] // 2, 2, -1).mean(axis=2)


def _wald_test_with_reference_covariance(
    covariance_observations: np.ndarray,
    mean_observations: np.ndarray,
    *,
    correlation_eigenvalue_cutoff: float,
) -> dict[str, float | int]:
    """Test a mean in modes selected from a disjoint covariance sample."""

    standard_deviation = np.std(covariance_observations, axis=0, ddof=1)
    active = standard_deviation > max(
        1.0e-14,
        1.0e-10 * float(np.max(standard_deviation)),
    )
    covariance_values = covariance_observations[:, active] / standard_deviation[active]
    mean_values = mean_observations[:, active] / standard_deviation[active]
    correlation = np.atleast_2d(np.cov(covariance_values, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    retained = eigenvalues > correlation_eigenvalue_cutoff
    if not bool(np.any(retained)):
        raise RuntimeError("independent-covariance audit retained no directions")
    projected_mean = eigenvectors[:, retained].T @ np.mean(mean_values, axis=0)
    statistic = float(
        len(mean_values)
        * np.sum(np.square(projected_mean) / eigenvalues[retained])
    )
    degrees_of_freedom = int(np.sum(retained))
    return {
        "num_covariance_observations": len(covariance_observations),
        "num_mean_observations": len(mean_observations),
        "degrees_of_freedom": degrees_of_freedom,
        "statistic": statistic,
        "p_value": float(chi2.sf(statistic, degrees_of_freedom)),
        "minimum_retained_correlation_eigenvalue": float(
            np.min(eigenvalues[retained])
        ),
    }


def _split_sample_wald_tests(
    observations: np.ndarray,
    *,
    correlation_eigenvalue_cutoff: float,
) -> dict[str, object]:
    """Estimate covariance on one half and test the other half's mean."""

    even = observations[0::2]
    odd = observations[1::2]
    if min(len(even), len(odd)) < 2:
        raise ValueError("split-sample inference needs at least four observations")
    return {
        "even_covariance_odd_mean": _wald_test_with_reference_covariance(
            even,
            odd,
            correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
        ),
        "odd_covariance_even_mean": _wald_test_with_reference_covariance(
            odd,
            even,
            correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
        ),
    }


def _statewise_zero_mean_audit(
    observations: np.ndarray,
    *,
    state_indices: np.ndarray,
    state_dates: tuple[object, ...],
    correlation_eigenvalue_cutoff: float,
    standard_error_floor_bps_spot: float,
) -> dict[str, object]:
    """Audit each initial state's mean and combine independent state tests."""

    if observations.ndim != 3 or observations.shape[0] != len(state_indices):
        raise ValueError("observations must have shape (state, sample, moment)")
    state_results: list[dict[str, object]] = []
    p_values: list[float] = []
    coordinate_z: list[np.ndarray] = []
    resolved_counts: list[int] = []
    for state_position, state_observations in enumerate(observations):
        standard_deviation = np.std(state_observations, axis=0, ddof=1)
        coordinate_standard_error = standard_deviation / np.sqrt(
            len(state_observations)
        )
        resolved = (
            10_000.0 * coordinate_standard_error
            >= standard_error_floor_bps_spot
        )
        if not bool(np.any(resolved)):
            raise RuntimeError("the numerical resolution floor removed every moment")
        resolved_observations = state_observations[:, resolved]
        test = _wald_test(
            resolved_observations,
            correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
        )
        active = standard_deviation > max(
            1.0e-14,
            1.0e-10 * float(np.max(standard_deviation)),
        )
        active &= resolved
        z = np.full(state_observations.shape[1], np.nan)
        z[active] = (
            np.mean(state_observations, axis=0)[active]
            / standard_deviation[active]
            * np.sqrt(len(state_observations))
        )
        coordinate_mean = np.mean(state_observations, axis=0)
        coordinate_z.append(z)
        resolved_counts.append(int(np.sum(resolved)))
        test.update(
            {
                "state_position": state_position,
                "source_state_index": int(state_indices[state_position]),
                "source_state_date": state_dates[state_position].isoformat(),
                "maximum_absolute_mean_bps_spot": float(
                    10_000.0 * np.max(np.abs(coordinate_mean))
                ),
                "coordinate_mean_bps_spot": (10_000.0 * coordinate_mean).tolist(),
                "coordinate_standard_error_bps_spot": (
                    10_000.0 * coordinate_standard_error
                ).tolist(),
                "coordinate_z": z.tolist(),
                "num_resolved_moments": int(np.sum(resolved)),
                "num_numerically_unresolved_moments": int(np.sum(~resolved)),
                "independent_covariance_split": _split_sample_wald_tests(
                    resolved_observations,
                    correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
                ),
                "correlation_cutoff_sensitivity": {
                    f"{cutoff:.3f}": _wald_test(
                        resolved_observations,
                        correlation_eigenvalue_cutoff=cutoff,
                    )
                    for cutoff in (0.005, 0.01, 0.02, 0.05)
                },
            }
        )
        state_results.append(test)
        p_values.append(float(test["p_value"]))
    all_z = np.stack(coordinate_z)
    clipped_p = np.clip(np.asarray(p_values), np.finfo(float).tiny, 1.0)
    fisher_statistic = float(-2.0 * np.sum(np.log(clipped_p)))
    return {
        "num_states": len(state_indices),
        "num_independent_observations_per_state": observations.shape[1],
        "num_moments_per_state": observations.shape[2],
        "minimum_resolved_moments_per_state": int(np.min(resolved_counts)),
        "maximum_resolved_moments_per_state": int(np.max(resolved_counts)),
        "z_resolution_standard_error_floor_bps_spot": (
            standard_error_floor_bps_spot
        ),
        "maximum_absolute_mean_bps_spot": float(
            10_000.0 * np.max(np.abs(np.mean(observations, axis=1)))
        ),
        "root_mean_square_mean_bps_spot": float(
            10_000.0 * np.sqrt(np.mean(np.square(np.mean(observations, axis=1))))
        ),
        "maximum_absolute_coordinate_z": float(np.nanmax(np.abs(all_z))),
        "root_mean_square_coordinate_z": float(np.sqrt(np.nanmean(np.square(all_z)))),
        "num_absolute_coordinate_z_above_2": int(np.sum(np.abs(all_z) > 2.0)),
        "minimum_state_omnibus_p_value": float(np.min(p_values)),
        "median_state_omnibus_p_value": float(np.median(p_values)),
        "fisher_combined_statistic": fisher_statistic,
        "fisher_combined_degrees_of_freedom": 2 * len(state_indices),
        "fisher_combined_p_value": float(
            chi2.sf(fisher_statistic, 2 * len(state_indices))
        ),
        "states": state_results,
    }


def _maximum(current: float, value: torch.Tensor) -> float:
    return max(current, float(value.detach().cpu()))


def _minimum(current: float, value: torch.Tensor) -> float:
    return min(current, float(value.detach().cpu()))


def _spot_reveal_tower_residual(
    decoder: object,
    *,
    raw_states: torch.Tensor,
) -> float:
    """Enumerate spot nodes and verify the roll for actual endpoint SDE states."""

    geometry = decoder.geometry
    parameters = bound_native_raw(raw_states, geometry=geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    num_states = len(raw_states)
    support_size = geometry.num_inner_strikes + 2
    support = surface.support_strikes[:, 0]
    nodes = torch.arange(support_size, device=raw_states.device)
    conditional = fixed_grid_attempt2_conditional_densities(
        decoder,
        support=(
            support[:, None]
            .expand(num_states, support_size, support_size)
        ),
        endpoint_dlvsqs=(
            parameters.dlvsqs[:, None]
            .expand(
                num_states,
                support_size,
                geometry.num_model_expiries,
                geometry.num_inner_strikes,
            )
        ),
        front_node_indices=nodes[None].expand(num_states, -1),
    )
    mean = torch.einsum("br,brts->bts", surface.densities[:, 0], conditional)
    return float(torch.max(torch.abs(mean - surface.densities)).detach().cpu())


def run(arguments: argparse.Namespace) -> None:
    """Execute the Attempt 2.0 fixed-grid SDE and martingale audits."""

    requested_state_indices = (
        tuple(int(value) for value in arguments.state_indices.split(","))
        if arguments.state_indices
        else None
    )
    config = FixedGridDlvAttempt2Config(
        num_states=(
            len(requested_state_indices)
            if requested_state_indices is not None
            else arguments.num_states
        ),
        num_paths_per_state=arguments.num_paths_per_state,
        path_batch_size=arguments.path_batch_size,
        num_substeps=arguments.num_substeps,
        seed=arguments.seed,
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
        raise ValueError("Attempt 2.0 requires the fitted fixed grid")
    if config.num_states > len(latent_states):
        raise ValueError("num_states exceeds the static state table")

    with torch.no_grad():
        all_raw = model.raw_states_from_states(latent_states)
        if requested_state_indices is not None:
            if (
                len(set(requested_state_indices)) != len(requested_state_indices)
                or min(requested_state_indices) < 0
                or max(requested_state_indices) >= len(all_raw)
            ):
                raise ValueError("state_indices must be unique valid source rows")
            state_indices = np.asarray(requested_state_indices, dtype=np.int64)
        else:
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
        if requested_state_indices is None and worst_cap_index not in state_indices:
            state_indices[len(state_indices) // 2] = worst_cap_index
            state_indices.sort()
        index = torch.as_tensor(state_indices, dtype=torch.long, device=env.device)
        initial_raw = all_raw[index]
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
        fixed_cash_strikes = torch.as_tensor(
            geometry.inner_relative_strikes,
            dtype=env.fdtype,
            device=env.device,
        )
        num_audit_strikes = len(fixed_cash_strikes)

        aligned_options = torch.empty(
            (
                config.num_states,
                config.num_paths_per_state,
                geometry.num_model_expiries,
                num_audit_strikes,
            ),
            dtype=torch.float64,
            device="cpu",
        )
        endpoint_survivor_options = torch.empty(
            (
                config.num_states,
                config.num_paths_per_state,
                geometry.num_model_expiries - 1,
                num_audit_strikes,
            ),
            dtype=torch.float64,
            device="cpu",
        )
        normalized_spot = torch.empty(
            (config.num_states, config.num_paths_per_state),
            dtype=torch.float64,
            device="cpu",
        )
        initial_options: torch.Tensor | None = None
        maximum_conditional_mass_residual = 0.0
        maximum_conditional_forward_residual = 0.0
        maximum_fixed_grid_change = 0.0
        maximum_frozen_coordinate_change = 0.0
        maximum_volatility_equation_residual = 0.0
        maximum_drift_equation_residual = 0.0
        maximum_density_diffusion_mass_residual = 0.0
        maximum_density_diffusion_forward_residual = 0.0
        minimum_density_during_sde = float("inf")
        minimum_raw_cap_margin = float("inf")
        spot_reveal_tower_residual: float | None = None
        generator = torch.Generator(device=torch.device(env.device)).manual_seed(
            config.seed
        )

        _synchronize(env.device)
        started = perf_counter()
        for start in range(0, config.num_paths_per_state, config.path_batch_size):
            stop = min(start + config.path_batch_size, config.num_paths_per_state)
            randoms = make_fixed_grid_attempt2_randoms(
                num_states=config.num_states,
                num_paths=stop - start,
                num_substeps=config.num_substeps,
                num_factors=loadings.shape[-1],
                dtype=env.fdtype,
                device=env.device,
                generator=generator,
                antithetic=config.antithetic,
            )
            rolled = fixed_grid_attempt2_sde_roll(
                model.sanos_decoder,
                initial_raw,
                raw_log_loadings=loadings,
                calendar_time_increment_y=(
                    config.calendar_business_days / DAYS_PER_YEAR
                ),
                randoms=FixedGridAttempt2Randoms(
                    surface_standard_normals=randoms.surface_standard_normals,
                    front_node_uniforms=randoms.front_node_uniforms,
                ),
                fixed_cash_strikes=fixed_cash_strikes,
            )
            aligned_options[:, start:stop] = (
                rolled.aligned_conditional_option_values.detach().cpu().double()
            )
            endpoint_survivor_options[:, start:stop] = (
                rolled.endpoint_unconditional_survivor_option_values
                .detach()
                .cpu()
                .double()
            )
            normalized_spot[:, start:stop] = (
                rolled.normalized_spot.detach().cpu().double()
            )
            initial_options = rolled.initial_option_values.detach().cpu().double()
            maximum_conditional_mass_residual = _maximum(
                maximum_conditional_mass_residual,
                rolled.maximum_conditional_mass_residual,
            )
            maximum_conditional_forward_residual = _maximum(
                maximum_conditional_forward_residual,
                rolled.maximum_conditional_forward_residual,
            )
            maximum_fixed_grid_change = _maximum(
                maximum_fixed_grid_change,
                rolled.maximum_fixed_grid_change,
            )
            maximum_frozen_coordinate_change = _maximum(
                maximum_frozen_coordinate_change,
                rolled.maximum_frozen_coordinate_change,
            )
            maximum_volatility_equation_residual = _maximum(
                maximum_volatility_equation_residual,
                rolled.maximum_volatility_equation_residual,
            )
            maximum_drift_equation_residual = _maximum(
                maximum_drift_equation_residual,
                rolled.maximum_drift_equation_residual,
            )
            maximum_density_diffusion_mass_residual = _maximum(
                maximum_density_diffusion_mass_residual,
                rolled.maximum_density_diffusion_mass_residual,
            )
            maximum_density_diffusion_forward_residual = _maximum(
                maximum_density_diffusion_forward_residual,
                rolled.maximum_density_diffusion_forward_residual,
            )
            minimum_density_during_sde = _minimum(
                minimum_density_during_sde,
                rolled.minimum_density_during_sde,
            )
            minimum_raw_cap_margin = _minimum(
                minimum_raw_cap_margin,
                rolled.minimum_raw_cap_margin,
            )
            if spot_reveal_tower_residual is None:
                spot_reveal_tower_residual = _spot_reveal_tower_residual(
                    model.sanos_decoder,
                    raw_states=rolled.endpoint_raw_states[:, 0],
                )
            print(f"paths {stop}/{config.num_paths_per_state}", flush=True)
        _synchronize(env.device)
        simulation_seconds = perf_counter() - started

    assert initial_options is not None
    assert spot_reveal_tower_residual is not None
    selected_dates = tuple(dates[int(value)] for value in state_indices)
    surface_changes = (
        endpoint_survivor_options.numpy()
        - initial_options[:, None, 1:].numpy()
    ).reshape(config.num_states, config.num_paths_per_state, -1)
    rolled_option_changes = (
        aligned_options.numpy() - initial_options[:, None].numpy()
    ).reshape(config.num_states, config.num_paths_per_state, -1)
    rolled_changes = np.concatenate(
        (
            normalized_spot.numpy()[..., None] - 1.0,
            rolled_option_changes,
        ),
        axis=-1,
    )
    surface_observations = _pair_mean_observations(
        surface_changes,
        antithetic=config.antithetic,
    )
    rolled_observations = _pair_mean_observations(
        rolled_changes,
        antithetic=config.antithetic,
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_kind": "fixed_grid_dlv_attempt_2_0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "config": asdict(config),
        "config_version": config.version.unique_id48,
        "device": str(env.device),
        "precision": arguments.precision,
        "model": {
            "factor": "X-only discrete DLV factor; no Black Y factor",
            "cash_grid": "unchanged across the calendar step",
            "surface_sde": (
                "Equation (15) in log excess-DLV coordinates, recomputed at "
                "every Euler substep; q_t^1 is the fixed boundary of this SDE"
            ),
            "spot_reveal": "S_{t+1} is sampled from q_t^1",
            "expiry_roll": "old DLVsq slabs 2,3,4 become new slabs 1,2,3",
            "far_expiry_boundary": (
                "not required or applied in this one-step audit of existing contracts"
            ),
            "option_martingale_scope": (
                "a call at every inner fixed-grid strike at each of the four old "
                "maturities; puts follow from call-put parity and the separately "
                "tested spot martingale"
            ),
            "option_coordinate_order": (
                "expiry-major then increasing strike; the full-roll audit prepends spot"
            ),
            "inner_fixed_cash_strikes": fixed_cash_strikes.detach().cpu().tolist(),
            "num_option_martingale_conditions_per_state": (
                geometry.num_model_expiries * num_audit_strikes
            ),
        },
        "num_source_states": len(latent_states),
        "source_state_indices": state_indices.tolist(),
        "source_state_dates": [value.isoformat() for value in selected_dates],
        "worst_cap_source_state_index": worst_cap_index,
        "sources": sources,
        "structural_diagnostics": {
            "maximum_spot_reveal_density_tower_residual": (
                spot_reveal_tower_residual
            ),
            "maximum_conditional_mass_residual": (
                maximum_conditional_mass_residual
            ),
            "maximum_conditional_forward_residual": (
                maximum_conditional_forward_residual
            ),
            "maximum_fixed_grid_change": maximum_fixed_grid_change,
            "maximum_frozen_coordinate_change": (
                maximum_frozen_coordinate_change
            ),
            "maximum_volatility_equation_residual": (
                maximum_volatility_equation_residual
            ),
            "maximum_drift_equation_residual": maximum_drift_equation_residual,
            "maximum_density_diffusion_mass_residual": (
                maximum_density_diffusion_mass_residual
            ),
            "maximum_density_diffusion_forward_residual": (
                maximum_density_diffusion_forward_residual
            ),
            "minimum_density_during_sde": minimum_density_during_sde,
            "minimum_raw_cap_margin": minimum_raw_cap_margin,
        },
        "martingale_audits": {
            "surface_sde_old_expiries_2_to_4": _statewise_zero_mean_audit(
                surface_observations,
                state_indices=state_indices,
                state_dates=selected_dates,
                correlation_eigenvalue_cutoff=(
                    config.correlation_eigenvalue_cutoff
                ),
                standard_error_floor_bps_spot=(
                    config.z_resolution_standard_error_floor_bps_spot
                ),
            ),
            "full_roll_spot_and_old_expiries_1_to_4": (
                _statewise_zero_mean_audit(
                    rolled_observations,
                    state_indices=state_indices,
                    state_dates=selected_dates,
                    correlation_eigenvalue_cutoff=(
                        config.correlation_eigenvalue_cutoff
                    ),
                    standard_error_floor_bps_spot=(
                        config.z_resolution_standard_error_floor_bps_spot
                    ),
                )
            ),
        },
        "speed": {
            "simulation_seconds": simulation_seconds,
            "path_state_steps_per_second": (
                config.num_states
                * config.num_paths_per_state
                * config.num_substeps
                / simulation_seconds
            ),
        },
    }
    write_json_atomic(report, arguments.output)
    for name, audit in report["martingale_audits"].items():
        print(
            f"{name}: max|z|={audit['maximum_absolute_coordinate_z']:.3f} "
            f"Fisher p={audit['fisher_combined_p_value']:.6g} "
            f"max mean={audit['maximum_absolute_mean_bps_spot']:.6g}bp",
            flush=True,
        )
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    """Command-line entry point."""

    run(_parser().parse_args())


if __name__ == "__main__":
    main()
