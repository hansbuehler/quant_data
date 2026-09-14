"""Audit 200 fixed-grid calls over four rolling four-tenor Attempt 2.0 steps."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.stats import t as student_t

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig
from quantdata.dysanos.flex.fixed_grid_dlv_attempt2_triangular import (
    TriangularFixedGridState,
    regrid_triangular_state_to_revealed_spot,
    triangular_fixed_grid_roll,
)
from quantdata.dysanos.flex.last_expiry_ar1 import (
    fit_last_expiry_ar1,
    interpolate_ar1_raw_curves,
    torch_last_expiry_ar1,
)
from quantdata.dysanos.flex.lvsq_drift import smooth_lvsq_log_loadings
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.run_fixed_grid_dlv_attempt2 import (
    _synchronize,
)
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


def _simultaneous_coordinate_audit(
    observations: np.ndarray, *, standard_error_floor_bps_spot: float
) -> dict[str, object]:
    """Bonferroni simultaneous test, valid without a covariance inverse."""

    sample_size = observations.shape[1]
    means = np.mean(observations, axis=1)
    standard_errors = np.std(observations, axis=1, ddof=1) / np.sqrt(sample_size)
    active = 10_000.0 * standard_errors >= standard_error_floor_bps_spot
    z = np.full_like(means, np.nan)
    z[active] = means[active] / standard_errors[active]
    maximum_z = float(np.nanmax(np.abs(z)))
    num_active = int(np.sum(active))
    degrees_of_freedom = sample_size - 1
    return {
        "method": (
            "two-sided Bonferroni max-|t| over all active state/option coordinates; "
            "each observation is an independent randomized-stratification block"
        ),
        "num_independent_antithetic_pair_observations_per_state": sample_size,
        "num_total_coordinates": int(means.size),
        "num_active_coordinates": num_active,
        "num_numerically_unresolved_coordinates": int(means.size - num_active),
        "standard_error_floor_bps_spot": standard_error_floor_bps_spot,
        "maximum_absolute_t": maximum_z,
        "familywise_p_value": float(
            min(1.0, 2.0 * student_t.sf(maximum_z, degrees_of_freedom) * num_active)
        ),
        "five_percent_familywise_t_cutoff": float(
            student_t.ppf(1.0 - 0.05 / (2.0 * num_active), degrees_of_freedom)
        ),
        "maximum_absolute_mean_bps_spot": float(10_000.0 * np.max(np.abs(means))),
        "maximum_unresolved_absolute_mean_bps_spot": float(
            10_000.0 * np.max(np.abs(means[~active])) if np.any(~active) else 0.0
        ),
        "coordinate_mean_bps_spot_by_state": (10_000.0 * means).tolist(),
        "coordinate_standard_error_bps_spot_by_state": (
            10_000.0 * standard_errors
        ).tolist(),
        "coordinate_t_by_state": z.tolist(),
    }


def _stratified_antithetic_uniforms(
    *,
    num_states: int,
    num_pairs: int,
    num_weeks: int,
    pairs_per_block: int,
    dtype: torch.dtype,
    device: torch.device | str,
    generator: torch.Generator,
    complement_pairs: bool = True,
) -> torch.Tensor:
    """Generate randomized blocks, optionally with adjacent complements."""

    if num_pairs % pairs_per_block:
        raise ValueError("num_pairs must be divisible by pairs_per_block")
    num_blocks = num_pairs // pairs_per_block
    points_per_block = pairs_per_block if complement_pairs else 2 * pairs_per_block
    random_order = torch.rand(
        num_states,
        num_blocks,
        num_weeks,
        points_per_block,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    ranks = torch.argsort(random_order, dim=-1)
    jitter = torch.rand(
        ranks.shape, dtype=dtype, device=device, generator=generator
    )
    base = ((ranks + jitter) / points_per_block).permute(0, 1, 3, 2).reshape(
        num_states,
        num_pairs if complement_pairs else 2 * num_pairs,
        num_weeks,
    )
    if not complement_pairs:
        return base
    maximum_uniform = torch.nextafter(
        torch.ones((), dtype=dtype, device=device),
        torch.zeros((), dtype=dtype, device=device),
    )
    return torch.stack(
        (base, torch.clamp(1.0 - base, max=maximum_uniform)), dim=2
    ).reshape(num_states, 2 * num_pairs, num_weeks)


def _stratified_block_mean_observations(
    values: np.ndarray, *, pairs_per_block: int
) -> np.ndarray:
    """Aggregate adjacent antithetic pairs into independent strata-block means."""

    num_pairs = values.shape[1] // 2
    if values.shape[1] % 2 or num_pairs % pairs_per_block:
        raise ValueError("paths must form complete antithetic stratification blocks")
    return values.reshape(
        values.shape[0],
        num_pairs // pairs_per_block,
        pairs_per_block,
        2,
        values.shape[2],
    ).mean(axis=(2, 3))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=64, choices=(32, 64))
    parser.add_argument("--num-states", type=int, default=21)
    parser.add_argument("--state-indices", type=str)
    parser.add_argument("--num-paths-per-state", type=int, default=10_000)
    parser.add_argument("--path-batch-size", type=int, default=500)
    parser.add_argument("--num-substeps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2_026_091_3)
    parser.add_argument("--regrid", action="store_true")
    parser.add_argument("--stratification-pairs-per-block", type=int, default=50)
    parser.add_argument("--antithetic-spot", action="store_true")
    parser.add_argument("--iid-spot", action="store_true")
    return parser


def run(arguments: argparse.Namespace) -> None:
    """Run four weekly transitions with an AR(1)-generated far expiry."""

    env = BaseEnvironment(precision=arguments.precision, device=arguments.device, debug=False)
    model, dates, latent_states, sources = load_static_q_inputs(
        arguments.static_decoder_checkpoint, arguments.static_report, env=env
    )
    geometry = model.config.geometry
    if not isinstance(geometry, FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("the triangular audit requires the fitted fixed grid")
    requested_indices = (
        tuple(int(value) for value in arguments.state_indices.split(","))
        if arguments.state_indices
        else None
    )
    num_states = len(requested_indices) if requested_indices is not None else arguments.num_states
    if num_states > len(latent_states):
        raise ValueError("num-states exceeds the static state table")
    state_indices = (
        np.asarray(requested_indices, dtype=np.int64)
        if requested_indices is not None
        else np.linspace(0, len(latent_states) - 1, num_states, dtype=np.int64)
    )
    index = torch.as_tensor(state_indices, device=env.device)
    with torch.no_grad():
        raw = model.raw_states_from_states(latent_states)[index]
        parameters = bound_native_raw(raw, geometry=geometry)
        surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        base_support = surface.support_strikes[:, 0]
        strikes = torch.as_tensor(
            geometry.inner_relative_strikes, dtype=env.fdtype, device=env.device
        )
        base_loadings = smooth_lvsq_log_loadings(
            base_support[0],
            num_model_expiries=geometry.num_model_expiries,
            annualized_log_volatility=0.10,
        )
        historical_last_raw = (
            model.raw_states_from_states(latent_states)
            .reshape(len(latent_states), geometry.num_model_expiries, -1)[
                :, -1, : geometry.num_inner_strikes
            ]
            .cpu()
            .double()
            .numpy()
        )
        boundary_fit = fit_last_expiry_ar1(
            historical_last_raw,
            dates,
            np.asarray(geometry.inner_relative_strikes),
            rank=25,
            training_end=date(2024, 12, 31),
        )
        boundary_model = torch_last_expiry_ar1(
            boundary_fit, dtype=env.fdtype, device=env.device
        )
        changes = [
            torch.empty(
                num_states,
                arguments.num_paths_per_state,
                geometry.num_model_expiries * geometry.num_inner_strikes,
                dtype=torch.float64,
            )
            for week in range(geometry.num_model_expiries)
        ]
        integrated_changes = [torch.empty_like(value) for value in changes]
        spot_changes = [
            torch.empty(num_states, arguments.num_paths_per_state, dtype=torch.float64)
            for _ in range(geometry.num_model_expiries)
        ]
        maximum_mass_residual = 0.0
        maximum_forward_residual = 0.0
        minimum_density = float("inf")
        generator = torch.Generator(device=env.device).manual_seed(arguments.seed)
        _synchronize(env.device)
        started = perf_counter()
        for start in range(0, arguments.num_paths_per_state, arguments.path_batch_size):
            stop = min(start + arguments.path_batch_size, arguments.num_paths_per_state)
            paths = stop - start
            if paths % 2:
                raise ValueError("every path batch must contain adjacent antithetic pairs")
            draws = paths // 2
            normals_half = torch.randn(
                num_states, draws, 4, arguments.num_substeps, 4,
                dtype=env.fdtype, device=env.device, generator=generator,
            )
            normals = torch.stack((normals_half, -normals_half), 2).reshape(
                num_states, paths, 4, arguments.num_substeps, 4
            )
            uniforms = (
                torch.rand(
                    num_states,
                    paths,
                    4,
                    dtype=env.fdtype,
                    device=env.device,
                    generator=generator,
                )
                if arguments.iid_spot
                else _stratified_antithetic_uniforms(
                    num_states=num_states,
                    num_pairs=draws,
                    num_weeks=4,
                    pairs_per_block=arguments.stratification_pairs_per_block,
                    dtype=env.fdtype,
                    device=env.device,
                    generator=generator,
                    complement_pairs=arguments.antithetic_spot,
                )
            )
            batch = num_states * paths
            ar_half = torch.randn(
                num_states, draws, 4, boundary_model.rank,
                dtype=env.fdtype, device=env.device, generator=generator,
            )
            ar_normals = torch.stack((ar_half, -ar_half), 2).reshape(
                num_states, paths, 4, boundary_model.rank
            )
            boundary_scores = boundary_model.project(
                raw.reshape(num_states, geometry.num_model_expiries, -1)[
                    :, -1, : geometry.num_inner_strikes
                ]
            )[:, None].expand(-1, paths, -1).reshape(batch, boundary_model.rank)
            state = TriangularFixedGridState(
                support=base_support[:, None].expand(-1, paths, -1).reshape(batch, -1),
                densities=surface.densities[:, None].expand(-1, paths, -1, -1).reshape(
                    batch, geometry.num_model_expiries, geometry.num_inner_strikes + 2
                ),
                dlvsqs=parameters.dlvsqs[:, None].expand(-1, paths, -1, -1).reshape(
                    batch, geometry.num_model_expiries, geometry.num_inner_strikes
                ),
            )
            current_spot = torch.ones(batch, dtype=env.fdtype, device=env.device)
            for week in range(geometry.num_model_expiries):
                roll = triangular_fixed_grid_roll(
                    model.sanos_decoder,
                    state,
                    raw_log_loadings=base_loadings,
                    minimum_dlvsq=geometry.min_dlvsq,
                    expiry_time_increment_y=model.sanos_decoder.model_dts_y[0],
                    calendar_time_increment_y=5.0 / DAYS_PER_YEAR,
                    standard_normals=normals[:, :, week].reshape(
                        batch, arguments.num_substeps, 4
                    ),
                    uniforms=uniforms[:, :, week].reshape(batch),
                    fixed_cash_strikes=(
                        state.support[:, 1:-1] if arguments.regrid else strikes
                    ),
                )
                scale = current_spot[:, None, None] if arguments.regrid else 1.0
                direct_option_change = (
                    roll.final_call_values - roll.initial_call_values
                ) * scale
                integrated_option_change = (
                    roll.endpoint_unconditional_call_values
                    - roll.initial_call_values
                ) * scale
                direct_option_change = direct_option_change.reshape(
                    num_states,
                    paths,
                    geometry.num_model_expiries * geometry.num_inner_strikes,
                )
                integrated_option_change = integrated_option_change.reshape_as(
                    direct_option_change
                )
                changes[week][:, start:stop] = direct_option_change.cpu().double()
                integrated_changes[week][:, start:stop] = (
                    integrated_option_change.cpu().double()
                )
                next_spot = (
                    current_spot * roll.revealed_spot
                    if arguments.regrid
                    else roll.revealed_spot
                )
                spot_changes[week][:, start:stop] = (next_spot - current_spot).reshape(
                    num_states, paths
                ).cpu().double()
                maximum_mass_residual = max(maximum_mass_residual, float(roll.maximum_mass_residual))
                maximum_forward_residual = max(
                    maximum_forward_residual, float(roll.maximum_forward_residual)
                )
                minimum_density = min(minimum_density, float(roll.minimum_density))
                assert roll.next_state is not None
                boundary_scores, boundary_raw = boundary_model.step(
                    boundary_scores,
                    ar_normals[:, :, week].reshape(batch, boundary_model.rank),
                )
                boundary_raw, _ = boundary_model.clamp_to_training_envelope(boundary_raw)
                next_state = (
                    regrid_triangular_state_to_revealed_spot(
                        roll.next_state, roll.revealed_spot
                    )
                    if arguments.regrid
                    else roll.next_state
                )
                if arguments.regrid:
                    boundary_raw, _ = interpolate_ar1_raw_curves(
                        boundary_model, boundary_raw, next_state.support[:, 1:-1]
                    )
                boundary_dlvsq = geometry.min_dlvsq + torch.exp(boundary_raw)
                far_density = model.sanos_decoder.solve_density_step_batch(
                    support=next_state.support,
                    dlvsqs=boundary_dlvsq,
                    time_increment_y=model.sanos_decoder.model_dts_y[0],
                    rhs_density=next_state.densities[:, -1],
                )
                state = TriangularFixedGridState(
                    support=next_state.support,
                    densities=torch.cat(
                        (next_state.densities, far_density[:, None]), dim=1
                    ),
                    dlvsqs=torch.cat(
                        (next_state.dlvsqs, boundary_dlvsq[:, None]), dim=1
                    ),
                )
                current_spot = next_spot
            print(f"paths {stop}/{arguments.num_paths_per_state}", flush=True)
        _synchronize(env.device)
        simulation_seconds = perf_counter() - started

    selected_dates = tuple(dates[int(value)] for value in state_indices)
    pairs_per_block = arguments.stratification_pairs_per_block
    direct_blocks = [
        _stratified_block_mean_observations(
            value.numpy(), pairs_per_block=pairs_per_block
        )
        for value in changes
    ]
    integrated_blocks = [
        _stratified_block_mean_observations(
            value.numpy(), pairs_per_block=pairs_per_block
        )
        for value in integrated_changes
    ]
    spot_blocks = [
        _stratified_block_mean_observations(
            value.numpy()[..., None], pairs_per_block=pairs_per_block
        )
        for value in spot_changes
    ]
    direct_combined = np.concatenate(direct_blocks, axis=2)
    integrated_combined = np.concatenate(integrated_blocks, axis=2)
    reveal_error_combined = direct_combined - integrated_combined
    audits: dict[str, object] = {}
    audits["direct_sampled_option_by_week"] = [
        {
            "week": week + 1,
            **_simultaneous_coordinate_audit(
                direct_blocks[week], standard_error_floor_bps_spot=1.0e-6
            ),
        }
        for week in range(4)
    ]
    audits["integrated_control_option_by_week"] = [
        {
            "week": week + 1,
            **_simultaneous_coordinate_audit(
                integrated_blocks[week], standard_error_floor_bps_spot=1.0e-6
            ),
        }
        for week in range(4)
    ]
    audits["direct_sampled_spot_by_week"] = [
        {
            "week": week + 1,
            **_simultaneous_coordinate_audit(
                spot_blocks[week], standard_error_floor_bps_spot=1.0e-6
            ),
        }
        for week in range(4)
    ]
    audits["direct_sampled_all_3200_option_conditions"] = (
        _simultaneous_coordinate_audit(
            direct_combined,
            standard_error_floor_bps_spot=1.0e-6,
        )
    )
    audits["integrated_control_all_3200_option_conditions"] = (
        _simultaneous_coordinate_audit(
            integrated_combined,
            standard_error_floor_bps_spot=1.0e-6,
        )
    )
    audits["direct_minus_integrated_reveal_error"] = _simultaneous_coordinate_audit(
        reveal_error_combined,
        standard_error_floor_bps_spot=1.0e-6,
    )
    report = {
        "schema_version": 1,
        "experiment_kind": "attempt_2_0_rolling_four_tenor_ar1_200_strikes",
        "regridded_to_revealed_spot_after_each_week": arguments.regrid,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_state_indices": state_indices.tolist(),
        "source_state_dates": [value.isoformat() for value in selected_dates],
        "sources": sources,
        "num_paths_per_state": arguments.num_paths_per_state,
        "sde_integrator": "stochastic Heun in log excess-DLV; additive diffusion",
        "num_sde_substeps_per_week": arguments.num_substeps,
        "num_independent_antithetic_pairs_per_state": arguments.num_paths_per_state // 2,
        "stratification_pairs_per_independent_block": pairs_per_block,
        "spot_uniforms_antithetic_within_surface_pair": arguments.antithetic_spot,
        "spot_uniform_sampling": (
            "iid" if arguments.iid_spot else "randomized_stratified"
        ),
        "num_independent_stratification_blocks_per_state": (
            arguments.num_paths_per_state // (2 * pairs_per_block)
        ),
        "option_conditions_by_week": [800, 800, 800, 800],
        "total_option_conditions_per_state": 3200,
        "floating_four_week_boundary_ar1": boundary_fit.diagnostics.to_dict(),
        "fixed_cash_strikes": strikes.cpu().tolist(),
        "martingale_audits": audits,
        "numerical": {
            "maximum_mass_residual": maximum_mass_residual,
            "maximum_forward_residual": maximum_forward_residual,
            "minimum_density": minimum_density,
        },
        "speed": {"simulation_seconds": simulation_seconds},
    }
    write_json_atomic(report, arguments.output)
    for name, audit in audits.items():
        if isinstance(audit, list):
            continue
        if "3200_option_conditions" in name or name == "direct_minus_integrated_reveal_error":
            print(
                f"{name}: max|t|={audit['maximum_absolute_t']:.3f} "
                f"familywise p={audit['familywise_p_value']:.6g} "
                f"max mean={audit['maximum_absolute_mean_bps_spot']:.6g}bp",
                flush=True,
            )
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
