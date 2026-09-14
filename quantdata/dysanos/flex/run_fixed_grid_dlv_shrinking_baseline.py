"""Audit an X-only analytic DLVsq tower on one fixed cash-strike grid."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.stats import chi2
from scipy.stats import t as student_t

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.fixed_expiry_ar1 import (
    fit_fixed_expiry_ar1,
    torch_fixed_expiry_raw_log_loadings,
)
from quantdata.dysanos.flex.fixed_grid_dlv_attempt2_triangular import (
    TriangularFixedGridState,
    triangular_fixed_grid_roll,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.run_fixed_grid_dlv_attempt2 import _synchronize
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=64, choices=(32, 64))
    parser.add_argument("--state-index", type=int, default=44)
    parser.add_argument("--num-paths", type=int, default=100_000)
    parser.add_argument("--path-batch-size", type=int, default=1_000)
    parser.add_argument("--antithetic-pairs-per-block", type=int, default=50)
    parser.add_argument("--num-substeps", type=int, default=10)
    parser.add_argument("--innovation-rank", type=int, default=25)
    parser.add_argument("--correlation-eigenvalue-cutoff", type=float, default=0.01)
    parser.add_argument("--multiplier-bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2_026_091_6)
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    paths_per_block = 2 * arguments.antithetic_pairs_per_block
    if arguments.num_paths < 4 or arguments.num_paths % paths_per_block:
        raise ValueError("num-paths must be divisible by a full antithetic block")
    if arguments.path_batch_size <= 0 or arguments.path_batch_size % paths_per_block:
        raise ValueError("path-batch-size must be divisible by a full antithetic block")
    if arguments.num_paths % arguments.path_batch_size:
        raise ValueError("num-paths must be divisible by path-batch-size")
    if arguments.num_substeps <= 0 or arguments.innovation_rank <= 0:
        raise ValueError("num-substeps and innovation-rank must be positive")
    if not 0.0 < arguments.correlation_eigenvalue_cutoff < 1.0:
        raise ValueError("correlation-eigenvalue-cutoff must lie in (0, 1)")
    if arguments.multiplier_bootstrap_samples <= 0:
        raise ValueError("multiplier-bootstrap-samples must be positive")


def _randomized_stratified_antithetic_uniforms(
    *,
    num_paths: int,
    num_weeks: int,
    pairs_per_block: int,
    dtype: torch.dtype,
    device: torch.device | str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate independent randomized-Latin blocks with paired complements."""

    num_pairs = num_paths // 2
    if num_paths % 2 or num_pairs % pairs_per_block:
        raise ValueError("paths must form complete antithetic stratification blocks")
    num_blocks = num_pairs // pairs_per_block
    random_order = torch.rand(
        num_blocks,
        num_weeks,
        pairs_per_block,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    ranks = torch.argsort(random_order, dim=-1)
    jitter = torch.rand(
        ranks.shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    base = ((ranks + jitter) / pairs_per_block).permute(0, 2, 1).reshape(num_pairs, num_weeks)
    maximum_uniform = torch.nextafter(
        torch.ones((), dtype=dtype, device=device),
        torch.zeros((), dtype=dtype, device=device),
    )
    return torch.stack((base, torch.clamp(1.0 - base, max=maximum_uniform)), dim=1).reshape(
        num_paths, num_weeks
    )


def _block_means(values: torch.Tensor, *, pairs_per_block: int) -> np.ndarray:
    """Reduce adjacent antithetic paths to independent stratification blocks."""

    num_paths, num_coordinates = values.shape
    paths_per_block = 2 * pairs_per_block
    if num_paths % paths_per_block:
        raise ValueError("values do not contain complete antithetic blocks")
    return (
        values.reshape(-1, pairs_per_block, 2, num_coordinates)
        .mean(dim=(1, 2))
        .cpu()
        .double()
        .numpy()
    )


def _spectral_wald_with_reference_covariance(
    covariance_observations: np.ndarray,
    mean_observations: np.ndarray,
    *,
    correlation_eigenvalue_cutoff: float,
) -> dict[str, float | int]:
    """Test a mean in covariance modes estimated from a reference sample."""

    standard_deviation = np.std(covariance_observations, axis=0, ddof=1)
    active = standard_deviation > max(
        1.0e-14,
        1.0e-10 * float(np.max(standard_deviation)),
    )
    reference = covariance_observations[:, active] / standard_deviation[active]
    tested = mean_observations[:, active] / standard_deviation[active]
    centered = reference - np.mean(reference, axis=0)
    scaled = centered / np.sqrt(len(reference) - 1)
    _, singular_values, right_vectors = np.linalg.svd(
        scaled,
        full_matrices=False,
    )
    eigenvalues = np.square(singular_values)
    retained = eigenvalues > correlation_eigenvalue_cutoff
    if not bool(np.any(retained)):
        raise RuntimeError("covariance-aware audit retained no directions")
    projected_mean = right_vectors[retained] @ np.mean(tested, axis=0)
    statistic = float(len(tested) * np.sum(np.square(projected_mean) / eigenvalues[retained]))
    degrees_of_freedom = int(np.sum(retained))
    return {
        "num_covariance_observations": len(reference),
        "num_mean_observations": len(tested),
        "num_active_coordinates": int(np.sum(active)),
        "degrees_of_freedom": degrees_of_freedom,
        "statistic": statistic,
        "statistic_per_degree_of_freedom": statistic / degrees_of_freedom,
        "p_value": float(chi2.sf(statistic, degrees_of_freedom)),
        "minimum_retained_correlation_eigenvalue": float(np.min(eigenvalues[retained])),
    }


def _zero_mean_audit(
    observations: np.ndarray,
    *,
    correlation_eigenvalue_cutoff: float,
    multiplier_bootstrap_samples: int,
    multiplier_bootstrap_seed: int,
) -> dict[str, object]:
    """Report coordinatewise and covariance-aware zero-mean tests."""

    num_blocks, num_coordinates = observations.shape
    means = np.mean(observations, axis=0)
    standard_deviations = np.std(observations, axis=0, ddof=1)
    active = standard_deviations > max(
        1.0e-14,
        1.0e-10 * float(np.max(standard_deviations)),
    )
    standard_errors = standard_deviations / np.sqrt(num_blocks)
    t_statistics = np.full(num_coordinates, np.nan)
    t_statistics[active] = means[active] / standard_errors[active]
    maximum_absolute_t = float(np.nanmax(np.abs(t_statistics)))
    root_mean_square_t = float(np.sqrt(np.nanmean(np.square(t_statistics))))
    num_active = int(np.sum(active))
    degrees_of_freedom = num_blocks - 1
    even = observations[0::2]
    odd = observations[1::2]
    centered = observations[:, active] - means[active]
    generator = np.random.default_rng(multiplier_bootstrap_seed)
    bootstrap_maximum: list[np.ndarray] = []
    bootstrap_root_mean_square: list[np.ndarray] = []
    remaining = multiplier_bootstrap_samples
    while remaining:
        count = min(128, remaining)
        signs = generator.choice((-1.0, 1.0), size=(count, num_blocks))
        bootstrap_t = signs @ centered / (
            np.sqrt(num_blocks) * standard_deviations[active]
        )
        bootstrap_maximum.append(np.max(np.abs(bootstrap_t), axis=1))
        bootstrap_root_mean_square.append(
            np.sqrt(np.mean(np.square(bootstrap_t), axis=1))
        )
        remaining -= count
    bootstrap_maximum_values = np.concatenate(bootstrap_maximum)
    bootstrap_rms_values = np.concatenate(bootstrap_root_mean_square)
    maximum_p_value = float(
        (1 + np.sum(bootstrap_maximum_values >= maximum_absolute_t))
        / (multiplier_bootstrap_samples + 1)
    )
    rms_p_value = float(
        (1 + np.sum(bootstrap_rms_values >= root_mean_square_t))
        / (multiplier_bootstrap_samples + 1)
    )
    combined_p_value = min(1.0, 2.0 * min(maximum_p_value, rms_p_value))
    return {
        "num_independent_blocks": num_blocks,
        "num_coordinates": num_coordinates,
        "num_active_coordinates": num_active,
        "num_numerically_unresolved_coordinates": num_coordinates - num_active,
        "maximum_absolute_t": maximum_absolute_t,
        "root_mean_square_t": root_mean_square_t,
        "num_absolute_t_above_2": int(np.sum(np.abs(t_statistics) > 2.0)),
        "bonferroni_familywise_p_value": float(
            min(
                1.0,
                2.0 * student_t.sf(maximum_absolute_t, degrees_of_freedom) * num_active,
            )
        ),
        "five_percent_bonferroni_t_cutoff": float(
            student_t.ppf(
                1.0 - 0.05 / (2.0 * num_active),
                degrees_of_freedom,
            )
        ),
        "maximum_absolute_mean": float(np.max(np.abs(means))),
        "coordinate_mean": means.tolist(),
        "coordinate_standard_error": standard_errors.tolist(),
        "coordinate_t": t_statistics.tolist(),
        "correlation_aware_multiplier_test": {
            "method": (
                "Rademacher shared-sign multiplier bootstrap over centered "
                "independent block means; shared signs retain all cross-option "
                "dependence. The combined p-value Bonferroni-combines the "
                "maximum-|t| and RMS-t statistics."
            ),
            "samples": multiplier_bootstrap_samples,
            "seed": multiplier_bootstrap_seed,
            "maximum_absolute_t_p_value": maximum_p_value,
            "root_mean_square_t_p_value": rms_p_value,
            "combined_p_value": combined_p_value,
            "zero_mean_not_rejected_at_five_percent": combined_p_value >= 0.05,
            "maximum_absolute_t_95_percent_cutoff": float(
                np.quantile(bootstrap_maximum_values, 0.95)
            ),
            "root_mean_square_t_95_percent_cutoff": float(
                np.quantile(bootstrap_rms_values, 0.95)
            ),
        },
        "spectral_wald_chi_square_diagnostic": {
            "warning": (
                "Not used as the decision test: chi-square calibration can be "
                "anti-conservative when retained rank is large relative to "
                "the number of covariance observations."
            ),
            "even_covariance_odd_mean": _spectral_wald_with_reference_covariance(
                even,
                odd,
                correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
            ),
            "odd_covariance_even_mean": _spectral_wald_with_reference_covariance(
                odd,
                even,
                correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
            ),
        },
    }


def _coordinate_layout(num_strikes: int) -> list[dict[str, int]]:
    """Describe the expiry-major coordinates of a four-week shrinking tower."""

    result: list[dict[str, int]] = []
    for trade_week in range(4):
        for weeks_to_expiry in range(1, 5 - trade_week):
            for strike_index in range(num_strikes):
                result.append(
                    {
                        "trade_week": trade_week,
                        "weeks_to_expiry": weeks_to_expiry,
                        "cash_expiry_week": trade_week + weeks_to_expiry,
                        "strike_index": strike_index,
                    }
                )
    return result


def run(arguments: argparse.Namespace) -> None:
    """Run and audit the requested fixed-grid shrinking tower."""

    _validate_arguments(arguments)
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
    if not isinstance(geometry, FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig):
        raise ValueError("the shrinking baseline requires the fitted fixed grid")
    if not 0 <= arguments.state_index < len(latent_states):
        raise ValueError("state-index lies outside the static state table")

    calendar_time_increment_y = 5.0 / DAYS_PER_YEAR
    with torch.no_grad():
        all_raw = model.raw_states_from_states(latent_states)
        fixed_expiry_fit = fit_fixed_expiry_ar1(
            all_raw.cpu().double().numpy(),
            dates,
            num_inner_strikes=geometry.num_inner_strikes,
            rank=arguments.innovation_rank,
            training_end=date(2024, 12, 31),
            calendar_time_increment_y=calendar_time_increment_y,
        )
        raw_log_loadings = torch_fixed_expiry_raw_log_loadings(
            fixed_expiry_fit,
            dtype=env.fdtype,
            device=env.device,
        )
        selected_raw = all_raw[arguments.state_index : arguments.state_index + 1]
        parameters = bound_native_raw(selected_raw, geometry=geometry)
        surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        base_support = surface.support_strikes[:, 0]
        fixed_cash_strikes = torch.as_tensor(
            geometry.inner_relative_strikes,
            dtype=env.fdtype,
            device=env.device,
        )

        generator = torch.Generator(device=env.device).manual_seed(arguments.seed)
        terminal_return_blocks: list[np.ndarray] = []
        terminal_price_change_blocks: list[np.ndarray] = []
        local_return_blocks: list[list[np.ndarray]] = [[] for _ in range(4)]
        local_integrated_return_blocks: list[list[np.ndarray]] = [[] for _ in range(4)]
        spot_return_blocks: list[list[np.ndarray]] = [[] for _ in range(4)]
        maximum_mass_residual = 0.0
        maximum_forward_residual = 0.0
        minimum_density = float("inf")
        _synchronize(env.device)
        started = perf_counter()

        for start in range(0, arguments.num_paths, arguments.path_batch_size):
            paths = arguments.path_batch_size
            pairs = paths // 2
            normal_half = torch.randn(
                pairs,
                4,
                arguments.num_substeps,
                arguments.innovation_rank,
                dtype=env.fdtype,
                device=env.device,
                generator=generator,
            )
            normals = torch.stack((normal_half, -normal_half), dim=1).reshape(
                paths,
                4,
                arguments.num_substeps,
                arguments.innovation_rank,
            )
            uniforms = _randomized_stratified_antithetic_uniforms(
                num_paths=paths,
                num_weeks=4,
                pairs_per_block=arguments.antithetic_pairs_per_block,
                dtype=env.fdtype,
                device=env.device,
                generator=generator,
            )
            state = TriangularFixedGridState(
                support=base_support.expand(paths, -1),
                densities=surface.densities.expand(paths, -1, -1),
                dlvsqs=parameters.dlvsqs.expand(paths, -1, -1),
            )
            current_spot = torch.ones(paths, dtype=env.fdtype, device=env.device)
            spots: list[torch.Tensor] = []
            trade_prices: list[torch.Tensor] = []
            trade_signs: list[torch.Tensor] = []

            for trade_week in range(4):
                num_live_expiries = 4 - trade_week
                option_signs = torch.where(
                    fixed_cash_strikes[None] <= current_spot[:, None],
                    1.0,
                    -1.0,
                )
                roll = triangular_fixed_grid_roll(
                    model.sanos_decoder,
                    state,
                    raw_log_loadings=raw_log_loadings[:num_live_expiries],
                    minimum_dlvsq=geometry.min_dlvsq,
                    expiry_time_increment_y=model.sanos_decoder.model_dts_y[0],
                    calendar_time_increment_y=calendar_time_increment_y,
                    standard_normals=normals[:, trade_week],
                    uniforms=uniforms[:, trade_week],
                    fixed_cash_strikes=fixed_cash_strikes,
                    option_signs=option_signs,
                )
                initial_prices = roll.initial_call_values
                direct_local_returns = (roll.final_call_values - initial_prices) / initial_prices
                integrated_local_returns = (
                    roll.endpoint_unconditional_call_values - initial_prices
                ) / initial_prices
                local_return_blocks[trade_week].append(
                    _block_means(
                        direct_local_returns.reshape(paths, -1),
                        pairs_per_block=arguments.antithetic_pairs_per_block,
                    )
                )
                local_integrated_return_blocks[trade_week].append(
                    _block_means(
                        integrated_local_returns.reshape(paths, -1),
                        pairs_per_block=arguments.antithetic_pairs_per_block,
                    )
                )
                next_spot = roll.revealed_spot
                spot_return_blocks[trade_week].append(
                    _block_means(
                        (next_spot / current_spot - 1.0)[:, None],
                        pairs_per_block=arguments.antithetic_pairs_per_block,
                    )
                )
                trade_prices.append(initial_prices)
                trade_signs.append(option_signs)
                spots.append(next_spot)
                maximum_mass_residual = max(
                    maximum_mass_residual,
                    float(roll.maximum_mass_residual),
                )
                maximum_forward_residual = max(
                    maximum_forward_residual,
                    float(roll.maximum_forward_residual),
                )
                minimum_density = min(minimum_density, float(roll.minimum_density))
                current_spot = next_spot
                if trade_week < 3:
                    if roll.next_state is None:
                        raise RuntimeError("shrinking tower ended before its fourth expiry")
                    state = roll.next_state
                elif roll.next_state is not None:
                    raise RuntimeError("shrinking tower retained a state after final expiry")

            terminal_returns: list[torch.Tensor] = []
            terminal_price_changes: list[torch.Tensor] = []
            for trade_week, (prices, signs) in enumerate(
                zip(trade_prices, trade_signs, strict=True)
            ):
                for expiry_index in range(prices.shape[1]):
                    terminal_spot = spots[trade_week + expiry_index]
                    payoff = torch.relu(signs * (terminal_spot[:, None] - fixed_cash_strikes[None]))
                    price = prices[:, expiry_index]
                    terminal_returns.append(payoff / price - 1.0)
                    terminal_price_changes.append(payoff - price)
            terminal_return_blocks.append(
                _block_means(
                    torch.cat(terminal_returns, dim=1),
                    pairs_per_block=arguments.antithetic_pairs_per_block,
                )
            )
            terminal_price_change_blocks.append(
                _block_means(
                    torch.cat(terminal_price_changes, dim=1),
                    pairs_per_block=arguments.antithetic_pairs_per_block,
                )
            )
            print(f"paths {start + paths}/{arguments.num_paths}", flush=True)

        _synchronize(env.device)
        simulation_seconds = perf_counter() - started

    local_returns = np.concatenate(
        [np.concatenate(values, axis=0) for values in local_return_blocks],
        axis=1,
    )
    local_integrated_returns = np.concatenate(
        [np.concatenate(values, axis=0) for values in local_integrated_return_blocks],
        axis=1,
    )
    terminal_returns = np.concatenate(terminal_return_blocks, axis=0)
    terminal_price_changes = np.concatenate(terminal_price_change_blocks, axis=0)
    spot_returns = np.concatenate(
        [np.concatenate(values, axis=0) for values in spot_return_blocks],
        axis=1,
    )
    cutoff = arguments.correlation_eigenvalue_cutoff
    terminal_return_audit = _zero_mean_audit(
        terminal_returns,
        correlation_eigenvalue_cutoff=cutoff,
        multiplier_bootstrap_samples=arguments.multiplier_bootstrap_samples,
        multiplier_bootstrap_seed=arguments.seed + 101,
    )
    terminal_price_change_audit = _zero_mean_audit(
        terminal_price_changes,
        correlation_eigenvalue_cutoff=cutoff,
        multiplier_bootstrap_samples=arguments.multiplier_bootstrap_samples,
        multiplier_bootstrap_seed=arguments.seed + 102,
    )
    local_return_audit = _zero_mean_audit(
        local_returns,
        correlation_eigenvalue_cutoff=cutoff,
        multiplier_bootstrap_samples=arguments.multiplier_bootstrap_samples,
        multiplier_bootstrap_seed=arguments.seed + 103,
    )
    local_integrated_return_audit = _zero_mean_audit(
        local_integrated_returns,
        correlation_eigenvalue_cutoff=cutoff,
        multiplier_bootstrap_samples=arguments.multiplier_bootstrap_samples,
        multiplier_bootstrap_seed=arguments.seed + 104,
    )
    spot_return_audit = _zero_mean_audit(
        spot_returns,
        correlation_eigenvalue_cutoff=cutoff,
        multiplier_bootstrap_samples=arguments.multiplier_bootstrap_samples,
        multiplier_bootstrap_seed=arguments.seed + 105,
    )
    report = {
        "schema_version": 1,
        "experiment_kind": "x_only_analytic_dlvsq_fixed_cash_shrinking_tower",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_state_index": arguments.state_index,
        "source_state_date": dates[arguments.state_index].isoformat(),
        "sources": sources,
        "fixed_cash_grid_no_regridding": True,
        "tower_expiries_by_trade_week": [4, 3, 2, 1],
        "option_conditions_by_trade_week": [800, 600, 400, 200],
        "total_option_conditions": 2_000,
        "num_paths": arguments.num_paths,
        "num_independent_antithetic_pairs": arguments.num_paths // 2,
        "num_independent_stratification_blocks": (
            arguments.num_paths // (2 * arguments.antithetic_pairs_per_block)
        ),
        "antithetic_pairs_per_block": arguments.antithetic_pairs_per_block,
        "spot_uniforms": "randomized-stratified with antithetic complements",
        "surface_normals": "iid across factors/weeks and antithetic within pairs",
        "multiplier_bootstrap_samples": arguments.multiplier_bootstrap_samples,
        "sde_integrator": "stochastic Heun in log excess-DLV",
        "num_sde_substeps_per_week": arguments.num_substeps,
        "fixed_expiry_ar1": fixed_expiry_fit.diagnostics.to_dict(),
        "coordinate_layout": _coordinate_layout(geometry.num_inner_strikes),
        "fixed_cash_strikes": fixed_cash_strikes.cpu().tolist(),
        "martingale_audits": {
            "terminal_itm_option_returns": terminal_return_audit,
            "terminal_itm_option_price_changes": terminal_price_change_audit,
            "one_step_mark_to_market_itm_option_returns": local_return_audit,
            "one_step_integrated_dlv_drift_control": local_integrated_return_audit,
            "weekly_spot_returns": spot_return_audit,
        },
        "numerical": {
            "minimum_density": minimum_density,
            "maximum_mass_residual": maximum_mass_residual,
            "maximum_forward_residual": maximum_forward_residual,
        },
        "speed": {
            "simulation_seconds": simulation_seconds,
            "paths_per_second": arguments.num_paths / simulation_seconds,
        },
    }
    write_json_atomic(report, arguments.output)
    for name, audit in report["martingale_audits"].items():
        print(
            f"{name}: max|t|={audit['maximum_absolute_t']:.4g}, "
            f"Bonferroni p={audit['bonferroni_familywise_p_value']:.4g}",
            flush=True,
        )
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
