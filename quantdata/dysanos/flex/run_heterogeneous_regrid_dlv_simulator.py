"""Audit four DLV rolls on 10,000 heterogeneous antithetic paths."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.fixed_grid_dlv import (
    build_fixed_grid_dlv_kernel,
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
from quantdata.dysanos.flex.run_multistep_regrid_dlv_simulator import (
    _factor_covariance_audit,
    _factor_scores,
    _intrinsic_values,
    _synchronize,
    _wald_test,
)
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.torch import BaseEnvironment


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class HeterogeneousRegridDlvConfig:
    """Random-origin, adjacent-antithetic four-week specification."""

    num_paths: int = 10_000
    num_steps: int = 4
    num_state_features: int = 4
    calendar_business_days: int = 5
    dlvsq_annualized_log_volatility: float = 0.10
    bs_backbone_annualized_volatility: float = 0.10
    num_option_strikes: int = 11
    correlation_eigenvalue_cutoff: float = 0.01
    seed: int = 2_026_091_0

    def __post_init__(self) -> None:
        verify_inp(
            self.num_paths >= 4 and self.num_paths % 2 == 0,
            lambda: "num_paths must be even and at least four",
        )
        verify_inp(
            self.num_steps == 4,
            lambda: "the current static model has exactly four weekly steps",
        )
        verify_inp(
            self.num_state_features > 0,
            lambda: "num_state_features must be positive",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-comparison-report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=64, choices=(32, 64))
    parser.add_argument("--num-paths", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2_026_091_0)
    return parser


def _coordinate_z(observations: np.ndarray) -> np.ndarray:
    standard_deviation = np.std(observations, axis=0, ddof=1)
    active = standard_deviation > max(
        1.0e-14,
        1.0e-10 * float(np.max(standard_deviation)),
    )
    result = np.full(observations.shape[1], np.nan)
    result[active] = (
        np.mean(observations, axis=0)[active]
        / standard_deviation[active]
        * np.sqrt(len(observations))
    )
    return result


def _heterogeneous_horizon_audit(
    *,
    week: int,
    spot: torch.Tensor,
    options: torch.Tensor,
    initial_options: torch.Tensor,
    state_features: np.ndarray,
    correlation_eigenvalue_cutoff: float,
) -> dict[str, object]:
    """Test raw and state-feature-weighted means of antithetic pair averages."""

    pairs, pair_size = spot.shape
    if pair_size != 2:
        raise ValueError("heterogeneous audit expects adjacent two-path pairs")
    spot_change = spot.numpy() - 1.0
    option_change = options.numpy() - initial_options.numpy()[:, None]
    path_observations = np.concatenate(
        (
            spot_change[..., None],
            option_change.reshape(pairs, pair_size, -1),
        ),
        axis=-1,
    )
    pair_observations = np.mean(path_observations, axis=1)
    pair_option_change = np.mean(option_change, axis=1).reshape(pairs, -1)
    unconditional = _wald_test(
        pair_observations,
        correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
    )
    feature_mean = np.mean(state_features, axis=0, keepdims=True)
    feature_std = np.std(state_features, axis=0, ddof=1, keepdims=True)
    standardized_features = (state_features - feature_mean) / feature_std
    design = np.concatenate((np.ones((pairs, 1)), standardized_features), axis=1)
    weighted_observations = np.einsum(
        "nf,nm->nfm",
        design,
        pair_observations,
    ).reshape(pairs, -1)
    state_weighted = _wald_test(
        weighted_observations,
        correlation_eigenvalue_cutoff=correlation_eigenvalue_cutoff,
    )
    raw_z = _coordinate_z(pair_observations)
    weighted_z = _coordinate_z(weighted_observations)
    return {
        "horizon_weeks": week,
        "num_antithetic_pairs": pairs,
        "num_paths": pairs * pair_size,
        "num_live_original_expiries": options.shape[2],
        "maximum_absolute_spot_mean_bps": float(
            10_000.0 * abs(np.mean(spot_change))
        ),
        "maximum_absolute_option_mean_bps_spot": float(
            10_000.0 * np.max(np.abs(np.mean(pair_option_change, axis=0)))
        ),
        "maximum_absolute_unconditional_z": float(np.nanmax(np.abs(raw_z))),
        "maximum_absolute_state_weighted_z": float(
            np.nanmax(np.abs(weighted_z))
        ),
        "unconditional_omnibus": unconditional,
        "state_weighted_omnibus": state_weighted,
        "state_weighting": {
            "features": "intercept plus first latent static-decoder coordinates",
            "num_nonconstant_features": state_features.shape[1],
        },
    }


def _legacy_comparison(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    simulations = payload["simulations"]
    result: dict[str, object] = {
        "source": str(path),
        "warning": (
            "Legacy raw-coordinate re-grid sampled spot independently and is not "
            "a valid conditional joint spot/surface construction."
        ),
    }
    for name in ("local_sde_no_regrid", "local_sde_regrid", "finite_h_exact_regrid"):
        simulation = simulations[name]
        result[name] = {
            "transition_seconds": simulation["speed"]["transition_seconds"],
            "path_steps_per_transition_second": simulation["speed"][
                "path_steps_per_transition_second"
            ],
            "fixed_cash_maximum_absolute_z_by_horizon": [
                value["maximum_absolute_coordinate_z"]
                for value in simulation["risk_neutrality"][
                    "fixed_cash_with_spot_by_horizon"
                ]
            ],
            "fixed_cash_p_value_by_horizon": [
                value["unconditional_omnibus"]["p_value"]
                for value in simulation["risk_neutrality"][
                    "fixed_cash_with_spot_by_horizon"
                ]
            ],
        }
    return result


def run(arguments: argparse.Namespace) -> None:
    """Run the heterogeneous-path experiment and write its audit."""

    config = HeterogeneousRegridDlvConfig(
        num_paths=arguments.num_paths,
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
        raise ValueError("heterogeneous DLV simulation requires the fixed grid")
    if config.num_state_features > latent_states.shape[1]:
        raise ValueError("num_state_features exceeds the latent state width")
    pairs = config.num_paths // 2
    calendar_dt = config.calendar_business_days / DAYS_PER_YEAR

    with torch.no_grad():
        generator = torch.Generator(device=torch.device(env.device)).manual_seed(
            config.seed
        )
        if pairs < len(latent_states):
            raise ValueError(
                "num_paths must supply at least one antithetic pair per state"
            )
        mandatory_indices = torch.arange(len(latent_states), device=env.device)
        additional_indices = torch.randint(
            len(latent_states),
            (pairs - len(latent_states),),
            device=env.device,
            generator=generator,
        )
        origin_indices = torch.cat((mandatory_indices, additional_indices))
        origin_indices = origin_indices[
            torch.randperm(
                pairs,
                device=env.device,
                generator=generator,
            )
        ]
        sampled_latent_states = latent_states[origin_indices]
        all_raw = model.raw_states_from_states(latent_states)
        initial_raw, forward_parameters = set_constant_backbone_volatility(
            all_raw[origin_indices],
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
        weekly_randoms = [
            make_fixed_grid_dlv_randoms(
                batch_size=pairs,
                num_paths=2,
                dtype=env.fdtype,
                device=env.device,
                generator=generator,
                antithetic=True,
            )
            for _ in range(config.num_steps)
        ]
        spot_by_week = torch.empty(
            (config.num_steps, pairs, 2),
            dtype=torch.float64,
            device="cpu",
        )
        option_by_week = [
            torch.empty(
                (
                    pairs,
                    2,
                    geometry.num_model_expiries - week,
                    config.num_option_strikes,
                ),
                dtype=torch.float64,
                device="cpu",
            )
            for week in range(config.num_steps)
        ]
        factor_scores = torch.empty(
            (config.num_steps, 1, config.num_paths, fixed_loadings.shape[-1]),
            dtype=torch.float64,
            device="cpu",
        )
        intended_factors = torch.empty_like(factor_scores)
        density_weighted_factor_scores = torch.empty_like(factor_scores)

        _synchronize(env.device)
        started = perf_counter()
        fixed_kernel = build_fixed_grid_dlv_kernel(
            model.sanos_decoder,
            initial_raw,
            raw_log_loadings=fixed_loadings,
            calendar_time_increment_y=calendar_dt,
        )
        fixed_kernel_seconds = perf_counter() - started
        sampled = sample_fixed_grid_dlv_paths(
            model.sanos_decoder,
            fixed_kernel,
            fixed_cash_strikes=fixed_cash_strikes,
            randoms=weekly_randoms[0],
        )
        spot_by_week[0] = sampled.normalized_spot.cpu().double()
        option_by_week[0] = sampled.fixed_cash_option_values.cpu().double()
        initial_options = sampled.initial_fixed_cash_option_values.cpu().double()
        pair_index = torch.arange(pairs, device=env.device)[:, None]
        selected_dlvsqs = fixed_kernel.finite_step.dlvsqs_by_branch[
            pair_index,
            sampled.surface_branch_indices,
        ]
        fixed_design = fixed_loadings.reshape(
            1,
            1,
            geometry.num_model_expiries,
            geometry.num_inner_strikes,
            -1,
        ).expand(pairs, 2, -1, -1, -1)
        factor_scores[0, 0] = _factor_scores(
            initial_dlvsqs=initial_surface.dlvsqs[:, None]
            .expand(-1, 2, -1, -1)
            .reshape(config.num_paths, geometry.num_model_expiries, -1),
            final_dlvsqs=selected_dlvsqs.reshape(
                config.num_paths,
                geometry.num_model_expiries,
                -1,
            ),
            raw_log_loadings=fixed_design.reshape(
                config.num_paths,
                geometry.num_model_expiries,
                geometry.num_inner_strikes,
                -1,
            ),
            minimum_dlvsq=geometry.min_dlvsq,
            calendar_time_increment_y=calendar_dt,
        ).cpu().double()
        density_weighted_factor_scores[0, 0] = _factor_scores(
            initial_dlvsqs=initial_surface.dlvsqs[:, None]
            .expand(-1, 2, -1, -1)
            .reshape(config.num_paths, geometry.num_model_expiries, -1),
            final_dlvsqs=selected_dlvsqs.reshape(
                config.num_paths,
                geometry.num_model_expiries,
                -1,
            ),
            raw_log_loadings=fixed_design.reshape(
                config.num_paths,
                geometry.num_model_expiries,
                geometry.num_inner_strikes,
                -1,
            ),
            minimum_dlvsq=geometry.min_dlvsq,
            calendar_time_increment_y=calendar_dt,
            coordinate_weights=initial_surface.densities[:, None, :, 1:-1]
            .expand(-1, 2, -1, -1)
            .reshape(config.num_paths, geometry.num_model_expiries, -1),
        ).cpu().double()
        intended_factors[0, 0] = fixed_kernel.finite_step.standardized_innovations[
            sampled.surface_branch_indices
        ].reshape(config.num_paths, -1).cpu().double()
        maximum_density_tower_residual = float(
            fixed_kernel.finite_step.maximum_density_mean_residual.cpu()
        )
        maximum_decoder_residual = float(
            fixed_kernel.finite_step.maximum_decoder_density_residual.cpu()
        )
        maximum_mass_residual = float(sampled.maximum_conditional_mass_residual.cpu())
        maximum_mean_residual = float(sampled.maximum_conditional_mean_residual.cpu())
        maximum_regrid_price_residual = float(
            sampled.regridded_state.maximum_cash_price_residual.cpu()
        )
        maximum_fallback_fraction = float(
            fixed_kernel.finite_step.fallback_coordinate_fraction.cpu()
        )
        minimum_density = float(fixed_kernel.finite_step.minimum_density.cpu())
        minimum_dlvsq = float(fixed_kernel.finite_step.minimum_dlvsq.cpu())
        maximum_dlvsq = float(fixed_kernel.finite_step.maximum_dlvsq.cpu())
        current_spot = sampled.normalized_spot.reshape(-1)
        surface = floating_surface_from_regrid(
            model.sanos_decoder,
            sampled.regridded_state,
        )
        del fixed_kernel, selected_dlvsqs, fixed_design

        for step_index in range(1, config.num_steps):
            loadings = smooth_floating_log_loadings(
                surface.support_strikes[:, 0],
                num_model_expiries=geometry.num_model_expiries,
                annualized_log_volatility=config.dlvsq_annualized_log_volatility,
            )
            kernel = build_floating_grid_dlv_kernel(
                model.sanos_decoder,
                surface,
                raw_log_loadings=loadings,
                calendar_time_increment_y=calendar_dt,
            )
            randoms = weekly_randoms[step_index]
            rolled = roll_floating_grid_dlv_kernel(
                model.sanos_decoder,
                kernel,
                surface_branch_uniforms=randoms.surface_branch_uniforms.reshape(-1),
                front_node_uniforms=randoms.front_node_uniforms.reshape(-1),
                front_backbone_normals=randoms.front_backbone_normals.reshape(-1),
            )
            path_index = torch.arange(config.num_paths, device=env.device)
            selected_dlvsqs = kernel.dlvsqs_by_branch[
                path_index,
                rolled.surface_branch_indices,
            ]
            factor_scores[step_index, 0] = _factor_scores(
                initial_dlvsqs=surface.dlvsqs,
                final_dlvsqs=selected_dlvsqs,
                raw_log_loadings=loadings,
                minimum_dlvsq=geometry.min_dlvsq,
                calendar_time_increment_y=calendar_dt,
            ).cpu().double()
            density_weighted_factor_scores[step_index, 0] = _factor_scores(
                initial_dlvsqs=surface.dlvsqs,
                final_dlvsqs=selected_dlvsqs,
                raw_log_loadings=loadings,
                minimum_dlvsq=geometry.min_dlvsq,
                calendar_time_increment_y=calendar_dt,
                coordinate_weights=surface.densities[..., 1:-1],
            ).cpu().double()
            intended_factors[step_index, 0] = kernel.standardized_innovations[
                rolled.surface_branch_indices
            ].cpu().double()
            current_spot = current_spot * rolled.gross_spot_return
            spot_by_week[step_index] = current_spot.reshape(pairs, 2).cpu().double()
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
            option_by_week[step_index] = live_values.reshape(
                pairs,
                2,
                geometry.num_model_expiries - step_index,
                config.num_option_strikes,
            ).cpu().double()
            maximum_density_tower_residual = max(
                maximum_density_tower_residual,
                float(kernel.maximum_density_mean_residual.cpu()),
            )
            maximum_decoder_residual = max(
                maximum_decoder_residual,
                float(kernel.maximum_decoder_density_residual.cpu()),
            )
            maximum_mass_residual = max(
                maximum_mass_residual,
                float(kernel.maximum_mass_residual.cpu()),
                float(rolled.maximum_conditional_mass_residual.cpu()),
            )
            maximum_mean_residual = max(
                maximum_mean_residual,
                float(kernel.maximum_mean_residual.cpu()),
                float(rolled.maximum_conditional_mean_residual.cpu()),
            )
            maximum_fallback_fraction = max(
                maximum_fallback_fraction,
                float(kernel.fallback_coordinate_fraction.cpu()),
            )
            minimum_density = min(minimum_density, float(kernel.minimum_density.cpu()))
            minimum_dlvsq = min(minimum_dlvsq, float(kernel.minimum_dlvsq.cpu()))
            maximum_dlvsq = max(maximum_dlvsq, float(kernel.maximum_dlvsq.cpu()))
            surface = rolled.surface
            del kernel, rolled, selected_dlvsqs, loadings
        _synchronize(env.device)
        total_seconds = perf_counter() - started

    state_features = sampled_latent_states[:, : config.num_state_features].cpu().numpy()
    horizon_audits = [
        _heterogeneous_horizon_audit(
            week=week + 1,
            spot=spot_by_week[week],
            options=option_by_week[week],
            initial_options=initial_options[:, week:],
            state_features=state_features,
            correlation_eigenvalue_cutoff=config.correlation_eigenvalue_cutoff,
        )
        for week in range(config.num_steps)
    ]
    sampled_indices_np = origin_indices.cpu().numpy()
    unique_indices, counts = np.unique(sampled_indices_np, return_counts=True)
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_kind": "heterogeneous_multistep_price_preserving_regrid_dlv",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "config": asdict(config),
        "config_version": config.version.unique_id48,
        "device": str(env.device),
        "precision": arguments.precision,
        "sources": sources,
        "initial_state_sampling": {
            "kind": (
                "all source states once, then uniform with replacement and "
                "random shuffle over antithetic origin pairs"
            ),
            "num_source_states": len(latent_states),
            "num_sampled_origin_pairs": pairs,
            "num_paths": config.num_paths,
            "num_unique_source_states": len(unique_indices),
            "minimum_repetitions": int(np.min(counts)),
            "maximum_repetitions": int(np.max(counts)),
            "minimum_source_date": min(dates[int(value)] for value in unique_indices).isoformat(),
            "maximum_source_date": max(dates[int(value)] for value in unique_indices).isoformat(),
        },
        "backbone": {
            "annualized_volatility": config.bs_backbone_annualized_volatility,
            "maximum_total_volatility_residual": float(backbone_residual.cpu()),
            "forward_parameters_minimum": float(forward_parameters.min().cpu()),
            "forward_parameters_maximum": float(forward_parameters.max().cpu()),
        },
        "numerical_invariants": {
            "maximum_density_tower_residual": maximum_density_tower_residual,
            "maximum_decoder_density_residual": maximum_decoder_residual,
            "maximum_conditional_mass_residual": maximum_mass_residual,
            "maximum_conditional_mean_residual": maximum_mean_residual,
            "maximum_first_regrid_cash_price_residual": (
                maximum_regrid_price_residual
            ),
            "maximum_fallback_coordinate_fraction": maximum_fallback_fraction,
            "minimum_density": minimum_density,
            "minimum_dlvsq": minimum_dlvsq,
            "maximum_dlvsq": maximum_dlvsq,
        },
        "speed": {
            "fixed_kernel_seconds": fixed_kernel_seconds,
            "total_simulation_seconds": total_seconds,
            "path_week_steps_per_second": (
                config.num_paths * config.num_steps / total_seconds
            ),
        },
        "antithetic_sources": {
            "origin": "both paths in a pair share one sampled initial state",
            "surface_branch": "u and 1-u",
            "discrete_spot_node": "inverse-CDF u and 1-u",
            "black_spot_factor": "Z and -Z",
            "pair_layout": "adjacent through every flatten and weekly roll",
        },
        "martingale_audits_by_horizon": horizon_audits,
        "realized_dlvsq_covariance_audit": {
            "unweighted_all_coordinates": _factor_covariance_audit(
                factor_scores,
                intended_factors,
            ),
            "initial_density_weighted_coordinates": _factor_covariance_audit(
                density_weighted_factor_scores,
                intended_factors,
            ),
            "interpretation": (
                "The density-weighted projection is the economically relevant "
                "comparison and uses a declared 1e-8 relative weight floor plus "
                "a 1e-10 Gram ridge when interior probability loses rank. The "
                "unweighted projection also counts DLV nodes at zero or "
                "negligible probability, where DLV is unidentified."
            ),
        },
        "legacy_comparison": _legacy_comparison(arguments.legacy_comparison_report),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, arguments.output)
    print(
        " ".join(
            f"w{value['horizon_weeks']}:"
            f"z={value['maximum_absolute_unconditional_z']:.3f},"
            f"p={value['unconditional_omnibus']['p_value']:.4g},"
            f"state_p={value['state_weighted_omnibus']['p_value']:.4g}"
            for value in horizon_audits
        ),
        flush=True,
    )
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    run(_parser().parse_args())
