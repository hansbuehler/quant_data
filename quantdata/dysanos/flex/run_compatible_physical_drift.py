"""Fit a statistically supported DLV physical drift compatible with Q."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.compatible_physical_drift import (
    CompatibleConstantDriftFit,
    factor_drift_audit,
    fit_compatible_constant_drift,
)
from quantdata.dysanos.flex.fixed_grid_dlv_attempt2_triangular import (
    TriangularFixedGridState,
    dlv_option_return_loadings,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.static_q_inputs import load_static_q_inputs
from quantdata.dysanos.flex.theoretical_traded_density import (
    theoretical_exponential_action,
    viable_exponential_action,
)
from quantdata.torch import BaseEnvironment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--baseline-estimate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array-output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hac-lag", type=int, default=4)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--horizon-weeks", type=int, default=4)
    parser.add_argument("--target-horizon-ess-fraction", type=float, default=0.10)
    parser.add_argument("--traded-state-index", type=int, default=44)
    parser.add_argument("--num-traded-strikes", type=int, default=11)
    parser.add_argument("--traded-log-strike-bound", type=float, default=0.30)
    parser.add_argument("--traded-covariance-se-multiple", type=float, default=1.0)
    parser.add_argument("--maximum-traded-action-l1", type=float, default=256.0)
    parser.add_argument(
        "--ess-frontier",
        default="0.50,0.25,0.10,0.05,0.01",
        help="comma-separated four-week expected ESS fractions",
    )
    return parser


def _fit_to_dict(fit: CompatibleConstantDriftFit) -> dict[str, Any]:
    return {
        "num_supported_modes": int(np.sum(fit.supported_modes)),
        "supported_mode_indices_one_based": (np.flatnonzero(fit.supported_modes) + 1).tolist(),
        "raw_weekly_factor_mean_l2": float(
            np.linalg.vector_norm(fit.raw_weekly_factor_mean)
        ),
        "supported_weekly_factor_mean_l2": float(
            np.linalg.vector_norm(fit.supported_weekly_factor_mean)
        ),
        "regularized_weekly_factor_mean_l2": float(
            np.linalg.vector_norm(fit.weekly_factor_mean)
        ),
        "annualized_market_price_of_risk_l2": float(
            np.linalg.vector_norm(fit.annualized_market_price_of_risk)
        ),
        "radial_scale": fit.radial_scale,
        "equivalent_ridge_penalty": fit.equivalent_ridge_penalty,
        "one_week_log_weight_variance": fit.one_week_log_weight_variance,
        "one_week_expected_ess_fraction": math.exp(-fit.one_week_log_weight_variance),
        "horizon_log_weight_variance": fit.horizon_log_weight_variance,
        "horizon_expected_ess_fraction": fit.horizon_expected_ess_fraction,
        "factor_statistics": [
            {
                "mode": index + 1,
                "weekly_factor_mean": float(fit.raw_weekly_factor_mean[index]),
                "hac_standard_error": float(fit.hac_standard_error[index]),
                "hac_t_value": float(fit.hac_t_value[index]),
                "raw_p_value": float(fit.raw_p_value[index]),
                "holm_adjusted_p_value": float(fit.holm_adjusted_p_value[index]),
                "supported": bool(fit.supported_modes[index]),
                "regularized_weekly_factor_mean": float(fit.weekly_factor_mean[index]),
                "annualized_market_price_of_risk": float(
                    fit.annualized_market_price_of_risk[index]
                ),
            }
            for index in range(len(fit.weekly_factor_mean))
        ],
    }


def _parse_frontier(value: str) -> tuple[float, ...]:
    fractions = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not fractions or any(not 0.0 < item <= 1.0 for item in fractions):
        raise ValueError("ess-frontier must contain values in (0, 1]")
    return fractions


def _traded_option_diagnostics(
    *,
    model: Any,
    geometry: Any,
    raw_states: torch.Tensor,
    state_index: int,
    annualized_loading: np.ndarray,
    fit: CompatibleConstantDriftFit,
    num_training_pairs: int,
    num_strikes: int,
    log_strike_bound: float,
    covariance_se_multiple: float,
    maximum_action_l1: float,
    horizon_weeks: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Test whether narrow ITM options can express the fitted density."""

    if not 0 <= state_index < len(raw_states):
        raise ValueError("traded-state-index lies outside the static state table")
    if num_strikes < 3 or num_strikes % 2 == 0:
        raise ValueError("num-traded-strikes must be an odd integer of at least three")
    if min(log_strike_bound, covariance_se_multiple, maximum_action_l1) <= 0.0:
        raise ValueError("traded diagnostic scales and action cap must be positive")

    parameters = bound_native_raw(raw_states[state_index : state_index + 1], geometry=geometry)
    surface = model.sanos_decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    state = TriangularFixedGridState(
        support=surface.support_strikes[:, 0],
        densities=surface.densities,
        dlvsqs=parameters.dlvsqs,
    )
    annualized = torch.as_tensor(
        annualized_loading.reshape(3, geometry.num_inner_strikes, -1),
        dtype=raw_states.dtype,
        device=raw_states.device,
    )
    raw_log_loadings = torch.cat(
        (
            torch.zeros_like(annualized[:1]),
            annualized,
        ),
        dim=0,
    )
    strikes = torch.exp(
        torch.linspace(
            -log_strike_bound,
            log_strike_bound,
            num_strikes,
            dtype=raw_states.dtype,
            device=raw_states.device,
        )
    )
    option_signs = torch.where(strikes <= 1.0, 1.0, -1.0)
    option_values, return_loadings = dlv_option_return_loadings(
        model.sanos_decoder,
        state,
        raw_log_loadings=raw_log_loadings,
        minimum_dlvsq=geometry.min_dlvsq,
        expiry_time_increment_y=model.sanos_decoder.model_dts_y[0],
        fixed_cash_strikes=strikes,
        option_signs=option_signs,
    )
    loading = return_loadings[0]
    row_norm = torch.linalg.vector_norm(loading, dim=1)
    active = row_norm > torch.finfo(loading.dtype).eps * torch.max(row_norm)
    singular_values = torch.linalg.svdvals(loading[active])
    average_variance = torch.sum(singular_values.square()) / torch.sum(active)
    covariance_variance_floor = (
        covariance_se_multiple
        * math.sqrt(2.0 / (num_training_pairs - 1.0))
        * average_variance
    )
    ridge_ratio = float(covariance_variance_floor / singular_values[0].square())
    requested_factor_shift = torch.as_tensor(
        fit.annualized_market_price_of_risk[None],
        dtype=loading.dtype,
        device=loading.device,
    )
    exact = theoretical_exponential_action(return_loadings, requested_factor_shift)
    viable = viable_exponential_action(
        return_loadings,
        requested_factor_shift,
        traded_ridge_ratio=ridge_ratio,
        maximum_factor_shift_norm=float("inf"),
        maximum_action_l1=maximum_action_l1,
    )
    viable_shift = viable.viable_factor_shift[0]
    time_increment_y = 5.0 / DAYS_PER_YEAR
    one_week_cost = time_increment_y * torch.sum(viable_shift.square())
    return (
        {
            "state_index": state_index,
            "num_strikes_per_expiry": num_strikes,
            "relative_strikes": strikes.cpu().double().tolist(),
            "option_type": "ITM call for K<=1 and ITM put for K>1",
            "num_option_coordinates": int(loading.shape[0]),
            "num_surface_sensitive_option_coordinates": int(torch.sum(active)),
            "minimum_option_value": float(torch.min(option_values)),
            "exact_numerical_rank": int(exact.numerical_rank[0]),
            "statistically_resolved_rank": int(
                torch.sum(singular_values.square() > covariance_variance_floor)
            ),
            "minimum_singular_value": float(torch.min(singular_values)),
            "maximum_singular_value": float(torch.max(singular_values)),
            "condition_number": float(torch.max(singular_values) / torch.min(singular_values)),
            "covariance_variance_floor": float(covariance_variance_floor),
            "covariance_se_multiple": covariance_se_multiple,
            "traded_ridge_ratio": ridge_ratio,
            "exact_action_l1": float(torch.sum(torch.abs(exact.action))),
            "exact_maximum_factor_replication_residual": float(
                torch.max(torch.abs(exact.factor_replication_residual))
            ),
            "viable_action_l1": float(torch.sum(torch.abs(viable.theoretical_action.action))),
            "maximum_action_l1": maximum_action_l1,
            "viable_factor_retention_fraction": float(viable.factor_retention_fraction[0]),
            "viable_annualized_market_price_of_risk_l2": float(
                torch.linalg.vector_norm(viable_shift)
            ),
            "viable_one_week_log_weight_variance": float(one_week_cost),
            "viable_four_week_expected_ess_fraction": float(
                torch.exp(-horizon_weeks * one_week_cost)
            ),
            "interpretation": (
                "the exact 4x11 book spans all fitted factors algebraically but is ill-conditioned; "
                "the covariance-noise ridge retains only directions resolvable from the historical sample"
            ),
        },
        viable_shift.cpu().double().numpy(),
    )


def run(arguments: argparse.Namespace) -> None:
    """Fit the constrained drift from the existing matched historical/Q data."""

    baseline = np.load(arguments.baseline_estimate)
    required = {
        "weekly_loading",
        "q_change",
        "origin_indices",
        "endpoint_indices",
        "training_mask",
    }
    missing = required.difference(baseline.files)
    if missing:
        raise ValueError(f"baseline estimate is missing arrays: {sorted(missing)}")

    env = BaseEnvironment(precision=64, device=arguments.device, debug=False)
    model, dates, latent_states, sources = load_static_q_inputs(
        arguments.static_decoder_checkpoint,
        arguments.static_report,
        env=env,
    )
    geometry = model.config.geometry
    with torch.no_grad():
        raw = model.raw_states_from_states(latent_states).cpu().double().numpy()
    raw_blocks = raw.reshape(
        len(raw),
        geometry.num_model_expiries,
        geometry.expiry_block_state_size,
    )
    origin = baseline["origin_indices"].astype(np.int64)
    endpoint = baseline["endpoint_indices"].astype(np.int64)
    training = baseline["training_mask"].astype(bool)
    observed_change = (
        raw_blocks[endpoint, :3, : geometry.num_inner_strikes]
        - raw_blocks[origin, 1:, : geometry.num_inner_strikes]
    ).reshape(len(origin), -1)
    q_change = np.asarray(baseline["q_change"], dtype=np.float64)
    weekly_loading = np.asarray(baseline["weekly_loading"], dtype=np.float64)
    if observed_change.shape != q_change.shape or weekly_loading.shape[0] != q_change.shape[1]:
        raise ValueError("baseline drift arrays do not align with the static model")
    weekly_loading_pinv = np.linalg.pinv(weekly_loading, rcond=1.0e-12)
    factor_innovations = (observed_change - q_change) @ weekly_loading_pinv.T
    time_increment_y = 5.0 / DAYS_PER_YEAR

    fit = fit_compatible_constant_drift(
        factor_innovations[training],
        time_increment_y=time_increment_y,
        horizon_steps=arguments.horizon_weeks,
        target_horizon_ess_fraction=arguments.target_horizon_ess_fraction,
        familywise_alpha=arguments.familywise_alpha,
        hac_lag=arguments.hac_lag,
    )
    frontier: list[dict[str, Any]] = []
    for target in _parse_frontier(arguments.ess_frontier):
        candidate = fit_compatible_constant_drift(
            factor_innovations[training],
            time_increment_y=time_increment_y,
            horizon_steps=arguments.horizon_weeks,
            target_horizon_ess_fraction=target,
            familywise_alpha=arguments.familywise_alpha,
            hac_lag=arguments.hac_lag,
        )
        frontier.append(
            {
                "target_horizon_expected_ess_fraction": target,
                **{
                    key: value
                    for key, value in _fit_to_dict(candidate).items()
                    if key != "factor_statistics"
                },
                "training_audit": factor_drift_audit(
                    factor_innovations[training],
                    candidate.weekly_factor_mean,
                    hac_lag=arguments.hac_lag,
                ),
                "validation_audit": factor_drift_audit(
                    factor_innovations[~training],
                    candidate.weekly_factor_mean,
                    hac_lag=arguments.hac_lag,
                ),
            }
        )

    supported_unregularized = fit.supported_weekly_factor_mean
    with torch.no_grad():
        traded_diagnostics, traded_market_price_of_risk = _traded_option_diagnostics(
            model=model,
            geometry=geometry,
            raw_states=torch.as_tensor(raw, dtype=env.fdtype, device=env.device),
            state_index=arguments.traded_state_index,
            annualized_loading=np.asarray(baseline["annualized_loading"], dtype=np.float64),
            fit=fit,
            num_training_pairs=int(np.sum(training)),
            num_strikes=arguments.num_traded_strikes,
            log_strike_bound=arguments.traded_log_strike_bound,
            covariance_se_multiple=arguments.traded_covariance_se_multiple,
            maximum_action_l1=arguments.maximum_traded_action_l1,
            horizon_weeks=arguments.horizon_weeks,
        )
    traded_weekly_factor_mean = math.sqrt(time_increment_y) * traded_market_price_of_risk
    traded_diagnostics["training_historical_fit"] = factor_drift_audit(
        factor_innovations[training],
        traded_weekly_factor_mean,
        hac_lag=arguments.hac_lag,
    )
    traded_diagnostics["validation_historical_fit"] = factor_drift_audit(
        factor_innovations[~training],
        traded_weekly_factor_mean,
        hac_lag=arguments.hac_lag,
    )
    result = {
        "schema_version": 1,
        "experiment_kind": "likelihood_regularized_compatible_physical_drift",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": {
            "state": "z=log(DLVsq-minimum), fixed-expiry rollover coordinates",
            "historical_innovation": "e=B^+(Delta z-h*mu_Q(z))",
            "compatible_physical_drift": "mu_P(z)=mu_Q(z)+L*lambda",
            "factor_mean": "E_P[e]=sqrt(h)*lambda",
            "mode_selection": (
                "two-sided HAC z tests with Holm familywise-error correction; "
                "lag four accounts for overlapping five-session returns"
            ),
            "regularization": (
                "maximum Gaussian likelihood subject to a four-week expected "
                "density ESS floor; equivalently isotropic ridge on lambda"
            ),
            "density_cost": (
                "one-week log-weight variance h*||lambda||^2=||E[e]||^2; "
                "idealized expected ESS fraction exp(-cost)"
            ),
        },
        "sources": sources,
        "baseline_estimate": str(arguments.baseline_estimate),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "sample": {
            "num_pairs": int(len(origin)),
            "num_training_pairs": int(np.sum(training)),
            "num_validation_pairs": int(np.sum(~training)),
            "first_origin_date": dates[int(origin[0])].isoformat(),
            "last_training_endpoint_date": dates[int(endpoint[training][-1])].isoformat(),
            "first_validation_endpoint_date": dates[int(endpoint[~training][0])].isoformat(),
            "last_endpoint_date": dates[int(endpoint[-1])].isoformat(),
            "num_dlv_coordinates": int(observed_change.shape[1]),
            "num_factors": int(weekly_loading.shape[1]),
        },
        "selected_estimate": _fit_to_dict(fit),
        "benchmarks": {
            "q": {
                "training_audit": factor_drift_audit(
                    factor_innovations[training],
                    np.zeros(weekly_loading.shape[1]),
                    hac_lag=arguments.hac_lag,
                ),
                "validation_audit": factor_drift_audit(
                    factor_innovations[~training],
                    np.zeros(weekly_loading.shape[1]),
                    hac_lag=arguments.hac_lag,
                ),
            },
            "supported_unregularized_mle": {
                "one_week_log_weight_variance": float(
                    supported_unregularized @ supported_unregularized
                ),
                "four_week_expected_ess_fraction": float(
                    math.exp(
                        -arguments.horizon_weeks
                        * float(supported_unregularized @ supported_unregularized)
                    )
                ),
                "training_audit": factor_drift_audit(
                    factor_innovations[training],
                    supported_unregularized,
                    hac_lag=arguments.hac_lag,
                ),
                "validation_audit": factor_drift_audit(
                    factor_innovations[~training],
                    supported_unregularized,
                    hac_lag=arguments.hac_lag,
                ),
            },
        },
        "selected_training_audit": factor_drift_audit(
            factor_innovations[training],
            fit.weekly_factor_mean,
            hac_lag=arguments.hac_lag,
        ),
        "selected_validation_audit": factor_drift_audit(
            factor_innovations[~training],
            fit.weekly_factor_mean,
            hac_lag=arguments.hac_lag,
        ),
        "narrow_traded_option_diagnostic": traded_diagnostics,
        "ess_frontier": frontier,
    }
    write_json_atomic(result, arguments.output)
    arguments.array_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.array_output,
        weekly_loading=weekly_loading,
        factor_innovations=factor_innovations,
        training_mask=training,
        supported_modes=fit.supported_modes,
        weekly_factor_mean=fit.weekly_factor_mean,
        annualized_market_price_of_risk=fit.annualized_market_price_of_risk,
        traded_regularized_annualized_market_price_of_risk=traded_market_price_of_risk,
        compatible_q_relative_weekly_change=(
            math.sqrt(time_increment_y)
            * fit.annualized_market_price_of_risk[None]
            @ weekly_loading.T
        ),
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {arguments.output}", flush=True)
    print(f"wrote {arguments.array_output}", flush=True)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
