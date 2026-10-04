"""Test a daily OCE measure change using spot and six near-ATM calls."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import chi2

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.few_instrument_oce import (
    entropic_oce_values,
    fit_batched_entropic_oce,
)
from quantdata.dysanos.flex.front_spot_copula import (
    _front_quantile_spline,
    fit_front_spot_copula_from_history,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.presentation_hedging import (
    _call_prices,
    _decode_surface,
    exact_daily_transition,
    load_presentation_model,
)

_RELATIVE_STRIKES = (0.99, 1.0, 1.01, 0.99, 1.0, 1.01)
_INITIAL_DAYS = (5.0, 5.0, 5.0, 10.0, 10.0, 10.0)
_FINAL_DAYS = (4.0, 4.0, 4.0, 9.0, 9.0, 9.0)
_LABELS = (
    "spot",
    "5D call K/S=0.99",
    "5D call K/S=1.00",
    "5D call K/S=1.01",
    "10D call K/S=0.99",
    "10D call K/S=1.00",
    "10D call K/S=1.01",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-origins", type=int, default=21)
    parser.add_argument("--training-paths", type=int, default=100_000)
    parser.add_argument("--validation-paths", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2_026_092_8)
    parser.add_argument("--maximum-iterations", type=int, default=50)
    parser.add_argument("--mean-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--copula-fit-before", type=date.fromisoformat, default=date(2025, 1, 1))
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.num_origins <= 0:
        raise ValueError("num-origins must be positive")
    if min(arguments.training_paths, arguments.validation_paths) <= 2:
        raise ValueError("path counts must exceed two")
    if arguments.training_paths % 2 or arguments.validation_paths % 2:
        raise ValueError("antithetic path counts must be even")
    if arguments.maximum_iterations <= 0 or arguments.mean_tolerance <= 0.0:
        raise ValueError("invalid OCE solver controls")


def _draw_returns(
    *,
    model: Any,
    current_state: torch.Tensor,
    daily_transition: torch.Tensor,
    daily_factor: torch.Tensor,
    copula_loading: torch.Tensor,
    copula_orthogonal_std: float,
    copula_total_std: float,
    num_paths: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one-session fixed-cash price changes and return issue prices."""

    device = current_state.device
    dtype = current_state.dtype
    dimension = current_state.shape[0]
    half_state = torch.randn(
        (num_paths // 2, dimension), dtype=dtype, device=device, generator=generator
    )
    half_spot = torch.randn(
        num_paths // 2, dtype=dtype, device=device, generator=generator
    )
    state_normals = torch.cat((half_state, -half_state), dim=0)
    independent_spot = torch.cat((half_spot, -half_spot), dim=0)
    del half_state, half_spot

    equilibrium = model.equilibrium_mean.detach()
    expected = equilibrium + (current_state - equilibrium) @ daily_transition
    next_states = expected[None] + state_normals @ daily_factor.T
    gaussian_score = (
        state_normals @ copula_loading + copula_orthogonal_std * independent_spot
    ) / copula_total_std

    current_batch = current_state[None]
    with torch.inference_mode():
        current_surface = _decode_surface(model, current_batch)
        support = current_surface.support_strikes[0, 0].cpu().double().numpy()
        density = current_surface.densities[0, 0].cpu().double().numpy()
        sqrt_variance = float(current_surface.backbone_total_vols[0, 0].cpu())
    gross_spot = _front_quantile_spline(
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        gaussian_score=gaussian_score.cpu().double().numpy(),
    )
    gross_spot_tensor = torch.as_tensor(gross_spot, dtype=dtype, device=device)

    strikes = torch.tensor(_RELATIVE_STRIKES, dtype=dtype, device=device)
    initial_days = torch.tensor(_INITIAL_DAYS, dtype=dtype, device=device)
    final_days = torch.tensor(_FINAL_DAYS, dtype=dtype, device=device)
    with torch.inference_mode():
        issue_prices = _call_prices(
            model,
            current_batch,
            torch.ones(1, dtype=dtype, device=device),
            remaining_days=initial_days,
            cash_strikes=strikes,
        )[0]
        final_prices = _call_prices(
            model,
            next_states,
            gross_spot_tensor,
            remaining_days=final_days,
            cash_strikes=strikes,
        )
    returns = torch.cat(
        (
            (gross_spot_tensor - 1.0)[:, None],
            final_prices - issue_prices[None],
        ),
        dim=1,
    )
    if not bool(torch.all(torch.isfinite(returns))):
        raise FloatingPointError("reduced OCE book produced non-finite price changes")
    return returns.double(), issue_prices.double()


def _paired_audit(returns: torch.Tensor, weights: torch.Tensor) -> dict[str, Any]:
    """Audit one self-normalized measure using antithetic pairs as observations."""

    if returns.ndim != 2 or len(returns) % 2 or weights.shape != (len(returns),):
        raise ValueError("paired audit geometry is invalid")
    paths = len(returns)
    pairs = paths // 2
    mass = torch.mean(weights)
    mean = torch.mean(weights[:, None] * returns, dim=0) / mass
    influence = weights[:, None] * (returns - mean[None]) / mass
    paired = 0.5 * (influence[:pairs] + influence[pairs:])
    centered = paired - torch.mean(paired, dim=0)
    covariance_of_mean = centered.T @ centered / (pairs * (pairs - 1))
    covariance_of_mean = 0.5 * (covariance_of_mean + covariance_of_mean.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_of_mean)
    tolerance = (
        torch.finfo(returns.dtype).eps
        * torch.amax(torch.abs(eigenvalues)).clamp_min(torch.finfo(returns.dtype).tiny)
        * returns.shape[1]
    )
    retained = eigenvalues > tolerance
    projected = eigenvectors[:, retained].T @ mean
    statistic = torch.sum(projected.square() / eigenvalues[retained])
    standard_error = torch.sqrt(torch.clamp_min(torch.diag(covariance_of_mean), 0.0))
    z = mean / standard_error.clamp_min(torch.finfo(returns.dtype).tiny)
    degrees = int(torch.sum(retained))
    return {
        "weighted_mean": mean.cpu().tolist(),
        "standard_error": standard_error.cpu().tolist(),
        "z": z.cpu().tolist(),
        "maximum_absolute_mean_bps_spot": float(torch.amax(torch.abs(mean))) * 1.0e4,
        "maximum_absolute_z": float(torch.amax(torch.abs(z))),
        "root_mean_square_z": float(torch.sqrt(torch.mean(z.square()))),
        "joint_statistic": float(statistic),
        "joint_degrees_of_freedom": degrees,
        "joint_p_value": float(chi2.sf(float(statistic), degrees)),
        "density_mass": float(mass),
        "effective_sample_size": float(paths**2 / torch.sum(weights.square())),
        "effective_sample_fraction": float(paths / torch.sum(weights.square())),
        "minimum_weight": float(torch.amin(weights)),
        "maximum_weight": float(torch.amax(weights)),
        "zero_weight_count": int(torch.sum(weights == 0.0)),
    }


def _aggregate_audits(values: list[dict[str, Any]]) -> dict[str, Any]:
    statistics = np.asarray([value["joint_statistic"] for value in values])
    degrees = np.asarray([value["joint_degrees_of_freedom"] for value in values])
    joint_p = np.asarray([value["joint_p_value"] for value in values])
    maximum_z = np.asarray([value["maximum_absolute_z"] for value in values])
    ess_fraction = np.asarray([value["effective_sample_fraction"] for value in values])
    all_z = np.concatenate([np.abs(np.asarray(value["z"])) for value in values])
    total_statistic = float(np.sum(statistics))
    total_degrees = int(np.sum(degrees))
    return {
        "origins": len(values),
        "coordinates": int(len(values) * len(_LABELS)),
        "omnibus_statistic": total_statistic,
        "omnibus_degrees_of_freedom": total_degrees,
        "omnibus_p_value": float(chi2.sf(total_statistic, total_degrees)),
        "origin_joint_p_value_minimum": float(np.min(joint_p)),
        "origin_joint_p_value_median": float(np.median(joint_p)),
        "origin_joint_rejection_fraction_5pct": float(np.mean(joint_p < 0.05)),
        "maximum_absolute_z": float(np.max(maximum_z)),
        "root_mean_square_z": float(np.sqrt(np.mean(np.square(all_z)))),
        "coordinate_fraction_abs_z_above_1p96": float(np.mean(all_z > 1.96)),
        "effective_sample_fraction_minimum": float(np.min(ess_fraction)),
        "effective_sample_fraction_median": float(np.median(ess_fraction)),
    }


def run(arguments: argparse.Namespace) -> None:
    """Run the conditional reduced-book OCE experiment."""

    _validate_arguments(arguments)
    env, model = load_presentation_model(arguments.checkpoint, device=arguments.device)
    history = torch.load(arguments.target_history, map_location="cpu", weights_only=False)
    dates = history.get("dates")
    states_value = history.get("states")
    spots_value = history.get("spot_refs")
    if (
        not isinstance(dates, list)
        or not isinstance(states_value, torch.Tensor)
        or not isinstance(spots_value, torch.Tensor)
    ):
        raise ValueError("target history is missing dates, states, or spot references")
    states = states_value.double().numpy()
    spots = spots_value.double().numpy()
    if arguments.num_origins > len(states):
        raise ValueError("num-origins exceeds the historical state count")

    daily_transition, daily_factor, transition_diagnostics = exact_daily_transition(model)
    copula_fit, copula_diagnostics = fit_front_spot_copula_from_history(
        model=model,
        states=states,
        spots=spots,
        dates=dates,
        daily_transition=daily_transition.cpu().double().numpy(),
        daily_factor=daily_factor.cpu().double().numpy(),
        fit_before=arguments.copula_fit_before,
    )
    origin_indices = np.linspace(
        0, len(states) - 1, arguments.num_origins, dtype=np.int64
    )
    copula_loading = torch.as_tensor(
        copula_fit.state_loading, dtype=env.fdtype, device=env.device
    )
    generator = torch.Generator(device=env.device).manual_seed(arguments.seed)

    records: list[dict[str, Any]] = []
    p_audits: list[dict[str, Any]] = []
    q_audits: list[dict[str, Any]] = []
    training_audits: list[dict[str, Any]] = []
    for position, origin_index in enumerate(origin_indices.tolist(), start=1):
        current_state = torch.as_tensor(
            states[origin_index], dtype=env.fdtype, device=env.device
        )
        training_returns, issue_prices = _draw_returns(
            model=model,
            current_state=current_state,
            daily_transition=daily_transition,
            daily_factor=daily_factor,
            copula_loading=copula_loading,
            copula_orthogonal_std=copula_fit.orthogonal_std,
            copula_total_std=copula_fit.total_std,
            num_paths=arguments.training_paths,
            generator=generator,
        )
        scale = torch.sqrt(torch.mean(training_returns.square(), dim=0)).clamp_min(1.0e-10)
        scaled_training = (training_returns / scale[None]).unsqueeze(0)
        fit = fit_batched_entropic_oce(
            scaled_training,
            maximum_iterations=arguments.maximum_iterations,
            mean_tolerance=arguments.mean_tolerance,
        )
        raw_action = fit.action[0] / scale
        training_audit = _paired_audit(training_returns, fit.weights[0])
        training_audits.append(training_audit)
        del scaled_training

        validation_returns, validation_issue_prices = _draw_returns(
            model=model,
            current_state=current_state,
            daily_transition=daily_transition,
            daily_factor=daily_factor,
            copula_loading=copula_loading,
            copula_orthogonal_std=copula_fit.orthogonal_std,
            copula_total_std=copula_fit.total_std,
            num_paths=arguments.validation_paths,
            generator=generator,
        )
        if not torch.allclose(issue_prices, validation_issue_prices, atol=0.0, rtol=0.0):
            raise RuntimeError("issue prices changed between training and validation draws")
        _, validation_weights, _ = entropic_oce_values(
            validation_returns.unsqueeze(0), raw_action.unsqueeze(0)
        )
        p_audit = _paired_audit(
            validation_returns, torch.ones(arguments.validation_paths, dtype=torch.float64, device=env.device)
        )
        q_audit = _paired_audit(validation_returns, validation_weights[0])
        p_audits.append(p_audit)
        q_audits.append(q_audit)
        record = {
            "origin_index": origin_index,
            "date": str(dates[origin_index]),
            "issue_prices": issue_prices.cpu().tolist(),
            "return_rms_scale": scale.cpu().tolist(),
            "raw_action": raw_action.cpu().tolist(),
            "solver_converged": bool(fit.converged[0]),
            "solver_iterations": int(fit.iterations[0]),
            "training_objective": float(fit.objective[0]),
            "training": training_audit,
            "validation_under_p": p_audit,
            "validation_under_q": q_audit,
        }
        records.append(record)
        print(
            f"origin {position}/{len(origin_indices)} {dates[origin_index]} "
            f"converged={record['solver_converged']} "
            f"ESS={q_audit['effective_sample_size']:.1f}/{arguments.validation_paths} "
            f"P|maxz|={p_audit['maximum_absolute_z']:.3f} "
            f"Q|maxz|={q_audit['maximum_absolute_z']:.3f} "
            f"Qp={q_audit['joint_p_value']:.4g}",
            flush=True,
        )
        del training_returns, validation_returns, validation_weights, fit
        if env.device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "schema": "dysanos-few-instrument-conditional-entropic-oce-v1",
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": {
            "time_step": "one business day",
            "trading_book": list(_LABELS),
            "option_timing": (
                "fixed cash strikes K=(0.99,1.00,1.01)*S_t; 5D and 10D calls "
                "are revalued one session later at 4D and 9D"
            ),
            "price_change": "S_{t+1}-S_t and C_{t+1}(T,K)-C_t(T,K)",
            "spot_law": (
                "current front-expiry SANOS inverse CDF with the fitted Gaussian "
                "spot/state copula"
            ),
            "state_law": "exact daily factor of the fitted Direct-200 weekly Gaussian AR law",
            "oce": "min_a log E_P[exp(-a'R)]; dQ/dP=exp(-a'R)/E_P[exp(-a'R)]",
            "action_scope": (
                "one unrestricted seven-coordinate action per historical origin; "
                "no neural-network approximation or generalization requirement"
            ),
            "audit": (
                "actions fit on training paths and evaluated on an independent antithetic "
                "validation path set; antithetic pairs are the independent observations"
            ),
        },
        "arguments": {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in vars(arguments).items()
        },
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        "target_history": str(arguments.target_history),
        "target_history_sha256": checkpoint_digest(arguments.target_history),
        "transition_diagnostics": transition_diagnostics,
        "spot_copula_fit": copula_fit.to_dict(),
        "spot_copula_diagnostics": copula_diagnostics,
        "origin_indices": origin_indices.tolist(),
        "solver_convergence_fraction": float(
            np.mean([record["solver_converged"] for record in records])
        ),
        "training_under_fitted_q": _aggregate_audits(training_audits),
        "independent_validation_under_p": _aggregate_audits(p_audits),
        "independent_validation_under_fitted_q": _aggregate_audits(q_audits),
        "origins": records,
    }
    write_json_atomic(result, arguments.output)
    print(json.dumps({key: result[key] for key in (
        "solver_convergence_fraction",
        "training_under_fitted_q",
        "independent_validation_under_p",
        "independent_validation_under_fitted_q",
    )}, indent=2), flush=True)
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_arguments())


if __name__ == "__main__":
    main()
