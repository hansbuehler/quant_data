"""Test OCE recovery of a known Q drift using directly traded risk factors."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import chi2

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.direct_factor_oce import (
    antithetic_gaussian_factor_returns,
    exponential_oce_values,
    fit_exponential_oce_action,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-paths", type=int, default=1_000_000)
    parser.add_argument("--validation-paths", type=int, default=1_000_000)
    parser.add_argument("--num-weeks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2_026_092_4)
    parser.add_argument("--maximum-iterations", type=int, default=30)
    parser.add_argument("--mean-tolerance", type=float, default=1.0e-10)
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _density_summary(weights: torch.Tensor) -> dict[str, float | int]:
    quantiles = torch.quantile(
        weights,
        torch.tensor((0.0, 0.5, 0.9, 0.99, 1.0), dtype=weights.dtype, device=weights.device),
    )
    return {
        "density_mass": float(torch.mean(weights)),
        "effective_sample_size": float(len(weights) ** 2 / torch.sum(weights.square())),
        "effective_sample_fraction": float(len(weights) / torch.sum(weights.square())),
        "minimum_weight": float(quantiles[0]),
        "median_weight": float(quantiles[1]),
        "q90_weight": float(quantiles[2]),
        "q99_weight": float(quantiles[3]),
        "maximum_weight": float(quantiles[4]),
        "numerically_zero_weight_count": int(torch.sum(weights == 0.0)),
    }


def _paired_weighted_mean_audit(
    returns: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, float | int]:
    """Use antithetic pairs as the independent observations in a joint test."""

    if len(returns) % 2 or weights.shape != (len(returns),):
        raise ValueError("paired audit inputs have incompatible shapes")
    mean = torch.mean(weights[:, None] * returns, dim=0)
    # The denominator of the self-normalized importance estimator is random.
    # Its influence is w*(R-E_Q[R]), not merely w*R. Pair this influence before
    # estimating uncertainty so the antithetic dependence is respected.
    influence = (weights[:, None] * (returns - mean)).reshape(
        len(returns) // 2, 2, -1
    ).mean(dim=1)
    centered = influence - torch.mean(influence, dim=0)
    num_pairs = len(influence)
    covariance_of_mean = centered.T @ centered / (num_pairs * (num_pairs - 1))
    covariance_of_mean = 0.5 * (covariance_of_mean + covariance_of_mean.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_of_mean)
    tolerance = (
        torch.finfo(eigenvalues.dtype).eps
        * torch.max(torch.abs(eigenvalues)).clamp_min(1.0)
        * len(eigenvalues)
    )
    retained = eigenvalues > tolerance
    projected = eigenvectors[:, retained].T @ mean
    statistic = torch.sum(projected.square() / eigenvalues[retained])
    standard_error = torch.sqrt(torch.clamp_min(torch.diag(covariance_of_mean), 0.0))
    z = mean / standard_error.clamp_min(torch.finfo(standard_error.dtype).tiny)
    return {
        "num_antithetic_pairs": num_pairs,
        "num_coordinates": returns.shape[1],
        "maximum_absolute_weighted_mean": float(torch.max(torch.abs(mean))),
        "weighted_mean_l2": float(torch.linalg.vector_norm(mean)),
        "maximum_absolute_z": float(torch.max(torch.abs(z))),
        "root_mean_square_z": float(torch.sqrt(torch.mean(z.square()))),
        "joint_statistic": float(statistic),
        "joint_degrees_of_freedom": int(torch.sum(retained)),
        "joint_p_value": float(chi2.sf(float(statistic), int(torch.sum(retained)))),
    }


def _action_summary(action: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = action - target
    target_norm = torch.linalg.vector_norm(target)
    return {
        "action_l1": float(torch.sum(torch.abs(action))),
        "action_l2": float(torch.linalg.vector_norm(action)),
        "target_l2": float(target_norm),
        "error_l2": float(torch.linalg.vector_norm(error)),
        "relative_error_l2": float(
            torch.linalg.vector_norm(error) / target_norm.clamp_min(torch.finfo(target.dtype).tiny)
        ),
        "maximum_absolute_error": float(torch.max(torch.abs(error))),
        "cosine_similarity": float(
            torch.dot(action, target)
            / (
                torch.linalg.vector_norm(action).clamp_min(torch.finfo(action.dtype).tiny)
                * target_norm.clamp_min(torch.finfo(target.dtype).tiny)
            )
        ),
    }


def _dlv_drift_audit(
    weighted_factor_mean: torch.Tensor,
    annualized_loading: torch.Tensor,
    *,
    num_weeks: int,
) -> dict[str, float]:
    factor_by_week = weighted_factor_mean.reshape(num_weeks, -1)
    dlv_mean = factor_by_week @ annualized_loading.T
    return {
        "maximum_absolute_q_relative_log_dlv_mean": float(torch.max(torch.abs(dlv_mean))),
        "root_mean_square_q_relative_log_dlv_mean": float(torch.sqrt(torch.mean(dlv_mean.square()))),
    }


def run(arguments: argparse.Namespace) -> None:
    """Fit and independently audit the direct-factor entropic OCE."""

    if min(arguments.num_paths, arguments.validation_paths) <= 0:
        raise ValueError("path counts must be positive")
    if arguments.num_paths % 2 or arguments.validation_paths % 2:
        raise ValueError("antithetic path counts must be even")
    if arguments.num_weeks <= 0:
        raise ValueError("num-weeks must be positive")

    estimate = np.load(arguments.estimate)
    required = {
        "annualized_market_price_of_risk",
        "factor_innovations",
        "training_mask",
        "supported_modes",
        "weekly_loading",
    }
    missing = required.difference(estimate.files)
    if missing:
        raise ValueError(f"compatible estimate is missing arrays: {sorted(missing)}")
    device = torch.device(arguments.device)
    dtype = torch.float64
    time_increment_y = 5.0 / DAYS_PER_YEAR
    regularized_lambda = torch.as_tensor(
        estimate["annualized_market_price_of_risk"],
        dtype=dtype,
        device=device,
    )
    supported = estimate["supported_modes"].astype(bool)
    training = estimate["training_mask"].astype(bool)
    raw_weekly_mean = np.mean(estimate["factor_innovations"][training], axis=0)
    raw_lambda = torch.as_tensor(
        np.where(supported, raw_weekly_mean, 0.0) / math.sqrt(time_increment_y),
        dtype=dtype,
        device=device,
    )
    annualized_loading = torch.as_tensor(
        estimate["weekly_loading"] / math.sqrt(time_increment_y),
        dtype=dtype,
        device=device,
    )
    regularized_target = regularized_lambda.repeat(arguments.num_weeks)
    raw_target = raw_lambda.repeat(arguments.num_weeks)

    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    _synchronize(device)
    training_returns = antithetic_gaussian_factor_returns(
        num_paths=arguments.num_paths,
        factor_drift=regularized_target,
        time_increment_y=time_increment_y,
        generator=generator,
    )
    direct_factor_mle_action = torch.mean(training_returns, dim=0) / time_increment_y
    direct_factor_mle = exponential_oce_values(training_returns, direct_factor_mle_action)
    _synchronize(device)
    fit = fit_exponential_oce_action(
        training_returns,
        return_variance=time_increment_y,
        maximum_iterations=arguments.maximum_iterations,
        mean_tolerance=arguments.mean_tolerance,
    )
    _synchronize(device)
    training_known = exponential_oce_values(training_returns, regularized_target)
    training_report = {
        "learned_action": _action_summary(fit.action, regularized_target),
        "solver_converged": fit.converged,
        "solver_history": list(fit.history),
        "learned_density": _density_summary(fit.values.weights),
        "known_density": _density_summary(training_known.weights),
        "direct_factor_drift_mle_action": _action_summary(
            direct_factor_mle_action,
            regularized_target,
        ),
        "direct_factor_drift_mle_density": _density_summary(direct_factor_mle.weights),
        "learned_factor_audit": _paired_weighted_mean_audit(
            training_returns,
            fit.values.weights,
        ),
        "known_factor_audit": _paired_weighted_mean_audit(
            training_returns,
            training_known.weights,
        ),
        "direct_factor_drift_mle_audit": _paired_weighted_mean_audit(
            training_returns,
            direct_factor_mle.weights,
        ),
        "learned_dlv_drift_audit": _dlv_drift_audit(
            fit.values.weighted_return_mean,
            annualized_loading,
            num_weeks=arguments.num_weeks,
        ),
    }
    del training_returns, training_known, direct_factor_mle
    if device.type == "cuda":
        torch.cuda.empty_cache()

    validation_returns = antithetic_gaussian_factor_returns(
        num_paths=arguments.validation_paths,
        factor_drift=regularized_target,
        time_increment_y=time_increment_y,
        generator=generator,
    )
    learned_validation = exponential_oce_values(validation_returns, fit.action)
    known_validation = exponential_oce_values(validation_returns, regularized_target)
    direct_factor_mle_validation = exponential_oce_values(
        validation_returns,
        direct_factor_mle_action,
    )
    validation_report = {
        "learned_density": _density_summary(learned_validation.weights),
        "known_density": _density_summary(known_validation.weights),
        "direct_factor_drift_mle_density": _density_summary(
            direct_factor_mle_validation.weights
        ),
        "learned_factor_audit": _paired_weighted_mean_audit(
            validation_returns,
            learned_validation.weights,
        ),
        "known_factor_audit": _paired_weighted_mean_audit(
            validation_returns,
            known_validation.weights,
        ),
        "direct_factor_drift_mle_audit": _paired_weighted_mean_audit(
            validation_returns,
            direct_factor_mle_validation.weights,
        ),
        "learned_dlv_drift_audit": _dlv_drift_audit(
            learned_validation.weighted_return_mean,
            annualized_loading,
            num_weeks=arguments.num_weeks,
        ),
        "known_dlv_drift_audit": _dlv_drift_audit(
            known_validation.weighted_return_mean,
            annualized_loading,
            num_weeks=arguments.num_weeks,
        ),
    }

    raw_returns = validation_returns + time_increment_y * (raw_target - regularized_target)[None]
    raw_known = exponential_oce_values(raw_returns, raw_target)
    raw_cost = time_increment_y * torch.sum(raw_target.square())
    raw_control = {
        "known_action_l2": float(torch.linalg.vector_norm(raw_target)),
        "theoretical_log_weight_variance": float(raw_cost),
        "theoretical_log10_expected_ess_fraction": float(-raw_cost / math.log(10.0)),
        "empirical_density": _density_summary(raw_known.weights),
        "empirical_maximum_absolute_weighted_factor_mean": float(
            torch.max(torch.abs(raw_known.weighted_return_mean))
        ),
        "interpretation": (
            "the exact action is representable, but the finite P sample contains no useful Q overlap"
        ),
    }

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_kind": "direct_gaussian_factor_entropic_oce",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": {
            "traded_return": "G_t=lambda*h+sqrt(h)*epsilon_t under P",
            "objective": "min_a log E_P[exp(-sum_t a_t'G_t)]",
            "known_solution": "a_t=lambda and dQ/dP proportional to exp(-sum_t lambda'G_t)",
            "learning": (
                "fixed-path natural-gradient root solve of E_Q[G]=0; no DLV or option-return "
                "loading matrix enters the action"
            ),
            "dlv_implication": "Delta z-h*mu_Q=L*G, so zero factor drift zeros every spanned DLV drift",
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "num_factors_per_week": len(regularized_lambda),
        "num_traded_coordinates": len(regularized_target),
        "time_increment_y": time_increment_y,
        "regularized_target": {
            "single_week_lambda_l2": float(torch.linalg.vector_norm(regularized_lambda)),
            "four_week_log_weight_variance": float(
                time_increment_y * torch.sum(regularized_target.square())
            ),
            "theoretical_expected_ess_fraction": float(
                torch.exp(-time_increment_y * torch.sum(regularized_target.square()))
            ),
        },
        "training": training_report,
        "independent_validation": validation_report,
        "raw_unregularized_control": raw_control,
    }
    write_json_atomic(result, arguments.output)
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
