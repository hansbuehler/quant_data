"""Audit a stationary Tail-Diffusion bridge from the capped weekly AR fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.presentation_hedging import exact_daily_transition
from quantdata.dysanos.v11.atm_spot_correlation import (
    _antithetic_correlation_standard_errors,
    _decode_atm_volatilities,
)
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.daily_joint_dynamics import _terminal_front_gross_returns
from quantdata.dysanos.v11.factorial_statistics import (
    AUDIT_BUSINESS_DTES,
    _leverage_values,
)
from quantdata.dysanos.v11.radius_cap_diagnostics import decoder_cap_diagnostics
from quantdata.dysanos.v11.tail_copula import normal_score_from_scale_mixture


def run(
    *,
    joint_checkpoint: Path,
    training_state_history: Path,
    normal_one_step_report: Path,
    weekly_tail_fit_report: Path,
    output: Path,
    path_count: int = 10_000,
    batch_size: int = 256,
    seed: int = 20261018,
    device: str = "cpu",
) -> dict[str, object]:
    """Draw a terminal one-day pilot, preserving the front SANOS marginal."""
    if path_count <= 0 or path_count % 2 or batch_size <= 0:
        raise ValueError("path count must be positive and even")
    model, audit = load_capped_joint_model(joint_checkpoint, device=device)
    history = torch.load(training_state_history, map_location="cpu", weights_only=False)
    states = history["states"].double().numpy()
    dates = history["dates"]
    if states.shape != tuple(model.daily_states.shape):
        raise ValueError("terminal state history and decoder checkpoint differ")
    normal = json.loads(normal_one_step_report.read_text(encoding="utf-8"))
    if normal["simulation_origin_date"] != dates[-1]:
        raise ValueError("Normal and Tail pilots have different origins")
    copula = normal["front_spot_copula"]
    loading = np.asarray(copula["state_loading"], dtype=np.float64)
    orthogonal_std = float(copula["orthogonal_std"])
    total_std = float(copula["total_std"])
    tail = json.loads(weekly_tail_fit_report.read_text(encoding="utf-8"))
    config = tail["fit"]["config"]
    variance = float(config["stationary_log_variance"])
    scale = float(config["unconditional_scale"])
    transition_t, factor_t, transition_audit = exact_daily_transition(model)
    transition = transition_t.cpu().double().numpy()
    factor = factor_t.cpu().double().numpy()
    rng = np.random.default_rng(seed)
    half_epsilon = rng.standard_normal((path_count // 2, model.config.state_size))
    half_eta = rng.standard_normal(path_count // 2)
    latent_half = rng.normal(scale=np.sqrt(variance), size=path_count // 2)
    epsilon = np.concatenate((half_epsilon, -half_epsilon), axis=0)
    eta = np.concatenate((half_eta, -half_eta), axis=0)
    latent = np.concatenate((latent_half, latent_half), axis=0)
    multiplier = np.sqrt(scale * np.exp(latent - 0.5 * variance))
    mean = model.equilibrium_mean.detach().cpu().double().numpy()
    conditional_mean = mean + (states[-1] - mean) @ transition
    next_states = conditional_mean[None, :] + (
        multiplier[:, None] * epsilon
    ) @ factor.T
    raw_spot = multiplier * (
        epsilon @ loading + orthogonal_std * eta
    ) / total_std
    normal_spot_score = normal_score_from_scale_mixture(
        raw_spot,
        stationary_log_variance=variance,
        unconditional_scale=scale,
    )
    gross = _terminal_front_gross_returns(model, states[-1], normal_spot_score)
    terminal_atm = _decode_atm_volatilities(
        model, next_states, batch_size=batch_size
    )
    initial_atm = _decode_atm_volatilities(
        model, states[-1][None, :], batch_size=1
    )[0]
    model_dtes = np.asarray(model.config.geometry.model_expiries_y) * 252.0
    indices = []
    for tenor in AUDIT_BUSINESS_DTES:
        matches = np.flatnonzero(np.isclose(model_dtes, tenor, rtol=0, atol=1e-9))
        if len(matches) != 1:
            raise ValueError(f"audit tenor {tenor}D unavailable")
        indices.append(int(matches[0]))
    spot_return = np.log(gross)
    iv_change = np.log(terminal_atm[:, indices] / initial_atm[None, indices])
    correlation, covariance, slope = _leverage_values(spot_return, iv_change)
    report: dict[str, object] = {
        "schema": "dysanos-v11-weekly-tail-one-step-v1",
        "status": "stationary_tail_pilot_complete",
        "checkpoint": audit,
        "normal_one_step_report": str(normal_one_step_report.resolve()),
        "weekly_tail_fit_report": str(weekly_tail_fit_report.resolve()),
        "simulation_origin_date": dates[-1],
        "path_count": path_count,
        "independent_antithetic_pairs": path_count // 2,
        "seed": seed,
        "business_dtes": list(AUDIT_BUSINESS_DTES),
        "tail_scale_assumption": "stationary weekly OU log variance applied to the one-day Gaussian bridge",
        "front_marginal_transform": "Gaussianize the scale-mixture spot shock before applying the decoded front SANOS quantile",
        "leverage": {
            "correlation": correlation.tolist(),
            "covariance": covariance.tolist(),
            "slope": slope.tolist(),
            "antithetic_pair_correlation_standard_error": (
                _antithetic_correlation_standard_errors(
                    spot_return, iv_change
                ).tolist()
            ),
        },
        "spot_return_quantiles": {
            str(p): float(np.quantile(spot_return, p))
            for p in (0.01, 0.05, 0.50, 0.95, 0.99)
        },
        "decoder_cap": decoder_cap_diagnostics(model, next_states),
        "transition_diagnostics": transition_audit,
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--training-state-history", type=Path, required=True)
    parser.add_argument("--normal-one-step-report", type=Path, required=True)
    parser.add_argument("--weekly-tail-fit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-count", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(f"weekly Tail one-step pilot {report['status']}: {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
