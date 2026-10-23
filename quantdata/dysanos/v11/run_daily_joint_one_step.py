"""Audit one-step ATM leverage of a fitted daily joint state/spot pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.v11.atm_spot_correlation import (
    _antithetic_correlation_standard_errors,
    _decode_atm_volatilities,
)
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.daily_joint_ar_spot import (
    DailyJointArSpotFit,
    sample_daily_joint_ar_spot,
)
from quantdata.dysanos.v11.daily_joint_dynamics import _terminal_front_gross_returns
from quantdata.dysanos.v11.factorial_statistics import (
    AUDIT_BUSINESS_DTES,
    _leverage_values,
)
from quantdata.dysanos.v11.radius_cap_diagnostics import decoder_cap_diagnostics
from quantdata.dysanos.v11.tail_copula import normal_score_from_scale_mixture


def _load_fit(path: Path) -> DailyJointArSpotFit:
    with np.load(path, allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    transition = values["factor_transition"]
    return DailyJointArSpotFit(
        **values,
        spectral_radius=float(np.max(np.abs(np.linalg.eigvals(transition)))),
        spot_direction_fraction=float("nan"),
    )


def run(
    *,
    joint_checkpoint: Path,
    training_state_history: Path,
    daily_normal_parameters: Path,
    output: Path,
    path_count: int = 10_000,
    batch_size: int = 256,
    seed: int = 20261018,
    device: str = "cpu",
    tail_fit_report: Path | None = None,
) -> dict[str, object]:
    """Draw independent antithetic pairs conditional on 2024-12-31."""
    if path_count <= 0 or path_count % 2 or batch_size <= 0:
        raise ValueError("path count must be positive and even, batch size positive")
    model, audit = load_capped_joint_model(joint_checkpoint, device=device)
    history = torch.load(training_state_history, map_location="cpu", weights_only=False)
    states = history["states"].double().numpy()
    dates = history["dates"]
    fit = _load_fit(daily_normal_parameters)
    if states.shape != tuple(model.daily_states.shape) or len(states) != len(dates):
        raise ValueError("state history does not match decoder checkpoint")
    tail_config = None
    if tail_fit_report is None:
        next_states, spot_scores = sample_daily_joint_ar_spot(
            replace(fit, spot_score_mean=0.0, spot_score_std=1.0),
            states[-1],
            count=path_count,
            seed=seed,
        )
        next_states = next_states[0]
        spot_scores = spot_scores[0]
    else:
        tail_report = json.loads(tail_fit_report.read_text(encoding="utf-8"))
        tail_config = tail_report["fit"]["config"]
        variance = float(tail_config["stationary_log_variance"])
        scale = float(tail_config["unconditional_scale"])
        generator = np.random.default_rng(seed)
        half_epsilon = generator.standard_normal((path_count // 2, fit.rank))
        half_eta = generator.standard_normal(path_count // 2)
        latent_half = generator.normal(scale=np.sqrt(variance), size=path_count // 2)
        epsilon = np.concatenate((half_epsilon, -half_epsilon), axis=0)
        eta = np.concatenate((half_eta, -half_eta), axis=0)
        latent = np.concatenate((latent_half, latent_half), axis=0)
        multiplier = np.sqrt(scale * np.exp(latent - 0.5 * variance))
        centered = states[-1] - fit.state_mean
        factor_mean = fit.factor_intercept + (
            centered @ fit.factor_basis
        ) @ fit.factor_transition
        conditional_mean = fit.state_mean + factor_mean @ fit.factor_basis.T
        next_states = conditional_mean[None, :] + (
            (multiplier[:, None] * epsilon) @ fit.innovation_factor.T
        ) @ fit.factor_basis.T
        raw_spot = multiplier * (
            epsilon @ fit.gamma
            + np.sqrt(1.0 - float(fit.gamma @ fit.gamma)) * eta
        )
        spot_scores = normal_score_from_scale_mixture(
            raw_spot,
            stationary_log_variance=variance,
            unconditional_scale=scale,
        )
    gross = _terminal_front_gross_returns(model, states[-1], spot_scores)
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
        "schema": "dysanos-v11-daily-joint-one-step-v1",
        "innovation_law": "Normal" if tail_config is None else "Tail Diffusion",
        "tail_fit_report": (
            None if tail_fit_report is None else str(tail_fit_report.resolve())
        ),
        "status": "pilot_audit_complete",
        "checkpoint": audit,
        "daily_normal_parameters": str(daily_normal_parameters.resolve()),
        "simulation_origin_date": dates[-1],
        "path_count": path_count,
        "independent_antithetic_pairs": path_count // 2,
        "seed": seed,
        "business_dtes": list(AUDIT_BUSINESS_DTES),
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
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--training-state-history", type=Path, required=True)
    parser.add_argument("--daily-normal-parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-count", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tail-fit-report", type=Path)
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(
        f"daily joint {report['innovation_law']} one-step pilot "
        f"{report['status']}: {arguments.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
