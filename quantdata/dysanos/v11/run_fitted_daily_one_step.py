"""Audit one-step ATM leverage from the full daily spot-joint DYSANOS fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from quantdata.dysanos.flex.market_encoder import write_json_atomic
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


def _simulate_one_step(
    initial_state: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    path_count: int,
    seed: int,
    tail_config: dict[str, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return next states and standard-normal front-spot scores in pair order."""
    if path_count <= 0 or path_count % 2:
        raise ValueError("path count must be positive and even")
    dimension = len(initial_state)
    mean = parameters["equilibrium_mean"]
    transition = parameters["transition"]
    factor = parameters["innovation_cholesky"]
    loading = parameters["spot_loading"]
    if (
        mean.shape != (dimension,)
        or transition.shape != (dimension, dimension)
        or factor.shape != (dimension, dimension)
        or loading.shape != (dimension,)
    ):
        raise ValueError("daily dynamics parameter dimensions do not match state")
    score_mean = float(parameters["training_spot_score_mean"])
    rng = np.random.default_rng(seed)
    half_spot = rng.standard_normal(path_count // 2)
    half_surface = rng.standard_normal((path_count // 2, dimension))
    raw_spot = np.concatenate((half_spot, -half_spot))
    surface_normal = np.concatenate((half_surface, -half_surface))
    if tail_config is None:
        spot_score = raw_spot
        surface_shock = surface_normal
    else:
        variance = float(tail_config["stationary_log_variance"])
        scale = float(tail_config["unconditional_scale"])
        if variance < 0.0 or scale <= 0.0:
            raise ValueError("invalid Tail Diffusion scale")
        latent_half = rng.normal(scale=np.sqrt(variance), size=path_count // 2)
        latent = np.concatenate((latent_half, latent_half))
        multiplier = np.sqrt(scale * np.exp(latent - 0.5 * variance))
        spot_score = normal_score_from_scale_mixture(
            multiplier * raw_spot,
            stationary_log_variance=variance,
            unconditional_scale=scale,
        )
        surface_shock = multiplier[:, None] * surface_normal
    predicted_state = mean + (initial_state - mean) @ transition
    next_states = (
        predicted_state[None, :]
        + (spot_score - score_mean)[:, None] * loading[None, :]
        + surface_shock @ factor.T
    )
    return next_states, spot_score


def run(
    *,
    joint_checkpoint: Path,
    target_history: Path,
    dynamics_parameters: Path,
    output: Path,
    path_count: int = 10_000,
    batch_size: int = 256,
    seed: int = 20261018,
    device: str = "cpu",
    tail_fit_report: Path | None = None,
) -> dict[str, object]:
    """Evaluate one-step leverage and decoder caps from the terminal train state."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    model, audit = load_capped_joint_model(joint_checkpoint, device=device)
    history = torch.load(target_history, map_location="cpu", weights_only=False)
    if history.get("schema") != "dysanos-v11-pre2025-daily-spot-joint-targets-v1":
        raise ValueError("not a daily spot-joint target history")
    states = history["states"].double().numpy()
    dates = history["dates"]
    if states.shape != tuple(model.daily_states.shape) or len(states) != len(dates):
        raise ValueError("state history does not match daily joint checkpoint")
    with np.load(dynamics_parameters, allow_pickle=False) as payload:
        parameters = {name: payload[name] for name in payload.files}
    tail_config = None
    if tail_fit_report is not None:
        tail_report = json.loads(tail_fit_report.read_text(encoding="utf-8"))
        if tail_report.get("schema") != "dysanos-v11-daily-joint-tail-diffusion-v1":
            raise ValueError("tail report does not match the daily joint model")
        tail_config = tail_report["fit"]["config"]
    next_states, spot_scores = _simulate_one_step(
        states[-1], parameters, path_count=path_count, seed=seed,
        tail_config=tail_config,
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
        "schema": "dysanos-v11-fitted-daily-one-step-v1",
        "status": "complete",
        "innovation_law": "Normal" if tail_config is None else "Tail Diffusion",
        "checkpoint": audit,
        "target_history": str(target_history.resolve()),
        "dynamics_parameters": str(dynamics_parameters.resolve()),
        "tail_fit_report": None if tail_fit_report is None else str(tail_fit_report.resolve()),
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
            for p in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
        "decoder_cap": decoder_cap_diagnostics(model, next_states),
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--dynamics-parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-count", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tail-fit-report", type=Path)
    args = parser.parse_args()
    report = run(**vars(args))
    print(f"daily {report['innovation_law']} one-step {report['status']}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
