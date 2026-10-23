"""Fit a Tail Diffusion to the daily joint state/spot AR innovation radius."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import torch

from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.stochastic_volatility_diffusion import (
    fit_log_variance_diffusion,
)
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.daily_joint_dynamics import _historical_spot_scores
from quantdata.dysanos.v11.native_atm_history import load_atm_history


def joint_white_innovations(
    states: np.ndarray,
    spot_scores: np.ndarray,
    *,
    state_mean: np.ndarray,
    factor_basis: np.ndarray,
    factor_intercept: np.ndarray,
    factor_transition: np.ndarray,
    innovation_factor: np.ndarray,
    spot_score_mean: float,
    spot_score_std: float,
    gamma: np.ndarray,
) -> np.ndarray:
    """Recover independent normal coordinates of a fitted daily joint AR."""
    if len(states) != len(spot_scores) + 1:
        raise ValueError("states and spot scores must describe adjacent sessions")
    factors_now = (states[:-1] - state_mean) @ factor_basis
    factors_next = (states[1:] - state_mean) @ factor_basis
    residual = factors_next - factor_intercept - factors_now @ factor_transition
    epsilon = np.linalg.solve(innovation_factor, residual.T).T
    remaining_variance = 1.0 - float(gamma @ gamma)
    if remaining_variance <= 0.0 or spot_score_std <= 0.0:
        raise ValueError("daily joint spot law is not positive definite")
    normalized_spot = (spot_scores - spot_score_mean) / spot_score_std
    eta = (normalized_spot - epsilon @ gamma) / np.sqrt(remaining_variance)
    whitened = np.column_stack((epsilon, eta))
    if not np.all(np.isfinite(whitened)):
        raise ValueError("non-finite daily joint innovations")
    return whitened


def run(
    *,
    joint_checkpoint: Path,
    training_state_history: Path,
    historical_sanos_history: Path,
    historical_dataset: Path,
    daily_normal_parameters: Path,
    output: Path,
    lookback_sessions: int = 252,
    device: str = "cpu",
) -> dict[str, object]:
    """Fit one daily scale process without altering the shared decoder or AR."""
    model, checkpoint_audit = load_capped_joint_model(joint_checkpoint, device=device)
    history = torch.load(training_state_history, map_location="cpu", weights_only=False)
    dates = tuple(date.fromisoformat(value) for value in history["dates"])
    states = history["states"].double().numpy()
    if (
        states.shape != tuple(model.daily_states.shape)
        or dates[-1] >= date(2025, 1, 1)
        or lookback_sessions < 52
        or lookback_sessions > len(states)
    ):
        raise ValueError("daily Tail Diffusion requires sufficient pre-2025 states")
    observed_dates, _, _, simple_spot = load_atm_history(
        history_path=historical_sanos_history,
        dataset_path=historical_dataset,
    )
    if list(observed_dates[: len(dates)]) != [value.isoformat() for value in dates]:
        raise ValueError("daily spot and state dates differ")
    selected_states = states[-lookback_sessions:]
    selected_gross = 1.0 + simple_spot[len(states) - lookback_sessions : len(states) - 1]
    spot_scores = _historical_spot_scores(
        model, selected_states[:-1], selected_gross, batch_size=128
    )
    with np.load(daily_normal_parameters, allow_pickle=False) as parameters:
        whitened = joint_white_innovations(
            selected_states,
            spot_scores,
            **{name: parameters[name] for name in parameters.files},
        )
    radii = np.mean(whitened * whitened, axis=1)
    starts = np.zeros(len(radii), dtype=np.bool_)
    starts[0] = True
    fit = fit_log_variance_diffusion(
        radii, starts, dimension=whitened.shape[1]
    )
    report: dict[str, object] = {
        "schema": "dysanos-v11-daily-joint-tail-diffusion-v1",
        "status": "complete" if fit.success else "optimizer_warning",
        "estimation": "daily joint AR frozen; OU log-variance fitted to its 201-dimensional white residual radius",
        "not_joint_decoder_refit": True,
        "checkpoint": checkpoint_audit,
        "daily_normal_parameters": str(daily_normal_parameters.resolve()),
        "training_range": [dates[-lookback_sessions].isoformat(), dates[-1].isoformat()],
        "daily_residual_count": len(radii),
        "innovation_dimension": whitened.shape[1],
        "whitened_radius_quantiles": {
            str(p): float(np.quantile(radii, p)) for p in (0.50, 0.90, 0.95, 0.99)
        },
        "fit": fit.to_dict(step_sessions=1),
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--training-state-history", type=Path, required=True)
    parser.add_argument("--historical-sanos-history", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--daily-normal-parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lookback-sessions", type=int, default=252)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(f"daily Tail Diffusion {report['status']}: {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
