"""Fit a last-252-session, rank-201 daily state/spot Gaussian AR pilot."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import torch

from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.daily_joint_ar_spot import fit_daily_joint_ar_spot
from quantdata.dysanos.v11.daily_joint_dynamics import _historical_spot_scores
from quantdata.dysanos.v11.native_atm_history import load_atm_history


def run(
    *,
    joint_checkpoint: Path,
    training_state_history: Path,
    historical_sanos_history: Path,
    historical_dataset: Path,
    output_report: Path,
    output_parameters: Path,
    lookback_sessions: int = 252,
    state_rank: int = 200,
    device: str = "cpu",
) -> dict[str, object]:
    """Estimate a stable daily AR and its front-spot dependence on training data."""
    model, checkpoint_audit = load_capped_joint_model(joint_checkpoint, device=device)
    history = torch.load(training_state_history, map_location="cpu", weights_only=False)
    dates = tuple(date.fromisoformat(value) for value in history["dates"])
    states = history["states"].double().numpy()
    if (
        states.shape != tuple(model.daily_states.shape)
        or dates[-1] >= date(2025, 1, 1)
        or lookback_sessions <= state_rank + 5
        or lookback_sessions > len(states)
    ):
        raise ValueError("daily joint fit requires sufficient strictly pre-2025 states")
    observed_dates, _, _, simple_spot = load_atm_history(
        history_path=historical_sanos_history,
        dataset_path=historical_dataset,
    )
    if list(observed_dates[: len(dates)]) != [value.isoformat() for value in dates]:
        raise ValueError("daily joint spot and latent-state dates differ")
    selected_states = states[-lookback_sessions:]
    selected_gross = 1.0 + simple_spot[len(states) - lookback_sessions : len(states) - 1]
    spot_scores = _historical_spot_scores(
        model, selected_states[:-1], selected_gross, batch_size=128
    )
    fit = fit_daily_joint_ar_spot(
        selected_states,
        spot_scores,
        rank=state_rank,
        transition_structure="diagonal",
    )
    output_parameters.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_parameters,
        state_mean=fit.state_mean,
        factor_basis=fit.factor_basis,
        factor_intercept=fit.factor_intercept,
        factor_transition=fit.factor_transition,
        innovation_factor=fit.innovation_factor,
        spot_score_mean=fit.spot_score_mean,
        spot_score_std=fit.spot_score_std,
        gamma=fit.gamma,
    )
    report: dict[str, object] = {
        "schema": "dysanos-v11-daily-joint-normal-v1",
        "status": "pilot_complete",
        "estimation": "closed-form last-252-session diagonal daily AR and joint spot score on the frozen 600-epoch affine states",
        "transition_structure": "diagonal",
        "not_joint_decoder_refit": True,
        "checkpoint": checkpoint_audit,
        "joint_checkpoint_sha256": checkpoint_digest(joint_checkpoint),
        "training_range": [dates[-lookback_sessions].isoformat(), dates[-1].isoformat()],
        "lookback_sessions": lookback_sessions,
        "daily_transitions": lookback_sessions - 1,
        "state_rank": fit.rank,
        "joint_state_spot_innovation_dimension": fit.rank + 1,
        "spectral_radius": fit.spectral_radius,
        "spot_direction_fraction": fit.spot_direction_fraction,
        "spot_score_mean": fit.spot_score_mean,
        "spot_score_std": fit.spot_score_std,
        "gamma_norm": float(np.linalg.norm(fit.gamma)),
        "parameters": str(output_parameters.resolve()),
    }
    write_json_atomic(report, output_report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--training-state-history", type=Path, required=True)
    parser.add_argument("--historical-sanos-history", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-parameters", type=Path, required=True)
    parser.add_argument("--lookback-sessions", type=int, default=252)
    parser.add_argument("--state-rank", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(
        f"daily joint Normal pilot: rank={report['joint_state_spot_innovation_dimension']} "
        f"spectral_radius={report['spectral_radius']:.5f} "
        f"output={arguments.output_report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
