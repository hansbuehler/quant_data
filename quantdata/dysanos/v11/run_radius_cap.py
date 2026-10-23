"""Audit 1--4-week frozen-model radii and decoder DLV-cap activity."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import torch

from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.radius_cap_diagnostics import (
    decoder_cap_diagnostics,
    innovation_radius_panels,
)


def _load_state_history(path: Path) -> tuple[tuple[date, ...], np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_dates = payload.get("dates")
    states = payload.get("states")
    if not isinstance(raw_dates, list) or not isinstance(states, torch.Tensor):
        raise ValueError(f"invalid state history in {path}")
    dates = tuple(date.fromisoformat(value) for value in raw_dates)
    if states.ndim != 2 or len(states) != len(dates):
        raise ValueError(f"unaligned state history in {path}")
    return dates, states.cpu().double().numpy()


def run(
    *,
    joint_checkpoint: Path,
    training_state_history: Path,
    forward_state_history: Path | None,
    output: Path,
    device: str = "cpu",
) -> dict[str, object]:
    """Compare realized radius tails against the frozen Gaussian AR reference."""
    model, checkpoint_audit = load_capped_joint_model(joint_checkpoint, device=device)
    train_dates, train_states = _load_state_history(training_state_history)
    if train_states.shape != tuple(model.daily_states.shape):
        raise ValueError("training state history does not match the joint checkpoint")
    with torch.inference_mode():
        stored = model.daily_states.detach().cpu().double().numpy()
        if not np.allclose(train_states, stored, rtol=0.0, atol=1.0e-7):
            raise ValueError("training history states differ from the joint checkpoint")
        mean = model.equilibrium_mean.detach().cpu().double().numpy()
        transition = model.weekly_transition().cpu().double().numpy()
        factor = model.innovation_cholesky().cpu().double().numpy()

    dates = train_dates
    states = train_states
    cap_panels = {"pre_evaluation": decoder_cap_diagnostics(model, train_states)}
    if forward_state_history is not None:
        forward_dates, forward_states = _load_state_history(forward_state_history)
        if not train_dates[-1] < forward_dates[0]:
            raise ValueError("forward state dates overlap the training sample")
        dates += forward_dates
        states = np.concatenate((train_states, forward_states), axis=0)
        cap_panels["forward_evaluation"] = decoder_cap_diagnostics(model, forward_states)

    report: dict[str, object] = {
        "schema": "dysanos-v11-radius-cap-v1",
        "checkpoint": checkpoint_audit,
        "training_state_history": str(training_state_history.resolve()),
        "forward_state_history": (
            None if forward_state_history is None else str(forward_state_history.resolve())
        ),
        "definition": "squared Mahalanobis radius divided by 200 using the actual compounded weekly AR covariance",
        "weeks": innovation_radius_panels(
            states,
            dates,
            mean=mean,
            weekly_transition=transition,
            weekly_innovation_factor=factor,
        ),
        "decoder_cap": cap_panels,
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--training-state-history", type=Path, required=True)
    parser.add_argument("--forward-state-history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(f"radius/cap report complete: {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
