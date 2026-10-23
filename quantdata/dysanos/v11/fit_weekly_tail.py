"""Fit a no-replay Tail Diffusion to pre-2025 weekly joint-AR residuals."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_triangular

from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.run_score_driven_student_t import (
    _filter_order,
    _segment_starts,
)
from quantdata.dysanos.flex.stochastic_volatility_diffusion import (
    fit_log_variance_diffusion,
)
from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs
from quantdata.dysanos.v11.capped_model import load_capped_joint_model


def run(
    *,
    joint_checkpoint: Path,
    training_state_history: Path,
    output: Path,
) -> dict[str, object]:
    """Fit only the innovation law; preserve the 600-epoch decoder and AR."""
    model, checkpoint_audit = load_capped_joint_model(
        joint_checkpoint, device="cpu"
    )
    payload = torch.load(training_state_history, map_location="cpu", weights_only=False)
    dates = tuple(date.fromisoformat(value) for value in payload["dates"])
    states = payload["states"].double().numpy()
    if states.shape != tuple(model.daily_states.shape) or dates[-1] >= date(2025, 1, 1):
        raise ValueError("weekly Tail Diffusion must use only pre-2025 fitted states")
    with torch.inference_mode():
        mean = model.equilibrium_mean.cpu().double().numpy()
        transition = model.weekly_transition().cpu().double().numpy()
        factor = model.innovation_cholesky().cpu().double().numpy()
    origin, endpoint = exact_weekly_pairs(dates, weeks=1)
    residual = states[endpoint] - mean - (states[origin] - mean) @ transition
    white = solve_triangular(factor, residual.T, lower=True).T
    normalized_radii = np.sum(white * white, axis=1) / model.config.state_size
    order = _filter_order(origin, dates, sessions=5)
    sorted_origin = origin[order]
    sorted_endpoint = endpoint[order]
    fit = fit_log_variance_diffusion(
        normalized_radii[order],
        _segment_starts(sorted_origin, sorted_endpoint),
        dimension=model.config.state_size,
    )
    report: dict[str, object] = {
        "schema": "dysanos-v11-weekly-tail-diffusion-v1",
        "status": "complete" if fit.success else "optimizer_warning",
        "shared_decoder_and_ar": "frozen at the capped 600-epoch weekly-Normal fit",
        "joint_checkpoint": checkpoint_audit,
        "joint_checkpoint_sha256": checkpoint_digest(joint_checkpoint),
        "training_state_history": str(training_state_history.resolve()),
        "training_date_range": [dates[0].isoformat(), dates[-1].isoformat()],
        "weekly_residual_count": len(origin),
        "innovation_dimension": model.config.state_size,
        "fit": fit.to_dict(step_sessions=5),
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--training-state-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(
        f"weekly Tail Diffusion {report['status']}: "
        f"persistence={report['fit']['config']['persistence']:.5f} "
        f"output={arguments.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
