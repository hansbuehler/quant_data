"""Fit the scalar Tail Diffusion after a capped daily spot-joint AR refit."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_triangular

from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.stochastic_volatility_diffusion import (
    fit_log_variance_diffusion,
)
from quantdata.dysanos.v11.capped_model import load_capped_joint_model


def run(
    *,
    joint_checkpoint: Path,
    target_history: Path,
    output: Path,
    lookback_sessions: int = 252,
) -> dict[str, object]:
    """Fit latent log variance to the spot-conditioned surface residual radius."""
    model, checkpoint_audit = load_capped_joint_model(
        joint_checkpoint, device="cpu", maximum_epochs=600
    )
    payload = torch.load(target_history, map_location="cpu", weights_only=False)
    if payload.get("schema") != "dysanos-v11-pre2025-daily-spot-joint-targets-v1":
        raise ValueError("not a daily spot-joint target history")
    dates = tuple(date.fromisoformat(item) for item in payload["dates"])
    states = payload["states"].double().numpy()
    scores = np.asarray(payload["front_spot_scores"], dtype=np.float64)
    if (
        dates[-1] >= date(2025, 1, 1)
        or len(states) != model.num_days
        or lookback_sessions < 52
        or lookback_sessions > len(states)
        or scores.shape != (lookback_sessions - 1,)
        or np.any(~np.isfinite(scores))
    ):
        raise ValueError("tail fit needs aligned pre-2025 daily states and scores")
    with torch.inference_mode():
        mean = model.equilibrium_mean.cpu().double().numpy()
        transition = model.weekly_transition().cpu().double().numpy()
        factor = model.innovation_cholesky().cpu().double().numpy()
    recent = states[-lookback_sessions:]
    residual = recent[1:] - mean - (recent[:-1] - mean) @ transition
    centered_score = scores - float(np.mean(scores))
    loading = np.sum(residual * centered_score[:, None], axis=0) / (
        np.sum(centered_score**2) + 1.0e-6
    )
    residual -= centered_score[:, None] * loading[None, :]
    white = solve_triangular(factor, residual.T, lower=True).T
    radius = np.mean(white * white, axis=1)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("nonpositive or nonfinite whitened innovation radius")
    starts = np.zeros(len(radius), dtype=np.bool_)
    starts[0] = True
    fit = fit_log_variance_diffusion(radius, starts, dimension=white.shape[1])
    report: dict[str, object] = {
        "schema": "dysanos-v11-daily-joint-tail-diffusion-v1",
        "status": "complete" if fit.success else "optimizer_warning",
        "joint_checkpoint": checkpoint_audit,
        "joint_checkpoint_sha256": checkpoint_digest(joint_checkpoint),
        "target_history": str(target_history.resolve()),
        "training_range": [dates[-lookback_sessions].isoformat(), dates[-1].isoformat()],
        "daily_residual_count": len(radius),
        "innovation_dimension": white.shape[1],
        "spot_loading_norm": float(np.linalg.norm(loading)),
        "whitened_radius_quantiles": {
            str(p): float(np.quantile(radius, p)) for p in (0.5, 0.9, 0.95, 0.99)
        },
        "estimation": "scalar OU log variance fitted after the 600-epoch joint decoder/AR fit",
        "fit": fit.to_dict(step_sessions=1),
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lookback-sessions", type=int, default=252)
    args = parser.parse_args()
    report = run(**vars(args))
    print(f"daily joint Tail Diffusion {report['status']}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
