"""Exact 5–20-session innovation-radius audit for the daily joint law."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.stats import chi2

from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.radius_cap_diagnostics import decoder_cap_diagnostics


def daily_radius_panels(
    states: np.ndarray,
    dates: tuple[date, ...],
    *,
    equilibrium_mean: np.ndarray,
    transition: np.ndarray,
    innovation_cholesky: np.ndarray,
    spot_loading: np.ndarray,
    training_spot_score_mean: float,
    forward_start: date = date(2025, 1, 1),
) -> dict[int, dict[str, object]]:
    """Whiten observed 5/10/15/20-session returns under the daily law."""
    values = np.asarray(states, dtype=np.float64)
    center = np.asarray(equilibrium_mean, dtype=np.float64)
    matrix = np.asarray(transition, dtype=np.float64)
    factor = np.asarray(innovation_cholesky, dtype=np.float64)
    loading = np.asarray(spot_loading, dtype=np.float64)
    if (
        values.ndim != 2
        or len(dates) != len(values)
        or center.shape != (values.shape[1],)
        or matrix.shape != (values.shape[1], values.shape[1])
        or factor.shape != matrix.shape
        or loading.shape != center.shape
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("daily radius geometry is inconsistent")
    dimension = values.shape[1]
    one_day_covariance = factor @ factor.T + np.outer(loading, loading)
    gaussian_p99 = float(chi2.ppf(0.99, dimension) / dimension)
    report: dict[int, dict[str, object]] = {}
    for weeks in (1, 2, 3, 4):
        sessions = 5 * weeks
        if len(values) <= sessions:
            raise ValueError("not enough daily states for radius horizon")
        transition_power = np.linalg.matrix_power(matrix, sessions)
        covariance = np.zeros_like(one_day_covariance)
        power = np.eye(dimension)
        deterministic_shift = np.zeros(dimension)
        for _ in range(sessions):
            covariance += power.T @ one_day_covariance @ power
            deterministic_shift += loading @ power
            power = power @ matrix
        covariance = 0.5 * (covariance + covariance.T)
        cholesky = np.linalg.cholesky(covariance)
        residual = values[sessions:] - (
            center + (values[:-sessions] - center) @ transition_power
            - training_spot_score_mean * deterministic_shift
        )
        white = solve_triangular(cholesky, residual.T, lower=True).T
        radii = np.mean(white * white, axis=1)
        forward = np.asarray(
            [dates[index] >= forward_start for index in range(sessions, len(dates))]
        )

        def summarize(selection: np.ndarray) -> dict[str, float | int]:
            selected = radii[selection]
            if len(selected) == 0:
                return {"observations": 0}
            return {
                "observations": len(selected),
                "mean": float(np.mean(selected)),
                "median": float(np.median(selected)),
                "p95": float(np.quantile(selected, 0.95)),
                "p99": float(np.quantile(selected, 0.99)),
                "maximum": float(np.max(selected)),
                "fraction_above_gaussian_p99": float(np.mean(selected > gaussian_p99)),
            }

        report[weeks] = {
            "sessions": sessions,
            "gaussian_p99": gaussian_p99,
            "pre_evaluation": summarize(~forward),
            "forward_evaluation": summarize(forward),
        }
    return report


def _load_states(path: Path) -> tuple[tuple[date, ...], np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    dates = tuple(date.fromisoformat(item) for item in payload["dates"])
    states = payload["states"].double().numpy()
    if len(dates) != len(states):
        raise ValueError("state-history dates and rows differ")
    return dates, states


def run(
    *,
    joint_checkpoint: Path,
    target_history: Path,
    forward_state_history: Path,
    dynamics_parameters: Path,
    output: Path,
) -> dict[str, object]:
    """Audit empirical radius tails and decoder caps in both periods."""
    model, checkpoint_audit = load_capped_joint_model(joint_checkpoint, device="cpu")
    train_dates, train_states = _load_states(target_history)
    forward_dates, forward_states = _load_states(forward_state_history)
    if train_states.shape != tuple(model.daily_states.shape) or not train_dates[-1] < forward_dates[0]:
        raise ValueError("daily training and forward state histories differ")
    with np.load(dynamics_parameters, allow_pickle=False) as stream:
        parameters = {name: stream[name] for name in stream.files}
    states = np.concatenate((train_states, forward_states), axis=0)
    dates = train_dates + forward_dates
    report: dict[str, object] = {
        "schema": "dysanos-v11-daily-joint-radius-cap-v1",
        "checkpoint": checkpoint_audit,
        "definition": "squared Mahalanobis state radius / 200 over exact 5–20-session transitions; covariance includes front-spot loading",
        "weeks": daily_radius_panels(states, dates, **parameters),
        "decoder_cap": {
            "pre_evaluation": decoder_cap_diagnostics(model, train_states),
            "forward_evaluation": decoder_cap_diagnostics(model, forward_states),
        },
    }
    write_json_atomic(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--forward-state-history", type=Path, required=True)
    parser.add_argument("--dynamics-parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(**vars(args))
    print(f"daily radius/cap audit complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
