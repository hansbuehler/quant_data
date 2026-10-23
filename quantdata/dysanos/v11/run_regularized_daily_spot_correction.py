"""One-origin shrinkage direct correction to the daily DYSANOS AR transition."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from quantdata.dysanos.v11.atm_spot_correlation import _decode_atm_volatilities
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.daily_joint_dynamics import _terminal_front_gross_returns


@dataclass(frozen=True, kw_only=True)
class CorrectionChoice:
    """Validation-selected low-rank ridge correction geometry."""

    response_rank: int
    ridge: float
    validation_mse: float


def _features(
    factors: np.ndarray,
    scores: np.ndarray,
    *,
    score_location: float | None = None,
    score_scale: float | None = None,
) -> np.ndarray:
    """State, spot, and state-times-spot features available at the origin."""
    location = scores.mean() if score_location is None else score_location
    scale = scores.std(ddof=1) if score_scale is None else score_scale
    standardized = (scores - location) / max(scale, 1.0e-6)
    return np.column_stack((factors, standardized, standardized**2 - 1.0, factors * standardized[:, None]))


def _fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1.0e-6)
    x = (features - mean) / scale
    intercept = target.mean(axis=0)
    y = target - intercept
    weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    return mean, scale, intercept, weights


def _predict(features: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mean, scale, intercept, weights = fit
    return intercept + ((features - mean) / scale) @ weights


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def run(
    *,
    checkpoint: Path,
    target_history: Path,
    dynamics_parameters: Path,
    checkpoint_output: Path,
    report_output: Path,
    paths: int = 1_024,
    seed: int = 20261019,
) -> dict[str, object]:
    """Select a correction on late 2024, then refit solely through 2024-12-31."""
    if paths <= 0 or paths % 2:
        raise ValueError("paths must be positive and antithetic-pair compatible")
    model, source_audit = load_capped_joint_model(checkpoint, device="cpu")
    history = torch.load(target_history, map_location="cpu", weights_only=False)
    states = history["states"].double().numpy()[-252:]
    dates = history["dates"][-252:]
    scores = np.asarray(history["front_spot_scores"], dtype=np.float64)
    with np.load(dynamics_parameters, allow_pickle=False) as stream:
        parameters = {key: stream[key] for key in stream.files}
    if scores.shape != (251,):
        raise ValueError("the daily target history must contain the trailing 251 spot scores")
    equilibrium = parameters["equilibrium_mean"]
    transition = parameters["transition"]
    loading = parameters["spot_loading"]
    score_mean = float(parameters["training_spot_score_mean"])
    origin, endpoint = states[:-1], states[1:]
    baseline = equilibrium + (origin - equilibrium) @ transition + (scores - score_mean)[:, None] * loading
    state_mean = origin.mean(axis=0)
    _, _, right = np.linalg.svd(origin - state_mean, full_matrices=False)
    basis = right[:8].T
    factors = (origin - state_mean) @ basis
    residual = (endpoint - baseline) @ basis
    features = _features(factors, scores)
    split = 209
    choices: list[CorrectionChoice] = []
    for rank in (1, 2, 4):
        centered = residual[:split] - residual[:split].mean(axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        response_basis = vectors[:rank].T
        projected = residual @ response_basis
        for ridge in (1.0, 10.0, 100.0, 1_000.0):
            fit = _fit_ridge(features[:split], projected[:split], ridge)
            predicted = _predict(features[split:], fit) @ response_basis.T
            value = float(np.mean((residual[split:] - predicted) ** 2))
            choices.append(
                CorrectionChoice(
                    response_rank=rank,
                    ridge=ridge,
                    validation_mse=value,
                )
            )
    selected = min(choices, key=lambda item: item.validation_mse)
    centered = residual - residual.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    response_basis = vectors[: selected.response_rank].T
    full_fit = _fit_ridge(features, residual @ response_basis, selected.ridge)
    fitted = _predict(features, full_fit) @ response_basis.T
    # Preserve the complete fitted AR innovation law.  The direct correction
    # changes only its conditional mean; replacing it by an 8-factor residual
    # would spuriously remove the 192 omitted state-noise directions.
    innovation_factor = parameters["innovation_cholesky"]

    generator = np.random.default_rng(seed)
    half = paths // 2
    score_half = generator.standard_normal(half)
    terminal_scores = scores.mean() + scores.std(ddof=1) * np.concatenate((score_half, -score_half))
    current_factor = (states[-1] - state_mean) @ basis
    terminal_features = _features(
        np.repeat(current_factor[None], paths, axis=0),
        terminal_scores,
        score_location=float(scores.mean()),
        score_scale=float(scores.std(ddof=1)),
    )
    correction = _predict(terminal_features, full_fit) @ response_basis.T
    terminal_baseline = equilibrium + (states[-1] - equilibrium) @ transition + (terminal_scores - score_mean)[:, None] * loading
    innovation_half = generator.standard_normal((half, states.shape[1])) @ innovation_factor.T
    innovations = np.concatenate((innovation_half, -innovation_half))
    next_states = terminal_baseline + correction @ basis.T + innovations
    current_atm = _decode_atm_volatilities(model, states[-1:], batch_size=1)[0]
    next_atm = _decode_atm_volatilities(model, next_states, batch_size=256)
    log_spot = np.log(_terminal_front_gross_returns(model, states[-1], terminal_scores))
    leverage = [_correlation(log_spot, np.log(next_atm[:, j]) - np.log(current_atm[j])) for j in range(next_atm.shape[1])]
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(checkpoint_output, basis=basis, response_basis=response_basis, innovation_factor=innovation_factor, feature_mean=full_fit[0], feature_scale=full_fit[1], response_intercept=full_fit[2], ridge_weights=full_fit[3])
    report: dict[str, object] = {"schema": "dysanos-v11-regularized-direct-correction-v1", "status": "complete", "scope": "252 dates through 2024-12-31; late-2024 validation selects shrinkage; final fit has no 2025 data", "window": [dates[0], dates[-1]], "transitions": 251, "validation_transitions": 42, "selected": {"response_rank": selected.response_rank, "ridge": selected.ridge, "validation_mse": selected.validation_mse}, "all_choices": [choice.__dict__ for choice in choices], "paths": paths, "business_dtes": (np.asarray(model.config.geometry.model_expiries_y) * 252.0).tolist(), "terminal_atm_spot_correlations": leverage, "source_audit": source_audit, "checkpoint": str(checkpoint_output.resolve())}
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--dynamics-parameters", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1_024)
    report = run(**vars(parser.parse_args()))
    print(f"regularized direct correction complete: {report['checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
