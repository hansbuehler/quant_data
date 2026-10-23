"""Jointly refit DYSANOS quotes and one-session state/spot AR dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from quantdata.dysanos.flex.config import (
    FlexObjectiveConfig,
    JointDirectDynamicsConfig,
)
from quantdata.dysanos.flex.joint_direct import (
    JointDirectTrainConfig,
    JointInitialization,
    compact_evaluation,
    fit_joint_direct_model,
    initialize_joint_model,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import JointDirectDynamicsModel
from quantdata.dysanos.flex.run_all_history_joint_direct import (
    _geometry_from_completion,
    _market_config_from_completion,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _load_markets,
    _market_dates_from_completion,
)
from quantdata.dysanos.flex.training import (
    TrainingConvergenceError,
    evaluate_model,
    materialize_day,
)
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.dysanos.v11.daily_joint_dynamics import _historical_spot_scores
from quantdata.dysanos.v11.native_atm_history import load_atm_history
from quantdata.dysanos.v11.run_strict_pre2025_direct import (
    _capped_joint_checkpoint,
    _train_count,
)
from quantdata.torch import BaseEnvironment


def _identity_with_spot(
    source_checkpoint: Path,
    origin: np.ndarray,
    endpoint: np.ndarray,
    scores: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(checkpoint_digest(source_checkpoint).encode("ascii"))
    for array in (origin, endpoint, scores):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def run(
    *,
    weekly_checkpoint: Path,
    training_state_history: Path,
    native_completion: Path,
    native_atm_history: Path,
    historical_dataset: Path,
    checkpoint_directory: Path,
    report_directory: Path,
    ticker: str = "SPX",
    workers: int = 4,
    device: str = "cuda:0",
    joint_epochs: int = 600,
    lookback_sessions: int = 252,
) -> dict[str, object]:
    """Fit one-day dynamics with the spot Gaussian score in every AR step."""
    if not 0 < joint_epochs <= 600:
        raise ValueError("joint epoch cap must lie in [1, 600]")
    source_model, source_audit = load_capped_joint_model(
        weekly_checkpoint, device="cpu", maximum_epochs=600
    )
    source_history = torch.load(
        training_state_history, map_location="cpu", weights_only=False
    )
    dates = _market_dates_from_completion(native_completion)
    train_count = _train_count(dates)
    if lookback_sessions < 52 or lookback_sessions > train_count:
        raise ValueError("daily dynamics lookback must cover 52 to all training sessions")
    if (
        train_count != source_model.num_days
        or source_history["dates"]
        != [value.isoformat() for value in dates[:train_count]]
    ):
        raise ValueError("weekly source and daily market dates differ")
    completion = json.loads(native_completion.read_text(encoding="utf-8"))
    geometry = _geometry_from_completion(
        completion, requested_model_business_dtes=None
    )
    if geometry != source_model.config.geometry:
        raise ValueError("weekly source and daily native geometry differ")
    observed_dates, _, _, spot_return = load_atm_history(
        history_path=native_atm_history, dataset_path=historical_dataset
    )
    if list(observed_dates[:train_count]) != source_history["dates"]:
        raise ValueError("native ATM and weekly source dates differ")
    source_states = source_history["states"].double().numpy()
    if source_states.shape != tuple(source_model.daily_states.shape):
        raise ValueError("source states and weekly checkpoint differ")
    if not np.allclose(
        source_states,
        source_model.daily_states.detach().cpu().double().numpy(),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise ValueError("source state history is not the weekly checkpoint")
    origin = np.arange(train_count - lookback_sessions, train_count - 1, dtype=np.int64)
    endpoint = origin + 1
    score = _historical_spot_scores(
        source_model,
        source_states[origin],
        1.0 + spot_return[origin],
        batch_size=128,
    )
    if np.any(~np.isfinite(score)):
        raise ValueError("non-finite historical front spot scores")
    source_identity = _identity_with_spot(weekly_checkpoint, origin, endpoint, score)
    markets = _load_markets(
        ticker=ticker,
        dates=dates[:train_count],
        geometry=geometry,
        market_config=_market_config_from_completion(completion),
        workers=workers,
    )
    train_days = markets
    env = BaseEnvironment(precision=32, device=device, debug=False)
    model_config = JointDirectDynamicsConfig(
        label="v11-pre2025-dte1-daily-spot-joint-direct-200",
        group="dysanos/v11/pre2025-factorial",
        state_size=200,
        geometry=geometry,
        minimum_mean_reversion_per_week=2.0e-4,
    )
    model = JointDirectDynamicsModel(env, num_days=train_count, config=model_config)
    initialization = JointInitialization(
        states=source_history["states"].double(),
        decoder_weight=source_model.raw_decoder.I0.weight.detach().cpu().double(),
        decoder_bias=source_model.raw_decoder.I0.bias.detach().cpu().double(),
        maximum_raw_reconstruction_error=0.0,
        minimum_factor_scale=1.0,
        maximum_factor_scale=1.0,
    )
    initialization_audit = initialize_joint_model(
        model, initialization, origin_indices=origin, endpoint_indices=endpoint,
        diagonal_transition=True, covariance_shrinkage=0.5,
    )
    train_config = JointDirectTrainConfig(
        num_epochs=joint_epochs,
        day_batch_size=16,
        state_learning_rate=3.0e-5,
        decoder_learning_rate=1.0e-5,
        dynamics_learning_rate=1.0e-4,
        warmup_epochs=min(20, joint_epochs - 1),
        dynamics_ramp_epochs=min(50, joint_epochs),
        report_every=1,
        checkpoint_every=5,
        compile_sanos_solver=device.startswith("cuda"),
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / "daily-spot-joint-direct-200.pt"
    with FileLock(str(checkpoint.with_suffix(".lock"))):
        try:
            history, convergence = fit_joint_direct_model(
                model,
                train_days,
                dates=dates[:train_count],
                objective_config=FlexObjectiveConfig(fit_weight=0.05),
                train_config=train_config,
                checkpoint_path=checkpoint,
                source_decoder_sha256=source_identity,
                env=env,
                transition_pair_indices=(origin, endpoint),
                transition_spot_scores=score,
            )
        except TrainingConvergenceError as error:
            weights, history, convergence = _capped_joint_checkpoint(
                checkpoint,
                train_config=train_config,
                model_config=model_config,
                market_fingerprints=tuple(
                    day.training_fingerprint for day in train_days
                ),
                source_decoder_sha256=source_identity,
                error=error,
            )
            model.load_state_dict(weights)
            print(
                f"daily-spot joint accepted at {joint_epochs}-epoch cap without convergence",
                flush=True,
            )
    targets = checkpoint_directory / "daily-spot-joint-direct-200-targets.pt"
    with torch.no_grad():
        state_tensor = model.daily_states.detach().double()
        mean_tensor = model.equilibrium_mean.detach().double()
        transition_tensor = model.weekly_transition().detach().double()
        origin_tensor = torch.as_tensor(origin, dtype=torch.long, device=env.device)
        endpoint_tensor = torch.as_tensor(endpoint, dtype=torch.long, device=env.device)
        residual_tensor = state_tensor[endpoint_tensor] - (
            mean_tensor + (state_tensor[origin_tensor] - mean_tensor) @ transition_tensor
        )
        score_tensor = torch.as_tensor(score, dtype=torch.float64, device=env.device)
        score_mean = torch.mean(score_tensor)
        centered_score = score_tensor - score_mean
        spot_loading = (
            torch.sum(residual_tensor * centered_score[:, None], dim=0)
            / (torch.sum(centered_score.square()) + 1.0e-6)
        )
        surface_innovation = residual_tensor - centered_score[:, None] * spot_loading
        fitted_cholesky = model.innovation_cholesky().detach().double()
    dynamics_parameters = checkpoint_directory / "daily-spot-joint-dynamics.npz"
    np.savez_compressed(
        dynamics_parameters,
        equilibrium_mean=mean_tensor.cpu().numpy(),
        transition=transition_tensor.cpu().numpy(),
        innovation_cholesky=fitted_cholesky.cpu().numpy(),
        spot_loading=spot_loading.cpu().numpy(),
        training_spot_score_mean=np.asarray(float(score_mean.cpu())),
    )
    torch.save(
        {
            "schema": "dysanos-v11-pre2025-daily-spot-joint-targets-v1",
            "dates": source_history["dates"],
            "states": model.daily_states.detach().cpu().double(),
            "spot_refs": source_history["spot_refs"],
            "front_spot_scores": score,
            "source_identity": source_identity,
        },
        targets,
    )
    report: dict[str, object] = {
        "schema": "dysanos-v11-pre2025-daily-spot-joint-v1",
        "status": "complete" if convergence.get("converged") is True else "capped",
        "training_semantics": "quotes and one-session spot-conditioned Gaussian AR jointly fitted through 2024-12-31",
        "shared_start": source_audit,
        "source_identity": source_identity,
        "train_range": [dates[0].isoformat(), dates[train_count - 1].isoformat()],
        "daily_dynamics_lookback_sessions": lookback_sessions,
        "daily_dynamics_transition_count": len(origin),
        "validation_range": [dates[train_count].isoformat(), dates[-1].isoformat()],
        "joint_convergence": convergence,
        "joint_history_tail": history[-50:],
        "joint_initialization": initialization_audit,
        "joint_dynamics": {
            "spot_loading_norm": float(torch.linalg.vector_norm(spot_loading).cpu()),
            "spot_score_mean": float(score_mean.cpu()),
            "empirical_surface_innovation_rms": float(
                torch.sqrt(torch.mean(surface_innovation.square())).cpu()
            ),
            "transition_spectral_radius": float(
                np.max(np.abs(np.linalg.eigvals(transition_tensor.cpu().numpy())))
            ),
        },
        "joint_fit": compact_evaluation(
            evaluate_model(
                model,
                tuple(materialize_day(day, env=env) for day in train_days),
                env=env,
                objective_config=FlexObjectiveConfig(fit_weight=0.05),
                day_batch_size=16,
                equal_date_weighting=True,
            )
        ),
        "model_config": asdict(model_config),
        "joint_checkpoint": str(checkpoint.resolve()),
        "target_history": str(targets.resolve()),
        "dynamics_parameters": str(dynamics_parameters.resolve()),
        "native_atm_history": str(native_atm_history.resolve()),
    }
    write_json_atomic(report, report_directory / "daily-spot-joint-direct-200.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly-checkpoint", type=Path, required=True)
    parser.add_argument("--training-state-history", type=Path, required=True)
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--native-atm-history", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--joint-epochs", type=int, default=600)
    parser.add_argument("--lookback-sessions", type=int, default=252)
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(f"daily-spot joint fit {report['status']}: {report['joint_checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
