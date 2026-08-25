"""Jointly fit Direct-200 SANOS factors and stable weekly P dynamics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from filelock import FileLock

from quantdata.dysanos.flex.config import (
    FlexMarketConfig,
    FlexObjectiveConfig,
    JointDirectDynamicsConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.joint_direct import (
    JointDirectTrainConfig,
    compact_evaluation,
    evaluate_projected_market_days,
    fit_joint_direct_model,
    initialize_joint_model,
    joint_dynamics_diagnostics,
    project_native_raw_to_joint_states,
    standardize_affine_direct_model,
)
from quantdata.dysanos.flex.market_encoder import (
    checkpoint_digest,
    write_json_atomic,
)
from quantdata.dysanos.flex.model import (
    JointDirectDynamicsModel,
    LatentStateSanosModel,
)
from quantdata.dysanos.flex.run_shared_strikes_compressed_encoders import (
    _candidate_config,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _load_markets,
    _load_native_targets,
    _market_dates_from_completion,
    _split_markets,
)
from quantdata.dysanos.flex.training import evaluate_model, materialize_day
from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs
from quantdata.torch import BaseEnvironment


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--state-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--decoder-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--dynamics-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--dynamics-weight", type=float, default=0.10)
    parser.add_argument("--gauge-weight", type=float, default=0.10)
    return parser.parse_args()


def _load_source_decoder(
    checkpoint_path: Path,
    *,
    num_days: int,
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    env: BaseEnvironment,
) -> LatentStateSanosModel:
    payload = torch.load(checkpoint_path, map_location=env.device, weights_only=False)
    if payload.get("schema_version") != 2 or payload.get("phase") != "complete":
        raise ValueError("Source Direct-200 decoder checkpoint is not complete")
    audits = payload.get("training_audits")
    if not isinstance(audits, dict) or not all(
        isinstance(audits.get(name), dict)
        and audits[name].get("converged") is True
        for name in ("supervised", "market")
    ):
        raise ValueError("Source Direct-200 decoder has no convergence proof")
    model = LatentStateSanosModel(
        env,
        num_days=num_days,
        config=_candidate_config("direct", geometry=geometry),
    )
    model.load_state_dict(payload["model_state_dict"])
    return model


def _state_sha256(states: torch.Tensor) -> str:
    import hashlib

    values = states.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def run(arguments: argparse.Namespace) -> None:
    """Run the train-only joint Direct experiment and oracle OOS diagnostics."""
    if arguments.ticker != "SPX":
        raise ValueError("The shared-strikes Direct experiment is specified for SPX")
    if arguments.device != "cuda:0":
        raise ValueError("Joint Direct training is restricted to physical GPU 0")
    if arguments.workers < 1 or arguments.workers > 4:
        raise ValueError("'workers' must lie in [1,4]")
    if not arguments.native_completion.exists():
        raise FileNotFoundError(arguments.native_completion)
    completion = json.loads(arguments.native_completion.read_text(encoding="utf-8"))
    if completion.get("status") != "native_complete":
        raise ValueError("Native shared-strikes completion proof is not complete")

    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    objective_config = FlexObjectiveConfig(fit_weight=0.05)
    train_config = JointDirectTrainConfig(
        num_epochs=arguments.epochs,
        state_learning_rate=arguments.state_learning_rate,
        decoder_learning_rate=arguments.decoder_learning_rate,
        dynamics_learning_rate=arguments.dynamics_learning_rate,
        dynamics_weight=arguments.dynamics_weight,
        gauge_weight=arguments.gauge_weight,
    )
    model_config = JointDirectDynamicsConfig(
        label="shared-strikes-train-through-2024-joint-direct-200",
        state_size=200,
        geometry=geometry,
    )
    dates = _market_dates_from_completion(arguments.native_completion)
    days = _load_markets(
        ticker=arguments.ticker,
        dates=dates,
        geometry=geometry,
        market_config=FlexMarketConfig(),
        workers=arguments.workers,
    )
    train_days, validation_days = _split_markets(days)
    native_raw, _ = _load_native_targets(
        arguments.native_checkpoint,
        days,
        geometry=geometry,
    )
    split = len(train_days)
    validation_raw = native_raw[split:]
    train_dates = tuple(day.trading_date for day in train_days)
    validation_dates = tuple(day.trading_date for day in validation_days)

    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    source = _load_source_decoder(
        arguments.direct_decoder_checkpoint,
        num_days=len(train_days),
        geometry=geometry,
        env=env,
    )
    source_decoder_sha256 = checkpoint_digest(arguments.direct_decoder_checkpoint)
    initialization = standardize_affine_direct_model(source)
    model = JointDirectDynamicsModel(
        env,
        num_days=len(train_days),
        config=model_config,
    )
    origin, endpoint = exact_weekly_pairs(train_dates, weeks=1)
    initialization_audit = initialize_joint_model(
        model,
        initialization,
        origin_indices=origin,
        endpoint_indices=endpoint,
    )
    train_torch_days = tuple(materialize_day(day, env=env) for day in train_days)
    baseline = evaluate_model(
        model,
        train_torch_days,
        env=env,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        equal_date_weighting=True,
    )

    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.checkpoint_directory / "joint-direct-200.pt"
    report_path = arguments.report_directory / "joint-direct-200.json"
    completion_path = arguments.report_directory / "joint-direct-200-complete.json"
    with FileLock(str(arguments.checkpoint_directory / "joint-direct-200.lock")):
        history, convergence = fit_joint_direct_model(
            model,
            train_days,
            dates=train_dates,
            objective_config=objective_config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            source_decoder_sha256=source_decoder_sha256,
            env=env,
        )

    train_evaluation = evaluate_model(
        model,
        train_torch_days,
        env=env,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        equal_date_weighting=True,
    )
    baseline_inside = baseline.statistics.all.percent_zero
    final_inside = train_evaluation.statistics.all.percent_zero
    minimum_inside = (
        1.0 - train_config.maximum_quote_containment_ratio_drop
    ) * baseline_inside
    if final_inside < minimum_inside:
        raise RuntimeError(
            "Joint Direct quote containment failed its two-percent ratio gate: "
            f"{final_inside:.6g}% < {minimum_inside:.6g}%"
        )

    validation_states = project_native_raw_to_joint_states(model, validation_raw)
    validation_evaluation = evaluate_projected_market_days(
        model,
        validation_states,
        validation_days,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        env=env,
    )
    train_states = model.daily_states.detach().cpu().double()
    dynamics_training = joint_dynamics_diagnostics(
        model,
        train_states.numpy(),
        train_states.numpy(),
        train_dates,
    )
    dynamics = joint_dynamics_diagnostics(
        model,
        train_states.numpy(),
        validation_states.numpy(),
        validation_dates,
    )
    history_path = arguments.checkpoint_directory / "joint-direct-200-targets.pt"
    temporary = history_path.with_suffix(history_path.suffix + ".tmp")
    torch.save(
        {
            "schema": "flex-joint-direct-target-history-v1",
            "train_dates": [value.isoformat() for value in train_dates],
            "validation_dates": [value.isoformat() for value in validation_dates],
            "train_states": train_states,
            "validation_projection_oracle_states": validation_states,
            "spot_refs": torch.as_tensor(
                [cast(float, day.spot_ref) for day in days], dtype=torch.float64
            ),
            "joint_checkpoint": str(checkpoint_path),
            "joint_checkpoint_sha256": checkpoint_digest(checkpoint_path),
            "source_decoder_checkpoint_sha256": source_decoder_sha256,
        },
        temporary,
    )
    temporary.replace(history_path)

    report: dict[str, object] = {
        "schema": "flex-shared-strikes-joint-direct-v1",
        "status": "complete",
        "measure": "P",
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "objective_config": asdict(objective_config),
        "train_range": [train_dates[0].isoformat(), train_dates[-1].isoformat()],
        "validation_range": [
            validation_dates[0].isoformat(),
            validation_dates[-1].isoformat(),
        ],
        "num_train_dates": len(train_dates),
        "num_validation_dates": len(validation_dates),
        "date_weighting": "mean quotes within each date, then mean dates",
        "initialization": {
            **initialization_audit,
            "maximum_raw_reconstruction_error": (
                initialization.maximum_raw_reconstruction_error
            ),
            "minimum_original_factor_scale": initialization.minimum_factor_scale,
            "maximum_original_factor_scale": initialization.maximum_factor_scale,
        },
        "baseline_two_stage_train_fit": compact_evaluation(baseline),
        "joint_train_fit": compact_evaluation(train_evaluation),
        "validation_2025_projection_oracle_fit": compact_evaluation(
            validation_evaluation
        ),
        "validation_semantics": (
            "The 2025 states are least-squares projections of independently "
            "calibrated native states. They test the learned dynamic manifold "
            "but are not a deployable market encoder and never enter training."
        ),
        "dynamics_training": dynamics_training,
        "dynamics": dynamics,
        "convergence": convergence,
        "history_tail": history[-50:],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest(checkpoint_path),
        "target_history": str(history_path),
        "target_history_sha256": checkpoint_digest(history_path),
        "train_state_sha256": _state_sha256(train_states),
        "validation_oracle_state_sha256": _state_sha256(validation_states),
        "source_decoder_checkpoint": str(arguments.direct_decoder_checkpoint),
        "source_decoder_checkpoint_sha256": source_decoder_sha256,
        "source_native_checkpoint": str(arguments.native_checkpoint),
        "source_native_checkpoint_sha256": checkpoint_digest(
            arguments.native_checkpoint
        ),
        "ad_hoc_choices": [
            "200 Direct factors",
            "dissipative K=LL'+S+k_min I parameterization",
            "alternating quote minibatches and full exact-week dynamics steps",
            "configured dynamics/gauge weights and learning-rate schedule",
            "full Gaussian weekly innovation covariance",
        ],
    }
    write_json_atomic(report, report_path)
    write_json_atomic(
        {
            "schema": "flex-shared-strikes-joint-direct-completion-v1",
            "status": "complete",
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest(checkpoint_path),
        },
        completion_path,
    )
    print(f"joint Direct experiment complete: {report_path}", flush=True)


def main() -> None:
    """CLI entry point."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_shared_strikes_joint_direct")
