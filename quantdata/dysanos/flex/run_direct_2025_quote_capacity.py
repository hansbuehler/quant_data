"""Calibrate quote-optimal 2025 states for the frozen joint Direct-200 decoder."""

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
    NativeStateConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.direct_validation_capacity import (
    DirectValidationCapacityTrainConfig,
    evaluate_quote_capacity_model,
    fit_quote_optimal_validation_states,
    frozen_capacity_decoder_sha256,
    frozen_joint_state_sha256,
    make_frozen_capacity_model,
    tensor_state_sha256,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.joint_direct import (
    compact_evaluation,
    joint_dynamics_diagnostics,
    project_native_raw_to_joint_states,
)
from quantdata.dysanos.flex.market import FlexMarketDay
from quantdata.dysanos.flex.market_encoder import (
    checkpoint_digest,
    write_json_atomic,
)
from quantdata.dysanos.flex.model import (
    JointDirectDynamicsModel,
    NativeStateSanosModel,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _load_markets,
    _load_native_targets,
    _market_dates_from_completion,
    _split_markets,
)
from quantdata.dysanos.flex.training import (
    FitEvaluation,
    evaluate_model,
    materialize_day,
)
from quantdata.torch import BaseEnvironment


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument(
        "--minimum-learning-rate-fraction",
        type=float,
        default=0.10,
    )
    return parser.parse_args()


def _load_joint_model(
    path: Path,
    *,
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    env: BaseEnvironment,
) -> JointDirectDynamicsModel:
    """Load the completed train-through-2024 joint model without mutation."""
    payload = torch.load(path, map_location=env.device, weights_only=False)
    if payload.get("schema") != "flex-joint-direct-dynamics-v1":
        raise ValueError(f"Unsupported joint Direct checkpoint in {path}")
    if payload.get("phase") != "complete":
        raise ValueError(f"Joint Direct checkpoint is not complete: {path}")
    raw_config = cast(dict[str, object], payload["model_config"])
    config = JointDirectDynamicsConfig(
        label=str(raw_config["label"]),
        state_size=int(raw_config["state_size"]),
        group=str(raw_config["group"]),
        user_version=str(raw_config["user_version"]),
        geometry=geometry,
        minimum_mean_reversion_per_week=float(
            raw_config["minimum_mean_reversion_per_week"]
        ),
        innovation_std_floor=float(raw_config["innovation_std_floor"]),
    )
    if payload.get("model_config_version") != config.version.unique_id48:
        raise ValueError("Joint Direct model code version changed")
    if payload.get("model_user_version") != config.user_version:
        raise ValueError("Joint Direct user version changed")
    if payload.get("model_config") != asdict(config):
        raise ValueError("Joint Direct model configuration cannot be reconstructed")
    state_dict = cast(dict[str, torch.Tensor], payload["model_state_dict"])
    training_states = state_dict.get("daily_state_embedding.weight")
    if training_states is None or training_states.ndim != 2:
        raise ValueError("Joint Direct checkpoint has no valid date-state matrix")
    model = JointDirectDynamicsModel(
        env,
        num_days=len(training_states),
        config=config,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _native_evaluation(
    validation_raw: torch.Tensor,
    validation_days: tuple[FlexMarketDay, ...],
    *,
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    objective_config: FlexObjectiveConfig,
    day_batch_size: int,
    env: BaseEnvironment,
) -> FitEvaluation:
    """Evaluate the independently calibrated full-native 2025 ceiling."""
    model = NativeStateSanosModel(
        env,
        num_days=len(validation_days),
        config=NativeStateConfig(
            label="shared-strikes-independent-native-2025-ceiling",
            geometry=geometry,
        ),
    )
    with torch.no_grad():
        model.daily_raw_states.copy_(
            validation_raw.to(dtype=env.fdtype, device=env.device)
        )
    days = tuple(materialize_day(day, env=env) for day in validation_days)
    return evaluate_model(
        model,
        days,
        env=env,
        objective_config=objective_config,
        day_batch_size=day_batch_size,
        equal_date_weighting=True,
    )


def _comparison(
    quote_optimal: dict[str, object],
    other: dict[str, object],
) -> dict[str, object]:
    """Return quote-optimal deltas against one exact comparison fit."""
    quote_stats = cast(dict[str, dict[str, float]], quote_optimal["statistics"])
    other_stats = cast(dict[str, dict[str, float]], other["statistics"])
    buckets: dict[str, dict[str, float]] = {}
    for name in ("all", "0DTE", "1DTE", "2DTE"):
        buckets[name] = {
            "percent_zero_delta": (
                float(quote_stats[name]["percent_zero"])
                - float(other_stats[name]["percent_zero"])
            ),
            "positive_mean_delta": (
                float(quote_stats[name]["positive_mean"])
                - float(other_stats[name]["positive_mean"])
            ),
            "positive_median_delta": (
                float(quote_stats[name]["positive_median"])
                - float(other_stats[name]["positive_median"])
            ),
        }
    return {
        "objective_delta": (
            float(quote_optimal["objective"]) - float(other["objective"])
        ),
        "statistics": buckets,
    }


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(arguments: argparse.Namespace) -> None:
    """Run the quote-optimal frozen-decoder 2025 calibration/capacity test."""
    if arguments.ticker != "SPX":
        raise ValueError("The shared-strikes capacity test is specified for SPX")
    if arguments.device != "cuda:0":
        raise ValueError("Direct validation capacity is restricted to physical GPU 0")
    if arguments.workers < 1 or arguments.workers > 4:
        raise ValueError("'workers' must lie in [1,4]")
    completion = json.loads(
        arguments.native_completion.read_text(encoding="utf-8")
    )
    if completion.get("status") != "native_complete":
        raise ValueError("Native shared-strikes completion proof is incomplete")

    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    objective_config = FlexObjectiveConfig(fit_weight=0.05)
    train_config = DirectValidationCapacityTrainConfig(
        num_epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        warmup_epochs=arguments.warmup_epochs,
        minimum_learning_rate_fraction=(
            arguments.minimum_learning_rate_fraction
        ),
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
    validation_raw = native_raw[-len(validation_days) :]
    train_dates = tuple(day.trading_date for day in train_days)
    validation_dates = tuple(day.trading_date for day in validation_days)
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    source_joint = _load_joint_model(
        arguments.joint_checkpoint,
        geometry=geometry,
        env=env,
    )
    source_joint_sha256 = checkpoint_digest(arguments.joint_checkpoint)
    source_native_sha256 = checkpoint_digest(arguments.native_checkpoint)
    source_frozen_before = frozen_joint_state_sha256(source_joint)
    projection_states = project_native_raw_to_joint_states(
        source_joint,
        validation_raw,
    )
    model = make_frozen_capacity_model(
        source_joint,
        projection_states,
        env=env,
    )
    capacity_frozen_before = frozen_capacity_decoder_sha256(model)
    raw_projection_evaluation = evaluate_quote_capacity_model(
        model,
        validation_days,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        env=env,
    )
    native_evaluation = _native_evaluation(
        validation_raw,
        validation_days,
        geometry=geometry,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        env=env,
    )

    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.checkpoint_directory / "direct-200-2025-capacity.pt"
    report_path = arguments.report_directory / "direct-200-2025-capacity.json"
    completion_path = (
        arguments.report_directory / "direct-200-2025-capacity-complete.json"
    )
    with FileLock(str(arguments.checkpoint_directory / "direct-200-2025-capacity.lock")):
        history, convergence = fit_quote_optimal_validation_states(
            model,
            validation_days,
            source_joint_model=source_joint,
            config=train_config,
            objective_config=objective_config,
            checkpoint_path=checkpoint_path,
            source_joint_sha256=source_joint_sha256,
            source_native_sha256=source_native_sha256,
            env=env,
        )

    quote_optimal_evaluation = evaluate_quote_capacity_model(
        model,
        validation_days,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        env=env,
    )
    if quote_optimal_evaluation.objective > raw_projection_evaluation.objective:
        raise RuntimeError(
            "Quote-optimal validation calibration regressed from its raw-L2 "
            f"initialization: {quote_optimal_evaluation.objective:.8g} > "
            f"{raw_projection_evaluation.objective:.8g}"
        )
    source_frozen_after = frozen_joint_state_sha256(source_joint)
    capacity_frozen_after = frozen_capacity_decoder_sha256(model)
    if source_frozen_after != source_frozen_before:
        raise RuntimeError("Source joint decoder/dynamics changed during calibration")
    if capacity_frozen_after != capacity_frozen_before:
        raise RuntimeError("Frozen capacity decoder changed during calibration")

    learned_states = model.daily_states.detach().cpu().double()
    train_states = source_joint.daily_states.detach().cpu().double()
    dynamics_training = joint_dynamics_diagnostics(
        source_joint,
        train_states.numpy(),
        train_states.numpy(),
        train_dates,
    )
    dynamics_raw_l2_projection = joint_dynamics_diagnostics(
        source_joint,
        train_states.numpy(),
        projection_states.cpu().double().numpy(),
        validation_dates,
    )
    dynamics_quote_optimal = joint_dynamics_diagnostics(
        source_joint,
        train_states.numpy(),
        learned_states.numpy(),
        validation_dates,
    )
    target_path = arguments.checkpoint_directory / "direct-200-2025-capacity-states.pt"
    _atomic_torch_save(
        {
            "schema": "flex-direct-2025-quote-capacity-states-v1",
            "semantics": (
                "One quote-optimal h_t per 2025 date for static frozen-decoder "
                "capacity and later dynamics diagnostics; not an encoder output, "
                "forecast, or modification of the fitted P model."
            ),
            "validation_dates": [value.isoformat() for value in validation_dates],
            "quote_optimal_states": learned_states,
            "raw_l2_projection_initial_states": projection_states.cpu().double(),
            "source_joint_checkpoint": str(arguments.joint_checkpoint),
            "source_joint_checkpoint_sha256": source_joint_sha256,
            "capacity_checkpoint": str(checkpoint_path),
            "capacity_checkpoint_sha256": checkpoint_digest(checkpoint_path),
        },
        target_path,
    )

    raw_projection = compact_evaluation(raw_projection_evaluation)
    quote_optimal = compact_evaluation(quote_optimal_evaluation)
    native_ceiling = compact_evaluation(native_evaluation)
    report: dict[str, object] = {
        "schema": "flex-direct-2025-quote-capacity-report-v1",
        "status": "complete",
        "measure": "static equal-date calibration; no dynamics measure",
        "semantics": (
            "Quote-optimal frozen-decoder 2025 calibration/capacity test. Each "
            "date's h_t starts at the native-raw least-squares projection and is "
            "then learned directly against only that date's real bid/ask and fit "
            "quotes. This is neither OOS prediction nor an oracle raw projection."
        ),
        "validation_range": [
            validation_dates[0].isoformat(),
            validation_dates[-1].isoformat(),
        ],
        "num_validation_dates": len(validation_dates),
        "date_weighting": "mean quotes within each date, then mean dates",
        "model_config": asdict(model.config),
        "train_config": asdict(train_config),
        "objective_config": asdict(objective_config),
        "independent_native_2025_ceiling": native_ceiling,
        "raw_state_l2_projection_initialization": raw_projection,
        "quote_optimal_frozen_decoder_2025_capacity": quote_optimal,
        "quote_optimal_delta_vs_raw_l2_projection": _comparison(
            quote_optimal,
            raw_projection,
        ),
        "quote_optimal_delta_vs_native_ceiling": _comparison(
            quote_optimal,
            native_ceiling,
        ),
        "dynamics_training": dynamics_training,
        "dynamics_2025_raw_l2_projection": dynamics_raw_l2_projection,
        "dynamics_2025_quote_optimal_states": dynamics_quote_optimal,
        "dynamics_comparison_semantics": (
            "Both 2025 diagnostics use the frozen train-through-2024 Direct P "
            "transition. The raw-L2 row is the superseded coordinate projection; "
            "the quote-optimal row uses one independently quote-calibrated h_t "
            "per 2025 date and is the appropriate static-capacity control."
        ),
        "convergence": convergence,
        "history_tail": history[-100:],
        "frozen_identity": {
            "source_decoder_and_dynamics_sha256_before": source_frozen_before,
            "source_decoder_and_dynamics_sha256_after": source_frozen_after,
            "capacity_decoder_sha256_before": capacity_frozen_before,
            "capacity_decoder_sha256_after": capacity_frozen_after,
            "verified_identical": True,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest(checkpoint_path),
        "learned_state_history": str(target_path),
        "learned_state_history_sha256": checkpoint_digest(target_path),
        "learned_state_sha256": tensor_state_sha256({"h": learned_states}),
        "source_joint_checkpoint": str(arguments.joint_checkpoint),
        "source_joint_checkpoint_sha256": source_joint_sha256,
        "source_native_checkpoint": str(arguments.native_checkpoint),
        "source_native_checkpoint_sha256": source_native_sha256,
    }
    write_json_atomic(report, report_path)
    write_json_atomic(
        {
            "schema": "flex-direct-2025-quote-capacity-completion-v1",
            "status": "complete",
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest(checkpoint_path),
            "state_history": str(target_path),
            "state_history_sha256": checkpoint_digest(target_path),
        },
        completion_path,
    )
    print(f"Direct 2025 quote capacity complete: {report_path}", flush=True)


def main() -> None:
    """CLI entry point."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_direct_2025_quote_capacity")
