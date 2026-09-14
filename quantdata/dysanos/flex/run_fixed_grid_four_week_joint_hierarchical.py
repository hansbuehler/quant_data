"""Jointly fit four-week fixed-grid hierarchical factors and weekly P dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from filelock import FileLock

from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    FlexObjectiveConfig,
    JointHierarchicalDynamicsConfig,
    TriangularPcaDecoderConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.fixed_relative_grid import (
    representative_relative_strike_grid,
)
from quantdata.dysanos.flex.joint_hierarchical import (
    JointHierarchicalTrainConfig,
    evaluate_states,
    fit_joint_hierarchical_model,
    hierarchical_dynamics_diagnostics,
    initialize_joint_hierarchical_model,
    standardize_affine_hierarchical_model,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import (
    JointHierarchicalDynamicsModel,
    LatentStateSanosModel,
)
from quantdata.dysanos.flex.run_all_history_joint_direct import (
    _market_config_from_completion,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _load_markets,
    _market_dates_from_completion,
)
from quantdata.dysanos.flex.training import evaluate_model, materialize_day
from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs
from quantdata.torch import BaseEnvironment

_ALLOCATION = (100, 50, 25, 25)
_MODEL_NAME = "joint-hierarchical-front-loaded-200"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-joint-checkpoint", type=Path, required=True)
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--hierarchical-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--day-batch-size", type=int, default=16)
    parser.add_argument("--state-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--decoder-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--dynamics-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--dynamics-weight", type=float, default=0.10)
    parser.add_argument("--gauge-weight", type=float, default=0.10)
    return parser.parse_args()


def _state_sha256(states: torch.Tensor) -> str:
    values = states.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _load_source_decoder(
    path: Path,
    *,
    num_days: int,
    geometry: FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    env: BaseEnvironment,
) -> LatentStateSanosModel:
    payload = torch.load(path, map_location=env.device, weights_only=False)
    if payload.get("schema_version") != 2 or payload.get("phase") != "complete":
        raise ValueError("Four-week hierarchical decoder is not complete")
    audits = payload.get("training_audits")
    if not isinstance(audits, dict) or not all(
        isinstance(audits.get(name), dict) and audits[name].get("converged") is True
        for name in ("supervised", "market")
    ):
        raise ValueError("Four-week hierarchical decoder has no convergence proof")
    identity = payload.get("model_identity")
    model_values = identity.get("model_config") if isinstance(identity, dict) else None
    if not isinstance(model_values, dict):
        raise ValueError("Four-week hierarchical decoder identity is missing")
    if tuple(model_values.get("state_sizes_by_expiry", ())) != _ALLOCATION:
        raise ValueError("Four-week hierarchical decoder allocation changed")
    model = LatentStateSanosModel(
        env,
        num_days=num_days,
        config=TriangularPcaDecoderConfig(
            label=cast(str, model_values["label"]),
            state_size=200,
            state_sizes_by_expiry=_ALLOCATION,
            geometry=geometry,
        ),
    )
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model_state_dict"]))
    return model


def run(arguments: argparse.Namespace) -> None:
    """Fit all 891 dates jointly with no temporal holdout."""
    if arguments.ticker != "SPX" or arguments.device != "cuda:0":
        raise ValueError("This experiment requires SPX on physical GPU 0")
    if arguments.workers < 1 or arguments.workers > 4:
        raise ValueError("'workers' must lie in [1,4]")
    for path in (
        arguments.reference_joint_checkpoint,
        arguments.native_completion,
        arguments.hierarchical_decoder_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    grid = representative_relative_strike_grid(arguments.reference_joint_checkpoint)
    geometry = FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_model_expiries=4,
        explicit_model_business_dtes=(5.0, 10.0, 15.0, 20.0),
        inner_relative_strikes=grid.inner_relative_strikes,
    )
    completion = json.loads(arguments.native_completion.read_text(encoding="utf-8"))
    if completion.get("status") != "native_complete":
        raise ValueError("Fixed-grid native completion proof is incomplete")
    dates = _market_dates_from_completion(arguments.native_completion)
    market_config = _market_config_from_completion(completion)
    days = _load_markets(
        ticker=arguments.ticker,
        dates=dates,
        geometry=geometry,
        market_config=market_config,
        workers=arguments.workers,
    )
    if tuple(day.trading_date for day in days) != dates:
        raise RuntimeError("Loaded fixed-grid markets changed order")

    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    source = _load_source_decoder(
        arguments.hierarchical_decoder_checkpoint,
        num_days=len(days),
        geometry=geometry,
        env=env,
    )
    source_sha256 = checkpoint_digest(arguments.hierarchical_decoder_checkpoint)
    initialization = standardize_affine_hierarchical_model(source)
    model_config = JointHierarchicalDynamicsConfig(
        label="fixed-grid-four-week-all-history-joint-hierarchical-front-loaded-200",
        state_sizes_by_expiry=_ALLOCATION,
        state_size=200,
        geometry=geometry,  # type: ignore[arg-type]
    )
    model = JointHierarchicalDynamicsModel(env, num_days=len(days), config=model_config)
    origin, endpoint = exact_weekly_pairs(dates, weeks=1)
    initialization_audit = initialize_joint_hierarchical_model(
        model,
        initialization,
        origin_indices=origin,
        endpoint_indices=endpoint,
    )
    objective_config = FlexObjectiveConfig(fit_weight=0.05)
    train_config = JointHierarchicalTrainConfig(
        num_epochs=arguments.epochs,
        day_batch_size=arguments.day_batch_size,
        state_learning_rate=arguments.state_learning_rate,
        decoder_learning_rate=arguments.decoder_learning_rate,
        dynamics_learning_rate=arguments.dynamics_learning_rate,
        dynamics_weight=arguments.dynamics_weight,
        gauge_weight=arguments.gauge_weight,
        seed=9329,
    )
    torch_days = tuple(materialize_day(day, env=env) for day in days)
    baseline = evaluate_model(
        model,
        torch_days,
        env=env,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        equal_date_weighting=True,
    )
    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.checkpoint_directory / f"{_MODEL_NAME}.pt"
    with FileLock(str(arguments.checkpoint_directory / f"{_MODEL_NAME}.lock")):
        history, convergence = fit_joint_hierarchical_model(
            model,
            days,
            dates=dates,
            objective_config=objective_config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            source_decoder_sha256=source_sha256,
            env=env,
        )
    evaluation = evaluate_model(
        model,
        torch_days,
        env=env,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        equal_date_weighting=True,
    )
    states = model.daily_states.detach().cpu().double()
    dynamics = hierarchical_dynamics_diagnostics(
        model,
        states.numpy(),
        states.numpy(),
        dates,
    )
    target_path = arguments.checkpoint_directory / f"{_MODEL_NAME}-targets.pt"
    temporary_target = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(
        {
            "schema": "flex-fixed-grid-four-week-target-history-v1",
            "dates": [value.isoformat() for value in dates],
            "states": states,
            "spot_refs": torch.as_tensor(
                [cast(float, day.spot_ref) for day in days], dtype=torch.float64
            ),
            "joint_checkpoint": str(checkpoint_path),
            "joint_checkpoint_sha256": checkpoint_digest(checkpoint_path),
        },
        temporary_target,
    )
    temporary_target.replace(target_path)
    report_path = arguments.report_directory / f"{_MODEL_NAME}.json"
    report = {
        "schema": "flex-fixed-grid-four-week-all-history-joint-hierarchical-v1",
        "status": "complete",
        "measure": "P",
        "num_dates": len(dates),
        "train_range": [dates[0].isoformat(), dates[-1].isoformat()],
        "model_config": asdict(model_config),
        "market_config": asdict(market_config),
        "train_config": asdict(train_config),
        "objective_config": asdict(objective_config),
        "representative_grid": asdict(grid),
        "initialization": {
            **initialization_audit,
            "maximum_raw_reconstruction_error": initialization.maximum_raw_reconstruction_error,
        },
        "baseline_fit": baseline.to_dict(),
        "all_history_fit": evaluation.to_dict(),
        "dynamics": dynamics,
        "convergence": convergence,
        "history_tail": history[-50:],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest(checkpoint_path),
        "target_history": str(target_path),
        "target_history_sha256": checkpoint_digest(target_path),
        "state_sha256": _state_sha256(states),
        "source_decoder_checkpoint": str(arguments.hierarchical_decoder_checkpoint),
        "source_decoder_checkpoint_sha256": source_sha256,
    }
    write_json_atomic(report, report_path)
    write_json_atomic(
        {
            "schema": "flex-fixed-grid-four-week-joint-hierarchical-completion-v1",
            "status": "complete",
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest(checkpoint_path),
            "target_history": str(target_path),
            "target_history_sha256": checkpoint_digest(target_path),
        },
        arguments.report_directory / f"{_MODEL_NAME}-complete.json",
    )


def main() -> None:
    """CLI entry point."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_fixed_grid_four_week_joint_hierarchical")
