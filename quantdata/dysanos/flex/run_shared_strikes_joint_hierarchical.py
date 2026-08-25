"""Jointly fit Hierarchical-200 SANOS factors and stationary weekly P dynamics."""

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
    FlexMarketConfig,
    FlexObjectiveConfig,
    JointHierarchicalDynamicsConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.joint_hierarchical import (
    JointHierarchicalTrainConfig,
    QuoteOptimalStateContinuationConfig,
    QuoteOptimalStateTrainConfig,
    continue_quote_optimal_states,
    evaluate_states,
    fit_joint_hierarchical_model,
    fit_quote_optimal_states,
    hierarchical_dynamics_diagnostics,
    initialize_joint_hierarchical_model,
    project_native_raw_to_hierarchical_states,
    standardize_affine_hierarchical_model,
)
from quantdata.dysanos.flex.market_encoder import (
    checkpoint_digest,
    write_json_atomic,
)
from quantdata.dysanos.flex.model import (
    JointHierarchicalDynamicsModel,
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
from quantdata.dysanos.flex.training import (
    TrainingConvergenceError,
    evaluate_model,
    materialize_day,
)
from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs
from quantdata.torch import BaseEnvironment


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--hierarchical-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--state-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--decoder-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--dynamics-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--dynamics-weight", type=float, default=0.10)
    parser.add_argument("--gauge-weight", type=float, default=0.10)
    parser.add_argument("--validation-state-epochs", type=int, default=600)
    parser.add_argument(
        "--validation-state-learning-rate",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument("--validation-continuation-epochs", type=int, default=400)
    parser.add_argument(
        "--validation-continuation-learning-rate",
        type=float,
        default=1.0e-4,
    )
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
        raise ValueError("Source Hierarchical-200 checkpoint is not complete")
    audits = payload.get("training_audits")
    if not isinstance(audits, dict) or not all(
        isinstance(audits.get(name), dict)
        and audits[name].get("converged") is True
        for name in ("supervised", "market")
    ):
        raise ValueError("Source Hierarchical-200 decoder has no convergence proof")
    model = LatentStateSanosModel(
        env,
        num_days=num_days,
        config=_candidate_config("hierarchical", geometry=geometry),
    )
    model.load_state_dict(payload["model_state_dict"])
    return model


def _state_sha256(states: torch.Tensor) -> str:
    values = states.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _compact_evaluation(evaluation) -> dict[str, object]:
    return evaluation.to_dict()


def run(arguments: argparse.Namespace) -> None:
    """Run joint hierarchical P training and quote-optimal 2025 capacity."""
    if arguments.ticker != "SPX":
        raise ValueError("The shared-strikes experiment is specified for SPX")
    if arguments.device != "cuda:0":
        raise ValueError("Joint hierarchical training is restricted to GPU 0")
    if arguments.workers < 1 or arguments.workers > 4:
        raise ValueError("'workers' must lie in [1,4]")
    completion = json.loads(arguments.native_completion.read_text(encoding="utf-8"))
    if completion.get("status") != "native_complete":
        raise ValueError("Native shared-strikes completion proof is not complete")

    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    objective_config = FlexObjectiveConfig(fit_weight=0.05)
    train_config = JointHierarchicalTrainConfig(
        num_epochs=arguments.epochs,
        state_learning_rate=arguments.state_learning_rate,
        decoder_learning_rate=arguments.decoder_learning_rate,
        dynamics_learning_rate=arguments.dynamics_learning_rate,
        dynamics_weight=arguments.dynamics_weight,
        gauge_weight=arguments.gauge_weight,
    )
    validation_train_config = QuoteOptimalStateTrainConfig(
        num_epochs=arguments.validation_state_epochs,
        learning_rate=arguments.validation_state_learning_rate,
    )
    validation_continuation_config = QuoteOptimalStateContinuationConfig(
        num_epochs=arguments.validation_continuation_epochs,
        learning_rate=arguments.validation_continuation_learning_rate,
    )
    model_config = JointHierarchicalDynamicsConfig(
        label="shared-strikes-train-through-2024-joint-hierarchical-200",
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
        arguments.hierarchical_decoder_checkpoint,
        num_days=len(train_days),
        geometry=geometry,
        env=env,
    )
    source_decoder_sha256 = checkpoint_digest(
        arguments.hierarchical_decoder_checkpoint
    )
    initialization = standardize_affine_hierarchical_model(source)
    model = JointHierarchicalDynamicsModel(
        env,
        num_days=len(train_days),
        config=model_config,
    )
    origin, endpoint = exact_weekly_pairs(train_dates, weeks=1)
    initialization_audit = initialize_joint_hierarchical_model(
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
    checkpoint_path = arguments.checkpoint_directory / "joint-hierarchical-200.pt"
    report_path = arguments.report_directory / "joint-hierarchical-200.json"
    completion_path = (
        arguments.report_directory / "joint-hierarchical-200-complete.json"
    )
    with FileLock(str(arguments.checkpoint_directory / "joint-hierarchical-200.lock")):
        history, convergence = fit_joint_hierarchical_model(
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
    minimum_inside = (
        1.0 - train_config.maximum_quote_containment_ratio_drop
    ) * baseline.statistics.all.percent_zero
    if train_evaluation.statistics.all.percent_zero < minimum_inside:
        raise RuntimeError(
            "Joint hierarchical quote containment failed its two-percent ratio gate: "
            f"{train_evaluation.statistics.all.percent_zero:.6g}% < "
            f"{minimum_inside:.6g}%"
        )

    # Raw-coordinate least squares is retained only as an initializer and as a
    # diagnostic.  The reported capacity result below learns every 2025 h_t
    # directly against that date's bid/ask-plus-fit quote objective.
    validation_ls_states = project_native_raw_to_hierarchical_states(
        model,
        validation_raw,
    )
    validation_ls_evaluation = evaluate_states(
        model,
        validation_ls_states,
        validation_days,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        env=env,
    )
    validation_state_checkpoint = (
        arguments.checkpoint_directory / "validation-2025-quote-optimal-states.pt"
    )
    validation_continuation_checkpoint = arguments.checkpoint_directory / (
        "validation-2025-quote-optimal-states-continuation-v1.pt"
    )
    validation_base_convergence: dict[str, object] | None = None
    validation_continuation_provenance: dict[str, object] | None = None
    with FileLock(
        str(arguments.checkpoint_directory / "validation-2025-quote-optimal.lock")
    ):
        base_schedule_exhausted = False
        if validation_state_checkpoint.exists():
            base_payload = torch.load(
                validation_state_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            base_schedule_exhausted = (
                base_payload.get("schema")
                == "flex-quote-optimal-hierarchical-states-v1"
                and base_payload.get("phase") == "training"
                and int(base_payload.get("completed_epochs", -1))
                == validation_train_config.num_epochs
            )
        if base_schedule_exhausted:
            print(
                "found exhausted non-complete base quote-state checkpoint; "
                "validating it before explicit continuation",
                flush=True,
            )
            (
                validation_states,
                validation_history,
                validation_convergence,
                continuation_initialization,
            ) = continue_quote_optimal_states(
                model,
                validation_days,
                objective_config=objective_config,
                base_train_config=validation_train_config,
                continuation_config=validation_continuation_config,
                source_state_checkpoint_path=validation_state_checkpoint,
                continuation_checkpoint_path=validation_continuation_checkpoint,
                source_model_path=checkpoint_path,
                env=env,
            )
            validation_base_convergence = (
                continuation_initialization.source_convergence
            )
            validation_continuation_provenance = (
                continuation_initialization.provenance
            )
            final_validation_state_checkpoint = validation_continuation_checkpoint
        else:
            try:
                validation_states, validation_history, validation_convergence = (
                    fit_quote_optimal_states(
                        model,
                        validation_ls_states,
                        validation_days,
                        objective_config=objective_config,
                        train_config=validation_train_config,
                        checkpoint_path=validation_state_checkpoint,
                        source_model_path=checkpoint_path,
                        env=env,
                    )
                )
                final_validation_state_checkpoint = validation_state_checkpoint
            except TrainingConvergenceError:
                # This is an explicit new phase, not an identical retry or a
                # relabelling of the exhausted base checkpoint. The loader
                # independently validates the exact permitted failure.
                print(
                    "base quote-state schedule failed its plateau gate; starting "
                    "provenance-bound lower-rate continuation",
                    flush=True,
                )
                (
                    validation_states,
                    validation_history,
                    validation_convergence,
                    continuation_initialization,
                ) = continue_quote_optimal_states(
                    model,
                    validation_days,
                    objective_config=objective_config,
                    base_train_config=validation_train_config,
                    continuation_config=validation_continuation_config,
                    source_state_checkpoint_path=validation_state_checkpoint,
                    continuation_checkpoint_path=validation_continuation_checkpoint,
                    source_model_path=checkpoint_path,
                    env=env,
                )
                validation_base_convergence = (
                    continuation_initialization.source_convergence
                )
                validation_continuation_provenance = (
                    continuation_initialization.provenance
                )
                final_validation_state_checkpoint = (
                    validation_continuation_checkpoint
                )
    validation_evaluation = evaluate_states(
        model,
        validation_states,
        validation_days,
        objective_config=objective_config,
        day_batch_size=train_config.day_batch_size,
        env=env,
    )

    train_states = model.daily_states.detach().cpu().double()
    dynamics_training = hierarchical_dynamics_diagnostics(
        model,
        train_states.numpy(),
        train_states.numpy(),
        train_dates,
    )
    dynamics = hierarchical_dynamics_diagnostics(
        model,
        train_states.numpy(),
        validation_states.numpy(),
        validation_dates,
    )
    history_path = (
        arguments.checkpoint_directory / "joint-hierarchical-200-targets.pt"
    )
    temporary = history_path.with_suffix(history_path.suffix + ".tmp")
    torch.save(
        {
            "schema": "flex-joint-hierarchical-target-history-v1",
            "train_dates": [value.isoformat() for value in train_dates],
            "validation_dates": [value.isoformat() for value in validation_dates],
            "train_states": train_states,
            "validation_quote_optimal_states": validation_states,
            "validation_raw_l2_initializer_states": validation_ls_states,
            "spot_refs": torch.as_tensor(
                [cast(float, day.spot_ref) for day in days],
                dtype=torch.float64,
            ),
            "joint_checkpoint": str(checkpoint_path),
            "joint_checkpoint_sha256": checkpoint_digest(checkpoint_path),
            "source_decoder_checkpoint_sha256": source_decoder_sha256,
        },
        temporary,
    )
    temporary.replace(history_path)

    report: dict[str, object] = {
        "schema": "flex-shared-strikes-joint-hierarchical-v1",
        "status": "complete",
        "measure": "P",
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "validation_state_train_config": asdict(validation_train_config),
        "validation_state_continuation_config": asdict(
            validation_continuation_config
        ),
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
        "baseline_two_stage_train_fit": _compact_evaluation(baseline),
        "joint_train_fit": _compact_evaluation(train_evaluation),
        "validation_2025_raw_l2_initializer_fit": _compact_evaluation(
            validation_ls_evaluation
        ),
        "validation_2025_quote_optimal_capacity_fit": _compact_evaluation(
            validation_evaluation
        ),
        "validation_semantics": (
            "The shared decoder and P dynamics are frozen. One independent h_t "
            "per untouched 2025 date is optimized directly against real quotes; "
            "this is the hierarchical decoder's static OOS capacity ceiling, not "
            "a deployable market encoder. Raw-state L2 is initialization only."
        ),
        "dynamics_training": dynamics_training,
        "dynamics": dynamics,
        "convergence": convergence,
        "validation_state_convergence": validation_convergence,
        "validation_state_base_convergence": validation_base_convergence,
        "validation_state_continuation_provenance": (
            validation_continuation_provenance
        ),
        "history_tail": history[-50:],
        "validation_state_history_tail": validation_history[-50:],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest(checkpoint_path),
        "validation_state_checkpoint": str(final_validation_state_checkpoint),
        "validation_state_checkpoint_sha256": checkpoint_digest(
            final_validation_state_checkpoint
        ),
        "validation_state_base_checkpoint": str(validation_state_checkpoint),
        "validation_state_base_checkpoint_sha256": checkpoint_digest(
            validation_state_checkpoint
        ),
        "target_history": str(history_path),
        "target_history_sha256": checkpoint_digest(history_path),
        "train_state_sha256": _state_sha256(train_states),
        "validation_quote_optimal_state_sha256": _state_sha256(validation_states),
        "source_decoder_checkpoint": str(
            arguments.hierarchical_decoder_checkpoint
        ),
        "source_decoder_checkpoint_sha256": source_decoder_sha256,
        "source_native_checkpoint": str(arguments.native_checkpoint),
        "source_native_checkpoint_sha256": checkpoint_digest(
            arguments.native_checkpoint
        ),
        "ad_hoc_choices": [
            "Strong 200-factor expiry allocation",
            "stable diagonal K blocks and unrestricted strict-upper couplings",
            "alternating quote minibatches and exact-week dynamics steps",
            "configured loss weights and learning-rate schedules",
            "full ordered lower-Cholesky Gaussian innovations",
        ],
    }
    write_json_atomic(report, report_path)
    write_json_atomic(
        {
            "schema": "flex-shared-strikes-joint-hierarchical-completion-v1",
            "status": "complete",
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest(checkpoint_path),
            "validation_state_checkpoint": str(final_validation_state_checkpoint),
            "validation_state_checkpoint_sha256": checkpoint_digest(
                final_validation_state_checkpoint
            ),
            "validation_state_base_checkpoint": str(validation_state_checkpoint),
            "validation_state_base_checkpoint_sha256": checkpoint_digest(
                validation_state_checkpoint
            ),
        },
        completion_path,
    )
    print(f"joint hierarchical experiment complete: {report_path}", flush=True)


def main() -> None:
    """CLI entry point."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_shared_strikes_joint_hierarchical")
