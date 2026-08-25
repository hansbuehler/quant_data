"""Extend the accepted hierarchical Gaussian Q law one weekly expiry at a time."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from filelock import FileLock

from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.joint_spot_correlation import (
    fit_joint_spot_correlation_under_p,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.run_joint_hierarchical_q1 import (
    _load_inputs,
    _load_json,
    _subset_evenly,
)
from quantdata.dysanos.flex.weekly_measure import (
    JointHierarchicalWeeklyDynamicsAdapter,
    NormalWeeklyMeasureKernel,
    WeeklyMeasureConfig,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    WeeklyMeasureTrainConfig,
    audit_weekly_roll_line,
    extend_weekly_measure_checkpoint,
    fit_weekly_measure_kernel,
    precompute_measure_training_data,
)
from quantdata.dysanos.flex.weekly_simulation import WeeklySanosSimulator
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1-completion", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-roll-lines", type=int, default=13)
    parser.add_argument("--full-audit-paths-per-origin", type=int, default=4096)
    parser.add_argument("--full-audit-origin-batch-size", type=int, default=4)
    parser.add_argument("--train-audit-origins", type=int, default=128)
    parser.add_argument(
        "--material-ratio-relative-increase",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--material-ratio-absolute-increase",
        type=float,
        default=0.10,
    )
    return parser.parse_args()


def _require_positive_audit_controls(arguments: argparse.Namespace) -> None:
    if arguments.device != "cuda:0":
        raise ValueError("Joint hierarchical Q training is restricted to GPU 0")
    if (
        arguments.maximum_roll_lines < 2
        or arguments.full_audit_paths_per_origin < 4
        or arguments.full_audit_paths_per_origin % 2 != 0
        or arguments.full_audit_origin_batch_size <= 0
        or arguments.train_audit_origins <= 0
        or arguments.material_ratio_relative_increase <= 0.0
        or arguments.material_ratio_absolute_increase <= 0.0
    ):
        raise ValueError("invalid sequential Q audit or stopping controls")


def _load_completed_q_stage(
    completion_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path, Mapping[str, Any]]:
    completion = _load_json(completion_path)
    if completion.get("status") != "complete":
        raise ValueError(f"Q completion is not complete: {completion_path}")
    report_path = Path(str(completion["report"]))
    checkpoint_path = Path(str(completion["checkpoint"]))
    if checkpoint_digest(report_path) != completion.get("report_sha256"):
        raise ValueError(f"Q report digest mismatch: {report_path}")
    if checkpoint_digest(checkpoint_path) != completion.get("checkpoint_sha256"):
        raise ValueError(f"Q checkpoint digest mismatch: {checkpoint_path}")
    report = _load_json(report_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        report.get("status") != "complete"
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema") != "flex-weekly-measure-kernel-v1"
        or checkpoint.get("phase") != "complete"
    ):
        raise ValueError("Q report or checkpoint lacks semantic completion")
    return completion, report, checkpoint_path, checkpoint


def _measure_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> WeeklyMeasureConfig:
    raw = checkpoint.get("kernel_config")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("context_network"), Mapping):
        raise ValueError("Q checkpoint has no reconstructable measure config")
    context = FeedForwardConfig(**dict(cast(Mapping[str, Any], raw["context_network"])))
    config = WeeklyMeasureConfig(
        label=str(raw["label"]),
        state_sizes_by_expiry=tuple(int(value) for value in raw["state_sizes_by_expiry"]),
        hierarchical=bool(raw["hierarchical"]),
        group=str(raw["group"]),
        user_version=str(raw["user_version"]),
        context_network=context,
        maximum_log_scale=float(raw["maximum_log_scale"]),
        maximum_shift=float(raw["maximum_shift"]),
        generic_amplitude_fraction=float(raw["generic_amplitude_fraction"]),
    )
    if asdict(config) != dict(raw):
        raise ValueError("Q measure config differs from current code")
    if checkpoint.get("kernel_config_version") != config.version.unique_id48:
        raise ValueError("Q measure config code version is stale")
    return config


def _train_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> WeeklyMeasureTrainConfig:
    raw = checkpoint.get("train_config")
    if not isinstance(raw, Mapping):
        raise ValueError("Q checkpoint has no reconstructable train config")
    config = WeeklyMeasureTrainConfig(**dict(raw))
    if asdict(config) != dict(raw):
        raise ValueError("Q train config differs from current code")
    if checkpoint.get("train_config_version") != config.version.unique_id48:
        raise ValueError("Q train config code version is stale")
    return config


def _q_to_p_ratio(
    p_audit: Mapping[str, object],
    q_audit: Mapping[str, object],
) -> float:
    p_rmse = float(p_audit["conditional_moment_rmse_bps_spot"])
    q_rmse = float(q_audit["conditional_moment_rmse_bps_spot"])
    if p_rmse <= 0.0 or q_rmse < 0.0:
        raise ValueError("invalid P/Q audit RMSE")
    return q_rmse / p_rmse


def _q1_validation_ratio(report: Mapping[str, Any]) -> float:
    roll = report.get("two_week_roll")
    if not isinstance(roll, Mapping):
        raise ValueError("Q1 report has no two-week roll audit")
    return _q_to_p_ratio(
        cast(Mapping[str, object], roll["p_baseline_validation_2025"]),
        cast(Mapping[str, object], roll["q1_validation_2025"]),
    )


def _is_material_degradation(
    previous_ratio: float,
    current_ratio: float,
    *,
    relative_increase: float,
    absolute_increase: float,
) -> bool:
    """Require both a relative and a percentage-point deterioration."""
    return (
        current_ratio - previous_ratio >= absolute_increase
        and current_ratio >= previous_ratio * (1.0 + relative_increase)
    )


def _write_sequence_report(
    path: Path,
    *,
    q1_completion: Path,
    maximum_roll_lines: int,
    stages: list[dict[str, object]],
    status: str,
    termination_reason: str | None,
    stopping_rule: Mapping[str, object],
) -> None:
    write_json_atomic(
        {
            "schema": "flex-joint-hierarchical-normal-q-sequence-v1",
            "status": status,
            "source_q1_completion": str(q1_completion),
            "source_q1_completion_sha256": checkpoint_digest(q1_completion),
            "maximum_roll_lines": maximum_roll_lines,
            "completed_roll_lines": 1 + len(stages),
            "stages": stages,
            "stopping_rule": dict(stopping_rule),
            "termination_reason": termination_reason,
        },
        path,
    )


def run(arguments: argparse.Namespace) -> None:
    """Continue Q1 through later weekly roll restrictions until its fit degrades."""
    _require_positive_audit_controls(arguments)
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    _, q1_report, source_checkpoint_path, source_checkpoint = (
        _load_completed_q_stage(arguments.q1_completion)
    )
    q1_sources = q1_report.get("sources")
    if not isinstance(q1_sources, Mapping):
        raise ValueError("Q1 report has no P-model provenance")
    joint_completion = Path(str(q1_sources["completion"]))
    (
        model,
        train_dates,
        train_states,
        train_log_spot,
        validation_dates,
        validation_states,
        _,
        sources,
    ) = _load_inputs(joint_completion, env=env)
    measure_config = _measure_config_from_checkpoint(source_checkpoint)
    source_train_config = _train_config_from_checkpoint(source_checkpoint)
    if (
        source_train_config.maximum_roll_lines != 1
        or int(source_checkpoint.get("line_index", 0)) != 1
        or not measure_config.hierarchical
        or measure_config.state_sizes_by_expiry != model.config.state_sizes_by_expiry
    ):
        raise ValueError("Q1 source is not the accepted first hierarchical line")
    maximum_roll_lines = min(
        arguments.maximum_roll_lines,
        len(measure_config.state_sizes_by_expiry) - 1,
    )
    spot_fit = fit_joint_spot_correlation_under_p(
        model,
        train_states.detach().cpu().double().numpy(),
        train_log_spot,
        train_dates,
    )
    dynamics = JointHierarchicalWeeklyDynamicsAdapter(
        env,
        model=model,
        spot_front_beta=torch.as_tensor(
            spot_fit.front_beta,
            dtype=env.fdtype,
            device=env.device,
        ),
        spot_orthogonal_std=spot_fit.orthogonal_std,
    )
    simulator = WeeklySanosSimulator(
        env,
        dynamics=dynamics,
        calibrated_model=model,
    )
    normalized_strikes = torch.exp(
        torch.tensor(
            source_train_config.log_moneyness,
            dtype=env.fdtype,
            device=env.device,
        )
    )
    training_data = precompute_measure_training_data(
        simulator,
        train_states,
        normalized_strikes,
        num_expiries=maximum_roll_lines + 1,
    )
    validation_data = precompute_measure_training_data(
        simulator,
        validation_states,
        normalized_strikes,
        num_expiries=maximum_roll_lines + 1,
    )
    training_audit_data = _subset_evenly(
        training_data,
        arguments.train_audit_origins,
    )
    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    sequence_report_path = arguments.report_directory / "normal-q-sequence.json"
    sequence_completion_path = (
        arguments.report_directory / "normal-q-sequence-complete.json"
    )
    stopping_rule = {
        "statistic": "2025 conditional-moment RMSE ratio Q/P for the active roll line",
        "relative_ratio_increase": arguments.material_ratio_relative_increase,
        "absolute_ratio_increase": arguments.material_ratio_absolute_increase,
        "decision": (
            "stop only when both thresholds are breached relative to the preceding line"
        ),
        "ad_hoc": True,
    }
    stages: list[dict[str, object]] = []
    _write_sequence_report(
        sequence_report_path,
        q1_completion=arguments.q1_completion,
        maximum_roll_lines=maximum_roll_lines,
        stages=stages,
        status="running",
        termination_reason=None,
        stopping_rule=stopping_rule,
    )
    previous_ratio = _q1_validation_ratio(q1_report)
    previous_completion = arguments.q1_completion
    previous_checkpoint = source_checkpoint_path

    with FileLock(str(arguments.checkpoint_directory / "joint-q-sequence.lock")):
        for roll_lines in range(2, maximum_roll_lines + 1):
            target_train_config = replace(
                source_train_config,
                maximum_roll_lines=roll_lines,
            )
            target_checkpoint = (
                arguments.checkpoint_directory
                / f"joint-hierarchical-normal-q{roll_lines}.pt"
            )
            kernel = NormalWeeklyMeasureKernel(
                env,
                dynamics=dynamics,
                config=measure_config,
            )
            seed_result = extend_weekly_measure_checkpoint(
                previous_checkpoint,
                target_checkpoint,
                kernel=kernel,
                training_data=training_data,
                source_config=source_train_config,
                target_config=target_train_config,
            )
            training = fit_weekly_measure_kernel(
                kernel,
                simulator,
                training_data,
                validation_data,
                checkpoint_path=target_checkpoint,
                config=target_train_config,
            )
            p_kernel = NormalWeeklyMeasureKernel(
                env,
                dynamics=dynamics,
                config=measure_config,
            )
            p_train = audit_weekly_roll_line(
                p_kernel,
                simulator,
                training_audit_data,
                normalized_strikes,
                current_expiry_index=roll_lines,
                paths_per_origin=arguments.full_audit_paths_per_origin,
                origins_per_batch=arguments.full_audit_origin_batch_size,
                seed=target_train_config.seed + 30_000 + roll_lines,
            )
            p_validation = audit_weekly_roll_line(
                p_kernel,
                simulator,
                validation_data,
                normalized_strikes,
                current_expiry_index=roll_lines,
                paths_per_origin=arguments.full_audit_paths_per_origin,
                origins_per_batch=arguments.full_audit_origin_batch_size,
                seed=target_train_config.seed + 40_000 + roll_lines,
            )
            q_train = audit_weekly_roll_line(
                kernel,
                simulator,
                training_audit_data,
                normalized_strikes,
                current_expiry_index=roll_lines,
                paths_per_origin=arguments.full_audit_paths_per_origin,
                origins_per_batch=arguments.full_audit_origin_batch_size,
                seed=target_train_config.seed + 30_000 + roll_lines,
            )
            q_validation = audit_weekly_roll_line(
                kernel,
                simulator,
                validation_data,
                normalized_strikes,
                current_expiry_index=roll_lines,
                paths_per_origin=arguments.full_audit_paths_per_origin,
                origins_per_batch=arguments.full_audit_origin_batch_size,
                seed=target_train_config.seed + 40_000 + roll_lines,
            )
            validation_ratio = _q_to_p_ratio(p_validation, q_validation)
            train_ratio = _q_to_p_ratio(p_train, q_train)
            material_degradation = _is_material_degradation(
                previous_ratio,
                validation_ratio,
                relative_increase=arguments.material_ratio_relative_increase,
                absolute_increase=arguments.material_ratio_absolute_increase,
            )
            report_path = (
                arguments.report_directory
                / f"joint-hierarchical-normal-q{roll_lines}.json"
            )
            completion_path = (
                arguments.report_directory
                / f"joint-hierarchical-normal-q{roll_lines}-complete.json"
            )
            report: dict[str, object] = {
                "schema": "flex-joint-hierarchical-normal-q-stage-v1",
                "status": "complete",
                "roll_line": roll_lines,
                "target": (
                    f"today {roll_lines + 1}W fixed-cash-strike calls equal expected "
                    f"next-week {roll_lines}W calls"
                ),
                "current_expiry_index": roll_lines,
                "rolled_expiry_index": roll_lines - 1,
                "train_range": [
                    train_dates[0].isoformat(),
                    train_dates[-1].isoformat(),
                ],
                "validation_range": [
                    validation_dates[0].isoformat(),
                    validation_dates[-1].isoformat(),
                ],
                "date_weighting": "equal origin dates; no quote-count weighting",
                "path_generation": (
                    "fresh origin-wise antithetic Torch normals inside every batch"
                ),
                "measure_config": asdict(measure_config),
                "train_config": asdict(target_train_config),
                "sources": sources,
                "previous_q_completion": str(previous_completion),
                "previous_q_completion_sha256": checkpoint_digest(previous_completion),
                "checkpoint_seed": seed_result,
                "training": training,
                "roll_audit": {
                    "p_baseline_training": p_train,
                    "q_training": q_train,
                    "p_baseline_validation_2025": p_validation,
                    "q_validation_2025": q_validation,
                    "training_q_to_p_rmse_ratio": train_ratio,
                    "validation_q_to_p_rmse_ratio": validation_ratio,
                    "preceding_line_validation_q_to_p_rmse_ratio": previous_ratio,
                },
                "material_degradation": material_degradation,
                "stopping_rule": stopping_rule,
                "checkpoint": str(target_checkpoint),
                "checkpoint_sha256": checkpoint_digest(target_checkpoint),
            }
            write_json_atomic(report, report_path)
            write_json_atomic(
                {
                    "schema": "flex-joint-hierarchical-normal-q-stage-completion-v1",
                    "status": "complete",
                    "roll_line": roll_lines,
                    "report": str(report_path),
                    "report_sha256": checkpoint_digest(report_path),
                    "checkpoint": str(target_checkpoint),
                    "checkpoint_sha256": checkpoint_digest(target_checkpoint),
                },
                completion_path,
            )
            stages.append(
                {
                    "roll_line": roll_lines,
                    "status": "complete",
                    "report": str(report_path),
                    "report_sha256": checkpoint_digest(report_path),
                    "completion": str(completion_path),
                    "completion_sha256": checkpoint_digest(completion_path),
                    "training_q_to_p_rmse_ratio": train_ratio,
                    "validation_q_to_p_rmse_ratio": validation_ratio,
                    "material_degradation": material_degradation,
                }
            )
            if material_degradation:
                termination_reason = (
                    f"Q{roll_lines} validation Q/P ratio {validation_ratio:.6g} "
                    f"materially exceeded preceding {previous_ratio:.6g}"
                )
                _write_sequence_report(
                    sequence_report_path,
                    q1_completion=arguments.q1_completion,
                    maximum_roll_lines=maximum_roll_lines,
                    stages=stages,
                    status="complete",
                    termination_reason=termination_reason,
                    stopping_rule=stopping_rule,
                )
                break
            previous_ratio = validation_ratio
            previous_completion = completion_path
            previous_checkpoint = target_checkpoint
            source_train_config = target_train_config
            _write_sequence_report(
                sequence_report_path,
                q1_completion=arguments.q1_completion,
                maximum_roll_lines=maximum_roll_lines,
                stages=stages,
                status="running",
                termination_reason=None,
                stopping_rule=stopping_rule,
            )
        else:
            _write_sequence_report(
                sequence_report_path,
                q1_completion=arguments.q1_completion,
                maximum_roll_lines=maximum_roll_lines,
                stages=stages,
                status="complete",
                termination_reason="maximum requested roll line completed",
                stopping_rule=stopping_rule,
            )

    write_json_atomic(
        {
            "schema": "flex-joint-hierarchical-normal-q-sequence-completion-v1",
            "status": "complete",
            "report": str(sequence_report_path),
            "report_sha256": checkpoint_digest(sequence_report_path),
            "completed_roll_lines": 1 + len(stages),
        },
        sequence_completion_path,
    )


def main() -> None:
    """CLI entry point with durable exception reporting."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_joint_hierarchical_q_sequence")
