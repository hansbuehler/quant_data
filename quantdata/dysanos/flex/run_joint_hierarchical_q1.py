"""Fit the first hierarchical target-Q line to the 2W-to-1W roll restriction."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from filelock import FileLock

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.config import JointHierarchicalDynamicsConfig
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.joint_spot_correlation import (
    evaluate_joint_spot_correlation_under_p,
    fit_joint_spot_correlation_under_p,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import JointHierarchicalDynamicsModel
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    JointHierarchicalWeeklyDynamicsAdapter,
    NormalWeeklyMeasureKernel,
    WeeklyMeasureConfig,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    WeeklyMeasureTrainConfig,
    WeeklyMeasureTrainingData,
    audit_front_week_consistency,
    audit_weekly_roll_line,
    fit_weekly_measure_kernel,
    precompute_measure_training_data,
)
from quantdata.dysanos.flex.weekly_simulation import WeeklySanosSimulator
from quantdata.torch import BaseEnvironment


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--kernel-kind",
        choices=("normal", "generic"),
        default="normal",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--origins-per-batch", type=int, default=16)
    parser.add_argument("--paths-per-origin", type=int, default=256)
    parser.add_argument("--audit-origins", type=int, default=64)
    parser.add_argument("--audit-paths-per-origin", type=int, default=2048)
    parser.add_argument("--full-audit-paths-per-origin", type=int, default=4096)
    parser.add_argument("--full-audit-origin-batch-size", type=int, default=4)
    parser.add_argument("--train-audit-origins", type=int, default=128)
    parser.add_argument("--p-kl-weight", type=float, default=1.0e-4)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def _sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_training_artifact(
    recorded_path: object,
    expected_sha256: object,
    *,
    training_root: Path | None = None,
) -> Path:
    """Resolve a provenance path after moving the QuantData training root.

    Completion proofs retain the original absolute path for auditability.  A
    copied artifact may instead live below the current
    ``quantdata.env.locations.training`` root.  The rebased candidate is
    accepted only when its content digest is unchanged.
    """
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("Training artifact has no valid SHA-256 provenance")
    recorded = Path(str(recorded_path))
    root = (
        Path(quantdata_env.locations.training.path)
        if training_root is None
        else training_root
    )
    candidates = [recorded]
    for marker in ("flex-shared-strikes", "flex-fixed-weekly", "flex-phase3"):
        if marker in recorded.parts:
            marker_index = recorded.parts.index(marker)
            candidates.append(root.joinpath(*recorded.parts[marker_index:]))
            break
    checked: list[str] = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            checked.append(f"{candidate} (missing)")
            continue
        actual_sha256 = checkpoint_digest(candidate)
        if actual_sha256 == expected_sha256:
            return candidate
        checked.append(f"{candidate} (digest {actual_sha256})")
    raise ValueError(
        "Could not resolve a checksum-valid migrated training artifact; checked "
        + "; ".join(checked)
    )


def _load_inputs(
    completion_path: Path,
    *,
    env: BaseEnvironment,
) -> tuple[
    JointHierarchicalDynamicsModel,
    tuple[date, ...],
    torch.Tensor,
    np.ndarray,
    tuple[date, ...],
    torch.Tensor,
    np.ndarray,
    dict[str, str],
]:
    completion = _load_json(completion_path)
    if (
        completion.get("schema")
        != "flex-shared-strikes-joint-hierarchical-completion-v1"
        or completion.get("status") != "complete"
    ):
        raise ValueError("Joint hierarchical completion proof is not complete")
    report_path = _resolve_training_artifact(
        completion["report"],
        completion.get("report_sha256"),
    )
    report = _load_json(report_path)
    if report.get("status") != "complete" or not cast(
        Mapping[str, object], report.get("convergence", {})
    ).get("converged"):
        raise ValueError("Joint hierarchical report lacks convergence evidence")

    checkpoint_path = _resolve_training_artifact(
        completion["checkpoint"],
        completion.get("checkpoint_sha256"),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema") != "flex-joint-hierarchical-dynamics-v1"
        or checkpoint.get("phase") != "complete"
    ):
        raise ValueError("Joint hierarchical model checkpoint is not complete")

    history_path = _resolve_training_artifact(
        report["target_history"],
        report.get("target_history_sha256"),
    )
    history = torch.load(history_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(history, Mapping)
        or history.get("schema") != "flex-joint-hierarchical-target-history-v1"
        or history.get("joint_checkpoint_sha256")
        != completion.get("checkpoint_sha256")
    ):
        raise ValueError("Joint hierarchical target history has invalid provenance")
    train_states = cast(torch.Tensor, history["train_states"]).detach().cpu().double()
    validation_states = cast(
        torch.Tensor, history["validation_quote_optimal_states"]
    ).detach().cpu().double()
    train_dates = tuple(date.fromisoformat(str(value)) for value in history["train_dates"])
    validation_dates = tuple(
        date.fromisoformat(str(value)) for value in history["validation_dates"]
    )
    spot_refs = cast(torch.Tensor, history["spot_refs"]).detach().cpu().double()
    if (
        train_states.ndim != 2
        or validation_states.ndim != 2
        or train_states.shape[1] != validation_states.shape[1]
        or len(train_dates) != len(train_states)
        or len(validation_dates) != len(validation_states)
        or any(value.year >= 2025 for value in train_dates)
        or any(value.year != 2025 for value in validation_dates)
        or spot_refs.shape != (len(train_dates) + len(validation_dates),)
        or not bool(torch.all(torch.isfinite(spot_refs) & (spot_refs > 0.0)))
    ):
        raise ValueError("Joint hierarchical train/validation history is malformed")
    if _sha256_tensor(train_states) != report.get("train_state_sha256"):
        raise ValueError("Training-state digest differs from the joint report")
    if _sha256_tensor(validation_states) != report.get(
        "validation_quote_optimal_state_sha256"
    ):
        raise ValueError("Validation-state digest differs from the joint report")

    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, Mapping) or not isinstance(raw_config.get("label"), str):
        raise ValueError("Joint checkpoint has no reconstructable model config")
    model_config = JointHierarchicalDynamicsConfig(label=raw_config["label"])
    if asdict(model_config) != dict(raw_config):
        raise ValueError("Joint checkpoint geometry differs from current code")
    model = JointHierarchicalDynamicsModel(
        env,
        num_days=len(train_states),
        config=model_config,
    )
    model.load_state_dict(cast(Mapping[str, torch.Tensor], checkpoint["model_state_dict"]))
    if not torch.allclose(
        model.daily_states.detach().cpu().double(), train_states, atol=0.0, rtol=0.0
    ):
        raise ValueError("Joint checkpoint and target history training states differ")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    log_spot = torch.log(spot_refs).numpy()
    return (
        model,
        train_dates,
        train_states.to(dtype=env.fdtype, device=env.device),
        log_spot[: len(train_dates)],
        validation_dates,
        validation_states.to(dtype=env.fdtype, device=env.device),
        log_spot[len(train_dates) :],
        {
            "completion": str(completion_path),
            "completion_sha256": checkpoint_digest(completion_path),
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest(checkpoint_path),
            "target_history": str(history_path),
            "target_history_sha256": checkpoint_digest(history_path),
        },
    )


def _subset_evenly(
    data: WeeklyMeasureTrainingData,
    maximum_rows: int,
) -> WeeklyMeasureTrainingData:
    rows = min(len(data.reduced_states), maximum_rows)
    indices = torch.linspace(
        0,
        len(data.reduced_states) - 1,
        rows,
        device=data.reduced_states.device,
    ).round().long()
    return WeeklyMeasureTrainingData(
        reduced_states=data.reduced_states[indices],
        front_support=data.front_support[indices],
        front_density=data.front_density[indices],
        front_backbone_total_vol=data.front_backbone_total_vol[indices],
        current_calls=data.current_calls[indices],
    )


def _require_front_week_consistency(
    audit: Mapping[str, object],
    *,
    label: str,
) -> None:
    """Reject a run whose implied spot draw does not reproduce front calls."""
    mass_error = float(audit["maximum_density_mass_error"])
    analytic_mean_error = float(
        audit["maximum_analytic_mean_gross_return_error"]
    )
    within_three_se = float(
        audit["sampled_mean_gross_return_percent_within_3_mc_se"]
    )
    option_within_three_se = float(audit["percent_within_3_mc_se"])
    if (
        mass_error > 5.0e-5
        or analytic_mean_error > 5.0e-5
        or within_three_se < 95.0
        or option_within_three_se < 95.0
    ):
        raise RuntimeError(
            f"{label} front-week implied-spot consistency failed: "
            f"mass_error={mass_error:.6g}, "
            f"analytic_mean_error={analytic_mean_error:.6g}, "
            f"spot_within_3se={within_three_se:.3f}%, "
            f"calls_within_3se={option_within_three_se:.3f}%"
        )


def run(arguments: argparse.Namespace) -> None:
    """Certify 1W implied spot, then fit only Q1 against the 2W roll."""
    if arguments.device != "cuda:0":
        raise ValueError("Joint hierarchical Q1 training is restricted to GPU 0")
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    (
        model,
        train_dates,
        train_states,
        train_log_spot,
        validation_dates,
        validation_states,
        validation_log_spot,
        sources,
    ) = _load_inputs(arguments.joint_completion, env=env)
    spot_fit = fit_joint_spot_correlation_under_p(
        model,
        train_states.detach().cpu().double().numpy(),
        train_log_spot,
        train_dates,
    )
    spot_fit_training = evaluate_joint_spot_correlation_under_p(
        spot_fit,
        model,
        train_states.detach().cpu().double().numpy(),
        train_log_spot,
        train_dates,
    )
    spot_fit_validation = evaluate_joint_spot_correlation_under_p(
        spot_fit,
        model,
        validation_states.detach().cpu().double().numpy(),
        validation_log_spot,
        validation_dates,
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
    log_moneyness = (
        -0.30,
        -0.24,
        -0.18,
        -0.12,
        -0.06,
        0.0,
        0.06,
        0.12,
        0.18,
        0.24,
        0.30,
    )
    normalized_strikes = torch.exp(
        torch.tensor(log_moneyness, dtype=env.fdtype, device=env.device)
    )
    training_data = precompute_measure_training_data(
        simulator,
        train_states,
        normalized_strikes,
        num_expiries=2,
    )
    validation_data = precompute_measure_training_data(
        simulator,
        validation_states,
        normalized_strikes,
        num_expiries=2,
    )
    training_audit_data = _subset_evenly(
        training_data,
        arguments.train_audit_origins,
    )
    kernel_kind = str(arguments.kernel_kind)
    kernel_type = (
        NormalWeeklyMeasureKernel
        if kernel_kind == "normal"
        else GenericWeeklyMeasureKernel
    )
    measure_config = WeeklyMeasureConfig(
        label=f"joint-hierarchical-200-{kernel_kind}-q1-v1",
        state_sizes_by_expiry=model.config.state_sizes_by_expiry,
        hierarchical=True,
    )
    train_config = WeeklyMeasureTrainConfig(
        epochs_per_line=arguments.epochs,
        origins_per_batch=arguments.origins_per_batch,
        paths_per_origin=arguments.paths_per_origin,
        audit_origins=arguments.audit_origins,
        audit_paths_per_origin=arguments.audit_paths_per_origin,
        learning_rate=arguments.learning_rate,
        p_regularization_weight=arguments.p_kl_weight,
        compile_objective=not arguments.no_compile,
        antithetic_paths=True,
        maximum_roll_lines=1,
        log_moneyness=log_moneyness,
    )

    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    stem = f"joint-hierarchical-{kernel_kind}-q1"
    report_path = arguments.report_directory / f"{stem}.json"
    completion_path = (
        arguments.report_directory / f"{stem}-complete.json"
    )
    checkpoint_path = arguments.checkpoint_directory / f"{stem}.pt"

    baseline_kernel = kernel_type(
        env,
        dynamics=dynamics,
        config=measure_config,
    )
    front_train = audit_front_week_consistency(
        training_audit_data,
        normalized_strikes,
        paths_per_origin=arguments.full_audit_paths_per_origin,
        origins_per_batch=arguments.full_audit_origin_batch_size,
        seed=train_config.seed + 20_000,
    )
    front_validation = audit_front_week_consistency(
        validation_data,
        normalized_strikes,
        paths_per_origin=arguments.full_audit_paths_per_origin,
        origins_per_batch=arguments.full_audit_origin_batch_size,
        seed=train_config.seed + 20_001,
    )
    _require_front_week_consistency(front_train, label="training")
    _require_front_week_consistency(front_validation, label="2025 validation")
    p_train = audit_weekly_roll_line(
        baseline_kernel,
        simulator,
        training_audit_data,
        normalized_strikes,
        current_expiry_index=1,
        paths_per_origin=arguments.full_audit_paths_per_origin,
        origins_per_batch=arguments.full_audit_origin_batch_size,
        seed=train_config.seed + 30_000,
    )
    p_validation = audit_weekly_roll_line(
        baseline_kernel,
        simulator,
        validation_data,
        normalized_strikes,
        current_expiry_index=1,
        paths_per_origin=arguments.full_audit_paths_per_origin,
        origins_per_batch=arguments.full_audit_origin_batch_size,
        seed=train_config.seed + 30_001,
    )

    kernel = kernel_type(env, dynamics=dynamics, config=measure_config)
    with FileLock(
        str(arguments.checkpoint_directory / f"joint-{kernel_kind}-q1.lock")
    ):
        training = fit_weekly_measure_kernel(
            kernel,
            simulator,
            training_data,
            validation_data,
            checkpoint_path=checkpoint_path,
            config=train_config,
        )
    q_train = audit_weekly_roll_line(
        kernel,
        simulator,
        training_audit_data,
        normalized_strikes,
        current_expiry_index=1,
        paths_per_origin=arguments.full_audit_paths_per_origin,
        origins_per_batch=arguments.full_audit_origin_batch_size,
        seed=train_config.seed + 30_000,
    )
    q_validation = audit_weekly_roll_line(
        kernel,
        simulator,
        validation_data,
        normalized_strikes,
        current_expiry_index=1,
        paths_per_origin=arguments.full_audit_paths_per_origin,
        origins_per_batch=arguments.full_audit_origin_batch_size,
        seed=train_config.seed + 30_001,
    )
    report: dict[str, object] = {
        "schema": f"flex-joint-hierarchical-{kernel_kind}-q1-report-v1",
        "status": "complete",
        "measure": (
            "Q1 conditional Gaussian change on the exact one-week step"
            if kernel_kind == "normal"
            else (
                "Q1 conditional generic triangular full-support transport "
                "on the exact one-week step"
            )
        ),
        "kernel_kind": kernel_kind,
        "target": "today 2W fixed-cash-strike calls equal expected next-week 1W calls",
        "spot_law": (
            "current front-week SANOS inverse CDF; its standard-normal rank is "
            "unchanged by Q1"
        ),
        "train_range": [train_dates[0].isoformat(), train_dates[-1].isoformat()],
        "validation_range": [
            validation_dates[0].isoformat(),
            validation_dates[-1].isoformat(),
        ],
        "num_train_states": len(train_states),
        "num_validation_states": len(validation_states),
        "num_train_full_audit_states": len(training_audit_data.reduced_states),
        "date_weighting": "equal origin dates; no quote-count weighting",
        "path_generation": (
            "fresh origin-wise antithetic Torch normals generated inside every batch"
        ),
        "measure_config": asdict(measure_config),
        "train_config": asdict(train_config),
        "sources": sources,
        "spot_correlation_under_p": {
            "fit": spot_fit.to_dict(),
            "training": spot_fit_training,
            "validation_2025": spot_fit_validation,
            "implied_sampling_semantics": (
                "standardize the P-fitted correlated Gaussian spot innovation, "
                "apply Phi, then the current front SANOS inverse CDF"
            ),
        },
        "one_week_consistency": {
            "training": front_train,
            "validation_2025": front_validation,
        },
        "two_week_roll": {
            "p_baseline_training": p_train,
            "q1_training": q_train,
            "p_baseline_validation_2025": p_validation,
            "q1_validation_2025": q_validation,
        },
        "training": training,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest(checkpoint_path),
        "ad_hoc_choices": [
            "11 log-moneyness calls from -0.30 to 0.30",
            (
                "two-layer width-256 SiLU Q1 context network with conditional "
                "mean/scale"
                if kernel_kind == "normal"
                else (
                    "two-layer width-256 SiLU Q1 context network with triangular "
                    "shift/scale/tanh-amplitude/gate transport"
                )
            ),
            f"P-proximity regularization weight {arguments.p_kl_weight:g}",
            f"{arguments.epochs} constant-learning-rate epochs",
            f"{arguments.origins_per_batch} origins x "
            f"{arguments.paths_per_origin} live antithetic paths per update",
        ],
    }
    write_json_atomic(report, report_path)
    write_json_atomic(
        {
            "schema": f"flex-joint-hierarchical-{kernel_kind}-q1-completion-v1",
            "status": "complete",
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest(checkpoint_path),
        },
        completion_path,
    )


def main() -> None:
    """CLI entry point with durable exception reporting."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_joint_hierarchical_q1")
