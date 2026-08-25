"""Fit normal and generic weekly Q kernels after shared-strike P dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from filelock import FileLock

from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.weekly_dynamics import (
    PHASE3_TRAIN_END,
    PHASE3_TRAIN_START,
    PHASE3_VALIDATION_END,
    PHASE3_VALIDATION_START,
    TorchWeeklyDynamics,
    WeeklyTrainingHistory,
    load_weekly_dynamics_fit,
)
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    NormalWeeklyMeasureKernel,
    WeeklyMeasureConfig,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    WeeklyMeasureTrainConfig,
    fit_weekly_measure_kernel,
    precompute_measure_training_data,
)
from quantdata.dysanos.flex.weekly_simulation import (
    WeeklySanosSimulator,
    load_shared_strikes_calibrated_model,
)
from quantdata.dysanos.flex.weekly_validation import project_strong_states
from quantdata.torch import BaseEnvironment


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-completion", type=Path, required=True)
    parser.add_argument("--compressed-completion", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    parser.add_argument("--epochs-per-line", type=int, default=200)
    parser.add_argument("--paths-per-origin", type=int, default=128)
    parser.add_argument("--origins-per-batch", type=int, default=16)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wait_for_completion(path: Path, *, wait_seconds: float) -> dict[str, object]:
    if not 0.0 < wait_seconds <= 60.0:
        raise ValueError("'wait_seconds' must lie in (0,60]")
    while not path.exists():
        print(f"waiting for completion proof {path}", flush=True)
        time.sleep(wait_seconds)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(
            f"Upstream completion has status {payload.get('status')!r}: {path}"
        )
    return payload


def _compressed_reports(completion: Mapping[str, object]) -> Mapping[str, object]:
    comparison_path = Path(str(completion["comparison_report"]))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    reports = comparison.get("reports")
    if not isinstance(reports, Mapping):
        raise ValueError("Compressed completion has no candidate reports")
    return reports


def _load_encoded_history(report: Mapping[str, object]) -> WeeklyTrainingHistory:
    encoded = report.get("encoded_history")
    if not isinstance(encoded, Mapping) or not isinstance(encoded.get("path"), str):
        raise ValueError("Compressed report has no encoded history")
    path = Path(encoded["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "flex-shared-encoded-history-v1"
    ):
        raise ValueError(f"Invalid compressed history {path}")
    raw_dates = payload.get("dates")
    states = payload.get("states")
    spots = payload.get("spot_refs")
    if (
        not isinstance(raw_dates, list)
        or not isinstance(states, torch.Tensor)
        or not isinstance(spots, torch.Tensor)
    ):
        raise ValueError("Compressed history is incomplete")
    dates = tuple(date.fromisoformat(str(value)) for value in raw_dates)
    spot_values = spots.detach().cpu().double().numpy()
    if np.any(spot_values <= 0.0) or any(value.year >= 2026 for value in dates):
        raise ValueError("Compressed history has invalid spot or 2026 data")
    return WeeklyTrainingHistory(
        dates=dates,
        states=states.detach().cpu().double().numpy(),
        log_spot=np.log(spot_values),
    )


def _long_horizon_diagnostics(
    simulator: WeeklySanosSimulator,
    kernel: NormalWeeklyMeasureKernel | GenericWeeklyMeasureKernel,
    training_states: torch.Tensor,
    *,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device=kernel.device)
    generator.manual_seed(seed)
    initial = torch.mean(training_states, dim=0)
    with torch.no_grad():
        paths = simulator.simulate_paths(
            initial,
            torch.tensor(1.0, dtype=kernel.fdtype, device=kernel.device),
            num_paths=256,
            num_steps=520,
            spot_mode="implied",
            generator=generator,
            measure_kernel=kernel,
        )
    state = paths.reduced_states
    log_spot = paths.log_spots
    finite = bool(torch.all(torch.isfinite(state)) and torch.all(torch.isfinite(log_spot)))
    maximum_abs_state = float(torch.max(torch.abs(state)).cpu())
    diagnostics: dict[str, object] = {
        "paths": 256,
        "weeks": 520,
        "finite": finite,
        "maximum_abs_reduced_state": maximum_abs_state,
        "terminal_log_spot_quantiles": [
            float(value)
            for value in torch.quantile(
                log_spot[:, -1],
                torch.tensor(
                    [0.01, 0.5, 0.99], dtype=kernel.fdtype, device=kernel.device
                ),
            ).cpu()
        ],
        "non_explosive_gate": finite and maximum_abs_state < 1.0e6,
    }
    if isinstance(kernel, GenericWeeklyMeasureKernel):
        normals = torch.randn(
            (len(training_states), kernel.config.state_size + 1),
            dtype=kernel.fdtype,
            device=kernel.device,
            generator=generator,
        )
        diagnostics["minimum_triangular_jacobian_diagonal"] = float(
            kernel.minimum_coordinate_derivative(
                training_states,
                torch.ones(
                    len(training_states), dtype=kernel.fdtype, device=kernel.device
                ),
                normals,
            ).detach().cpu()
        )
    if diagnostics["non_explosive_gate"] is not True:
        raise RuntimeError("Target-measure long-horizon simulation exploded")
    return diagnostics


def _fit_candidate(
    *,
    structure: str,
    kernel_kind: str,
    p_variant: Mapping[str, object],
    encoder_report: Mapping[str, object],
    checkpoint_directory: Path,
    train_config: WeeklyMeasureTrainConfig,
    env: BaseEnvironment,
) -> dict[str, object]:
    dynamics_checkpoint = Path(str(p_variant["checkpoint"]))
    fit, dynamics_source = load_weekly_dynamics_fit(dynamics_checkpoint)
    dynamics = TorchWeeklyDynamics(env, fit=fit)
    decoder_checkpoint = Path(str(encoder_report["decoder_checkpoint"]))
    calibrated = load_shared_strikes_calibrated_model(env, decoder_checkpoint)
    simulator = WeeklySanosSimulator(
        env,
        dynamics=dynamics,
        calibrated_model=calibrated,
    )
    history = _load_encoded_history(encoder_report)
    training_history = history.subset(PHASE3_TRAIN_START, PHASE3_TRAIN_END)
    validation_history = history.subset(
        PHASE3_VALIDATION_START, PHASE3_VALIDATION_END
    )
    training_reduced = torch.as_tensor(
        project_strong_states(fit, training_history.states),
        dtype=env.fdtype,
        device=env.device,
    )
    validation_reduced = torch.as_tensor(
        project_strong_states(fit, validation_history.states),
        dtype=env.fdtype,
        device=env.device,
    )
    normalized_strikes = torch.exp(
        torch.tensor(
            train_config.log_moneyness,
            dtype=env.fdtype,
            device=env.device,
        )
    )
    training_data = precompute_measure_training_data(
        simulator, training_reduced, normalized_strikes
    )
    validation_data = precompute_measure_training_data(
        simulator, validation_reduced, normalized_strikes
    )
    measure_config = WeeklyMeasureConfig(
        label=f"shared-strikes-{structure}-{kernel_kind}-q-v1",
        state_sizes_by_expiry=fit.reduced_sizes_by_expiry,
        hierarchical=structure == "hierarchical-200",
    )
    kernel: NormalWeeklyMeasureKernel | GenericWeeklyMeasureKernel
    if kernel_kind == "normal":
        kernel = NormalWeeklyMeasureKernel(
            env, dynamics=dynamics, config=measure_config
        )
    elif kernel_kind == "generic":
        kernel = GenericWeeklyMeasureKernel(
            env, dynamics=dynamics, config=measure_config
        )
    else:
        raise ValueError(f"unknown Q kernel kind {kernel_kind!r}")
    checkpoint_path = checkpoint_directory / f"{structure}-{kernel_kind}.pt"
    training_report = fit_weekly_measure_kernel(
        kernel,
        simulator,
        training_data,
        validation_data,
        checkpoint_path=checkpoint_path,
        config=train_config,
    )
    long_horizon = _long_horizon_diagnostics(
        simulator,
        kernel,
        training_reduced,
        seed=train_config.seed + (0 if kernel_kind == "normal" else 1),
    )
    return {
        "status": "complete",
        "structure": structure,
        "kernel": kernel_kind,
        "measure_config": asdict(measure_config),
        "train_config": asdict(train_config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "p_dynamics_checkpoint": str(dynamics_checkpoint),
        "p_dynamics_checkpoint_sha256": _sha256(dynamics_checkpoint),
        "p_dynamics_source": dynamics_source,
        "decoder_checkpoint": str(decoder_checkpoint),
        "decoder_checkpoint_sha256": _sha256(decoder_checkpoint),
        "num_train_origins": len(training_history.dates),
        "num_validation_origins": len(validation_history.dates),
        "training": training_report,
        "long_horizon": long_horizon,
    }


def run(arguments: argparse.Namespace) -> None:
    """Fit the hierarchical kernels first, then direct joint controls."""
    if arguments.device != "cuda:0":
        raise ValueError("Shared-strike Q training is restricted to GPU 0")
    p_proof = _wait_for_completion(
        arguments.p_completion, wait_seconds=arguments.wait_seconds
    )
    compressed_proof = _wait_for_completion(
        arguments.compressed_completion, wait_seconds=arguments.wait_seconds
    )
    p_report_path = Path(str(p_proof["report"]))
    p_report = json.loads(p_report_path.read_text(encoding="utf-8"))
    variants = p_report.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("Weekly P report has no variants")
    encoder_reports = _compressed_reports(compressed_proof)
    train_config = WeeklyMeasureTrainConfig(
        epochs_per_line=arguments.epochs_per_line,
        paths_per_origin=arguments.paths_per_origin,
        origins_per_batch=arguments.origins_per_batch,
        compile_objective=not arguments.no_compile,
    )
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    report_path = arguments.report_directory / "shared-strikes-weekly-q.json"
    report: dict[str, object] = {
        "schema": "flex-shared-strikes-weekly-q-v1",
        "status": "fitting",
        "source_p_completion": str(arguments.p_completion),
        "source_p_completion_sha256": _sha256(arguments.p_completion),
        "train_range": [PHASE3_TRAIN_START.isoformat(), PHASE3_TRAIN_END.isoformat()],
        "validation_range": [
            PHASE3_VALIDATION_START.isoformat(),
            PHASE3_VALIDATION_END.isoformat(),
        ],
        "spot_law": "current front-week SANOS inverse CDF under every Q candidate",
        "date_weighting": "one equal current-date origin, independent of quote count",
        "training_semantics": (
            "Q has no observed transition labels; fit conditional fixed-cash-strike "
            "roll moments with Monte Carlo paths and regularization to P"
        ),
        "primary_candidates": [
            "hierarchical-200/normal",
            "hierarchical-200/generic",
        ],
        "ad_hoc_choices": [
            "11 log-moneyness strikes from -0.30 to 0.30",
            "two-layer width-256 SiLU conditional networks",
            f"{train_config.epochs_per_line} epochs per roll line",
            f"{train_config.origins_per_batch} origins x "
            f"{train_config.paths_per_origin} paths per update",
            "P-kernel regularization weight 1e-4",
            "520-week, 256-path non-explosion diagnostic",
        ],
        "variants": {},
    }
    write_json_atomic(report, report_path)
    order = (
        ("hierarchical-200", "normal"),
        ("hierarchical-200", "generic"),
        ("direct-200", "normal"),
        ("direct-200", "generic"),
    )
    completed: dict[str, object] = {}
    with FileLock(str(arguments.checkpoint_directory / "weekly-q.lock")):
        for structure, kernel_kind in order:
            key = f"{structure}/{kernel_kind}"
            p_variant = variants.get(f"{structure}/pca-all")
            encoder_key = "hierarchical" if structure.startswith("hierarchical") else "direct"
            encoder_report = encoder_reports.get(encoder_key)
            if not isinstance(p_variant, Mapping) or p_variant.get("status") != "complete":
                completed[key] = {
                    "status": "failed_prerequisite",
                    "error": f"P variant {structure}/pca-all is unavailable",
                }
            elif not isinstance(encoder_report, Mapping):
                completed[key] = {
                    "status": "failed_prerequisite",
                    "error": f"encoder report {encoder_key} is unavailable",
                }
            else:
                print(f"starting weekly Q candidate {key}", flush=True)
                try:
                    completed[key] = _fit_candidate(
                        structure=structure,
                        kernel_kind=kernel_kind,
                        p_variant=p_variant,
                        encoder_report=encoder_report,
                        checkpoint_directory=arguments.checkpoint_directory,
                        train_config=train_config,
                        env=env,
                    )
                except (RuntimeError, ValueError) as error:
                    completed[key] = {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    print(f"weekly Q candidate failed {key}: {error}", flush=True)
            report["variants"] = completed
            report["status"] = f"evaluated_{key.replace('/', '_')}"
            write_json_atomic(report, report_path)

    primary_complete = all(
        isinstance(completed.get(key), Mapping)
        and completed[key].get("status") == "complete"
        for key in cast(list[str], report["primary_candidates"])
    )
    report["status"] = "complete" if primary_complete else "failed_primary"
    write_json_atomic(report, report_path)
    if not primary_complete:
        write_json_atomic(
            {
                "schema": "flex-shared-strikes-weekly-q-failure-v1",
                "status": "failed_primary",
                "report": str(report_path),
                "report_sha256": _sha256(report_path),
            },
            arguments.report_directory / "shared-strikes-weekly-q-failed.json",
        )
        raise RuntimeError("One or both primary hierarchical Q kernels failed")
    write_json_atomic(
        {
            "schema": "flex-shared-strikes-weekly-q-completion-v1",
            "status": "complete",
            "report": str(report_path),
            "report_sha256": _sha256(report_path),
        },
        arguments.report_directory / "shared-strikes-weekly-q-complete.json",
    )


def main() -> None:
    """CLI entry point with persistent top-level failure reporting."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_shared_strikes_weekly_q")
