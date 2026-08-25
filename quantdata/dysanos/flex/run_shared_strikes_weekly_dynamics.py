"""Fit shared-strike full/direct/hierarchical weekly physical dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from filelock import FileLock

from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.run_shared_strikes_compressed_encoders import (
    _HIERARCHICAL_200_ALLOCATION,
)
from quantdata.dysanos.flex.weekly_dynamics import (
    PHASE3_TRAIN_END,
    PHASE3_TRAIN_START,
    PHASE3_VALIDATION_END,
    PHASE3_VALIDATION_START,
    WeeklyDynamicsConfig,
    WeeklyTrainingHistory,
    fit_phase3_training_history,
    load_weekly_dynamics_fit,
    save_weekly_dynamics_fit,
)
from quantdata.dysanos.flex.weekly_validation import evaluate_oos_dynamics

_FULL_EXPIRY_STATE_SIZE = 202
_NUM_EXPIRIES = 14


@dataclass(frozen=True)
class _HistorySource:
    """One immutable market-encoded state history and its P-model geometry."""

    name: str
    history_path: Path
    state_sizes_by_expiry: tuple[int, ...]
    structure: str
    provenance: dict[str, object]


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-completion", type=Path, required=True)
    parser.add_argument("--compressed-completion", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=30.0)
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


def _history_reference(report: Mapping[str, object]) -> Path:
    encoded = report.get("encoded_history")
    if not isinstance(encoded, Mapping) or not isinstance(encoded.get("path"), str):
        raise ValueError("Encoder report has no immutable encoded-history path")
    return Path(encoded["path"])


def _load_sources(
    *,
    full_completion: Path,
    compressed_completion: Path,
) -> tuple[_HistorySource, ...]:
    full_proof = json.loads(full_completion.read_text(encoding="utf-8"))
    full_report_path = Path(str(full_proof["report"]))
    full_report = json.loads(full_report_path.read_text(encoding="utf-8"))
    if full_report.get("status") != "complete":
        raise ValueError("Full encoder report is not complete")

    compressed_proof = json.loads(compressed_completion.read_text(encoding="utf-8"))
    comparison_path = Path(str(compressed_proof["comparison_report"]))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    reports = comparison.get("reports")
    if not isinstance(reports, Mapping):
        raise ValueError("Compressed encoder comparison has no candidate reports")
    direct = reports.get("direct")
    hierarchical = reports.get("hierarchical")
    if not isinstance(direct, Mapping) or not isinstance(hierarchical, Mapping):
        raise ValueError("Compressed encoder comparison is missing a candidate")
    if direct.get("status") != "complete" or hierarchical.get("status") != "complete":
        raise ValueError("A compressed market encoder is not complete")

    return (
        _HistorySource(
            name="full-identity-2828",
            history_path=_history_reference(full_report),
            state_sizes_by_expiry=(_FULL_EXPIRY_STATE_SIZE,) * _NUM_EXPIRIES,
            structure="expiry-hierarchical identity decoder",
            provenance={
                "completion": str(full_completion),
                "completion_sha256": _sha256(full_completion),
                "report": str(full_report_path),
                "report_sha256": _sha256(full_report_path),
            },
        ),
        _HistorySource(
            name="direct-200",
            history_path=_history_reference(direct),
            state_sizes_by_expiry=(200,),
            structure="one global state block; unrestricted direct decoder",
            provenance={
                "completion": str(compressed_completion),
                "completion_sha256": _sha256(compressed_completion),
                "comparison": str(comparison_path),
                "comparison_sha256": _sha256(comparison_path),
            },
        ),
        _HistorySource(
            name="hierarchical-200",
            history_path=_history_reference(hierarchical),
            state_sizes_by_expiry=_HIERARCHICAL_200_ALLOCATION,
            structure="expiry-triangular decoder and lag-only expiry hierarchy",
            provenance={
                "completion": str(compressed_completion),
                "completion_sha256": _sha256(compressed_completion),
                "comparison": str(comparison_path),
                "comparison_sha256": _sha256(comparison_path),
            },
        ),
    )


def _load_history(source: _HistorySource) -> WeeklyTrainingHistory:
    payload = torch.load(source.history_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "flex-shared-encoded-history-v1"
    ):
        raise ValueError(f"Invalid shared encoded history: {source.history_path}")
    raw_dates = payload.get("dates")
    states = payload.get("states")
    spot_refs = payload.get("spot_refs")
    if not isinstance(raw_dates, list) or not isinstance(states, torch.Tensor):
        raise ValueError("Encoded history is missing dates or state tensor")
    if not isinstance(spot_refs, torch.Tensor):
        raise ValueError("Encoded history is missing spot references")
    dates = tuple(date.fromisoformat(str(value)) for value in raw_dates)
    state_values = states.detach().cpu().double().numpy()
    spot_values = spot_refs.detach().cpu().double().numpy()
    if state_values.shape != (len(dates), sum(source.state_sizes_by_expiry)):
        raise ValueError(
            f"{source.name} history shape {state_values.shape} does not match "
            f"{sum(source.state_sizes_by_expiry)} state coordinates"
        )
    if np.any(spot_values <= 0.0):
        raise ValueError(f"{source.name} contains a non-positive reference spot")
    if any(value.year >= 2026 for value in dates):
        raise ValueError(f"2026 data entered {source.name}")
    return WeeklyTrainingHistory(
        dates=dates,
        states=state_values,
        log_spot=np.log(spot_values),
    )


def _mean_reversion_summary(generator: np.ndarray) -> dict[str, object]:
    eigenvalues = np.linalg.eigvals(generator)
    real = eigenvalues.real
    positive = real[real > 0.0]
    half_lives = np.log(2.0) / positive
    return {
        "dimension": len(real),
        "minimum_real_rate_per_week": float(np.min(real)),
        "median_real_rate_per_week": float(np.median(real)),
        "maximum_real_rate_per_week": float(np.max(real)),
        "minimum_half_life_weeks": float(np.min(half_lives)),
        "median_half_life_weeks": float(np.median(half_lives)),
        "maximum_half_life_weeks": float(np.max(half_lives)),
        "maximum_imaginary_rate_abs": float(np.max(np.abs(eigenvalues.imag))),
    }


def _fit_variant(
    source: _HistorySource,
    history: WeeklyTrainingHistory,
    *,
    pca_mode: str,
    checkpoint_directory: Path,
) -> dict[str, object]:
    training = history.subset(PHASE3_TRAIN_START, PHASE3_TRAIN_END)
    validation = history.subset(PHASE3_VALIDATION_START, PHASE3_VALIDATION_END)
    label_mode = "pca-all" if pca_mode == "full" else "pca-995"
    config = WeeklyDynamicsConfig(
        label=f"shared-strikes-{source.name}-{label_mode}",
        state_sizes_by_expiry=source.state_sizes_by_expiry,
        pca_mode=pca_mode,
        pca_variance_threshold=0.995,
        ridge_penalty=0.0,
    )
    checkpoint_path = checkpoint_directory / f"{source.name}-{label_mode}.pt"
    checkpoint_source = {
        **source.provenance,
        "history": str(source.history_path),
        "history_sha256": _sha256(source.history_path),
        "structure": source.structure,
    }
    if checkpoint_path.exists():
        fit, saved_source = load_weekly_dynamics_fit(checkpoint_path)
        if fit.config != config or saved_source != checkpoint_source:
            raise ValueError(
                "Existing weekly P checkpoint identity/source mismatch; "
                f"refusing to overwrite {checkpoint_path}"
            )
        checkpoint_action = "loaded"
    else:
        fit = fit_phase3_training_history(training, config=config)
        save_weekly_dynamics_fit(
            fit,
            checkpoint_path,
            source=checkpoint_source,
        )
        checkpoint_action = "fitted"
    validation_report = evaluate_oos_dynamics(fit, validation, training)
    if not validation_report["all_horizons_finite"]:
        raise RuntimeError(f"{source.name}/{label_mode} OOS scores are non-finite")
    if not validation_report["all_horizons_nondegenerate"]:
        raise RuntimeError(f"{source.name}/{label_mode} OOS dynamics collapsed")
    return {
        "status": "complete",
        "source": source.name,
        "structure": source.structure,
        "pca_label": label_mode,
        "config": asdict(config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_action": checkpoint_action,
        "num_train_dates": len(training.dates),
        "num_validation_dates": len(validation.dates),
        "reduced_sizes_by_expiry": list(fit.reduced_sizes_by_expiry),
        "reduced_state_size": fit.reduced_state_size,
        "mean_reversion": _mean_reversion_summary(fit.mean_reversion_generator),
        "diagnostics": fit.diagnostics,
        "validation_2025": validation_report,
    }


def run(arguments: argparse.Namespace) -> None:
    """Wait for clean encoders, then fit every declared P-model control."""
    _wait_for_completion(arguments.full_completion, wait_seconds=arguments.wait_seconds)
    _wait_for_completion(
        arguments.compressed_completion,
        wait_seconds=arguments.wait_seconds,
    )
    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    report_path = arguments.report_directory / "shared-strikes-weekly-p.json"
    completion_path = arguments.report_directory / "shared-strikes-weekly-p-complete.json"
    report: dict[str, object] = {
        "schema": "flex-shared-strikes-weekly-p-v1",
        "status": "fitting",
        "measure": "P",
        "train_range": [PHASE3_TRAIN_START.isoformat(), PHASE3_TRAIN_END.isoformat()],
        "validation_range": [
            PHASE3_VALIDATION_START.isoformat(),
            PHASE3_VALIDATION_END.isoformat(),
        ],
        "date_weighting": "one equal date/transition; never weighted by quote count",
        "time_scaling": (
            "fit exact adjacent-session standardized state levels, compute the full "
            "K=-log(A), and precompute exp(-5K); weekly innovations use exact "
            "five-business-session residuals"
        ),
        "primary_q_source": "hierarchical-200/pca-all",
        "no_fallback_policy": (
            "candidate failures are recorded explicitly and never replaced by a "
            "different dynamics without review"
        ),
        "ad_hoc_choices": [
            "99.5% per-block level-PCA threshold for reduced diagnostics",
            "zero ridge penalty",
            "Gaussian exact-week innovation likelihood",
            "DYSANOS/grid lag-only hierarchy: next expiry j uses lagged j and j-1",
        ],
        "variants": {},
    }
    write_json_atomic(report, report_path)
    sources = _load_sources(
        full_completion=arguments.full_completion,
        compressed_completion=arguments.compressed_completion,
    )
    # The two 200-state candidates establish the practical simulator first.
    # The 2,828-dimensional PCA-all control is deliberately last because its
    # dense matrix logarithm and Lyapunov solve are substantially more costly.
    ordered = (
        (sources[0], "reduced"),
        (sources[1], "reduced"),
        (sources[1], "full"),
        (sources[2], "reduced"),
        (sources[2], "full"),
        (sources[0], "full"),
    )
    variants: dict[str, object] = {}
    with FileLock(str(arguments.checkpoint_directory / "weekly-p.lock")):
        for source, pca_mode in ordered:
            label_mode = "pca-all" if pca_mode == "full" else "pca-995"
            key = f"{source.name}/{label_mode}"
            print(f"starting weekly P candidate {key}", flush=True)
            try:
                variants[key] = _fit_variant(
                    source,
                    _load_history(source),
                    pca_mode=pca_mode,
                    checkpoint_directory=arguments.checkpoint_directory,
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                variants[key] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                print(f"weekly P candidate failed {key}: {error}", flush=True)
            report["variants"] = variants
            report["status"] = f"evaluated_{key.replace('/', '_')}"
            write_json_atomic(report, report_path)
            if key == "hierarchical-200/pca-all" and (
                isinstance(variants[key], Mapping)
                and variants[key].get("status") == "complete"
            ):
                # Q depends only on the primary hierarchical 200-state law.
                # Publish that immutable gate before the intentionally costly
                # 2,828-dimensional PCA-all control, which continues below.
                write_json_atomic(
                    {
                        "schema": "flex-shared-strikes-weekly-p-primary-v1",
                        "status": "complete",
                        "report": str(report_path),
                        "primary": "hierarchical-200/pca-all",
                        "primary_checkpoint": variants[key]["checkpoint"],
                        "full_control_status": "pca-all pending",
                    },
                    arguments.report_directory
                    / "shared-strikes-weekly-p-primary-complete.json",
                )

    primary = variants.get("hierarchical-200/pca-all")
    primary_complete = isinstance(primary, Mapping) and primary.get("status") == (
        "complete"
    )
    failures = [
        key
        for key, value in variants.items()
        if isinstance(value, Mapping) and value.get("status") == "failed"
    ]
    report["status"] = "complete" if primary_complete else "failed_primary"
    report["candidate_failures"] = failures
    write_json_atomic(report, report_path)
    if not primary_complete:
        write_json_atomic(
            {
                "schema": "flex-shared-strikes-weekly-p-failure-v1",
                "status": "failed_primary",
                "report": str(report_path),
                "report_sha256": _sha256(report_path),
                "primary": "hierarchical-200/pca-all",
                "candidate_failures": failures,
            },
            arguments.report_directory / "shared-strikes-weekly-p-failed.json",
        )
        raise RuntimeError(
            "The primary hierarchical-200/pca-all P dynamics failed; no "
            "alternative candidate was selected silently"
        )
    write_json_atomic(
        {
            "schema": "flex-shared-strikes-weekly-p-completion-v1",
            "status": report["status"],
            "report": str(report_path),
            "report_sha256": _sha256(report_path),
            "primary": "hierarchical-200/pca-all",
            "primary_checkpoint": (
                primary.get("checkpoint") if isinstance(primary, Mapping) else None
            ),
            "candidate_failures": failures,
        },
        completion_path,
    )


def main() -> None:
    """CLI entry point with persistent top-level failure reporting."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_shared_strikes_weekly_dynamics")
