"""Fit and validate the Phase 3 weekly physical-measure dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market_encoder import load_encoded_strong_history
from quantdata.dysanos.flex.weekly_dynamics import (
    PHASE3_TRAIN_END,
    PHASE3_TRAIN_START,
    PHASE3_VALIDATION_END,
    PHASE3_VALIDATION_START,
    WeeklyDynamicsConfig,
    WeeklyTrainingHistory,
    fit_phase3_training_history,
    load_strong_weekly_history,
    load_weekly_dynamics_fit,
    save_weekly_dynamics_fit,
)
from quantdata.dysanos.flex.weekly_simulation_validation import (
    run_simulation_diagnostics,
)
from quantdata.dysanos.flex.weekly_validation import (
    evaluate_one_week_interval,
    evaluate_oos_dynamics,
)

_FLEX_SOURCE_DIRECTORY = Path(__file__).resolve().parent
_TRAINING_ROOT = Path(quantdata_env.locations.training.path)
_DEFAULT_DIRECT_CHECKPOINT = (
    _TRAINING_ROOT
    / "flex-fixed-weekly"
    / "spx-2022-2025-forward-mu-v1-stronger-parallel"
    / "triangular-stronger-140.pt"
)
_DEFAULT_DIRECT_REPORT = (
    _FLEX_SOURCE_DIRECTORY
    / "reports"
    / "batch-descent"
    / "rebalanced-stronger"
    / "spx-2022-2025-forward-mu-v1"
    / "spx-2022-2025-forward-mu-v1.json"
)
_DEFAULT_ENCODER_CHECKPOINT = (
    _TRAINING_ROOT
    / "flex-phase3"
    / "market-encoder-v1"
    / "market-to-strong.pt"
)
_DEFAULT_ENCODER_REPORT = (
    _FLEX_SOURCE_DIRECTORY
    / "reports"
    / "phase3-encoder"
    / "phase3-market-encoder.json"
)
_DEFAULT_CHECKPOINT_DIRECTORY = (
    _TRAINING_ROOT / "flex-phase3" / "weekly-dynamics-grid-v1"
)
_DEFAULT_REPORT_DIRECTORY = _FLEX_SOURCE_DIRECTORY / "reports" / "phase3-weekly"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_history(
    *,
    source_kind: str,
    source_checkpoint: Path,
    source_report: Path,
) -> tuple[WeeklyTrainingHistory, dict[str, Any]]:
    if source_kind == "encoder":
        encoded = load_encoded_strong_history(source_checkpoint, source_report)
        encoder_report = json.loads(source_report.read_text(encoding="utf-8"))
        expected_train_range = [
            PHASE3_TRAIN_START.isoformat(),
            PHASE3_TRAIN_END.isoformat(),
        ]
        expected_validation_range = [
            PHASE3_VALIDATION_START.isoformat(),
            PHASE3_VALIDATION_END.isoformat(),
        ]
        strong = encoder_report.get("strong")
        strong_training = strong.get("training") if isinstance(strong, dict) else None
        strong_convergence = (
            strong_training.get("convergence")
            if isinstance(strong_training, dict)
            else None
        )
        if (
            encoder_report.get("train_range") != expected_train_range
            or encoder_report.get("validation_range") != expected_validation_range
            or encoder_report.get("num_train_days") != 712
            or encoder_report.get("num_validation_days") != 162
            or not isinstance(strong_convergence, dict)
            or strong_convergence.get("converged") is not True
        ):
            raise ValueError(
                "Encoder report does not prove the locked train/validation split "
                "and converged train-only Strong decoder"
            )
        history = WeeklyTrainingHistory(
            dates=encoded.dates,
            states=encoded.states.astype(np.float64, copy=False),
            log_spot=np.log(encoded.spot_refs.astype(np.float64, copy=False)),
        )
        source = {
            "kind": "market-to-strong-encoder",
            "checkpoint": str(source_checkpoint),
            "checkpoint_sha256": encoded.encoder_checkpoint_sha256,
            "strong_decoder_sha256": encoded.strong_decoder_sha256,
            "report": str(source_report),
            "report_sha256": _sha256(source_report),
            "train_range": expected_train_range,
            "validation_range": expected_validation_range,
            "num_train_dates": 712,
            "num_validation_dates": 162,
        }
    elif source_kind == "direct-oracle":
        history = load_strong_weekly_history(
            source_checkpoint,
            source_report,
        )
        source = {
            "kind": "direct-strong-oracle",
            "checkpoint": str(source_checkpoint),
            "checkpoint_sha256": _sha256(source_checkpoint),
            "report": str(source_report),
            "report_sha256": _sha256(source_report),
            "warning": (
                "Phase 2 direct per-date states are an oracle benchmark, not a "
                "live market-to-state result."
            ),
        }
    else:
        raise ValueError(
            "source_kind must be 'encoder' or 'direct-oracle', found "
            f"{source_kind!r}"
        )
    if (
        history.dates[0] != PHASE3_TRAIN_START
        or history.dates[-1] != PHASE3_VALIDATION_END
        or any(value.year >= 2026 for value in history.dates)
    ):
        raise ValueError("Weekly Phase 3 history has invalid endpoints or includes 2026")
    return history, source


def _fit_or_load(
    training: WeeklyTrainingHistory,
    *,
    config: WeeklyDynamicsConfig,
    checkpoint_path: Path,
    source: dict[str, Any],
) -> tuple[Any, str]:
    if checkpoint_path.exists():
        fit, saved_source = load_weekly_dynamics_fit(checkpoint_path)
        if fit.config != config or saved_source != source:
            raise ValueError(
                "Existing weekly checkpoint identity/source mismatch; do not "
                f"overwrite it silently: {checkpoint_path}"
            )
        return fit, "loaded"
    fit = fit_phase3_training_history(training, config=config)
    save_weekly_dynamics_fit(fit, checkpoint_path, source=source)
    return fit, "fitted"


def _ridge_sensitivity(
    history: WeeklyTrainingHistory,
) -> dict[str, Any]:
    """Evaluate ad-hoc ridge choices without using 2025 observations."""
    fitting = history.subset(PHASE3_TRAIN_START, date(2023, 12, 31))
    scoring = history.subset(date(2024, 1, 1), PHASE3_TRAIN_END)
    variants: dict[str, Any] = {}
    for pca_mode in ("reduced", "full"):
        mode_results: dict[str, Any] = {}
        for ridge in (0.0, 1.0e-6, 1.0e-4):
            config = WeeklyDynamicsConfig(
                label=f"phase3-ridge-sensitivity-{pca_mode}-{ridge:.0e}",
                pca_mode=pca_mode,
                ridge_penalty=ridge,
            )
            try:
                fit = fit_phase3_training_history(fitting, config=config)
                mode_results[f"{ridge:.0e}"] = {
                    "status": "complete",
                    "score_2024": evaluate_one_week_interval(fit, scoring),
                    "reduced_state_size": fit.reduced_state_size,
                    "standardized_transition_maximum_power_norm": fit.diagnostics[
                        "standardized_transition_maximum_power_norm"
                    ],
                    "transition_condition_number": fit.diagnostics[
                        "transition_condition_number"
                    ],
                    "generator_roundtrip_relative_error": fit.diagnostics[
                        "generator_roundtrip_relative_error"
                    ],
                }
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                mode_results[f"{ridge:.0e}"] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        variants[pca_mode] = mode_results
    return {
        "fit_range": [PHASE3_TRAIN_START.isoformat(), "2023-12-31"],
        "score_range": ["2024-01-01", PHASE3_TRAIN_END.isoformat()],
        "selection_policy": (
            "diagnostic only; the DYSANOS/grid ordinary least-squares fit is "
            "retained and "
            "2025 is not used to select it"
        ),
        "variants": variants,
    }


def run_weekly_phase3(
    *,
    source_kind: str,
    source_checkpoint: Path,
    source_report: Path,
    checkpoint_directory: Path,
    report_directory: Path,
    run_simulation: bool,
) -> dict[str, Any]:
    """Fit full/reduced weekly dynamics and score untouched 2025 horizons."""
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    report_path = report_directory / f"phase3-weekly-{source_kind}.json"
    lock_path = checkpoint_directory / f"phase3-weekly-{source_kind}.lock"
    with FileLock(str(lock_path)):
        history, source = _load_history(
            source_kind=source_kind,
            source_checkpoint=source_checkpoint,
            source_report=source_report,
        )
        training = history.subset(PHASE3_TRAIN_START, PHASE3_TRAIN_END)
        validation = history.subset(
            PHASE3_VALIDATION_START,
            PHASE3_VALIDATION_END,
        )
        report: dict[str, Any] = {
            "schema": "flex-phase3-weekly-v2-grid-hierarchy",
            "status": "fitting",
            "source": source,
            "train_range": [
                PHASE3_TRAIN_START.isoformat(),
                PHASE3_TRAIN_END.isoformat(),
            ],
            "validation_range": [
                PHASE3_VALIDATION_START.isoformat(),
                PHASE3_VALIDATION_END.isoformat(),
            ],
            "num_train_dates": len(training.dates),
            "num_validation_dates": len(validation.dates),
            "date_weighting": (
                "one equal exact daily origin for drift and one equal exact "
                "five-session origin for innovations"
            ),
            "research_gate": (
                "Weak forecast fit is retained if finite, stationary, and "
                "non-degenerate; statistics are diagnostics, not production gates."
            ),
            "variants": {},
            "ad_hoc_choices": [
                "per-expiry PCA variance threshold 99.5% for reduced mode",
                "ordinary least squares for the lag-only daily affine transition",
                "training-coordinate standardization before fitting the generator",
                "full real stable matrix logarithm with no fallback",
                "transient-power cap 10 in standardized coordinates",
                "stationary/training variance trace-ratio gate [0.01,100]",
                "Gaussian likelihood for weekly innovations",
            ],
        }
        _write_json_atomic(report, report_path)
        report["ridge_sensitivity"] = _ridge_sensitivity(training)
        report["status"] = "ridge_sensitivity_complete"
        _write_json_atomic(report, report_path)
        variants: dict[str, Any] = {}
        strong_checkpoint: Path | None = None
        if run_simulation:
            if source_kind == "direct-oracle":
                strong_checkpoint = source_checkpoint
            else:
                source_payload = json.loads(source_report.read_text(encoding="utf-8"))
                strong_payload = source_payload.get("strong")
                if not isinstance(strong_payload, dict) or not isinstance(
                    strong_payload.get("checkpoint"), str
                ):
                    raise ValueError("Encoder report has no train-only Strong checkpoint")
                strong_checkpoint = Path(strong_payload["checkpoint"])
                if _sha256(strong_checkpoint) != source["strong_decoder_sha256"]:
                    raise ValueError("Encoder Strong decoder checkpoint digest mismatch")
        for pca_mode in ("reduced", "full"):
            config = WeeklyDynamicsConfig(
                label=f"phase3-weekly-{source_kind}-{pca_mode}",
                pca_mode=pca_mode,
            )
            checkpoint_path = checkpoint_directory / (
                f"phase3-weekly-{source_kind}-{pca_mode}.pt"
            )
            fit, action = _fit_or_load(
                training,
                config=config,
                checkpoint_path=checkpoint_path,
                source=source,
            )
            validation_result = evaluate_oos_dynamics(
                fit,
                validation,
                training,
            )
            if not (
                validation_result["all_horizons_finite"]
                and validation_result["all_horizons_nondegenerate"]
            ):
                report["status"] = f"failed_{pca_mode}_validation_gate"
                _write_json_atomic(report, report_path)
                raise RuntimeError(
                    f"Phase 3 {pca_mode} validation is non-finite or collapsed"
                )
            variants[pca_mode] = {
                "config": asdict(config),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "checkpoint_action": action,
                "diagnostics": fit.diagnostics,
                "validation_2025": validation_result,
            }
            if run_simulation:
                assert strong_checkpoint is not None
                simulation = run_simulation_diagnostics(
                    fit,
                    validation,
                    strong_checkpoint=strong_checkpoint,
                )
                variants[pca_mode]["simulation_validation"] = simulation
                if simulation["nondegenerate"] is not True:
                    report["variants"] = variants
                    report["status"] = f"failed_{pca_mode}_simulation_gate"
                    _write_json_atomic(report, report_path)
                    raise RuntimeError(
                        f"Phase 3 {pca_mode} real-checkpoint simulation collapsed"
                    )
            report["variants"] = variants
            report["status"] = f"{pca_mode}_complete"
            _write_json_atomic(report, report_path)
            print(
                f"phase3 weekly {pca_mode} d={fit.reduced_state_size} "
                f"rho={fit.diagnostics['maximum_transition_spectral_radius']:.6g} "
                f"oos={validation_result['all_horizons_nondegenerate']}",
                flush=True,
            )
        report["production_complete"] = run_simulation
        report["status"] = (
            "complete"
            if run_simulation
            else "statistical_validation_complete_simulation_not_run"
        )
        _write_json_atomic(report, report_path)
        return report


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-kind",
        choices=("encoder", "direct-oracle"),
        default="encoder",
    )
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=_DEFAULT_CHECKPOINT_DIRECTORY,
    )
    parser.add_argument(
        "--report-directory",
        type=Path,
        default=_DEFAULT_REPORT_DIRECTORY,
    )
    parser.add_argument(
        "--run-simulation",
        action="store_true",
        help="Run real-decoder normal/implied and option-return checks on GPU 0.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    if arguments.source_checkpoint is None:
        source_checkpoint = (
            _DEFAULT_ENCODER_CHECKPOINT
            if arguments.source_kind == "encoder"
            else _DEFAULT_DIRECT_CHECKPOINT
        )
    else:
        source_checkpoint = arguments.source_checkpoint
    if arguments.source_report is None:
        source_report = (
            _DEFAULT_ENCODER_REPORT
            if arguments.source_kind == "encoder"
            else _DEFAULT_DIRECT_REPORT
        )
    else:
        source_report = arguments.source_report
    run_weekly_phase3(
        source_kind=arguments.source_kind,
        source_checkpoint=source_checkpoint,
        source_report=source_report,
        checkpoint_directory=arguments.checkpoint_directory,
        report_directory=arguments.report_directory,
        run_simulation=arguments.run_simulation,
    )


if __name__ == "__main__":
    run_guarded_cli(main, program="run_weekly_phase3")
