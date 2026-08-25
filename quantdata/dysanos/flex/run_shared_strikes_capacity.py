"""Run flat-200 then expiry-causal-200 compression for shared strikes."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from quantdata.biz_calendar import get_business_days
from quantdata.dysanos.flex.config import (
    FlexMarketConfig,
    FlexObjectiveConfig,
    FlexTrainConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.run_production_gate import run_production_gates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--start", default="2022-01-03")
    parser.add_argument("--end", default="2025-08-29")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--supervised-epochs", type=int, default=300)
    parser.add_argument("--day-batch-size", type=int, default=16)
    parser.add_argument("--supervised-day-batch-size", type=int, default=128)
    # The analytic PCA decoder is already a strong market-price initializer.
    # Fine-tuning therefore uses a conservative rate: 2e-2 destroyed that
    # initialization before the full-surface regression guard could reject it.
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--supervised-learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--fit-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    parser.add_argument(
        "--prerequisite-completion",
        type=Path,
        help="Wait for this upstream completion proof before starting compression.",
    )
    parser.add_argument("--label", default="spx-2022-2025-shared-strikes-v1")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    return parser.parse_args()


def _wait_for_native_completion(
    *,
    output_directory: Path,
    label: str,
    wait_seconds: float,
) -> None:
    completion_path = output_directory / f"{label}-native-complete.json"
    report_path = output_directory / f"{label}.json"
    while not completion_path.exists():
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            status = str(report.get("status", "unknown"))
            if status.endswith("_failed"):
                raise RuntimeError(
                    "Native shared-strikes calibration failed before "
                    f"compression: {report.get('failure', status)}"
                )
            print(
                f"waiting for shared native completion status={status}",
                flush=True,
            )
        else:
            print("waiting for shared native report", flush=True)
        time.sleep(wait_seconds)


def _wait_for_prerequisite(path: Path | None, *, wait_seconds: float) -> None:
    """Wait for an explicit upstream validation gate when one is configured."""
    if path is None:
        return
    while not path.exists():
        print(f"waiting for prerequisite completion {path}", flush=True)
        time.sleep(wait_seconds)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(
            f"Prerequisite completion has unexpected status: {payload.get('status')!r}"
        )


def _candidate_summary(report: dict[str, object]) -> dict[str, object]:
    """Retain the fit, convergence, and checkpoint proof for one candidate."""
    latent = report.get("latent")
    if not isinstance(latent, list) or len(latent) != 1:
        raise RuntimeError("Expected exactly one completed 200-state candidate")
    candidate = latent[0]
    if not isinstance(candidate, dict):
        raise RuntimeError("Invalid completed 200-state candidate report")
    retained = (
        "variant",
        "state_size",
        "state_sizes_by_expiry",
        "statistics",
        "surface_fit",
        "objective",
        "initial_evaluation",
        "elapsed_seconds",
        "supervised_convergence",
        "market_convergence",
        "checkpoint_proof",
    )
    return {key: candidate.get(key) for key in retained}


def main() -> None:
    """Wait for certified native states, then run both 200-state decoders."""
    args = _parse_args()
    if args.device not in {"cuda:0", "cpu"}:
        raise ValueError("Shared capacity training is restricted to cuda:0 or CPU")
    if args.wait_seconds <= 0.0 or args.wait_seconds > 60.0:
        raise ValueError("'wait_seconds' must lie in (0,60]")
    dates = tuple(
        get_business_days(
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
        )
    )
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    market_config = FlexMarketConfig()
    objective_config = FlexObjectiveConfig(fit_weight=args.fit_weight)
    market_train_config = FlexTrainConfig(
        num_epochs=args.epochs,
        day_batch_size=args.day_batch_size,
        learning_rate=args.learning_rate,
        precision=args.precision,
        device=args.device,
        compile_sanos_solver=True,
    )
    supervised_train_config = FlexTrainConfig(
        num_epochs=args.supervised_epochs,
        day_batch_size=args.supervised_day_batch_size,
        learning_rate=args.supervised_learning_rate,
        precision=args.precision,
        device=args.device,
    )
    _wait_for_prerequisite(
        args.prerequisite_completion,
        wait_seconds=args.wait_seconds,
    )
    _wait_for_native_completion(
        output_directory=args.output_directory,
        label=args.label,
        wait_seconds=args.wait_seconds,
    )

    print("starting shared-strikes flat-200 compression", flush=True)
    direct_report = run_production_gates(
        ticker=args.ticker,
        dates=dates,
        geometry=geometry,
        market_config=market_config,
        objective_config=objective_config,
        train_config=market_train_config,
        supervised_train_config=supervised_train_config,
        workers=args.workers,
        output_directory=args.output_directory,
        checkpoint_directory=args.checkpoint_directory,
        label=args.label,
        state_sizes=(200,),
        run_compression=True,
        pca_decoder=True,
    )

    print("starting shared-strikes hierarchical-200 compression", flush=True)
    hierarchical_report = run_production_gates(
        ticker=args.ticker,
        dates=dates,
        geometry=geometry,
        market_config=market_config,
        objective_config=objective_config,
        train_config=market_train_config,
        supervised_train_config=supervised_train_config,
        workers=args.workers,
        output_directory=args.output_directory,
        checkpoint_directory=args.checkpoint_directory,
        label=args.label,
        state_sizes=(200,),
        run_compression=True,
        expiry_benchmark=True,
        expiry_benchmark_variants=("triangular-strong-200",),
    )
    comparison_path = args.output_directory / (
        f"{args.label}-shared-capacity-200-comparison.json"
    )
    write_json_atomic(
        {
            "schema": "flex-shared-capacity-comparison-v1",
            "status": "complete",
            "label": args.label,
            "native_completion": str(
                args.output_directory / f"{args.label}-native-complete.json"
            ),
            "prerequisite_completion": (
                None
                if args.prerequisite_completion is None
                else str(args.prerequisite_completion)
            ),
            "direct_200": _candidate_summary(direct_report),
            "hierarchical_200": _candidate_summary(hierarchical_report),
        },
        comparison_path,
    )
    completion_path = args.output_directory / (
        f"{args.label}-shared-capacity-200-complete.json"
    )
    write_json_atomic(
        {
            "schema": "flex-shared-capacity-completion-v1",
            "status": "complete",
            "comparison_report": str(comparison_path),
        },
        completion_path,
    )


if __name__ == "__main__":
    run_guarded_cli(main, program="flex-shared-strikes-capacity")
