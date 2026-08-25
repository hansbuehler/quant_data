"""Compare short train-only direct-200 decoder learning-rate trajectories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from cdxcore.util import is_filename
from filelock import FileLock

from quantdata.dysanos.flex.config import (
    FlexMarketConfig,
    FlexObjectiveConfig,
    FlexTrainConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.run_shared_strikes_compressed_encoders import (
    _candidate_config,
    _native_training_target,
    _wait_for_completion,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _load_markets,
    _load_native_targets,
    _market_dates_from_completion,
    _split_markets,
)
from quantdata.dysanos.flex.training import (
    TrainingConvergenceError,
    fit_latent_states,
)
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class ProbeCandidate:
    """One peak rate and schedule starting from the same analytic PCA state."""

    label: str
    learning_rate: float
    schedule: Literal["constant", "cosine"]
    warmup_epochs: int
    minimum_fraction: float


def _parse_candidate(value: str) -> ProbeCandidate:
    fields = value.split(",")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "Expected LABEL,PEAK_LR,SCHEDULE,WARMUP_EPOCHS,MINIMUM_FRACTION"
        )
    label, raw_rate, schedule, raw_warmup, raw_floor = fields
    if not is_filename(label):
        raise argparse.ArgumentTypeError(f"Invalid candidate label {label!r}")
    if schedule not in {"constant", "cosine"}:
        raise argparse.ArgumentTypeError(f"Invalid schedule {schedule!r}")
    try:
        learning_rate = float(raw_rate)
        warmup_epochs = int(raw_warmup)
        minimum_fraction = float(raw_floor)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return ProbeCandidate(
        label=label,
        learning_rate=learning_rate,
        schedule=cast(Literal["constant", "cosine"], schedule),
        warmup_epochs=warmup_epochs,
        minimum_fraction=minimum_fraction,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=_parse_candidate,
    )
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--capacity-completion", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _checkpoint_history(path: Path) -> tuple[dict[str, object], list[dict[str, float]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_training_state = payload["training_state"]
    if raw_training_state is None:
        training_state: dict[str, object] = {"phase": payload["phase"]}
        history = cast(list[dict[str, float]], payload["market_history"])
    else:
        training_state = cast(dict[str, object], raw_training_state)
        history = cast(list[dict[str, float]], training_state["history"])
    return training_state, history


def _history_summary(
    *,
    training_state: dict[str, object],
    history: list[dict[str, float]],
) -> dict[str, object]:
    if not history:
        raise RuntimeError("A decoder LR probe produced no history")
    best = min(history, key=lambda item: item["objective"])
    first = history[0]
    last = history[-1]
    return {
        "pre_training_score": training_state["pre_training_score"],
        "num_epochs": len(history),
        "first": first,
        "best": best,
        "last": last,
        "maximum_dlvsq_cap_penalty": max(
            item["dlvsq_cap_penalty"] for item in history
        ),
    }


def run(arguments: argparse.Namespace) -> None:
    """Run every candidate serially on GPU 0 and persist each trajectory."""
    if arguments.ticker != "SPX":
        raise ValueError("Shared-strikes LR probes are specified for SPX")
    if arguments.device != "cuda:0":
        raise ValueError("Shared-strikes LR probes are restricted to GPU 0")
    if not 1 <= arguments.workers <= 4:
        raise ValueError("Shared-strikes LR probes require one to four workers")
    if arguments.epochs < 2:
        raise ValueError("Shared-strikes LR probes require at least two epochs")
    _wait_for_completion(arguments.capacity_completion, wait_seconds=30.0)
    native_proof = json.loads(arguments.native_completion.read_text(encoding="utf-8"))
    if native_proof.get("status") != "native_complete":
        raise RuntimeError("Native completion proof is not complete")
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    dates = _market_dates_from_completion(arguments.native_completion)
    days = _load_markets(
        ticker=arguments.ticker,
        dates=dates,
        geometry=geometry,
        market_config=FlexMarketConfig(),
        workers=arguments.workers,
    )
    train_days, _ = _split_markets(days)
    native_raw, _ = _load_native_targets(
        arguments.native_checkpoint,
        days,
        geometry=geometry,
    )
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    objective_config = FlexObjectiveConfig(fit_weight=0.05)
    native_fit = _native_training_target(
        train_days,
        native_raw[: len(train_days)],
        geometry=geometry,
        objective_config=objective_config,
        env=env,
    )
    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    with FileLock(str(arguments.checkpoint_directory / "decoder-lr-probe.lock")):
        for candidate in arguments.candidate:
            print(f"starting decoder LR probe {candidate.label}", flush=True)
            checkpoint = arguments.checkpoint_directory / f"{candidate.label}.pt"
            if checkpoint.exists():
                raise FileExistsError(
                    f"Probe checkpoint already exists; use a new run directory: {checkpoint}"
                )
            train_config = FlexTrainConfig(
                num_epochs=arguments.epochs,
                day_batch_size=16,
                learning_rate=candidate.learning_rate,
                learning_rate_schedule=candidate.schedule,
                learning_rate_warmup_epochs=candidate.warmup_epochs,
                minimum_learning_rate_fraction=candidate.minimum_fraction,
                precision=32,
                device=arguments.device,
                compile_sanos_solver=True,
                checkpoint_every=5,
                report_every=5,
                convergence_window=min(10, arguments.epochs - 1),
                convergence_relative_tolerance=0.99,
            )
            error: str | None = None
            final_evaluation: dict[str, object] | None = None
            try:
                fit = fit_latent_states(
                    train_days,
                    native_fit,
                    config=_candidate_config("direct", geometry=geometry),
                    objective_config=objective_config,
                    supervised_train_config=FlexTrainConfig(
                        num_epochs=1,
                        day_batch_size=128,
                        learning_rate=2.0e-3,
                        precision=32,
                        device=arguments.device,
                    ),
                    market_train_config=train_config,
                    checkpoint_path=checkpoint,
                )
                final_evaluation = fit.evaluation.to_dict()
                pre_training_score: object = fit.initial_evaluation.to_dict()
            except TrainingConvergenceError as exception:
                error = str(exception)
                pre_training_score = None
            training_state, history = _checkpoint_history(checkpoint)
            if pre_training_score is not None:
                training_state["pre_training_score"] = pre_training_score
            reports.append(
                {
                    "candidate": asdict(candidate),
                    "train_config": asdict(train_config),
                    "status": "complete" if error is None else "rejected",
                    "error": error,
                    "history": history,
                    "summary": _history_summary(
                        training_state=training_state,
                        history=history,
                    ),
                    "final_evaluation": final_evaluation,
                    "checkpoint": str(checkpoint),
                }
            )
            write_json_atomic(
                {
                    "schema": "flex-shared-decoder-lr-probe-v1",
                    "status": "running",
                    "epochs": arguments.epochs,
                    "reports": reports,
                },
                arguments.output,
            )
    write_json_atomic(
        {
            "schema": "flex-shared-decoder-lr-probe-v1",
            "status": "complete",
            "epochs": arguments.epochs,
            "reports": reports,
        },
        arguments.output,
    )


def main() -> None:
    """CLI entry point."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_shared_strikes_decoder_lr_probe")
