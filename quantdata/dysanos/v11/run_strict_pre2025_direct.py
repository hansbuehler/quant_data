"""Fit the frozen pre-2025 one-day-front Direct-200 DYSANOS representation.

The source native rows are per-date SANOS calibrations: their only learned
parameter is the row for that date.  They are therefore safe as a target
source, but the affine decoder and every transition parameter below are fit
only through 2024-12-31.  The resulting checkpoint is the common frozen
representation for the weekly/daily and Gaussian/Tail-Diffusion factorial
comparison.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from typing import cast

import torch
from filelock import FileLock

from quantdata.dysanos.flex.config import (
    FlexObjectiveConfig,
    JointDirectDynamicsConfig,
    NativeStateConfig,
    PcaDecoderConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.joint_direct import (
    JointDirectTrainConfig,
    _convergence_audit,
    compact_evaluation,
    fit_joint_direct_model,
    initialize_joint_model,
    standardize_affine_direct_model,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import JointDirectDynamicsModel, NativeStateSanosModel
from quantdata.dysanos.flex.run_all_history_joint_direct import (
    _geometry_from_completion,
    _market_config_from_completion,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _load_markets,
    _load_native_targets,
    _market_dates_from_completion,
)
from quantdata.dysanos.flex.training import (
    FlexTrainConfig,
    NativeStateFitResult,
    TrainingConvergenceError,
    evaluate_model,
    fit_latent_states,
    materialize_day,
)
from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs
from quantdata.torch import BaseEnvironment

_TRAIN_END = date(2024, 12, 31)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--native-completion", required=True, type=Path)
    parser.add_argument("--native-checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-directory", required=True, type=Path)
    parser.add_argument("--report-directory", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decoder-epochs", type=int, default=600)
    parser.add_argument("--decoder-warm-start", type=Path)
    parser.add_argument("--state-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--network-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--joint-epochs", type=int, default=300)
    parser.add_argument("--accept-joint-epoch-cap", action="store_true")
    return parser.parse_args()


def _train_count(dates: tuple[date, ...]) -> int:
    count = sum(value <= _TRAIN_END for value in dates)
    if not 0 < count < len(dates) or any(value <= _TRAIN_END for value in dates[count:]):
        raise ValueError("dates must have a non-empty contiguous 2025 validation suffix")
    return count


def _gpu_weights_only_checkpoint(source: Path, *, device: str) -> Path:
    """Promote a CPU checkpoint to a fresh GPU optimizer run once.

    Adam's device-specific resume signature intentionally rejects CPU moments
    under CUDA.  The affine decoder parameters remain valid, so retain them
    but mark a sibling checkpoint as supervised-complete: market fitting then
    starts a new GPU optimizer and scheduler from those weights.
    """
    if not device.startswith("cuda") or not source.exists():
        return source
    target = source.with_name(f"{source.stem}-gpu{source.suffix}")
    if target.exists():
        return target
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("phase") not in {"market", "complete"}:
        raise ValueError("CPU decoder checkpoint is not suitable for GPU warm start")
    payload["phase"] = "supervised_complete"
    payload["training_state"] = None
    payload["market_history"] = []
    payload["gpu_weights_only_source"] = str(source)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def _continuation_checkpoint(source: Path, target: Path) -> Path:
    """Retain fitted weights while starting a new market optimizer schedule."""
    if not source.exists():
        raise FileNotFoundError(source)
    source_digest = checkpoint_digest(source)
    if target.exists():
        existing = torch.load(target, map_location="cpu", weights_only=False)
        if existing.get("phase") == "complete":
            # The final trainer checkpoint replaces the provisional payload and
            # no longer retains the warm-start source hash.  Its own completed
            # audits and market identity are checked again by fit_latent_states.
            audits = existing.get("training_audits")
            if not isinstance(audits, dict) or not all(
                isinstance(audits.get(name), dict)
                and audits[name].get("converged") is True
                for name in ("supervised", "market")
            ):
                raise ValueError("Existing complete decoder lacks passing audits")
            return target
        if existing.get("weights_only_source_sha256") != source_digest:
            raise ValueError("Existing decoder continuation has another source")
        return target
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("phase") not in {"market", "complete"}:
        raise ValueError("Decoder continuation requires fitted market weights")
    state = payload.get("training_state")
    signature = state.get("resume_signature") if isinstance(state, dict) else None
    if payload.get("phase") == "market" and not (
        isinstance(state, dict)
        and isinstance(signature, dict)
        and int(state.get("completed_epochs", -1))
        == int(signature.get("num_epochs", -2))
    ):
        raise ValueError("Decoder continuation requires a completed prior schedule")
    payload["phase"] = "supervised_complete"
    payload["training_state"] = None
    payload["market_history"] = []
    audits = payload.get("training_audits")
    if audits is None:
        audits = {}
    elif not isinstance(audits, dict):
        raise ValueError("Decoder checkpoint training_audits must be a dictionary or None")
    payload["training_audits"] = {
        name: audit
        for name, audit in audits.items()
        if name == "supervised"
    }
    payload["weights_only_source"] = str(source)
    payload["weights_only_source_sha256"] = source_digest
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def _decoder_train_config_for_checkpoint(
    config: FlexTrainConfig, checkpoint: Path
) -> FlexTrainConfig:
    """Explicitly load a completed decoder during a joint-only restart."""
    if not checkpoint.exists():
        return config
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return replace(config, just_load=True) if payload.get("phase") == "complete" else config


def _capped_joint_checkpoint(
    checkpoint_path: Path,
    *,
    train_config: JointDirectTrainConfig,
    model_config: JointDirectDynamicsConfig,
    market_fingerprints: tuple[str, ...],
    source_decoder_sha256: str,
    error: TrainingConvergenceError,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], dict[str, object]]:
    """Accept only a fully saved epoch-cap checkpoint, without calling it converged."""
    if not str(error).startswith("Joint Direct training failed convergence:"):
        raise error
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "flex-joint-direct-dynamics-v1"
        or payload.get("phase") != "training"
        or payload.get("model_config") != asdict(model_config)
        or payload.get("model_config_version") != model_config.version.unique_id48
        or payload.get("model_user_version") != model_config.user_version
        or payload.get("train_config") != asdict(train_config)
        or tuple(payload.get("market_fingerprints", ())) != market_fingerprints
        or payload.get("source_decoder_sha256") != source_decoder_sha256
        or payload.get("completed_epochs") != train_config.num_epochs
    ):
        raise ValueError("Capped joint checkpoint does not match this completed run") from error
    history = payload.get("history")
    state_dict = payload.get("model_state_dict")
    if (
        not isinstance(history, list)
        or len(history) != train_config.num_epochs
        or not isinstance(state_dict, dict)
    ):
        raise ValueError("Capped joint checkpoint has incomplete history or weights") from error
    convergence = _convergence_audit(history, config=train_config)
    if convergence.get("reason") != "trailing_window_not_converged":
        raise ValueError("Capped joint checkpoint failed for another reason") from error
    return state_dict, history, {**convergence, "accepted_at_epoch_cap": True}


def run(
    *,
    ticker: str,
    native_completion: Path,
    native_checkpoint: Path,
    checkpoint_directory: Path,
    report_directory: Path,
    workers: int,
    device: str,
    decoder_epochs: int,
    decoder_warm_start: Path | None,
    state_learning_rate: float,
    network_learning_rate: float,
    joint_epochs: int,
    accept_joint_epoch_cap: bool = False,
) -> dict[str, object]:
    """Fit the reusable strict representation and its weekly joint baseline."""
    completion = json.loads(native_completion.read_text(encoding="utf-8"))
    geometry = _geometry_from_completion(completion, requested_model_business_dtes=None)
    market_config = _market_config_from_completion(completion)
    dates = _market_dates_from_completion(native_completion)
    train_count = _train_count(dates)
    all_days = _load_markets(
        ticker=ticker,
        dates=dates,
        geometry=geometry,
        market_config=market_config,
        workers=workers,
    )
    train_days = all_days[:train_count]
    native_raw, source = _load_native_targets(
        native_checkpoint, all_days, geometry=geometry
    )
    env = BaseEnvironment(precision=32, device=device, debug=False)
    native_model = NativeStateSanosModel(
        env,
        num_days=train_count,
        config=NativeStateConfig(label="v11-pre2025-native-target", geometry=geometry),
    )
    with torch.no_grad():
        native_model.daily_raw_states.copy_(native_raw[:train_count].to(env.device, env.fdtype))
    torch_days = tuple(materialize_day(day, env=env) for day in train_days)
    native_fit = NativeStateFitResult(
        model=native_model,
        evaluation=evaluate_model(
            native_model,
            torch_days,
            env=env,
            objective_config=FlexObjectiveConfig(fit_weight=0.05),
            day_batch_size=16,
            equal_date_weighting=True,
        ),
        history=(),
        elapsed_seconds=0.0,
        market_fingerprints=tuple(day.training_fingerprint for day in train_days),
        state_dtype=str(env.fdtype),
        convergence={"converged": True, "source": "independent_per_date_native_rows"},
    )
    decoder_config = PcaDecoderConfig(
        label="v11-pre2025-dte1-direct-200",
        state_size=200,
        group="dysanos/v11/pre2025-factorial",
        geometry=geometry,
    )
    decoder_train = FlexTrainConfig(
        num_epochs=decoder_epochs,
        day_batch_size=16,
        learning_rate=state_learning_rate,
        decoder_learning_rate=network_learning_rate,
        learning_rate_schedule="cosine",
        minimum_learning_rate_fraction=0.1,
        device=device,
        precision=32,
        checkpoint_every=5,
        # CPU fitting can take several minutes per epoch.  Persisted stdout is
        # therefore also the required liveness signal for a detached job.
        report_every=1,
        compile_sanos_solver=True,
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    decoder_checkpoint = (
        _gpu_weights_only_checkpoint(
            checkpoint_directory / "direct-200-decoder.pt",
            device=device,
        )
        if decoder_warm_start is None
        else _continuation_checkpoint(
            decoder_warm_start,
            checkpoint_directory / "direct-200-decoder-continuation.pt",
        )
    )
    with FileLock(str(decoder_checkpoint.with_suffix(".lock"))):
        decoder_fit = fit_latent_states(
            train_days,
            native_fit,
            config=decoder_config,
            objective_config=FlexObjectiveConfig(fit_weight=0.05),
            supervised_train_config=decoder_train,
            market_train_config=_decoder_train_config_for_checkpoint(
                decoder_train, decoder_checkpoint
            ),
            checkpoint_path=decoder_checkpoint,
        )
    standardized = standardize_affine_direct_model(decoder_fit.model)
    model_config = JointDirectDynamicsConfig(
        label="v11-pre2025-dte1-weekly-joint-direct-200",
        group="dysanos/v11/pre2025-factorial",
        state_size=200,
        geometry=geometry,
    )
    joint_model = JointDirectDynamicsModel(env, num_days=train_count, config=model_config)
    origin, endpoint = exact_weekly_pairs(dates[:train_count], weeks=1)
    initialization = initialize_joint_model(
        joint_model,
        standardized,
        origin_indices=origin,
        endpoint_indices=endpoint,
    )
    joint_config = JointDirectTrainConfig(
        num_epochs=joint_epochs,
        day_batch_size=16,
        state_learning_rate=3.0e-5,
        decoder_learning_rate=1.0e-5,
        dynamics_learning_rate=1.0e-4,
        warmup_epochs=20,
        dynamics_ramp_epochs=50,
        report_every=1,
        checkpoint_every=5,
    )
    joint_checkpoint = checkpoint_directory / "joint-direct-200.pt"
    source_decoder_sha256 = checkpoint_digest(decoder_checkpoint)
    with FileLock(str(joint_checkpoint.with_suffix(".lock"))):
        try:
            history, convergence = fit_joint_direct_model(
                joint_model,
                train_days,
                dates=dates[:train_count],
                objective_config=FlexObjectiveConfig(fit_weight=0.05),
                train_config=joint_config,
                checkpoint_path=joint_checkpoint,
                source_decoder_sha256=source_decoder_sha256,
                env=env,
            )
        except TrainingConvergenceError as error:
            if not accept_joint_epoch_cap:
                raise
            weights, history, convergence = _capped_joint_checkpoint(
                joint_checkpoint,
                train_config=joint_config,
                model_config=model_config,
                market_fingerprints=tuple(day.training_fingerprint for day in train_days),
                source_decoder_sha256=source_decoder_sha256,
                error=error,
            )
            joint_model.load_state_dict(weights)
            print(
                f"joint-direct accepted at {joint_epochs}-epoch cap without convergence",
                flush=True,
            )
    targets = checkpoint_directory / "joint-direct-200-targets.pt"
    torch.save(
        {
            "schema": "dysanos-v11-pre2025-joint-direct-targets-v1",
            "dates": [value.isoformat() for value in dates[:train_count]],
            "states": joint_model.daily_states.detach().cpu().double(),
            "spot_refs": torch.as_tensor([cast(float, day.spot_ref) for day in train_days]),
            "native_source": source,
        },
        targets,
    )
    report: dict[str, object] = {
        "schema": "dysanos-v11-pre2025-direct-v1",
        "status": "complete" if convergence.get("converged") is True else "capped",
        "training_semantics": "Decoder and weekly joint dynamics fitted through 2024-12-31 only; 2025 is untouched.",
        "train_range": [dates[0].isoformat(), dates[train_count - 1].isoformat()],
        "validation_range": [dates[train_count].isoformat(), dates[-1].isoformat()],
        "geometry": asdict(geometry),
        "native_source": source,
        "decoder_fit": decoder_fit.to_dict(),
        "joint_initialization": initialization,
        "joint_convergence": convergence,
        "joint_history_tail": history[-50:],
        "joint_fit": compact_evaluation(evaluate_model(joint_model, torch_days, env=env, objective_config=FlexObjectiveConfig(fit_weight=0.05), day_batch_size=16, equal_date_weighting=True)),
        "decoder_checkpoint": str(decoder_checkpoint),
        "decoder_warm_start": (
            None if decoder_warm_start is None else str(decoder_warm_start)
        ),
        "joint_checkpoint": str(joint_checkpoint),
        "target_history": str(targets),
    }
    write_json_atomic(report, report_directory / "pre2025-direct-200.json")
    return report


def main() -> None:
    arguments = _arguments()
    report = run(**vars(arguments))
    print(
        f"strict pre-2025 Direct fit {report['status']}: {report['joint_checkpoint']}",
        flush=True,
    )


if __name__ == "__main__":
    run_guarded_cli(main, program="run_strict_pre2025_direct")
