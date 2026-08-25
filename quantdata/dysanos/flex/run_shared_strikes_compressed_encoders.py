"""Fit clean direct/hierarchical 200-state decoders and frozen-2025 encoders."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

import torch
from filelock import FileLock

from quantdata.dysanos.flex.config import (
    FlexMarketConfig,
    FlexObjectiveConfig,
    FlexTrainConfig,
    NativeStateConfig,
    PcaDecoderConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    TriangularPcaDecoderConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market import FlexMarketDay
from quantdata.dysanos.flex.market_encoder import (
    EncoderTrainConfig,
    NativeRbfQuoteEncoder,
    NativeRbfQuoteEncoderConfig,
    checkpoint_digest,
    fit_rbf_feature_statistics,
    transform_market_days,
    write_json_atomic,
)
from quantdata.dysanos.flex.model import NativeStateSanosModel
from quantdata.dysanos.flex.run_phase3_native_encoder import (
    _evaluate,
    _save_checkpoint,
    _sha256_tensor,
    _train_phase,
)
from quantdata.dysanos.flex.run_production_gate import _latent_restart_config
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _load_markets,
    _load_native_targets,
    _market_dates_from_completion,
    _restore_encoder_checkpoint,
    _split_markets,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.dysanos.flex.training import (
    NativeStateFitResult,
    evaluate_model,
    fit_latent_states,
    materialize_day,
    model_checkpoint_phase,
)
from quantdata.torch import BaseEnvironment

_HIERARCHICAL_200_ALLOCATION = (
    23,
    27,
    25,
    22,
    19,
    14,
    14,
    14,
    7,
    7,
    7,
    7,
    7,
    7,
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    parser.add_argument("--decoder-supervised-epochs", type=int, default=300)
    parser.add_argument("--decoder-market-epochs", type=int, default=1_000)
    parser.add_argument("--decoder-market-learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--decoder-learning-rate-schedule",
        choices=("constant", "cosine"),
        default="cosine",
    )
    parser.add_argument("--decoder-learning-rate-warmup-epochs", type=int, default=10)
    parser.add_argument(
        "--decoder-minimum-learning-rate-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--recover-incomplete-decoder",
        action="store_true",
        help=(
            "warm-start an existing incomplete market-phase decoder at the "
            "explicit recovery learning rate instead of replaying its optimizer"
        ),
    )
    parser.add_argument("--decoder-recovery-epochs", type=int, default=200)
    parser.add_argument(
        "--decoder-recovery-learning-rate",
        type=float,
        default=3.0e-5,
    )
    parser.add_argument("--encoder-supervised-epochs", type=int, default=1_200)
    parser.add_argument("--encoder-market-epochs", type=int, default=600)
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--capacity-completion", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    return parser.parse_args()


def _wait_for_completion(path: Path, *, wait_seconds: float) -> dict[str, object]:
    while not path.exists():
        print(f"waiting for completion proof {path}", flush=True)
        time.sleep(wait_seconds)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(
            f"Completion proof has unexpected status {payload.get('status')!r}: {path}"
        )
    return payload


def _native_training_target(
    train_days: tuple[FlexMarketDay, ...],
    train_raw: torch.Tensor,
    *,
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    objective_config: FlexObjectiveConfig,
    env: BaseEnvironment,
) -> NativeStateFitResult:
    """Represent certified independent train rows as a converged native target."""
    model = NativeStateSanosModel(
        env,
        num_days=len(train_days),
        config=NativeStateConfig(
            label="shared-strikes-train-through-2024-native-target",
            geometry=geometry,
        ),
    )
    with torch.no_grad():
        model.daily_raw_states.copy_(
            train_raw.to(device=env.device, dtype=env.fdtype)
        )
    torch_days = tuple(materialize_day(day, env=env) for day in train_days)
    evaluation = evaluate_model(
        model,
        torch_days,
        env=env,
        objective_config=objective_config,
        day_batch_size=16,
    )
    return NativeStateFitResult(
        model=model,
        evaluation=evaluation,
        history=(),
        elapsed_seconds=0.0,
        market_fingerprints=tuple(day.training_fingerprint for day in train_days),
        state_dtype=str(env.fdtype),
        convergence={
            "converged": True,
            "reason": "certified independent per-date native target",
        },
    )


def _decoder_market_train_config(
    checkpoint_path: Path,
    *,
    num_epochs: int,
    learning_rate: float,
    learning_rate_schedule: Literal["constant", "cosine"],
    learning_rate_warmup_epochs: int,
    minimum_learning_rate_fraction: float,
    recover_incomplete: bool,
    recovery_epochs: int,
    recovery_learning_rate: float,
    device: str,
) -> FlexTrainConfig:
    """Select exact resume, completed load, or explicit low-LR recovery."""
    incomplete_market_checkpoint = (
        checkpoint_path.exists()
        and model_checkpoint_phase(checkpoint_path) == "market"
    )
    if recover_incomplete and incomplete_market_checkpoint:
        print(
            "recovering incomplete decoder checkpoint with a weights-only "
            f"warm start: epochs={recovery_epochs} lr={recovery_learning_rate:.8g}",
            flush=True,
        )
        # A completed but unconverged schedule cannot be continued exactly with
        # more epochs because the scheduler horizon is part of the serialized
        # continuation signature. Preserve its learned weights, deliberately
        # reset optimizer state, and test a short plateau at the former LR floor.
        return FlexTrainConfig(
            num_epochs=recovery_epochs,
            day_batch_size=16,
            learning_rate=recovery_learning_rate,
            learning_rate_schedule="constant",
            learning_rate_warmup_epochs=0,
            minimum_learning_rate_fraction=1.0,
            warm_start=True,
            device=device,
            precision=32,
            compile_sanos_solver=True,
        )
    return _latent_restart_config(
        FlexTrainConfig(
            num_epochs=num_epochs,
            day_batch_size=16,
            learning_rate=learning_rate,
            learning_rate_schedule=learning_rate_schedule,
            learning_rate_warmup_epochs=learning_rate_warmup_epochs,
            minimum_learning_rate_fraction=minimum_learning_rate_fraction,
            device=device,
            precision=32,
            compile_sanos_solver=True,
        ),
        checkpoint_path,
    )


def _candidate_config(
    kind: str,
    *,
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
) -> PcaDecoderConfig | TriangularPcaDecoderConfig:
    if kind == "direct":
        return PcaDecoderConfig(
            label="shared-strikes-train-through-2024-direct-200",
            state_size=200,
            geometry=geometry,
        )
    if kind == "hierarchical":
        return TriangularPcaDecoderConfig(
            label="shared-strikes-train-through-2024-hierarchical-200",
            state_size=200,
            state_sizes_by_expiry=_HIERARCHICAL_200_ALLOCATION,
            geometry=geometry,
        )
    raise ValueError(f"Unknown compressed decoder kind {kind!r}")


def _effective_decoder(
    model: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_decoder = getattr(model, "raw_decoder")
    linear = raw_decoder.I0
    weight = linear.weight.detach()
    mask = getattr(model, "decoder_weight_mask", None)
    if isinstance(mask, torch.Tensor):
        weight = weight * mask
    bias = cast(torch.Tensor, linear.bias).detach()
    return weight, bias


def _project_validation_states(
    model: torch.nn.Module,
    validation_raw: torch.Tensor,
) -> torch.Tensor:
    weight, bias = _effective_decoder(model)
    weight_cpu = weight.cpu().double()
    centered = validation_raw.cpu().double() - bias.cpu().double()
    return torch.linalg.lstsq(
        weight_cpu,
        centered.transpose(0, 1),
    ).solution.transpose(0, 1)


def _install_frozen_decoder(
    encoder: NativeRbfQuoteEncoder,
    decoder_model: torch.nn.Module,
) -> None:
    if encoder.native_decoder is None:
        raise RuntimeError("A 200-state encoder must contain an affine decoder")
    source_weight, source_bias = _effective_decoder(decoder_model)
    target = encoder.native_decoder.I0
    with torch.no_grad():
        target.weight.copy_(source_weight)
        cast(torch.Tensor, target.bias).copy_(source_bias)
    for parameter in encoder.native_decoder.parameters():
        parameter.requires_grad_(False)


def _state_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float]:
    error = predicted.detach().cpu() - target.detach().cpu()
    normalized = error / scale.detach().cpu()
    return {
        "mse": float(torch.mean(error.square())),
        "mae": float(torch.mean(torch.abs(error))),
        "standardized_mse": float(torch.mean(normalized.square())),
        "standardized_mae": float(torch.mean(torch.abs(normalized))),
    }


def _fit_one(
    kind: str,
    *,
    train_days: tuple[FlexMarketDay, ...],
    validation_days: tuple[FlexMarketDay, ...],
    train_summary: torch.Tensor,
    validation_summary: torch.Tensor,
    native_fit: NativeStateFitResult,
    train_raw: torch.Tensor,
    validation_raw: torch.Tensor,
    feature_statistics: object,
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    objective_config: FlexObjectiveConfig,
    decoder_supervised_epochs: int,
    decoder_market_epochs: int,
    decoder_market_learning_rate: float,
    decoder_learning_rate_schedule: Literal["constant", "cosine"],
    decoder_learning_rate_warmup_epochs: int,
    decoder_minimum_learning_rate_fraction: float,
    recover_incomplete_decoder: bool,
    decoder_recovery_epochs: int,
    decoder_recovery_learning_rate: float,
    encoder_supervised_epochs: int,
    encoder_market_epochs: int,
    checkpoint_directory: Path,
    report_directory: Path,
    source_checkpoint_sha256: str,
    env: BaseEnvironment,
) -> dict[str, object]:
    decoder_checkpoint = checkpoint_directory / f"train-only-{kind}-200.pt"
    decoder_config = _candidate_config(kind, geometry=geometry)
    decoder_supervised_config = FlexTrainConfig(
        num_epochs=decoder_supervised_epochs,
        day_batch_size=128,
        learning_rate=2.0e-3,
        device=str(env.device),
        precision=32,
    )
    decoder_market_config = _decoder_market_train_config(
        decoder_checkpoint,
        num_epochs=decoder_market_epochs,
        learning_rate=decoder_market_learning_rate,
        learning_rate_schedule=decoder_learning_rate_schedule,
        learning_rate_warmup_epochs=decoder_learning_rate_warmup_epochs,
        minimum_learning_rate_fraction=decoder_minimum_learning_rate_fraction,
        recover_incomplete=recover_incomplete_decoder,
        recovery_epochs=decoder_recovery_epochs,
        recovery_learning_rate=decoder_recovery_learning_rate,
        device=str(env.device),
    )
    decoder_fit = fit_latent_states(
        train_days,
        native_fit,
        config=decoder_config,
        objective_config=objective_config,
        supervised_train_config=decoder_supervised_config,
        market_train_config=decoder_market_config,
        checkpoint_path=decoder_checkpoint,
    )
    decoder_model = decoder_fit.model
    validation_h = _project_validation_states(decoder_model, validation_raw)
    train_h = decoder_model.daily_states.detach().cpu()
    state_mean = torch.mean(train_h, dim=0)
    state_scale = torch.clamp(torch.std(train_h, dim=0, unbiased=False), min=1.0e-5)
    encoder_config = NativeRbfQuoteEncoderConfig(
        label=f"shared-strikes-{kind}-200-market-encoder-v1",
        state_size=200,
        geometry=geometry,
    )
    encoder = NativeRbfQuoteEncoder(
        env,
        config=encoder_config,
        native_mean=torch.zeros(geometry.native_state_size),
        native_scale=torch.ones(geometry.native_state_size),
        state_mean=state_mean,
        state_scale=state_scale,
        pca_components=torch.zeros((200, geometry.native_state_size)),
    )
    _install_frozen_decoder(encoder, decoder_model)
    decoder = FlexSanosDecoder(env, geometry=geometry)
    target_h = train_h.to(device=env.device, dtype=env.fdtype)
    target_sha256 = _sha256_tensor(train_h)
    decoder_sha256 = checkpoint_digest(decoder_checkpoint)
    encoder_checkpoint = checkpoint_directory / f"market-to-{kind}-200.pt"
    (
        phase,
        supervised_history,
        market_history,
        audits,
        resume_state,
    ) = _restore_encoder_checkpoint(
        encoder_checkpoint,
        encoder,
        train_days=train_days,
        target_sha256=target_sha256,
        source_checkpoint_sha256=decoder_sha256,
        feature_statistics=cast(object, feature_statistics),
    )
    supervised_config = EncoderTrainConfig(
        num_epochs=encoder_supervised_epochs,
        batch_size=16,
        learning_rate=1.0e-3,
    )
    market_config = EncoderTrainConfig(
        num_epochs=encoder_market_epochs,
        batch_size=16,
        learning_rate=2.0e-4,
        seed=3141,
    )

    def checkpoint_callback(
        active_phase: str,
        history: list[dict[str, float]],
        training_state: dict[str, object],
    ) -> None:
        _save_checkpoint(
            encoder,
            encoder_checkpoint,
            phase=active_phase,
            train_days=train_days,
            target_sha256=target_sha256,
            source_checkpoint_sha256=decoder_sha256,
            feature_statistics=cast(object, feature_statistics),
            supervised_history=(
                history if active_phase == "supervised" else supervised_history
            ),
            market_history=(history if active_phase == "market" else market_history),
            audits=audits,
            training_state=training_state,
        )

    if phase in {None, "supervised"}:
        supervised_history, supervised_audit = _train_phase(
            encoder,
            train_summary,
            train_days,
            phase="supervised",
            config=supervised_config,
            target_states=target_h,
            objective_config=objective_config,
            decoder=decoder,
            checkpoint_callback=checkpoint_callback,
            resume_state=resume_state if phase == "supervised" else None,
        )
        audits["supervised"] = supervised_audit
        _save_checkpoint(
            encoder,
            encoder_checkpoint,
            phase="supervised_complete",
            train_days=train_days,
            target_sha256=target_sha256,
            source_checkpoint_sha256=decoder_sha256,
            feature_statistics=cast(object, feature_statistics),
            supervised_history=supervised_history,
            market_history=[],
            audits=audits,
            training_state=None,
        )
        phase = "supervised_complete"
        resume_state = None
    elif audits.get("supervised", {}).get("converged") is not True:
        raise RuntimeError(f"{kind} encoder lacks supervised convergence proof")

    if phase != "complete":
        market_history, market_audit = _train_phase(
            encoder,
            train_summary,
            train_days,
            phase="market",
            config=market_config,
            target_states=target_h,
            objective_config=objective_config,
            decoder=decoder,
            checkpoint_callback=checkpoint_callback,
            resume_state=resume_state if phase == "market" else None,
        )
        audits["market"] = market_audit
    elif audits.get("market", {}).get("converged") is not True:
        raise RuntimeError(f"{kind} encoder lacks market convergence proof")

    train_evaluation = _evaluate(
        encoder,
        train_summary,
        train_days,
        train_raw,
        objective_config=objective_config,
        decoder=decoder,
        batch_size=16,
    )
    validation_evaluation = _evaluate(
        encoder,
        validation_summary,
        validation_days,
        validation_raw,
        objective_config=objective_config,
        decoder=decoder,
        batch_size=16,
    )
    with torch.no_grad():
        predicted_train_h = encoder.encode_state(train_summary)
        predicted_validation_h = encoder.encode_state(validation_summary)
    state_errors = {
        "train": _state_error(predicted_train_h, target_h, encoder.state_scale),
        "validation_projection_oracle": _state_error(
            predicted_validation_h,
            validation_h,
            encoder.state_scale,
        ),
    }
    encoded_history = torch.cat(
        (
            predicted_train_h.detach().cpu(),
            predicted_validation_h.detach().cpu(),
        ),
        dim=0,
    )
    all_days = train_days + validation_days
    if any(day.spot_ref is None for day in all_days):
        raise RuntimeError("A shared-strikes market day is missing spot_ref")
    spot_refs = torch.as_tensor(
        [cast(float, day.spot_ref) for day in all_days],
        dtype=torch.float64,
    )
    history_path = checkpoint_directory / f"encoded-history-{kind}-200.pt"
    history_temporary = history_path.with_suffix(history_path.suffix + ".tmp")
    torch.save(
        {
            "schema": "flex-shared-encoded-history-v1",
            "kind": kind,
            "dates": [day.trading_date.isoformat() for day in all_days],
            "states": encoded_history,
            "spot_refs": spot_refs,
            "train_end": "2024-12-31",
            "validation_start": "2025-01-02",
            "decoder_checkpoint_sha256": decoder_sha256,
        },
        history_temporary,
    )
    history_temporary.replace(history_path)
    _save_checkpoint(
        encoder,
        encoder_checkpoint,
        phase="complete",
        train_days=train_days,
        target_sha256=target_sha256,
        source_checkpoint_sha256=decoder_sha256,
        feature_statistics=cast(object, feature_statistics),
        supervised_history=supervised_history,
        market_history=market_history,
        audits=audits,
        training_state=None,
    )
    report: dict[str, object] = {
        "schema": "flex-shared-compressed-market-encoder-v1",
        "status": "complete",
        "kind": kind,
        "state_size": 200,
        "state_sizes_by_expiry": (
            list(_HIERARCHICAL_200_ALLOCATION) if kind == "hierarchical" else None
        ),
        "decoder_config": asdict(decoder_config),
        "decoder_checkpoint": str(decoder_checkpoint),
        "decoder_checkpoint_sha256": decoder_sha256,
        "decoder_fit": decoder_fit.to_dict(),
        "encoder_config": asdict(encoder_config),
        "encoder_checkpoint": str(encoder_checkpoint),
        "encoder_checkpoint_sha256": checkpoint_digest(encoder_checkpoint),
        "encoder_supervised_history": supervised_history,
        "encoder_market_history": market_history,
        "encoder_training_audits": audits,
        "train_evaluation": train_evaluation,
        "validation_2025": validation_evaluation,
        "state_errors": state_errors,
        "encoded_history": {
            "path": str(history_path),
            "state_shape": list(encoded_history.shape),
            "state_sha256": _sha256_tensor(encoded_history),
            "spot_ref_sha256": _sha256_tensor(spot_refs),
        },
        "source_native_checkpoint_sha256": source_checkpoint_sha256,
        "date_weighting": "mean quotes within each date, then mean dates",
        "validation_semantics": (
            "The decoder and market encoder use only dates through 2024. The "
            "2025 h projection is an evaluation target only and is never optimized."
        ),
    }
    write_json_atomic(report, report_directory / f"{kind}-200-encoder.json")
    return report


def run(arguments: argparse.Namespace) -> None:
    """Run both clean compressed candidates after the oracle capacity gate."""
    if arguments.ticker != "SPX":
        raise ValueError("Shared-strikes compressed encoders are specified for SPX")
    if arguments.device != "cuda:0":
        raise ValueError("Shared-strikes encoder training is restricted to GPU 0")
    if not 0.0 < arguments.wait_seconds <= 60.0:
        raise ValueError("'wait-seconds' must lie in (0,60]")
    if arguments.workers < 1 or arguments.workers > 4:
        raise ValueError("'workers' must lie in [1,4]")
    if arguments.decoder_recovery_epochs < 1:
        raise ValueError("'decoder-recovery-epochs' must be positive")
    if arguments.decoder_recovery_learning_rate <= 0.0:
        raise ValueError("'decoder-recovery-learning-rate' must be positive")
    _wait_for_completion(
        arguments.capacity_completion,
        wait_seconds=arguments.wait_seconds,
    )
    native_proof = json.loads(arguments.native_completion.read_text(encoding="utf-8"))
    if native_proof.get("status") != "native_complete":
        raise RuntimeError("Native completion proof is not complete")
    dates = _market_dates_from_completion(arguments.native_completion)
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    market_config = FlexMarketConfig()
    objective_config = FlexObjectiveConfig(fit_weight=0.05)
    days = _load_markets(
        ticker=arguments.ticker,
        dates=dates,
        geometry=geometry,
        market_config=market_config,
        workers=arguments.workers,
    )
    train_days, validation_days = _split_markets(days)
    native_raw, _ = _load_native_targets(
        arguments.native_checkpoint,
        days,
        geometry=geometry,
    )
    split = len(train_days)
    train_raw = native_raw[:split]
    validation_raw = native_raw[split:]
    feature_config = NativeRbfQuoteEncoderConfig(
        label="shared-strikes-compressed-feature-map-v1",
        state_size=200,
        geometry=geometry,
    )
    feature_statistics, train_summary_np = fit_rbf_feature_statistics(
        train_days,
        config=feature_config,
    )
    validation_summary_np = transform_market_days(
        validation_days,
        config=feature_config,
        statistics=feature_statistics,
    )
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    train_summary = torch.as_tensor(
        train_summary_np,
        dtype=env.fdtype,
        device=env.device,
    )
    validation_summary = torch.as_tensor(
        validation_summary_np,
        dtype=env.fdtype,
        device=env.device,
    )
    native_fit = _native_training_target(
        train_days,
        train_raw,
        geometry=geometry,
        objective_config=objective_config,
        env=env,
    )
    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    source_sha256 = checkpoint_digest(arguments.native_checkpoint)
    reports: dict[str, object] = {}
    with FileLock(str(arguments.checkpoint_directory / "compressed-encoders.lock")):
        for kind in ("direct", "hierarchical"):
            print(f"starting clean shared-strikes {kind}-200 encoder", flush=True)
            reports[kind] = _fit_one(
                kind,
                train_days=train_days,
                validation_days=validation_days,
                train_summary=train_summary,
                validation_summary=validation_summary,
                native_fit=native_fit,
                train_raw=train_raw,
                validation_raw=validation_raw,
                feature_statistics=feature_statistics,
                geometry=geometry,
                objective_config=objective_config,
                decoder_supervised_epochs=arguments.decoder_supervised_epochs,
                decoder_market_epochs=arguments.decoder_market_epochs,
                decoder_market_learning_rate=(
                    arguments.decoder_market_learning_rate
                ),
                decoder_learning_rate_schedule=cast(
                    Literal["constant", "cosine"],
                    arguments.decoder_learning_rate_schedule,
                ),
                decoder_learning_rate_warmup_epochs=(
                    arguments.decoder_learning_rate_warmup_epochs
                ),
                decoder_minimum_learning_rate_fraction=(
                    arguments.decoder_minimum_learning_rate_fraction
                ),
                recover_incomplete_decoder=arguments.recover_incomplete_decoder,
                decoder_recovery_epochs=arguments.decoder_recovery_epochs,
                decoder_recovery_learning_rate=(
                    arguments.decoder_recovery_learning_rate
                ),
                encoder_supervised_epochs=arguments.encoder_supervised_epochs,
                encoder_market_epochs=arguments.encoder_market_epochs,
                checkpoint_directory=arguments.checkpoint_directory,
                report_directory=arguments.report_directory,
                source_checkpoint_sha256=source_sha256,
                env=env,
            )
    comparison_path = arguments.report_directory / "compressed-200-comparison.json"
    write_json_atomic(
        {
            "schema": "flex-shared-compressed-encoder-comparison-v1",
            "status": "complete",
            "train_days": len(train_days),
            "validation_days": len(validation_days),
            "reports": reports,
        },
        comparison_path,
    )
    write_json_atomic(
        {
            "schema": "flex-shared-compressed-encoder-completion-v1",
            "status": "complete",
            "comparison_report": str(comparison_path),
        },
        arguments.report_directory / "compressed-200-encoders-complete.json",
    )


def main() -> None:
    """CLI entry point."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_shared_strikes_compressed_encoders")
