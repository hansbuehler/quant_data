"""Train the full shared-strike market encoder and test frozen 2025 dates."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import torch
from filelock import FileLock

from quantdata.dysanos.flex.config import (
    FlexGeometryConfig,
    FlexMarketConfig,
    FlexObjectiveConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market import FlexMarketDay, market_process_count
from quantdata.dysanos.flex.market_encoder import (
    EncoderTrainConfig,
    NativeRbfQuoteEncoder,
    NativeRbfQuoteEncoderConfig,
    RbfFeatureStatistics,
    checkpoint_digest,
    fit_rbf_feature_statistics,
    transform_market_days,
    write_json_atomic,
)
from quantdata.dysanos.flex.run_phase3_encoder import _load_market_job
from quantdata.dysanos.flex.run_phase3_native_encoder import (
    _CHECKPOINT_SCHEMA,
    _evaluate,
    _model_identity,
    _pca_initialization,
    _save_checkpoint,
    _sha256_tensor,
    _train_phase,
    _write_markdown,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.pool import PoolConfig
from quantdata.torch import BaseEnvironment

_TRAIN_END_DATE = date(2024, 12, 31)
_VALIDATION_START_DATE = date(2025, 1, 2)
_VALIDATION_END_DATE = date(2025, 8, 29)


def _market_cache_geometry(
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
) -> FlexGeometryConfig:
    """Retain every field that defines the canonical market universe."""
    return FlexGeometryConfig(
        **{
            field.name: getattr(geometry, field.name)
            for field in fields(FlexGeometryConfig)
        }
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    parser.add_argument("--supervised-epochs", type=int, default=1_200)
    parser.add_argument("--market-epochs", type=int, default=600)
    parser.add_argument(
        "--source-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-label",
        default="spx-2022-2025-shared-strikes-v1",
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--report-directory",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def _wait_for_native_completion(
    *,
    source_directory: Path,
    source_label: str,
    wait_seconds: float,
) -> Path:
    """Wait for the immutable native completion proof, rejecting source failure."""
    if not 0.0 < wait_seconds <= 60.0:
        raise ValueError("'wait_seconds' must lie in (0,60]")
    completion_path = source_directory / f"{source_label}-native-complete.json"
    live_report_path = source_directory / f"{source_label}.json"
    while not completion_path.exists():
        status = "missing"
        if live_report_path.exists():
            report = json.loads(live_report_path.read_text(encoding="utf-8"))
            status = str(report.get("status", "unknown"))
            if status.endswith("_failed"):
                raise RuntimeError(
                    "Shared-strike native calibration failed before encoder "
                    f"training: {report.get('failure', status)}"
                )
        print(f"waiting for shared native completion status={status}", flush=True)
        time.sleep(wait_seconds)
    return completion_path


def _market_dates_from_completion(path: Path) -> tuple[date, ...]:
    """Read and validate the exact date universe represented by the checkpoint."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "native_complete":
        raise ValueError(
            "Shared native completion proof has unexpected status "
            f"{report.get('status')!r}"
        )
    raw_dates = report.get("market_dates")
    if not isinstance(raw_dates, list) or not raw_dates:
        raise ValueError("Shared native completion proof has no market dates")
    market_dates = tuple(date.fromisoformat(str(value)) for value in raw_dates)
    if tuple(sorted(set(market_dates))) != market_dates:
        raise ValueError("Shared native market dates must be unique and sorted")
    if market_dates[-1] != _VALIDATION_END_DATE:
        raise ValueError(
            "Shared native market dates have unexpected final date "
            f"{market_dates[-1]}"
        )
    if any(value.year >= 2026 for value in market_dates):
        raise ValueError("2026 data entered the shared-strike encoder universe")
    if report.get("num_days") != len(market_dates):
        raise ValueError("Shared native report day count does not match its dates")
    proof = report.get("native_checkpoint_proof")
    if not isinstance(proof, dict) or proof.get("num_market_days") != len(
        market_dates
    ):
        raise ValueError("Shared native checkpoint proof has the wrong day count")
    return market_dates


def _load_markets(
    *,
    ticker: str,
    dates: tuple[date, ...],
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    market_config: FlexMarketConfig,
    workers: int,
) -> tuple[FlexMarketDay, ...]:
    """Reload exactly the completed native universe from the existing cache."""
    # Market filtering is independent of the pricing parameterization, but its
    # horizon and NMS interval are part of the market identity.  Preserve every
    # base-geometry field, including an explicit expiry grid, exactly as the
    # production gate does when it creates the certified checkpoint.
    cache_geometry = _market_cache_geometry(geometry)
    pool = PoolConfig(
        num_workers=market_process_count(workers),
        threading=False,
    ).pool()
    jobs = {
        trading_date: pool.delayed(_load_market_job)(
            ticker,
            trading_date,
            geometry=cache_geometry,
            market_config=market_config,
        )
        for trading_date in dates
    }
    days: list[FlexMarketDay] = []
    for index, (trading_date, day) in enumerate(pool.parallel(jobs)):
        if day is None:
            raise RuntimeError(
                "A certified shared native market became unavailable: "
                f"{trading_date}"
            )
        days.append(day)
        if (index + 1) % 50 == 0 or index + 1 == len(dates):
            print(
                f"shared encoder markets={index + 1}/{len(dates)} "
                f"date={trading_date}",
                flush=True,
            )
    days.sort(key=lambda value: value.trading_date)
    output = tuple(days)
    if tuple(day.trading_date for day in output) != dates:
        raise RuntimeError("Reloaded shared encoder universe changed date ordering")
    return output


def _split_markets(
    days: tuple[FlexMarketDay, ...],
) -> tuple[tuple[FlexMarketDay, ...], tuple[FlexMarketDay, ...]]:
    """Use pre-2025 training and the untouched 2025 temporal validation set."""
    train = tuple(day for day in days if day.trading_date <= _TRAIN_END_DATE)
    validation = tuple(
        day
        for day in days
        if _VALIDATION_START_DATE <= day.trading_date <= _VALIDATION_END_DATE
    )
    if not train or not validation or len(train) + len(validation) != len(days):
        raise RuntimeError("Shared encoder train/validation split is incomplete")
    return train, validation


def _load_native_targets(
    path: Path,
    days: tuple[FlexMarketDay, ...],
    *,
    geometry: SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Load the already capped-unbounded independent per-date native rows."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 2 or payload.get("phase") != "complete":
        raise ValueError(f"Shared native target checkpoint is incomplete: {path}")
    expected_fingerprints = [day.training_fingerprint for day in days]
    if payload.get("market_fingerprints") != expected_fingerprints:
        raise ValueError("Shared native target market fingerprints changed")
    state_dict = cast(dict[str, torch.Tensor], payload["model_state_dict"])
    expected_keys = {
        "daily_raw_embedding.weight",
        "sanos_decoder.inner_nms",
        "sanos_decoder.model_expiries_y",
        "sanos_decoder.model_dts_y",
        "sanos_decoder.shared_first_nms",
        "sanos_decoder.shared_last_nms",
    }
    if set(state_dict) != expected_keys:
        raise ValueError(
            "Shared native checkpoint has unexpected learned/buffer tensors: "
            f"{sorted(set(state_dict).symmetric_difference(expected_keys))}"
        )
    raw = state_dict["daily_raw_embedding.weight"].detach().cpu().clone()
    expected_shape = (len(days), geometry.native_state_size)
    if raw.shape != expected_shape or not torch.all(torch.isfinite(raw)):
        raise ValueError(
            f"Expected finite shared native targets {expected_shape}, found "
            f"{tuple(raw.shape)}"
        )
    return raw, {
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": checkpoint_digest(path),
        "raw_state_sha256": _sha256_tensor(raw),
        "shape": list(raw.shape),
        "shared_learned_parameters": [],
        "proof": (
            "The only learned tensor is sparse daily_raw_embedding.weight; all "
            "SANOS tensors are fixed geometry buffers. Encoder fitting uses only "
            "rows dated through 2024."
        ),
    }


def _restore_encoder_checkpoint(
    path: Path,
    model: NativeRbfQuoteEncoder,
    *,
    train_days: tuple[FlexMarketDay, ...],
    target_sha256: str,
    source_checkpoint_sha256: str,
    feature_statistics: RbfFeatureStatistics,
) -> tuple[
    str | None,
    list[dict[str, float]],
    list[dict[str, float]],
    dict[str, dict[str, object]],
    dict[str, object] | None,
]:
    """Restore a compatible resumable encoder or return a cold-start state."""
    if not path.exists():
        return None, [], [], {}, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != _CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported shared encoder checkpoint schema")
    if payload.get("model_identity") != _model_identity(model):
        raise ValueError("Shared encoder checkpoint model identity mismatch")
    if payload.get("train_market_fingerprints") != [
        day.training_fingerprint for day in train_days
    ]:
        raise ValueError("Shared encoder checkpoint training universe mismatch")
    if payload.get("target_sha256") != target_sha256:
        raise ValueError("Shared encoder checkpoint target mismatch")
    if payload.get("source_checkpoint_sha256") != source_checkpoint_sha256:
        raise ValueError("Shared encoder checkpoint source mismatch")
    saved_statistics = RbfFeatureStatistics.from_dict(
        cast(dict[str, object], payload["feature_statistics"])
    )
    if any(
        not np.array_equal(
            getattr(saved_statistics, name),
            getattr(feature_statistics, name),
        )
        for name in ("quote_mean", "quote_scale", "summary_mean", "summary_scale")
    ):
        raise ValueError("Shared encoder feature normalizer mismatch")
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model_state_dict"]))
    phase = str(payload.get("phase"))
    if phase not in {"supervised", "supervised_complete", "market", "complete"}:
        raise ValueError(f"Unknown shared encoder checkpoint phase {phase!r}")
    return (
        phase,
        cast(list[dict[str, float]], payload["supervised_history"]),
        cast(list[dict[str, float]], payload["market_history"]),
        cast(dict[str, dict[str, object]], payload["training_audits"]),
        cast(dict[str, object] | None, payload.get("training_state")),
    )


def run_shared_strikes_full_encoder(
    *,
    ticker: str,
    workers: int,
    device: str,
    supervised_epochs: int,
    market_epochs: int,
    source_directory: Path,
    source_label: str,
    checkpoint_directory: Path,
    report_directory: Path,
    wait_seconds: float,
) -> dict[str, object]:
    """Fit market-to-full-native on pre-2025 data and evaluate frozen 2025."""
    if ticker != "SPX":
        raise ValueError("The shared-strike encoder is currently specified for SPX")
    if device not in {"cuda:0", "cpu"}:
        raise ValueError("The shared-strike encoder is restricted to GPU 0 or CPU")
    if supervised_epochs <= 0 or market_epochs <= 0:
        raise ValueError("Encoder epoch counts must be positive")
    completion_path = _wait_for_native_completion(
        source_directory=source_directory,
        source_label=source_label,
        wait_seconds=wait_seconds,
    )
    source_checkpoint = source_directory.parent / "checkpoints" / "native.pt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(
            f"Shared native checkpoint is missing: {source_checkpoint}"
        )
    dates = _market_dates_from_completion(completion_path)
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig()
    market_config = FlexMarketConfig()
    objective_config = FlexObjectiveConfig(fit_weight=0.05)
    days = _load_markets(
        ticker=ticker,
        dates=dates,
        geometry=geometry,
        market_config=market_config,
        workers=workers,
    )
    train_days, validation_days = _split_markets(days)
    native_raw, native_source_audit = _load_native_targets(
        source_checkpoint,
        days,
        geometry=geometry,
    )
    train_raw = native_raw[: len(train_days)]
    validation_raw = native_raw[len(train_days) :]
    state_size = geometry.native_state_size
    encoder_config = NativeRbfQuoteEncoderConfig(
        label="shared-strikes-full-native-v1",
        state_size=state_size,
        group="flex/shared-strikes/full-native-encoder",
        geometry=geometry,
    )
    supervised_config = EncoderTrainConfig(
        num_epochs=supervised_epochs,
        batch_size=32,
        learning_rate=1.0e-3,
        seed=2718,
        convergence_relative_tolerance=3.0e-3,
    )
    market_train_config = EncoderTrainConfig(
        num_epochs=market_epochs,
        batch_size=16,
        learning_rate=2.0e-4,
        seed=3141,
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / "market-to-native-2828.pt"
    report_path = report_directory / "shared-strikes-full-encoder.json"
    lock_path = checkpoint_directory / "market-to-native-2828.lock"
    with FileLock(str(lock_path)):
        report: dict[str, object] = {
            "schema": "flex-shared-strikes-full-encoder-v1",
            "status": "feature_fit",
            "ticker": ticker,
            "state_size": state_size,
            "native_state_size": state_size,
            "date_range": [dates[0].isoformat(), dates[-1].isoformat()],
            "train_range": [dates[0].isoformat(), _TRAIN_END_DATE.isoformat()],
            "validation_range": [
                _VALIDATION_START_DATE.isoformat(),
                _VALIDATION_END_DATE.isoformat(),
            ],
            "num_train_days": len(train_days),
            "num_validation_days": len(validation_days),
            "num_train_quotes": sum(day.num_quotes for day in train_days),
            "num_validation_quotes": sum(day.num_quotes for day in validation_days),
            "date_weighting": "mean quotes within each date, then mean dates",
            "evaluation_semantics": "same-set amortized calibration",
            "geometry": asdict(geometry),
            "objective_config": asdict(objective_config),
            "encoder_config": asdict(encoder_config),
            "supervised_train_config": asdict(supervised_config),
            "market_train_config": asdict(market_train_config),
            "native_source_audit": native_source_audit,
            "native_targets_parameterization": "capped-unbounded shared-strike raw gauge",
            "ad_hoc_choices": [
                "14 x 11 fixed expiry/NMS Gaussian-RBF anchors",
                "5-business-day expiry bandwidth and 0.5-NMS bandwidth",
                "two SiLU hidden layers of width 512",
                f"{supervised_epochs} supervised and {market_epochs} market epochs",
                "0.3% full-native supervised plateau tolerance",
            ],
        }
        write_json_atomic(report, report_path)
        feature_statistics, train_summary_np = fit_rbf_feature_statistics(
            train_days,
            config=encoder_config,
        )
        validation_summary_np = transform_market_days(
            validation_days,
            config=encoder_config,
            statistics=feature_statistics,
        )
        env = BaseEnvironment(precision=32, device=device, debug=False)
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
        initialization = _pca_initialization(
            train_raw,
            state_size=state_size,
            env=env,
        )
        target_states = cast(torch.Tensor, initialization["target_states"])
        torch.manual_seed(supervised_config.seed)
        if device == "cuda:0":
            torch.cuda.manual_seed(supervised_config.seed)
        model = NativeRbfQuoteEncoder(
            env,
            config=encoder_config,
            native_mean=cast(torch.Tensor, initialization["native_mean"]),
            native_scale=cast(torch.Tensor, initialization["native_scale"]),
            state_mean=cast(torch.Tensor, initialization["state_mean"]),
            state_scale=cast(torch.Tensor, initialization["state_scale"]),
            pca_components=None,
        )
        decoder = FlexSanosDecoder(env, geometry=geometry)
        report["initialization"] = {
            "method": "identity standardized native state",
            "retained_standardized_native_variance": 1.0,
        }
        source_sha256 = checkpoint_digest(source_checkpoint)
        target_sha256 = _sha256_tensor(native_raw)
        (
            phase,
            supervised_history,
            market_history,
            audits,
            resume_state,
        ) = _restore_encoder_checkpoint(
            checkpoint_path,
            model,
            train_days=train_days,
            target_sha256=target_sha256,
            source_checkpoint_sha256=source_sha256,
            feature_statistics=feature_statistics,
        )

        def checkpoint_callback(
            active_phase: str,
            history: list[dict[str, float]],
            training_state: dict[str, object],
        ) -> None:
            _save_checkpoint(
                model,
                checkpoint_path,
                phase=active_phase,
                train_days=train_days,
                target_sha256=target_sha256,
                source_checkpoint_sha256=source_sha256,
                feature_statistics=feature_statistics,
                supervised_history=(
                    history if active_phase == "supervised" else supervised_history
                ),
                market_history=(history if active_phase == "market" else market_history),
                audits=audits,
                training_state=training_state,
            )

        if phase in {None, "supervised"}:
            report["status"] = "supervised_training"
            write_json_atomic(report, report_path)
            supervised_history, supervised_audit = _train_phase(
                model,
                train_summary,
                train_days,
                phase="supervised",
                config=supervised_config,
                target_states=target_states,
                objective_config=objective_config,
                decoder=decoder,
                checkpoint_callback=checkpoint_callback,
                resume_state=resume_state if phase == "supervised" else None,
            )
            audits["supervised"] = supervised_audit
            _save_checkpoint(
                model,
                checkpoint_path,
                phase="supervised_complete",
                train_days=train_days,
                target_sha256=target_sha256,
                source_checkpoint_sha256=source_sha256,
                feature_statistics=feature_statistics,
                supervised_history=supervised_history,
                market_history=[],
                audits=audits,
                training_state=None,
            )
            phase = "supervised_complete"
            resume_state = None
        elif audits.get("supervised", {}).get("converged") is not True:
            raise ValueError("Shared encoder lacks supervised convergence proof")

        if phase != "complete":
            report["status"] = "market_training"
            report["supervised"] = {
                "history": supervised_history,
                "convergence": audits["supervised"],
            }
            write_json_atomic(report, report_path)
            market_history, market_audit = _train_phase(
                model,
                train_summary,
                train_days,
                phase="market",
                config=market_train_config,
                target_states=target_states,
                objective_config=objective_config,
                decoder=decoder,
                checkpoint_callback=checkpoint_callback,
                resume_state=resume_state if phase == "market" else None,
            )
            audits["market"] = market_audit
        else:
            if audits.get("market", {}).get("converged") is not True:
                raise ValueError("Complete shared encoder lacks market convergence proof")
            market_audit = audits["market"]

        train_evaluation = _evaluate(
            model,
            train_summary,
            train_days,
            train_raw,
            objective_config=objective_config,
            decoder=decoder,
            batch_size=16,
        )
        validation_evaluation = _evaluate(
            model,
            validation_summary,
            validation_days,
            validation_raw,
            objective_config=objective_config,
            decoder=decoder,
            batch_size=16,
        )
        validation_native_ceiling = _evaluate(
            model,
            validation_summary,
            validation_days,
            validation_raw,
            objective_config=objective_config,
            decoder=decoder,
            batch_size=16,
            prediction_native_raw=validation_raw,
        )
        with torch.no_grad():
            train_state = model.encode_state(train_summary)
            validation_state = model.encode_state(validation_summary)
        encoded_history = torch.cat(
            (train_state.detach().cpu(), validation_state.detach().cpu()),
            dim=0,
        )
        all_days = train_days + validation_days
        if any(day.spot_ref is None for day in all_days):
            raise RuntimeError("A shared-strikes market day is missing spot_ref")
        spot_refs = torch.as_tensor(
            [cast(float, day.spot_ref) for day in all_days],
            dtype=torch.float64,
        )
        history_path = checkpoint_directory / "encoded-history-full-2828.pt"
        history_temporary = history_path.with_suffix(history_path.suffix + ".tmp")
        torch.save(
            {
                "schema": "flex-shared-encoded-history-v1",
                "kind": "full",
                "dates": [day.trading_date.isoformat() for day in all_days],
                "states": encoded_history,
                "spot_refs": spot_refs,
                "train_end": _TRAIN_END_DATE.isoformat(),
                "validation_start": _VALIDATION_START_DATE.isoformat(),
                "decoder": "identity",
                "source_native_checkpoint_sha256": source_sha256,
            },
            history_temporary,
        )
        history_temporary.replace(history_path)
        state_variation = {
            "train_mean_coordinate_std": float(
                torch.mean(torch.std(train_state, dim=0, unbiased=False)).cpu()
            ),
            "validation_mean_coordinate_std": float(
                torch.mean(torch.std(validation_state, dim=0, unbiased=False)).cpu()
            ),
        }
        if not all(np.isfinite(value) and value > 0.0 for value in state_variation.values()):
            raise RuntimeError("Shared encoder state variation collapsed")
        _save_checkpoint(
            model,
            checkpoint_path,
            phase="complete",
            train_days=train_days,
            target_sha256=target_sha256,
            source_checkpoint_sha256=source_sha256,
            feature_statistics=feature_statistics,
            supervised_history=supervised_history,
            market_history=market_history,
            audits=audits,
            training_state=None,
        )
        report["supervised"] = {
            "history": supervised_history,
            "convergence": audits["supervised"],
        }
        report["market"] = {
            "history": market_history,
            "convergence": market_audit,
        }
        report["train_evaluation"] = train_evaluation
        report["validation_2025"] = validation_evaluation
        report["validation_2025_direct_native_ceiling"] = {
            "semantics": (
                "Independently calibrated per-day shared-strike native state; "
                "this is the SANOS geometry ceiling, not an OOS encoder."
            ),
            "evaluation": validation_native_ceiling,
        }
        report["state_variation"] = state_variation
        report["encoded_history"] = {
            "path": str(history_path),
            "state_shape": list(encoded_history.shape),
            "state_sha256": _sha256_tensor(encoded_history),
            "spot_ref_sha256": _sha256_tensor(spot_refs),
        }
        report["checkpoint"] = str(checkpoint_path)
        report["checkpoint_sha256"] = checkpoint_digest(checkpoint_path)
        report["status"] = "complete"
        write_json_atomic(report, report_path)
        _write_markdown(
            report,
            report_directory / "shared-strikes-full-encoder.md",
        )
        completion = report_directory / "shared-strikes-full-encoder-complete.json"
        write_json_atomic(
            {
                "schema": "flex-shared-strikes-full-encoder-completion-v1",
                "status": "complete",
                "report": str(report_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": report["checkpoint_sha256"],
            },
            completion,
        )
        return report


def main() -> None:
    """CLI entry point."""
    arguments = _parse_arguments()
    run_shared_strikes_full_encoder(
        ticker=arguments.ticker,
        workers=arguments.workers,
        device=arguments.device,
        supervised_epochs=arguments.supervised_epochs,
        market_epochs=arguments.market_epochs,
        source_directory=arguments.source_directory,
        source_label=arguments.source_label,
        checkpoint_directory=arguments.checkpoint_directory,
        report_directory=arguments.report_directory,
        wait_seconds=arguments.wait_seconds,
    )


if __name__ == "__main__":
    run_guarded_cli(main, program="run_shared_strikes_full_encoder")
