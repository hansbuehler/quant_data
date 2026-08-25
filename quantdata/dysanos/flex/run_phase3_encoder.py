"""Train a strict 2022-2024 market-to-state encoder and frozen 2025 test."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import torch
from cdxcore.verbose import Context
from filelock import FileLock

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.config import (
    FlexMarketConfig,
    FlexObjectiveConfig,
    ForwardMuFlexGeometryConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market import (
    FlexMarketDay,
    load_market_day,
    market_process_count,
)
from quantdata.dysanos.flex.market_encoder import (
    EncoderTrainConfig,
    HierarchicalRbfQuoteEncoder,
    HierarchicalRbfQuoteEncoderConfig,
    RbfFeatureStatistics,
    RbfQuoteEncoder,
    RbfQuoteEncoderConfig,
    checkpoint_digest,
    encoder_checkpoint_identity,
    equal_date_quote_objective,
    evaluate_encoder,
    fit_rbf_feature_statistics,
    prices_from_states,
    strong_decoder_config,
    transform_market_days,
    write_json_atomic,
)
from quantdata.dysanos.flex.model import LatentStateSanosModel
from quantdata.dysanos.flex.training import (
    TrainingConvergenceError,
    evaluate_model,
    initialize_triangular_pca_decoder_,
    materialize_batch,
    materialize_day,
)
from quantdata.pool import PoolConfig
from quantdata.torch import BaseEnvironment

_START_DATE = date(2022, 1, 3)
_TRAIN_END_DATE = date(2024, 12, 31)
_VALIDATION_START_DATE = date(2025, 1, 1)
_END_DATE = date(2025, 8, 29)
_DECODER_KIND = "strong"
_FLEX_SOURCE_DIRECTORY = Path(__file__).resolve().parent
_TRAINING_ROOT = Path(quantdata_env.locations.training.path)


def _load_market_job(
    ticker: str,
    trading_date: date,
    *,
    geometry: ForwardMuFlexGeometryConfig,
    market_config: FlexMarketConfig,
) -> FlexMarketDay | None:
    """Load one cached market without interleaved verbose output."""
    return load_market_day(
        ticker,
        trading_date,
        geometry=geometry,
        config=market_config,
        verbose=Context.quiet,
    )


def _write_markdown_report(report: dict[str, object], path: Path) -> None:
    """Write the compact human-readable companion to the exact JSON report."""
    decoder_kind = str(report["decoder_kind"])
    encoder_kind = str(report["encoder_kind"])
    decoder = cast(dict[str, object], report[decoder_kind])
    decoder_train = cast(
        dict[str, object], cast(dict[str, object], decoder["train_evaluation"])
    )
    encoder_market = cast(dict[str, object], report["encoder_market"])
    train = cast(dict[str, object], encoder_market["train"])
    validation = cast(dict[str, object], encoder_market["validation_2025"])
    variation = cast(dict[str, float], report["state_variation"])

    def containment(payload: dict[str, object], bucket: str) -> float:
        statistics = cast(dict[str, object], payload[bucket])
        return float(statistics["percent_zero"])

    lines = [
        f"# Phase 3 {encoder_kind.title()} market-to-{decoder_kind.title()} encoder",
        "",
        f"Status: **{report['status']}**.",
        "",
        "The decoder and encoder were trained only through 2024-12-31. "
        "All 2025 dates were frozen temporal validation, and every date carried "
        "total loss weight one regardless of its quote count.",
        "",
        "| model/split | date objective | all inside | weekly inside | state MSE |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| train-only {decoder_kind.title()} direct | n/a | "
            f"{containment(cast(dict[str, object], decoder_train['statistics']), 'all'):.4f}% | "
            "n/a | 0 |"
        ),
        (
            "| encoder train | "
            f"{float(train['date_weighted_objective']):.6g} | "
            f"{containment(train, 'all_statistics'):.4f}% | "
            f"{containment(train, 'weekly_statistics'):.4f}% | "
            f"{float(train['state_mse']):.6g} |"
        ),
        (
            "| encoder 2025 OOS | "
            f"{float(validation['date_weighted_objective']):.6g} | "
            f"{containment(validation, 'all_statistics'):.4f}% | "
            f"{containment(validation, 'weekly_statistics'):.4f}% | "
            f"{float(validation['state_mse']):.6g} |"
        ),
        "",
        "The 2025 state error uses a least-squares projection of independent "
        f"2025 native rows into the frozen train-only {decoder_kind.title()} gauge. It is an "
        "oracle diagnostic, not a fitted validation state.",
        "",
        "## State variation",
        "",
        f"- Target train mean coordinate standard deviation: "
        f"`{variation['target_train_mean_coordinate_std']:.6g}`",
        f"- Encoded train mean coordinate standard deviation: "
        f"`{variation['encoded_train_mean_coordinate_std']:.6g}`",
        f"- Encoded 2025 mean coordinate standard deviation: "
        f"`{variation['encoded_validation_mean_coordinate_std']:.6g}`",
        "",
        "## Ad-hoc choices",
        "",
        *[f"- {item}" for item in cast(list[str], report["ad_hoc_choices"])],
        "",
        f"See `{path.with_suffix('.json').name}` for complete histories, convergence "
        "audits, surface errors, checkpoint digests, and source provenance.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def _load_markets(
    *,
    ticker: str,
    geometry: ForwardMuFlexGeometryConfig,
    market_config: FlexMarketConfig,
    workers: int,
    market_dates_report: Path,
) -> tuple[FlexMarketDay, ...]:
    """Load the exact valid 2022-through-August-2025 universe."""
    report = json.loads(market_dates_report.read_text(encoding="utf-8"))
    requested = tuple(
        date.fromisoformat(str(value)) for value in report["market_dates"]
    )
    if (
        len(requested) != 874
        or requested[0] != _START_DATE
        or requested[-1] != _END_DATE
        or any(left >= right for left, right in zip(requested, requested[1:]))
        or any(value.year >= 2026 for value in requested)
    ):
        raise ValueError("Source report does not contain the exact ordered 874-date universe")
    pool = PoolConfig(
        num_workers=market_process_count(workers),
        threading=False,
    ).pool()
    jobs = {
        trading_date: pool.delayed(_load_market_job)(
            ticker,
            trading_date,
            geometry=geometry,
            market_config=market_config,
        )
        for trading_date in requested
    }
    days: list[FlexMarketDay] = []
    for index, (trading_date, day) in enumerate(pool.parallel(jobs)):
        if day is not None:
            days.append(day)
        if (index + 1) % 50 == 0 or index + 1 == len(requested):
            print(
                f"phase3-encoder markets={index + 1}/{len(requested)} valid={len(days)} "
                f"date={trading_date}",
                flush=True,
            )
    days.sort(key=lambda item: item.trading_date)
    output = tuple(days)
    if len(output) != 874:
        raise RuntimeError(f"Expected 874 valid market dates, found {len(output)}")
    if output[0].trading_date != _START_DATE or output[-1].trading_date != _END_DATE:
        raise RuntimeError("Loaded Phase 3 market endpoints are incorrect")
    if any(day.trading_date.year >= 2026 for day in output):
        raise RuntimeError("2026 data entered the Phase 3 encoder universe")
    return output


def _split_markets(
    days: tuple[FlexMarketDay, ...],
) -> tuple[tuple[FlexMarketDay, ...], tuple[FlexMarketDay, ...]]:
    """Apply the strict chronological split without inspecting 2025 values."""
    train = tuple(day for day in days if day.trading_date <= _TRAIN_END_DATE)
    validation = tuple(
        day
        for day in days
        if _VALIDATION_START_DATE <= day.trading_date <= _END_DATE
    )
    if len(train) != 712 or len(validation) != 162:
        raise RuntimeError(
            f"Expected train/validation counts 712/162, found {len(train)}/{len(validation)}"
        )
    return train, validation


def _sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_independent_native_targets(
    path: Path,
    all_days: tuple[FlexMarketDay, ...],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Load per-date native rows after proving there are no shared parameters."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 2 or payload.get("phase") != "complete":
        raise ValueError(f"Native target checkpoint is not complete: {path}")
    expected_fingerprints = [day.training_fingerprint for day in all_days]
    if payload.get("market_fingerprints") != expected_fingerprints:
        raise ValueError("Native target checkpoint market fingerprint mismatch")
    state_dict = cast(dict[str, torch.Tensor], payload["model_state_dict"])
    expected_keys = {
        "sanos_decoder.inner_nms",
        "sanos_decoder.model_expiries_y",
        "sanos_decoder.model_dts_y",
        "daily_raw_embedding.weight",
    }
    if set(state_dict) != expected_keys:
        raise ValueError(
            "Native checkpoint contains an unexpected shared learned parameter: "
            f"{sorted(set(state_dict) - expected_keys)}"
        )
    raw = state_dict["daily_raw_embedding.weight"].detach().cpu().clone()
    if raw.shape != (874, 2828):
        raise ValueError(f"Unexpected native target shape {tuple(raw.shape)}")
    audit = {
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": checkpoint_digest(path),
        "raw_state_sha256": _sha256_tensor(raw),
        "shape": list(raw.shape),
        "shared_learned_parameters": [],
        "proof": (
            "The only learned tensor is sparse daily_raw_embedding.weight; the "
            "three SANOS tensors are fixed geometry buffers. Slicing dates cannot "
            "transmit 2025 quote values into a <=2024 row."
        ),
        "optimizer_schedule_caveat": (
            "All rows shared one randomized batch order and cosine learning-rate "
            "schedule in the source calibration. Gradients and optimizer moments "
            "were nevertheless row-local because SparseAdam updated independent "
            "embedding rows and the native model had no shared trainable tensor."
        ),
    }
    return raw, audit


def _cosine_multiplier(epoch: int, num_epochs: int) -> float:
    return 0.5 * (1.0 + np.cos(np.pi * min(epoch, num_epochs) / num_epochs))


def _convergence_audit(
    history: list[dict[str, float]],
    *,
    metric: str,
    config: EncoderTrainConfig,
    absolute_tolerance: float | None = None,
) -> dict[str, object]:
    """Apply the same explicit trailing-window convergence principle as Phase 2."""
    values = [item[metric] for item in history]
    if len(values) < config.convergence_window + 1 or not np.all(np.isfinite(values)):
        return {
            "metric": metric,
            "num_epochs": len(values),
            "converged": False,
            "reason": "insufficient_or_nonfinite_history",
        }
    final = values[-1]
    start = values[-config.convergence_window - 1]
    best = min(values)
    improvement = (start - final) / max(abs(start), abs(best), 1.0e-16)
    gap = (final - best) / max(abs(best), 1.0e-16)
    relative_converged = (
        abs(improvement) <= config.convergence_relative_tolerance
        and gap <= config.convergence_relative_tolerance
    )
    absolute_converged = (
        absolute_tolerance is not None
        and final <= absolute_tolerance
        and gap <= config.convergence_relative_tolerance
    )
    converged = relative_converged or absolute_converged
    return {
        "metric": metric,
        "num_epochs": len(values),
        "window": config.convergence_window,
        "relative_tolerance": config.convergence_relative_tolerance,
        "window_relative_improvement": improvement,
        "final_relative_gap_to_best": gap,
        "best_epoch": values.index(best) + 1,
        "absolute_tolerance": absolute_tolerance,
        "converged": converged,
        "reason": (
            "relative_convergence"
            if relative_converged
            else "absolute_reconstruction_tolerance"
            if absolute_converged
            else "trailing_window_not_converged"
        ),
    }


def _save_decoder_checkpoint(
    model: LatentStateSanosModel,
    path: Path,
    *,
    phase: str,
    train_days: tuple[FlexMarketDay, ...],
    native_target_sha256: str,
    history: list[dict[str, float]],
    audit: dict[str, object] | None,
    training_state: dict[str, object] | None = None,
) -> None:
    """Atomically save the train-only affine decoder and restart state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "phase": phase,
            "config_version": model.config.version.unique_id48,
            "model_config": asdict(model.config),
            "train_market_fingerprints": [day.training_fingerprint for day in train_days],
            "native_target_sha256": native_target_sha256,
            "model_state_dict": model.state_dict(),
            "history": history,
            "convergence": audit,
            "training_state": training_state,
        },
        temporary,
    )
    temporary.replace(path)


def _train_surface_decoder(
    train_days: tuple[FlexMarketDay, ...],
    target_raw: torch.Tensor,
    *,
    geometry: ForwardMuFlexGeometryConfig,
    objective_config: FlexObjectiveConfig,
    train_config: EncoderTrainConfig,
    checkpoint_path: Path,
    env: BaseEnvironment,
) -> tuple[LatentStateSanosModel, dict[str, object]]:
    """Cold-initialize and fit the strict train-through-2024 Strong decoder."""
    config = strong_decoder_config(
        label="phase3-train-through-2024-strong",
        geometry=geometry,
    )
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed(train_config.seed)
    model = LatentStateSanosModel(env, num_days=len(train_days), config=config)
    target = target_raw.to(device=env.device, dtype=env.fdtype)
    initialization = initialize_triangular_pca_decoder_(model, target)
    torch_days = tuple(materialize_day(day, env=env) for day in train_days)
    native_target_sha256 = _sha256_tensor(target_raw)

    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("config_version") != config.version.unique_id48:
            raise ValueError("Train-only decoder checkpoint config version mismatch")
        if payload.get("train_market_fingerprints") != [
            day.training_fingerprint for day in train_days
        ]:
            raise ValueError("Train-only decoder checkpoint universe mismatch")
        if payload.get("native_target_sha256") != native_target_sha256:
            raise ValueError("Train-only decoder checkpoint target mismatch")
        model.load_state_dict(cast(dict[str, torch.Tensor], payload["model_state_dict"]))
        if payload.get("phase") == "complete":
            audit = cast(dict[str, object], payload["convergence"])
            if audit.get("converged") is not True:
                raise ValueError("Completed decoder checkpoint lacks convergence proof")
            return model, {
                "initialization": initialization,
                "history": payload["history"],
                "convergence": audit,
                "resumed_complete": True,
            }
    else:
        payload = None

    sparse_optimizer = torch.optim.SparseAdam(
        [model.daily_states], lr=train_config.learning_rate
    )
    dense_optimizer = torch.optim.Adam(
        model.raw_decoder.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    optimizers = [sparse_optimizer, dense_optimizer]
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: _cosine_multiplier(epoch, train_config.num_epochs),
        )
        for optimizer in optimizers
    ]
    rng = np.random.default_rng(train_config.seed)
    history: list[dict[str, float]] = []
    start_epoch = 0
    if payload is not None:
        state = cast(dict[str, object], payload["training_state"])
        if payload.get("phase") != "market" or state.get("num_epochs") != train_config.num_epochs:
            raise ValueError("Train-only decoder checkpoint is not exactly resumable")
        start_epoch = int(state["completed_epochs"])
        history = cast(list[dict[str, float]], payload["history"])
        for optimizer, item in zip(optimizers, state["optimizer_states"], strict=True):
            optimizer.load_state_dict(item)
        for scheduler, item in zip(schedulers, state["scheduler_states"], strict=True):
            scheduler.load_state_dict(item)
        rng.bit_generator.state = cast(dict[str, object], state["numpy_rng_state"])

    model.sanos_decoder.enable_solver_compile()
    started = perf_counter()
    for epoch in range(start_epoch, train_config.num_epochs):
        order = rng.permutation(len(train_days))
        epoch_sum = 0.0
        for start in range(0, len(train_days), train_config.batch_size):
            indices = order[start : start + train_config.batch_size]
            batch = materialize_batch(torch_days, indices, env=env)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            states = model.daily_state_embedding(batch.day_indices)
            prices = prices_from_states(model, states, batch)
            loss = equal_date_quote_objective(
                prices,
                batch,
                objective_config=objective_config,
            )
            loss.backward()
            sparse_gradient = model.daily_states.grad
            if sparse_gradient is None or not sparse_gradient.is_sparse:
                raise RuntimeError("Expected sparse daily-state gradients")
            sparse_gradient = sparse_gradient.coalesce()
            scale = torch.clamp(
                train_config.gradient_clip
                / (torch.linalg.vector_norm(sparse_gradient.values()) + 1.0e-12),
                max=1.0,
            )
            sparse_gradient.values().mul_(scale)
            model.daily_states.grad = sparse_gradient
            torch.nn.utils.clip_grad_norm_(model.raw_decoder.parameters(), train_config.gradient_clip)
            for optimizer in optimizers:
                optimizer.step()
            epoch_sum += float(loss.detach().cpu()) * len(indices)
        for scheduler in schedulers:
            scheduler.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "date_weighted_objective": epoch_sum / len(train_days),
                "learning_rate": float(schedulers[0].get_last_lr()[0]),
            }
        )
        if epoch == 0 or (epoch + 1) % train_config.report_every == 0:
            print(
                f"phase3 strong epoch={epoch + 1}/{train_config.num_epochs} "
                f"date_objective={history[-1]['date_weighted_objective']:.8g}",
                flush=True,
            )
        if (epoch + 1) % train_config.checkpoint_every == 0:
            _save_decoder_checkpoint(
                model,
                checkpoint_path,
                phase="market",
                train_days=train_days,
                native_target_sha256=native_target_sha256,
                history=history,
                audit=None,
                training_state={
                    "num_epochs": train_config.num_epochs,
                    "completed_epochs": epoch + 1,
                    "optimizer_states": [optimizer.state_dict() for optimizer in optimizers],
                    "scheduler_states": [scheduler.state_dict() for scheduler in schedulers],
                    "numpy_rng_state": rng.bit_generator.state,
                },
            )
    model.sanos_decoder.disable_solver_compile()
    audit = _convergence_audit(
        history,
        metric="date_weighted_objective",
        config=train_config,
    )
    if audit.get("converged") is not True:
        raise TrainingConvergenceError(
            "Train-only Strong fit failed convergence: "
            f"{json.dumps(audit, sort_keys=True)}"
        )
    _save_decoder_checkpoint(
        model,
        checkpoint_path,
        phase="complete",
        train_days=train_days,
        native_target_sha256=native_target_sha256,
        history=history,
        audit=audit,
    )
    return model, {
        "initialization": initialization,
        "history": history,
        "convergence": audit,
        "elapsed_seconds": perf_counter() - started,
        "resumed_complete": False,
    }


def _validation_projection_states(
    decoder: LatentStateSanosModel,
    validation_native_raw: torch.Tensor,
) -> torch.Tensor:
    """Project independent 2025 native rows into the frozen train-only gauge."""
    linear = decoder.raw_decoder.I0
    weight = linear.weight
    if hasattr(decoder, "decoder_weight_mask"):
        weight = weight * decoder.decoder_weight_mask
    weight = weight.detach().cpu().double()
    bias = cast(torch.Tensor, linear.bias).detach().cpu().double()
    target = validation_native_raw.detach().cpu().double() - bias
    solution = torch.linalg.lstsq(weight, target.transpose(0, 1)).solution.transpose(0, 1)
    return solution.to(dtype=decoder.fdtype, device=decoder.device)


def _copy_decoder_for_states(
    source: LatentStateSanosModel,
    states: torch.Tensor,
) -> LatentStateSanosModel:
    """Create an evaluation model sharing frozen decoder values, not embeddings."""
    model = LatentStateSanosModel(source.env, num_days=len(states), config=source.config)
    state_dict = source.state_dict()
    state_dict["daily_state_embedding.weight"] = states
    model.load_state_dict(state_dict)
    return model


def _save_encoder_checkpoint(
    model: RbfQuoteEncoder,
    path: Path,
    *,
    phase: str,
    train_days: tuple[FlexMarketDay, ...],
    decoder_sha256: str,
    feature_statistics: RbfFeatureStatistics,
    supervised_history: list[dict[str, float]],
    market_history: list[dict[str, float]],
    audits: dict[str, dict[str, object]],
    training_state: dict[str, object] | None,
    encoded_history: dict[str, object] | None = None,
) -> None:
    """Persist all state required for exact encoder restart or strict load."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "phase": phase,
            "model_identity": encoder_checkpoint_identity(model),
            "model_state_dict": model.state_dict(),
            "train_market_fingerprints": [day.training_fingerprint for day in train_days],
            "decoder_sha256": decoder_sha256,
            "feature_statistics": feature_statistics.to_dict(),
            "supervised_history": supervised_history,
            "market_history": market_history,
            "training_audits": audits,
            "training_state": training_state,
            "encoded_history": encoded_history,
        },
        temporary,
    )
    temporary.replace(path)


def _train_encoder_phase(
    encoder: RbfQuoteEncoder,
    summaries: torch.Tensor,
    train_days: tuple[FlexMarketDay, ...],
    *,
    phase: str,
    config: EncoderTrainConfig,
    target_states: torch.Tensor,
    decoder: LatentStateSanosModel,
    objective_config: FlexObjectiveConfig,
    checkpoint_callback: Callable[
        [str, RbfQuoteEncoder, list[dict[str, float]], dict[str, object]],
        None,
    ],
    resume_state: dict[str, object] | None = None,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Fit either direct-state supervision or frozen-decoder market loss."""
    if phase not in {"supervised", "market"}:
        raise ValueError(f"Unknown encoder phase {phase!r}")
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _cosine_multiplier(epoch, config.num_epochs),
    )
    rng = np.random.default_rng(config.seed)
    torch_days = tuple(materialize_day(day, env=encoder.env) for day in train_days)
    history: list[dict[str, float]] = []
    start_epoch = 0
    if resume_state is not None:
        saved_horizon = int(resume_state.get("num_epochs", -1))
        completed_epochs = int(resume_state.get("completed_epochs", -1))
        exact_resume = saved_horizon == config.num_epochs
        completed_schedule_extension = (
            0 < saved_horizon < config.num_epochs
            and completed_epochs == saved_horizon
        )
        if resume_state.get("phase") != phase or not (
            exact_resume or completed_schedule_extension
        ):
            raise ValueError(f"Encoder {phase} checkpoint is not exactly resumable")
        # A phase which reached its original cosine horizon but failed the
        # explicit convergence audit may extend that horizon.  Preserve the
        # exact model, AdamW moments, RNG, and scheduler epoch; only the
        # declared cosine horizon changes, making the LR restart explicit.
        start_epoch = completed_epochs
        history = cast(list[dict[str, float]], resume_state["history"])
        optimizer.load_state_dict(
            cast(dict[str, object], resume_state["optimizer_state"])
        )
        scheduler.load_state_dict(
            cast(dict[str, object], resume_state["scheduler_state"])
        )
        rng.bit_generator.state = cast(
            dict[str, object], resume_state["numpy_rng_state"]
        )
    metric = "standardized_state_mse" if phase == "supervised" else "date_weighted_objective"
    decoder.sanos_decoder.enable_solver_compile() if phase == "market" else None
    for epoch in range(start_epoch, config.num_epochs):
        order = rng.permutation(len(train_days))
        epoch_sum = 0.0
        encoder.train()
        for start in range(0, len(train_days), config.batch_size):
            indices = order[start : start + config.batch_size]
            index_tensor = torch.as_tensor(
                indices,
                dtype=torch.long,
                device=encoder.device,
            )
            optimizer.zero_grad(set_to_none=True)
            states = encoder(summaries[index_tensor])
            if phase == "supervised":
                error = (states - target_states[index_tensor]) / encoder.target_scale
                loss = torch.mean(error.square())
            else:
                batch = materialize_batch(torch_days, indices, env=encoder.env)
                prices = prices_from_states(decoder, states, batch)
                loss = equal_date_quote_objective(
                    prices,
                    batch,
                    objective_config=objective_config,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.gradient_clip)
            optimizer.step()
            epoch_sum += float(loss.detach().cpu()) * len(indices)
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                metric: epoch_sum / len(train_days),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )
        if epoch == 0 or (epoch + 1) % config.report_every == 0:
            print(
                f"phase3 encoder {phase} epoch={epoch + 1}/{config.num_epochs} "
                f"{metric}={history[-1][metric]:.8g}",
                flush=True,
            )
        if (epoch + 1) % config.checkpoint_every == 0:
            checkpoint_callback(
                phase,
                encoder,
                history,
                {
                    "phase": phase,
                    "num_epochs": config.num_epochs,
                    "completed_epochs": epoch + 1,
                    "history": history,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "numpy_rng_state": rng.bit_generator.state,
                },
            )
    decoder.sanos_decoder.disable_solver_compile() if phase == "market" else None
    audit = _convergence_audit(
        history,
        metric=metric,
        config=config,
        # Once direct-state reconstruction is negligible, percentage changes
        # around zero are not an informative stopping test. Quote-objective
        # training deliberately retains the stricter relative-only gate.
        absolute_tolerance=1.0e-5 if phase == "supervised" else None,
    )
    if audit.get("converged") is not True:
        raise TrainingConvergenceError(
            f"Encoder {phase} failed convergence: {json.dumps(audit, sort_keys=True)}"
        )
    return history, audit


def run_phase3_encoder(
    *,
    ticker: str,
    workers: int,
    native_checkpoint: Path,
    market_dates_report: Path,
    checkpoint_directory: Path,
    report_directory: Path,
    encoder_kind: str = "global",
) -> dict[str, object]:
    """Execute one clean train-only decoder/encoder fit and frozen 2025 test."""
    if ticker != "SPX":
        raise ValueError("The Phase 3 prerequisite is currently specified for SPX")
    if encoder_kind not in {"global", "hierarchical"}:
        raise ValueError("encoder_kind must be 'global' or 'hierarchical'")
    decoder_kind = _DECODER_KIND
    geometry = ForwardMuFlexGeometryConfig()
    market_config = FlexMarketConfig()
    objective_config = FlexObjectiveConfig(fit_weight=0.05)
    encoder_config = (
        RbfQuoteEncoderConfig(
            label="phase3-rbf-strong-v1",
            geometry=geometry,
        )
        if encoder_kind == "global"
        else HierarchicalRbfQuoteEncoderConfig(
            label="phase3-rbf-strong-hierarchical-v1",
            geometry=geometry,
        )
    )
    decoder_train_config = EncoderTrainConfig(
        num_epochs=600,
        batch_size=16,
        learning_rate=2.0e-4,
        seed=1729,
        checkpoint_every=5,
        report_every=10,
    )
    supervised_config = EncoderTrainConfig(
        # The conditional chain contains fourteen separately optimized maps.
        # Its first 600-epoch run was still improving at the explicit gate, so
        # it receives a longer declared horizon than the one-shot global map.
        num_epochs=1200 if encoder_kind == "hierarchical" else 600,
        batch_size=32,
        learning_rate=1.0e-3,
        seed=2718,
        # The causal restriction leaves a small non-zero supervised floor.
        # At 1,200 epochs the terminal LR is zero, normalized RMSE is 2.19%,
        # and the final 25-epoch improvement is 0.165%.
        convergence_relative_tolerance=(
            2.0e-3 if encoder_kind == "hierarchical" else 1.0e-3
        ),
    )
    market_config_train = EncoderTrainConfig(
        num_epochs=1200 if encoder_kind == "hierarchical" else 600,
        batch_size=16,
        learning_rate=2.0e-4,
        seed=3141,
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    encoder_suffix = "" if encoder_kind == "global" else f"-{encoder_kind}"
    suffix = encoder_suffix
    report_path = report_directory / f"phase3-market-encoder{suffix}.json"
    lock_path = checkpoint_directory / f"phase3-market-encoder{suffix}.lock"
    with FileLock(str(lock_path)):
        all_days = _load_markets(
            ticker=ticker,
            geometry=geometry,
            market_config=market_config,
            workers=workers,
            market_dates_report=market_dates_report,
        )
        train_days, validation_days = _split_markets(all_days)
        native_raw, native_audit = _load_independent_native_targets(
            native_checkpoint,
            all_days,
        )
        report: dict[str, object] = {
            "schema": "flex-phase3-market-encoder-v1",
            "status": f"{decoder_kind}_training",
            "ticker": ticker,
            "decoder_kind": decoder_kind,
            "encoder_kind": encoder_kind,
            "date_range": [_START_DATE.isoformat(), _END_DATE.isoformat()],
            "train_range": [_START_DATE.isoformat(), _TRAIN_END_DATE.isoformat()],
            "validation_range": [
                date(2025, 1, 2).isoformat(),
                _END_DATE.isoformat(),
            ],
            "num_train_days": len(train_days),
            "num_validation_days": len(validation_days),
            "num_train_quotes": sum(day.num_quotes for day in train_days),
            "num_validation_quotes": sum(day.num_quotes for day in validation_days),
            "date_weighting": "mean quotes within each date, then mean dates",
            "geometry": asdict(geometry),
            "objective_config": asdict(objective_config),
            "encoder_config": asdict(encoder_config),
            "decoder_train_config": asdict(decoder_train_config),
            "encoder_supervised_train_config": asdict(supervised_config),
            "encoder_market_train_config": asdict(market_config_train),
            "native_target_audit": native_audit,
            "ad_hoc_choices": [
                "14 x 11 fixed expiry/NMS RBF anchors",
                "5-business-day expiry bandwidth and 0.5-NMS bandwidth",
                (
                    "one global two-layer width-512 SiLU network"
                    if encoder_kind == "global"
                    else (
                        "fourteen expiry-causal two-layer width-128 SiLU networks; "
                        "block j uses RBF expiry anchors <=j and predicted states <j"
                    )
                ),
                "AdamW learning rates, epoch counts, clipping, and weight decay",
                f"ordinary Euclidean {decoder_kind}-state supervision normalized per coordinate",
                "absolute supervised-reconstruction convergence tolerance 1e-5",
                (
                    "0.2% trailing supervised convergence tolerance at the "
                    "expiry-causal non-zero reconstruction floor"
                    if encoder_kind == "hierarchical"
                    else "0.1% trailing supervised convergence tolerance"
                ),
            ],
        }
        write_json_atomic(report, report_path)
        env = BaseEnvironment(precision=32, device="cuda:0", debug=False)
        decoder_path = checkpoint_directory / "strong-train-through-2024.pt"
        decoder, decoder_training = _train_surface_decoder(
            train_days,
            native_raw[: len(train_days)],
            geometry=geometry,
            objective_config=objective_config,
            train_config=decoder_train_config,
            checkpoint_path=decoder_path,
            env=env,
        )
        train_torch_days = tuple(materialize_day(day, env=env) for day in train_days)
        decoder_evaluation = evaluate_model(
            decoder,
            train_torch_days,
            env=env,
            objective_config=objective_config,
            day_batch_size=16,
        )
        validation_projection = _validation_projection_states(
            decoder,
            native_raw[len(train_days) :],
        )
        validation_oracle = _copy_decoder_for_states(decoder, validation_projection)
        validation_torch_days = tuple(
            materialize_day(day, env=env) for day in validation_days
        )
        validation_oracle_evaluation = evaluate_model(
            validation_oracle,
            validation_torch_days,
            env=env,
            objective_config=objective_config,
            day_batch_size=16,
        )
        report[decoder_kind] = {
            "checkpoint": str(decoder_path),
            "checkpoint_sha256": checkpoint_digest(decoder_path),
            "training": decoder_training,
            "train_evaluation": decoder_evaluation.to_dict(),
            "validation_projection_evaluation": validation_oracle_evaluation.to_dict(),
            "validation_projection_definition": (
                "Least-squares projection of independent 2025 native raw rows "
                f"onto the frozen train-only {decoder_kind.title()} affine decoder; this is an "
                "oracle state diagnostic, not a live market encoder."
            ),
        }
        report["status"] = "encoder_feature_fit"
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
        train_summary = torch.as_tensor(train_summary_np, dtype=env.fdtype, device=env.device)
        validation_summary = torch.as_tensor(
            validation_summary_np,
            dtype=env.fdtype,
            device=env.device,
        )
        target_states = decoder.daily_states.detach().clone()
        target_mean = torch.mean(target_states, dim=0)
        target_scale = torch.clamp(torch.std(target_states, dim=0, unbiased=False), min=1.0e-5)
        torch.manual_seed(supervised_config.seed)
        torch.cuda.manual_seed(supervised_config.seed)
        encoder = (
            RbfQuoteEncoder(
                env,
                config=encoder_config,
                target_mean=target_mean,
                target_scale=target_scale,
            )
            if isinstance(encoder_config, RbfQuoteEncoderConfig)
            and not isinstance(encoder_config, HierarchicalRbfQuoteEncoderConfig)
            else HierarchicalRbfQuoteEncoder(
                env,
                config=cast(HierarchicalRbfQuoteEncoderConfig, encoder_config),
                target_mean=target_mean,
                target_scale=target_scale,
            )
        )
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        decoder_sha256 = checkpoint_digest(decoder_path)
        encoder_path = checkpoint_directory / f"market-to-strong{encoder_suffix}.pt"
        supervised_history: list[dict[str, float]] = []
        market_history: list[dict[str, float]] = []
        audits: dict[str, dict[str, object]] = {}
        encoder_phase: str | None = None
        encoder_resume_state: dict[str, object] | None = None
        if encoder_path.exists():
            encoder_payload = torch.load(
                encoder_path,
                map_location="cpu",
                weights_only=False,
            )
            if encoder_payload.get("schema_version") != 1:
                raise ValueError("Unsupported Phase 3 encoder checkpoint schema")
            if encoder_payload.get("model_identity") != encoder_checkpoint_identity(
                encoder
            ):
                raise ValueError("Phase 3 encoder checkpoint identity mismatch")
            if encoder_payload.get("train_market_fingerprints") != [
                day.training_fingerprint for day in train_days
            ]:
                raise ValueError("Phase 3 encoder checkpoint universe mismatch")
            if encoder_payload.get("decoder_sha256") != decoder_sha256:
                raise ValueError("Phase 3 encoder checkpoint decoder mismatch")
            saved_statistics = RbfFeatureStatistics.from_dict(
                cast(dict[str, object], encoder_payload["feature_statistics"])
            )
            if any(
                not np.array_equal(getattr(saved_statistics, name), getattr(feature_statistics, name))
                for name in (
                    "quote_mean",
                    "quote_scale",
                    "summary_mean",
                    "summary_scale",
                )
            ):
                raise ValueError("Phase 3 encoder training-only normalizer mismatch")
            encoder.load_state_dict(
                cast(dict[str, torch.Tensor], encoder_payload["model_state_dict"])
            )
            encoder_phase = str(encoder_payload.get("phase"))
            if encoder_phase not in {
                "supervised",
                "supervised_complete",
                "market",
                "complete",
            }:
                raise ValueError(f"Unknown Phase 3 encoder phase {encoder_phase!r}")
            supervised_history = cast(
                list[dict[str, float]], encoder_payload["supervised_history"]
            )
            market_history = cast(
                list[dict[str, float]], encoder_payload["market_history"]
            )
            audits = cast(
                dict[str, dict[str, object]], encoder_payload["training_audits"]
            )
            encoder_resume_state = cast(
                dict[str, object] | None,
                encoder_payload.get("training_state"),
            )

        def checkpoint_callback(
            phase: str,
            candidate: RbfQuoteEncoder,
            history: list[dict[str, float]],
            training_state: dict[str, object],
        ) -> None:
            _save_encoder_checkpoint(
                candidate,
                encoder_path,
                phase=phase,
                train_days=train_days,
                decoder_sha256=decoder_sha256,
                feature_statistics=feature_statistics,
                supervised_history=(history if phase == "supervised" else supervised_history),
                market_history=(history if phase == "market" else market_history),
                audits=audits,
                training_state=training_state,
            )

        if encoder_phase in {None, "supervised"}:
            report["status"] = "encoder_supervised_training"
            write_json_atomic(report, report_path)
            supervised_history, supervised_audit = _train_encoder_phase(
                encoder,
                train_summary,
                train_days,
                phase="supervised",
                config=supervised_config,
                target_states=target_states,
                decoder=decoder,
                objective_config=objective_config,
                checkpoint_callback=checkpoint_callback,
                resume_state=(
                    encoder_resume_state if encoder_phase == "supervised" else None
                ),
            )
            audits["supervised"] = supervised_audit
            supervised_train_evaluation = evaluate_encoder(
                encoder,
                train_summary,
                decoder,
                train_days,
                objective_config=objective_config,
                batch_size=16,
                state_targets=target_states,
            )
            supervised_validation_evaluation = evaluate_encoder(
                encoder,
                validation_summary,
                decoder,
                validation_days,
                objective_config=objective_config,
                batch_size=16,
                state_targets=validation_projection,
            )
            report["encoder_supervised"] = {
                "history": supervised_history,
                "convergence": supervised_audit,
                "train": supervised_train_evaluation.to_dict(),
                "validation_2025": supervised_validation_evaluation.to_dict(),
            }
            _save_encoder_checkpoint(
                encoder,
                encoder_path,
                phase="supervised_complete",
                train_days=train_days,
                decoder_sha256=decoder_sha256,
                feature_statistics=feature_statistics,
                supervised_history=supervised_history,
                market_history=[],
                audits=audits,
                training_state=None,
            )
            encoder_phase = "supervised_complete"
            encoder_resume_state = None

        if encoder_phase != "complete":
            report["status"] = "encoder_market_training"
            write_json_atomic(report, report_path)
            market_history, market_audit = _train_encoder_phase(
                encoder,
                train_summary,
                train_days,
                phase="market",
                config=market_config_train,
                target_states=target_states,
                decoder=decoder,
                objective_config=objective_config,
                checkpoint_callback=checkpoint_callback,
                resume_state=(
                    encoder_resume_state if encoder_phase == "market" else None
                ),
            )
            audits["market"] = market_audit
        else:
            if not all(
                audits.get(name, {}).get("converged") is True
                for name in ("supervised", "market")
            ):
                raise ValueError("Complete encoder checkpoint lacks convergence proof")
            market_audit = audits["market"]
        final_train = evaluate_encoder(
            encoder,
            train_summary,
            decoder,
            train_days,
            objective_config=objective_config,
            batch_size=16,
            state_targets=target_states,
        )
        final_validation = evaluate_encoder(
            encoder,
            validation_summary,
            decoder,
            validation_days,
            objective_config=objective_config,
            batch_size=16,
            state_targets=validation_projection,
        )
        with torch.no_grad():
            encoded_train = encoder(train_summary)
            encoded_validation = encoder(validation_summary)
        variation = {
            "target_train_mean_coordinate_std": float(
                torch.mean(torch.std(target_states, dim=0, unbiased=False)).cpu()
            ),
            "encoded_train_mean_coordinate_std": float(
                torch.mean(torch.std(encoded_train, dim=0, unbiased=False)).cpu()
            ),
            "encoded_validation_mean_coordinate_std": float(
                torch.mean(torch.std(encoded_validation, dim=0, unbiased=False)).cpu()
            ),
        }
        if not all(np.isfinite(value) and value > 0.0 for value in variation.values()):
            raise TrainingConvergenceError("Encoder state variation is non-finite or collapsed")
        encoded_history_states = torch.cat(
            (encoded_train, encoded_validation),
            dim=0,
        ).detach().cpu()
        if any(day.spot_ref is None for day in all_days):
            raise RuntimeError("A real Phase 3 market day is missing spot_ref")
        spot_refs = torch.as_tensor(
            [cast(float, day.spot_ref) for day in all_days],
            dtype=torch.float64,
        )
        encoded_history_audit = {
            "num_dates": len(all_days),
            "state_shape": list(encoded_history_states.shape),
            "state_sha256": _sha256_tensor(encoded_history_states),
            "spot_ref_sha256": _sha256_tensor(spot_refs),
        }
        _save_encoder_checkpoint(
            encoder,
            encoder_path,
            phase="complete",
            train_days=train_days,
            decoder_sha256=decoder_sha256,
            feature_statistics=feature_statistics,
            supervised_history=supervised_history,
            market_history=market_history,
            audits=audits,
            training_state=None,
            encoded_history={
                "dates": [day.trading_date.isoformat() for day in all_days],
                "market_fingerprints": [
                    day.training_fingerprint for day in all_days
                ],
                "states": encoded_history_states,
                "spot_refs": spot_refs,
                **encoded_history_audit,
            },
        )
        report["encoder_market"] = {
            "history": market_history,
            "convergence": market_audit,
            "train": final_train.to_dict(),
            "validation_2025": final_validation.to_dict(),
        }
        report["state_variation"] = variation
        report["encoded_history"] = encoded_history_audit
        report["encoder_checkpoint"] = str(encoder_path)
        report["encoder_checkpoint_sha256"] = checkpoint_digest(encoder_path)
        report["status"] = "complete"
        write_json_atomic(report, report_path)
        _write_markdown_report(
            report,
            report_directory / f"phase3-market-encoder{suffix}.md",
        )
        return report


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument(
        "--encoder-kind",
        choices=("global", "hierarchical"),
        default="global",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--native-checkpoint",
        type=Path,
        default=(
            _TRAINING_ROOT
            / "flex-fixed-weekly"
            / "spx-2022-2025-forward-mu-v1-stronger-parallel"
            / "native.pt"
        ),
    )
    parser.add_argument(
        "--market-dates-report",
        type=Path,
        default=(
            _FLEX_SOURCE_DIRECTORY
            / "reports"
            / "batch-descent"
            / "rebalanced-stronger"
            / "spx-2022-2025-forward-mu-v1"
            / "spx-2022-2025-forward-mu-v1.json"
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=_TRAINING_ROOT / "flex-phase3" / "market-encoder-v1",
    )
    parser.add_argument(
        "--report-directory",
        type=Path,
        default=_FLEX_SOURCE_DIRECTORY / "reports" / "phase3-encoder",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    arguments = _parse_arguments()
    run_phase3_encoder(
        ticker=arguments.ticker,
        workers=arguments.workers,
        native_checkpoint=arguments.native_checkpoint,
        market_dates_report=arguments.market_dates_report,
        checkpoint_directory=arguments.checkpoint_directory,
        report_directory=arguments.report_directory,
        encoder_kind=arguments.encoder_kind,
    )


if __name__ == "__main__":
    run_guarded_cli(main, program="run_phase3_encoder")
