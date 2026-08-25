"""Train flat market-to-native FLEX encoders and test on frozen 2025 data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import torch
from filelock import FileLock

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.config import (
    CappedUnboundedDlvFlexGeometryConfig,
    FlexMarketConfig,
    FlexObjectiveConfig,
    ForwardMuFlexGeometryConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market import FlexMarketDay
from quantdata.dysanos.flex.market_encoder import (
    EncoderTrainConfig,
    NativeRbfQuoteEncoder,
    NativeRbfQuoteEncoderConfig,
    RbfFeatureStatistics,
    checkpoint_digest,
    equal_date_quote_objective,
    fit_rbf_feature_statistics,
    transform_market_days,
    weekly_quote_mask,
    write_json_atomic,
)
from quantdata.dysanos.flex.metrics import (
    excess_statistics,
    normalized_excess,
    surface_fit_statistics,
)
from quantdata.dysanos.flex.model import (
    NativeParameters,
    bound_native_raw,
    dlvsq_cap_penalty,
    raw_from_native_parameters,
)
from quantdata.dysanos.flex.run_phase3_encoder import (
    _convergence_audit,
    _cosine_multiplier,
    _load_independent_native_targets,
    _load_markets,
    _sha256_tensor,
    _split_markets,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder
from quantdata.dysanos.flex.training import (
    TorchMarketBatch,
    materialize_batch,
    materialize_day,
)
from quantdata.torch import BaseEnvironment

_FULL_NATIVE_STATE_SIZE = 2_828
_HIERARCHICAL_TOTAL_CAPACITY = 140
_SUPPORTED_STATE_SIZES = (_HIERARCHICAL_TOTAL_CAPACITY, _FULL_NATIVE_STATE_SIZE)
_CHECKPOINT_SCHEMA = 1
_FLEX_SOURCE_DIRECTORY = Path(__file__).resolve().parent
_TRAINING_ROOT = Path(quantdata_env.locations.training.path)


def _convert_sigmoid_native_targets(
    legacy_raw: torch.Tensor,
    *,
    target_geometry: CappedUnboundedDlvFlexGeometryConfig,
    legacy_geometry: ForwardMuFlexGeometryConfig | None = None,
) -> torch.Tensor:
    """Preserve physical parameters while replacing only the DLV raw gauge."""
    legacy_geometry = (
        ForwardMuFlexGeometryConfig()
        if legacy_geometry is None
        else legacy_geometry
    )
    if (
        legacy_geometry.num_model_expiries
        != target_geometry.num_model_expiries
        or legacy_geometry.num_inner_strikes
        != target_geometry.num_inner_strikes
    ):
        raise ValueError("Legacy and target native geometries must share dimensions")
    if legacy_raw.ndim != 2 or legacy_raw.shape[1] != legacy_geometry.native_state_size:
        raise ValueError(
            "Legacy native target must have shape (*,2828), found "
            f"{tuple(legacy_raw.shape)}"
        )
    with torch.no_grad():
        legacy_parameters = bound_native_raw(
            legacy_raw,
            geometry=legacy_geometry,
        )
        converted = raw_from_native_parameters(
            NativeParameters(
                dlvsqs=legacy_parameters.dlvsqs,
                forward_vols=legacy_parameters.forward_vols,
                eta=legacy_parameters.eta,
            ),
            geometry=target_geometry,
        )
    if converted.shape != legacy_raw.shape or not torch.all(torch.isfinite(converted)):
        raise RuntimeError("Converted capped-unbounded native targets are invalid")
    return converted


def _pca_initialization(
    train_native_raw: torch.Tensor,
    *,
    state_size: int,
    env: BaseEnvironment,
) -> dict[str, torch.Tensor | float]:
    """Build train-only native normalization and an exact affine PCA gauge."""
    target = train_native_raw.to(device=env.device, dtype=env.fdtype)
    native_mean = torch.mean(target, dim=0)
    native_scale = torch.clamp(
        torch.std(target, dim=0, unbiased=False),
        min=1.0e-5,
    )
    standardized = (target - native_mean) / native_scale
    native_size = target.shape[1]
    if state_size == native_size:
        target_states = standardized
        state_mean = torch.zeros(native_size, dtype=env.fdtype, device=env.device)
        state_scale = torch.ones(native_size, dtype=env.fdtype, device=env.device)
        components = None
        retained_variance = 1.0
    else:
        _, singular_values, vh = torch.linalg.svd(
            standardized,
            full_matrices=False,
        )
        components = vh[:state_size]
        target_states = standardized @ components.T
        state_mean = torch.mean(target_states, dim=0)
        state_scale = torch.clamp(
            torch.std(target_states, dim=0, unbiased=False),
            min=1.0e-5,
        )
        retained_variance = float(
            torch.sum(singular_values[:state_size].square())
            / torch.sum(singular_values.square())
        )
    return {
        "native_mean": native_mean,
        "native_scale": native_scale,
        "target_states": target_states,
        "state_mean": state_mean,
        "state_scale": state_scale,
        "pca_components": components,
        "retained_standardized_native_variance": retained_variance,
    }


def _prices_from_native_raw(
    raw: torch.Tensor,
    batch: TorchMarketBatch,
    *,
    geometry: CappedUnboundedDlvFlexGeometryConfig,
    decoder: FlexSanosDecoder,
) -> torch.Tensor:
    """Price each unique call location from explicit native raw coordinates."""
    parameters = bound_native_raw(raw, geometry=geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    call_locations = decoder.price_calls_batch(
        surface,
        expiries_y=batch.price_expiries_y,
        strikes=batch.price_strikes,
    )
    calls = torch.gather(call_locations, dim=1, index=batch.quote_price_index)
    return torch.where(batch.is_call, calls, calls - 1.0 + batch.strikes)


def _cap_diagnostics(
    raw: torch.Tensor,
    *,
    geometry: CappedUnboundedDlvFlexGeometryConfig,
) -> dict[str, int | float]:
    """Measure pricing-cap contact in the unbounded log-DLV² coordinates."""
    native = raw.detach().cpu().double()
    blocks = native.reshape(
        native.shape[0],
        geometry.num_model_expiries,
        geometry.expiry_block_state_size,
    )
    dlv_raw = blocks[..., : geometry.num_inner_strikes]
    log_excess = dlv_raw - geometry.maximum_log_dlvsq
    at_or_above = log_excess >= 0.0
    above = log_excess > 0.0
    penalty = dlvsq_cap_penalty(native, geometry=geometry)
    return {
        "num_dlvsq_nodes": int(dlv_raw.numel()),
        "num_at_or_above_price_cap": int(torch.sum(at_or_above)),
        "fraction_at_or_above_price_cap": float(torch.mean(at_or_above.double())),
        "num_above_price_cap": int(torch.sum(above)),
        "fraction_above_price_cap": float(torch.mean(above.double())),
        "maximum_log_excess_above_cap": float(torch.clamp(log_excess, min=0.0).max()),
        "mean_surface_cap_penalty": float(torch.mean(penalty)),
        "median_surface_cap_penalty": float(torch.quantile(penalty, 0.5)),
        "maximum_surface_cap_penalty": float(torch.max(penalty)),
    }


def _evaluate(
    model: NativeRbfQuoteEncoder,
    summaries: torch.Tensor,
    market_days: tuple[FlexMarketDay, ...],
    target_native_raw: torch.Tensor,
    *,
    objective_config: FlexObjectiveConfig,
    decoder: FlexSanosDecoder,
    batch_size: int,
    prediction_native_raw: torch.Tensor | None = None,
) -> dict[str, object]:
    """Evaluate same-set amortized calibration on one frozen temporal split."""
    if summaries.shape != (len(market_days), model.config.summary_size):
        raise ValueError("Native encoder summaries do not match market dates")
    if target_native_raw.shape != (
        len(market_days),
        model.config.geometry.native_state_size,
    ):
        raise ValueError("Native encoder targets do not match market dates")
    if prediction_native_raw is not None and prediction_native_raw.shape != (
        len(market_days),
        model.config.geometry.native_state_size,
    ):
        raise ValueError("Explicit native predictions do not match market dates")
    explicit_prediction = (
        None
        if prediction_native_raw is None
        else prediction_native_raw.to(device=model.device, dtype=model.fdtype)
    )
    torch_days = tuple(materialize_day(day, env=model.env) for day in market_days)
    fits: list[np.ndarray] = []
    predicted_raw: list[torch.Tensor] = []
    per_day_objectives: list[float] = []
    quote_loss_sum = 0.0
    quote_count = 0
    model.eval()
    decoder.eval()
    with torch.no_grad():
        for start in range(0, len(market_days), batch_size):
            indices = np.arange(start, min(start + batch_size, len(market_days)))
            batch = materialize_batch(torch_days, indices, env=model.env)
            raw = (
                model(summaries[batch.day_indices])
                if explicit_prediction is None
                else explicit_prediction[batch.day_indices]
            )
            prices = _prices_from_native_raw(
                raw,
                batch,
                geometry=model.config.geometry,
                decoder=decoder,
            )
            half_spread = 0.5 * (batch.asks - batch.bids)
            safe = torch.where(batch.mask, half_spread, torch.ones_like(half_spread))
            values = (
                torch.relu(prices - batch.asks)
                + torch.relu(batch.bids - prices)
                + objective_config.fit_weight
                * torch.abs(prices - batch.fit_prices)
            ) / safe
            valid = batch.mask.to(dtype=prices.dtype)
            day_values = torch.sum(values * valid, dim=1) / torch.sum(valid, dim=1)
            per_day_objectives.extend(day_values.cpu().tolist())
            quote_loss_sum += float(torch.sum(values * valid).cpu())
            quote_count += int(torch.sum(batch.mask).cpu())
            predicted_raw.append(raw.cpu())
            for row, day_index in enumerate(indices):
                count = market_days[int(day_index)].num_quotes
                fits.append(prices[row, :count].cpu().numpy().astype(np.float64))

    raw_prediction = torch.cat(predicted_raw, dim=0)
    target = target_native_raw.detach().cpu().to(dtype=raw_prediction.dtype)
    normalized_error = (
        raw_prediction - target
    ) / model.native_scale.detach().cpu()
    fit = np.concatenate(fits)
    bid = np.concatenate([day.bids for day in market_days])
    ask = np.concatenate([day.asks for day in market_days])
    market_fit = np.concatenate([day.fit_prices for day in market_days])
    weekly_masks = tuple(
        weekly_quote_mask(day, geometry=model.config.geometry)
        for day in market_days
    )
    weekly_fit = np.concatenate(
        [value[mask] for value, mask in zip(fits, weekly_masks, strict=True)]
    )
    weekly_bid = np.concatenate(
        [day.bids[mask] for day, mask in zip(market_days, weekly_masks, strict=True)]
    )
    weekly_ask = np.concatenate(
        [day.asks[mask] for day, mask in zip(market_days, weekly_masks, strict=True)]
    )
    weekly_market_fit = np.concatenate(
        [
            day.fit_prices[mask]
            for day, mask in zip(market_days, weekly_masks, strict=True)
        ]
    )
    return {
        "num_days": len(market_days),
        "num_quotes": quote_count,
        "quote_weighted_objective": quote_loss_sum / quote_count,
        "date_weighted_objective": float(np.mean(per_day_objectives)),
        "all_statistics": excess_statistics(
            normalized_excess(fit, bid, ask)
        ).to_dict(),
        "weekly_statistics": excess_statistics(
            normalized_excess(weekly_fit, weekly_bid, weekly_ask)
        ).to_dict(),
        "surface_fit": surface_fit_statistics(
            fit,
            market_fit,
            bid,
            ask,
        ).to_dict(),
        "weekly_surface_fit": surface_fit_statistics(
            weekly_fit,
            weekly_market_fit,
            weekly_bid,
            weekly_ask,
        ).to_dict(),
        "native_raw_mse": float(torch.mean((raw_prediction - target).square())),
        "standardized_native_raw_mse": float(torch.mean(normalized_error.square())),
        "standardized_native_raw_mae": float(torch.mean(torch.abs(normalized_error))),
        "cap": _cap_diagnostics(raw_prediction, geometry=model.config.geometry),
    }


def _model_identity(model: NativeRbfQuoteEncoder) -> dict[str, object]:
    return {
        "model_class": type(model).__qualname__,
        "config_version": model.config.version.unique_id48,
        "user_version": model.config.user_version,
        "model_config": asdict(model.config),
    }


def _save_checkpoint(
    model: NativeRbfQuoteEncoder,
    path: Path,
    *,
    phase: str,
    train_days: tuple[FlexMarketDay, ...],
    target_sha256: str,
    source_checkpoint_sha256: str,
    feature_statistics: RbfFeatureStatistics,
    supervised_history: list[dict[str, float]],
    market_history: list[dict[str, float]],
    audits: dict[str, dict[str, object]],
    training_state: dict[str, object] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": _CHECKPOINT_SCHEMA,
            "phase": phase,
            "model_identity": _model_identity(model),
            "model_state_dict": model.state_dict(),
            "train_market_fingerprints": [
                day.training_fingerprint for day in train_days
            ],
            "target_sha256": target_sha256,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "feature_statistics": feature_statistics.to_dict(),
            "supervised_history": supervised_history,
            "market_history": market_history,
            "training_audits": audits,
            "training_state": training_state,
        },
        temporary,
    )
    temporary.replace(path)


def _train_phase(
    model: NativeRbfQuoteEncoder,
    summaries: torch.Tensor,
    train_days: tuple[FlexMarketDay, ...],
    *,
    phase: str,
    config: EncoderTrainConfig,
    target_states: torch.Tensor,
    objective_config: FlexObjectiveConfig,
    decoder: FlexSanosDecoder,
    checkpoint_callback: Callable[
        [str, list[dict[str, float]], dict[str, object]],
        None,
    ],
    resume_state: dict[str, object] | None,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Train supervised state reconstruction or the equal-date market loss."""
    if phase not in {"supervised", "market"}:
        raise ValueError(f"Unknown native encoder phase {phase!r}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _cosine_multiplier(epoch, config.num_epochs),
    )
    rng = np.random.default_rng(config.seed)
    torch_days = tuple(materialize_day(day, env=model.env) for day in train_days)
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
            raise ValueError(f"Native encoder {phase} checkpoint is not resumable")
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
        torch.random.set_rng_state(cast(torch.Tensor, resume_state["torch_rng_state"]))
        torch.cuda.set_rng_state(
            cast(torch.Tensor, resume_state["cuda_rng_state"]),
            device=model.device,
        )

    metric = (
        "standardized_state_mse"
        if phase == "supervised"
        else "total_date_weighted_objective"
    )
    if phase == "market":
        decoder.enable_solver_compile()
    started = perf_counter()
    for epoch in range(start_epoch, config.num_epochs):
        order = rng.permutation(len(train_days))
        total_sum = 0.0
        quote_sum = 0.0
        cap_sum = 0.0
        model.train()
        for start in range(0, len(train_days), config.batch_size):
            indices = order[start : start + config.batch_size]
            index_tensor = torch.as_tensor(
                indices,
                dtype=torch.long,
                device=model.device,
            )
            optimizer.zero_grad(set_to_none=True)
            if phase == "supervised":
                state = model.encode_state(summaries[index_tensor])
                error = (
                    state - target_states[index_tensor]
                ) / model.state_scale
                loss = torch.mean(error.square())
                quote_loss = torch.zeros_like(loss)
                cap_penalty = torch.zeros_like(loss)
            else:
                batch = materialize_batch(torch_days, indices, env=model.env)
                raw = model(summaries[index_tensor])
                prices = _prices_from_native_raw(
                    raw,
                    batch,
                    geometry=model.config.geometry,
                    decoder=decoder,
                )
                quote_loss = equal_date_quote_objective(
                    prices,
                    batch,
                    objective_config=objective_config,
                )
                cap_penalty = torch.mean(
                    dlvsq_cap_penalty(raw, geometry=model.config.geometry)
                )
                loss = quote_loss + cap_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            weight = len(indices)
            total_sum += float(loss.detach().cpu()) * weight
            quote_sum += float(quote_loss.detach().cpu()) * weight
            cap_sum += float(cap_penalty.detach().cpu()) * weight
        scheduler.step()
        item = {
            "epoch": float(epoch + 1),
            metric: total_sum / len(train_days),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        if phase == "market":
            item["date_weighted_quote_objective"] = quote_sum / len(train_days)
            item["date_weighted_dlvsq_cap_penalty"] = cap_sum / len(train_days)
        history.append(item)
        if epoch == 0 or (epoch + 1) % config.report_every == 0:
            print(
                f"phase3 native encoder state={model.config.state_size} "
                f"{phase} epoch={epoch + 1}/{config.num_epochs} "
                f"{metric}={history[-1][metric]:.8g}",
                flush=True,
            )
        if (epoch + 1) % config.checkpoint_every == 0:
            checkpoint_callback(
                phase,
                history,
                {
                    "phase": phase,
                    "num_epochs": config.num_epochs,
                    "completed_epochs": epoch + 1,
                    "history": history,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.random.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(model.device),
                },
            )
    if phase == "market":
        decoder.disable_solver_compile()
    audit = _convergence_audit(
        history,
        metric=metric,
        config=config,
        absolute_tolerance=1.0e-5 if phase == "supervised" else None,
    )
    audit["elapsed_seconds"] = perf_counter() - started
    if audit.get("converged") is not True:
        raise RuntimeError(
            f"Native encoder {phase} failed convergence: "
            f"{json.dumps(audit, sort_keys=True)}"
        )
    return history, audit


def _write_markdown(report: dict[str, object], path: Path) -> None:
    train = cast(dict[str, object], report["train_evaluation"])
    validation = cast(dict[str, object], report["validation_2025"])
    validation_ceiling = cast(
        dict[str, object],
        cast(
            dict[str, object],
            report["validation_2025_direct_native_ceiling"],
        )["evaluation"],
    )

    def statistic(payload: dict[str, object], bucket: str, key: str) -> float:
        return float(cast(dict[str, object], payload[bucket])[key])

    def cap_statistic(payload: dict[str, object], key: str) -> float:
        return float(cast(dict[str, object], payload["cap"])[key])

    lines = [
        f"# Phase 3 flat native encoder, state {report['state_size']}",
        "",
        f"Status: **{report['status']}**.",
        "",
        "This is same-set amortized calibration: each date's full eligible quote "
        "set enters the invariant RBF summary and is reconstructed through the "
        "capped-unbounded native SANOS state. The frozen 2025 split measures "
        "temporal generalization of that calibration map, not masked-quote valuation.",
        "",
        "| split | date objective | all inside | weekly inside | positive mean | positive median |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| train through 2024 | {float(train['date_weighted_objective']):.6g} | "
            f"{statistic(train, 'all_statistics', 'percent_zero'):.4f}% | "
            f"{statistic(train, 'weekly_statistics', 'percent_zero'):.4f}% | "
            f"{statistic(train, 'all_statistics', 'positive_mean'):.6g} | "
            f"{statistic(train, 'all_statistics', 'positive_median'):.6g} |"
        ),
        (
            f"| frozen 2025 | {float(validation['date_weighted_objective']):.6g} | "
            f"{statistic(validation, 'all_statistics', 'percent_zero'):.4f}% | "
            f"{statistic(validation, 'weekly_statistics', 'percent_zero'):.4f}% | "
            f"{statistic(validation, 'all_statistics', 'positive_mean'):.6g} | "
            f"{statistic(validation, 'all_statistics', 'positive_median'):.6g} |"
        ),
        (
            "| 2025 direct native ceiling | "
            f"{float(validation_ceiling['date_weighted_objective']):.6g} | "
            f"{statistic(validation_ceiling, 'all_statistics', 'percent_zero'):.4f}% | "
            f"{statistic(validation_ceiling, 'weekly_statistics', 'percent_zero'):.4f}% | "
            f"{statistic(validation_ceiling, 'all_statistics', 'positive_mean'):.6g} | "
            f"{statistic(validation_ceiling, 'all_statistics', 'positive_median'):.6g} |"
        ),
        "",
        f"The explicit surface-state size is `{report['state_size']}`; the native "
        f"SANOS output size is `{report['native_state_size']}`.",
        "",
        "DLV² uses `min_dlvsq + exp(z)`, is mechanically capped for pricing at "
        "`max_dlvsq`, and carries the AAE smooth log-coordinate cap penalty.",
        "",
        "| split | DLV² nodes at/above cap | mean surface cap penalty |",
        "| --- | ---: | ---: |",
        (
            "| train through 2024 | "
            f"{100.0 * cap_statistic(train, 'fraction_at_or_above_price_cap'):.6f}% | "
            f"{cap_statistic(train, 'mean_surface_cap_penalty'):.6g} |"
        ),
        (
            "| frozen 2025 | "
            f"{100.0 * cap_statistic(validation, 'fraction_at_or_above_price_cap'):.6f}% | "
            f"{cap_statistic(validation, 'mean_surface_cap_penalty'):.6g} |"
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def _write_comparison(report_directory: Path) -> None:
    paths = {
        state_size: report_directory / f"phase3-native-encoder-{state_size}.json"
        for state_size in _SUPPORTED_STATE_SIZES
    }
    if not all(path.exists() for path in paths.values()):
        return
    reports = {
        state_size: json.loads(path.read_text(encoding="utf-8"))
        for state_size, path in paths.items()
    }
    if not all(report.get("status") == "complete" for report in reports.values()):
        return
    baseline_paths = {
        "strong_global_140": report_directory.parent
        / "phase3-encoder/phase3-market-encoder.json",
        "strong_hierarchical_140": report_directory.parent
        / "phase3-encoder/phase3-market-encoder-hierarchical.json",
    }
    rows: dict[str, dict[str, float]] = {}
    for state_size, report in reports.items():
        evaluation = cast(dict[str, object], report["validation_2025"])
        all_statistics = cast(dict[str, object], evaluation["all_statistics"])
        weekly_statistics = cast(dict[str, object], evaluation["weekly_statistics"])
        rows[f"flat_native_{state_size}"] = {
            "date_weighted_objective": float(evaluation["date_weighted_objective"]),
            "all_percent_zero": float(all_statistics["percent_zero"]),
            "weekly_percent_zero": float(weekly_statistics["percent_zero"]),
            "positive_mean": float(all_statistics["positive_mean"]),
            "positive_median": float(all_statistics["positive_median"]),
        }
    ceiling = cast(
        dict[str, object],
        cast(
            dict[str, object],
            reports[_FULL_NATIVE_STATE_SIZE][
                "validation_2025_direct_native_ceiling"
            ],
        )["evaluation"],
    )
    ceiling_all = cast(dict[str, object], ceiling["all_statistics"])
    ceiling_weekly = cast(dict[str, object], ceiling["weekly_statistics"])
    rows["direct_native_ceiling"] = {
        "date_weighted_objective": float(ceiling["date_weighted_objective"]),
        "all_percent_zero": float(ceiling_all["percent_zero"]),
        "weekly_percent_zero": float(ceiling_weekly["percent_zero"]),
        "positive_mean": float(ceiling_all["positive_mean"]),
        "positive_median": float(ceiling_all["positive_median"]),
    }
    for name, path in baseline_paths.items():
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        evaluation = cast(
            dict[str, object],
            cast(dict[str, object], report["encoder_market"])["validation_2025"],
        )
        all_statistics = cast(dict[str, object], evaluation["all_statistics"])
        weekly_statistics = cast(dict[str, object], evaluation["weekly_statistics"])
        rows[name] = {
            "date_weighted_objective": float(evaluation["date_weighted_objective"]),
            "all_percent_zero": float(all_statistics["percent_zero"]),
            "weekly_percent_zero": float(weekly_statistics["percent_zero"]),
            "positive_mean": float(all_statistics["positive_mean"]),
            "positive_median": float(all_statistics["positive_median"]),
        }
    payload: dict[str, object] = {
        "schema": "flex-phase3-native-encoder-comparison-v1",
        "status": "complete",
        "validation_period": ["2025-01-02", "2025-08-29"],
        "evaluation_semantics": "same-set amortized calibration",
        "rows": rows,
    }
    write_json_atomic(payload, report_directory / "phase3-native-encoder-comparison.json")
    lines = [
        "# Phase 3 native-encoder comparison",
        "",
        "Frozen 2025 same-set amortized-calibration results:",
        "",
        "| model | date objective | all inside | weekly inside | positive mean | positive median |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['date_weighted_objective']:.6g} | "
            f"{row['all_percent_zero']:.4f}% | {row['weekly_percent_zero']:.4f}% | "
            f"{row['positive_mean']:.6g} | {row['positive_median']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The Strong baselines use the earlier sigmoid-DLV parameterization; "
            "the flat-native rows use the new capped-unbounded DLV² gauge.",
            "",
        ]
    )
    path = report_directory / "phase3-native-encoder-comparison.md"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def run_phase3_native_encoder(
    *,
    ticker: str,
    state_size: int,
    workers: int,
    native_checkpoint: Path,
    market_dates_report: Path,
    checkpoint_directory: Path,
    report_directory: Path,
) -> dict[str, object]:
    """Train one flat native-state capacity and evaluate frozen 2025 dates."""
    if ticker != "SPX":
        raise ValueError("The Phase 3 native encoder is currently specified for SPX")
    if state_size not in _SUPPORTED_STATE_SIZES:
        raise ValueError(
            f"state_size must be one of {_SUPPORTED_STATE_SIZES}, found {state_size}"
        )
    geometry = CappedUnboundedDlvFlexGeometryConfig()
    if geometry.native_state_size != _FULL_NATIVE_STATE_SIZE:
        raise RuntimeError("Unexpected production native-state geometry")
    market_config = FlexMarketConfig()
    objective_config = FlexObjectiveConfig(fit_weight=0.05)
    encoder_config = NativeRbfQuoteEncoderConfig(
        label=f"phase3-rbf-native-state-{state_size}-v1",
        state_size=state_size,
        geometry=geometry,
    )
    supervised_config = EncoderTrainConfig(
        # The full-state run was still improving at epoch 600.  The explicit
        # convergence gate therefore extends the resumable cosine horizon;
        # no market fine-tuning is allowed before this phase truly plateaus.
        num_epochs=1_200,
        batch_size=32,
        learning_rate=1.0e-3,
        seed=2718,
        convergence_relative_tolerance=(
            3.0e-3
            if state_size == geometry.native_state_size
            else 1.0e-3
        ),
    )
    market_train_config = EncoderTrainConfig(
        num_epochs=600,
        batch_size=16,
        learning_rate=2.0e-4,
        seed=3141,
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / f"market-to-native-{state_size}.pt"
    report_path = report_directory / f"phase3-native-encoder-{state_size}.json"
    lock_path = checkpoint_directory / f"market-to-native-{state_size}.lock"
    with FileLock(str(lock_path)):
        all_days = _load_markets(
            ticker=ticker,
            geometry=geometry,
            market_config=market_config,
            workers=workers,
            market_dates_report=market_dates_report,
        )
        train_days, validation_days = _split_markets(all_days)
        legacy_raw, native_source_audit = _load_independent_native_targets(
            native_checkpoint,
            all_days,
        )
        converted_raw = _convert_sigmoid_native_targets(
            legacy_raw,
            target_geometry=geometry,
        )
        train_raw = converted_raw[: len(train_days)]
        validation_raw = converted_raw[len(train_days) :]
        source_checkpoint_sha256 = checkpoint_digest(native_checkpoint)
        target_sha256 = _sha256_tensor(converted_raw)
        report: dict[str, object] = {
            "schema": "flex-phase3-native-market-encoder-v1",
            "status": "feature_fit",
            "ticker": ticker,
            "state_size": state_size,
            "native_state_size": geometry.native_state_size,
            "hierarchical_total_capacity": _HIERARCHICAL_TOTAL_CAPACITY,
            "date_range": ["2022-01-03", "2025-08-29"],
            "train_range": ["2022-01-03", "2024-12-31"],
            "validation_range": ["2025-01-02", "2025-08-29"],
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
            "converted_target_sha256": target_sha256,
            "dlvsq_parameterization": {
                "raw": "min_dlvsq + exp(z)",
                "pricing": "min(raw_dlvsq, max_dlvsq)",
                "cap_barrier": (
                    "0.5 * sum((width * softplus((z-log(max-min))/width) "
                    "/ std)^2) per surface"
                ),
            },
            "ad_hoc_choices": [
                "14 x 11 fixed expiry/NMS RBF anchors",
                "5-business-day expiry bandwidth and 0.5-NMS bandwidth",
                "two SiLU hidden layers of width 512",
                "1,200 supervised and 600 market epochs with mandatory convergence gates",
                (
                    "0.3% full-native supervised plateau tolerance at the "
                    "observed non-zero reconstruction floor"
                    if state_size == geometry.native_state_size
                    else "0.1% compressed-state supervised plateau tolerance"
                ),
                "AAE cap barrier log width 0.10 and log standard deviation 0.35",
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
        env = BaseEnvironment(precision=32, device="cuda:0", debug=False)
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
        torch.cuda.manual_seed(supervised_config.seed)
        model = NativeRbfQuoteEncoder(
            env,
            config=encoder_config,
            native_mean=cast(torch.Tensor, initialization["native_mean"]),
            native_scale=cast(torch.Tensor, initialization["native_scale"]),
            state_mean=cast(torch.Tensor, initialization["state_mean"]),
            state_scale=cast(torch.Tensor, initialization["state_scale"]),
            pca_components=cast(
                torch.Tensor | None,
                initialization["pca_components"],
            ),
        )
        decoder = FlexSanosDecoder(env, geometry=geometry)
        report["initialization"] = {
            "method": (
                "identity standardized native state"
                if state_size == geometry.native_state_size
                else "exact train-only PCA of standardized native raw state"
            ),
            "retained_standardized_native_variance": float(
                initialization["retained_standardized_native_variance"]
            ),
        }
        report["status"] = "supervised_training"
        write_json_atomic(report, report_path)

        supervised_history: list[dict[str, float]] = []
        market_history: list[dict[str, float]] = []
        audits: dict[str, dict[str, object]] = {}
        phase: str | None = None
        resume_state: dict[str, object] | None = None
        if checkpoint_path.exists():
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if payload.get("schema_version") != _CHECKPOINT_SCHEMA:
                raise ValueError("Unsupported native encoder checkpoint schema")
            if payload.get("model_identity") != _model_identity(model):
                raise ValueError("Native encoder checkpoint model identity mismatch")
            if payload.get("train_market_fingerprints") != [
                day.training_fingerprint for day in train_days
            ]:
                raise ValueError("Native encoder checkpoint universe mismatch")
            if payload.get("target_sha256") != target_sha256:
                raise ValueError("Native encoder checkpoint target mismatch")
            if payload.get("source_checkpoint_sha256") != source_checkpoint_sha256:
                raise ValueError("Native encoder checkpoint source mismatch")
            saved_statistics = RbfFeatureStatistics.from_dict(
                cast(dict[str, object], payload["feature_statistics"])
            )
            if any(
                not np.array_equal(
                    getattr(saved_statistics, name),
                    getattr(feature_statistics, name),
                )
                for name in (
                    "quote_mean",
                    "quote_scale",
                    "summary_mean",
                    "summary_scale",
                )
            ):
                raise ValueError("Native encoder feature normalizer mismatch")
            model.load_state_dict(
                cast(dict[str, torch.Tensor], payload["model_state_dict"])
            )
            phase = str(payload.get("phase"))
            if phase not in {
                "supervised",
                "supervised_complete",
                "market",
                "complete",
            }:
                raise ValueError(f"Unknown native encoder phase {phase!r}")
            supervised_history = cast(
                list[dict[str, float]], payload["supervised_history"]
            )
            market_history = cast(list[dict[str, float]], payload["market_history"])
            audits = cast(dict[str, dict[str, object]], payload["training_audits"])
            resume_state = cast(
                dict[str, object] | None,
                payload.get("training_state"),
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
                source_checkpoint_sha256=source_checkpoint_sha256,
                feature_statistics=feature_statistics,
                supervised_history=(
                    history if active_phase == "supervised" else supervised_history
                ),
                market_history=(
                    history if active_phase == "market" else market_history
                ),
                audits=audits,
                training_state=training_state,
            )

        if phase in {None, "supervised"}:
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
                source_checkpoint_sha256=source_checkpoint_sha256,
                feature_statistics=feature_statistics,
                supervised_history=supervised_history,
                market_history=[],
                audits=audits,
                training_state=None,
            )
            phase = "supervised_complete"
            resume_state = None
        elif phase in {"supervised_complete", "market", "complete"}:
            if audits.get("supervised", {}).get("converged") is not True:
                raise ValueError("Native encoder lacks supervised convergence proof")

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
                raise ValueError("Complete native encoder lacks market convergence proof")
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
        state_variation = {
            "train_mean_coordinate_std": float(
                torch.mean(torch.std(train_state, dim=0, unbiased=False)).cpu()
            ),
            "validation_mean_coordinate_std": float(
                torch.mean(torch.std(validation_state, dim=0, unbiased=False)).cpu()
            ),
        }
        if not all(
            np.isfinite(value) and value > 0.0
            for value in state_variation.values()
        ):
            raise RuntimeError("Native encoder state variation collapsed")
        _save_checkpoint(
            model,
            checkpoint_path,
            phase="complete",
            train_days=train_days,
            target_sha256=target_sha256,
            source_checkpoint_sha256=source_checkpoint_sha256,
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
                "independently calibrated per-day native state; this is a "
                "SANOS geometry ceiling, not an out-of-sample encoder"
            ),
            "evaluation": validation_native_ceiling,
        }
        report["state_variation"] = state_variation
        report["checkpoint"] = str(checkpoint_path)
        report["checkpoint_sha256"] = checkpoint_digest(checkpoint_path)
        report["status"] = "complete"
        write_json_atomic(report, report_path)
        _write_markdown(
            report,
            report_directory / f"phase3-native-encoder-{state_size}.md",
        )
        _write_comparison(report_directory)
        return report


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument(
        "--state-size",
        type=int,
        choices=_SUPPORTED_STATE_SIZES,
        required=True,
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
        default=_TRAINING_ROOT / "flex-phase3" / "market-native-encoder-v1",
    )
    parser.add_argument(
        "--report-directory",
        type=Path,
        default=_FLEX_SOURCE_DIRECTORY / "reports" / "phase3-native-encoder",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    arguments = _parse_arguments()
    run_phase3_native_encoder(
        ticker=arguments.ticker,
        state_size=arguments.state_size,
        workers=arguments.workers,
        native_checkpoint=arguments.native_checkpoint,
        market_dates_report=arguments.market_dates_report,
        checkpoint_directory=arguments.checkpoint_directory,
        report_directory=arguments.report_directory,
    )


if __name__ == "__main__":
    run_guarded_cli(main, program="run_phase3_native_encoder")
