"""Quote-optimal 2025 capacity calibration for a frozen Direct decoder."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.dysanos.flex.config import (
    FlexObjectiveConfig,
    PcaDecoderConfig,
)
from quantdata.dysanos.flex.market import FlexMarketDay
from quantdata.dysanos.flex.model import (
    JointDirectDynamicsModel,
    LatentStateSanosModel,
)
from quantdata.dysanos.flex.training import (
    FitEvaluation,
    TrainingConvergenceError,
    _model_prices,
    equal_date_quote_objective,
    evaluate_model,
    materialize_batch,
    materialize_day,
    model_dlvsq_cap_penalty,
    training_convergence_audit,
)
from quantdata.torch import BaseEnvironment


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class DirectValidationCapacityTrainConfig:
    """Operational schedule for independent 2025 factor calibration."""

    num_epochs: int = 1_000
    day_batch_size: int = 16
    learning_rate: float = 1.0e-3
    warmup_epochs: int = 10
    minimum_learning_rate_fraction: float = 0.10
    gradient_clip: float = 10.0
    cap_penalty_weight: float = 1.0
    seed: int = 7499
    checkpoint_every: int = 10
    report_every: int = 10
    convergence_window: int = 50
    convergence_relative_tolerance: float = 1.0e-3
    compile_sanos_solver: bool = True

    def __post_init__(self) -> None:
        verify_inp(self.num_epochs > 0, lambda: "'num_epochs' must be positive")
        verify_inp(
            self.day_batch_size > 0,
            lambda: "'day_batch_size' must be positive",
        )
        verify_inp(
            self.learning_rate > 0.0 and np.isfinite(self.learning_rate),
            lambda: "'learning_rate' must be finite and positive",
        )
        verify_inp(
            0 <= self.warmup_epochs < self.num_epochs,
            lambda: "'warmup_epochs' must lie in [0,num_epochs)",
        )
        verify_inp(
            0.0 < self.minimum_learning_rate_fraction <= 1.0,
            lambda: "'minimum_learning_rate_fraction' must lie in (0,1]",
        )
        verify_inp(
            self.gradient_clip > 0.0 and self.cap_penalty_weight >= 0.0,
            lambda: "Invalid gradient clip or cap-penalty weight",
        )
        verify_inp(
            self.checkpoint_every > 0
            and self.report_every > 0
            and self.convergence_window > 1,
            lambda: "Checkpoint/report/convergence intervals must be positive",
        )
        verify_inp(
            0.0 < self.convergence_relative_tolerance < 1.0,
            lambda: "Invalid convergence tolerance",
        )


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash an ordered tensor state without Torch serialization metadata."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def frozen_joint_state_sha256(model: JointDirectDynamicsModel) -> str:
    """Hash the source decoder, SANOS buffers, and dynamics, excluding dates."""
    return tensor_state_sha256(
        {
            name: value
            for name, value in model.state_dict().items()
            if name != "daily_state_embedding.weight"
        }
    )


def frozen_capacity_decoder_sha256(model: LatentStateSanosModel) -> str:
    """Hash every fixed tensor of the quote-capacity model."""
    return tensor_state_sha256(
        {
            name: value
            for name, value in model.state_dict().items()
            if name != "daily_state_embedding.weight"
        }
    )


def make_frozen_capacity_model(
    source: JointDirectDynamicsModel,
    initial_states: torch.Tensor,
    *,
    env: BaseEnvironment,
) -> LatentStateSanosModel:
    """Copy a joint Direct affine decoder into a state-only calibration model."""
    if initial_states.ndim != 2 or initial_states.shape[1] != source.config.state_size:
        raise ValueError(
            "Initial Direct factors have the wrong shape: "
            f"{tuple(initial_states.shape)}"
        )
    config = PcaDecoderConfig(
        label="shared-strikes-frozen-joint-direct-2025-capacity",
        state_size=source.config.state_size,
        geometry=source.config.geometry,
    )
    model = LatentStateSanosModel(
        env,
        num_days=len(initial_states),
        config=config,
    )
    with torch.no_grad():
        model.raw_decoder.load_state_dict(source.raw_decoder.state_dict())
        model.sanos_decoder.load_state_dict(source.sanos_decoder.state_dict())
        model.daily_states.copy_(
            initial_states.to(dtype=env.fdtype, device=env.device)
        )
    for parameter in model.raw_decoder.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.raw_decoder.parameters()):
        raise RuntimeError("The frozen Direct decoder still has trainable parameters")
    return model


def _learning_rate_multiplier(
    epoch: int,
    config: DirectValidationCapacityTrainConfig,
) -> float:
    epoch = min(max(epoch, 0), config.num_epochs - 1)
    floor = config.minimum_learning_rate_fraction
    if epoch < config.warmup_epochs:
        return floor + (1.0 - floor) * (epoch + 1) / max(
            config.warmup_epochs,
            1,
        )
    progress = (epoch - config.warmup_epochs) / max(
        config.num_epochs - config.warmup_epochs - 1,
        1,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (1.0 - floor) * cosine


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_payload(
    model: LatentStateSanosModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    history: list[dict[str, float]],
    *,
    phase: str,
    config: DirectValidationCapacityTrainConfig,
    objective_config: FlexObjectiveConfig,
    market_fingerprints: tuple[str, ...],
    source_joint_sha256: str,
    source_native_sha256: str,
    frozen_source_state_sha256: str,
    frozen_decoder_sha256: str,
    numpy_rng_state: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "flex-direct-2025-quote-capacity-v1",
        "phase": phase,
        "model_config": asdict(model.config),
        "model_config_version": model.config.version.unique_id48,
        "model_user_version": model.config.user_version,
        "train_config": asdict(config),
        "objective_config": asdict(objective_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "completed_epochs": len(history),
        "history": history,
        "market_fingerprints": list(market_fingerprints),
        "source_joint_sha256": source_joint_sha256,
        "source_native_sha256": source_native_sha256,
        "frozen_source_state_sha256": frozen_source_state_sha256,
        "frozen_decoder_sha256": frozen_decoder_sha256,
        "numpy_rng_state": numpy_rng_state,
        "torch_rng_state": torch.random.get_rng_state(),
    }
    selected_device = str(model.device)
    if selected_device.startswith("cuda"):
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(selected_device).cpu()
    return payload


def _restore_checkpoint(
    model: LatentStateSanosModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    path: Path,
    *,
    config: DirectValidationCapacityTrainConfig,
    objective_config: FlexObjectiveConfig,
    market_fingerprints: tuple[str, ...],
    source_joint_sha256: str,
    source_native_sha256: str,
    frozen_source_state_sha256: str,
    frozen_decoder_sha256: str,
) -> tuple[int, list[dict[str, float]], dict[str, object], str] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location=model.device, weights_only=False)
    if payload.get("schema") != "flex-direct-2025-quote-capacity-v1":
        raise ValueError(f"Unsupported validation-capacity checkpoint in {path}")
    expected = {
        "model_config": asdict(model.config),
        "model_config_version": model.config.version.unique_id48,
        "model_user_version": model.config.user_version,
        "train_config": asdict(config),
        "objective_config": asdict(objective_config),
        "market_fingerprints": list(market_fingerprints),
        "source_joint_sha256": source_joint_sha256,
        "source_native_sha256": source_native_sha256,
        "frozen_source_state_sha256": frozen_source_state_sha256,
        "frozen_decoder_sha256": frozen_decoder_sha256,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"Validation-capacity checkpoint {name} mismatch; "
                "cold start or explicit weights-only warm start required"
            )
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    torch.random.set_rng_state(payload["torch_rng_state"].cpu())
    selected_device = str(model.device)
    if selected_device.startswith("cuda"):
        torch.cuda.set_rng_state(
            payload["cuda_rng_state"].cpu(),
            device=selected_device,
        )
    if frozen_capacity_decoder_sha256(model) != frozen_decoder_sha256:
        raise ValueError("Frozen Direct decoder changed in the checkpoint")
    history = [
        {str(name): float(value) for name, value in item.items()}
        for item in cast(list[dict[str, float]], payload.get("history", []))
    ]
    return (
        int(payload["completed_epochs"]),
        history,
        cast(dict[str, object], payload["numpy_rng_state"]),
        str(payload["phase"]),
    )


def _capacity_convergence_audit(
    history: list[dict[str, float]],
    *,
    config: DirectValidationCapacityTrainConfig,
) -> dict[str, object]:
    # Reuse the established FLEX plateau definition with an operational config
    # carrying the same two convergence fields.
    return dict(
        training_convergence_audit(
            history,
            metric="objective",
            train_config=cast(object, config),
        )
    )


def fit_quote_optimal_validation_states(
    model: LatentStateSanosModel,
    market_days: tuple[FlexMarketDay, ...],
    *,
    source_joint_model: JointDirectDynamicsModel,
    config: DirectValidationCapacityTrainConfig,
    objective_config: FlexObjectiveConfig,
    checkpoint_path: Path,
    source_joint_sha256: str,
    source_native_sha256: str,
    env: BaseEnvironment,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Fit only one free 200-vector per 2025 date against that date's quotes."""
    if model.num_days != len(market_days):
        raise ValueError("Capacity model and validation-market counts differ")
    if any(parameter.requires_grad for parameter in model.raw_decoder.parameters()):
        raise ValueError("Quote-capacity fitting requires a frozen decoder")
    market_fingerprints = tuple(day.training_fingerprint for day in market_days)
    torch_days = tuple(materialize_day(day, env=env) for day in market_days)
    frozen_source = frozen_joint_state_sha256(source_joint_model)
    frozen_decoder = frozen_capacity_decoder_sha256(model)
    optimizer = torch.optim.SparseAdam(
        [model.daily_states],
        lr=config.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _learning_rate_multiplier(epoch, config),
    )
    restored = _restore_checkpoint(
        model,
        optimizer,
        scheduler,
        checkpoint_path,
        config=config,
        objective_config=objective_config,
        market_fingerprints=market_fingerprints,
        source_joint_sha256=source_joint_sha256,
        source_native_sha256=source_native_sha256,
        frozen_source_state_sha256=frozen_source,
        frozen_decoder_sha256=frozen_decoder,
    )
    rng = np.random.default_rng(config.seed)
    if restored is None:
        start_epoch = 0
        history: list[dict[str, float]] = []
    else:
        start_epoch, history, rng_state, phase = restored
        rng.bit_generator.state = rng_state
        if phase == "complete":
            return history, _capacity_convergence_audit(history, config=config)
        if phase != "training":
            raise ValueError(f"Unexpected validation-capacity phase {phase!r}")

    if config.compile_sanos_solver:
        model.sanos_decoder.enable_solver_compile()
    started = perf_counter()
    for epoch in range(start_epoch, config.num_epochs):
        order = rng.permutation(len(torch_days))
        objective_sum = 0.0
        quote_sum = 0.0
        cap_sum = 0.0
        for start in range(0, len(torch_days), config.day_batch_size):
            indices = order[start : start + config.day_batch_size]
            batch = materialize_batch(torch_days, indices, env=env)
            optimizer.zero_grad(set_to_none=True)
            prices = _model_prices(model, batch)
            quote = equal_date_quote_objective(
                prices,
                batch,
                objective_config=objective_config,
            )
            cap = model_dlvsq_cap_penalty(model, batch.day_indices)
            loss = quote + config.cap_penalty_weight * cap
            if not bool(torch.isfinite(loss)):
                raise TrainingConvergenceError(
                    f"Non-finite Direct validation loss at epoch {epoch + 1}"
                )
            loss.backward()
            gradient = model.daily_states.grad
            if gradient is None or not gradient.is_sparse:
                raise RuntimeError("Expected sparse gradients for validation states")
            gradient = gradient.coalesce()
            norm = torch.linalg.vector_norm(gradient.values())
            scale = torch.clamp(config.gradient_clip / (norm + 1.0e-12), max=1.0)
            gradient.values().mul_(scale)
            model.daily_states.grad = gradient
            optimizer.step()
            count = len(indices)
            objective_sum += float(loss.detach().cpu()) * count
            quote_sum += float(quote.detach().cpu()) * count
            cap_sum += float(cap.detach().cpu()) * count
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "objective": objective_sum / len(torch_days),
                "quote_objective": quote_sum / len(torch_days),
                "dlvsq_cap_penalty": cap_sum / len(torch_days),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if (epoch + 1) % config.report_every == 0 or epoch == start_epoch:
            latest = history[-1]
            print(
                "direct-2025-quote-capacity "
                f"epoch={epoch + 1}/{config.num_epochs} "
                f"objective={latest['objective']:.7g} "
                f"quote={latest['quote_objective']:.7g} "
                f"cap={latest['dlvsq_cap_penalty']:.7g} "
                f"lr={latest['learning_rate']:.7g} "
                f"elapsed={perf_counter() - started:.1f}s",
                flush=True,
            )
        if (epoch + 1) % config.checkpoint_every == 0:
            if frozen_joint_state_sha256(source_joint_model) != frozen_source:
                raise RuntimeError(
                    "Source joint decoder or dynamics changed during fitting"
                )
            if frozen_capacity_decoder_sha256(model) != frozen_decoder:
                raise RuntimeError("Frozen capacity decoder changed during fitting")
            _atomic_torch_save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    history,
                    phase="training",
                    config=config,
                    objective_config=objective_config,
                    market_fingerprints=market_fingerprints,
                    source_joint_sha256=source_joint_sha256,
                    source_native_sha256=source_native_sha256,
                    frozen_source_state_sha256=frozen_source,
                    frozen_decoder_sha256=frozen_decoder,
                    numpy_rng_state=cast(dict[str, object], rng.bit_generator.state),
                ),
                checkpoint_path,
            )

    if config.compile_sanos_solver:
        model.sanos_decoder.disable_solver_compile()
    if frozen_joint_state_sha256(source_joint_model) != frozen_source:
        raise RuntimeError("Source joint decoder or dynamics changed during fitting")
    if frozen_capacity_decoder_sha256(model) != frozen_decoder:
        raise RuntimeError("Frozen capacity decoder changed during fitting")
    audit = _capacity_convergence_audit(history, config=config)
    phase = "complete" if audit.get("converged") is True else "training"
    _atomic_torch_save(
        _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            history,
            phase=phase,
            config=config,
            objective_config=objective_config,
            market_fingerprints=market_fingerprints,
            source_joint_sha256=source_joint_sha256,
            source_native_sha256=source_native_sha256,
            frozen_source_state_sha256=frozen_source,
            frozen_decoder_sha256=frozen_decoder,
            numpy_rng_state=cast(dict[str, object], rng.bit_generator.state),
        ),
        checkpoint_path,
    )
    if phase != "complete":
        raise TrainingConvergenceError(
            "Direct 2025 quote-capacity calibration failed convergence: "
            f"{json.dumps(audit, sort_keys=True)}"
        )
    return history, audit


def evaluate_quote_capacity_model(
    model: LatentStateSanosModel,
    market_days: tuple[FlexMarketDay, ...],
    *,
    objective_config: FlexObjectiveConfig,
    day_batch_size: int,
    env: BaseEnvironment,
) -> FitEvaluation:
    """Evaluate every validation quote with equal date weighting."""
    torch_days = tuple(materialize_day(day, env=env) for day in market_days)
    return evaluate_model(
        model,
        torch_days,
        env=env,
        objective_config=objective_config,
        day_batch_size=day_batch_size,
        equal_date_weighting=True,
    )


__all__ = [
    "DirectValidationCapacityTrainConfig",
    "evaluate_quote_capacity_model",
    "fit_quote_optimal_validation_states",
    "frozen_capacity_decoder_sha256",
    "frozen_joint_state_sha256",
    "make_frozen_capacity_model",
    "tensor_state_sha256",
]
