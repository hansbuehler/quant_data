"""Joint affine Direct SANOS factors and stationary weekly dynamics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.dysanos.flex.config import FlexObjectiveConfig
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
)
from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs
from quantdata.torch import BaseEnvironment


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class JointDirectTrainConfig:
    """Operational schedule for alternating quote and dynamics updates."""

    num_epochs: int = 300
    day_batch_size: int = 16
    state_learning_rate: float = 1.0e-4
    decoder_learning_rate: float = 3.0e-5
    dynamics_learning_rate: float = 3.0e-4
    dynamics_weight: float = 0.10
    gauge_weight: float = 0.10
    cap_penalty_weight: float = 1.0
    dynamics_steps_per_epoch: int = 1
    warmup_epochs: int = 20
    minimum_learning_rate_fraction: float = 0.10
    dynamics_ramp_epochs: int = 50
    gradient_clip: float = 10.0
    seed: int = 8263
    checkpoint_every: int = 5
    report_every: int = 5
    convergence_window: int = 30
    convergence_relative_tolerance: float = 2.0e-3
    maximum_quote_containment_ratio_drop: float = 0.02
    compile_sanos_solver: bool = True

    def __post_init__(self) -> None:
        verify_inp(self.num_epochs > 0, lambda: "'num_epochs' must be positive")
        verify_inp(
            self.day_batch_size > 0,
            lambda: "'day_batch_size' must be positive",
        )
        verify_inp(
            min(
                self.state_learning_rate,
                self.decoder_learning_rate,
                self.dynamics_learning_rate,
            )
            > 0.0,
            lambda: "Every joint learning rate must be positive",
        )
        verify_inp(
            self.dynamics_weight > 0.0
            and self.gauge_weight > 0.0
            and self.cap_penalty_weight >= 0.0,
            lambda: "Joint loss weights must be positive (cap may be zero)",
        )
        verify_inp(
            self.dynamics_steps_per_epoch > 0,
            lambda: "'dynamics_steps_per_epoch' must be positive",
        )
        verify_inp(
            0 <= self.warmup_epochs < self.num_epochs
            and 0 <= self.dynamics_ramp_epochs <= self.num_epochs,
            lambda: "Invalid joint warm-up or dynamics-ramp horizon",
        )
        verify_inp(
            0.0 < self.minimum_learning_rate_fraction <= 1.0,
            lambda: "'minimum_learning_rate_fraction' must lie in (0,1]",
        )
        verify_inp(
            self.gradient_clip > 0.0,
            lambda: "'gradient_clip' must be positive",
        )
        verify_inp(
            self.checkpoint_every > 0
            and self.report_every > 0
            and self.convergence_window > 1,
            lambda: "Checkpoint/report/convergence intervals must be positive",
        )
        verify_inp(
            0.0 < self.convergence_relative_tolerance < 1.0,
            lambda: "Invalid joint convergence tolerance",
        )
        verify_inp(
            0.0 <= self.maximum_quote_containment_ratio_drop < 1.0,
            lambda: "Invalid quote-containment degradation limit",
        )


@dataclass(frozen=True)
class JointInitialization:
    """Exact affine re-gauging of an existing Direct decoder."""

    states: torch.Tensor
    decoder_weight: torch.Tensor
    decoder_bias: torch.Tensor
    maximum_raw_reconstruction_error: float
    minimum_factor_scale: float
    maximum_factor_scale: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardize_affine_direct_model(
    source: LatentStateSanosModel,
) -> JointInitialization:
    """Whiten Direct factors while preserving every decoded raw state."""
    if source.config.decoder_network.num_depth != 0:
        raise ValueError("Joint Direct initialization requires an affine decoder")
    states = source.daily_states.detach().cpu().double()
    weight = source.raw_decoder.I0.weight.detach().cpu().double()
    bias = cast(torch.Tensor, source.raw_decoder.I0.bias).detach().cpu().double()
    mean = torch.mean(states, dim=0)
    centered = states - mean
    _, singular_values, right = torch.linalg.svd(centered, full_matrices=False)
    scales = singular_values / math.sqrt(len(states))
    scale_floor = math.sqrt(torch.finfo(scales.dtype).eps)
    if bool(torch.any(scales <= scale_floor)):
        raise ValueError(
            "Direct factors are rank deficient and cannot be exactly whitened: "
            f"minimum scale={float(torch.min(scales)):.6g}"
        )
    factors = (centered @ right.transpose(0, 1)) / scales
    transformed_weight = weight @ right.transpose(0, 1) @ torch.diag(scales)
    transformed_bias = bias + weight @ mean
    original_raw = states @ weight.transpose(0, 1) + bias
    transformed_raw = (
        factors @ transformed_weight.transpose(0, 1) + transformed_bias
    )
    maximum_error = float(torch.max(torch.abs(original_raw - transformed_raw)))
    tolerance = 1.0e-9 * max(1.0, float(torch.max(torch.abs(original_raw))))
    if maximum_error > tolerance:
        raise RuntimeError(
            "Affine Direct whitening changed decoded native states: "
            f"maximum error={maximum_error:.6g}, tolerance={tolerance:.6g}"
        )
    return JointInitialization(
        states=factors,
        decoder_weight=transformed_weight,
        decoder_bias=transformed_bias,
        maximum_raw_reconstruction_error=maximum_error,
        minimum_factor_scale=float(torch.min(scales)),
        maximum_factor_scale=float(torch.max(scales)),
    )


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def initialize_joint_model(
    model: JointDirectDynamicsModel,
    initialization: JointInitialization,
    *,
    origin_indices: np.ndarray,
    endpoint_indices: np.ndarray,
) -> dict[str, float | int]:
    """Install the exact decoder gauge and a real stable dynamics warm start."""
    states = initialization.states.numpy()
    origin = states[origin_indices]
    endpoint = states[endpoint_indices]
    design = np.concatenate(
        (np.ones((len(origin), 1), dtype=np.float64), origin), axis=1
    )
    coefficients = np.linalg.lstsq(design, endpoint, rcond=None)[0]
    discrete_transition = coefficients[1:]

    # I-B is the real first-order logarithm approximation. Project only its
    # symmetric part to the dissipative cone; retain the fitted skew component.
    approximate_generator = np.eye(model.config.state_size) - discrete_transition
    symmetric = 0.5 * (approximate_generator + approximate_generator.T)
    skew = 0.5 * (approximate_generator - approximate_generator.T)
    values, vectors = np.linalg.eigh(symmetric)
    minimum_rate = model.config.minimum_mean_reversion_per_week
    projected_values = np.maximum(values, minimum_rate + 1.0e-5)
    projected_symmetric = (vectors * projected_values[None, :]) @ vectors.T
    factor = np.linalg.cholesky(
        projected_symmetric - minimum_rate * np.eye(model.config.state_size)
    )
    generator = projected_symmetric + skew
    from scipy.linalg import expm

    transition = expm(-generator)
    equilibrium = np.zeros(model.config.state_size, dtype=np.float64)
    residual = endpoint - (
        equilibrium + (origin - equilibrium) @ transition
    )
    covariance = np.cov(residual, rowvar=False, bias=True)
    covariance += 1.0e-6 * np.eye(model.config.state_size)
    innovation_factor = np.linalg.cholesky(covariance)

    with torch.no_grad():
        model.daily_states.copy_(
            initialization.states.to(dtype=model.fdtype, device=model.device)
        )
        model.raw_decoder.I0.weight.copy_(
            initialization.decoder_weight.to(
                dtype=model.fdtype, device=model.device
            )
        )
        cast(torch.Tensor, model.raw_decoder.I0.bias).copy_(
            initialization.decoder_bias.to(dtype=model.fdtype, device=model.device)
        )
        model.damping_factor.copy_(
            torch.as_tensor(factor, dtype=model.fdtype, device=model.device)
        )
        model.rotation_raw.copy_(
            torch.as_tensor(skew, dtype=model.fdtype, device=model.device)
        )
        model.equilibrium_mean.zero_()
        innovation = torch.as_tensor(
            innovation_factor,
            dtype=model.fdtype,
            device=model.device,
        )
        raw_innovation = torch.tril(innovation, diagonal=-1)
        adjusted_diagonal = torch.clamp(
            torch.diagonal(innovation) - model.config.innovation_std_floor,
            min=torch.finfo(model.fdtype).eps,
        )
        raw_innovation += torch.diag(_inverse_softplus(adjusted_diagonal))
        model.innovation_cholesky_raw.copy_(raw_innovation)

    return {
        "num_weekly_pairs": len(origin_indices),
        "unrestricted_weekly_spectral_radius": float(
            np.max(np.abs(np.linalg.eigvals(discrete_transition)))
        ),
        "projected_generator_minimum_real_eigenvalue": float(
            np.min(np.linalg.eigvals(generator).real)
        ),
        "projected_weekly_spectral_radius": float(
            np.max(np.abs(np.linalg.eigvals(transition)))
        ),
        "symmetric_projection_frobenius": float(
            np.linalg.norm(projected_symmetric - symmetric)
        ),
    }


def joint_dynamics_loss(
    model: JointDirectDynamicsModel,
    origin_indices: torch.Tensor,
    endpoint_indices: torch.Tensor,
) -> torch.Tensor:
    """Return exact-week Gaussian transition NLL per state dimension."""
    states = model.daily_states
    transition = model.weekly_transition()
    mean = model.equilibrium_mean
    prediction = mean + (states[origin_indices] - mean) @ transition
    residual = states[endpoint_indices] - prediction
    factor = model.innovation_cholesky()
    white = torch.linalg.solve_triangular(
        factor,
        residual.transpose(0, 1),
        upper=False,
    ).transpose(0, 1)
    dimension = model.config.state_size
    return (
        0.5
        * (
            torch.mean(torch.sum(white.square(), dim=1))
            + 2.0 * torch.sum(torch.log(torch.diagonal(factor)))
            + dimension * math.log(2.0 * math.pi)
        )
        / dimension
    )


def factor_gauge_penalty(states: torch.Tensor) -> torch.Tensor:
    """Fix the affine factor gauge at zero mean and unit population covariance."""
    centered = states - torch.mean(states, dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered / states.shape[0]
    identity = torch.eye(
        states.shape[1], dtype=states.dtype, device=states.device
    )
    return torch.mean(torch.mean(states, dim=0).square()) + (
        torch.sum((covariance - identity).square()) / states.shape[1]
    )


def _learning_rate_multiplier(epoch: int, config: JointDirectTrainConfig) -> float:
    if epoch < config.warmup_epochs:
        return (epoch + 1) / max(config.warmup_epochs, 1)
    progress = (epoch - config.warmup_epochs) / max(
        config.num_epochs - config.warmup_epochs - 1,
        1,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    floor = config.minimum_learning_rate_fraction
    return floor + (1.0 - floor) * cosine


def _convergence_audit(
    history: list[dict[str, float]],
    *,
    config: JointDirectTrainConfig,
) -> dict[str, object]:
    # Before the dynamics ramp reaches one, ``joint_objective`` values use
    # different loss weights and therefore are not comparable.  In particular,
    # the almost quote-only first epoch can be lower than every fully weighted
    # epoch even when the fixed-weight objective has converged.  Audit only the
    # constant-weight phase.
    eligible = [
        item
        for item in history
        if float(item.get("dynamics_ramp", 1.0)) >= 1.0 - 1.0e-12
    ]
    values = np.asarray(
        [item["joint_objective"] for item in eligible], dtype=np.float64
    )
    if len(values) <= config.convergence_window or not np.all(np.isfinite(values)):
        return {
            "converged": False,
            "reason": "insufficient_or_nonfinite_history",
            "num_epochs": len(history),
            "num_comparable_epochs": len(values),
        }
    start = float(values[-config.convergence_window - 1])
    final = float(values[-1])
    best = float(np.min(values))
    denominator = max(abs(start), abs(best), 1.0e-8)
    window_change = (start - final) / denominator
    final_gap = (final - best) / max(abs(best), 1.0e-8)
    converged = (
        abs(window_change) <= config.convergence_relative_tolerance
        and final_gap <= 2.0 * config.convergence_relative_tolerance
    )
    return {
        "converged": converged,
        "reason": "converged" if converged else "trailing_window_not_converged",
        "num_epochs": len(history),
        "num_comparable_epochs": len(values),
        "comparable_start_epoch": int(eligible[0]["epoch"]),
        "window": config.convergence_window,
        "window_relative_change": window_change,
        "final_relative_gap_to_best": final_gap,
        "best_epoch": int(eligible[int(np.argmin(values))]["epoch"]),
    }


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restart_train_config_identity(
    config: JointDirectTrainConfig | dict[str, object],
) -> dict[str, object]:
    """Exclude the operational batch size from checkpoint compatibility."""
    values = (
        asdict(config)
        if isinstance(config, JointDirectTrainConfig)
        else dict(config)
    )
    values.pop("day_batch_size", None)
    return values


def _save_checkpoint(
    model: JointDirectDynamicsModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    history: list[dict[str, float]],
    path: Path,
    *,
    train_config: JointDirectTrainConfig,
    market_fingerprints: tuple[str, ...],
    source_decoder_sha256: str,
    phase: str,
) -> None:
    payload: dict[str, object] = {
        "schema": "flex-joint-direct-dynamics-v1",
        "phase": phase,
        "model_config": asdict(model.config),
        "model_config_version": model.config.version.unique_id48,
        "model_user_version": model.config.user_version,
        "train_config": asdict(train_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "completed_epochs": len(history),
        "history": history,
        "market_fingerprints": list(market_fingerprints),
        "source_decoder_sha256": source_decoder_sha256,
        "torch_rng_state": torch.random.get_rng_state(),
    }
    selected_device = str(model.device)
    if selected_device.startswith("cuda"):
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(
            selected_device
        ).cpu()
    _atomic_torch_save(payload, path)


def _restore_checkpoint(
    model: JointDirectDynamicsModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    path: Path,
    *,
    train_config: JointDirectTrainConfig,
    market_fingerprints: tuple[str, ...],
    source_decoder_sha256: str,
) -> tuple[int, list[dict[str, float]], str] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location=model.device, weights_only=False)
    if payload.get("schema") != "flex-joint-direct-dynamics-v1":
        raise ValueError(f"Unsupported joint Direct checkpoint in {path}")
    if payload.get("model_config_version") != model.config.version.unique_id48 or (
        payload.get("model_user_version") != model.config.user_version
    ):
        raise ValueError(
            "Joint Direct checkpoint model version changed; cold start required"
        )
    if payload.get("model_config") != asdict(model.config):
        raise ValueError("Joint Direct checkpoint model configuration mismatch")
    saved_train_config = payload.get("train_config")
    if not isinstance(saved_train_config, dict) or _restart_train_config_identity(
        saved_train_config
    ) != _restart_train_config_identity(train_config):
        raise ValueError("Joint Direct checkpoint training configuration mismatch")
    if tuple(payload.get("market_fingerprints", ())) != market_fingerprints:
        raise ValueError("Joint Direct checkpoint market identity mismatch")
    if payload.get("source_decoder_sha256") != source_decoder_sha256:
        raise ValueError("Joint Direct source decoder checkpoint changed")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    torch.random.set_rng_state(payload["torch_rng_state"].cpu())
    selected_device = str(model.device)
    if selected_device.startswith("cuda"):
        torch.cuda.set_rng_state(
            payload["cuda_rng_state"].cpu(), device=selected_device
        )
    history = [
        {str(key): float(value) for key, value in item.items()}
        for item in payload.get("history", [])
    ]
    return int(payload["completed_epochs"]), history, str(payload["phase"])


def fit_joint_direct_model(
    model: JointDirectDynamicsModel,
    market_days: tuple[FlexMarketDay, ...],
    *,
    dates: tuple[date, ...],
    objective_config: FlexObjectiveConfig,
    train_config: JointDirectTrainConfig,
    checkpoint_path: Path,
    source_decoder_sha256: str,
    env: BaseEnvironment,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Alternate equal-date quote steps with exact-week dynamics steps."""
    if dates != tuple(day.trading_date for day in market_days):
        raise ValueError("Joint training dates and markets are not aligned")
    origin_np, endpoint_np = exact_weekly_pairs(dates, weeks=1)
    origin = torch.as_tensor(origin_np, dtype=torch.long, device=model.device)
    endpoint = torch.as_tensor(endpoint_np, dtype=torch.long, device=model.device)
    torch_days = tuple(materialize_day(day, env=env) for day in market_days)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [model.daily_states],
                "lr": train_config.state_learning_rate,
            },
            {
                "params": list(model.raw_decoder.parameters()),
                "lr": train_config.decoder_learning_rate,
            },
            {
                "params": [
                    model.damping_factor,
                    model.rotation_raw,
                    model.equilibrium_mean,
                    model.innovation_cholesky_raw,
                ],
                "lr": train_config.dynamics_learning_rate,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _learning_rate_multiplier(epoch, train_config),
    )
    market_fingerprints = tuple(day.training_fingerprint for day in market_days)
    restored = _restore_checkpoint(
        model,
        optimizer,
        scheduler,
        checkpoint_path,
        train_config=train_config,
        market_fingerprints=market_fingerprints,
        source_decoder_sha256=source_decoder_sha256,
    )
    if restored is None:
        start_epoch = 0
        history: list[dict[str, float]] = []
    else:
        start_epoch, history, phase = restored
        if phase == "complete":
            return history, _convergence_audit(history, config=train_config)
        if phase != "training":
            raise ValueError(f"Unexpected joint checkpoint phase {phase!r}")

    rng = np.random.default_rng(train_config.seed)
    # The only NumPy draw is one date permutation per completed epoch. Advance
    # deterministically so a checkpoint resume uses the exact next permutation
    # without serializing a Python-specific NumPy generator object.
    for _ in range(start_epoch):
        rng.permutation(len(torch_days))
    if train_config.compile_sanos_solver:
        model.sanos_decoder.enable_solver_compile()
    started = perf_counter()
    for epoch in range(start_epoch, train_config.num_epochs):
        model.train()
        permutation = rng.permutation(len(torch_days))
        quote_total = 0.0
        cap_total = 0.0
        batch_count = 0
        for start in range(0, len(torch_days), train_config.day_batch_size):
            indices = permutation[start : start + train_config.day_batch_size]
            batch = materialize_batch(torch_days, indices, env=env)
            optimizer.zero_grad(set_to_none=True)
            prices = _model_prices(model, batch)
            quote = equal_date_quote_objective(
                prices,
                batch,
                objective_config=objective_config,
            )
            cap = model_dlvsq_cap_penalty(model, batch.day_indices)
            quote_loss = quote + train_config.cap_penalty_weight * cap
            if not bool(torch.isfinite(quote_loss)):
                raise TrainingConvergenceError(
                    f"Non-finite quote loss at joint epoch {epoch + 1}"
                )
            quote_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [model.daily_states, *model.raw_decoder.parameters()],
                train_config.gradient_clip,
            )
            optimizer.step()
            quote_total += float(quote.detach().cpu())
            cap_total += float(cap.detach().cpu())
            batch_count += 1

        dynamics_ramp = min(
            1.0,
            (epoch + 1) / max(train_config.dynamics_ramp_epochs, 1),
        )
        dynamics_value = math.nan
        gauge_value = math.nan
        for _ in range(train_config.dynamics_steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            dynamics = joint_dynamics_loss(model, origin, endpoint)
            gauge = factor_gauge_penalty(model.daily_states)
            dynamics_objective = (
                dynamics_ramp * train_config.dynamics_weight * dynamics
                + train_config.gauge_weight * gauge
            )
            if not bool(torch.isfinite(dynamics_objective)):
                raise TrainingConvergenceError(
                    f"Non-finite dynamics loss at joint epoch {epoch + 1}"
                )
            dynamics_objective.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    model.daily_states,
                    model.damping_factor,
                    model.rotation_raw,
                    model.equilibrium_mean,
                    model.innovation_cholesky_raw,
                ],
                train_config.gradient_clip,
            )
            optimizer.step()
            dynamics_value = float(dynamics.detach().cpu())
            gauge_value = float(gauge.detach().cpu())
        scheduler.step()

        quote_value = quote_total / batch_count
        cap_value = cap_total / batch_count
        joint_value = (
            quote_value
            + train_config.cap_penalty_weight * cap_value
            + dynamics_ramp * train_config.dynamics_weight * dynamics_value
            + train_config.gauge_weight * gauge_value
        )
        item = {
            "epoch": float(epoch + 1),
            "quote_objective": quote_value,
            "dlvsq_cap_penalty": cap_value,
            "dynamics_nll_per_dimension": dynamics_value,
            "gauge_penalty": gauge_value,
            "dynamics_ramp": dynamics_ramp,
            "joint_objective": joint_value,
            "state_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "decoder_learning_rate": float(optimizer.param_groups[1]["lr"]),
            "dynamics_learning_rate": float(optimizer.param_groups[2]["lr"]),
        }
        history.append(item)
        if (epoch + 1) % train_config.report_every == 0 or epoch == start_epoch:
            elapsed = perf_counter() - started
            print(
                "joint-direct "
                f"epoch={epoch + 1}/{train_config.num_epochs} "
                f"quote={quote_value:.7g} dynamics={dynamics_value:.7g} "
                f"gauge={gauge_value:.7g} total={joint_value:.7g} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if (epoch + 1) % train_config.checkpoint_every == 0:
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                history,
                checkpoint_path,
                train_config=train_config,
                market_fingerprints=market_fingerprints,
                source_decoder_sha256=source_decoder_sha256,
                phase="training",
            )

    if train_config.compile_sanos_solver:
        model.sanos_decoder.disable_solver_compile()
    audit = _convergence_audit(history, config=train_config)
    if audit.get("converged") is not True:
        _save_checkpoint(
            model,
            optimizer,
            scheduler,
            history,
            checkpoint_path,
            train_config=train_config,
            market_fingerprints=market_fingerprints,
            source_decoder_sha256=source_decoder_sha256,
            phase="training",
        )
        raise TrainingConvergenceError(
            f"Joint Direct training failed convergence: {json.dumps(audit)}"
        )
    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        history,
        checkpoint_path,
        train_config=train_config,
        market_fingerprints=market_fingerprints,
        source_decoder_sha256=source_decoder_sha256,
        phase="complete",
    )
    return history, audit


def project_native_raw_to_joint_states(
    model: JointDirectDynamicsModel,
    native_raw: torch.Tensor,
) -> torch.Tensor:
    """Return least-squares factor projections for an evaluation-only surface."""
    weight = model.raw_decoder.I0.weight.detach().cpu().double()
    bias = cast(torch.Tensor, model.raw_decoder.I0.bias).detach().cpu().double()
    centered = native_raw.detach().cpu().double() - bias
    return torch.linalg.lstsq(
        weight,
        centered.transpose(0, 1),
    ).solution.transpose(0, 1)


def evaluate_projected_market_days(
    model: JointDirectDynamicsModel,
    states: torch.Tensor,
    market_days: tuple[FlexMarketDay, ...],
    *,
    objective_config: FlexObjectiveConfig,
    day_batch_size: int,
    env: BaseEnvironment,
) -> FitEvaluation:
    """Evaluate a fixed state matrix without mutating the trained checkpoint."""
    evaluation_model = JointDirectDynamicsModel(
        env,
        num_days=len(market_days),
        config=model.config,
    )
    state_dict = model.state_dict()
    copied = evaluation_model.state_dict()
    for key, value in state_dict.items():
        if key != "daily_state_embedding.weight":
            copied[key] = value
    copied["daily_state_embedding.weight"] = states.to(
        dtype=env.fdtype, device=env.device
    )
    evaluation_model.load_state_dict(copied)
    torch_days = tuple(materialize_day(day, env=env) for day in market_days)
    return evaluate_model(
        evaluation_model,
        torch_days,
        env=env,
        objective_config=objective_config,
        day_batch_size=day_batch_size,
        equal_date_weighting=True,
    )


def joint_dynamics_diagnostics(
    model: JointDirectDynamicsModel,
    train_states: np.ndarray,
    validation_states: np.ndarray,
    validation_dates: tuple[date, ...],
) -> dict[str, object]:
    """Report stability and oracle-projected 2025 state forecasts."""
    with torch.no_grad():
        generator = model.mean_reversion_generator().detach().cpu().double().numpy()
        transition = model.weekly_transition().detach().cpu().double().numpy()
        factor = model.innovation_cholesky().detach().cpu().double().numpy()
        equilibrium = model.equilibrium_mean.detach().cpu().double().numpy()
    covariance = factor @ factor.T
    eigenvalues = np.linalg.eigvals(generator)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if not np.all(np.isfinite(eigenvalues)) or np.min(eigenvalues.real) <= 0.0:
        raise RuntimeError("Joint Direct generator lost stationarity")
    if spectral_radius >= 1.0:
        raise RuntimeError("Joint Direct weekly transition is not stationary")
    powers: dict[str, float] = {}
    power = np.eye(len(transition))
    maximum_power_norm = 0.0
    maximum_power_week = 0
    for week in range(1, 1041):
        power = power @ transition
        norm = float(np.linalg.norm(power, ord=2))
        if norm > maximum_power_norm:
            maximum_power_norm = norm
            maximum_power_week = week
        if week in {1, 2, 4, 52, 260, 520, 1040}:
            powers[str(week)] = norm
    horizons: dict[str, object] = {}
    unconditional_mean = np.mean(train_states, axis=0)
    unconditional_rmse_scale = np.std(train_states, axis=0)
    unconditional_rmse_scale = np.maximum(unconditional_rmse_scale, 1.0e-8)
    for weeks in (1, 2, 4):
        origin, endpoint = exact_weekly_pairs(validation_dates, weeks=weeks)
        horizon_transition = np.linalg.matrix_power(transition, weeks)
        horizon_covariance = np.zeros_like(covariance)
        propagation = np.eye(len(transition))
        for _ in range(weeks):
            horizon_covariance += propagation.T @ covariance @ propagation
            propagation = propagation @ transition
        prediction = equilibrium + (
            validation_states[origin] - equilibrium
        ) @ horizon_transition
        realized = validation_states[endpoint]
        normalized_error = (
            prediction - realized
        ) / unconditional_rmse_scale
        persistence_error = (
            validation_states[origin] - realized
        ) / unconditional_rmse_scale
        unconditional_error = (
            unconditional_mean - realized
        ) / unconditional_rmse_scale
        sign, logdet = np.linalg.slogdet(horizon_covariance)
        if sign <= 0.0:
            raise RuntimeError("Propagated joint Direct covariance is not positive")
        residual = realized - prediction
        mahalanobis = np.sum(
            residual
            * np.linalg.solve(horizon_covariance, residual.T).T,
            axis=1,
        )
        nll_per_dimension = float(
            np.mean(
                0.5
                * (
                    mahalanobis
                    + logdet
                    + len(transition) * math.log(2.0 * math.pi)
                )
            )
            / len(transition)
        )
        horizons[f"{weeks}w"] = {
            "num_pairs": len(origin),
            "state_rmse": float(np.sqrt(np.mean(np.square(normalized_error)))),
            "persistence_state_rmse": float(
                np.sqrt(np.mean(np.square(persistence_error)))
            ),
            "unconditional_state_rmse": float(
                np.sqrt(np.mean(np.square(unconditional_error)))
            ),
            "negative_log_likelihood_per_dimension": nll_per_dimension,
            "mahalanobis_mean_per_dimension": float(
                np.mean(mahalanobis) / len(transition)
            ),
        }
    return {
        "generator_minimum_real_eigenvalue": float(np.min(eigenvalues.real)),
        "generator_maximum_real_eigenvalue": float(np.max(eigenvalues.real)),
        "generator_maximum_imaginary_abs": float(
            np.max(np.abs(eigenvalues.imag))
        ),
        "weekly_spectral_radius": spectral_radius,
        "maximum_transition_power_norm": maximum_power_norm,
        "maximum_transition_power_week": maximum_power_week,
        "transition_power_norms": powers,
        "transition_condition_number": float(np.linalg.cond(transition)),
        "horizons": horizons,
    }


def compact_evaluation(evaluation: FitEvaluation) -> dict[str, object]:
    """Return the standard quote diagnostics without retaining fitted vectors."""
    return evaluation.to_dict()


__all__ = [
    "JointDirectTrainConfig",
    "JointInitialization",
    "compact_evaluation",
    "evaluate_projected_market_days",
    "factor_gauge_penalty",
    "fit_joint_direct_model",
    "initialize_joint_model",
    "joint_dynamics_diagnostics",
    "joint_dynamics_loss",
    "project_native_raw_to_joint_states",
    "standardize_affine_direct_model",
]
