"""Joint expiry-causal SANOS factors and stationary weekly P dynamics."""

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
from quantdata.dysanos.flex.joint_direct import (
    JointDirectTrainConfig,
    _convergence_audit,
    _inverse_softplus,
    _learning_rate_multiplier,
)
from quantdata.dysanos.flex.market import FlexMarketDay
from quantdata.dysanos.flex.model import (
    JointHierarchicalDynamicsModel,
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


@version("1.0", dependencies=[JointDirectTrainConfig])
@dataclass(frozen=True, kw_only=True)
class JointHierarchicalTrainConfig(JointDirectTrainConfig):
    """Operational schedule for joint hierarchical quote/dynamics training."""


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class QuoteOptimalStateTrainConfig:
    """Optimize independent date states against quotes with a frozen decoder."""

    num_epochs: int = 600
    day_batch_size: int = 16
    learning_rate: float = 1.0e-3
    cap_penalty_weight: float = 1.0
    warmup_epochs: int = 10
    minimum_learning_rate_fraction: float = 0.10
    gradient_clip: float = 10.0
    seed: int = 9109
    checkpoint_every: int = 10
    report_every: int = 10
    convergence_window: int = 50
    convergence_relative_tolerance: float = 2.0e-3
    compile_sanos_solver: bool = True

    def __post_init__(self) -> None:
        verify_inp(self.num_epochs > 0, lambda: "'num_epochs' must be positive")
        verify_inp(
            self.day_batch_size > 0 and self.learning_rate > 0.0,
            lambda: "State batch size and learning rate must be positive",
        )
        verify_inp(
            self.cap_penalty_weight >= 0.0,
            lambda: "'cap_penalty_weight' must be non-negative",
        )
        verify_inp(
            0 <= self.warmup_epochs < self.num_epochs,
            lambda: "Invalid state warm-up horizon",
        )
        verify_inp(
            0.0 < self.minimum_learning_rate_fraction <= 1.0,
            lambda: "Invalid state learning-rate floor",
        )
        verify_inp(
            self.gradient_clip > 0.0
            and self.checkpoint_every > 0
            and self.report_every > 0
            and self.convergence_window > 1,
            lambda: "Invalid state-training operational controls",
        )
        verify_inp(
            0.0 < self.convergence_relative_tolerance < 1.0,
            lambda: "Invalid state convergence tolerance",
        )


@version("1.0", dependencies=[QuoteOptimalStateTrainConfig])
@dataclass(frozen=True, kw_only=True)
class QuoteOptimalStateContinuationConfig(QuoteOptimalStateTrainConfig):
    """Explicit lower-rate continuation after a completed non-plateau run."""

    num_epochs: int = 400
    learning_rate: float = 1.0e-4
    warmup_epochs: int = 0
    minimum_learning_rate_fraction: float = 0.10
    seed: int = 12347


@dataclass(frozen=True)
class QuoteStateContinuationInitialization:
    """Validated state/provenance carried into a new optimizer phase."""

    states: torch.Tensor
    source_checkpoint_sha256: str
    source_completed_epochs: int
    source_convergence: dict[str, object]
    provenance: dict[str, object]


@dataclass(frozen=True)
class HierarchicalInitialization:
    """Exact blockwise re-gauging of an expiry-causal affine decoder."""

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


def standardize_affine_hierarchical_model(
    source: LatentStateSanosModel,
) -> HierarchicalInitialization:
    """Whiten each expiry block while preserving raw states and causality."""
    config = source.config
    state_sizes = getattr(config, "state_sizes_by_expiry", None)
    if state_sizes is None or config.decoder_network.num_depth != 0:
        raise ValueError(
            "Joint hierarchical initialization requires an affine triangular decoder"
        )
    states = source.daily_states.detach().cpu().double()
    weight = (
        source.raw_decoder.I0.weight.detach().cpu().double()
        * source.decoder_weight_mask.detach().cpu().double()
    )
    bias = cast(torch.Tensor, source.raw_decoder.I0.bias).detach().cpu().double()
    transformed_states = torch.empty_like(states)
    transformed_weight = torch.zeros_like(weight)
    transformed_bias = bias.clone()
    scales_seen: list[torch.Tensor] = []
    start = 0
    for width in state_sizes:
        stop = start + width
        block = states[:, start:stop]
        mean = torch.mean(block, dim=0)
        centered = block - mean
        _, singular_values, right = torch.linalg.svd(
            centered,
            full_matrices=False,
        )
        scales = singular_values / math.sqrt(len(states))
        scale_floor = math.sqrt(torch.finfo(scales.dtype).eps)
        if bool(torch.any(scales <= scale_floor)):
            raise ValueError(
                "Hierarchical factor block is rank deficient: "
                f"minimum scale={float(torch.min(scales)):.6g}"
            )
        transformed_states[:, start:stop] = (
            centered @ right.transpose(0, 1)
        ) / scales
        transformed_weight[:, start:stop] = (
            weight[:, start:stop]
            @ right.transpose(0, 1)
            @ torch.diag(scales)
        )
        transformed_bias += weight[:, start:stop] @ mean
        scales_seen.append(scales)
        start = stop

    original_raw = states @ weight.transpose(0, 1) + bias
    transformed_raw = (
        transformed_states @ transformed_weight.transpose(0, 1)
        + transformed_bias
    )
    maximum_error = float(torch.max(torch.abs(original_raw - transformed_raw)))
    tolerance = 1.0e-9 * max(1.0, float(torch.max(torch.abs(original_raw))))
    if maximum_error > tolerance:
        raise RuntimeError(
            "Hierarchical whitening changed decoded raw states: "
            f"maximum error={maximum_error:.6g}, tolerance={tolerance:.6g}"
        )
    all_scales = torch.cat(scales_seen)
    return HierarchicalInitialization(
        states=transformed_states,
        decoder_weight=transformed_weight,
        decoder_bias=transformed_bias,
        maximum_raw_reconstruction_error=maximum_error,
        minimum_factor_scale=float(torch.min(all_scales)),
        maximum_factor_scale=float(torch.max(all_scales)),
    )


def initialize_joint_hierarchical_model(
    model: JointHierarchicalDynamicsModel,
    initialization: HierarchicalInitialization,
    *,
    origin_indices: np.ndarray,
    endpoint_indices: np.ndarray,
) -> dict[str, float | int]:
    """Install an exact decoder gauge and stable causal dynamics warm start."""
    states = initialization.states.numpy()
    origin = states[origin_indices]
    endpoint = states[endpoint_indices]
    dimension = model.config.state_size
    discrete_transition = np.zeros((dimension, dimension), dtype=np.float64)
    state_slices = model.state_slices
    for state_slice in state_slices:
        prefix_stop = state_slice.stop
        design = np.concatenate(
            (
                np.ones((len(origin), 1), dtype=np.float64),
                origin[:, :prefix_stop],
            ),
            axis=1,
        )
        coefficients = np.linalg.lstsq(
            design,
            endpoint[:, state_slice],
            rcond=None,
        )[0]
        discrete_transition[:prefix_stop, state_slice] = coefficients[1:]

    approximate_generator = np.eye(dimension) - discrete_transition
    generator = np.zeros_like(approximate_generator)
    factors: list[np.ndarray] = []
    minimum_rate = model.config.minimum_mean_reversion_per_week
    projection_norm_sq = 0.0
    for state_slice in state_slices:
        diagonal = approximate_generator[state_slice, state_slice]
        symmetric = 0.5 * (diagonal + diagonal.T)
        skew = 0.5 * (diagonal - diagonal.T)
        values, vectors = np.linalg.eigh(symmetric)
        projected_values = np.maximum(values, minimum_rate + 1.0e-5)
        projected = (vectors * projected_values[None, :]) @ vectors.T
        factors.append(
            np.linalg.cholesky(
                projected - minimum_rate * np.eye(state_slice.stop - state_slice.start)
            )
        )
        generator[state_slice, state_slice] = projected + skew
        projection_norm_sq += float(np.sum(np.square(projected - symmetric)))
        generator[: state_slice.start, state_slice] = approximate_generator[
            : state_slice.start, state_slice
        ]

    from scipy.linalg import expm

    transition = expm(-generator)
    residual = endpoint - origin @ transition
    covariance = np.cov(residual, rowvar=False, bias=True)
    covariance += 1.0e-6 * np.eye(dimension)
    innovation_factor = np.linalg.cholesky(covariance)

    with torch.no_grad():
        model.daily_states.copy_(
            initialization.states.to(dtype=model.fdtype, device=model.device)
        )
        model.raw_decoder.I0.weight.copy_(
            initialization.decoder_weight.to(
                dtype=model.fdtype,
                device=model.device,
            )
        )
        cast(torch.Tensor, model.raw_decoder.I0.bias).copy_(
            initialization.decoder_bias.to(dtype=model.fdtype, device=model.device)
        )
        model.damping_factor.zero_()
        model.rotation_raw.zero_()
        model.causal_coupling_raw.zero_()
        for state_slice, factor in zip(state_slices, factors, strict=True):
            diagonal = generator[state_slice, state_slice]
            skew = 0.5 * (diagonal - diagonal.T)
            model.damping_factor[state_slice, state_slice].copy_(
                torch.as_tensor(factor, dtype=model.fdtype, device=model.device)
            )
            model.rotation_raw[state_slice, state_slice].copy_(
                torch.as_tensor(skew, dtype=model.fdtype, device=model.device)
            )
            model.causal_coupling_raw[: state_slice.start, state_slice].copy_(
                torch.as_tensor(
                    generator[: state_slice.start, state_slice],
                    dtype=model.fdtype,
                    device=model.device,
                )
            )
        model.equilibrium_mean.zero_()
        innovation = torch.as_tensor(
            innovation_factor,
            dtype=model.fdtype,
            device=model.device,
        )
        adjusted_diagonal = torch.clamp(
            torch.diagonal(innovation) - model.config.innovation_std_floor,
            min=torch.finfo(model.fdtype).eps,
        )
        model.innovation_cholesky_raw.copy_(
            torch.tril(innovation, diagonal=-1)
            + torch.diag(_inverse_softplus(adjusted_diagonal))
        )

    eigenvalues = np.linalg.eigvals(generator)
    return {
        "num_weekly_pairs": len(origin_indices),
        "unrestricted_weekly_spectral_radius": float(
            np.max(np.abs(np.linalg.eigvals(discrete_transition)))
        ),
        "projected_generator_minimum_real_eigenvalue": float(
            np.min(eigenvalues.real)
        ),
        "projected_weekly_spectral_radius": float(
            np.max(np.abs(np.linalg.eigvals(transition)))
        ),
        "diagonal_projection_frobenius": math.sqrt(projection_norm_sq),
    }


def hierarchical_dynamics_loss(
    model: JointHierarchicalDynamicsModel,
    origin_indices: torch.Tensor,
    endpoint_indices: torch.Tensor,
) -> torch.Tensor:
    """Return exact-week Gaussian transition NLL per state coordinate."""
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


def hierarchical_factor_gauge_penalty(
    states: torch.Tensor,
    state_sizes_by_expiry: tuple[int, ...],
) -> torch.Tensor:
    """Standardize each expiry block without erasing cross-expiry dependence."""
    penalties: list[torch.Tensor] = []
    start = 0
    for width in state_sizes_by_expiry:
        stop = start + width
        block = states[:, start:stop]
        centered = block - torch.mean(block, dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / states.shape[0]
        identity = torch.eye(width, dtype=states.dtype, device=states.device)
        penalties.append(
            torch.mean(torch.mean(block, dim=0).square())
            + torch.sum((covariance - identity).square()) / width
        )
        start = stop
    return torch.mean(torch.stack(penalties))


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_checkpoint(
    model: JointHierarchicalDynamicsModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    history: list[dict[str, float]],
    path: Path,
    *,
    train_config: JointHierarchicalTrainConfig,
    market_fingerprints: tuple[str, ...],
    source_decoder_sha256: str,
    phase: str,
) -> None:
    payload: dict[str, object] = {
        "schema": "flex-joint-hierarchical-dynamics-v1",
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
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(selected_device).cpu()
    _atomic_torch_save(payload, path)


def _restore_checkpoint(
    model: JointHierarchicalDynamicsModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    path: Path,
    *,
    train_config: JointHierarchicalTrainConfig,
    market_fingerprints: tuple[str, ...],
    source_decoder_sha256: str,
) -> tuple[int, list[dict[str, float]], str] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location=model.device, weights_only=False)
    if payload.get("schema") != "flex-joint-hierarchical-dynamics-v1":
        raise ValueError(f"Unsupported joint hierarchical checkpoint in {path}")
    if payload.get("model_config_version") != model.config.version.unique_id48 or (
        payload.get("model_user_version") != model.config.user_version
    ):
        raise ValueError(
            "Joint hierarchical checkpoint model version changed; cold start required"
        )
    if payload.get("model_config") != asdict(model.config):
        raise ValueError("Joint hierarchical checkpoint model configuration mismatch")
    if payload.get("train_config") != asdict(train_config):
        raise ValueError("Joint hierarchical checkpoint training configuration mismatch")
    if tuple(payload.get("market_fingerprints", ())) != market_fingerprints:
        raise ValueError("Joint hierarchical checkpoint market identity mismatch")
    if payload.get("source_decoder_sha256") != source_decoder_sha256:
        raise ValueError("Joint hierarchical source decoder checkpoint changed")
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
    history = [
        {str(key): float(value) for key, value in item.items()}
        for item in payload.get("history", [])
    ]
    return int(payload["completed_epochs"]), history, str(payload["phase"])


def fit_joint_hierarchical_model(
    model: JointHierarchicalDynamicsModel,
    market_days: tuple[FlexMarketDay, ...],
    *,
    dates: tuple[date, ...],
    objective_config: FlexObjectiveConfig,
    train_config: JointHierarchicalTrainConfig,
    checkpoint_path: Path,
    source_decoder_sha256: str,
    env: BaseEnvironment,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Alternate equal-date quote steps with exact-week causal dynamics."""
    if dates != tuple(day.trading_date for day in market_days):
        raise ValueError("Joint training dates and markets are not aligned")
    origin_np, endpoint_np = exact_weekly_pairs(dates, weeks=1)
    origin = torch.as_tensor(origin_np, dtype=torch.long, device=model.device)
    endpoint = torch.as_tensor(endpoint_np, dtype=torch.long, device=model.device)
    torch_days = tuple(materialize_day(day, env=env) for day in market_days)
    optimizer = torch.optim.Adam(
        [
            {"params": [model.daily_states], "lr": train_config.state_learning_rate},
            {
                "params": list(model.raw_decoder.parameters()),
                "lr": train_config.decoder_learning_rate,
            },
            {
                "params": [
                    model.damping_factor,
                    model.rotation_raw,
                    model.causal_coupling_raw,
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
    fingerprints = tuple(day.training_fingerprint for day in market_days)
    restored = _restore_checkpoint(
        model,
        optimizer,
        scheduler,
        checkpoint_path,
        train_config=train_config,
        market_fingerprints=fingerprints,
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
            raise ValueError(f"Unexpected hierarchical checkpoint phase {phase!r}")

    rng = np.random.default_rng(train_config.seed)
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
                    f"Non-finite quote loss at hierarchical epoch {epoch + 1}"
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
            dynamics = hierarchical_dynamics_loss(model, origin, endpoint)
            gauge = hierarchical_factor_gauge_penalty(
                model.daily_states,
                model.config.state_sizes_by_expiry,
            )
            dynamics_objective = (
                dynamics_ramp * train_config.dynamics_weight * dynamics
                + train_config.gauge_weight * gauge
            )
            if not bool(torch.isfinite(dynamics_objective)):
                raise TrainingConvergenceError(
                    f"Non-finite dynamics loss at hierarchical epoch {epoch + 1}"
                )
            dynamics_objective.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    model.daily_states,
                    model.damping_factor,
                    model.rotation_raw,
                    model.causal_coupling_raw,
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
        history.append(
            {
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
        )
        if (epoch + 1) % train_config.report_every == 0 or epoch == start_epoch:
            print(
                "joint-hierarchical "
                f"epoch={epoch + 1}/{train_config.num_epochs} "
                f"quote={quote_value:.7g} dynamics={dynamics_value:.7g} "
                f"gauge={gauge_value:.7g} total={joint_value:.7g} "
                f"elapsed={perf_counter() - started:.1f}s",
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
                market_fingerprints=fingerprints,
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
            market_fingerprints=fingerprints,
            source_decoder_sha256=source_decoder_sha256,
            phase="training",
        )
        raise TrainingConvergenceError(
            f"Joint hierarchical training failed convergence: {json.dumps(audit)}"
        )
    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        history,
        checkpoint_path,
        train_config=train_config,
        market_fingerprints=fingerprints,
        source_decoder_sha256=source_decoder_sha256,
        phase="complete",
    )
    return history, audit


def project_native_raw_to_hierarchical_states(
    model: JointHierarchicalDynamicsModel,
    native_raw: torch.Tensor,
) -> torch.Tensor:
    """Return raw-L2 states solely as an initialization/diagnostic."""
    weight = model.effective_decoder_weight().detach().cpu().double()
    bias = cast(torch.Tensor, model.raw_decoder.I0.bias).detach().cpu().double()
    centered = native_raw.detach().cpu().double() - bias
    return torch.linalg.lstsq(
        weight,
        centered.transpose(0, 1),
    ).solution.transpose(0, 1)


def evaluation_model_with_states(
    model: JointHierarchicalDynamicsModel,
    states: torch.Tensor,
    *,
    env: BaseEnvironment,
) -> JointHierarchicalDynamicsModel:
    """Clone frozen shared parameters and install an independent state table."""
    evaluation = JointHierarchicalDynamicsModel(
        env,
        num_days=states.shape[0],
        config=model.config,
    )
    copied = evaluation.state_dict()
    for key, value in model.state_dict().items():
        if key != "daily_state_embedding.weight":
            copied[key] = value
    copied["daily_state_embedding.weight"] = states.to(
        dtype=env.fdtype,
        device=env.device,
    )
    evaluation.load_state_dict(copied)
    return evaluation


def evaluate_states(
    model: JointHierarchicalDynamicsModel,
    states: torch.Tensor,
    market_days: tuple[FlexMarketDay, ...],
    *,
    objective_config: FlexObjectiveConfig,
    day_batch_size: int,
    env: BaseEnvironment,
) -> FitEvaluation:
    """Evaluate a state matrix without mutating the trained model."""
    evaluation = evaluation_model_with_states(model, states, env=env)
    torch_days = tuple(materialize_day(day, env=env) for day in market_days)
    return evaluate_model(
        evaluation,
        torch_days,
        env=env,
        objective_config=objective_config,
        day_batch_size=day_batch_size,
        equal_date_weighting=True,
    )


def _state_learning_rate_multiplier(
    epoch: int,
    config: QuoteOptimalStateTrainConfig,
) -> float:
    if epoch < config.warmup_epochs:
        return (epoch + 1) / max(config.warmup_epochs, 1)
    progress = (epoch - config.warmup_epochs) / max(
        config.num_epochs - config.warmup_epochs - 1,
        1,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.minimum_learning_rate_fraction + (
        1.0 - config.minimum_learning_rate_fraction
    ) * cosine


def _state_convergence_audit(
    history: list[dict[str, float]],
    *,
    config: QuoteOptimalStateTrainConfig,
) -> dict[str, object]:
    # This is the exact optimized state objective, including the DLV cap
    # barrier, rather than the reporting-only quote component.
    values = np.asarray([item["total"] for item in history], dtype=np.float64)
    if len(values) <= config.convergence_window or not np.all(np.isfinite(values)):
        return {
            "converged": False,
            "reason": "insufficient_or_nonfinite_history",
            "num_epochs": len(values),
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
        "num_epochs": len(values),
        "window": config.convergence_window,
        "window_relative_change": window_change,
        "final_relative_gap_to_best": final_gap,
        "best_epoch": int(np.argmin(values)) + 1,
    }


def _save_state_checkpoint(
    model: JointHierarchicalDynamicsModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    history: list[dict[str, float]],
    path: Path,
    *,
    train_config: QuoteOptimalStateTrainConfig,
    market_fingerprints: tuple[str, ...],
    source_model_sha256: str,
    rng: np.random.Generator,
    phase: str,
    checkpoint_schema: str = "flex-quote-optimal-hierarchical-states-v1",
    continuation_provenance: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema": checkpoint_schema,
        "phase": phase,
        "model_config": asdict(model.config),
        "model_config_version": model.config.version.unique_id48,
        "model_user_version": model.config.user_version,
        "train_config": asdict(train_config),
        "daily_states": model.daily_states.detach().cpu(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "completed_epochs": len(history),
        "history": history,
        "market_fingerprints": list(market_fingerprints),
        "source_model_sha256": source_model_sha256,
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.random.get_rng_state(),
    }
    if continuation_provenance is not None:
        payload["continuation_provenance"] = continuation_provenance
    selected_device = str(model.device)
    if selected_device.startswith("cuda"):
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(selected_device).cpu()
    _atomic_torch_save(payload, path)


def fit_quote_optimal_states(
    source: JointHierarchicalDynamicsModel,
    initial_states: torch.Tensor,
    market_days: tuple[FlexMarketDay, ...],
    *,
    objective_config: FlexObjectiveConfig,
    train_config: QuoteOptimalStateTrainConfig,
    checkpoint_path: Path,
    source_model_path: Path,
    env: BaseEnvironment,
    checkpoint_schema: str = "flex-quote-optimal-hierarchical-states-v1",
    continuation_provenance: dict[str, object] | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]], dict[str, object]]:
    """Fit one 2025 ``h_t`` per date against quotes with all shared weights frozen."""
    if initial_states.shape != (len(market_days), source.config.state_size):
        raise ValueError(
            "Quote-optimal state initializer has wrong shape: "
            f"{initial_states.shape}"
        )
    model = evaluation_model_with_states(source, initial_states, env=env)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Validation dates are independent capacity parameters.  Sparse updates
    # avoid Adam momentum moving rows absent from the current date minibatch.
    model.daily_state_embedding.sparse = True
    model.daily_states.requires_grad_(True)
    optimizer = torch.optim.SparseAdam(
        [model.daily_states],
        lr=train_config.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _state_learning_rate_multiplier(epoch, train_config),
    )
    fingerprints = tuple(day.training_fingerprint for day in market_days)
    source_sha256 = _sha256(source_model_path)
    rng = np.random.default_rng(train_config.seed)
    history: list[dict[str, float]] = []
    start_epoch = 0
    if checkpoint_path.exists():
        payload = torch.load(
            checkpoint_path,
            map_location=model.device,
            weights_only=False,
        )
        expected = (
            payload.get("schema") == checkpoint_schema
            and payload.get("model_config_version")
            == model.config.version.unique_id48
            and payload.get("model_user_version") == model.config.user_version
            and payload.get("model_config") == asdict(model.config)
            and payload.get("train_config") == asdict(train_config)
            and tuple(payload.get("market_fingerprints", ())) == fingerprints
            and payload.get("source_model_sha256") == source_sha256
            and payload.get("continuation_provenance")
            == continuation_provenance
        )
        if not expected:
            raise ValueError("Quote-optimal state checkpoint identity mismatch")
        restored_states = cast(torch.Tensor, payload["daily_states"])
        if restored_states.shape != model.daily_states.shape or not bool(
            torch.all(torch.isfinite(restored_states))
        ):
            raise ValueError("Quote-optimal state checkpoint has invalid states")
        model.daily_states.data.copy_(restored_states)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = [
            {str(key): float(value) for key, value in item.items()}
            for item in payload.get("history", [])
        ]
        start_epoch = int(payload["completed_epochs"])
        phase = payload.get("phase")
        if phase == "complete":
            return (
                model.daily_states.detach().cpu().double(),
                history,
                _state_convergence_audit(history, config=train_config),
            )
        if phase != "training":
            raise ValueError(f"Unexpected quote-state checkpoint phase {phase!r}")
        if start_epoch > train_config.num_epochs or start_epoch != len(history):
            raise ValueError("Quote-state checkpoint epoch/history mismatch")
        rng.bit_generator.state = payload["numpy_rng_state"]
        torch.random.set_rng_state(payload["torch_rng_state"].cpu())
        selected_device = str(model.device)
        if selected_device.startswith("cuda"):
            torch.cuda.set_rng_state(
                payload["cuda_rng_state"].cpu(),
                device=selected_device,
            )

    torch_days = tuple(materialize_day(day, env=env) for day in market_days)
    if train_config.compile_sanos_solver:
        model.sanos_decoder.enable_solver_compile()
    started = perf_counter()
    for epoch in range(start_epoch, train_config.num_epochs):
        order = rng.permutation(len(torch_days))
        quote_total = 0.0
        cap_total = 0.0
        date_count = 0
        for start in range(0, len(torch_days), train_config.day_batch_size):
            indices = order[start : start + train_config.day_batch_size]
            batch = materialize_batch(torch_days, indices, env=env)
            optimizer.zero_grad(set_to_none=True)
            prices = _model_prices(model, batch)
            quote = equal_date_quote_objective(
                prices,
                batch,
                objective_config=objective_config,
            )
            cap = model_dlvsq_cap_penalty(model, batch.day_indices)
            loss = quote + train_config.cap_penalty_weight * cap
            if not bool(torch.isfinite(loss)):
                raise TrainingConvergenceError(
                    f"Non-finite quote-optimal state loss at epoch {epoch + 1}"
                )
            loss.backward()
            gradient = model.daily_states.grad
            if gradient is None or not gradient.is_sparse:
                raise RuntimeError(
                    "Expected sparse quote-optimal validation-state gradients"
                )
            gradient = gradient.coalesce()
            norm = torch.linalg.vector_norm(gradient.values())
            scale = torch.clamp(
                train_config.gradient_clip / (norm + 1.0e-12),
                max=1.0,
            )
            gradient.values().mul_(scale)
            model.daily_states.grad = gradient
            optimizer.step()
            num_dates = len(indices)
            quote_total += float(quote.detach().cpu()) * num_dates
            cap_total += float(cap.detach().cpu()) * num_dates
            date_count += num_dates
        scheduler.step()
        # ``quote`` and ``cap`` are date means within a batch.  Weight by the
        # number of dates so the final short batch has no excess influence.
        objective = quote_total / date_count
        cap_value = cap_total / date_count
        history.append(
            {
                "epoch": float(epoch + 1),
                "objective": objective,
                "dlvsq_cap_penalty": cap_value,
                "total": objective + train_config.cap_penalty_weight * cap_value,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if (epoch + 1) % train_config.report_every == 0 or epoch == start_epoch:
            print(
                "hierarchical-quote-optimal-state "
                f"epoch={epoch + 1}/{train_config.num_epochs} "
                f"quote={objective:.7g} cap={cap_value:.7g} "
                f"elapsed={perf_counter() - started:.1f}s",
                flush=True,
            )
        if (epoch + 1) % train_config.checkpoint_every == 0:
            _save_state_checkpoint(
                model,
                optimizer,
                scheduler,
                history,
                checkpoint_path,
                train_config=train_config,
                market_fingerprints=fingerprints,
                source_model_sha256=source_sha256,
                rng=rng,
                phase="training",
                checkpoint_schema=checkpoint_schema,
                continuation_provenance=continuation_provenance,
            )

    if train_config.compile_sanos_solver:
        model.sanos_decoder.disable_solver_compile()
    audit = _state_convergence_audit(history, config=train_config)
    if audit.get("converged") is not True:
        _save_state_checkpoint(
            model,
            optimizer,
            scheduler,
            history,
            checkpoint_path,
            train_config=train_config,
            market_fingerprints=fingerprints,
            source_model_sha256=source_sha256,
            rng=rng,
            phase="training",
            checkpoint_schema=checkpoint_schema,
            continuation_provenance=continuation_provenance,
        )
        raise TrainingConvergenceError(
            f"Quote-optimal hierarchical states failed convergence: {audit}"
        )
    _save_state_checkpoint(
        model,
        optimizer,
        scheduler,
        history,
        checkpoint_path,
        train_config=train_config,
        market_fingerprints=fingerprints,
        source_model_sha256=source_sha256,
        rng=rng,
        phase="complete",
        checkpoint_schema=checkpoint_schema,
        continuation_provenance=continuation_provenance,
    )
    return model.daily_states.detach().cpu().double(), history, audit


_CONTINUATION_CHECKPOINT_SCHEMA = (
    "flex-quote-optimal-hierarchical-states-continuation-v1"
)


def load_quote_state_continuation_initialization(
    source: JointHierarchicalDynamicsModel,
    market_days: tuple[FlexMarketDay, ...],
    *,
    source_state_checkpoint_path: Path,
    source_model_path: Path,
    base_train_config: QuoteOptimalStateTrainConfig,
) -> QuoteStateContinuationInitialization:
    """Validate an exhausted non-plateau phase before resetting optimization."""
    if not source_state_checkpoint_path.exists():
        raise FileNotFoundError(source_state_checkpoint_path)
    if not source_model_path.exists():
        raise FileNotFoundError(source_model_path)
    payload = torch.load(
        source_state_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    fingerprints = tuple(day.training_fingerprint for day in market_days)
    source_model_sha256 = _sha256(source_model_path)
    expected_identity = {
        "schema": "flex-quote-optimal-hierarchical-states-v1",
        "phase": "training",
        "model_config": asdict(source.config),
        "model_config_version": source.config.version.unique_id48,
        "model_user_version": source.config.user_version,
        "train_config": asdict(base_train_config),
        "market_fingerprints": list(fingerprints),
        "source_model_sha256": source_model_sha256,
    }
    for name, expected in expected_identity.items():
        if payload.get(name) != expected:
            raise ValueError(
                "Continuation source checkpoint identity mismatch for "
                f"{name!r}"
            )
    completed_epochs = int(payload.get("completed_epochs", -1))
    history_payload = payload.get("history")
    if not isinstance(history_payload, list):
        raise ValueError("Continuation source has no training history")
    history = [
        {str(key): float(value) for key, value in item.items()}
        for item in history_payload
        if isinstance(item, dict)
    ]
    if (
        len(history) != len(history_payload)
        or completed_epochs != base_train_config.num_epochs
        or len(history) != completed_epochs
    ):
        raise ValueError(
            "Continuation requires one complete, exhausted base schedule"
        )
    if not all(
        all(np.isfinite(value) for value in item.values()) for item in history
    ):
        raise ValueError("Continuation source history contains non-finite values")
    convergence = _state_convergence_audit(history, config=base_train_config)
    if convergence.get("converged") is True:
        raise ValueError(
            "A converged base checkpoint must be completed, not continued"
        )
    if convergence.get("reason") != "trailing_window_not_converged":
        raise ValueError(
            "Continuation is allowed only for a finite failed plateau audit: "
            f"{convergence}"
        )
    for required in (
        "optimizer_state_dict",
        "scheduler_state_dict",
        "numpy_rng_state",
        "torch_rng_state",
    ):
        if required not in payload:
            raise ValueError(
                f"Continuation source is missing restart evidence {required!r}"
            )
    states = payload.get("daily_states")
    if not isinstance(states, torch.Tensor):
        raise ValueError("Continuation source has no daily-state tensor")
    expected_shape = (len(market_days), source.config.state_size)
    if states.shape != expected_shape or not bool(torch.all(torch.isfinite(states))):
        raise ValueError(
            "Continuation source state shape/finiteness mismatch: "
            f"{tuple(states.shape)} versus {expected_shape}"
        )
    checkpoint_sha256 = _sha256(source_state_checkpoint_path)
    provenance: dict[str, object] = {
        "source_state_checkpoint": str(source_state_checkpoint_path),
        "source_state_checkpoint_sha256": checkpoint_sha256,
        "source_state_checkpoint_schema": payload["schema"],
        "source_model_checkpoint": str(source_model_path),
        "source_model_checkpoint_sha256": source_model_sha256,
        "source_completed_epochs": completed_epochs,
        "source_convergence": convergence,
        "source_final_metrics": history[-1],
        "optimizer_scheduler_reset": True,
        "reset_reason": (
            "base 600-epoch schedule exhausted with finite improving objective "
            "but failed the strict trailing-window plateau gate"
        ),
    }
    return QuoteStateContinuationInitialization(
        states=states.detach().cpu().double(),
        source_checkpoint_sha256=checkpoint_sha256,
        source_completed_epochs=completed_epochs,
        source_convergence=convergence,
        provenance=provenance,
    )


def continue_quote_optimal_states(
    source: JointHierarchicalDynamicsModel,
    market_days: tuple[FlexMarketDay, ...],
    *,
    objective_config: FlexObjectiveConfig,
    base_train_config: QuoteOptimalStateTrainConfig,
    continuation_config: QuoteOptimalStateContinuationConfig,
    source_state_checkpoint_path: Path,
    continuation_checkpoint_path: Path,
    source_model_path: Path,
    env: BaseEnvironment,
) -> tuple[
    torch.Tensor,
    list[dict[str, float]],
    dict[str, object],
    QuoteStateContinuationInitialization,
]:
    """Run a provenance-bound lower-rate phase with a fresh optimizer."""
    initialization = load_quote_state_continuation_initialization(
        source,
        market_days,
        source_state_checkpoint_path=source_state_checkpoint_path,
        source_model_path=source_model_path,
        base_train_config=base_train_config,
    )
    states, history, convergence = fit_quote_optimal_states(
        source,
        initialization.states,
        market_days,
        objective_config=objective_config,
        train_config=continuation_config,
        checkpoint_path=continuation_checkpoint_path,
        source_model_path=source_model_path,
        env=env,
        checkpoint_schema=_CONTINUATION_CHECKPOINT_SCHEMA,
        continuation_provenance=initialization.provenance,
    )
    return states, history, convergence, initialization


def hierarchical_dynamics_diagnostics(
    model: JointHierarchicalDynamicsModel,
    train_states: np.ndarray,
    evaluation_states: np.ndarray,
    evaluation_dates: tuple[date, ...],
) -> dict[str, object]:
    """Report stability, forecast skill, and innovation calibration."""
    with torch.no_grad():
        generator = model.mean_reversion_generator().detach().cpu().double().numpy()
        transition = model.weekly_transition().detach().cpu().double().numpy()
        factor = model.innovation_cholesky().detach().cpu().double().numpy()
        equilibrium = model.equilibrium_mean.detach().cpu().double().numpy()
    covariance = factor @ factor.T
    eigenvalues = np.linalg.eigvals(generator)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if np.min(eigenvalues.real) <= 0.0 or spectral_radius >= 1.0:
        raise RuntimeError("Joint hierarchical dynamics lost stationarity")
    forbidden_block_lower = (
        model.generator_strict_upper_mask.detach().cpu().double().numpy().T
    )
    strict_lower_generator = generator * forbidden_block_lower
    strict_lower_transition = transition * forbidden_block_lower
    causal_tolerance = 1.0e-8 * max(1.0, float(np.max(np.abs(generator))))
    if np.max(np.abs(strict_lower_generator)) > causal_tolerance or np.max(
        np.abs(strict_lower_transition)
    ) > causal_tolerance:
        raise RuntimeError("Joint hierarchical mean transition lost causality")

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

    unconditional_mean = np.mean(train_states, axis=0)
    scale = np.maximum(np.std(train_states, axis=0), 1.0e-8)
    state_stops = np.cumsum((0, *model.config.state_sizes_by_expiry))
    horizons: dict[str, object] = {}
    for weeks in (1, 2, 4):
        origin, endpoint = exact_weekly_pairs(evaluation_dates, weeks=weeks)
        horizon_transition = np.linalg.matrix_power(transition, weeks)
        horizon_covariance = np.zeros_like(covariance)
        propagation = np.eye(len(transition))
        for _ in range(weeks):
            horizon_covariance += propagation.T @ covariance @ propagation
            propagation = propagation @ transition
        prediction = equilibrium + (
            evaluation_states[origin] - equilibrium
        ) @ horizon_transition
        realized = evaluation_states[endpoint]
        errors = (prediction - realized) / scale
        persistence = (evaluation_states[origin] - realized) / scale
        unconditional = (unconditional_mean - realized) / scale
        residual = realized - prediction
        horizon_factor = np.linalg.cholesky(horizon_covariance)
        white = np.linalg.solve(horizon_factor, residual.T).T
        block_diagnostics: list[dict[str, float | int]] = []
        for expiry_index in range(len(model.config.state_sizes_by_expiry)):
            state_slice = slice(
                int(state_stops[expiry_index]),
                int(state_stops[expiry_index + 1]),
            )
            block_diagnostics.append(
                {
                    "expiry_index": expiry_index,
                    "state_size": state_slice.stop - state_slice.start,
                    "state_rmse": float(
                        np.sqrt(np.mean(np.square(errors[:, state_slice])))
                    ),
                    "persistence_state_rmse": float(
                        np.sqrt(np.mean(np.square(persistence[:, state_slice])))
                    ),
                    "unconditional_state_rmse": float(
                        np.sqrt(np.mean(np.square(unconditional[:, state_slice])))
                    ),
                    "whitened_innovation_mean": float(
                        np.mean(white[:, state_slice])
                    ),
                    "whitened_innovation_std": float(
                        np.std(white[:, state_slice])
                    ),
                }
            )
        sign, logdet = np.linalg.slogdet(horizon_covariance)
        if sign <= 0.0:
            raise RuntimeError("Hierarchical horizon covariance is not positive")
        mahalanobis = np.sum(np.square(white), axis=1)
        horizons[f"{weeks}w"] = {
            "num_pairs": len(origin),
            "state_rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "persistence_state_rmse": float(
                np.sqrt(np.mean(np.square(persistence)))
            ),
            "unconditional_state_rmse": float(
                np.sqrt(np.mean(np.square(unconditional)))
            ),
            "negative_log_likelihood_per_dimension": float(
                np.mean(
                    0.5
                    * (
                        mahalanobis
                        + logdet
                        + len(transition) * math.log(2.0 * math.pi)
                    )
                )
                / len(transition)
            ),
            "mahalanobis_mean_per_dimension": float(
                np.mean(mahalanobis) / len(transition)
            ),
            "whitened_innovation_mean": float(np.mean(white)),
            "whitened_innovation_std": float(np.std(white)),
            "by_expiry": block_diagnostics,
        }
    return {
        "generator_minimum_real_eigenvalue": float(np.min(eigenvalues.real)),
        "generator_maximum_real_eigenvalue": float(np.max(eigenvalues.real)),
        "generator_maximum_imaginary_abs": float(np.max(np.abs(eigenvalues.imag))),
        "weekly_spectral_radius": spectral_radius,
        "maximum_generator_strict_lower_abs": float(
            np.max(np.abs(strict_lower_generator))
        ),
        "maximum_transition_strict_lower_abs": float(
            np.max(np.abs(strict_lower_transition))
        ),
        "maximum_transition_power_norm": maximum_power_norm,
        "maximum_transition_power_week": maximum_power_week,
        "transition_power_norms": powers,
        "transition_condition_number": float(np.linalg.cond(transition)),
        "horizons": horizons,
    }


__all__ = [
    "HierarchicalInitialization",
    "JointHierarchicalTrainConfig",
    "QuoteOptimalStateTrainConfig",
    "evaluate_states",
    "evaluation_model_with_states",
    "fit_joint_hierarchical_model",
    "fit_quote_optimal_states",
    "hierarchical_dynamics_diagnostics",
    "hierarchical_dynamics_loss",
    "hierarchical_factor_gauge_penalty",
    "initialize_joint_hierarchical_model",
    "project_native_raw_to_hierarchical_states",
    "standardize_affine_hierarchical_model",
]
