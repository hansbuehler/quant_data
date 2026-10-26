"""Train the two neural maps used by generalized-Bates CRC Algorithm 1.

NN1 maps the fourteen generalized-Bates parameters to the paper's 10-by-13
implied-volatility surface.  NN2 receives the three recalibrated Heston
parameters and a protected pre-reset surface, and returns the ten
maturity-bucket jump parameters.  NN2 is trained through frozen NN1 so its
objective is implied-volatility-surface preservation; it may choose any
equivalent jump-parameter representative, as in the paper.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from cdxcore.version import version
from filelock import FileLock

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("3.0")
@dataclass(frozen=True, kw_only=True)
class CrcInverseNetworkConfig:
    """Logical architecture of the paper-scale forward and inverse maps."""

    label: str
    group: str = "dysanos/bates-crc"
    user_version: str = "3.0"
    parameter_count: int = 14
    surface_node_count: int = 130
    jump_parameter_count: int = 10
    forward_network: FeedForwardConfig = FeedForwardConfig(
        activation="ELU",
        num_depth=4,
        num_width=1_024,
    )
    inverse_network: FeedForwardConfig = FeedForwardConfig(
        activation="ELU",
        num_depth=4,
        num_width=1_024,
    )

    def __post_init__(self) -> None:
        if not self.label or not self.group or not self.user_version:
            raise ValueError("label, group, and user_version must be non-empty")
        if (
            self.parameter_count != 14
            or self.surface_node_count != 130
            or self.jump_parameter_count != 10
        ):
            raise ValueError("generalized-Bates inverse geometry must be 14 -> 130 and 133 -> 10")


@dataclass(frozen=True, kw_only=True)
class CrcInverseTrainConfig:
    """Operational training settings; these do not change model geometry."""

    forward_epochs: int = 40
    inverse_epochs: int = 40
    batch_size: int = 1_000
    validation_batch_size: int = 8_000
    learning_rate: float = 1.0e-3
    inverse_learning_rate: float = 2.0e-4
    final_learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-6
    inverse_parameter_regularizer: float = 0.0
    gradient_clip_norm: float = 10.0
    seed: int = 7_931

    def __post_init__(self) -> None:
        if self.forward_epochs <= 0 or self.inverse_epochs <= 0:
            raise ValueError("both training phases require positive epoch counts")
        if self.batch_size <= 0 or self.validation_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if not 0.0 < self.final_learning_rate <= min(
            self.learning_rate, self.inverse_learning_rate
        ):
            raise ValueError("learning-rate bounds are invalid")
        if self.weight_decay < 0.0 or self.inverse_parameter_regularizer < 0.0:
            raise ValueError("regularization weights must be non-negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")


class CrcInverseNetworks(BaseModel):
    """Normalized NN1/NN2 pair with bounded physical jump outputs."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: CrcInverseNetworkConfig,
        parameter_mean: torch.Tensor,
        parameter_scale: torch.Tensor,
        surface_mean: torch.Tensor,
        surface_scale: torch.Tensor,
        jump_lower: torch.Tensor,
        jump_upper: torch.Tensor,
    ) -> None:
        super().__init__(env)
        self.config = config
        for name, value, shape in (
            ("parameter_mean", parameter_mean, (config.parameter_count,)),
            ("parameter_scale", parameter_scale, (config.parameter_count,)),
            ("surface_mean", surface_mean, (config.surface_node_count,)),
            ("surface_scale", surface_scale, (config.surface_node_count,)),
            ("jump_lower", jump_lower, (config.jump_parameter_count,)),
            ("jump_upper", jump_upper, (config.jump_parameter_count,)),
        ):
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            self.register_buffer(name, value.detach().to(device=env.device, dtype=env.fdtype))
        if torch.any(self.parameter_scale <= 0.0) or torch.any(self.surface_scale <= 0.0):
            raise ValueError("normalization scales must be positive")
        if torch.any(self.jump_upper <= self.jump_lower):
            raise ValueError("jump bounds must be ordered")

        self.parameter_to_surface = FeedForward(
            env,
            num_input=config.parameter_count,
            num_output=config.surface_node_count,
            config=config.forward_network,
        )
        self.surface_to_jumps = FeedForward(
            env,
            num_input=3 + config.surface_node_count,
            num_output=config.jump_parameter_count,
            config=config.inverse_network,
        )

    def normalized_surface(self, parameters: torch.Tensor) -> torch.Tensor:
        """Evaluate NN1 in normalized surface coordinates."""

        normalized_parameters = (parameters - self.parameter_mean) / self.parameter_scale
        return self.parameter_to_surface(normalized_parameters)

    def surface(self, parameters: torch.Tensor) -> torch.Tensor:
        """Evaluate NN1 in annualized Black-volatility units."""

        return self.surface_mean + self.surface_scale * self.normalized_surface(parameters)

    def jump_parameters(
        self,
        heston_state: torch.Tensor,
        target_surface: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate NN2 and map its raw output to admissible jump bounds."""

        if heston_state.shape[-1] != 4 or target_surface.shape[-1] != self.config.surface_node_count:
            raise ValueError("inverse inputs have invalid final dimensions")
        # Algorithm 1 treats the realised variance as a state variable, not a
        # recalibrated parameter.  As in the paper, NN2 therefore receives the
        # three updated Heston parameters (theta, xi, rho) and the protected
        # surface; the unchanged variance is supplied directly to NN1 below.
        normalized_heston = (
            heston_state[..., 1:] - self.parameter_mean[1:4]
        ) / self.parameter_scale[1:4]
        normalized_surface = (target_surface - self.surface_mean) / self.surface_scale
        raw = self.surface_to_jumps(torch.cat((normalized_heston, normalized_surface), dim=-1))
        unit = torch.sigmoid(raw)
        return self.jump_lower + (self.jump_upper - self.jump_lower) * unit

    def candidate_parameters(
        self,
        heston_state: torch.Tensor,
        target_surface: torch.Tensor,
    ) -> torch.Tensor:
        """Combine post-reset Heston coordinates with NN2 jump parameters."""

        return torch.cat((heston_state, self.jump_parameters(heston_state, target_surface)), dim=-1)


def _load_split(directory: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(directory.glob(f"{split}-*.npz"))
    if not paths:
        raise RuntimeError(f"no {split!r} inverse-map shards found in {directory}")
    parameters: list[np.ndarray] = []
    surfaces: list[np.ndarray] = []
    maturity_reference: np.ndarray | None = None
    moneyness_reference: np.ndarray | None = None
    scalar_reference: tuple[float, float, int, float] | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            parameter = np.asarray(archive["parameters"], dtype=np.float32)
            surface = np.asarray(archive["implied_volatilities"], dtype=np.float32)
            maturities = np.asarray(archive["maturities"], dtype=np.float64)
            moneyness = np.asarray(archive["moneyness"], dtype=np.float64)
            scalars = (
                float(archive["kappa"]),
                float(archive["intensity"]),
                int(archive["quadrature_nodes"]),
                float(archive["quadrature_u_max"]),
            )
        if parameter.ndim != 2 or parameter.shape[1] != 14:
            raise RuntimeError(f"invalid parameter geometry in {path}")
        if surface.shape != (len(parameter), 130):
            raise RuntimeError(f"invalid surface geometry in {path}")
        if maturity_reference is None:
            maturity_reference = maturities
            moneyness_reference = moneyness
            scalar_reference = scalars
        elif not np.array_equal(maturities, maturity_reference) or not np.array_equal(
            moneyness, moneyness_reference
        ):
            raise RuntimeError(f"surface grid changed in {path}")
        elif scalars != scalar_reference:
            raise RuntimeError(f"model or quadrature configuration changed in {path}")
        parameters.append(parameter)
        surfaces.append(surface)
    return np.concatenate(parameters), np.concatenate(surfaces)


def _load_grid(
    directory: Path,
) -> tuple[np.ndarray, np.ndarray, float, float, int, float]:
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise RuntimeError(f"no inverse-map shards found in {directory}")
    with np.load(paths[0], allow_pickle=False) as archive:
        return (
            np.asarray(archive["maturities"], dtype=np.float64),
            np.asarray(archive["moneyness"], dtype=np.float64),
            float(archive["kappa"]),
            float(archive["intensity"]),
            int(archive["quadrature_nodes"]),
            float(archive["quadrature_u_max"]),
        )


def apply_reset_innovations(
    heston_state: torch.Tensor,
    innovations: torch.Tensor,
    *,
    kappa: float,
    relative_change_cap: float,
) -> torch.Tensor:
    """Apply paper-style reset innovations to ``(V, theta, xi, rho)``."""

    if heston_state.shape[-1] != 4 or innovations.shape != heston_state.shape[:-1] + (3,):
        raise ValueError("reset innovation geometry is invalid")
    variance, theta, xi, rho = torch.unbind(heston_state, dim=-1)
    theta = theta * torch.exp(torch.clamp(innovations[..., 0], -relative_change_cap, relative_change_cap))
    xi = xi * torch.exp(torch.clamp(innovations[..., 1], -relative_change_cap, relative_change_cap))
    rho = torch.tanh(torch.atanh(torch.clamp(rho, -0.999999, 0.999999)) + innovations[..., 2])
    theta = torch.clamp(theta, 0.01, 0.50)
    xi = torch.clamp(xi, 0.01, 0.50)
    feller_limit = torch.sqrt(torch.clamp(2.0 * kappa * theta, min=1.0e-12)) * (1.0 - 1.0e-6)
    xi = torch.minimum(xi, feller_limit)
    return torch.stack((variance, theta, xi, rho), dim=-1)


def _normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(values, axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, np.float32(1.0e-6))
    return mean, scale


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_progress(path: Path, record: dict[str, object]) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()


def _validate_forward(
    model: CrcInverseNetworks,
    parameters: torch.Tensor,
    surfaces: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    errors: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(parameters), batch_size):
            stop = min(start + batch_size, len(parameters))
            prediction = model.surface(parameters[start:stop].to(model.device))
            errors.append((prediction.cpu() - surfaces[start:stop]).reshape(-1))
    error = torch.cat(errors).double()
    return {
        "rmse_iv": float(torch.sqrt(torch.mean(error.square()))),
        "mae_iv": float(torch.mean(torch.abs(error))),
        "maximum_absolute_iv_error": float(torch.max(torch.abs(error))),
    }


def _validate_inverse_surface(
    model: CrcInverseNetworks,
    parameters: torch.Tensor,
    surfaces: torch.Tensor,
    *,
    innovation_mean: torch.Tensor,
    innovation_root: torch.Tensor,
    kappa: float,
    relative_change_cap: float,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    """Measure held-out IV preservation after deterministic reset innovations."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    errors: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(parameters), batch_size):
            stop = min(start + batch_size, len(parameters))
            heston = parameters[start:stop, :4]
            normal = torch.randn((len(heston), 3), generator=generator)
            innovations = innovation_mean + normal @ innovation_root.T
            post_reset = apply_reset_innovations(
                heston,
                innovations,
                kappa=kappa,
                relative_change_cap=relative_change_cap,
            ).to(model.device)
            target = surfaces[start:stop].to(model.device)
            candidate = model.candidate_parameters(post_reset, target)
            prediction = model.surface(candidate)
            errors.append((prediction - target).cpu().reshape(-1))
    error = torch.cat(errors).double()
    return {
        "inverse_rmse_iv": float(torch.sqrt(torch.mean(error.square()))),
        "inverse_mae_iv": float(torch.mean(torch.abs(error))),
        "inverse_maximum_absolute_iv_error": float(torch.max(torch.abs(error))),
    }


def _checkpoint_payload(
    model: CrcInverseNetworks,
    *,
    model_config: CrcInverseNetworkConfig,
    train_config: CrcInverseTrainConfig,
    phase: str,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cpu_generator: torch.Generator,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "gambara-teichmann-crc-inverse-networks-v1",
        "user_version": model_config.user_version,
        "model_version": model_config.version.unique_id48,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "phase": phase,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "cpu_generator_state": cpu_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        # A CPU-only service must not initialize an unallocated CUDA context
        # merely to serialize RNG state.  Detached CPU jobs explicitly hide
        # CUDA as well, but this device check is the correctness backstop.
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.device(model.device).type == "cuda"
            else None
        ),
    }
    return payload


def train_inverse_maps(
    *,
    dataset_directory: Path,
    target_fit_report: Path,
    output_directory: Path,
    model_config: CrcInverseNetworkConfig,
    train_config: CrcInverseTrainConfig,
    device: str,
) -> None:
    """Train NN1 followed by NN2, persisting restart-complete checkpoints."""

    output_directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(output_directory / "training.lock")
    with lock:
        started = time.monotonic()
        progress_path = output_directory / "training-progress.jsonl"
        report = json.loads(target_fit_report.read_text(encoding="utf-8"))
        reset_law = report["reset_innovation_law"]
        kappa = float(report["target_fit"]["kappa_fixed_from_history"])
        if report.get("target_date") != "2025-08-29":
            raise RuntimeError("inverse training received the wrong target-date fit")

        (
            maturities,
            _,
            dataset_kappa,
            dataset_intensity,
            dataset_quadrature_nodes,
            dataset_quadrature_u_max,
        ) = _load_grid(dataset_directory)
        target_maturities = np.asarray(
            report["target_fit"]["maturity_nodes"], dtype=np.float64
        )
        if not np.array_equal(maturities, target_maturities):
            raise RuntimeError(
                "inverse-map maturity grid does not match the target calibration; "
                "generate target-specific surfaces before training"
            )
        target_intensity = float(report["target_fit"]["intensity_fixed_from_history"])
        if dataset_kappa != kappa or dataset_intensity != target_intensity:
            raise RuntimeError(
                "inverse-map fixed kappa/intensity do not match the target calibration"
            )
        if dataset_quadrature_nodes < 240 or dataset_quadrature_u_max < 600.0:
            raise RuntimeError(
                "inverse-map surfaces were generated with insufficient Fourier quadrature"
            )

        train_parameters_np, train_surfaces_np = _load_split(dataset_directory, "train")
        validation_parameters_np, validation_surfaces_np = _load_split(
            dataset_directory, "heldout"
        )
        target_parameters = np.asarray(
            (
                report["target_fit"]["variance"],
                report["target_fit"]["theta"],
                report["target_fit"]["xi"],
                report["target_fit"]["rho"],
                *report["target_fit"]["jump_means"],
                *report["target_fit"]["jump_stds"],
            ),
            dtype=np.float64,
        )
        training_minimum = np.min(train_parameters_np, axis=0)
        training_maximum = np.max(train_parameters_np, axis=0)
        # Uniform draws almost surely miss an exact configured endpoint.  Permit
        # only numerical endpoint slack; material domain mismatches still fail.
        coverage_tolerance = 1.0e-5
        if np.any(target_parameters < training_minimum - coverage_tolerance) or np.any(
            target_parameters > training_maximum + coverage_tolerance
        ):
            outside = np.flatnonzero(
                (target_parameters < training_minimum - coverage_tolerance)
                | (target_parameters > training_maximum + coverage_tolerance)
            )
            raise RuntimeError(
                "target calibration lies outside the inverse-map training domain "
                f"at parameter coordinates {outside.tolist()}"
            )
        parameter_mean, parameter_scale = _normalization(train_parameters_np)
        surface_mean, surface_scale = _normalization(train_surfaces_np)
        jump_lower = np.concatenate((np.full(5, -0.30), np.full(5, 0.005))).astype(
            np.float32
        )
        jump_upper = np.concatenate((np.full(5, 0.15), np.full(5, 0.30))).astype(
            np.float32
        )

        env = BaseEnvironment(precision=32, device=device, debug=False)
        model = CrcInverseNetworks(
            env,
            config=model_config,
            parameter_mean=torch.from_numpy(parameter_mean),
            parameter_scale=torch.from_numpy(parameter_scale),
            surface_mean=torch.from_numpy(surface_mean),
            surface_scale=torch.from_numpy(surface_scale),
            jump_lower=torch.from_numpy(jump_lower),
            jump_upper=torch.from_numpy(jump_upper),
        )
        train_parameters = torch.from_numpy(train_parameters_np)
        train_surfaces = torch.from_numpy(train_surfaces_np)
        validation_parameters = torch.from_numpy(validation_parameters_np)
        validation_surfaces = torch.from_numpy(validation_surfaces_np)
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(train_config.seed)
        torch.manual_seed(train_config.seed)

        total_epochs = train_config.forward_epochs + train_config.inverse_epochs
        checkpoint_path = output_directory / "checkpoint.pt"
        resume_phase: str | None = None
        resume_epoch = 0
        resume_optimizer: dict[str, object] | None = None
        resume_scheduler: dict[str, object] | None = None
        if checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
            if payload.get("schema") != "gambara-teichmann-crc-inverse-networks-v1":
                raise RuntimeError(f"invalid inverse checkpoint schema in {checkpoint_path}")
            if payload.get("user_version") != model_config.user_version:
                raise RuntimeError("inverse checkpoint user version does not match")
            if payload.get("model_version") != model_config.version.unique_id48:
                raise RuntimeError("inverse checkpoint code version does not match")
            if payload.get("model_config") != asdict(model_config):
                raise RuntimeError("inverse checkpoint model configuration does not match")
            model.load_state_dict(payload["model_state_dict"])
            resume_phase = str(payload["phase"])
            resume_epoch = int(payload["epoch"])
            resume_optimizer = payload["optimizer_state_dict"]
            resume_scheduler = payload["scheduler_state_dict"]
            cpu_generator.set_state(payload["cpu_generator_state"])
            torch.set_rng_state(payload["torch_rng_state"])
            cuda_state = payload.get("cuda_rng_state")
            if cuda_state is not None and torch.device(model.device).type == "cuda":
                torch.cuda.set_rng_state_all(cuda_state)

        completed_epochs = 0
        final_validation: dict[str, float] = {}
        phase_specifications = (
            ("forward", train_config.forward_epochs),
            ("inverse", train_config.inverse_epochs),
        )
        phase_names = tuple(name for name, _ in phase_specifications)
        if resume_phase is not None and resume_phase not in phase_names:
            raise RuntimeError(f"unknown checkpoint phase {resume_phase!r}")
        innovation_mean = torch.tensor(reset_law["mean"], dtype=torch.float32)
        innovation_covariance = torch.tensor(reset_law["covariance"], dtype=torch.float32)
        innovation_root = torch.linalg.cholesky(
            innovation_covariance + 1.0e-10 * torch.eye(3)
        )
        last_progress_at = time.monotonic()

        for phase, epochs in phase_specifications:
            forward_phase = phase == "forward"
            phase_index = phase_names.index(phase)
            resume_phase_index = (
                phase_names.index(resume_phase) if resume_phase is not None else -1
            )
            if phase_index < resume_phase_index:
                completed_epochs += epochs
                continue
            first_epoch = resume_epoch if phase == resume_phase else 0
            completed_epochs += first_epoch
            if first_epoch >= epochs:
                if forward_phase:
                    final_validation = _validate_forward(
                        model,
                        validation_parameters,
                        validation_surfaces,
                        batch_size=train_config.validation_batch_size,
                    )
                else:
                    final_validation = {
                        **final_validation,
                        **_validate_inverse_surface(
                            model,
                            validation_parameters,
                            validation_surfaces,
                            innovation_mean=innovation_mean,
                            innovation_root=innovation_root,
                            kappa=kappa,
                            relative_change_cap=float(reset_law["relative_change_cap"]),
                            batch_size=train_config.validation_batch_size,
                            seed=train_config.seed + 1,
                        ),
                    }
                continue
            for parameter in model.parameter_to_surface.parameters():
                parameter.requires_grad_(forward_phase)
            for parameter in model.surface_to_jumps.parameters():
                parameter.requires_grad_(not forward_phase)
            optimized = (
                model.parameter_to_surface.parameters()
                if forward_phase
                else model.surface_to_jumps.parameters()
            )
            phase_learning_rate = (
                train_config.learning_rate
                if forward_phase
                else train_config.inverse_learning_rate
            )
            optimizer = torch.optim.AdamW(
                optimized,
                lr=phase_learning_rate,
                weight_decay=train_config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=train_config.final_learning_rate,
            )
            if phase == resume_phase and resume_optimizer is not None and resume_scheduler is not None:
                optimizer.load_state_dict(resume_optimizer)
                scheduler.load_state_dict(resume_scheduler)
            for epoch in range(first_epoch, epochs):
                permutation = torch.randperm(len(train_parameters), generator=cpu_generator)
                total_loss = 0.0
                total_observations = 0
                model.train()
                for start in range(0, len(permutation), train_config.batch_size):
                    indices = permutation[start : start + train_config.batch_size]
                    parameters_cpu = train_parameters[indices]
                    surfaces_cpu = train_surfaces[indices]
                    parameters = parameters_cpu.to(model.device)
                    surfaces = surfaces_cpu.to(model.device)
                    optimizer.zero_grad(set_to_none=True)
                    if forward_phase:
                        target = surfaces
                        prediction = model.surface(parameters)
                        loss = torch.mean((prediction - target).square())
                    else:
                        normal = torch.randn(
                            (len(indices), 3),
                            generator=cpu_generator,
                            dtype=torch.float32,
                        )
                        innovations = innovation_mean + normal @ innovation_root.T
                        post_reset = apply_reset_innovations(
                            parameters_cpu[:, :4],
                            innovations,
                            kappa=kappa,
                            relative_change_cap=float(reset_law["relative_change_cap"]),
                        ).to(model.device)
                        candidate = model.candidate_parameters(post_reset, surfaces)
                        reconstructed = model.surface(candidate)
                        target = surfaces
                        original_jumps = parameters[:, 4:]
                        predicted_jumps = candidate[:, 4:]
                        jump_scale = model.parameter_scale[4:]
                        loss = torch.mean((reconstructed - target).square())
                        loss = loss + train_config.inverse_parameter_regularizer * torch.mean(
                            ((predicted_jumps - original_jumps) / jump_scale).square()
                        )
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite {phase} loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_config.gradient_clip_norm
                    )
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(indices)
                    total_observations += len(indices)
                    now = time.monotonic()
                    if now - last_progress_at >= 240.0:
                        fractional_completed = completed_epochs + min(
                            total_observations / len(train_parameters), 1.0
                        )
                        elapsed = now - started
                        eta = (
                            elapsed * (total_epochs - fractional_completed) / fractional_completed
                        )
                        _write_progress(
                            progress_path,
                            {
                                "stage": f"train_{phase}",
                                "phase_epoch": epoch + 1,
                                "phase_epochs": epochs,
                                "epoch_observations": total_observations,
                                "epoch_total_observations": len(train_parameters),
                                "completed_work_units": fractional_completed,
                                "total_work_units": total_epochs,
                                "elapsed_seconds": elapsed,
                                "estimated_remaining_seconds": eta,
                                "training_loss": total_loss / total_observations,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "heartbeat": True,
                            },
                        )
                        last_progress_at = now
                scheduler.step()
                completed_epochs += 1
                if forward_phase:
                    final_validation = _validate_forward(
                        model,
                        validation_parameters,
                        validation_surfaces,
                        batch_size=train_config.validation_batch_size,
                    )
                else:
                    final_validation = {
                        **final_validation,
                        **_validate_inverse_surface(
                            model,
                            validation_parameters,
                            validation_surfaces,
                            innovation_mean=innovation_mean,
                            innovation_root=innovation_root,
                            kappa=kappa,
                            relative_change_cap=float(reset_law["relative_change_cap"]),
                            batch_size=train_config.validation_batch_size,
                            seed=train_config.seed + 1,
                        ),
                    }
                elapsed = time.monotonic() - started
                eta = elapsed * (total_epochs - completed_epochs) / completed_epochs
                record: dict[str, object] = {
                    "stage": f"train_{phase}",
                    "phase_epoch": epoch + 1,
                    "phase_epochs": epochs,
                    "completed_work_units": completed_epochs,
                    "total_work_units": total_epochs,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": eta,
                    "training_loss": total_loss / total_observations,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **final_validation,
                }
                _write_progress(progress_path, record)
                last_progress_at = time.monotonic()
                _atomic_torch_save(
                    _checkpoint_payload(
                        model,
                        model_config=model_config,
                        train_config=train_config,
                        phase=phase,
                        epoch=epoch + 1,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cpu_generator=cpu_generator,
                    ),
                    checkpoint_path,
                )

        if "rmse_iv" not in final_validation:
            final_validation = {
                **final_validation,
                **_validate_forward(
                    model,
                    validation_parameters,
                    validation_surfaces,
                    batch_size=train_config.validation_batch_size,
                ),
            }
        if "inverse_rmse_iv" not in final_validation:
            final_validation = {
                **final_validation,
                **_validate_inverse_surface(
                    model,
                    validation_parameters,
                    validation_surfaces,
                    innovation_mean=innovation_mean,
                    innovation_root=innovation_root,
                    kappa=kappa,
                    relative_change_cap=float(reset_law["relative_change_cap"]),
                    batch_size=train_config.validation_batch_size,
                    seed=train_config.seed + 1,
                ),
            }
        final_report = {
            "schema": "gambara-teichmann-crc-inverse-training-report-v1",
            "status": "complete",
            "dataset_directory": str(dataset_directory.resolve()),
            "dataset_quadrature": {
                "nodes": dataset_quadrature_nodes,
                "u_max": dataset_quadrature_u_max,
            },
            "target_fit_report": str(target_fit_report.resolve()),
            "train_samples": len(train_parameters),
            "heldout_samples": len(validation_parameters),
            "model_config": asdict(model_config),
            "model_version": model_config.version.unique_id48,
            "jump_output_bounds": {
                "lower": jump_lower.tolist(),
                "upper": jump_upper.tolist(),
            },
            "train_config": asdict(train_config),
            "heldout_metrics": final_validation,
            "elapsed_seconds": time.monotonic() - started,
        }
        temporary = output_directory / ".training-report.json.tmp"
        temporary.write_text(json.dumps(final_report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output_directory / "training-report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", type=Path, required=True)
    parser.add_argument("--target-fit-report", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--label", default="spx-2025-08-29-paper-aligned-v3")
    parser.add_argument("--forward-epochs", type=int, default=40)
    parser.add_argument("--inverse-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--inverse-learning-rate", type=float, default=2.0e-4)
    arguments = parser.parse_args()
    train_inverse_maps(
        dataset_directory=arguments.dataset_directory,
        target_fit_report=arguments.target_fit_report,
        output_directory=arguments.output_directory,
        model_config=CrcInverseNetworkConfig(label=arguments.label),
        train_config=CrcInverseTrainConfig(
            forward_epochs=arguments.forward_epochs,
            inverse_epochs=arguments.inverse_epochs,
            batch_size=arguments.batch_size,
            inverse_learning_rate=arguments.inverse_learning_rate,
        ),
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
