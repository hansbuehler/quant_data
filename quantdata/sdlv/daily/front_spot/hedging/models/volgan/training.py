"""Restartable VolGAN training on the distributable SANOS panel."""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.uniquehash import named_unique_filename48_8
from filelock import FileLock

from quantdata.biz_calendar import NMS
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    file_sha256,
    train_config_snapshot,
    training_data_fingerprint,
    validate_train_config_change,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import (
    VolganModelConfig,
    VolganTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.data import (
    VolganTrainingData,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.model import (
    VolganDiscriminator,
    VolganGenerator,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.pca import VolganSurfacePca
from quantdata.torch import BaseEnvironment, torch


@dataclass(frozen=True, kw_only=True)
class VolganTrainingResult:
    """Trained VolGAN and auditable convergence diagnostics."""

    generator: VolganGenerator
    discriminator: VolganDiscriminator
    completed_epochs: int
    generator_loss: float
    discriminator_loss: float
    validation_loss: float
    best_validation_loss: float
    moneyness_penalty_weight: float
    maturity_penalty_weight: float
    checkpoint_path: Path
    warm_start_checkpoint: Path | None
    warm_start_sha256: str | None
    training_end_date: date
    converged: bool
    level_pca_explained_fraction: float | None = None
    increment_pca_explained_fraction: float | None = None


def fit_volgan(
    data: VolganTrainingData,
    *,
    model_config: VolganModelConfig,
    checkpoint_directory: Path,
    train_config: VolganTrainConfig = VolganTrainConfig(),
    warm_start_checkpoint: Path | None = None,
    expected_warm_start_sha256: str | None = None,
    dataset_manifest_sha256: str | None = None,
    environment: BaseEnvironment | None = None,
) -> VolganTrainingResult:
    """Fit the published conditional GAN with exact restart state."""
    if data.conditions.shape[1] != model_config.full_condition_dimension:
        raise ValueError("VolGAN condition dimension does not match model config")
    if data.targets.shape[1] != model_config.full_output_dimension:
        raise ValueError("VolGAN target dimension does not match model config")
    if environment is None:
        environment = BaseEnvironment(precision=64, device="cpu", debug=False)
    data_fingerprint = _volgan_data_fingerprint(data)
    precision = str(environment.fdtype)
    warm_start_checkpoint = (
        None if warm_start_checkpoint is None else Path(warm_start_checkpoint)
    )
    warm_start_sha256 = (
        None
        if warm_start_checkpoint is None
        else file_sha256(warm_start_checkpoint)
    )
    if (
        expected_warm_start_sha256 is not None
        and warm_start_sha256 != expected_warm_start_sha256
    ):
        raise RuntimeError("VolGAN warm-start checkpoint changed before fitting")
    checkpoint_directory = Path(checkpoint_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_name = named_unique_filename48_8(
        model_config.label,
        model_config,
        data_fingerprint,
        precision,
        warm_start_sha256,
        dataset_manifest_sha256,
    )
    checkpoint_path = checkpoint_directory / f"{checkpoint_name}.pt"
    with FileLock(checkpoint_path.with_suffix(".pt.lock")):
        return _fit_locked(
            data,
            model_config=model_config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            environment=environment,
            data_fingerprint=data_fingerprint,
            precision=precision,
            warm_start_checkpoint=warm_start_checkpoint,
            warm_start_sha256=warm_start_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )


def _fit_locked(
    data: VolganTrainingData,
    *,
    model_config: VolganModelConfig,
    train_config: VolganTrainConfig,
    checkpoint_path: Path,
    environment: BaseEnvironment,
    data_fingerprint: str,
    precision: str,
    warm_start_checkpoint: Path | None,
    warm_start_sha256: str | None,
    dataset_manifest_sha256: str | None,
) -> VolganTrainingResult:
    train_count = len(data.conditions) - max(
        1,
        int(round(len(data.conditions) * train_config.validation_fraction)),
    )
    if train_count < 2:
        raise ValueError("VolGAN split has fewer than two training rows")
    if model_config.pca_rank is None:
        surface_pca = None
    elif warm_start_checkpoint is None:
        surface_pca = VolganSurfacePca.fit(
            conditions=data.conditions,
            targets=data.targets,
            train_count=train_count,
            rank=model_config.pca_rank,
        )
    else:
        source = torch.load(warm_start_checkpoint, map_location="cpu", weights_only=False)
        source_state = source.get("generator", {})
        required = (
            "level_mean",
            "level_components",
            "increment_mean",
            "increment_components",
        )
        if any(name not in source_state for name in required):
            raise ValueError("VolGAN PCA warm start lacks a frozen source basis")
        if (
            source_state["level_components"].shape != (model_config.pca_rank, model_config.surface_size)
            or source_state["increment_components"].shape != (model_config.pca_rank, model_config.surface_size)
        ):
            raise ValueError("VolGAN PCA warm-start basis has incompatible geometry")
        surface_pca = VolganSurfacePca.from_frozen_basis(
            conditions=data.conditions,
            targets=data.targets,
            train_count=train_count,
            **{name: source_state[name].cpu().numpy() for name in required},
        )
    encoded_conditions = (
        data.conditions
        if surface_pca is None
        else surface_pca.encode_conditions(data.conditions)
    )
    encoded_targets = (
        data.targets if surface_pca is None else surface_pca.encode_targets(data.targets)
    )
    condition_mean = np.mean(encoded_conditions[:train_count], axis=0)
    condition_scale = np.maximum(
        np.std(encoded_conditions[:train_count], axis=0, ddof=0),
        1.0e-8,
    )
    target_mean = np.mean(encoded_targets[:train_count], axis=0)
    target_scale = np.maximum(
        np.std(encoded_targets[:train_count], axis=0, ddof=0),
        1.0e-8,
    )
    torch.manual_seed(train_config.seed)
    if environment.device.type == "cuda":
        torch.cuda.manual_seed_all(train_config.seed)
    generator = VolganGenerator(
        environment,
        config=model_config,
        condition_mean=torch.as_tensor(
            condition_mean,
            dtype=environment.fdtype,
            device=environment.device,
        ),
        condition_scale=torch.as_tensor(
            condition_scale,
            dtype=environment.fdtype,
            device=environment.device,
        ),
        target_mean=torch.as_tensor(
            target_mean,
            dtype=environment.fdtype,
            device=environment.device,
        ),
        target_scale=torch.as_tensor(
            target_scale,
            dtype=environment.fdtype,
            device=environment.device,
        ),
        **(
            {}
            if surface_pca is None
            else {
                name: torch.as_tensor(value, dtype=environment.fdtype, device=environment.device)
                for name, value in (
                    ("level_mean", surface_pca.level_mean),
                    ("level_components", surface_pca.level_components),
                    ("increment_mean", surface_pca.increment_mean),
                    ("increment_components", surface_pca.increment_components),
                )
            }
        ),
    )
    discriminator = VolganDiscriminator(environment, config=model_config)
    optimizer_class = (
        torch.optim.RMSprop if train_config.optimizer == "rmsprop" else torch.optim.Adam
    )
    generator_optimizer = optimizer_class(
        generator.parameters(),
        lr=train_config.generator_learning_rate,
    )
    discriminator_optimizer = optimizer_class(
        discriminator.parameters(),
        lr=train_config.discriminator_learning_rate,
    )
    conditions = torch.as_tensor(
        encoded_conditions,
        dtype=environment.fdtype,
        device=environment.device,
    )
    targets = torch.as_tensor(
        encoded_targets,
        dtype=environment.fdtype,
        device=environment.device,
    )
    full_conditions = torch.as_tensor(
        data.conditions,
        dtype=environment.fdtype,
        device=environment.device,
    )
    generator_forward = torch.compile(generator) if train_config.compile_models else generator
    discriminator_forward = (
        torch.compile(discriminator) if train_config.compile_models else discriminator
    )
    start_epoch = 0
    best_validation = math.inf
    gradient_ratio_m_sum = 0.0
    gradient_ratio_t_sum = 0.0
    gradient_ratio_count = 0
    moneyness_weight = 0.0
    maturity_weight = 0.0
    warm_started = False
    current_train_config = train_config_snapshot(train_config)
    model_config_identity = named_unique_filename48_8(
        model_config.label,
        model_config,
    )
    if checkpoint_path.exists():
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        compatible = (
            payload.get("checkpoint_format_version") == CHECKPOINT_FORMAT_VERSION
            and payload.get("model_code_version") == model_config.version.unique_id48
            and payload.get("model_user_version") == model_config.user_version
            and payload.get("data_fingerprint") == data_fingerprint
            and payload.get("precision") == precision
            and payload.get("model_config_identity") == model_config_identity
            and payload.get("warm_start_source_sha256") == warm_start_sha256
            and payload.get("dataset_manifest_sha256")
            == dataset_manifest_sha256
        )
        if compatible:
            changed_train_fields = validate_train_config_change(
                payload.get("train_config"),
                current_train_config,
                ignored_fields=frozenset(
                    {
                        "epochs",
                        "progress_seconds",
                        "checkpoint_epochs",
                        "compile_models",
                        "allow_changed_config_warm_start",
                        "just_load",
                    }
                ),
                immutable_fields=frozenset({"seed", "validation_fraction", "optimizer"}),
                allow_warm_start=train_config.allow_changed_config_warm_start,
                model_name="VolGAN",
            )
            generator.load_state_dict(payload["generator"])
            discriminator.load_state_dict(payload["discriminator"])
            generator_optimizer.load_state_dict(payload["generator_optimizer"])
            discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
            if "generator_learning_rate" in changed_train_fields:
                for parameter_group in generator_optimizer.param_groups:
                    parameter_group["lr"] = train_config.generator_learning_rate
            if "discriminator_learning_rate" in changed_train_fields:
                for parameter_group in discriminator_optimizer.param_groups:
                    parameter_group["lr"] = train_config.discriminator_learning_rate
            torch.set_rng_state(payload["torch_rng_state"])
            if environment.device.type == "cuda" and payload.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
            start_epoch = int(payload["epoch"]) + 1
            best_validation = float(payload["best_validation"])
            gradient_ratio_m_sum = float(payload["gradient_ratio_m_sum"])
            gradient_ratio_t_sum = float(payload["gradient_ratio_t_sum"])
            gradient_ratio_count = int(payload["gradient_ratio_count"])
            moneyness_weight = float(payload["moneyness_weight"])
            maturity_weight = float(payload["maturity_weight"])
            warm_started = bool(payload.get("warm_started", False))
            if changed_train_fields and (
                train_config.just_load or start_epoch >= train_config.epochs
            ):
                raise ValueError(
                    "changed VolGAN training fields require additional epochs"
                )
        else:
            warnings.warn(
                f"Ignoring incompatible VolGAN checkpoint {checkpoint_path}",
                stacklevel=2,
            )
    if start_epoch == 0 and warm_start_checkpoint is not None and not train_config.just_load:
        (
            gradient_ratio_m_sum,
            gradient_ratio_t_sum,
            gradient_ratio_count,
            moneyness_weight,
            maturity_weight,
        ) = _load_rolling_warm_start(
            generator,
            discriminator,
            checkpoint=warm_start_checkpoint,
            expected_sha256=warm_start_sha256,
            model_config_identity=model_config_identity,
            precision=precision,
            optimizer=train_config.optimizer,
            current_training_end_date=data.target_dates[-1],
            expected_dataset_manifest_sha256=dataset_manifest_sha256,
        )
        warm_started = True
    standardized_conditions = (
        conditions - generator.condition_mean
    ) / generator.condition_scale
    standardized_targets = (targets - generator.target_mean) / generator.target_scale
    if train_config.just_load or start_epoch >= train_config.epochs:
        if start_epoch == 0:
            raise FileNotFoundError(f"no compatible VolGAN checkpoint at {checkpoint_path}")
        return _training_result(
            generator,
            discriminator,
            standardized_conditions=standardized_conditions,
            standardized_targets=standardized_targets,
            raw_conditions=conditions,
            full_conditions=full_conditions,
            surface_pca=surface_pca,
            train_count=train_count,
            model_config=model_config,
            completed_epochs=start_epoch,
            best_validation=best_validation,
            moneyness_weight=moneyness_weight,
            maturity_weight=maturity_weight,
            checkpoint_path=checkpoint_path,
            warm_start_checkpoint=warm_start_checkpoint,
            warm_start_sha256=warm_start_sha256,
            training_end_date=data.target_dates[-1],
            converged=start_epoch >= train_config.epochs,
        )

    started = time.monotonic()
    last_progress = started
    for epoch in range(start_epoch, train_config.epochs):
        permutation = torch.randperm(train_count, device=environment.device)
        for batch_start in range(0, train_count, train_config.batch_size):
            indices = permutation[batch_start : batch_start + train_config.batch_size]
            condition_batch = standardized_conditions[indices]
            raw_condition_batch = conditions[indices]
            full_condition_batch = full_conditions[indices]
            real_batch = standardized_targets[indices]
            noise = torch.randn(
                (len(indices), model_config.noise_dimension),
                dtype=environment.fdtype,
                device=environment.device,
            )
            fake_batch = generator_forward(noise, raw_condition_batch)
            discriminator_optimizer.zero_grad(set_to_none=True)
            real_logits = discriminator_forward(condition_batch, real_batch)
            fake_logits = discriminator_forward(condition_batch, fake_batch.detach())
            discriminator_loss = 0.5 * (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    real_logits,
                    torch.ones_like(real_logits),
                )
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    fake_logits,
                    torch.zeros_like(fake_logits),
                )
            )
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(
                (len(indices), model_config.noise_dimension),
                dtype=environment.fdtype,
                device=environment.device,
            )
            fake_batch = generator_forward(noise, raw_condition_batch)
            fake_logits = discriminator_forward(condition_batch, fake_batch)
            adversarial = torch.nn.functional.binary_cross_entropy_with_logits(
                fake_logits,
                torch.ones_like(fake_logits),
            )
            moneyness_penalty, maturity_penalty = _surface_penalties(
                generator,
                raw_conditions=full_condition_batch,
                standardized_targets=fake_batch,
                model_config=model_config,
            )
            if not warm_started and epoch < train_config.gradient_matching_epochs:
                bce_norm = _gradient_norm(adversarial, generator)
                m_norm = _gradient_norm(moneyness_penalty, generator)
                t_norm = _gradient_norm(maturity_penalty, generator)
                if m_norm > 0.0 and t_norm > 0.0:
                    gradient_ratio_m_sum += bce_norm / m_norm
                    gradient_ratio_t_sum += bce_norm / t_norm
                    gradient_ratio_count += 1
                generator_loss = adversarial
            else:
                if gradient_ratio_count == 0:
                    raise RuntimeError("VolGAN gradient matching produced no finite ratios")
                if moneyness_weight == 0.0 and maturity_weight == 0.0:
                    moneyness_weight = gradient_ratio_m_sum / gradient_ratio_count
                    maturity_weight = gradient_ratio_t_sum / gradient_ratio_count
                generator_loss = (
                    adversarial
                    + moneyness_weight * moneyness_penalty
                    + maturity_weight * maturity_penalty
                )
            generator_loss.backward()
            generator_optimizer.step()

        validation = _validation_loss(
            generator,
            discriminator,
            standardized_conditions=standardized_conditions[train_count:],
            standardized_targets=standardized_targets[train_count:],
            raw_conditions=conditions[train_count:],
            full_conditions=full_conditions[train_count:],
            model_config=model_config,
            seed=train_config.seed,
            moneyness_weight=moneyness_weight,
            maturity_weight=maturity_weight,
        )
        best_validation = min(best_validation, validation)
        completed_epochs = epoch + 1
        now = time.monotonic()
        checkpoint_due = completed_epochs % train_config.checkpoint_epochs == 0
        progress_due = now - last_progress >= train_config.progress_seconds
        if checkpoint_due or progress_due:
            _save_checkpoint(
                checkpoint_path,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                epoch=epoch,
                best_validation=best_validation,
                gradient_ratio_m_sum=gradient_ratio_m_sum,
                gradient_ratio_t_sum=gradient_ratio_t_sum,
                gradient_ratio_count=gradient_ratio_count,
                moneyness_weight=moneyness_weight,
                maturity_weight=maturity_weight,
                data_fingerprint=data_fingerprint,
                precision=precision,
                train_config=current_train_config,
                model_config_identity=model_config_identity,
                training_start_date=data.condition_dates[0],
                training_end_date=data.target_dates[-1],
                warm_start_source_sha256=warm_start_sha256,
                warm_started=warm_started,
                dataset_manifest_sha256=dataset_manifest_sha256,
            )
            if progress_due:
                elapsed = now - started
                print(
                    json.dumps(
                        {
                            "stage": "volgan-training",
                            "epoch": completed_epochs,
                            "total_epochs": train_config.epochs,
                            "elapsed_seconds": elapsed,
                            "eta_seconds": elapsed / max(completed_epochs - start_epoch, 1)
                            * (train_config.epochs - completed_epochs),
                            "validation_loss": validation,
                        }
                    ),
                    flush=True,
                )
                last_progress = now
    if not warm_started and train_config.gradient_matching_epochs == train_config.epochs:
        if gradient_ratio_count == 0:
            raise RuntimeError("VolGAN gradient matching produced no finite ratios")
        moneyness_weight = gradient_ratio_m_sum / gradient_ratio_count
        maturity_weight = gradient_ratio_t_sum / gradient_ratio_count
    _save_checkpoint(
        checkpoint_path,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        epoch=train_config.epochs - 1,
        best_validation=best_validation,
        gradient_ratio_m_sum=gradient_ratio_m_sum,
        gradient_ratio_t_sum=gradient_ratio_t_sum,
        gradient_ratio_count=gradient_ratio_count,
        moneyness_weight=moneyness_weight,
        maturity_weight=maturity_weight,
        data_fingerprint=data_fingerprint,
        precision=precision,
        train_config=current_train_config,
        model_config_identity=model_config_identity,
        training_start_date=data.condition_dates[0],
        training_end_date=data.target_dates[-1],
        warm_start_source_sha256=warm_start_sha256,
        warm_started=warm_started,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    return _training_result(
        generator,
        discriminator,
        standardized_conditions=standardized_conditions,
        standardized_targets=standardized_targets,
        raw_conditions=conditions,
        full_conditions=full_conditions,
        surface_pca=surface_pca,
        train_count=train_count,
        model_config=model_config,
        completed_epochs=train_config.epochs,
        best_validation=best_validation,
        moneyness_weight=moneyness_weight,
        maturity_weight=maturity_weight,
        checkpoint_path=checkpoint_path,
        warm_start_checkpoint=warm_start_checkpoint,
        warm_start_sha256=warm_start_sha256,
        training_end_date=data.target_dates[-1],
        converged=True,
    )


def _surface_penalties(
    generator: VolganGenerator,
    *,
    raw_conditions: torch.Tensor,
    standardized_targets: torch.Tensor,
    model_config: VolganModelConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    decoded = generator.decode_full_targets(standardized_targets)
    current = raw_conditions[:, 3:].reshape(
        -1,
        len(model_config.grid.expiries_y),
        model_config.grid.strike_count,
    )
    next_surface = current + decoded[:, 1:].reshape_as(current)
    expiry_grid = next_surface.new_tensor(model_config.grid.expiries_y)
    if model_config.grid.nms_nodes is None:
        strike_grid = next_surface.new_tensor(model_config.grid.strikes_over_forward).log()
        strike_steps = torch.diff(strike_grid)[None, None]
    else:
        nodes = next_surface.new_tensor(model_config.grid.nms_nodes)
        atm_index = model_config.grid.nms_nodes.index(0.0)
        atm_volatility = torch.exp(next_surface[..., atm_index])
        scale = torch.sqrt(
            (NMS.SCALE**2 * NMS.EPS + atm_volatility.square() * expiry_grid[None, :])
            / (NMS.EPS + 1.0)
        )
        strike_steps = torch.diff(nodes)[None, None, :] * scale[..., None]
    strike_slopes = torch.diff(next_surface, dim=-1) / strike_steps
    expiry_slopes = torch.diff(next_surface, dim=-2) / torch.diff(expiry_grid)[None, :, None]
    return torch.mean(strike_slopes.square()), torch.mean(expiry_slopes.square())


def _gradient_norm(loss: torch.Tensor, model: VolganGenerator) -> float:
    gradients = torch.autograd.grad(
        loss,
        tuple(model.parameters()),
        retain_graph=True,
        allow_unused=False,
    )
    total = sum(torch.sum(value.square()) for value in gradients)
    return float(torch.sqrt(total).detach().cpu())


def _validation_loss(
    generator: VolganGenerator,
    discriminator: VolganDiscriminator,
    *,
    standardized_conditions: torch.Tensor,
    standardized_targets: torch.Tensor,
    raw_conditions: torch.Tensor,
    full_conditions: torch.Tensor,
    model_config: VolganModelConfig,
    seed: int,
    moneyness_weight: float,
    maturity_weight: float,
) -> float:
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        noise_generator = torch.Generator(device=generator.device)
        noise_generator.manual_seed(seed + 1)
        noise = torch.randn(
            (len(raw_conditions), model_config.noise_dimension),
            generator=noise_generator,
            dtype=generator.fdtype,
            device=generator.device,
        )
        fake = generator(noise, raw_conditions)
        fake_logits = discriminator(standardized_conditions, fake)
        adversarial = torch.nn.functional.binary_cross_entropy_with_logits(
            fake_logits,
            torch.ones_like(fake_logits),
        )
        moneyness, maturity = _surface_penalties(
            generator,
            raw_conditions=full_conditions,
            standardized_targets=fake,
            model_config=model_config,
        )
        moment_error = torch.mean(
            (torch.mean(fake, dim=0) - torch.mean(standardized_targets, dim=0)).square()
        )
        return float(
            (
                adversarial
                + moneyness_weight * moneyness
                + maturity_weight * maturity
                + moment_error
            ).cpu()
        )


def _training_result(
    generator: VolganGenerator,
    discriminator: VolganDiscriminator,
    *,
    standardized_conditions: torch.Tensor,
    standardized_targets: torch.Tensor,
    raw_conditions: torch.Tensor,
    full_conditions: torch.Tensor,
    surface_pca: VolganSurfacePca | None,
    train_count: int,
    model_config: VolganModelConfig,
    completed_epochs: int,
    best_validation: float,
    moneyness_weight: float,
    maturity_weight: float,
    checkpoint_path: Path,
    warm_start_checkpoint: Path | None,
    warm_start_sha256: str | None,
    training_end_date: date,
    converged: bool,
) -> VolganTrainingResult:
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        noise = torch.zeros(
            (train_count, model_config.noise_dimension),
            dtype=generator.fdtype,
            device=generator.device,
        )
        fake = generator(noise, raw_conditions[:train_count])
        generator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            discriminator(standardized_conditions[:train_count], fake),
            torch.ones(train_count, dtype=generator.fdtype, device=generator.device),
        )
        real_logits = discriminator(
            standardized_conditions[:train_count],
            standardized_targets[:train_count],
        )
        fake_logits = discriminator(standardized_conditions[:train_count], fake)
        discriminator_loss = 0.5 * (
            torch.nn.functional.binary_cross_entropy_with_logits(
                real_logits,
                torch.ones_like(real_logits),
            )
            + torch.nn.functional.binary_cross_entropy_with_logits(
                fake_logits,
                torch.zeros_like(fake_logits),
            )
        )
    validation = _validation_loss(
        generator,
        discriminator,
        standardized_conditions=standardized_conditions[train_count:],
        standardized_targets=standardized_targets[train_count:],
        raw_conditions=raw_conditions[train_count:],
        full_conditions=full_conditions[train_count:],
        model_config=model_config,
        seed=20_261_012,
        moneyness_weight=moneyness_weight,
        maturity_weight=maturity_weight,
    )
    return VolganTrainingResult(
        generator=generator,
        discriminator=discriminator,
        completed_epochs=completed_epochs,
        generator_loss=float(generator_loss.cpu()),
        discriminator_loss=float(discriminator_loss.cpu()),
        validation_loss=validation,
        best_validation_loss=best_validation,
        moneyness_penalty_weight=moneyness_weight,
        maturity_penalty_weight=maturity_weight,
        checkpoint_path=checkpoint_path,
        warm_start_checkpoint=warm_start_checkpoint,
        warm_start_sha256=warm_start_sha256,
        training_end_date=training_end_date,
        converged=converged,
        level_pca_explained_fraction=(
            None if surface_pca is None else surface_pca.level_explained_fraction
        ),
        increment_pca_explained_fraction=(
            None if surface_pca is None else surface_pca.increment_explained_fraction
        ),
    )


def _save_checkpoint(
    path: Path,
    *,
    generator: VolganGenerator,
    discriminator: VolganDiscriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation: float,
    gradient_ratio_m_sum: float,
    gradient_ratio_t_sum: float,
    gradient_ratio_count: int,
    moneyness_weight: float,
    maturity_weight: float,
    data_fingerprint: str,
    precision: str,
    train_config: dict[str, object],
    model_config_identity: str,
    training_start_date: date,
    training_end_date: date,
    warm_start_source_sha256: str | None,
    warm_started: bool,
    dataset_manifest_sha256: str | None,
) -> None:
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_code_version": generator.config.version.unique_id48,
        "model_user_version": generator.config.user_version,
        "model_config_identity": model_config_identity,
        "data_fingerprint": data_fingerprint,
        "precision": precision,
        "training_start_date": training_start_date.isoformat(),
        "training_end_date": training_end_date.isoformat(),
        "warm_start_source_sha256": warm_start_source_sha256,
        "warm_started": warm_started,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "train_config": train_config,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if generator.device.type == "cuda" else None
        ),
        "epoch": epoch,
        "best_validation": best_validation,
        "gradient_ratio_m_sum": gradient_ratio_m_sum,
        "gradient_ratio_t_sum": gradient_ratio_t_sum,
        "gradient_ratio_count": gradient_ratio_count,
        "moneyness_weight": moneyness_weight,
        "maturity_weight": maturity_weight,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_rolling_warm_start(
    generator: VolganGenerator,
    discriminator: VolganDiscriminator,
    *,
    checkpoint: Path,
    expected_sha256: str | None,
    model_config_identity: str,
    precision: str,
    optimizer: str,
    current_training_end_date: date,
    expected_dataset_manifest_sha256: str | None,
) -> tuple[float, float, int, float, float]:
    """Load an earlier fit while preserving its economic input/output map."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VolGAN warm-start checkpoint does not exist: {checkpoint}")
    if expected_sha256 is None or file_sha256(checkpoint) != expected_sha256:
        raise RuntimeError("VolGAN warm-start checkpoint changed while loading")
    payload = torch.load(
        checkpoint,
        map_location=generator.device,
        weights_only=False,
    )
    if file_sha256(checkpoint) != expected_sha256:
        raise RuntimeError("VolGAN warm-start checkpoint changed while loading")
    source_end_raw = payload.get("training_end_date")
    try:
        source_end = date.fromisoformat(str(source_end_raw))
    except (TypeError, ValueError) as error:
        raise ValueError("VolGAN warm-start checkpoint has no valid training end date") from error
    source_train_config = payload.get("train_config")
    source_epochs = (
        source_train_config.get("epochs")
        if isinstance(source_train_config, dict)
        else None
    )
    compatible = (
        payload.get("checkpoint_format_version") == CHECKPOINT_FORMAT_VERSION
        and payload.get("model_code_version") == generator.config.version.unique_id48
        and payload.get("model_user_version") == generator.config.user_version
        and payload.get("model_config_identity") == model_config_identity
        and payload.get("precision") == precision
        and isinstance(source_train_config, dict)
        and source_train_config.get("optimizer", "adam") == optimizer
        and payload.get("dataset_manifest_sha256")
        == expected_dataset_manifest_sha256
        and isinstance(source_epochs, int)
        and int(payload.get("epoch", -1)) + 1 >= source_epochs
    )
    if not compatible:
        raise ValueError("VolGAN warm-start checkpoint is incomplete or incompatible")
    if source_end >= current_training_end_date:
        raise ValueError(
            "VolGAN warm-start checkpoint must predate the current training window end"
        )

    generator_state = _rebase_generator_state(
        payload["generator"],
        current_state=generator.state_dict(),
        noise_dimension=generator.config.noise_dimension,
    )
    if generator.config.pca_rank is not None:
        for name in (
            "level_mean",
            "level_components",
            "increment_mean",
            "increment_components",
        ):
            if not torch.equal(generator_state[name], generator.state_dict()[name]):
                raise ValueError(f"VolGAN PCA warm-start basis changed: {name}")
    discriminator_state = _rebase_discriminator_state(
        payload["discriminator"],
        source_generator_state=payload["generator"],
        current_generator_state=generator.state_dict(),
        condition_dimension=generator.config.condition_dimension,
    )
    generator.load_state_dict(generator_state)
    discriminator.load_state_dict(discriminator_state)
    return (
        float(payload["gradient_ratio_m_sum"]),
        float(payload["gradient_ratio_t_sum"]),
        int(payload["gradient_ratio_count"]),
        float(payload["moneyness_weight"]),
        float(payload["maturity_weight"]),
    )


def _rebase_generator_state(
    source_state: dict[str, torch.Tensor],
    *,
    current_state: dict[str, torch.Tensor],
    noise_dimension: int,
) -> dict[str, torch.Tensor]:
    """Express one trained generator under current-window normalization."""
    state = {name: value.detach().clone() for name, value in source_state.items()}
    old_condition_mean = state["condition_mean"]
    old_condition_scale = state["condition_scale"]
    old_target_mean = state["target_mean"]
    old_target_scale = state["target_scale"]
    new_condition_mean = current_state["condition_mean"]
    new_condition_scale = current_state["condition_scale"]
    new_target_mean = current_state["target_mean"]
    new_target_scale = current_state["target_scale"]

    first_weight = state["linear1.weight"]
    old_condition_mean = old_condition_mean.to(first_weight)
    old_condition_scale = old_condition_scale.to(first_weight)
    new_condition_mean = new_condition_mean.to(first_weight)
    new_condition_scale = new_condition_scale.to(first_weight)
    old_condition_weight = first_weight[:, noise_dimension:].clone()
    condition_ratio = new_condition_scale / old_condition_scale
    condition_shift = (new_condition_mean - old_condition_mean) / old_condition_scale
    first_weight[:, noise_dimension:] = old_condition_weight * condition_ratio[None]
    state["linear1.bias"] = state["linear1.bias"] + old_condition_weight @ condition_shift

    output_weight = state["linear3.weight"]
    old_target_mean = old_target_mean.to(output_weight)
    old_target_scale = old_target_scale.to(output_weight)
    new_target_mean = new_target_mean.to(output_weight)
    new_target_scale = new_target_scale.to(output_weight)
    target_ratio = old_target_scale / new_target_scale
    state["linear3.weight"] = target_ratio[:, None] * state["linear3.weight"]
    state["linear3.bias"] = (
        old_target_mean + old_target_scale * state["linear3.bias"] - new_target_mean
    ) / new_target_scale
    for name in ("condition_mean", "condition_scale", "target_mean", "target_scale"):
        state[name] = current_state[name].detach().clone()
    return state


def _rebase_discriminator_state(
    source_state: dict[str, torch.Tensor],
    *,
    source_generator_state: dict[str, torch.Tensor],
    current_generator_state: dict[str, torch.Tensor],
    condition_dimension: int,
) -> dict[str, torch.Tensor]:
    """Express discriminator logits under current standardized coordinates."""
    state = {name: value.detach().clone() for name, value in source_state.items()}
    old_condition_mean = source_generator_state["condition_mean"]
    old_condition_scale = source_generator_state["condition_scale"]
    old_target_mean = source_generator_state["target_mean"]
    old_target_scale = source_generator_state["target_scale"]
    new_condition_mean = current_generator_state["condition_mean"]
    new_condition_scale = current_generator_state["condition_scale"]
    new_target_mean = current_generator_state["target_mean"]
    new_target_scale = current_generator_state["target_scale"]

    first_weight = state["linear1.weight"]
    old_condition_mean = old_condition_mean.to(first_weight)
    old_condition_scale = old_condition_scale.to(first_weight)
    old_target_mean = old_target_mean.to(first_weight)
    old_target_scale = old_target_scale.to(first_weight)
    new_condition_mean = new_condition_mean.to(first_weight)
    new_condition_scale = new_condition_scale.to(first_weight)
    new_target_mean = new_target_mean.to(first_weight)
    new_target_scale = new_target_scale.to(first_weight)
    condition_weight = first_weight[:, :condition_dimension].clone()
    target_weight = first_weight[:, condition_dimension:].clone()
    condition_ratio = new_condition_scale / old_condition_scale
    target_ratio = new_target_scale / old_target_scale
    condition_shift = (new_condition_mean - old_condition_mean) / old_condition_scale
    target_shift = (new_target_mean - old_target_mean) / old_target_scale
    first_weight[:, :condition_dimension] = condition_weight * condition_ratio[None]
    first_weight[:, condition_dimension:] = target_weight * target_ratio[None]
    state["linear1.bias"] = (
        state["linear1.bias"]
        + condition_weight @ condition_shift
        + target_weight @ target_shift
    )
    return state


def _volgan_data_fingerprint(data: VolganTrainingData) -> str:
    return training_data_fingerprint(
        arrays={"conditions": data.conditions, "targets": data.targets},
        metadata={
            "schema": "volgan-training-data-v1",
            "condition_dates": data.condition_dates,
            "target_dates": data.target_dates,
        },
    )
