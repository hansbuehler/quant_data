"""Restartable AD-Seq-Vol and arbitrage-aware fine-tuning."""

from __future__ import annotations

import copy
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

from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    file_sha256,
    train_config_snapshot,
    training_data_fingerprint,
    validate_train_config_change,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionTrainConfig,
    DiffusionVariant,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.data import (
    DiffusionTrainingData,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.model import (
    SequentialVolatilityDenoiser,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.process import (
    cosine_beta_schedule,
    static_arbitrage_penalty,
)
from quantdata.torch import BaseEnvironment, torch

NORMALIZATION_POLICY = "source-fixed-v1"


@dataclass(frozen=True, kw_only=True)
class DiffusionTrainingResult:
    """Fitted denoiser, EMA model, and training diagnostics."""

    model: SequentialVolatilityDenoiser
    ema_model: SequentialVolatilityDenoiser
    completed_epochs: int
    training_loss: float
    validation_loss: float
    best_validation_loss: float
    checkpoint_path: Path
    warm_start_checkpoint: Path | None
    warm_start_sha256: str | None
    training_end_date: date
    converged: bool


def fit_diffusion(
    data: DiffusionTrainingData,
    *,
    model_config: DiffusionModelConfig,
    checkpoint_directory: Path,
    train_config: DiffusionTrainConfig = DiffusionTrainConfig(),
    base_checkpoint: Path | None = None,
    warm_start_checkpoint: Path | None = None,
    expected_base_checkpoint_sha256: str | None = None,
    expected_warm_start_sha256: str | None = None,
    dataset_manifest_sha256: str | None = None,
    environment: BaseEnvironment | None = None,
) -> DiffusionTrainingResult:
    """Fit one published diffusion variant with full restart state."""
    expected_tail = (
        model_config.sequence_length,
        model_config.channel_count,
        len(model_config.grid.expiries_y),
        model_config.grid.strike_count,
    )
    if data.sequences.shape[1:] != expected_tail:
        raise ValueError("diffusion data geometry does not match model config")
    if model_config.variant is DiffusionVariant.AD_SEQ_VOL_FT and base_checkpoint is None:
        raise ValueError("AD-Seq-Vol-FT requires a base AD-Seq-Vol checkpoint")
    if environment is None:
        environment = BaseEnvironment(precision=32, device="cpu", debug=False)
    data_fingerprint = _diffusion_data_fingerprint(data)
    precision = str(environment.fdtype)
    base_checkpoint = None if base_checkpoint is None else Path(base_checkpoint)
    base_checkpoint_fingerprint = (
        None if base_checkpoint is None else file_sha256(base_checkpoint)
    )
    if (
        expected_base_checkpoint_sha256 is not None
        and base_checkpoint_fingerprint != expected_base_checkpoint_sha256
    ):
        raise RuntimeError("base diffusion checkpoint changed before fitting")
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
        raise RuntimeError("diffusion warm-start checkpoint changed before fitting")
    checkpoint_directory = Path(checkpoint_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_name = named_unique_filename48_8(
        model_config.label,
        model_config,
        data_fingerprint,
        precision,
        base_checkpoint_fingerprint,
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
            base_checkpoint=base_checkpoint,
            environment=environment,
            data_fingerprint=data_fingerprint,
            precision=precision,
            base_checkpoint_fingerprint=base_checkpoint_fingerprint,
            warm_start_checkpoint=warm_start_checkpoint,
            warm_start_sha256=warm_start_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )


def _fit_locked(
    data: DiffusionTrainingData,
    *,
    model_config: DiffusionModelConfig,
    train_config: DiffusionTrainConfig,
    checkpoint_path: Path,
    base_checkpoint: Path | None,
    environment: BaseEnvironment,
    data_fingerprint: str,
    precision: str,
    base_checkpoint_fingerprint: str | None,
    warm_start_checkpoint: Path | None,
    warm_start_sha256: str | None,
    dataset_manifest_sha256: str | None,
) -> DiffusionTrainingResult:
    validation_count = max(1, int(round(len(data.sequences) * train_config.validation_fraction)))
    train_count = len(data.sequences) - validation_count
    if train_count < 2:
        raise ValueError("diffusion split has fewer than two training sequences")
    channel_mean = np.mean(data.sequences[:train_count], axis=(0, 1, 3, 4))
    channel_scale = np.maximum(
        np.std(data.sequences[:train_count], axis=(0, 1, 3, 4), ddof=0),
        1.0e-8,
    )
    torch.manual_seed(train_config.seed)
    if environment.device.type == "cuda":
        torch.cuda.manual_seed_all(train_config.seed)
    model = SequentialVolatilityDenoiser(
        environment,
        config=model_config,
        channel_mean=torch.as_tensor(
            channel_mean,
            dtype=environment.fdtype,
            device=environment.device,
        ),
        channel_scale=torch.as_tensor(
            channel_scale,
            dtype=environment.fdtype,
            device=environment.device,
        ),
    )
    if model_config.variant is DiffusionVariant.AD_SEQ_VOL_FT:
        _load_base_for_fine_tuning(
            model,
            Path(base_checkpoint),
            expected_sha256=base_checkpoint_fingerprint,
            precision=precision,
            current_training_end_date=data.target_dates[-1],
            expected_dataset_manifest_sha256=dataset_manifest_sha256,
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("output_adapter."))
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("diffusion training has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    steps_per_epoch = math.ceil(train_count / train_config.batch_size)
    total_steps = train_config.epochs * steps_per_epoch

    def learning_rate_multiplier(step: int) -> float:
        if train_config.warmup_steps > 0 and step < train_config.warmup_steps:
            return max((step + 1) / train_config.warmup_steps, 1.0e-8)
        denominator = max(total_steps - train_config.warmup_steps, 1)
        progress = min(max((step - train_config.warmup_steps) / denominator, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    amp_enabled = bool(train_config.mixed_precision and environment.device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    raw_sequences = torch.as_tensor(
        data.sequences,
        dtype=environment.fdtype,
        device=environment.device,
    )
    betas = cosine_beta_schedule(
        model_config.diffusion_steps,
        dtype=environment.fdtype,
        device=environment.device,
    )
    alphas = 1.0 - betas
    cumulative_alphas = torch.cumprod(alphas, dim=0)
    start_epoch = 0
    best_validation = math.inf
    global_step = 0
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
        if (
            payload.get("warm_start_source_sha256") is not None
            and payload.get("normalization_policy") != NORMALIZATION_POLICY
        ):
            raise ValueError("rolling diffusion checkpoint has an unsupported normalization policy")
        compatible = (
            payload.get("checkpoint_format_version") == CHECKPOINT_FORMAT_VERSION
            and payload.get("model_code_version") == model_config.version.unique_id48
            and payload.get("model_user_version") == model_config.user_version
            and payload.get("variant") == model_config.variant.value
            and payload.get("data_fingerprint") == data_fingerprint
            and payload.get("precision") == precision
            and payload.get("base_checkpoint_fingerprint")
            == base_checkpoint_fingerprint
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
                        "compile_model",
                        "allow_changed_config_warm_start",
                        "just_load",
                    }
                ),
                immutable_fields=frozenset({"seed", "validation_fraction"}),
                allow_warm_start=train_config.allow_changed_config_warm_start,
                model_name="diffusion",
            )
            model.load_state_dict(payload["model"])
            ema_model.load_state_dict(payload["ema_model"])
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            schedule_changed = bool(
                {
                    "learning_rate",
                    "warmup_steps",
                    "batch_size",
                    "validation_fraction",
                }
                & set(changed_train_fields)
            ) or payload["train_config"].get("epochs") != current_train_config.get(
                "epochs"
            )
            if schedule_changed:
                current_learning_rate = (
                    train_config.learning_rate
                    * learning_rate_multiplier(int(payload["global_step"]))
                )
                scheduler.base_lrs = [
                    train_config.learning_rate for _ in optimizer.param_groups
                ]
                scheduler._last_lr = [
                    current_learning_rate for _ in optimizer.param_groups
                ]
                for parameter_group in optimizer.param_groups:
                    parameter_group["initial_lr"] = train_config.learning_rate
                    parameter_group["lr"] = current_learning_rate
            if "weight_decay" in changed_train_fields:
                for parameter_group in optimizer.param_groups:
                    parameter_group["weight_decay"] = train_config.weight_decay
            scaler.load_state_dict(payload["scaler"])
            torch.set_rng_state(payload["torch_rng_state"])
            if environment.device.type == "cuda" and payload.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
            start_epoch = int(payload["epoch"]) + 1
            best_validation = float(payload["best_validation"])
            global_step = int(payload["global_step"])
            if changed_train_fields and (
                train_config.just_load or start_epoch >= train_config.epochs
            ):
                raise ValueError(
                    "changed diffusion training fields require additional epochs"
                )
        else:
            warnings.warn(
                f"Ignoring incompatible diffusion checkpoint {checkpoint_path}",
                stacklevel=2,
            )
    if start_epoch == 0 and warm_start_checkpoint is not None and not train_config.just_load:
        _load_rolling_warm_start(
            model,
            ema_model,
            checkpoint=warm_start_checkpoint,
            expected_sha256=warm_start_sha256,
            model_config_identity=model_config_identity,
            precision=precision,
            current_training_end_date=data.target_dates[-1],
            expected_dataset_manifest_sha256=dataset_manifest_sha256,
        )
    sequences = model.normalize(raw_sequences)
    if train_config.just_load or start_epoch >= train_config.epochs:
        if start_epoch == 0:
            raise FileNotFoundError(f"no compatible diffusion checkpoint at {checkpoint_path}")
        return _result(
            model,
            ema_model,
            sequences=sequences,
            train_count=train_count,
            cumulative_alphas=cumulative_alphas,
            model_config=model_config,
            train_config=train_config,
            completed_epochs=start_epoch,
            best_validation=best_validation,
            checkpoint_path=checkpoint_path,
            warm_start_checkpoint=warm_start_checkpoint,
            warm_start_sha256=warm_start_sha256,
            training_end_date=data.target_dates[-1],
            converged=start_epoch >= train_config.epochs,
        )

    forward_model = torch.compile(model) if train_config.compile_model else model
    started = time.monotonic()
    last_progress = started
    for epoch in range(start_epoch, train_config.epochs):
        model.train()
        permutation = torch.randperm(train_count, device=environment.device)
        for batch_start in range(0, train_count, train_config.batch_size):
            indices = permutation[batch_start : batch_start + train_config.batch_size]
            clean = sequences[indices]
            steps = torch.randint(
                0,
                model_config.diffusion_steps,
                (len(indices),),
                device=environment.device,
            )
            noise = torch.randn_like(clean)
            alpha_bar = cumulative_alphas[steps].reshape(-1, 1, 1, 1, 1)
            noisy = torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=environment.device.type,
                enabled=amp_enabled,
            ):
                predicted = forward_model(noisy, steps)
                loss = torch.mean((predicted - clean).square())
                return_channel = predicted[:, -1, 1]
                loss = loss + train_config.return_broadcast_penalty * torch.mean(
                    (return_channel - torch.mean(return_channel, dim=(-2, -1), keepdim=True)).square()
                )
                if model_config.variant is DiffusionVariant.AD_SEQ_VOL_FT:
                    predicted_economic = model.denormalize(predicted)
                    loss = loss + train_config.arbitrage_penalty * static_arbitrage_penalty(
                        predicted_economic[:, -1, 0],
                        config=model_config,
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            _update_ema(ema_model, model, decay=train_config.ema_decay)
            global_step += 1
        validation = _validation_loss(
            ema_model,
            sequences=sequences[train_count:],
            cumulative_alphas=cumulative_alphas,
            model_config=model_config,
            train_config=train_config,
            seed=train_config.seed + 1,
        )
        best_validation = min(best_validation, validation)
        completed_epochs = epoch + 1
        now = time.monotonic()
        checkpoint_due = completed_epochs % train_config.checkpoint_epochs == 0
        progress_due = now - last_progress >= train_config.progress_seconds
        if checkpoint_due or progress_due:
            _save_checkpoint(
                checkpoint_path,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_validation=best_validation,
                data_fingerprint=data_fingerprint,
                precision=precision,
                base_checkpoint_fingerprint=base_checkpoint_fingerprint,
                train_config=current_train_config,
                model_config_identity=model_config_identity,
                training_start_date=data.start_dates[0],
                training_end_date=data.target_dates[-1],
                warm_start_source_sha256=warm_start_sha256,
                dataset_manifest_sha256=dataset_manifest_sha256,
            )
            if progress_due:
                elapsed = now - started
                print(
                    json.dumps(
                        {
                            "stage": "diffusion-training",
                            "variant": model_config.variant.value,
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
    _save_checkpoint(
        checkpoint_path,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=train_config.epochs - 1,
        global_step=global_step,
        best_validation=best_validation,
        data_fingerprint=data_fingerprint,
        precision=precision,
        base_checkpoint_fingerprint=base_checkpoint_fingerprint,
        train_config=current_train_config,
        model_config_identity=model_config_identity,
        training_start_date=data.start_dates[0],
        training_end_date=data.target_dates[-1],
        warm_start_source_sha256=warm_start_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    return _result(
        model,
        ema_model,
        sequences=sequences,
        train_count=train_count,
        cumulative_alphas=cumulative_alphas,
        model_config=model_config,
        train_config=train_config,
        completed_epochs=train_config.epochs,
        best_validation=best_validation,
        checkpoint_path=checkpoint_path,
        warm_start_checkpoint=warm_start_checkpoint,
        warm_start_sha256=warm_start_sha256,
        training_end_date=data.target_dates[-1],
        converged=True,
    )


def _load_base_for_fine_tuning(
    model: SequentialVolatilityDenoiser,
    checkpoint: Path,
    *,
    expected_sha256: str | None,
    precision: str,
    current_training_end_date: date,
    expected_dataset_manifest_sha256: str | None,
) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"base diffusion checkpoint does not exist: {checkpoint}")
    if expected_sha256 is None or file_sha256(checkpoint) != expected_sha256:
        raise RuntimeError("base diffusion checkpoint changed while loading")
    payload = torch.load(checkpoint, map_location=model.device, weights_only=False)
    if file_sha256(checkpoint) != expected_sha256:
        raise RuntimeError("base diffusion checkpoint changed while loading")
    source_train_config = payload.get("train_config")
    source_epochs = (
        source_train_config.get("epochs")
        if isinstance(source_train_config, dict)
        else None
    )
    source_end_raw = payload.get("training_end_date")
    try:
        source_end = date.fromisoformat(str(source_end_raw))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "fine-tuning base checkpoint has no valid training end date"
        ) from error
    compatible = (
        payload.get("checkpoint_format_version") == CHECKPOINT_FORMAT_VERSION
        and payload.get("variant") == DiffusionVariant.AD_SEQ_VOL.value
        and payload.get("model_code_version") == model.config.version.unique_id48
        and payload.get("model_user_version") == model.config.user_version
        and payload.get("base_compatibility_identity")
        == _base_compatibility_identity(model.config)
        and payload.get("precision") == precision
        and payload.get("dataset_manifest_sha256")
        == expected_dataset_manifest_sha256
        and isinstance(source_epochs, int)
        and int(payload.get("epoch", -1)) + 1 >= source_epochs
    )
    if not compatible:
        raise ValueError("fine-tuning base checkpoint is incomplete or incompatible")
    if source_end > current_training_end_date:
        raise ValueError("fine-tuning base checkpoint uses future training data")
    missing, unexpected = model.load_state_dict(payload["ema_model"], strict=False)
    expected_missing = {
        "output_adapter.down.weight",
        "output_adapter.up.weight",
    }
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"base diffusion checkpoint geometry mismatch: missing={missing}, unexpected={unexpected}"
        )


def _validation_loss(
    model: SequentialVolatilityDenoiser,
    *,
    sequences: torch.Tensor,
    cumulative_alphas: torch.Tensor,
    model_config: DiffusionModelConfig,
    train_config: DiffusionTrainConfig,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    with torch.no_grad():
        steps = torch.full(
            (len(sequences),),
            model_config.diffusion_steps // 2,
            dtype=torch.long,
            device=model.device,
        )
        noise = torch.randn(
            sequences.shape,
            generator=generator,
            dtype=model.fdtype,
            device=model.device,
        )
        alpha_bar = cumulative_alphas[steps].reshape(-1, 1, 1, 1, 1)
        noisy = torch.sqrt(alpha_bar) * sequences + torch.sqrt(1.0 - alpha_bar) * noise
        predicted = model(noisy, steps)
        loss = torch.mean((predicted - sequences).square())
        return_channel = predicted[:, -1, 1]
        loss = loss + train_config.return_broadcast_penalty * torch.mean(
            (return_channel - torch.mean(return_channel, dim=(-2, -1), keepdim=True)).square()
        )
        if model_config.variant is DiffusionVariant.AD_SEQ_VOL_FT:
            economic = model.denormalize(predicted)
            loss = loss + train_config.arbitrage_penalty * static_arbitrage_penalty(
                economic[:, -1, 0],
                config=model_config,
            )
    return float(loss.cpu())


def _update_ema(
    ema_model: SequentialVolatilityDenoiser,
    model: SequentialVolatilityDenoiser,
    *,
    decay: float,
) -> None:
    with torch.no_grad():
        model_state = model.state_dict()
        for name, value in ema_model.state_dict().items():
            source = model_state[name]
            if value.is_floating_point():
                value.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                value.copy_(source)


def _result(
    model: SequentialVolatilityDenoiser,
    ema_model: SequentialVolatilityDenoiser,
    *,
    sequences: torch.Tensor,
    train_count: int,
    cumulative_alphas: torch.Tensor,
    model_config: DiffusionModelConfig,
    train_config: DiffusionTrainConfig,
    completed_epochs: int,
    best_validation: float,
    checkpoint_path: Path,
    warm_start_checkpoint: Path | None,
    warm_start_sha256: str | None,
    training_end_date: date,
    converged: bool,
) -> DiffusionTrainingResult:
    training = _validation_loss(
        ema_model,
        sequences=sequences[:train_count],
        cumulative_alphas=cumulative_alphas,
        model_config=model_config,
        train_config=train_config,
        seed=train_config.seed + 2,
    )
    validation = _validation_loss(
        ema_model,
        sequences=sequences[train_count:],
        cumulative_alphas=cumulative_alphas,
        model_config=model_config,
        train_config=train_config,
        seed=train_config.seed + 1,
    )
    return DiffusionTrainingResult(
        model=model,
        ema_model=ema_model,
        completed_epochs=completed_epochs,
        training_loss=training,
        validation_loss=validation,
        best_validation_loss=best_validation,
        checkpoint_path=checkpoint_path,
        warm_start_checkpoint=warm_start_checkpoint,
        warm_start_sha256=warm_start_sha256,
        training_end_date=training_end_date,
        converged=converged,
    )


def _save_checkpoint(
    path: Path,
    *,
    model: SequentialVolatilityDenoiser,
    ema_model: SequentialVolatilityDenoiser,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_validation: float,
    data_fingerprint: str,
    precision: str,
    base_checkpoint_fingerprint: str | None,
    train_config: dict[str, object],
    model_config_identity: str,
    training_start_date: date,
    training_end_date: date,
    warm_start_source_sha256: str | None,
    dataset_manifest_sha256: str | None,
) -> None:
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_code_version": model.config.version.unique_id48,
        "model_user_version": model.config.user_version,
        "model_config_identity": model_config_identity,
        "base_compatibility_identity": _base_compatibility_identity(model.config),
        "variant": model.config.variant.value,
        "data_fingerprint": data_fingerprint,
        "precision": precision,
        "base_checkpoint_fingerprint": base_checkpoint_fingerprint,
        "training_start_date": training_start_date.isoformat(),
        "training_end_date": training_end_date.isoformat(),
        "warm_start_source_sha256": warm_start_source_sha256,
        "normalization_policy": NORMALIZATION_POLICY,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "train_config": train_config,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if model.device.type == "cuda" else None
        ),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation": best_validation,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_rolling_warm_start(
    model: SequentialVolatilityDenoiser,
    ema_model: SequentialVolatilityDenoiser,
    *,
    checkpoint: Path,
    expected_sha256: str | None,
    model_config_identity: str,
    precision: str,
    current_training_end_date: date,
    expected_dataset_manifest_sha256: str | None,
) -> None:
    """Load an earlier diffusion fit under current-window normalization."""
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"diffusion warm-start checkpoint does not exist: {checkpoint}"
        )
    if expected_sha256 is None or file_sha256(checkpoint) != expected_sha256:
        raise RuntimeError("diffusion warm-start checkpoint changed while loading")
    payload = torch.load(
        checkpoint,
        map_location=model.device,
        weights_only=False,
    )
    if file_sha256(checkpoint) != expected_sha256:
        raise RuntimeError("diffusion warm-start checkpoint changed while loading")
    source_end_raw = payload.get("training_end_date")
    try:
        source_end = date.fromisoformat(str(source_end_raw))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "diffusion warm-start checkpoint has no valid training end date"
        ) from error
    source_train_config = payload.get("train_config")
    source_epochs = (
        source_train_config.get("epochs")
        if isinstance(source_train_config, dict)
        else None
    )
    compatible = (
        payload.get("checkpoint_format_version") == CHECKPOINT_FORMAT_VERSION
        and payload.get("model_code_version") == model.config.version.unique_id48
        and payload.get("model_user_version") == model.config.user_version
        and payload.get("model_config_identity") == model_config_identity
        and payload.get("variant") == model.config.variant.value
        and payload.get("precision") == precision
        and payload.get("dataset_manifest_sha256")
        == expected_dataset_manifest_sha256
        and isinstance(source_epochs, int)
        and int(payload.get("epoch", -1)) + 1 >= source_epochs
    )
    if not compatible:
        raise ValueError("diffusion warm-start checkpoint is incomplete or incompatible")
    if source_end >= current_training_end_date:
        raise ValueError(
            "diffusion warm-start checkpoint must predate the current training window end"
        )
    if (
        (source_end.year > 2022 or payload.get("warm_start_source_sha256") is not None)
        and payload.get("normalization_policy") != NORMALIZATION_POLICY
    ):
        raise ValueError("diffusion warm-start checkpoint has an unsupported normalization policy")

    model.load_state_dict(payload["model"])
    ema_model.load_state_dict(payload["ema_model"])


def _base_compatibility_identity(config: DiffusionModelConfig) -> str:
    """Identify geometry shared by base and fine-tuned diffusion variants."""
    return named_unique_filename48_8(
        "diffusion-base-compatibility",
        config.grid,
        config.sequence_length,
        config.channel_count,
        config.model_dimension,
        config.transformer_layers,
        config.attention_heads,
        config.feedforward_multiplier,
        config.dropout,
        config.diffusion_steps,
    )


def _rebase_denoiser_state(
    source_state: dict[str, torch.Tensor],
    *,
    current_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Preserve the deterministic map, but not the DDPM sampling law."""
    state = {name: value.detach().clone() for name, value in source_state.items()}
    old_mean = state["channel_mean"]
    old_scale = state["channel_scale"]
    new_mean = current_state["channel_mean"]
    new_scale = current_state["channel_scale"]
    surface_size = state["input_projection.weight"].shape[1] // len(old_mean)

    input_ratio = torch.repeat_interleave(new_scale / old_scale, surface_size)
    input_shift = torch.repeat_interleave(
        (new_mean - old_mean) / old_scale,
        surface_size,
    )
    input_weight = state["input_projection.weight"]
    old_input_weight = input_weight.clone()
    state["input_projection.weight"] = old_input_weight * input_ratio[None]
    state["input_projection.bias"] = (
        state["input_projection.bias"] + old_input_weight @ input_shift
    )

    output_ratio = torch.repeat_interleave(old_scale / new_scale, surface_size)
    output_shift = torch.repeat_interleave(
        (old_mean - new_mean) / new_scale,
        surface_size,
    )
    state["output_projection.weight"] = (
        output_ratio[:, None] * state["output_projection.weight"]
    )
    state["output_projection.bias"] = (
        output_shift + output_ratio * state["output_projection.bias"]
    )
    adapter_up = "output_adapter.up.weight"
    if adapter_up in state:
        state[adapter_up] = output_ratio[:, None] * state[adapter_up]
    state["channel_mean"] = new_mean.detach().clone()
    state["channel_scale"] = new_scale.detach().clone()
    return state


def _diffusion_data_fingerprint(data: DiffusionTrainingData) -> str:
    return training_data_fingerprint(
        arrays={"sequences": data.sequences},
        metadata={
            "schema": "diffusion-training-data-v1",
            "start_dates": data.start_dates,
            "target_dates": data.target_dates,
        },
    )
