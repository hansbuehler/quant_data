"""
Generic PyTorch training loop for in-memory AC datasets.

This module is intentionally model-agnostic. It assumes the caller already has
training and validation data in CPU memory as nested dictionaries / arrays and
can provide a ``step_fn`` that computes a loss (and optional metrics) from a
batch. Batches are produced directly by slicing the in-memory arrays, avoiding
per-sample ``DataLoader`` overhead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dys.env import environment
from dys.diagnostics import collect_nonfinite_or_missing_grads


Batch = Any
StepMetrics = dict[str, float]
StepReturn = Any
StepFunction = Callable[[Any, Batch], StepReturn]
ModelLike = Any
OptimizerLike = Any
SchedulerLike = Any
LoaderLike = Any
DeviceLike = Any
DatasetLike = Any
ArrayTree = Mapping[str, Any] | Sequence[Any]


@dataclass(frozen=True)
class WandbConfig:
    project: str = "dynamic-sanos"
    entity: str | None = None
    group: str | None = None
    name: str | None = None
    job_type: str = "train"
    tags: tuple[str, ...] = ()
    notes: str | None = None
    mode: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    eval_epochs: int = 1
    batch_size: int = 256
    eval_batch_size: int | None = None
    shuffle: bool = True
    num_workers: int = 0
    device: str | None = None
    max_grad_norm: float | None = None
    checkpoint_dir: str | os.PathLike[str] | None = None
    run_name: str | None = None
    resume_from: str | os.PathLike[str] | None = "latest"
    pin_memory: bool = True
    save_every_eval: bool = True
    log_train_metrics: bool = True
    debug_gradients: bool = False
    debug_gradients_limit: int = 40

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("'epochs' must be positive")
        if self.eval_epochs <= 0:
            raise ValueError("'eval_epochs' must be positive")
        if self.batch_size <= 0:
            raise ValueError("'batch_size' must be positive")
        if self.eval_batch_size is not None and self.eval_batch_size <= 0:
            raise ValueError("'eval_batch_size' must be positive when set")
        if self.num_workers < 0:
            raise ValueError("'num_workers' cannot be negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("'max_grad_norm' must be positive when set")
        if self.debug_gradients_limit <= 0:
            raise ValueError("'debug_gradients_limit' must be positive")


@dataclass
class TrainingResult:
    best_validation_loss: float
    best_checkpoint: Path
    latest_checkpoint: Path
    start_epoch: int
    end_epoch: int
    history: list[dict[str, float]] = field(default_factory=list)


class _MetricAccumulator:
    def __init__(self) -> None:
        self.weight = 0
        self.totals: dict[str, float] = {}

    def update(self, metrics: Mapping[str, float], batch_weight: int) -> None:
        self.weight += batch_weight
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * batch_weight

    def averages(self) -> dict[str, float]:
        if self.weight == 0:
            return {key: float("nan") for key in self.totals}
        return {key: total / self.weight for key, total in self.totals.items()}


def train_model(
    model: ModelLike,
    optimizer: OptimizerLike,
    train_data: ArrayTree,
    validation_data: ArrayTree,
    step_fn: StepFunction,
    *,
    config: TrainingConfig,
    wandb_config: WandbConfig | None = None,
    scheduler: SchedulerLike | None = None,
) -> TrainingResult:
    """
    Train a model for ``config.epochs`` epochs with periodic full evaluation.

    Parameters
    ----------
    model
        PyTorch model to train.
    optimizer
        Optimizer used for gradient updates.
    train_data, validation_data
        Nested mappings / sequences of CPU-resident numpy arrays, tensors,
        lists, and scalars. Leading dimensions are sliced directly to form
        batches.
    step_fn
        Callable taking ``(model, batch)`` and returning either a scalar loss or
        ``(loss, metrics)``. ``metrics`` is logged during both training and full
        evaluation.
    config
        Training loop configuration.
    wandb_config
        Optional Weights & Biases configuration. If omitted, WandB logging is
        disabled.
    scheduler
        Optional scheduler stepped once per epoch, after the training pass.
    """
    _require_torch()

    if config.num_workers != 0:
        warnings.warn("TrainingConfig.num_workers is ignored by the direct in-memory batch iterator.")

    train_dataset, train_num_examples = _prepare_array_tree(train_data)
    validation_dataset, validation_num_examples = _prepare_array_tree(validation_data)

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    eval_batch_size = config.eval_batch_size or config.batch_size

    checkpoint_dir = _resolve_checkpoint_dir(config)
    latest_checkpoint = checkpoint_dir / "latest.pt"
    best_checkpoint = checkpoint_dir / "best.pt"

    start_epoch = 0
    best_validation_loss = float("inf")
    if config.resume_from is not None:
        resume_path = _resolve_resume_path(config.resume_from, latest_checkpoint)
        if resume_path is not None:
            start_epoch, best_validation_loss = _load_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            warnings.warn(f"Resumed training from checkpoint '{resume_path}' at epoch {start_epoch}.")

    run = _init_wandb(
        model=model,
        config=config,
        wandb_config=wandb_config,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    history: list[dict[str, float]] = []
    training_start = perf_counter()

    for epoch in range(start_epoch + 1, config.epochs + 1):
        epoch_start = perf_counter()
        train_metrics, train_examples, train_batches = _train_one_epoch(
            model=model,
            optimizer=optimizer,
            data=train_dataset,
            num_examples=train_num_examples,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            step_fn=step_fn,
            device=device,
            max_grad_norm=config.max_grad_norm,
            pin_memory=config.pin_memory and device.type == "cuda",
            debug_gradients=config.debug_gradients,
            debug_gradients_limit=config.debug_gradients_limit,
        )
        if scheduler is not None:
            scheduler.step()

        epoch_seconds = perf_counter() - epoch_start
        log_row: dict[str, float] = {
            "epoch": float(epoch),
            "benchmark/epoch_seconds": epoch_seconds,
            "benchmark/train_examples_per_second": train_examples / max(epoch_seconds, 1e-12),
            "benchmark/train_batches_per_second": train_batches / max(epoch_seconds, 1e-12),
            "benchmark/elapsed_seconds": perf_counter() - training_start,
            "optimizer/lr": _current_learning_rate(optimizer),
        }
        if config.log_train_metrics:
            log_row.update({f"train/{key}": value for key, value in train_metrics.items()})

        should_evaluate = (epoch % config.eval_epochs == 0) or (epoch == config.epochs)
        if should_evaluate:
            eval_start = perf_counter()
            full_train_metrics, n_train_eval = evaluate_model(
                model,
                train_dataset,
                num_examples=train_num_examples,
                batch_size=eval_batch_size,
                step_fn=step_fn,
                device=device,
                pin_memory=config.pin_memory and device.type == "cuda",
            )
            validation_metrics, n_val_eval = evaluate_model(
                model,
                validation_dataset,
                num_examples=validation_num_examples,
                batch_size=eval_batch_size,
                step_fn=step_fn,
                device=device,
                pin_memory=config.pin_memory and device.type == "cuda",
            )
            eval_seconds = perf_counter() - eval_start

            log_row.update({f"eval/train_{key}": value for key, value in full_train_metrics.items()})
            log_row.update({f"eval/validation_{key}": value for key, value in validation_metrics.items()})
            log_row["benchmark/eval_seconds"] = eval_seconds
            log_row["benchmark/eval_examples_per_second"] = (n_train_eval + n_val_eval) / max(eval_seconds, 1e-12)

            validation_loss = validation_metrics.get("loss")
            if validation_loss is None:
                raise ValueError("Validation metrics must contain 'loss' for checkpoint selection")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                _save_checkpoint(
                    best_checkpoint,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_validation_loss=best_validation_loss,
                    config=config,
                )
                log_row["checkpoint/is_best"] = 1.0
            else:
                log_row["checkpoint/is_best"] = 0.0

            if config.save_every_eval:
                _save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_validation_loss=best_validation_loss,
                    config=config,
                )
        else:
            log_row["checkpoint/is_best"] = 0.0

        _save_checkpoint(
            latest_checkpoint,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_validation_loss=best_validation_loss,
            config=config,
        )

        history.append(log_row)
        if run is not None:
            run.log(log_row, step=epoch)

    if run is not None:
        run.finish()

    return TrainingResult(
        best_validation_loss=best_validation_loss,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        start_epoch=start_epoch,
        end_epoch=config.epochs,
        history=history,
    )


def evaluate_model(
    model: ModelLike,
    data: ArrayTree,
    *,
    num_examples: int,
    batch_size: int,
    step_fn: StepFunction,
    device: DeviceLike,
    pin_memory: bool = False,
) -> tuple[dict[str, float], int]:
    """Run a full evaluation pass and return averaged metrics and item count."""
    _require_torch()

    model.eval()
    accumulator = _MetricAccumulator()
    total_examples = 0
    with torch.no_grad():
        for batch in _iterate_batches(data, num_examples=num_examples, batch_size=batch_size, shuffle=False):
            batch = _move_to_device(batch, device, pin_memory=pin_memory)
            loss, metrics = _parse_step_output(step_fn(model, batch))
            batch_size = _infer_batch_size(batch)
            metrics = {"loss": float(loss.detach().item()), **metrics}
            accumulator.update(metrics, batch_size)
            total_examples += batch_size
    return accumulator.averages(), total_examples


def _train_one_epoch(
    *,
    model: ModelLike,
    optimizer: OptimizerLike,
    data: ArrayTree,
    num_examples: int,
    batch_size: int,
    shuffle: bool,
    step_fn: StepFunction,
    device: DeviceLike,
    max_grad_norm: float | None,
    pin_memory: bool,
    debug_gradients: bool,
    debug_gradients_limit: int,
) -> tuple[dict[str, float], int, int]:
    model.train()
    accumulator = _MetricAccumulator()
    total_examples = 0
    total_batches = 0

    for batch in _iterate_batches(data, num_examples=num_examples, batch_size=batch_size, shuffle=shuffle):
        batch = _move_to_device(batch, device, pin_memory=pin_memory)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = _parse_step_output(step_fn(model, batch))
        loss.backward()
        if debug_gradients:
            problems = collect_nonfinite_or_missing_grads((name, parameter) for name, parameter in model.named_parameters())
            if problems:
                raise RuntimeError("Gradient diagnostics failed before optimizer step:\n" + "\n".join(problems[:debug_gradients_limit]))
        if max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = _infer_batch_size(batch)
        metrics = {"loss": float(loss.detach().item()), **metrics}
        accumulator.update(metrics, batch_size)
        total_examples += batch_size
        total_batches += 1

    return accumulator.averages(), total_examples, total_batches


def _parse_step_output(output: StepReturn) -> tuple[Any, dict[str, float]]:
    if isinstance(output, tuple):
        loss, metrics = output
    else:
        loss, metrics = output, {}
    if not torch.is_tensor(loss):
        raise TypeError("step_fn must return a torch.Tensor loss or (loss, metrics)")
    parsed_metrics = {key: _metric_to_float(value) for key, value in dict(metrics).items()}
    return loss, parsed_metrics


def _metric_to_float(value: float | int | np.number | Any) -> float:
    if torch is not None and torch.is_tensor(value):
        return float(value.detach().item())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _prepare_array_tree(data: ArrayTree) -> tuple[ArrayTree, int]:
    normalized = _normalize_tree(data)
    length = _infer_tree_length(normalized)
    if length is None:
        raise ValueError("Could not infer dataset length from the provided in-memory data")
    return normalized, length


def _normalize_tree(node: Any) -> Any:
    if isinstance(node, Mapping):
        return {key: _normalize_tree(value) for key, value in node.items()}
    if isinstance(node, tuple):
        return tuple(_normalize_tree(value) for value in node)
    if isinstance(node, list):
        return [_normalize_tree(value) for value in node]
    return node


def _infer_tree_length(node: Any) -> int | None:
    if _is_array_like(node):
        return None if _leading_dimension(node) == 0 and _is_scalar_like(node) else _leading_dimension(node)
    if isinstance(node, Mapping):
        lengths = {_infer_tree_length(value) for value in node.values()}
        lengths.discard(None)
        if not lengths:
            return None
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent leading dimensions found in mapping: {sorted(lengths)}")
        return lengths.pop()
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        return len(node)
    return None


def _slice_tree(node: Any, indices: np.ndarray, expected_length: int) -> Any:
    if isinstance(node, Mapping):
        return {key: _slice_tree(value, indices, expected_length) for key, value in node.items()}
    if torch is not None and torch.is_tensor(node):
        if node.ndim == 0:
            return node
        if node.shape[0] == expected_length:
            return node[indices]
        return node
    if isinstance(node, np.ndarray):
        if node.ndim == 0:
            return node.item()
        if node.shape[0] == expected_length:
            return node[indices]
        return node
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        if len(node) == expected_length:
            sliced = [node[int(index)] for index in indices]
            return tuple(sliced) if isinstance(node, tuple) else sliced
        return node
    return node


def _move_to_device(node: Any, device: DeviceLike, *, pin_memory: bool = False) -> Any:
    if torch is not None and torch.is_tensor(node):
        if pin_memory and device.type == "cuda" and node.device.type == "cpu":
            node = node.pin_memory()
        return node.to(device, non_blocking=pin_memory and device.type == "cuda")
    if isinstance(node, np.ndarray):
        node = torch.as_tensor(node)
        if pin_memory and device.type == "cuda" and node.device.type == "cpu":
            node = node.pin_memory()
        return node.to(device, non_blocking=pin_memory and device.type == "cuda")
    if isinstance(node, Mapping):
        return {key: _move_to_device(value, device, pin_memory=pin_memory) for key, value in node.items()}
    if isinstance(node, tuple):
        return tuple(_move_to_device(value, device, pin_memory=pin_memory) for value in node)
    if isinstance(node, list):
        return [_move_to_device(value, device, pin_memory=pin_memory) for value in node]
    return node


def _iterate_batches(
    data: ArrayTree,
    *,
    num_examples: int,
    batch_size: int,
    shuffle: bool,
):
    indices = np.arange(num_examples, dtype=np.int64)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, num_examples, batch_size):
        yield _slice_tree(data, indices[start : start + batch_size], num_examples)


def _infer_batch_size(batch: Any) -> int:
    if isinstance(batch, Mapping):
        for value in batch.values():
            size = _infer_batch_size(value)
            if size > 0:
                return size
        return 0
    if torch is not None and torch.is_tensor(batch):
        return 1 if batch.ndim == 0 else int(batch.shape[0])
    if isinstance(batch, np.ndarray):
        return 1 if batch.ndim == 0 else int(batch.shape[0])
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes, bytearray)):
        if not batch:
            return 0
        if isinstance(batch[0], (str, bytes, bytearray)):
            return len(batch)
        return _infer_batch_size(batch[0])
    return 1


def _current_learning_rate(optimizer: OptimizerLike) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _resolve_checkpoint_dir(config: TrainingConfig) -> Path:
    if config.checkpoint_dir is not None:
        checkpoint_dir = Path(config.checkpoint_dir)
    else:
        checkpoint_dir = _default_checkpoint_dir(config.run_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _default_checkpoint_dir(run_name: str | None) -> Path:
    run_label = run_name or "default"
    base_path = getattr(environment.locations.training, "path", None)
    if base_path is not None:
        return Path(str(base_path)) / "wrds" / "ac" / "checkpoints" / run_label

    training_root = os.environ.get("DYS_TRAINING_DATA")
    if training_root is not None:
        return Path(training_root) / "wrds" / "ac" / "checkpoints" / run_label

    root = os.environ.get("DYS_ROOT")
    if root is not None:
        return Path(root) / "training" / "wrds" / "ac" / "checkpoints" / run_label

    return Path.cwd() / "checkpoints" / run_label


def _resolve_resume_path(resume_from: str | os.PathLike[str], latest_checkpoint: Path) -> Path | None:
    if str(resume_from).lower() == "latest":
        return latest_checkpoint if latest_checkpoint.exists() else None
    resume_path = Path(resume_from)
    return resume_path if resume_path.exists() else None


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: ModelLike,
    optimizer: OptimizerLike,
    scheduler: SchedulerLike | None,
    best_validation_loss: float,
    config: TrainingConfig,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "config": asdict(config),
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    *,
    model: ModelLike,
    optimizer: OptimizerLike,
    scheduler: SchedulerLike | None,
    device: DeviceLike,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_validation_loss", float("inf")))


def _init_wandb(
    *,
    model: ModelLike,
    config: TrainingConfig,
    wandb_config: WandbConfig | None,
    checkpoint_dir: Path,
    device: DeviceLike,
):
    if wandb_config is None or not wandb_config.enabled:
        return None

    run = wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        group=wandb_config.group,
        name=wandb_config.name or config.run_name,
        job_type=wandb_config.job_type,
        tags=list(wandb_config.tags),
        notes=wandb_config.notes,
        mode=wandb_config.mode,
        config={
            "training": asdict(config),
            "device": str(device),
            "checkpoint_dir": str(checkpoint_dir),
        },
    )
    run.watch(model, log="gradients", log_freq=max(100, config.eval_epochs))
    return run


def _is_array_like(node: Any) -> bool:
    return torch.is_tensor(node) or isinstance(node, np.ndarray)


def _leading_dimension(node: Any) -> int | None:
    if torch.is_tensor(node):
        return None if node.ndim == 0 else int(node.shape[0])
    if isinstance(node, np.ndarray):
        return None if node.ndim == 0 else int(node.shape[0])
    return None


def _is_scalar_like(node: Any) -> bool:
    if torch.is_tensor(node):
        return node.ndim == 0
    if isinstance(node, np.ndarray):
        return node.ndim == 0
    return np.isscalar(node)


def _require_torch() -> None:
    return None
