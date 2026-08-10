"""Full-universe training and one-sided-state analytics for conditional tails."""

from __future__ import annotations

import copy
import math
import os
import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock
from torch import Tensor

from quantdata.torch import BaseEnvironment

from .conditional_tail import (
    LONG_DIRECTION,
    SHORT_DIRECTION,
    TAIL_DIRECTIONS,
    ConditionalTailConfig,
    ConditionalTailQuantileModel,
)
from .direct_arbitrage import state_normalization
from .torch_drift_change import UniverseGeometry, validate_universe

_CHECKPOINT_SCHEMA = 1
_OBJECTIVE_SEMANTICS = "conditional-lower-quantile-raw-dc-v1"
_ROUNDOFF_MULTIPLIER = 32.0


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class ConditionalTailTrainConfig:
    """Operational configuration for conditional-tail optimization."""

    batch_size: int = 512
    valuation_batch_size: int = 2_048
    lr: float = 1.0e-3
    final_lr: float = 1.0e-6
    weight_decay: float = 1.0e-4
    compile: bool = True
    value_update: int = 5_000
    checkpoint_update: int = 10_000
    minimum_epochs: int = 50_000
    convergence_points: int = 5
    convergence_relative_tolerance: float = 2.0e-3
    convergence_log_coverage_tolerance: float = 0.10
    checkpoint_root: str = "~/.cache/rndysanos/conditional-tail"

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "valuation_batch_size",
            "value_update",
            "checkpoint_update",
            "minimum_epochs",
            "convergence_points",
        ):
            value = getattr(self, name)
            verify_inp(
                isinstance(value, int) and value > 0,
                lambda name=name, value=value: f"'{name}' must be positive, got {value}",
            )
        for name in ("lr", "final_lr"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: f"'{name}' must be finite and positive, got {value}",
            )
        for name in (
            "weight_decay",
            "convergence_relative_tolerance",
            "convergence_log_coverage_tolerance",
        ):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value >= 0.0,
                lambda name=name, value=value: f"'{name}' must be finite and nonnegative, got {value}",
            )


@dataclass
class ConditionalTailTracker:
    """Compact restartable optimization and full-valuation history."""

    epochs: list[int] = field(default_factory=list)
    batch_loss: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    gradient_norm: list[float] = field(default_factory=list)
    valuation_epochs: list[int] = field(default_factory=list)
    training_values: list[dict[str, Any]] = field(default_factory=list)
    validation_values: list[dict[str, Any]] = field(default_factory=list)
    restart_epochs: list[int] = field(default_factory=list)
    converged_epoch: int | None = None


@dataclass
class ConditionalTailTrainingResult:
    """Trained model, complete final valuations, and checkpoint metadata."""

    model: ConditionalTailQuantileModel
    training: dict[str, Any]
    validation: dict[str, Any]
    tracker: ConditionalTailTracker
    checkpoint_path: Path
    restarted: bool


def _floating_tensor(value: Any, *, name: str, env: BaseEnvironment) -> Tensor:
    array = np.asarray(value)
    verify_inp(
        np.issubdtype(array.dtype, np.floating),
        lambda: f"'{name}' must have floating dtype, got {array.dtype}",
    )
    tensor = torch.as_tensor(array, dtype=env.fdtype, device=env.device)
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"'{name}' contains non-finite values")
    return tensor


def prepare_raw_option_tail_data(
    universe: Mapping[str, Any],
    *,
    env: BaseEnvironment,
) -> tuple[dict[str, Tensor], UniverseGeometry]:
    """Prepare state, raw option price changes, values, and roundoff bands."""

    geometry = validate_universe(universe)
    S, TAU, T, N = (
        geometry.samples,
        geometry.steps,
        geometry.expiries,
        geometry.strikes,
    )
    xstate = _floating_tensor(universe["xstate"], name="xstate", env=env)[:, :TAU]
    log_strikes = _floating_tensor(
        universe["log_strikes"], name="log_strikes", env=env
    ).reshape(S, TAU, T * N)
    expiries = _floating_tensor(
        universe["expiries"], name="expiries", env=env
    ).reshape(1, 1, T).expand(S, TAU, T)
    returns = _floating_tensor(universe["dC"], name="dC", env=env).reshape(
        S, TAU, T * N
    )
    values = _floating_tensor(universe["C"], name="C", env=env).reshape(
        S, TAU, T * N
    )
    z = torch.cat((xstate, log_strikes, expiries), dim=-1)
    roundoff = _ROUNDOFF_MULTIPLIER * torch.finfo(torch.float32).eps * torch.maximum(
        torch.ones((), dtype=values.dtype, device=values.device),
        2.0 * values.abs(),
    )
    if not torch.isfinite(roundoff).all().item() or not torch.all(roundoff > 0.0).item():
        raise RuntimeError("raw-dC roundoff band is invalid")
    expected = (S, TAU, T * N)
    assert returns.shape == values.shape == roundoff.shape == expected
    assert z.shape == (S, TAU, geometry.state_features)
    return {
        "z": z,
        "returns": returns,
        "values": values,
        "roundoff": roundoff,
    }, geometry


def split_tail_data(
    data: Mapping[str, Tensor], *, validation_fraction: float
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Split paths contiguously so validation never enters normalization/training."""

    verify_inp(
        math.isfinite(validation_fraction) and 0.0 < validation_fraction < 1.0,
        lambda: f"validation fraction must be in (0, 1), got {validation_fraction}",
    )
    paths = data["z"].shape[0]
    validation_paths = int(math.floor(paths * validation_fraction))
    verify_inp(
        0 < validation_paths < paths,
        lambda: f"split of {paths} paths leaves an empty partition",
    )
    training_paths = paths - validation_paths
    return (
        {name: value[:training_paths] for name, value in data.items()},
        {name: value[training_paths:] for name, value in data.items()},
    )


def _iter_path_batches(
    data: Mapping[str, Tensor], batch_size: int
) -> Iterator[tuple[int, dict[str, Tensor]]]:
    paths = data["z"].shape[0]
    for start in range(0, paths, batch_size):
        stop = min(start + batch_size, paths)
        yield start, {name: value[start:stop] for name, value in data.items()}


def _sample_training_batch(
    data: Mapping[str, Tensor], *, batch_size: int, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    paths = data["z"].shape[0]
    indices = torch.randint(
        paths,
        (min(paths, batch_size),),
        device=data["z"].device,
        generator=generator,
    )
    return data["z"][indices], data["returns"][indices]


def _decode_example(
    *,
    flat_index: int,
    batch_paths: int,
    steps: int,
    instruments: int,
    batch_start: int,
    direction: int,
    level: int,
    quantiles: Tensor,
    batch: Mapping[str, Tensor],
) -> dict[str, Any]:
    del batch_paths
    path_in_batch, remainder = divmod(flat_index, steps * instruments)
    time_index, instrument = divmod(remainder, instruments)
    z = batch["z"][path_in_batch, time_index]
    return {
        "path": batch_start + path_in_batch,
        "time": time_index,
        "instrument": instrument,
        "direction": TAIL_DIRECTIONS[direction],
        "level_index": level,
        "quantile": float(quantiles[path_in_batch, time_index, direction, level, instrument].item()),
        "opposite_quantile": float(
            quantiles[path_in_batch, time_index, 1 - direction, level, instrument].item()
        ),
        "roundoff": float(batch["roundoff"][path_in_batch, time_index, instrument].item()),
        "realized_dC": float(batch["returns"][path_in_batch, time_index, instrument].item()),
        "option_value": float(batch["values"][path_in_batch, time_index, instrument].item()),
        "z": z.cpu().tolist(),
    }


@torch.no_grad()
def valuate_conditional_tail(
    model: ConditionalTailQuantileModel,
    data: Mapping[str, Tensor],
    *,
    batch_size: int,
    split: str,
    full: bool = True,
    example_count: int = 10,
) -> dict[str, Any]:
    """Value calibration and one-sided lower-tail candidates on a full split."""

    directions = 2
    levels = len(model.config.unique_effective_tail_probabilities)
    instruments = model.config.instruments
    output_shape = (directions, levels, instruments)
    device = model.device
    loss_total = torch.zeros((), dtype=torch.float64, device=device)
    loss_by_output = torch.zeros(output_shape, dtype=torch.float64, device=device)
    coverage_count = torch.zeros(output_shape, dtype=torch.int64, device=device)
    one_sided_count = torch.zeros(output_shape, dtype=torch.int64, device=device)
    strict_one_sided_count = torch.zeros(output_shape, dtype=torch.int64, device=device)
    realized_violation_count = torch.zeros(output_shape, dtype=torch.int64, device=device)
    both_sides_count = torch.zeros((levels, instruments), dtype=torch.int64, device=device)
    states_with_one_sided = torch.zeros((directions, levels), dtype=torch.int64, device=device)
    examples: list[list[list[dict[str, Any]]]] = [
        [[] for _ in range(levels)] for _ in range(directions)
    ]
    observations = 0
    states = 0
    model.eval()
    for batch_start, batch in _iter_path_batches(data, batch_size):
        quantiles = model.unique_quantiles(batch["z"])
        loss, by_output = model.loss(batch["z"], batch["returns"])
        batch_states = batch["z"].shape[0] * batch["z"].shape[1]
        loss_total += loss.to(torch.float64) * batch_states
        loss_by_output += by_output.to(torch.float64) * batch_states
        targets = torch.stack((batch["returns"], -batch["returns"]), dim=-2).unsqueeze(-2)
        coverage_count += (targets <= quantiles).sum(dim=(0, 1))
        tolerance = batch["roundoff"].unsqueeze(-2)
        long_positive = quantiles[..., LONG_DIRECTION, :, :] > tolerance
        short_positive = quantiles[..., SHORT_DIRECTION, :, :] > tolerance
        long_only = long_positive & ~short_positive
        short_only = short_positive & ~long_positive
        one_sided = torch.stack((long_only, short_only), dim=-3)
        one_sided_count += one_sided.sum(dim=(0, 1))
        strict_long = (quantiles[..., LONG_DIRECTION, :, :] > 0.0) & ~(
            quantiles[..., SHORT_DIRECTION, :, :] > 0.0
        )
        strict_short = (quantiles[..., SHORT_DIRECTION, :, :] > 0.0) & ~(
            quantiles[..., LONG_DIRECTION, :, :] > 0.0
        )
        strict_one_sided_count += torch.stack((strict_long, strict_short), dim=-3).sum(
            dim=(0, 1)
        )
        both_sides_count += (long_positive & short_positive).sum(dim=(0, 1))
        states_with_one_sided += one_sided.any(dim=-1).sum(dim=(0, 1))
        realized_long_loss = batch["returns"].unsqueeze(-2) < -tolerance
        realized_short_loss = batch["returns"].unsqueeze(-2) > tolerance
        realized_violation_count[LONG_DIRECTION] += (
            long_only & realized_long_loss
        ).sum(dim=(0, 1))
        realized_violation_count[SHORT_DIRECTION] += (
            short_only & realized_short_loss
        ).sum(dim=(0, 1))

        if full and example_count > 0:
            for direction, mask in enumerate((long_only, short_only)):
                margins = quantiles[..., direction, :, :] - tolerance
                for level in range(levels):
                    flat_margin = torch.where(
                        mask[..., level, :],
                        margins[..., level, :],
                        torch.full_like(margins[..., level, :], -torch.inf),
                    ).reshape(-1)
                    count = min(example_count, int(torch.count_nonzero(torch.isfinite(flat_margin)).item()))
                    if count == 0:
                        continue
                    top = torch.topk(flat_margin, count).indices.cpu().tolist()
                    examples[direction][level].extend(
                        _decode_example(
                            flat_index=index,
                            batch_paths=batch["z"].shape[0],
                            steps=batch["z"].shape[1],
                            instruments=instruments,
                            batch_start=batch_start,
                            direction=direction,
                            level=level,
                            quantiles=quantiles,
                            batch=batch,
                        )
                        for index in top
                    )
        observations += batch["returns"].numel()
        states += batch_states

    comparison_count = observations
    probabilities = torch.as_tensor(
        model.config.unique_effective_tail_probabilities,
        dtype=torch.float64,
        device=device,
    )
    # Each per-instrument output has one Bernoulli observation per sampled
    # state.  Dividing by all state-instrument comparisons here would apply
    # the instrument count twice when the vector is averaged below.
    coverage = coverage_count.to(torch.float64) / states
    count_by_direction_level = one_sided_count.sum(dim=-1)
    candidate_violation_rate = realized_violation_count.to(torch.float64) / one_sided_count.clamp_min(
        1
    )
    result: dict[str, Any] = {
        "split": split,
        "paths": data["z"].shape[0],
        "steps": data["z"].shape[1],
        "instruments": instruments,
        "states": states,
        "state_instrument_comparisons": comparison_count,
        "loss": float((loss_total / states).item()),
        "tail_probabilities": probabilities.cpu().tolist(),
        "tail_path_counts": [
            value * model.config.training_paths
            for value in model.config.unique_effective_tail_probabilities
        ],
        "coverage_by_direction_level": coverage.mean(dim=-1).cpu().tolist(),
        "coverage_ratio_by_direction_level": (
            coverage.mean(dim=-1) / probabilities.reshape(1, -1)
        ).cpu().tolist(),
        "pinball_by_direction_level": (
            loss_by_output.mean(dim=-1) / states
        ).cpu().tolist(),
        "roundoff_convention": (
            "32*float32_eps*max(1, 2*abs(C_t)); candidate lower quantile must exceed this band"
        ),
        "one_sided_count_by_direction_level": count_by_direction_level.cpu().tolist(),
        "one_sided_fraction_by_direction_level": (
            count_by_direction_level.to(torch.float64) / comparison_count
        ).cpu().tolist(),
        "strict_zero_one_sided_count_by_direction_level": strict_one_sided_count.sum(
            dim=-1
        ).cpu().tolist(),
        "states_with_any_one_sided_instrument_by_direction_level": states_with_one_sided.cpu().tolist(),
        "both_sides_positive_count_by_level": both_sides_count.sum(dim=-1).cpu().tolist(),
        "realized_counter_sign_count_by_direction_level": realized_violation_count.sum(
            dim=-1
        ).cpu().tolist(),
        "realized_counter_sign_rate_by_direction_level": (
            realized_violation_count.sum(dim=-1).to(torch.float64)
            / count_by_direction_level.clamp_min(1)
        ).cpu().tolist(),
        "has_any_roundoff_one_sided_state": bool(torch.any(one_sided_count > 0).item()),
    }
    if full:
        result.update(
            one_sided_count_by_direction_level_instrument=one_sided_count.cpu().tolist(),
            realized_counter_sign_rate_by_direction_level_instrument=candidate_violation_rate.cpu().tolist(),
            top_examples=[
                [
                    sorted(
                        level_examples,
                        key=lambda item: item["quantile"] - item["roundoff"],
                        reverse=True,
                    )[:example_count]
                    for level_examples in direction_examples
                ]
                for direction_examples in examples
            ],
        )
    for name, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"valuation output {name!r} is non-finite")
    return result


def _compact_value(value: dict[str, Any]) -> dict[str, Any]:
    names = (
        "loss",
        "coverage_by_direction_level",
        "coverage_ratio_by_direction_level",
        "one_sided_count_by_direction_level",
        "one_sided_fraction_by_direction_level",
        "realized_counter_sign_rate_by_direction_level",
        "both_sides_positive_count_by_level",
    )
    return {name: copy.deepcopy(value[name]) for name in names}


def _convergence(
    tracker: ConditionalTailTracker,
    *,
    config: ConditionalTailTrainConfig,
) -> dict[str, Any]:
    count = min(
        config.convergence_points,
        len(tracker.training_values),
        len(tracker.validation_values),
    )
    if count < config.convergence_points:
        return {"converged": False, "reason": "insufficient_full_valuations", "points": count}
    if tracker.valuation_epochs[-1] < config.minimum_epochs:
        return {"converged": False, "reason": "minimum_epochs_not_reached", "points": count}
    training_loss = np.asarray(
        [value["loss"] for value in tracker.training_values[-count:]], dtype=np.float64
    )
    validation_loss = np.asarray(
        [value["loss"] for value in tracker.validation_values[-count:]], dtype=np.float64
    )

    def relative_range(values: np.ndarray) -> float:
        return float(np.ptp(values) / max(abs(float(np.mean(values))), 1.0e-12))

    coverage = np.asarray(
        [
            value["coverage_ratio_by_direction_level"]
            for value in tracker.validation_values[-count:]
        ],
        dtype=np.float64,
    )
    log_coverage = np.log(np.clip(coverage, 1.0e-12, None))
    log_coverage_range = float(np.max(np.ptp(log_coverage, axis=0)))
    final_coverage = coverage[-1]
    coverage_calibrated = bool(np.all((final_coverage >= 0.5) & (final_coverage <= 2.0)))
    training_flat = relative_range(training_loss) <= config.convergence_relative_tolerance
    validation_flat = relative_range(validation_loss) <= config.convergence_relative_tolerance
    coverage_flat = log_coverage_range <= config.convergence_log_coverage_tolerance
    return {
        "converged": training_flat and validation_flat and coverage_flat and coverage_calibrated,
        "points": count,
        "training_loss_relative_range": relative_range(training_loss),
        "validation_loss_relative_range": relative_range(validation_loss),
        "validation_log_coverage_max_range": log_coverage_range,
        "final_validation_coverage_ratio": final_coverage.tolist(),
        "training_loss_flat": training_flat,
        "validation_loss_flat": validation_flat,
        "coverage_flat": coverage_flat,
        "coverage_calibrated_within_factor_two": coverage_calibrated,
    }


def _checkpoint_path(
    config: ConditionalTailConfig, train_config: ConditionalTailTrainConfig
) -> Path:
    root = Path(train_config.checkpoint_root).expanduser()
    if config.group:
        root = root.joinpath(*PurePath(config.group).parts)
    return root / f"{config.filename}.pt"


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_incompatibility(
    checkpoint: Mapping[str, Any], config: ConditionalTailConfig
) -> str | None:
    expected = {
        "schema": _CHECKPOINT_SCHEMA,
        "objective_semantics": _OBJECTIVE_SEMANTICS,
        "model_version": ConditionalTailQuantileModel.version.unique_id48,
        "config_version": ConditionalTailConfig.version.unique_id48,
        "user_version": config.user_version,
        "config": asdict(config),
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            return f"checkpoint {name} differs"
    return None


def train_conditional_tail(
    training: Mapping[str, Tensor],
    validation: Mapping[str, Tensor],
    *,
    config: ConditionalTailConfig,
    train_config: ConditionalTailTrainConfig,
    epochs: int,
    env: BaseEnvironment,
    restart: bool = False,
    progress: Callable[[int, dict[str, Any], dict[str, Any], dict[str, Any]], None]
    | None = None,
) -> ConditionalTailTrainingResult:
    """Fit, checkpoint, and fully value a conditional raw-dC tail model."""

    verify_inp(epochs > 0, lambda: f"'epochs' must be positive, got {epochs}")
    verify_inp(
        training["z"].shape[0] == config.training_paths,
        lambda: (
            f"config has {config.training_paths} training paths but data has "
            f"{training['z'].shape[0]}"
        ),
    )
    checkpoint_path = _checkpoint_path(config, train_config)
    checkpoint: dict[str, Any] | None = None
    if checkpoint_path.exists() and not restart:
        candidate = torch.load(checkpoint_path, map_location=env.device, weights_only=False)
        incompatibility = _checkpoint_incompatibility(candidate, config)
        if incompatibility is None:
            checkpoint = candidate
        else:
            warnings.warn(f"{incompatibility}; cold-starting {config.label!r}")

    if checkpoint is None:
        torch.manual_seed(config.seed)
        if env.device.type == "cuda":
            torch.cuda.manual_seed(config.seed)
        state_mean, state_scale = state_normalization(
            training, batch_size=train_config.valuation_batch_size
        )
    else:
        state_mean = checkpoint["state_mean"].to(env.device)
        state_scale = checkpoint["state_scale"].to(env.device)
    model = ConditionalTailQuantileModel(
        env,
        config=config,
        state_mean=state_mean,
        state_scale=state_scale,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )
    gamma = 1.0 if epochs <= 1 else (train_config.final_lr / train_config.lr) ** (
        1.0 / (epochs - 1)
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(config.seed + 1)
    tracker = ConditionalTailTracker()
    epoch = 0
    restarted = checkpoint is not None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        generator.set_state(checkpoint["generator_state"])
        tracker = ConditionalTailTracker(**checkpoint["tracker"])
        epoch = int(checkpoint["epoch"])
        tracker.restart_epochs.append(epoch)

    def save() -> None:
        _atomic_checkpoint(
            checkpoint_path,
            {
                "schema": _CHECKPOINT_SCHEMA,
                "objective_semantics": _OBJECTIVE_SEMANTICS,
                "model_version": ConditionalTailQuantileModel.version.unique_id48,
                "config_version": ConditionalTailConfig.version.unique_id48,
                "user_version": config.user_version,
                "config": asdict(config),
                "train_config": asdict(train_config),
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "generator_state": generator.get_state(),
                "state_mean": state_mean,
                "state_scale": state_scale,
                "tracker": asdict(tracker),
            },
        )

    def value() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        training_value = valuate_conditional_tail(
            model,
            training,
            batch_size=train_config.valuation_batch_size,
            split="training",
            full=False,
        )
        validation_value = valuate_conditional_tail(
            model,
            validation,
            batch_size=train_config.valuation_batch_size,
            split="validation",
            full=False,
        )
        if not tracker.valuation_epochs or tracker.valuation_epochs[-1] != epoch:
            tracker.valuation_epochs.append(epoch)
            tracker.training_values.append(_compact_value(training_value))
            tracker.validation_values.append(_compact_value(validation_value))
        else:
            tracker.training_values[-1] = _compact_value(training_value)
            tracker.validation_values[-1] = _compact_value(validation_value)
        convergence = _convergence(tracker, config=train_config)
        print(
            f"Tail epoch {epoch}/{epochs}: train={training_value['loss']:.8g}, "
            f"validation={validation_value['loss']:.8g}, validation coverage ratio="
            f"{validation_value['coverage_ratio_by_direction_level']}, "
            f"converged={convergence['converged']}",
            flush=True,
        )
        if progress is not None:
            progress(epoch, training_value, validation_value, convergence)
        return training_value, validation_value, convergence

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    training_kernel = (
        torch.compile(model.training_loss, dynamic=True)
        if train_config.compile
        else model.training_loss
    )
    final_training: dict[str, Any] | None = None
    final_validation: dict[str, Any] | None = None
    lock = FileLock(f"{checkpoint_path}.lock")
    with lock:
        try:
            for epoch_index in range(epoch, epochs):
                z, returns = _sample_training_batch(
                    training,
                    batch_size=train_config.batch_size,
                    generator=generator,
                )
                optimizer.zero_grad(set_to_none=True)
                loss, _ = training_kernel(z, returns)
                loss_value = float(loss.detach().cpu().item())
                if not math.isfinite(loss_value):
                    raise RuntimeError(
                        f"conditional-tail objective became non-finite on CPU at epoch {epoch_index + 1}"
                    )
                loss.backward()
                gradient_sqr = torch.zeros((), dtype=torch.float64, device=env.device)
                for parameter in model.parameters():
                    if parameter.grad is None:
                        continue
                    if not torch.isfinite(parameter.grad).all().item():
                        raise RuntimeError(f"non-finite gradient at epoch {epoch_index + 1}")
                    gradient_sqr += parameter.grad.to(torch.float64).square().sum()
                gradient_norm = float(gradient_sqr.sqrt().item())
                optimizer.step()
                epoch = epoch_index + 1
                if epoch < epochs:
                    scheduler.step()
                tracker.epochs.append(epoch)
                tracker.batch_loss.append(loss_value)
                tracker.learning_rate.append(float(optimizer.param_groups[0]["lr"]))
                tracker.gradient_norm.append(gradient_norm)
                if epoch == 1 or epoch % train_config.value_update == 0 or epoch == epochs:
                    final_training, final_validation, convergence = value()
                    save()
                    if convergence["converged"]:
                        tracker.converged_epoch = epoch
                        save()
                        break
                elif epoch % train_config.checkpoint_update == 0:
                    save()
        except BaseException:
            save()
            raise

    if final_training is None or final_validation is None:
        final_training, final_validation, _ = value()
    final_training = valuate_conditional_tail(
        model,
        training,
        batch_size=train_config.valuation_batch_size,
        split="training",
        full=True,
    )
    final_validation = valuate_conditional_tail(
        model,
        validation,
        batch_size=train_config.valuation_batch_size,
        split="validation",
        full=True,
    )
    save()
    return ConditionalTailTrainingResult(
        model=model,
        training=final_training,
        validation=final_validation,
        tracker=tracker,
        checkpoint_path=checkpoint_path,
        restarted=restarted,
    )


def process_memory(device: torch.device) -> dict[str, float]:
    """Return process and CUDA peak memory in GiB."""

    output: dict[str, float] = {}
    with Path("/proc/self/status").open(encoding="utf-8") as stream:
        for line in stream:
            name, _, value = line.partition(":")
            if name in {"VmRSS", "VmHWM"}:
                output[{"VmRSS": "rss_gib", "VmHWM": "peak_rss_gib"}[name]] = (
                    float(value.split()[0]) / 1024.0**2
                )
    if device.type == "cuda":
        output.update(
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024.0**3,
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024.0**3,
        )
    return output


__all__ = [
    "ConditionalTailTracker",
    "ConditionalTailTrainConfig",
    "ConditionalTailTrainingResult",
    "prepare_raw_option_tail_data",
    "process_memory",
    "split_tail_data",
    "train_conditional_tail",
    "valuate_conditional_tail",
]
