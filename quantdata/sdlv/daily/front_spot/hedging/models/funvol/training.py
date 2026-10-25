"""Chronological CPU fit and one-step simulation of single-asset FuNVol."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
from filelock import FileLock
from numpy.typing import NDArray

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    SanosFixedGridHistory,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    file_sha256,
    training_data_fingerprint,
)
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.config import (
    FunvolConfig,
    FunvolTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.fpc import (
    FunctionalPca,
    fit_functional_pca,
)
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.model import FunvolNeuralSde
from quantdata.torch import BaseEnvironment, torch

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class FunvolFit:
    """In-memory fitted functional basis, state scales, and neural SDE."""

    config: FunvolConfig
    fpc: FunctionalPca
    state_center: FloatArray
    state_scale: FloatArray
    model: FunvolNeuralSde
    training_loss: float
    validation_loss: float | None
    training_target_count: int
    validation_target_count: int
    last_training_surface_index: int
    completed_epochs: int = 0
    checkpoint_path: Path | None = None
    warm_start_date: str | None = None
    warm_start_sha256: str | None = None
    spot_log_anchor: float = 0.0


_CHECKPOINT_VERSION = 2


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _state_history(
    history: SanosFixedGridHistory,
    fpc: FunctionalPca,
    *,
    spot_log_anchor: float = 0.0,
) -> FloatArray:
    scores = fpc.encode(np.exp(history.log_implied_volatilities))
    returns = np.nan_to_num(history.forward_log_returns_into_date, nan=0.0)
    log_spot = spot_log_anchor + np.cumsum(returns)
    return np.column_stack((scores, log_spot))


def _windows(
    states: FloatArray,
    sessions: NDArray[np.int64],
    *,
    lag_count: int,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    windows = []
    targets = []
    indices = []
    for target_index in range(lag_count, len(states)):
        if np.any(sessions[target_index - lag_count + 1 : target_index + 1] != 1):
            continue
        windows.append(states[target_index - lag_count : target_index])
        targets.append(states[target_index] - states[target_index - 1])
        indices.append(target_index)
    if len(windows) < 4:
        raise ValueError("FuNVol needs at least four contiguous-session transition windows")
    return np.stack(windows), np.stack(targets), np.asarray(indices, dtype=np.int64)


def _gaussian_pit_loss(
    increment: torch.Tensor,
    mean: torch.Tensor,
    lower: torch.Tensor,
) -> torch.Tensor:
    """Smooth empirical-CDF penalty for marginal Gaussian PIT calibration."""
    variance = torch.sum(lower.square(), dim=-1).clamp_min(1.0e-10)
    score = (increment - mean) / torch.sqrt(variance)
    pit = 0.5 * (1.0 + torch.erf(score / np.sqrt(2.0)))
    grid = torch.linspace(0.1, 0.9, 9, dtype=pit.dtype, device=pit.device)
    empirical_cdf = torch.sigmoid((grid[None, None, :] - pit[..., None]) / 0.05).mean(0)
    return (empirical_cdf - grid).square().mean()


def _loss(
    model: FunvolNeuralSde,
    windows: torch.Tensor,
    increments: torch.Tensor,
    *,
    pit_weight: float,
) -> torch.Tensor:
    drift, lower = model(windows)
    dt = 1.0 / model.config.sessions_per_year
    mean = drift * dt
    covariance_factor = lower * np.sqrt(dt)
    distribution = torch.distributions.MultivariateNormal(
        loc=mean,
        scale_tril=covariance_factor,
    )
    likelihood = -distribution.log_prob(increments).mean()
    return likelihood + pit_weight * _gaussian_pit_loss(increments, mean, covariance_factor)


def fit_funvol(
    history: SanosFixedGridHistory,
    *,
    config: FunvolConfig = FunvolConfig(),
    train_config: FunvolTrainConfig = FunvolTrainConfig(),
    environment: BaseEnvironment | None = None,
    checkpoint_path: Path | None = None,
    warm_start_checkpoint: Path | None = None,
) -> FunvolFit:
    """Fit only on observations through the chronological training cutoff.

    FPC basis and normalization never see the held-out tail. A dated checkpoint
    resumes the same fit exactly, or inherits only neural weights from an
    explicitly supplied earlier-date checkpoint.
    """
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(path.with_suffix(path.suffix + ".lock")):
            return _fit_funvol_unlocked(
                history,
                config=config,
                train_config=train_config,
                environment=environment,
                checkpoint_path=path,
                warm_start_checkpoint=warm_start_checkpoint,
            )
    return _fit_funvol_unlocked(
        history,
        config=config,
        train_config=train_config,
        environment=environment,
        checkpoint_path=None,
        warm_start_checkpoint=warm_start_checkpoint,
    )


def _fit_funvol_unlocked(
    history: SanosFixedGridHistory,
    *,
    config: FunvolConfig,
    train_config: FunvolTrainConfig,
    environment: BaseEnvironment | None,
    checkpoint_path: Path | None,
    warm_start_checkpoint: Path | None,
) -> FunvolFit:
    env = environment or BaseEnvironment(device="cpu")
    if env.device.type != "cpu":
        raise ValueError("the standalone FuNVol fit is CPU-only")
    observation_count = len(history.dates)
    if observation_count < config.lag_count + config.factor_count + 6:
        raise ValueError("FuNVol history is too short")
    cutoff = int(np.floor(observation_count * (1.0 - train_config.validation_fraction)))
    if cutoff <= config.lag_count + config.factor_count + 2:
        raise ValueError("FuNVol training split is too short")
    as_of_date = history.dates[-1].isoformat()
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    saved = None
    if checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    elif warm_start_checkpoint is not None:
        saved = torch.load(Path(warm_start_checkpoint), map_location="cpu", weights_only=False)
    if saved is not None and (
        saved.get("version") != _CHECKPOINT_VERSION or saved.get("config") != repr(config)
    ):
        raise ValueError("FuNVol checkpoint geometry is incompatible")
    if (
        saved is not None
        and warm_start_checkpoint is not None
        and (checkpoint is None or not checkpoint.exists())
    ):
        previous_date = saved.get("as_of_date")
        if not isinstance(previous_date, str) or previous_date >= as_of_date:
            raise ValueError("FuNVol warm start must come from an earlier date")
        if int(saved.get("completed_epochs", 0)) < 1:
            raise ValueError("FuNVol warm-start checkpoint has no fitted epochs")
    surfaces = np.exp(history.log_implied_volatilities)
    fpc = (
        saved["fpc"] if saved is not None else fit_functional_pca(surfaces[:cutoff], config=config)
    )
    spot_log_anchor = 0.0
    if saved is not None:
        prior_dates = tuple(saved["history_dates"])
        try:
            origin_index = prior_dates.index(history.dates[0].isoformat())
        except ValueError as error:
            raise ValueError("FuNVol warm-start history has no overlapping spot anchor") from error
        prior_returns = np.asarray(saved["history_returns"], dtype=np.float64)
        spot_log_anchor = float(
            saved["spot_log_anchor"] + np.sum(prior_returns[1 : origin_index + 1])
        )
    states = _state_history(history, fpc, spot_log_anchor=spot_log_anchor)
    windows, increments, target_indices = _windows(
        states,
        history.sessions_from_previous_observation,
        lag_count=config.lag_count,
    )
    train_mask = target_indices < cutoff
    if np.count_nonzero(train_mask) < 2 or (
        train_config.validation_fraction > 0.0 and np.count_nonzero(~train_mask) < 1
    ):
        raise ValueError("FuNVol requires training and held-out contiguous transitions")
    train_states = states[:cutoff]
    if saved is not None:
        center = saved["state_center"]
        scale = saved["state_scale"]
    else:
        center = np.median(train_states, axis=0)
        scale = np.quantile(train_states, 0.75, axis=0) - np.quantile(train_states, 0.25, axis=0)
        scale = np.maximum(scale, 1.0e-5)
    normalized_windows = (windows - center) / scale
    normalized_increments = increments / scale
    fingerprint = training_data_fingerprint(
        arrays={
            "surfaces": history.log_implied_volatilities,
            "returns": history.forward_log_returns_into_date,
            "sessions": history.sessions_from_previous_observation,
        },
        metadata={"dates": [item.isoformat() for item in history.dates], "config": repr(config)},
    )
    torch.manual_seed(train_config.seed)
    model = FunvolNeuralSde(env, config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    tensor_windows = torch.as_tensor(normalized_windows, dtype=env.fdtype, device=env.device)
    tensor_increments = torch.as_tensor(normalized_increments, dtype=env.fdtype, device=env.device)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(~train_mask)
    generator = np.random.default_rng(train_config.seed)
    start_epoch = 0
    warm_start_date = None
    warm_start_sha256 = None
    if checkpoint is not None and checkpoint.exists():
        if (
            saved.get("as_of_date") != as_of_date
            or saved.get("fingerprint") != fingerprint
            or saved.get("config") != repr(config)
            or saved.get("train_config_except_epochs")
            != repr(train_config.__class__(**{**train_config.__dict__, "epochs": 1}))
        ):
            raise ValueError("FuNVol dated checkpoint identity is incompatible")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        start_epoch = int(saved["completed_epochs"])
        warm_start_date = saved["warm_start_date"]
        warm_start_sha256 = saved["warm_start_sha256"]
        if (
            warm_start_checkpoint is not None
            and file_sha256(Path(warm_start_checkpoint)) != warm_start_sha256
        ):
            raise ValueError("FuNVol warm-start lineage changed since dated checkpoint")
    elif warm_start_checkpoint is not None:
        previous_path = Path(warm_start_checkpoint)
        warm_start_date = saved.get("as_of_date")
        model.load_state_dict(saved["model"])
        warm_start_sha256 = file_sha256(previous_path)
    model.train()
    started_at = monotonic()
    last_report = started_at
    for epoch in range(start_epoch, train_config.epochs):
        for batch_indices in np.array_split(
            generator.permutation(train_indices),
            max(1, int(np.ceil(len(train_indices) / train_config.batch_size))),
        ):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = _loss(
                model,
                tensor_windows[batch_indices],
                tensor_increments[batch_indices],
                pit_weight=train_config.pit_weight,
            )
            if not torch.isfinite(batch_loss):
                raise FloatingPointError("FuNVol training loss is nonfinite")
            batch_loss.backward()
            optimizer.step()
        if checkpoint is not None:
            _save_checkpoint(
                checkpoint,
                {
                    "version": _CHECKPOINT_VERSION,
                    "as_of_date": as_of_date,
                    "fingerprint": fingerprint,
                    "config": repr(config),
                    "train_config_except_epochs": repr(
                        train_config.__class__(**{**train_config.__dict__, "epochs": 1})
                    ),
                    "completed_epochs": epoch + 1,
                    "model": deepcopy(model.state_dict()),
                    "fpc": fpc,
                    "state_center": center,
                    "state_scale": scale,
                    "spot_log_anchor": spot_log_anchor,
                    "history_dates": tuple(day.isoformat() for day in history.dates),
                    "history_returns": history.forward_log_returns_into_date,
                    "optimizer": optimizer.state_dict(),
                    "numpy_rng": generator.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "warm_start_date": warm_start_date,
                    "warm_start_sha256": warm_start_sha256,
                },
            )
        now = monotonic()
        if now - last_report >= 300.0 or epoch + 1 == train_config.epochs:
            completed = epoch + 1 - start_epoch
            remaining = train_config.epochs - epoch - 1
            eta = (now - started_at) * remaining / max(completed, 1)
            print(
                f"FuNVol {as_of_date}: {epoch + 1}/{train_config.epochs} epochs, "
                f"elapsed={now - started_at:.0f}s, eta={eta:.0f}s",
                flush=True,
            )
            last_report = now
    model.eval()
    with torch.no_grad():
        training_loss = float(
            _loss(
                model,
                tensor_windows[train_indices],
                tensor_increments[train_indices],
                pit_weight=train_config.pit_weight,
            )
        )
        validation_loss = (
            float(
                _loss(
                    model,
                    tensor_windows[validation_indices],
                    tensor_increments[validation_indices],
                    pit_weight=train_config.pit_weight,
                )
            )
            if len(validation_indices)
            else None
        )
    return FunvolFit(
        config=config,
        fpc=fpc,
        state_center=center,
        state_scale=scale,
        model=model,
        training_loss=training_loss,
        validation_loss=validation_loss,
        training_target_count=len(train_indices),
        validation_target_count=len(validation_indices),
        last_training_surface_index=cutoff - 1,
        completed_epochs=max(start_epoch, train_config.epochs),
        checkpoint_path=checkpoint,
        warm_start_date=warm_start_date,
        warm_start_sha256=warm_start_sha256,
        spot_log_anchor=spot_log_anchor,
    )


def sample_funvol_one_day(
    fit: FunvolFit,
    history: SanosFixedGridHistory,
    *,
    path_count: int,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Sample one-day forward log returns and next implied-volatility surfaces."""
    if path_count < 1 or len(history.dates) < fit.config.lag_count:
        raise ValueError("FuNVol sample requires positive paths and sufficient history")
    if np.any(history.sessions_from_previous_observation[-fit.config.lag_count + 1 :] != 1):
        raise ValueError("FuNVol sample context is not contiguous one-session history")
    states = _state_history(history, fit.fpc, spot_log_anchor=fit.spot_log_anchor)
    last_window = (states[-fit.config.lag_count :] - fit.state_center) / fit.state_scale
    env = fit.model.env
    context = torch.as_tensor(last_window, dtype=env.fdtype, device=env.device)
    context = context.unsqueeze(0).expand(path_count, -1, -1)
    generator = torch.Generator(device=env.device).manual_seed(seed)
    fit.model.eval()
    with torch.no_grad():
        drift, lower = fit.model(context)
        noise = torch.randn(
            path_count,
            fit.config.state_dimension,
            generator=generator,
            dtype=env.fdtype,
            device=env.device,
        )
        dt = 1.0 / fit.config.sessions_per_year
        normalized_increment = drift * dt + (lower @ noise[..., None]).squeeze(-1) * np.sqrt(dt)
    increments = normalized_increment.cpu().numpy() * fit.state_scale
    next_scores = states[-1, : fit.config.factor_count] + increments[:, : fit.config.factor_count]
    next_volatilities = fit.fpc.decode(next_scores)
    log_returns = increments[:, -1]
    if not np.all(np.isfinite(log_returns)) or not np.all(np.isfinite(next_volatilities)):
        raise FloatingPointError("FuNVol generated a nonfinite spot return or IV surface")
    return log_returns, next_volatilities
