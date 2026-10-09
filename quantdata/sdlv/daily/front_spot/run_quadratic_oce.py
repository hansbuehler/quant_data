"""Run cross-fitted quadratic second-moment no-arbitrage diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from quantdata.sdlv.daily.front_spot.oce import (
    FrontSpotOcePolicy,
    FrontSpotOcePolicyConfig,
    paired_path_average_statistics,
    quadratic_second_moment_reward,
)
from quantdata.sdlv.daily.front_spot.run_oce import (
    EXPIRY_DAYS,
    NORMALIZED_MONEYNESS,
    _active_mask,
    _instrument_labels,
)
from quantdata.sdlv.daily.front_spot.run_surface_oce import (
    POPULATION_SCHEMA,
    SurfacePopulation,
    _atomic_json,
    _atomic_torch,
    _dynamic_active_mask,
    _policy_from_state,
    _progress,
)
from quantdata.sdlv.daily.front_spot.simulation import load_initial_model_state

POLICY_SCHEMA = "front-spot-quadratic-second-moment-policy-v1"
PROJECTION_SCHEMA = "front-spot-all-node-to-restricted-projection-v1"
REPORT_SCHEMA = "front-spot-quadratic-second-moment-crossfit-v1"


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class QuadraticSecondMomentTrainConfig:
    """Controls for exact empirical quadratic-objective optimization."""

    batch_size: int = 2_048
    maximum_epochs: int = 30
    learning_rate: float = 1.0e-3
    maximum_gradient_norm: float = 10.0
    backtracking_factor: float = 0.2
    maximum_line_search_steps: int = 20
    lambda_second_moment: float = 1.0
    seed: int = 20261009

    def __post_init__(self) -> None:
        verify_inp(self.batch_size > 0, lambda: "batch size must be positive")
        verify_inp(self.maximum_epochs > 0, lambda: "maximum epochs must be positive")
        verify_inp(
            math.isfinite(self.learning_rate) and self.learning_rate > 0.0,
            lambda: "learning rate must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.maximum_gradient_norm)
            and self.maximum_gradient_norm > 0.0,
            lambda: "maximum gradient norm must be finite-positive",
        )
        verify_inp(
            0.0 < self.backtracking_factor < 1.0,
            lambda: "backtracking factor must lie strictly between zero and one",
        )
        verify_inp(
            self.maximum_line_search_steps > 0,
            lambda: "maximum line-search steps must be positive",
        )
        verify_inp(
            math.isfinite(self.lambda_second_moment)
            and self.lambda_second_moment >= 0.0,
            lambda: "second-moment lambda must be finite and non-negative",
        )


@dataclass(frozen=True)
class QuadraticPopulation:
    """Disk-backed features, returns, and predictable availability masks."""

    features: NDArray[np.float32]
    returns: NDArray[np.float32]
    label: str
    source: dict[str, Any]
    static_active_mask: NDArray[np.bool_] | None = None
    surface_population: SurfacePopulation | None = None

    def __post_init__(self) -> None:
        if self.features.shape[:2] != self.returns.shape[:2]:
            raise ValueError("population feature and return path-time axes must match")
        if (self.static_active_mask is None) == (self.surface_population is None):
            raise ValueError("population requires exactly one active-mask representation")
        if self.static_active_mask is not None:
            if self.static_active_mask.shape != self.returns.shape[1:]:
                raise ValueError("static active mask has incompatible geometry")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interpolation_plan(
    source_strikes: NDArray[np.float64],
    target_strikes: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Return bracketing source nodes and linear weights for each target strike."""

    if source_strikes.ndim != 2 or target_strikes.ndim != 2:
        raise ValueError("source and target strikes must be expiry-by-strike matrices")
    if source_strikes.shape[0] != target_strikes.shape[0]:
        raise ValueError("source and target expiry counts must match")
    lower = np.empty(target_strikes.shape, dtype=np.int64)
    upper = np.empty_like(lower)
    weight = np.empty(target_strikes.shape, dtype=np.float64)
    for expiry_index in range(source_strikes.shape[0]):
        source = source_strikes[expiry_index]
        target = target_strikes[expiry_index]
        if np.any(np.diff(source) <= 0.0):
            raise ValueError("source strikes must be strictly increasing")
        if np.min(target) < source[0] or np.max(target) > source[-1]:
            raise ValueError("target strikes lie outside the source strike grid")
        upper_row = np.searchsorted(source, target, side="right")
        upper_row = np.clip(upper_row, 1, len(source) - 1)
        lower_row = upper_row - 1
        denominator = source[upper_row] - source[lower_row]
        lower[expiry_index] = lower_row
        upper[expiry_index] = upper_row
        weight[expiry_index] = (target - source[lower_row]) / denominator
    return lower, upper, weight


def _project_all_node_batch_to_restricted(
    source_features: NDArray[np.float32],
    source_returns: NDArray[np.float32],
    *,
    factor_rank: int,
    source_strikes: NDArray[np.float64],
    target_strikes: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Project node calls to the original fixed-strike call/put book."""

    path_count, step_count, source_feature_count = source_features.shape
    source_strike_count = source_strikes.shape[1]
    target_strike_count = target_strikes.shape[1]
    source_instrument_count = 1 + len(EXPIRY_DAYS) * source_strike_count
    expected_feature_count = (
        factor_rank + source_instrument_count + 1 + len(EXPIRY_DAYS)
    )
    if source_feature_count != expected_feature_count:
        raise ValueError(
            f"source feature count is {source_feature_count}, expected "
            f"{expected_feature_count}"
        )
    if source_returns.shape != (path_count, step_count, source_instrument_count):
        raise ValueError("source return geometry is incompatible with its strike grid")

    target_instrument_count = 1 + len(EXPIRY_DAYS) * target_strike_count
    target_feature_count = (
        factor_rank + target_instrument_count + 1 + len(EXPIRY_DAYS)
    )
    target_features = np.zeros(
        (path_count, step_count, target_feature_count),
        dtype=np.float32,
    )
    target_returns = np.zeros(
        (path_count, step_count, target_instrument_count),
        dtype=np.float32,
    )
    target_features[..., : factor_rank + 1] = source_features[..., : factor_rank + 1]
    target_returns[..., 0] = source_returns[..., 0]

    source_option_start = factor_rank + 1
    source_option_stop = source_option_start + len(EXPIRY_DAYS) * source_strike_count
    target_option_start = factor_rank + 1
    target_option_stop = target_option_start + len(EXPIRY_DAYS) * target_strike_count
    target_features[..., target_option_stop:] = source_features[..., source_option_stop:]
    lower, upper, weight = _interpolation_plan(source_strikes, target_strikes)
    put_mask = np.asarray(NORMALIZED_MONEYNESS, dtype=np.float64) > 0.0
    if len(put_mask) != target_strike_count:
        raise ValueError("target strike count must match normalized-moneyness count")
    log_spot = np.asarray(source_features[..., factor_rank], dtype=np.float64)
    inverse_spot = np.exp(-log_spot)

    for expiry_index, expiry_day in enumerate(EXPIRY_DAYS):
        live = np.arange(step_count) < expiry_day
        source_feature_slice = slice(
            source_option_start + expiry_index * source_strike_count,
            source_option_start + (expiry_index + 1) * source_strike_count,
        )
        source_return_slice = slice(
            1 + expiry_index * source_strike_count,
            1 + (expiry_index + 1) * source_strike_count,
        )
        target_feature_slice = slice(
            target_option_start + expiry_index * target_strike_count,
            target_option_start + (expiry_index + 1) * target_strike_count,
        )
        target_return_slice = slice(
            1 + expiry_index * target_strike_count,
            1 + (expiry_index + 1) * target_strike_count,
        )
        source_call_ratio = np.exp(
            np.asarray(source_features[:, live, source_feature_slice], dtype=np.float64)
        )
        source_call_return = np.asarray(
            source_returns[:, live, source_return_slice],
            dtype=np.float64,
        )
        interpolation_weight = weight[expiry_index]
        target_call_ratio = (
            source_call_ratio[..., lower[expiry_index]]
            * (1.0 - interpolation_weight)
            + source_call_ratio[..., upper[expiry_index]] * interpolation_weight
        )
        target_call_return = (
            source_call_return[..., lower[expiry_index]]
            * (1.0 - interpolation_weight)
            + source_call_return[..., upper[expiry_index]] * interpolation_weight
        )
        target_value_ratio = np.where(
            put_mask,
            target_call_ratio
            - 1.0
            + target_strikes[expiry_index] * inverse_spot[:, live, None],
            target_call_ratio,
        )
        if np.any(~np.isfinite(target_value_ratio)) or np.min(target_value_ratio) < -1e-7:
            raise FloatingPointError("projected restricted option values are invalid")
        target_features[:, live, target_feature_slice] = np.log(
            np.maximum(target_value_ratio, 1.0e-12)
        ).astype(np.float32)
        projected_return = np.where(
            put_mask,
            target_call_return - source_returns[:, live, 0, None],
            target_call_return,
        )
        target_returns[:, live, target_return_slice] = projected_return.astype(np.float32)
    return target_features, target_returns


def _project_restricted_population(
    source_directory: Path,
    projection_directory: Path,
    reference_result: Path,
    *,
    factor_rank: int,
    dlv_substeps: int,
    batch_size: int,
) -> None:
    """Create or resume the restricted-book projection of retained Q paths."""

    source_metadata_path = source_directory / "metadata.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(source_metadata_path)
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    source_features = np.load(
        source_directory / "features.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    source_returns = np.load(
        source_directory / "returns.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    source_strikes = np.load(source_directory / "strikes.npy", allow_pickle=False)
    reference = json.loads(reference_result.read_text(encoding="utf-8"))
    target_strikes = np.asarray(reference["population"]["fixed_cash_strikes"])
    path_count, step_count = source_features.shape[:2]
    target_feature_count = factor_rank + 1 + 4 * len(NORMALIZED_MONEYNESS) + 1 + 4
    target_instrument_count = 1 + 4 * len(NORMALIZED_MONEYNESS)
    expected = {
        "schema": PROJECTION_SCHEMA,
        "path_count": path_count,
        "factor_rank": factor_rank,
        "dlv_substeps": dlv_substeps,
        "source_metadata_digest": _file_digest(source_metadata_path),
        "reference_result_digest": _file_digest(reference_result),
        "feature_shape": [path_count, step_count, target_feature_count],
        "return_shape": [path_count, step_count, target_instrument_count],
    }
    projection_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = projection_directory / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    if metadata.get("complete") and all(
        metadata.get(key) == value for key, value in expected.items()
    ):
        return
    if metadata and any(
        metadata.get(key) != value
        for key, value in expected.items()
        if key in metadata
    ):
        raise RuntimeError(f"incompatible restricted projection {projection_directory}")

    feature_path = projection_directory / "features.npy"
    return_path = projection_directory / "returns.npy"
    completed = int(metadata.get("completed_paths", 0))
    if completed:
        target_features = np.load(feature_path, mmap_mode="r+", allow_pickle=False)
        target_returns = np.load(return_path, mmap_mode="r+", allow_pickle=False)
    else:
        target_features = open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float32,
            shape=tuple(expected["feature_shape"]),
        )
        target_returns = open_memmap(
            return_path,
            mode="w+",
            dtype=np.float32,
            shape=tuple(expected["return_shape"]),
        )
        np.save(
            projection_directory / "active_mask.npy",
            _active_mask(len(NORMALIZED_MONEYNESS)),
            allow_pickle=False,
        )
        np.save(projection_directory / "strikes.npy", target_strikes, allow_pickle=False)
        np.save(
            projection_directory / "instrument_labels.npy",
            np.asarray(
                _instrument_labels(
                    len(NORMALIZED_MONEYNESS),
                    instrument_universe="restricted",
                )
            ),
            allow_pickle=False,
        )
    started = time.perf_counter()
    for start in range(completed, path_count, batch_size):
        stop = min(start + batch_size, path_count)
        features, returns = _project_all_node_batch_to_restricted(
            np.asarray(source_features[start:stop]),
            np.asarray(source_returns[start:stop]),
            factor_rank=factor_rank,
            source_strikes=np.asarray(source_strikes, dtype=np.float64),
            target_strikes=target_strikes,
        )
        target_features[start:stop] = features
        target_returns[start:stop] = returns
        target_features.flush()
        target_returns.flush()
        _atomic_json(
            {
                **expected,
                "source_directory": str(source_directory),
                "reference_result": str(reference_result),
                "completed_paths": stop,
                "complete": stop == path_count,
            },
            metadata_path,
        )
        _progress("project-restricted-population", stop, path_count, started)


def _load_restricted_population(
    projection_directory: Path,
    *,
    label: str,
) -> QuadraticPopulation:
    metadata = json.loads(
        (projection_directory / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("schema") != PROJECTION_SCHEMA or not metadata.get("complete"):
        raise RuntimeError(f"restricted projection is incomplete: {projection_directory}")
    return QuadraticPopulation(
        features=np.load(
            projection_directory / "features.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        returns=np.load(
            projection_directory / "returns.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        static_active_mask=np.load(
            projection_directory / "active_mask.npy",
            allow_pickle=False,
        ),
        label=label,
        source=metadata,
    )


def _load_surface_population(directory: Path, *, label: str) -> QuadraticPopulation:
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != POPULATION_SCHEMA or not metadata.get("complete"):
        raise RuntimeError(f"full-surface population is incomplete: {directory}")
    initial = load_initial_model_state(
        fit_label=str(metadata["fit_label"]),
        date_index=int(metadata["date_index"]),
    )
    surface = SurfacePopulation(
        features=np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False),
        returns=np.load(directory / "returns.npy", mmap_mode="r", allow_pickle=False),
        strikes=np.asarray(initial.support[1:-1], dtype=np.float64),
        factor_rank=int(metadata["factor_rank"]),
        nms_annualized_atm_volatility=float(
            metadata["nms_annualized_atm_volatility"]
        ),
        nms_lower_bound=float(metadata["nms_lower_bound"]),
        nms_upper_bound=float(metadata["nms_upper_bound"]),
        initial_date=str(metadata["initial_date"]),
        generation_seconds=float(metadata["generation_seconds"]),
    )
    return QuadraticPopulation(
        features=surface.features,
        returns=surface.returns,
        surface_population=surface,
        label=label,
        source={**metadata, "directory": str(directory)},
    )


def _active_mask_for_batch(
    population: QuadraticPopulation,
    feature_array: NDArray[np.float32],
) -> NDArray[np.bool_]:
    if population.static_active_mask is not None:
        return population.static_active_mask
    assert population.surface_population is not None
    return _dynamic_active_mask(feature_array, population.surface_population)


def _feature_normalization(
    population: QuadraticPopulation,
    start: int,
    stop: int,
    *,
    batch_size: int,
    label: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    feature_count = population.features.shape[-1]
    total = np.zeros(feature_count, dtype=np.float64)
    square = np.zeros(feature_count, dtype=np.float64)
    observation_count = 0
    started = time.perf_counter()
    for batch_start in range(start, stop, batch_size):
        batch_stop = min(batch_start + batch_size, stop)
        values = np.asarray(population.features[batch_start:batch_stop], dtype=np.float64)
        total += np.sum(values, axis=(0, 1), dtype=np.float64)
        square += np.sum(values**2, axis=(0, 1), dtype=np.float64)
        observation_count += values.shape[0] * values.shape[1]
        if batch_stop == stop or (batch_stop - start) % max(4_096, batch_size) == 0:
            _progress(f"normalize-{label}", batch_stop - start, stop - start, started)
    mean = total / observation_count
    variance = np.maximum(square / observation_count - mean**2, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def _objective_loss(
    policy: FrontSpotOcePolicy,
    population: QuadraticPopulation,
    start: int,
    stop: int,
    *,
    train_config: QuadraticSecondMomentTrainConfig,
    stage: str,
) -> float:
    reward_sum = 0.0
    observation_count = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_start in range(start, stop, train_config.batch_size):
            batch_stop = min(batch_start + train_config.batch_size, stop)
            feature_array = population.features[batch_start:batch_stop]
            feature = torch.as_tensor(
                feature_array,
                dtype=policy.fdtype,
                device=policy.device,
            )
            instrument_return = torch.as_tensor(
                population.returns[batch_start:batch_stop],
                dtype=policy.fdtype,
                device=policy.device,
            )
            active_mask = torch.as_tensor(
                _active_mask_for_batch(population, feature_array),
                dtype=torch.bool,
                device=policy.device,
            )
            action = policy(feature, active_mask)
            reward = quadratic_second_moment_reward(
                action,
                instrument_return,
                lambda_second_moment=train_config.lambda_second_moment,
            )
            reward_sum += float(torch.sum(reward.double()).cpu())
            observation_count += reward.numel()
            if batch_stop == stop or (batch_stop - start) % max(
                4_096, train_config.batch_size
            ) == 0:
                _progress(stage, batch_stop - start, stop - start, started)
    return -reward_sum / observation_count


def _checkpoint_payload(
    *,
    policy: FrontSpotOcePolicy,
    best_state: dict[str, torch.Tensor],
    best_loss: float,
    history: list[dict[str, float | int]],
    next_epoch: int,
    complete: bool,
    affine_only: bool,
    train_config: QuadraticSecondMomentTrainConfig,
    fold_start: int,
    fold_stop: int,
    population_label: str,
) -> dict[str, Any]:
    return {
        "schema": POLICY_SCHEMA,
        "policy_code_version": FrontSpotOcePolicyConfig.version.unique_id48,
        "policy_config": asdict(policy.config),
        "train_config": asdict(train_config),
        "fold_start": fold_start,
        "fold_stop": fold_stop,
        "population_label": population_label,
        "affine_only": affine_only,
        "current_state": {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        },
        "best_state": best_state,
        "best_loss": best_loss,
        "history": history,
        "next_epoch": next_epoch,
        "complete": complete,
        "torch_rng_state": torch.get_rng_state(),
    }


def _fit_policy(
    population: QuadraticPopulation,
    start: int,
    stop: int,
    *,
    label: str,
    checkpoint_path: Path,
    train_config: QuadraticSecondMomentTrainConfig,
    device: str,
    affine_only: bool,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[FrontSpotOcePolicy, list[dict[str, float | int]]]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but Torch cannot access it")
    torch.manual_seed(train_config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(train_config.seed)
    checkpoint: dict[str, Any] | None = None
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = {
            "schema": POLICY_SCHEMA,
            "policy_code_version": FrontSpotOcePolicyConfig.version.unique_id48,
            "fold_start": start,
            "fold_stop": stop,
            "population_label": population.label,
            "affine_only": affine_only,
        }
        mismatches = [
            key for key, value in expected.items() if checkpoint.get(key) != value
        ]
        previous_config = checkpoint.get("train_config", {})
        if previous_config.get("lambda_second_moment") != (
            train_config.lambda_second_moment
        ):
            mismatches.append("lambda_second_moment")
        if mismatches:
            raise RuntimeError(
                f"incompatible policy checkpoint {checkpoint_path}: "
                f"{', '.join(mismatches)}"
            )
        state_dict = checkpoint["current_state"]
        feature_mean = state_dict["feature_mean"].numpy()
        feature_scale = state_dict["feature_scale"].numpy()
    else:
        feature_mean, feature_scale = _feature_normalization(
            population,
            start,
            stop,
            batch_size=train_config.batch_size,
            label=label,
        )
        state_dict = initial_state_dict

    policy = _policy_from_state(
        label=label,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        feature_count=population.features.shape[-1],
        instrument_count=population.returns.shape[-1],
        device=device,
        state_dict=state_dict,
    )
    parameters = (
        tuple(policy.affine.parameters()) if affine_only else tuple(policy.parameters())
    )
    if checkpoint is None:
        history: list[dict[str, float | int]] = []
        next_epoch = 0
        best_loss = math.inf
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        }
    else:
        history = checkpoint["history"]
        next_epoch = int(checkpoint["next_epoch"])
        best_loss = float(checkpoint["best_loss"])
        best_state = checkpoint["best_state"]
        if next_epoch > train_config.maximum_epochs:
            raise ValueError("checkpoint has more epochs than the requested schedule")
        if checkpoint["complete"] and next_epoch == train_config.maximum_epochs:
            policy.load_state_dict(best_state)
            return policy, history
        if checkpoint["complete"]:
            policy.load_state_dict(best_state)

    training_started = time.perf_counter()
    observation_count = (stop - start) * population.features.shape[1]
    for epoch in range(next_epoch, train_config.maximum_epochs):
        loss = _objective_loss(
            policy,
            population,
            start,
            stop,
            train_config=train_config,
            stage=f"{label}-epoch-{epoch}-loss",
        )
        if loss < best_loss:
            best_loss = loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            }

        policy.zero_grad(set_to_none=True)
        gradient_started = time.perf_counter()
        for batch_start in range(start, stop, train_config.batch_size):
            batch_stop = min(batch_start + train_config.batch_size, stop)
            feature_array = population.features[batch_start:batch_stop]
            feature = torch.as_tensor(
                feature_array,
                dtype=policy.fdtype,
                device=policy.device,
            )
            instrument_return = torch.as_tensor(
                population.returns[batch_start:batch_stop],
                dtype=policy.fdtype,
                device=policy.device,
            )
            active_mask = torch.as_tensor(
                _active_mask_for_batch(population, feature_array),
                dtype=torch.bool,
                device=policy.device,
            )
            action = policy(feature, active_mask)
            reward = quadratic_second_moment_reward(
                action,
                instrument_return,
                lambda_second_moment=train_config.lambda_second_moment,
            )
            (-torch.sum(reward) / observation_count).backward()
            if batch_stop == stop or (batch_stop - start) % max(
                4_096, train_config.batch_size
            ) == 0:
                _progress(
                    f"{label}-epoch-{epoch}-gradient",
                    batch_stop - start,
                    stop - start,
                    gradient_started,
                )
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters,
                train_config.maximum_gradient_norm,
            )
        )
        direction_norm_square = float(
            sum(
                torch.sum(parameter.grad.detach().double().square()).cpu()
                for parameter in parameters
                if parameter.grad is not None
            )
        )
        original_parameters = tuple(
            parameter.detach().clone() for parameter in parameters
        )
        accepted_step = 0.0
        candidate_loss = loss
        line_search_evaluations = 0
        step_size = train_config.learning_rate
        for line_search_index in range(train_config.maximum_line_search_steps):
            line_search_evaluations = line_search_index + 1
            with torch.no_grad():
                for parameter, original in zip(
                    parameters,
                    original_parameters,
                    strict=True,
                ):
                    if parameter.grad is None:
                        parameter.copy_(original)
                    else:
                        parameter.copy_(original - step_size * parameter.grad)
            candidate_loss = _objective_loss(
                policy,
                population,
                start,
                stop,
                train_config=train_config,
                stage=f"{label}-epoch-{epoch}-line-search-{line_search_index}",
            )
            if candidate_loss < loss - 1.0e-15:
                accepted_step = step_size
                break
            step_size *= train_config.backtracking_factor
        if accepted_step == 0.0:
            with torch.no_grad():
                for parameter, original in zip(
                    parameters,
                    original_parameters,
                    strict=True,
                ):
                    parameter.copy_(original)

        record: dict[str, float | int] = {
            "epoch": epoch,
            "loss": loss,
            "candidate_loss": candidate_loss,
            "objective": -loss,
            "candidate_objective": -candidate_loss,
            "gradient_l2_norm_before_clipping": gradient_norm,
            "gradient_l2_norm_after_clipping": math.sqrt(direction_norm_square),
            "accepted_step_size": accepted_step,
            "line_search_evaluations": line_search_evaluations,
            "elapsed_seconds": time.perf_counter() - training_started,
        }
        history.append(record)
        _atomic_torch(
            _checkpoint_payload(
                policy=policy,
                best_state=best_state,
                best_loss=best_loss,
                history=history,
                next_epoch=epoch + 1,
                complete=False,
                affine_only=affine_only,
                train_config=train_config,
                fold_start=start,
                fold_stop=stop,
                population_label=population.label,
            ),
            checkpoint_path,
        )
        print(json.dumps({"stage": "train-epoch", "label": label, **record}), flush=True)
        if accepted_step == 0.0:
            break

    final_loss = _objective_loss(
        policy,
        population,
        start,
        stop,
        train_config=train_config,
        stage=f"{label}-final-loss",
    )
    if final_loss < best_loss:
        best_loss = final_loss
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        }
    policy.load_state_dict(best_state)
    _atomic_torch(
        _checkpoint_payload(
            policy=policy,
            best_state=best_state,
            best_loss=best_loss,
            history=history,
            next_epoch=len(history),
            complete=True,
            affine_only=affine_only,
            train_config=train_config,
            fold_start=start,
            fold_stop=stop,
            population_label=population.label,
        ),
        checkpoint_path,
    )
    return policy, history


def _evaluate_policy(
    policy: FrontSpotOcePolicy,
    population: QuadraticPopulation,
    start: int,
    stop: int,
    *,
    batch_size: int,
    lambda_second_moment: float,
    label: str,
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    path_average = np.empty(stop - start, dtype=np.float64)
    raw_return_sum = 0.0
    action_energy_sum = 0.0
    squared_portfolio_return_sum = 0.0
    objective_sum = 0.0
    maximum_absolute_action = 0.0
    observation_count = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_start in range(start, stop, batch_size):
            batch_stop = min(batch_start + batch_size, stop)
            relative = slice(batch_start - start, batch_stop - start)
            feature_array = population.features[batch_start:batch_stop]
            feature = torch.as_tensor(
                feature_array,
                dtype=policy.fdtype,
                device=policy.device,
            )
            instrument_return = torch.as_tensor(
                population.returns[batch_start:batch_stop],
                dtype=policy.fdtype,
                device=policy.device,
            )
            active_mask = torch.as_tensor(
                _active_mask_for_batch(population, feature_array),
                dtype=torch.bool,
                device=policy.device,
            )
            action = policy(feature, active_mask)
            portfolio_return = torch.sum(action * instrument_return, dim=-1)
            action_energy = torch.sum(action.square(), dim=-1)
            reward = (
                portfolio_return
                - 0.5 * action_energy
                - 0.5 * lambda_second_moment * portfolio_return.square()
            )
            portfolio_return_array = portfolio_return.double().cpu().numpy()
            path_average[relative] = np.mean(
                portfolio_return_array,
                axis=1,
                dtype=np.float64,
            )
            raw_return_sum += float(torch.sum(portfolio_return.double()).cpu())
            action_energy_sum += float(torch.sum(action_energy.double()).cpu())
            squared_portfolio_return_sum += float(
                torch.sum(portfolio_return.double().square()).cpu()
            )
            objective_sum += float(torch.sum(reward.double()).cpu())
            maximum_absolute_action = max(
                maximum_absolute_action,
                float(torch.max(torch.abs(action)).cpu()),
            )
            observation_count += portfolio_return.numel()
            if batch_stop == stop or (batch_stop - start) % max(4_096, batch_size) == 0:
                _progress(label, batch_stop - start, stop - start, started)
    paired_drift, _ = paired_path_average_statistics(path_average)
    summary = {
        "path_count": stop - start,
        "transition_count": observation_count,
        "mean_quadratic_objective": objective_sum / observation_count,
        "mean_raw_portfolio_return": raw_return_sum / observation_count,
        "expected_pointwise_action_energy": action_energy_sum / observation_count,
        "expected_squared_portfolio_return": (
            squared_portfolio_return_sum / observation_count
        ),
        "maximum_absolute_action": maximum_absolute_action,
        "paired_conditional_drift": paired_drift,
    }
    return summary, path_average


def run(arguments: argparse.Namespace) -> None:
    """Run one resumable two-fold quadratic conditional-drift experiment."""

    if arguments.paths < 4 or arguments.paths % 4:
        raise ValueError("path count must be divisible by four")
    arguments.output.mkdir(parents=True, exist_ok=True)
    arguments.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(arguments.torch_threads)
    lock = FileLock(arguments.checkpoint_dir / "quadratic-second-moment.lock")
    with lock:
        if arguments.population_kind == "restricted-projection":
            if arguments.source_population is None or arguments.reference_result is None:
                raise ValueError("restricted projection requires source and reference paths")
            _project_restricted_population(
                arguments.source_population,
                arguments.projection_cache,
                arguments.reference_result,
                factor_rank=arguments.factor_rank,
                dlv_substeps=arguments.dlv_substeps,
                batch_size=arguments.projection_batch_size,
            )
            population = _load_restricted_population(
                arguments.projection_cache,
                label=arguments.label,
            )
        else:
            population = _load_surface_population(
                arguments.surface_population,
                label=arguments.label,
            )
        if len(population.features) != arguments.paths:
            raise ValueError("population path count does not match --paths")
        train_config = QuadraticSecondMomentTrainConfig(
            batch_size=arguments.training_batch_size,
            maximum_epochs=arguments.maximum_epochs,
            learning_rate=arguments.learning_rate,
            maximum_gradient_norm=arguments.maximum_gradient_norm,
            lambda_second_moment=arguments.lambda_second_moment,
            seed=arguments.policy_seed,
        )
        split = arguments.paths // 2
        policies: dict[str, FrontSpotOcePolicy] = {}
        histories: dict[str, list[dict[str, float | int]]] = {}
        for fold_label, fold_start, fold_stop in (
            ("A", 0, split),
            ("B", split, arguments.paths),
        ):
            affine_policy, affine_history = _fit_policy(
                population,
                fold_start,
                fold_stop,
                label=f"{arguments.label}-affine-{fold_label}",
                checkpoint_path=arguments.checkpoint_dir
                / f"affine_{fold_label}.checkpoint.pt",
                train_config=train_config,
                device=arguments.device,
                affine_only=True,
            )
            affine_state = {
                name: value.detach().cpu().clone()
                for name, value in affine_policy.state_dict().items()
            }
            policy, history = _fit_policy(
                population,
                fold_start,
                fold_stop,
                label=f"{arguments.label}-{fold_label}",
                checkpoint_path=arguments.checkpoint_dir
                / f"policy_{fold_label}.checkpoint.pt",
                train_config=train_config,
                device=arguments.device,
                affine_only=False,
                initial_state_dict=affine_state,
            )
            policies[fold_label] = policy
            histories[f"affine_{fold_label}"] = affine_history
            histories[fold_label] = history

        training_A, _ = _evaluate_policy(
            policies["A"],
            population,
            0,
            split,
            batch_size=train_config.batch_size,
            lambda_second_moment=train_config.lambda_second_moment,
            label="evaluate-A-on-A",
        )
        heldout_B, scores_B = _evaluate_policy(
            policies["A"],
            population,
            split,
            arguments.paths,
            batch_size=train_config.batch_size,
            lambda_second_moment=train_config.lambda_second_moment,
            label="evaluate-A-on-B",
        )
        training_B, _ = _evaluate_policy(
            policies["B"],
            population,
            split,
            arguments.paths,
            batch_size=train_config.batch_size,
            lambda_second_moment=train_config.lambda_second_moment,
            label="evaluate-B-on-B",
        )
        heldout_A, scores_A = _evaluate_policy(
            policies["B"],
            population,
            0,
            split,
            batch_size=train_config.batch_size,
            lambda_second_moment=train_config.lambda_second_moment,
            label="evaluate-B-on-A",
        )
        out_of_fold_scores = np.concatenate((scores_A, scores_B))
        out_of_fold_summary, out_of_fold_pair_scores = paired_path_average_statistics(
            out_of_fold_scores
        )
        score_path = arguments.output / f"{arguments.label}_oof_pair_scores.npy"
        np.save(score_path, out_of_fold_pair_scores, allow_pickle=False)
        report = {
            "schema": REPORT_SCHEMA,
            "objective": {
                "definition": (
                    "sup_a E[a(X)'R-|a(X)|^2/2-"
                    "lambda*(a(X)'R)^2/2]"
                ),
                "lambda_second_moment": train_config.lambda_second_moment,
                "covariance_implementation": (
                    "sample portfolio-return square; no covariance matrix is formed"
                ),
            },
            "population": {
                "label": population.label,
                "measure": "Q",
                "path_count": arguments.paths,
                "antithetic_pair_count": arguments.paths // 2,
                "horizon_days": population.features.shape[1],
                "feature_count": population.features.shape[2],
                "instrument_count": population.returns.shape[2],
                "source": population.source,
            },
            "policy": {
                "architecture": (
                    "uncapped signed affine plus two-layer width-32 SiLU network"
                ),
                "train_config": asdict(train_config),
                "normalization": "fitted separately on each training fold",
                "heldout_use": "frozen policy evaluated once on the opposite fold",
            },
            "training_A": training_A,
            "heldout_B": heldout_B,
            "training_B": training_B,
            "heldout_A": heldout_A,
            "combined_out_of_fold": out_of_fold_summary,
            "headline_statistic": out_of_fold_summary["z_score"],
            "training_epochs": {
                key: len(history) for key, history in histories.items()
            },
            "artifacts": {"out_of_fold_pair_scores": str(score_path)},
        }
        report_path = arguments.output / f"{arguments.label}.json"
        _atomic_json(report, report_path)
        print(json.dumps(report, indent=2), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--population-kind",
        choices=("restricted-projection", "full-surface"),
        required=True,
    )
    parser.add_argument("--source-population", type=Path)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--projection-cache", type=Path)
    parser.add_argument("--surface-population", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--factor-rank", type=int, default=8)
    parser.add_argument("--dlv-substeps", type=int, default=8)
    parser.add_argument("--projection-batch-size", type=int, default=512)
    parser.add_argument("--training-batch-size", type=int, default=2_048)
    parser.add_argument("--maximum-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--maximum-gradient-norm", type=float, default=10.0)
    parser.add_argument("--lambda-second-moment", type=float, default=1.0)
    parser.add_argument("--policy-seed", type=int, default=20261009)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main() -> None:
    """Run one quadratic second-moment cross-fit experiment."""

    arguments = _parser().parse_args()
    if arguments.population_kind == "restricted-projection":
        if arguments.projection_cache is None:
            raise ValueError("--projection-cache is required")
    elif arguments.surface_population is None:
        raise ValueError("--surface-population is required")
    run(arguments)


if __name__ == "__main__":
    main()
