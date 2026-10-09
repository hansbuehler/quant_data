"""Run the conditional no-arbitrage test on the full eligible SDLV surface."""

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

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from filelock import FileLock  # noqa: E402
from numpy.lib.format import open_memmap  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.special import ndtri  # noqa: E402

from quantdata.biz_calendar import DAYS_PER_YEAR  # noqa: E402
from quantdata.sdlv.daily.front_spot.calibration import (  # noqa: E402
    FrontSpotCalibration,
    calibrate_front_spot_model,
)
from quantdata.sdlv.daily.front_spot.jax_model import (  # noqa: E402
    RiskNeutralState,
    black_mixture_call_prices,
    one_day_risk_neutral_substepped,
    substep_brownian_weights,
    terminal_front_step,
)
from quantdata.sdlv.daily.front_spot.oce import (  # noqa: E402
    FrontSpotOcePolicy,
    FrontSpotOcePolicyConfig,
    negative_exponential_statistics,
    paired_path_average_statistics,
    regularized_transition_gain,
)
from quantdata.sdlv.daily.front_spot.run_oce import (  # noqa: E402
    FrontSpotOceTrainConfig,
)
from quantdata.sdlv.daily.front_spot.simulation import (  # noqa: E402
    load_initial_model_state,
)
from quantdata.torch import BaseEnvironment  # noqa: E402
from quantdata.torch.ffw import FeedForwardConfig  # noqa: E402

HORIZON_DAYS = 25
POPULATION_SCHEMA = "front-spot-full-eligible-surface-population-v1"
POLICY_SCHEMA = "front-spot-full-eligible-surface-policy-v1"


@dataclass(frozen=True)
class SurfacePopulation:
    """Disk-backed path population for the full-surface conditional test."""

    features: NDArray[np.float32]
    returns: NDArray[np.float32]
    strikes: NDArray[np.float64]
    factor_rank: int
    nms_annualized_atm_volatility: float
    nms_lower_bound: float
    nms_upper_bound: float
    initial_date: str
    generation_seconds: float


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _progress(
    stage: str,
    completed: int,
    total: int,
    started: float,
    **values: object,
) -> None:
    elapsed = time.perf_counter() - started
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    eta = (total - completed) / rate if rate > 0.0 else None
    print(
        json.dumps(
            {
                "stage": stage,
                "completed": completed,
                "total": total,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                **values,
            }
        ),
        flush=True,
    )


def _front_atm_volatility(
    support: jax.Array,
    density_tower: jax.Array,
) -> float:
    front_atm_call = float(
        np.asarray(
            black_mixture_call_prices(
                support,
                density_tower[:1],
                jnp.ones((1, 1), dtype=jnp.float64),
                jnp.asarray(1.0, dtype=jnp.float64),
                jnp.ones(1, dtype=jnp.float64),
            )
        )[0, 0]
    )
    total_volatility = 2.0 * ndtri(
        np.clip(0.5 * (front_atm_call + 1.0), 0.5, 1.0 - 1.0e-14)
    )
    annualized = total_volatility * math.sqrt(DAYS_PER_YEAR)
    if not math.isfinite(annualized) or annualized <= 0.0:
        raise RuntimeError("could not recover a finite-positive front ATM volatility")
    return annualized


def _population_geometry(
    *,
    path_count: int,
    factor_rank: int,
    strike_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    instrument_count = 1 + HORIZON_DAYS * strike_count
    feature_count = factor_rank + 1 + HORIZON_DAYS * strike_count + 1 + HORIZON_DAYS
    return (
        (path_count, HORIZON_DAYS, feature_count),
        (path_count, HORIZON_DAYS, instrument_count),
    )


def _population_metadata_expected(
    *,
    path_count: int,
    factor_rank: int,
    dlv_substeps: int,
    time_power: float,
    seed: int,
    fit_label: str,
    date_index: int,
    loading_digest: str,
    nms_lower_bound: float,
    nms_upper_bound: float,
) -> dict[str, object]:
    return {
        "schema": POPULATION_SCHEMA,
        "path_count": path_count,
        "factor_rank": factor_rank,
        "dlv_substeps": dlv_substeps,
        "time_power": time_power,
        "seed": seed,
        "fit_label": fit_label,
        "date_index": date_index,
        "loading_digest": loading_digest,
        "nms_lower_bound": nms_lower_bound,
        "nms_upper_bound": nms_upper_bound,
    }


def _write_pre_step_features(
    target: NDArray[np.float32],
    *,
    factor_history: NDArray[np.float64],
    current_spot: NDArray[np.float64],
    current_options: NDArray[np.float64],
    day_index: int,
) -> None:
    live = np.arange(1, HORIZON_DAYS + 1) > day_index
    normalized_options = current_options / current_spot[:, None, None]
    log_options = np.where(
        live[None, :, None],
        np.log(np.maximum(normalized_options, 1.0e-12)),
        0.0,
    )
    remaining = np.maximum(
        np.arange(1, HORIZON_DAYS + 1, dtype=np.float64) - day_index,
        0.0,
    ) / HORIZON_DAYS
    values = np.concatenate(
        (
            factor_history,
            np.log(current_spot)[:, None],
            log_options.reshape(len(current_spot), -1),
            np.full((len(current_spot), 1), day_index / HORIZON_DAYS),
            np.broadcast_to(remaining, (len(current_spot), HORIZON_DAYS)),
        ),
        axis=1,
    )
    if values.shape != target.shape:
        raise RuntimeError(
            f"unexpected full-surface feature shape {values.shape}, expected {target.shape}"
        )
    target[:] = values


def _eligible_option_mask(
    current_spot: NDArray[np.float64],
    strikes: NDArray[np.float64],
    *,
    day_index: int,
    annualized_atm_volatility: float,
    lower_bound: float,
    upper_bound: float,
) -> NDArray[np.bool_]:
    remaining_days = np.arange(1, HORIZON_DAYS + 1, dtype=np.float64) - day_index
    live = remaining_days > 0.0
    denominator = annualized_atm_volatility * np.sqrt(
        np.maximum(remaining_days, 1.0) / DAYS_PER_YEAR
    )
    nms = np.log(strikes[None, None, :] / current_spot[:, None, None]) / (
        denominator[None, :, None]
    )
    return live[None, :, None] & (nms > lower_bound) & (nms < upper_bound)


def _write_returns(
    target: NDArray[np.float32],
    *,
    current_spot: NDArray[np.float64],
    next_spot: NDArray[np.float64],
    current_options: NDArray[np.float64],
    next_options: NDArray[np.float64],
    eligible_options: NDArray[np.bool_],
) -> None:
    target[:, 0] = (next_spot - current_spot) / current_spot
    option_returns = np.where(
        eligible_options,
        (next_options - current_options) / current_spot[:, None, None],
        0.0,
    )
    target[:, 1:] = option_returns.reshape(len(current_spot), -1)


def _generate_population(
    calibration: FrontSpotCalibration,
    *,
    cache_dir: Path,
    path_count: int,
    batch_size: int,
    fit_label: str,
    date_index: int,
    dlv_substeps: int,
    time_power: float,
    seed: int,
    nms_lower_bound: float,
    nms_upper_bound: float,
) -> SurfacePopulation:
    if path_count < 4 or path_count % 2:
        raise ValueError("the full-surface population requires an even path count")
    if batch_size < 2 or batch_size % 2:
        raise ValueError("the generation batch size must be positive and even")
    if not jax.config.x64_enabled:
        raise RuntimeError("full-surface generation requires JAX 64-bit mode")

    initial = load_initial_model_state(fit_label=fit_label, date_index=date_index)
    expiry_count, node_count = initial.density_tower.shape
    if expiry_count != HORIZON_DAYS:
        raise RuntimeError(
            f"expected {HORIZON_DAYS} daily expiries, found {expiry_count}"
        )
    strike_count = node_count - 2
    factor_rank = calibration.innovations.factor_rank
    feature_shape, return_shape = _population_geometry(
        path_count=path_count,
        factor_rank=factor_rank,
        strike_count=strike_count,
    )
    expected_metadata = _population_metadata_expected(
        path_count=path_count,
        factor_rank=factor_rank,
        dlv_substeps=dlv_substeps,
        time_power=time_power,
        seed=seed,
        fit_label=fit_label,
        date_index=date_index,
        loading_digest=hashlib.sha256(
            np.asarray(
                calibration.innovations.annualized_loadings,
                dtype=np.float64,
            ).tobytes()
        ).hexdigest(),
        nms_lower_bound=nms_lower_bound,
        nms_upper_bound=nms_upper_bound,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    features_path = cache_dir / "features.npy"
    returns_path = cache_dir / "returns.npy"

    metadata: dict[str, Any] | None = None
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError(
                f"incompatible partial population cache at {cache_dir}; use a new cache"
            )
        if not features_path.is_file() or not returns_path.is_file():
            raise RuntimeError(f"population metadata exists without arrays at {cache_dir}")

    if metadata is None:
        features = open_memmap(
            features_path,
            mode="w+",
            dtype=np.float32,
            shape=feature_shape,
        )
        returns = open_memmap(
            returns_path,
            mode="w+",
            dtype=np.float32,
            shape=return_shape,
        )
        generator = np.random.default_rng(seed)
        metadata = {
            **expected_metadata,
            "complete": False,
            "completed_paths": 0,
            "rng_state": generator.bit_generator.state,
            "initial_date": initial.date,
            "generation_seconds": 0.0,
        }
        _atomic_json(metadata, metadata_path)
    else:
        features = open_memmap(features_path, mode="r+")
        returns = open_memmap(returns_path, mode="r+")
        if features.shape != feature_shape or returns.shape != return_shape:
            raise RuntimeError("cached full-surface arrays have incompatible shapes")
        generator = np.random.default_rng()
        generator.bit_generator.state = metadata["rng_state"]

    support = jnp.asarray(initial.support, dtype=jnp.float64)
    strikes = np.asarray(initial.support[1:-1], dtype=np.float64)
    initial_tower = jnp.asarray(initial.density_tower, dtype=jnp.float64)
    initial_calls = np.asarray(
        black_mixture_call_prices(
            support,
            initial_tower,
            jnp.asarray(strikes, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            jnp.arange(1, HORIZON_DAYS + 1, dtype=jnp.float64),
        )
    )
    annualized_atm_volatility = _front_atm_volatility(support, initial_tower)
    loadings = jnp.asarray(
        calibration.innovations.annualized_loadings,
        dtype=jnp.float64,
    )
    daily_step = jax.jit(
        one_day_risk_neutral_substepped,
        static_argnames=("expiry_dt", "time_power"),
    )
    final_step = jax.jit(terminal_front_step, static_argnames=("dt",))
    pricer = jax.jit(black_mixture_call_prices)
    brownian_weights = np.asarray(
        substep_brownian_weights(
            dlv_substeps,
            time_power=time_power,
            dtype=jnp.float64,
        )
    )
    dt = 1.0 / DAYS_PER_YEAR
    completed = int(metadata["completed_paths"])
    if completed % 2:
        raise RuntimeError("partial population cache split an antithetic pair")
    started = time.perf_counter() - float(metadata.get("generation_seconds", 0.0))
    last_reported = completed

    while completed < path_count:
        valid_count = min(batch_size, path_count - completed)
        if valid_count % 2:
            raise RuntimeError("generation batch split an antithetic pair")
        pair_count = valid_count // 2
        independent_factor_normals = generator.standard_normal(
            (HORIZON_DAYS, dlv_substeps, pair_count, factor_rank)
        )
        factor_normals = np.stack(
            (independent_factor_normals, -independent_factor_normals),
            axis=3,
        ).reshape(HORIZON_DAYS, dlv_substeps, valid_count, factor_rank)
        independent_background_normals = generator.standard_normal(
            (HORIZON_DAYS, pair_count)
        )
        background_normals = np.stack(
            (independent_background_normals, -independent_background_normals),
            axis=2,
        ).reshape(HORIZON_DAYS, valid_count)

        state = RiskNeutralState(
            initial_tower[0],
            jnp.asarray(initial.dlvsq[1:], dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
        )
        current_tower = np.broadcast_to(
            initial.density_tower,
            (valid_count,) + initial.density_tower.shape,
        ).copy()
        current_spot = np.ones(valid_count, dtype=np.float64)
        current_options = np.broadcast_to(
            initial_calls,
            (valid_count,) + initial_calls.shape,
        ).copy()
        factor_history = np.zeros((valid_count, factor_rank), dtype=np.float64)
        path_slice = slice(completed, completed + valid_count)

        for day_index in range(HORIZON_DAYS - 1):
            _write_pre_step_features(
                features[path_slice, day_index],
                factor_history=factor_history,
                current_spot=current_spot,
                current_options=current_options,
                day_index=day_index,
            )
            eligibility_spot = np.exp(
                np.asarray(
                    features[path_slice, day_index, factor_rank],
                    dtype=np.float64,
                )
            )
            eligible = _eligible_option_mask(
                eligibility_spot,
                strikes,
                day_index=day_index,
                annualized_atm_volatility=annualized_atm_volatility,
                lower_bound=nms_lower_bound,
                upper_bound=nms_upper_bound,
            )
            future_count = state.future_dlvsq.shape[-2]
            result = daily_step(
                support,
                state,
                loadings[:future_count],
                jnp.asarray(factor_normals[day_index], dtype=jnp.float64),
                jnp.asarray(background_normals[day_index], dtype=jnp.float64),
                expiry_dt=dt,
                time_power=time_power,
            )
            next_spot = np.asarray(result.spot)
            next_options = current_options.copy()
            next_options[:, day_index] = np.maximum(
                next_spot[:, None] - strikes[None],
                0.0,
            )
            live_count = HORIZON_DAYS - day_index - 1
            next_options[:, day_index + 1 :] = np.asarray(
                pricer(
                    support,
                    result.density_tower,
                    jnp.asarray(strikes, dtype=jnp.float64),
                    result.state.background_level,
                    jnp.arange(1, live_count + 1, dtype=jnp.float64),
                )
            )
            _write_returns(
                returns[path_slice, day_index],
                current_spot=current_spot,
                next_spot=next_spot,
                current_options=current_options,
                next_options=next_options,
                eligible_options=eligible,
            )
            daily_factor_normal = np.einsum(
                "s,sbf->bf",
                brownian_weights,
                factor_normals[day_index],
            )
            factor_history += math.sqrt(dt) * daily_factor_normal
            state = result.state
            current_tower = np.asarray(result.density_tower)
            current_spot = next_spot
            current_options = next_options

        terminal_day = HORIZON_DAYS - 1
        _write_pre_step_features(
            features[path_slice, terminal_day],
            factor_history=factor_history,
            current_spot=current_spot,
            current_options=current_options,
            day_index=terminal_day,
        )
        terminal_eligibility_spot = np.exp(
            np.asarray(
                features[path_slice, terminal_day, factor_rank],
                dtype=np.float64,
            )
        )
        terminal_eligible = _eligible_option_mask(
            terminal_eligibility_spot,
            strikes,
            day_index=terminal_day,
            annualized_atm_volatility=annualized_atm_volatility,
            lower_bound=nms_lower_bound,
            upper_bound=nms_upper_bound,
        )
        terminal_factor_normal = np.einsum(
            "s,sb->b",
            brownian_weights,
            factor_normals[terminal_day, :, :, 0],
        )
        terminal = final_step(
            support,
            state.front_density,
            state.background_level,
            jnp.asarray(terminal_factor_normal, dtype=jnp.float64),
            jnp.asarray(background_normals[terminal_day], dtype=jnp.float64),
            dt=dt,
        )
        terminal_spot = np.asarray(terminal.spot)
        terminal_options = current_options.copy()
        terminal_options[:, terminal_day] = np.maximum(
            terminal_spot[:, None] - strikes[None],
            0.0,
        )
        _write_returns(
            returns[path_slice, terminal_day],
            current_spot=current_spot,
            next_spot=terminal_spot,
            current_options=current_options,
            next_options=terminal_options,
            eligible_options=terminal_eligible,
        )

        completed += valid_count
        features.flush()
        returns.flush()
        metadata.update(
            {
                "complete": completed == path_count,
                "completed_paths": completed,
                "rng_state": generator.bit_generator.state,
                "initial_date": initial.date,
                "generation_seconds": time.perf_counter() - started,
                "nms_annualized_atm_volatility": annualized_atm_volatility,
                "strike_count": strike_count,
                "feature_shape": list(feature_shape),
                "return_shape": list(return_shape),
            }
        )
        _atomic_json(metadata, metadata_path)
        if completed - last_reported >= 1_024 or completed == path_count:
            _progress("generate-population", completed, path_count, started)
            last_reported = completed

    return SurfacePopulation(
        features=np.load(features_path, mmap_mode="r+", allow_pickle=False),
        returns=np.load(returns_path, mmap_mode="r+", allow_pickle=False),
        strikes=strikes,
        factor_rank=factor_rank,
        nms_annualized_atm_volatility=annualized_atm_volatility,
        nms_lower_bound=nms_lower_bound,
        nms_upper_bound=nms_upper_bound,
        initial_date=initial.date,
        generation_seconds=float(metadata["generation_seconds"]),
    )


def _dynamic_active_mask(
    features: NDArray[np.float32],
    population: SurfacePopulation,
) -> NDArray[np.bool_]:
    current_spot = np.exp(
        np.asarray(features[..., population.factor_rank], dtype=np.float64)
    )
    path_count, step_count = current_spot.shape
    if step_count != HORIZON_DAYS:
        raise RuntimeError(f"expected {HORIZON_DAYS} feature steps, found {step_count}")
    strike_count = len(population.strikes)
    mask = np.zeros(
        (path_count, HORIZON_DAYS, 1 + HORIZON_DAYS * strike_count),
        dtype=np.bool_,
    )
    mask[..., 0] = True
    for day_index in range(HORIZON_DAYS):
        eligible = _eligible_option_mask(
            current_spot[:, day_index],
            population.strikes,
            day_index=day_index,
            annualized_atm_volatility=population.nms_annualized_atm_volatility,
            lower_bound=population.nms_lower_bound,
            upper_bound=population.nms_upper_bound,
        )
        mask[:, day_index, 1:] = eligible.reshape(path_count, -1)
    return mask


def _feature_normalization(
    population: SurfacePopulation,
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
            _progress(
                f"normalize-{label}",
                batch_stop - start,
                stop - start,
                started,
            )
    mean = total / observation_count
    variance = np.maximum(square / observation_count - mean**2, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def _policy_from_state(
    *,
    label: str,
    feature_mean: NDArray[np.float32],
    feature_scale: NDArray[np.float32],
    feature_count: int,
    instrument_count: int,
    device: str,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> FrontSpotOcePolicy:
    environment = BaseEnvironment(precision=32, device=device, debug=False)
    config = FrontSpotOcePolicyConfig(
        label=label,
        feature_count=feature_count,
        instrument_count=instrument_count,
        network=FeedForwardConfig(
            activation="SiLU",
            num_depth=2,
            num_width=32,
        ),
    )
    policy = FrontSpotOcePolicy(
        environment,
        config=config,
        feature_mean=torch.as_tensor(
            feature_mean,
            dtype=environment.fdtype,
            device=environment.device,
        ),
        feature_scale=torch.as_tensor(
            feature_scale,
            dtype=environment.fdtype,
            device=environment.device,
        ),
    )
    if state_dict is not None:
        policy.load_state_dict(state_dict)
    return policy


def _transition_gain(
    policy: FrontSpotOcePolicy,
    population: SurfacePopulation,
    start: int,
    stop: int,
    *,
    batch_size: int,
    epsilon: float,
    stage: str,
) -> NDArray[np.float64]:
    values = np.empty((stop - start, HORIZON_DAYS), dtype=np.float64)
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
                _dynamic_active_mask(feature_array, population),
                dtype=torch.bool,
                device=policy.device,
            )
            action = policy(feature, active_mask)
            values[relative] = (
                regularized_transition_gain(
                    action,
                    instrument_return,
                    epsilon=epsilon,
                )
                .double()
                .cpu()
                .numpy()
            )
            if batch_stop == stop or (batch_stop - start) % max(4_096, batch_size) == 0:
                _progress(stage, batch_stop - start, stop - start, started)
    return values


def _checkpoint_payload(
    *,
    policy: FrontSpotOcePolicy,
    best_state: dict[str, torch.Tensor],
    best_loss: float,
    history: list[dict[str, float | int]],
    next_epoch: int,
    complete: bool,
    affine_only: bool,
    train_config: FrontSpotOceTrainConfig,
    fold_start: int,
    fold_stop: int,
) -> dict[str, object]:
    return {
        "schema": POLICY_SCHEMA,
        "policy_code_version": FrontSpotOcePolicyConfig.version.unique_id48,
        "policy_config": asdict(policy.config),
        "train_config": asdict(train_config),
        "fold_start": fold_start,
        "fold_stop": fold_stop,
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


def _checkpoint_metadata_mismatches(
    checkpoint: dict[str, Any],
    *,
    start: int,
    stop: int,
    affine_only: bool,
) -> list[str]:
    expected = {
        "schema": POLICY_SCHEMA,
        "policy_code_version": FrontSpotOcePolicyConfig.version.unique_id48,
        "fold_start": start,
        "fold_stop": stop,
        "affine_only": affine_only,
    }
    return [
        key for key, value in expected.items() if checkpoint.get(key) != value
    ]


def _fit_policy(
    population: SurfacePopulation,
    start: int,
    stop: int,
    *,
    label: str,
    checkpoint_path: Path,
    train_config: FrontSpotOceTrainConfig,
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
        mismatches = _checkpoint_metadata_mismatches(
            checkpoint,
            start=start,
            stop=stop,
            affine_only=affine_only,
        )
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
        tuple(policy.affine.parameters())
        if affine_only
        else tuple(policy.parameters())
    )
    if checkpoint is None:
        history: list[dict[str, float | int]] = []
        next_epoch = 0
        best_loss = math.inf
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        }
        _atomic_torch(
            _checkpoint_payload(
                policy=policy,
                best_state=best_state,
                best_loss=best_loss,
                history=history,
                next_epoch=next_epoch,
                complete=False,
                affine_only=affine_only,
                train_config=train_config,
                fold_start=start,
                fold_stop=stop,
            ),
            checkpoint_path,
        )
    else:
        history = checkpoint["history"]
        next_epoch = int(checkpoint["next_epoch"])
        best_loss = float(checkpoint["best_loss"])
        best_state = checkpoint["best_state"]
        if next_epoch > train_config.maximum_epochs:
            raise ValueError(
                f"checkpoint {checkpoint_path} has completed {next_epoch} epochs, "
                f"more than the requested {train_config.maximum_epochs}"
            )
        if checkpoint["complete"] and next_epoch == train_config.maximum_epochs:
            policy.load_state_dict(best_state)
            return policy, history
        if checkpoint["complete"]:
            policy.load_state_dict(best_state)
            print(
                json.dumps(
                    {
                        "stage": "resume-extended-training",
                        "label": label,
                        "completed_epochs": next_epoch,
                        "requested_epochs": train_config.maximum_epochs,
                        "previous_train_config": checkpoint.get("train_config"),
                        "current_train_config": asdict(train_config),
                    }
                ),
                flush=True,
            )

    training_started = time.perf_counter()
    for epoch in range(next_epoch, train_config.maximum_epochs):
        transition_gain = _transition_gain(
            policy,
            population,
            start,
            stop,
            batch_size=train_config.batch_size,
            epsilon=train_config.epsilon,
            stage=f"{label}-epoch-{epoch}-loss",
        )
        log_weight = -transition_gain
        maximum = float(np.max(log_weight))
        scaled_weight = np.exp(log_weight - maximum)
        scaled_weight /= np.sum(scaled_weight, dtype=np.float64)
        log_loss = maximum + math.log(
            float(np.mean(np.exp(log_weight - maximum), dtype=np.float64))
        )
        if log_loss < best_loss - train_config.minimum_improvement:
            best_loss = log_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            }

        policy.zero_grad(set_to_none=True)
        gradient_started = time.perf_counter()
        for batch_start in range(start, stop, train_config.batch_size):
            batch_stop = min(batch_start + train_config.batch_size, stop)
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
                _dynamic_active_mask(feature_array, population),
                dtype=torch.bool,
                device=policy.device,
            )
            action = policy(feature, active_mask)
            batch_gain = regularized_transition_gain(
                action,
                instrument_return,
                epsilon=train_config.epsilon,
            )
            probability = torch.as_tensor(
                scaled_weight[relative],
                dtype=batch_gain.dtype,
                device=batch_gain.device,
            )
            torch.sum(-probability * batch_gain).backward()
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
        candidate_loss = log_loss
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
            candidate_gain = _transition_gain(
                policy,
                population,
                start,
                stop,
                batch_size=train_config.batch_size,
                epsilon=train_config.epsilon,
                stage=f"{label}-epoch-{epoch}-line-search-{line_search_index}",
            )
            candidate_loss = negative_exponential_statistics(candidate_gain)[
                "log_expectation"
            ]
            if candidate_loss < log_loss - 1.0e-12:
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
            "log_expectation": log_loss,
            "candidate_log_expectation": candidate_loss,
            "gradient_l2_norm_before_clipping": gradient_norm,
            "gradient_l2_norm_after_clipping": math.sqrt(direction_norm_square),
            "accepted_step_size": accepted_step,
            "line_search_evaluations": line_search_evaluations,
            "tilted_effective_sample_size": float(
                1.0 / np.sum(scaled_weight**2, dtype=np.float64)
            ),
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
            ),
            checkpoint_path,
        )
        print(json.dumps({"stage": "train-epoch", "label": label, **record}), flush=True)
        if accepted_step == 0.0:
            break

    final_gain = _transition_gain(
        policy,
        population,
        start,
        stop,
        batch_size=train_config.batch_size,
        epsilon=train_config.epsilon,
        stage=f"{label}-final-loss",
    )
    final_loss = negative_exponential_statistics(final_gain)["log_expectation"]
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
        ),
        checkpoint_path,
    )
    return policy, history


def _evaluate_policy(
    policy: FrontSpotOcePolicy,
    population: SurfacePopulation,
    start: int,
    stop: int,
    *,
    batch_size: int,
    epsilon: float,
    label: str,
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    transition_gain = np.empty((stop - start, HORIZON_DAYS), dtype=np.float64)
    path_average = np.empty(stop - start, dtype=np.float64)
    path_energy_sum = 0.0
    action_square_sum = 0.0
    maximum_absolute_action = 0.0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_start in range(start, stop, batch_size):
            batch_stop = min(batch_start + batch_size, stop)
            relative = slice(batch_start - start, batch_stop - start)
            feature_array = population.features[batch_start:batch_stop]
            return_array = population.returns[batch_start:batch_stop]
            feature = torch.as_tensor(
                feature_array,
                dtype=policy.fdtype,
                device=policy.device,
            )
            instrument_return = torch.as_tensor(
                return_array,
                dtype=policy.fdtype,
                device=policy.device,
            )
            active_mask = torch.as_tensor(
                _dynamic_active_mask(feature_array, population),
                dtype=torch.bool,
                device=policy.device,
            )
            action = policy(feature, active_mask)
            transition_gain[relative] = (
                regularized_transition_gain(
                    action,
                    instrument_return,
                    epsilon=epsilon,
                )
                .double()
                .cpu()
                .numpy()
            )
            action_array = action.double().cpu().numpy()
            path_average[relative] = np.mean(
                np.sum(action_array * return_array.astype(np.float64), axis=-1),
                axis=1,
                dtype=np.float64,
            )
            path_energy = np.sum(action_array**2, axis=(1, 2), dtype=np.float64)
            path_energy_sum += float(np.sum(path_energy, dtype=np.float64))
            action_square_sum += float(np.sum(action_array**2, dtype=np.float64))
            maximum_absolute_action = max(
                maximum_absolute_action,
                float(np.max(np.abs(action_array))),
            )
            if batch_stop == stop or (batch_stop - start) % max(4_096, batch_size) == 0:
                _progress(label, batch_stop - start, stop - start, started)
    summary: dict[str, Any] = negative_exponential_statistics(transition_gain)
    paired_drift, _ = paired_path_average_statistics(path_average)
    summary.update(
        {
            "path_count": stop - start,
            "transition_count": (stop - start) * HORIZON_DAYS,
            "paired_conditional_drift": paired_drift,
            "mean_path_average_raw_action_return": float(np.mean(path_average)),
            "expected_path_action_energy": path_energy_sum / (stop - start),
            "expected_pointwise_action_energy": action_square_sum
            / ((stop - start) * HORIZON_DAYS),
            "maximum_absolute_action": maximum_absolute_action,
            "rms_action": math.sqrt(
                action_square_sum
                / ((stop - start) * HORIZON_DAYS * population.returns.shape[-1])
            ),
        }
    )
    return summary, path_average


def _instrument_label(index: int, strike_count: int) -> str:
    if index == 0:
        return "spot"
    option_index = index - 1
    expiry = option_index // strike_count + 1
    strike_index = option_index % strike_count
    return f"{expiry}D interior node {strike_index} call"


def _unconditional_audit(
    population: SurfacePopulation,
    *,
    batch_size: int,
) -> dict[str, Any]:
    path_count, step_count, instrument_count = population.returns.shape
    if path_count % 2:
        raise RuntimeError("unconditional audit requires complete antithetic pairs")
    total = np.zeros((step_count, instrument_count), dtype=np.float64)
    square = np.zeros_like(total)
    eligible_count = np.zeros_like(total, dtype=np.int64)
    pair_count = path_count // 2
    started = time.perf_counter()
    completed_pairs = 0
    pair_batch_size = max(1, batch_size // 2)
    for pair_start in range(0, pair_count, pair_batch_size):
        pair_stop = min(pair_start + pair_batch_size, pair_count)
        path_start = 2 * pair_start
        path_stop = 2 * pair_stop
        pair_return = np.mean(
            np.asarray(population.returns[path_start:path_stop], dtype=np.float64).reshape(
                pair_stop - pair_start,
                2,
                step_count,
                instrument_count,
            ),
            axis=1,
            dtype=np.float64,
        )
        total += np.sum(pair_return, axis=0, dtype=np.float64)
        square += np.sum(pair_return**2, axis=0, dtype=np.float64)
        active = _dynamic_active_mask(
            population.features[path_start:path_stop],
            population,
        )
        eligible_count += np.sum(active, axis=0, dtype=np.int64)
        completed_pairs = pair_stop
        if pair_stop == pair_count or pair_stop % max(2_048, pair_batch_size) == 0:
            _progress("unconditional-audit", pair_stop, pair_count, started)
    mean = total / pair_count
    variance = np.maximum(
        (square - total**2 / pair_count) / max(pair_count - 1, 1),
        0.0,
    )
    standard_error = np.sqrt(variance / pair_count)
    identifiable = (eligible_count >= 20) & (standard_error > 1.0e-15)
    z_score = np.zeros_like(mean)
    np.divide(mean, standard_error, out=z_score, where=identifiable)
    absolute_z = np.abs(z_score[identifiable])
    flat_identifiable = np.flatnonzero(identifiable)
    top_count = min(10, len(flat_identifiable))
    top_order = np.argsort(absolute_z)[-top_count:][::-1]
    top_coordinates = []
    strike_count = len(population.strikes)
    for active_index in top_order:
        flat_index = int(flat_identifiable[active_index])
        day_index, instrument_index = np.unravel_index(flat_index, mean.shape)
        top_coordinates.append(
            {
                "day_index": int(day_index),
                "instrument_index": int(instrument_index),
                "instrument_label": _instrument_label(instrument_index, strike_count),
                "mean": float(mean[day_index, instrument_index]),
                "pair_based_standard_error": float(
                    standard_error[day_index, instrument_index]
                ),
                "z_score": float(z_score[day_index, instrument_index]),
                "eligible_path_count": int(
                    eligible_count[day_index, instrument_index]
                ),
            }
        )
    return {
        "null_hypothesis": (
            "E_Q[I_t^i (V_{t+1}^i-V_t^i)/S_t]=0 for every daily coordinate, "
            "where I_t^i is the predictable NMS eligibility indicator"
        ),
        "path_count": path_count,
        "antithetic_pair_count": completed_pairs,
        "tested_coordinate_count": int(np.sum(identifiable)),
        "total_coordinate_count": int(np.sum(eligible_count > 0)),
        "max_absolute_z": float(np.max(absolute_z)) if absolute_z.size else 0.0,
        "p99_absolute_z": (
            float(np.quantile(absolute_z, 0.99)) if absolute_z.size else 0.0
        ),
        "rms_z": (
            float(np.sqrt(np.mean(absolute_z**2))) if absolute_z.size else 0.0
        ),
        "count_above_2_576": int(np.sum(absolute_z > 2.576)),
        "count_above_3": int(np.sum(absolute_z > 3.0)),
        "count_above_5": int(np.sum(absolute_z > 5.0)),
        "top_absolute_z_coordinates": top_coordinates,
    }


def run(arguments: argparse.Namespace) -> None:
    """Run the resumable two-fold conditional no-arbitrage experiment."""

    if arguments.paths < 4 or arguments.paths % 4:
        raise ValueError("path count must be divisible by four for two antithetic folds")
    if arguments.factor_rank not in (8, 16):
        raise ValueError("factor rank must be 8 or 16")
    torch.set_num_threads(arguments.torch_threads)
    arguments.output.mkdir(parents=True, exist_ok=True)
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(arguments.cache_dir / "full-surface-oce.lock")
    with lock:
        calibration = calibrate_front_spot_model(
            fit_label=arguments.fit_label,
            factor_rank=arguments.factor_rank,
            batch_size=arguments.calibration_batch_size,
            smoothing_sigma=(arguments.expiry_smoothing, arguments.strike_smoothing),
        )
        population = _generate_population(
            calibration,
            cache_dir=arguments.cache_dir / "population",
            path_count=arguments.paths,
            batch_size=arguments.generation_batch_size,
            fit_label=arguments.fit_label,
            date_index=arguments.date_index,
            dlv_substeps=arguments.dlv_substeps,
            time_power=arguments.time_power,
            seed=arguments.seed,
            nms_lower_bound=arguments.nms_lower_bound,
            nms_upper_bound=arguments.nms_upper_bound,
        )
        affine_train_config = FrontSpotOceTrainConfig(
            batch_size=arguments.training_batch_size,
            maximum_epochs=arguments.affine_maximum_epochs,
            learning_rate=arguments.learning_rate,
            minimum_improvement=arguments.minimum_improvement,
            maximum_gradient_norm=arguments.maximum_gradient_norm,
            epsilon=arguments.epsilon,
            seed=arguments.policy_seed,
        )
        full_train_config = FrontSpotOceTrainConfig(
            batch_size=arguments.training_batch_size,
            maximum_epochs=arguments.full_maximum_epochs,
            learning_rate=arguments.learning_rate,
            minimum_improvement=arguments.minimum_improvement,
            maximum_gradient_norm=arguments.maximum_gradient_norm,
            epsilon=arguments.epsilon,
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
                label=f"full-surface-affine-{fold_label}",
                checkpoint_path=arguments.cache_dir
                / f"affine_{fold_label}.checkpoint.pt",
                train_config=affine_train_config,
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
                label=f"full-surface-{fold_label}",
                checkpoint_path=arguments.cache_dir / f"policy_{fold_label}.checkpoint.pt",
                train_config=full_train_config,
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
            batch_size=arguments.training_batch_size,
            epsilon=arguments.epsilon,
            label="evaluate-A-on-A",
        )
        heldout_B, scores_B = _evaluate_policy(
            policies["A"],
            population,
            split,
            arguments.paths,
            batch_size=arguments.training_batch_size,
            epsilon=arguments.epsilon,
            label="evaluate-A-on-B",
        )
        training_B, _ = _evaluate_policy(
            policies["B"],
            population,
            split,
            arguments.paths,
            batch_size=arguments.training_batch_size,
            epsilon=arguments.epsilon,
            label="evaluate-B-on-B",
        )
        heldout_A, scores_A = _evaluate_policy(
            policies["B"],
            population,
            0,
            split,
            batch_size=arguments.training_batch_size,
            epsilon=arguments.epsilon,
            label="evaluate-B-on-A",
        )
        out_of_fold_scores = np.concatenate((scores_A, scores_B))
        out_of_fold_summary, out_of_fold_pair_scores = paired_path_average_statistics(
            out_of_fold_scores
        )
        artifact_stem = (
            f"rank{arguments.factor_rank}_{arguments.dlv_substeps}_step_"
            "full_surface_oce"
        )
        np.save(
            arguments.output / f"{artifact_stem}_oof_pair_scores.npy",
            out_of_fold_pair_scores,
            allow_pickle=False,
        )
        unconditional = _unconditional_audit(
            population,
            batch_size=arguments.training_batch_size,
        )
        report = {
            "schema": "front-spot-full-eligible-surface-oce-v1",
            "population": {
                "initial_date": population.initial_date,
                "measure": "Q",
                "path_count": arguments.paths,
                "antithetic_pair_count": arguments.paths // 2,
                "factor_rank": arguments.factor_rank,
                "dlv_substeps": arguments.dlv_substeps,
                "horizon_days": HORIZON_DAYS,
                "strike_count_per_expiry": len(population.strikes),
                "instrument_count": population.returns.shape[-1],
                "feature_count": population.features.shape[-1],
                "nms_annualized_atm_volatility": (
                    population.nms_annualized_atm_volatility
                ),
                "nms_bounds": [
                    population.nms_lower_bound,
                    population.nms_upper_bound,
                ],
                "return_definition": (
                    "predictably NMS-eligible daily spot and option price increments "
                    "divided by pre-step spot"
                ),
                "generation_seconds": population.generation_seconds,
            },
            "objective": {
                "local_loss": (
                    "log E[exp(-a_theta(X_t)'R_t + epsilon*"
                    "a_theta(X_t)'a_theta(X_t)/2)]"
                ),
                "epsilon": arguments.epsilon,
                "action": "uncapped signed affine plus two-layer width-32 SiLU network",
                "eligibility": (
                    "pathwise pre-step mask; ineligible and expired actions are exactly zero"
                ),
            },
            "split": {
                "sample_A_paths": split,
                "sample_B_paths": arguments.paths - split,
                "integrity": (
                    "complete paths and both members of each adjacent antithetic pair "
                    "remain in one fold"
                ),
                "normalization": "fitted separately on each training fold only",
                "affine_hyperparameters": asdict(affine_train_config),
                "full_hyperparameters": asdict(full_train_config),
            },
            "unconditional": unconditional,
            "conditional": {
                "training_A": training_A,
                "heldout_B": heldout_B,
                "training_B": training_B,
                "heldout_A": heldout_A,
                "combined_out_of_fold": out_of_fold_summary,
                "headline_statistic": out_of_fold_summary["z_score"],
                "interpretation": (
                    "Under zero conditional drift, every frozen predictable action has "
                    "zero expected held-out score; the statistic is conditional on the "
                    "stated affine-plus-deep strategy class"
                ),
            },
            "training_epochs": {
                label: len(history) for label, history in histories.items()
            },
            "checkpointing": (
                "population resumes after the last completed generation batch and each "
                "policy resumes after the last completed training epoch"
            ),
        }
        report_path = arguments.output / f"{artifact_stem}.json"
        _atomic_json(report, report_path)
        print(json.dumps(report, indent=2), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "quantdata/sdlv/daily/front_spot/results/"
            "oce_rank8_eight_step_full_surface"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "/home/hans/quantdata_data/tmp/quantdata/sdlv/"
            "front_spot/rank8_eight_step_full_surface_oce_100k"
        ),
    )
    parser.add_argument("--fit-label", default="full")
    parser.add_argument("--date-index", type=int, default=-1)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--factor-rank", type=int, default=8)
    parser.add_argument("--dlv-substeps", type=int, default=8)
    parser.add_argument("--time-power", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--policy-seed", type=int, default=20261008)
    parser.add_argument("--nms-lower-bound", type=float, default=-2.0)
    parser.add_argument("--nms-upper-bound", type=float, default=1.0)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--calibration-batch-size", type=int, default=64)
    parser.add_argument("--expiry-smoothing", type=float, default=1.0)
    parser.add_argument("--strike-smoothing", type=float, default=2.0)
    parser.add_argument("--training-batch-size", type=int, default=64)
    parser.add_argument("--affine-maximum-epochs", type=int, default=100)
    parser.add_argument("--full-maximum-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--minimum-improvement", type=float, default=1.0e-7)
    parser.add_argument("--maximum-gradient-norm", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main() -> None:
    """Run the full-surface conditional no-arbitrage experiment."""

    run(_parser().parse_args())


if __name__ == "__main__":
    main()
