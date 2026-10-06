"""Fit uncapped affine-plus-deep OCE policies on front-spot Q paths."""

from __future__ import annotations

import argparse
import copy
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
from cdxcore.err import verify_inp  # noqa: E402
from cdxcore.version import version  # noqa: E402
from filelock import FileLock  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.special import ndtri  # noqa: E402

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS  # noqa: E402
from quantdata.sdlv.daily.front_spot.calibration import (  # noqa: E402
    FrontSpotCalibration,
    calibrate_front_spot_model,
)
from quantdata.sdlv.daily.front_spot.jax_model import (  # noqa: E402
    RiskNeutralState,
    black_mixture_call_prices,
    one_day_risk_neutral_substepped,
    substep_brownian_weights,
)
from quantdata.sdlv.daily.front_spot.oce import (  # noqa: E402
    FrontSpotOcePolicy,
    FrontSpotOcePolicyConfig,
    negative_exponential_statistics,
    paired_path_average_statistics,
    regularized_transition_gain,
)
from quantdata.sdlv.daily.front_spot.simulation import (  # noqa: E402
    load_initial_model_state,
)
from quantdata.torch import BaseEnvironment  # noqa: E402
from quantdata.torch.ffw import FeedForwardConfig  # noqa: E402

EXPIRY_DAYS = (5, 10, 15, 20)
NORMALIZED_MONEYNESS = (
    -1.5,
    -1.3,
    -1.1,
    -0.9,
    -0.7,
    -0.5,
    -0.3,
    -0.1,
    0.1,
    0.3,
    0.5,
)


def _substep_name(dlv_substeps: int) -> str:
    names = {1: "one_step", 8: "eight_step"}
    return names.get(dlv_substeps, f"{dlv_substeps}_step")


def _artifact_stem(
    factor_rank: int,
    dlv_substeps: int,
    instrument_universe: str = "restricted",
) -> str:
    suffix = "" if instrument_universe == "restricted" else f"_{instrument_universe}"
    return f"rank{factor_rank}_{_substep_name(dlv_substeps)}{suffix}_local_oce"


def _population_cache_schema(
    factor_rank: int,
    dlv_substeps: int,
    instrument_universe: str = "restricted",
) -> str:
    if (
        factor_rank == 8
        and dlv_substeps == 8
        and instrument_universe == "restricted"
    ):
        return "front-spot-rank8-eight-step-antithetic-spot-scaled-population-v2"
    step_name = _substep_name(dlv_substeps).replace("_", "-")
    return (
        f"front-spot-rank{factor_rank}-{step_name}-{instrument_universe}-"
        "antithetic-spot-scaled-population-v3"
    )


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class FrontSpotOceTrainConfig:
    """Operational controls for exact empirical OCE optimization."""

    batch_size: int = 2_048
    maximum_epochs: int = 30
    learning_rate: float = 1.0e-3
    minimum_improvement: float = 1.0e-7
    maximum_gradient_norm: float = 10.0
    backtracking_factor: float = 0.2
    maximum_line_search_steps: int = 20
    epsilon: float = 1.0
    seed: int = 20261006

    def __post_init__(self) -> None:
        verify_inp(self.batch_size > 0, lambda: "batch size must be positive")
        verify_inp(self.maximum_epochs > 0, lambda: "maximum epochs must be positive")
        verify_inp(
            math.isfinite(self.learning_rate) and self.learning_rate > 0.0,
            lambda: "learning rate must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.minimum_improvement) and self.minimum_improvement >= 0.0,
            lambda: "minimum improvement must be finite and non-negative",
        )
        verify_inp(
            math.isfinite(self.maximum_gradient_norm) and self.maximum_gradient_norm > 0.0,
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
            math.isfinite(self.epsilon) and self.epsilon > 0.0,
            lambda: "epsilon must be finite-positive",
        )


@dataclass(frozen=True)
class FrontSpotOcePopulation:
    """Observable features and spot-normalized price increments on Q paths."""

    features: NDArray[np.float32]
    returns: NDArray[np.float32]
    active_mask: NDArray[np.bool_]
    strikes: NDArray[np.float64]
    instrument_labels: tuple[str, ...]
    instrument_universe: str
    initial_date: str
    generation_seconds: float


def _save_population_cache(
    population: FrontSpotOcePopulation,
    directory: Path,
    *,
    factor_rank: int,
    dlv_substeps: int,
    instrument_universe: str,
    seed: int,
) -> None:
    """Save the reproducible large population in ephemeral local storage."""

    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "features": population.features,
        "returns": population.returns,
        "active_mask": population.active_mask,
        "strikes": population.strikes,
        "instrument_labels": np.asarray(population.instrument_labels),
    }
    for name, values in arrays.items():
        final_path = directory / f"{name}.npy"
        temporary = directory / f".{name}.tmp.npy"
        np.save(temporary, values, allow_pickle=False)
        os.replace(temporary, final_path)
    _atomic_json(
        {
            "schema": _population_cache_schema(
                factor_rank,
                dlv_substeps,
                instrument_universe,
            ),
            "path_count": len(population.features),
            "factor_rank": factor_rank,
            "dlv_substeps": dlv_substeps,
            "instrument_universe": instrument_universe,
            "seed": seed,
            "initial_date": population.initial_date,
            "generation_seconds": population.generation_seconds,
        },
        directory / "metadata.json",
    )


def _load_population_cache(
    directory: Path,
    *,
    path_count: int,
    factor_rank: int,
    dlv_substeps: int,
    instrument_universe: str,
    seed: int,
) -> FrontSpotOcePopulation | None:
    """Load a complete compatible ephemeral population cache if present."""

    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema": _population_cache_schema(
            factor_rank,
            dlv_substeps,
            instrument_universe,
        ),
        "path_count": path_count,
        "factor_rank": factor_rank,
        "dlv_substeps": dlv_substeps,
        "instrument_universe": instrument_universe,
        "seed": seed,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    required = (
        "features.npy",
        "returns.npy",
        "active_mask.npy",
        "strikes.npy",
        "instrument_labels.npy",
    )
    if any(not (directory / name).is_file() for name in required):
        return None
    features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    returns = np.load(directory / "returns.npy", mmap_mode="r", allow_pickle=False)
    active_mask = np.load(directory / "active_mask.npy", allow_pickle=False)
    strikes = np.load(directory / "strikes.npy", allow_pickle=False)
    labels = tuple(
        str(value)
        for value in np.load(
            directory / "instrument_labels.npy",
            allow_pickle=False,
        )
    )
    if features.shape[:2] != (path_count, max(EXPIRY_DAYS)):
        raise RuntimeError("cached OCE features have the wrong path-time geometry")
    if returns.shape[:2] != features.shape[:2]:
        raise RuntimeError("cached OCE returns have the wrong path-time geometry")
    return FrontSpotOcePopulation(
        features=features,
        returns=returns,
        active_mask=active_mask,
        strikes=strikes,
        instrument_labels=labels,
        instrument_universe=instrument_universe,
        initial_date=str(metadata["initial_date"]),
        generation_seconds=float(metadata["generation_seconds"]),
    )


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


def _initial_nms_strikes(
    support: jax.Array,
    density_tower: jax.Array,
) -> NDArray[np.float64]:
    expiry_indices = np.asarray(EXPIRY_DAYS, dtype=np.int64) - 1
    selected_tower = density_tower[expiry_indices]
    expiry_days = jnp.asarray(EXPIRY_DAYS, dtype=jnp.float64)
    atm_calls = np.asarray(
        black_mixture_call_prices(
            support,
            selected_tower,
            jnp.ones((len(EXPIRY_DAYS), 1), dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            expiry_days,
        )
    )[:, 0]
    probabilities = np.clip(0.5 * (atm_calls + 1.0), 0.5, 1.0 - 1.0e-14)
    total_volatility = 2.0 * ndtri(probabilities)
    epsilon_volatility = total_volatility[0] / math.sqrt(EXPIRY_DAYS[0] / DAYS_PER_YEAR)
    nms_scale = np.sqrt(
        (
            epsilon_volatility**2 * float(NMS.EPS)
            + total_volatility**2
        )
        / (1.0 + float(NMS.EPS))
    )
    strikes = np.exp(
        nms_scale[:, None]
        * np.asarray(NORMALIZED_MONEYNESS, dtype=np.float64)[None]
    )
    if np.any(~np.isfinite(strikes) | (strikes <= 0.0)):
        raise FloatingPointError("NMS conversion produced invalid strikes")
    return strikes


def _option_values_from_calls(
    calls: NDArray[np.float64],
    spots: NDArray[np.float64] | float,
    strikes: NDArray[np.float64],
) -> NDArray[np.float64]:
    put_mask = np.asarray(NORMALIZED_MONEYNESS) > 0.0
    spot_array = np.asarray(spots, dtype=np.float64)
    if spot_array.ndim:
        spot_array = spot_array[:, None, None]
    puts = calls - spot_array + strikes
    values = np.where(put_mask[None, :], puts, calls)
    if np.any(~np.isfinite(values)) or np.min(values) < -2.0e-12:
        raise FloatingPointError("option values are negative or non-finite")
    return np.maximum(values, np.finfo(np.float64).tiny)


def _instrument_labels(
    strike_count: int,
    *,
    instrument_universe: str,
) -> tuple[str, ...]:
    labels = ["spot"]
    for expiry in EXPIRY_DAYS:
        if instrument_universe == "restricted":
            for score in NORMALIZED_MONEYNESS:
                option_type = "put" if score > 0.0 else "call"
                labels.append(f"{expiry}D NMS {score:g} {option_type}")
        elif instrument_universe == "all_nodes":
            labels.extend(
                f"{expiry}D interior node {node_index} call"
                for node_index in range(strike_count)
            )
        else:
            raise ValueError(f"unknown instrument universe {instrument_universe!r}")
    return tuple(labels)


def _active_mask(strike_count: int) -> NDArray[np.bool_]:
    horizon = max(EXPIRY_DAYS)
    mask = np.zeros(
        (horizon, 1 + len(EXPIRY_DAYS) * strike_count),
        dtype=np.bool_,
    )
    mask[:, 0] = True
    for expiry_index, expiry in enumerate(EXPIRY_DAYS):
        option_slice = slice(
            1 + expiry_index * strike_count,
            1 + (expiry_index + 1) * strike_count,
        )
        mask[:expiry, option_slice] = True
    return mask


def _spot_normalized_price_increments(
    current_spot: NDArray[np.float64],
    next_spot: NDArray[np.float64],
    current_options: NDArray[np.float64],
    next_options: NDArray[np.float64],
    valid_option_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return one-day price increments divided by the pre-step spot."""

    path_count = len(current_spot)
    instrument_count = 1 + current_options.shape[1] * current_options.shape[2]
    returns = np.zeros((path_count, instrument_count), dtype=np.float64)
    returns[:, 0] = (next_spot - current_spot) / current_spot
    option_increments = np.zeros_like(current_options)
    np.subtract(
        next_options,
        current_options,
        out=option_increments,
        where=valid_option_mask[None, :, None],
    )
    option_increments /= current_spot[:, None, None]
    returns[:, 1:] = option_increments.reshape(path_count, -1)
    return returns


def _unconditional_coordinate_summary(
    pair_returns: NDArray[np.float64],
    coordinate_mask: NDArray[np.bool_],
    instrument_labels: tuple[str, ...],
) -> dict[str, Any]:
    """Summarize unconditional zero-mean tests using antithetic pairs."""

    pair_count = len(pair_returns)
    mean = np.mean(pair_returns, axis=0, dtype=np.float64)
    standard_error = np.std(pair_returns, axis=0, ddof=1) / math.sqrt(pair_count)
    active = coordinate_mask & (standard_error > 1.0e-15)
    z_score = np.zeros_like(mean)
    np.divide(mean, standard_error, out=z_score, where=active)
    absolute_z = np.abs(z_score[active])
    flat_active = np.flatnonzero(active)
    top_count = min(10, len(flat_active))
    top_order = np.argsort(absolute_z)[-top_count:][::-1]
    top_coordinates = []
    for active_index in top_order:
        flat_index = int(flat_active[active_index])
        day_index, instrument_index = np.unravel_index(flat_index, mean.shape)
        top_coordinates.append(
            {
                "day_index": int(day_index),
                "instrument_index": int(instrument_index),
                "instrument_label": instrument_labels[instrument_index],
                "mean": float(mean[day_index, instrument_index]),
                "pair_based_standard_error": float(
                    standard_error[day_index, instrument_index]
                ),
                "z_score": float(z_score[day_index, instrument_index]),
            }
        )
    return {
        "tested_coordinate_count": int(np.sum(active)),
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


def unconditional_spot_scaled_martingale_audit(
    returns: NDArray[np.float32],
    active_mask: NDArray[np.bool_],
    instrument_labels: tuple[str, ...],
) -> dict[str, Any]:
    """Test unconditional means of spot-scaled increments coordinate by coordinate."""

    if len(returns) < 4 or len(returns) % 2:
        raise ValueError("unconditional audit requires an even number of paths")
    if returns.shape[1:] != active_mask.shape:
        raise ValueError("return and active-mask geometries do not match")
    pair_returns = 0.5 * (
        np.asarray(returns[0::2], dtype=np.float64)
        + np.asarray(returns[1::2], dtype=np.float64)
    )
    spot_mask = np.zeros_like(active_mask)
    spot_mask[:, 0] = active_mask[:, 0]
    option_mask = active_mask.copy()
    option_mask[:, 0] = False
    return {
        "null_hypothesis": "E_Q[(V_{t+1}^i-V_t^i)/S_t]=0 for every day and instrument",
        "path_count": len(returns),
        "antithetic_pair_count": len(pair_returns),
        "sampling_unit": (
            "complete antithetic path pair; each day-instrument coordinate is "
            "tested separately"
        ),
        "all_instruments": _unconditional_coordinate_summary(
            pair_returns,
            active_mask,
            instrument_labels,
        ),
        "spot": _unconditional_coordinate_summary(
            pair_returns,
            spot_mask,
            instrument_labels,
        ),
        "options": _unconditional_coordinate_summary(
            pair_returns,
            option_mask,
            instrument_labels,
        ),
    }


def generate_front_spot_oce_population(
    calibration: FrontSpotCalibration,
    *,
    path_count: int = 100_000,
    batch_size: int = 128,
    fit_label: str = "full",
    date_index: int = -1,
    dlv_substeps: int = 8,
    time_power: float = 1.0,
    seed: int = 20261004,
    instrument_universe: str = "restricted",
) -> FrontSpotOcePopulation:
    """Generate the rank-specific Q population used by the OCE experiment."""

    if path_count < 4 or path_count % 2 or batch_size < 2 or batch_size % 2:
        raise ValueError(
            "path and batch counts must be even, with at least four paths and two paths per batch"
        )
    if dlv_substeps < 1:
        raise ValueError("at least one DLV substep is required")
    started = time.perf_counter()
    initial = load_initial_model_state(fit_label=fit_label, date_index=date_index)
    factor_rank = calibration.innovations.factor_rank
    horizon = max(EXPIRY_DAYS)
    support = jnp.asarray(initial.support, dtype=jnp.float64)
    initial_tower = jnp.asarray(initial.density_tower, dtype=jnp.float64)
    loadings = jnp.asarray(
        calibration.innovations.annualized_loadings,
        dtype=jnp.float64,
    )
    if instrument_universe == "restricted":
        strikes = _initial_nms_strikes(support, initial_tower)
    elif instrument_universe == "all_nodes":
        strikes = np.broadcast_to(
            np.asarray(initial.support[1:-1], dtype=np.float64),
            (len(EXPIRY_DAYS), initial.support.size - 2),
        ).copy()
    else:
        raise ValueError(f"unknown instrument universe {instrument_universe!r}")
    strike_count = strikes.shape[1]
    instrument_count = 1 + len(EXPIRY_DAYS) * strike_count
    feature_count = factor_rank + 1 + instrument_count - 1 + 1 + len(EXPIRY_DAYS)
    features = np.empty((path_count, horizon, feature_count), dtype=np.float32)
    returns = np.empty((path_count, horizon, instrument_count), dtype=np.float32)
    active_mask = _active_mask(strike_count)
    selected_initial_tower = initial_tower[np.asarray(EXPIRY_DAYS) - 1]
    initial_calls = np.asarray(
        black_mixture_call_prices(
            support,
            selected_initial_tower,
            jnp.asarray(strikes, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            jnp.asarray(EXPIRY_DAYS, dtype=jnp.float64),
        )
    )
    if instrument_universe == "restricted":
        initial_options = _option_values_from_calls(initial_calls, 1.0, strikes)
    else:
        initial_options = initial_calls
    daily_step = jax.jit(
        one_day_risk_neutral_substepped,
        static_argnames=("expiry_dt", "time_power"),
    )
    selected_pricer = jax.jit(black_mixture_call_prices)
    brownian_weights = np.asarray(
        substep_brownian_weights(
            dlv_substeps,
            time_power=time_power,
            dtype=jnp.float64,
        )
    )
    dt = 1.0 / DAYS_PER_YEAR
    generator = np.random.default_rng(seed)

    completed = 0
    while completed < path_count:
        valid_count = min(batch_size, path_count - completed)
        if valid_count % 2:
            raise RuntimeError("a generation batch split an antithetic pair")
        pair_count = batch_size // 2
        independent_factor_normals = generator.standard_normal(
            (horizon, dlv_substeps, pair_count, factor_rank)
        )
        factor_normals = np.stack(
            (independent_factor_normals, -independent_factor_normals),
            axis=3,
        ).reshape(horizon, dlv_substeps, batch_size, factor_rank)
        independent_background_normals = generator.standard_normal(
            (horizon, pair_count)
        )
        background_normals = np.stack(
            (independent_background_normals, -independent_background_normals),
            axis=2,
        ).reshape(horizon, batch_size)
        q_state = RiskNeutralState(
            initial_tower[0],
            jnp.asarray(initial.dlvsq[1:], dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
        )
        current_spot = np.ones(batch_size, dtype=np.float64)
        current_options = np.broadcast_to(
            initial_options,
            (batch_size,) + initial_options.shape,
        ).copy()
        factor_history = np.zeros((batch_size, factor_rank), dtype=np.float64)

        for day_index in range(horizon):
            valid_option_mask = np.asarray(EXPIRY_DAYS) > day_index
            normalized_option = current_options / current_spot[:, None, None]
            log_option = np.where(
                valid_option_mask[:, None],
                np.log(np.maximum(normalized_option, 1.0e-12)),
                0.0,
            )
            time_to_expiry = np.maximum(
                np.asarray(EXPIRY_DAYS, dtype=np.float64) - day_index,
                0.0,
            ) / horizon
            batch_features = np.concatenate(
                (
                    factor_history,
                    np.log(current_spot)[:, None],
                    log_option.reshape(batch_size, -1),
                    np.full((batch_size, 1), day_index / horizon),
                    np.broadcast_to(time_to_expiry, (batch_size, len(EXPIRY_DAYS))),
                ),
                axis=1,
            )
            features[completed : completed + valid_count, day_index] = batch_features[
                :valid_count
            ]

            future_count = q_state.future_dlvsq.shape[-2]
            result = daily_step(
                support,
                q_state,
                loadings[:future_count],
                jnp.asarray(factor_normals[day_index], dtype=jnp.float64),
                jnp.asarray(background_normals[day_index], dtype=jnp.float64),
                expiry_dt=dt,
                time_power=time_power,
            )
            next_spot = np.asarray(result.spot)
            next_options = current_options.copy()
            settlement = np.asarray(EXPIRY_DAYS) == day_index + 1
            if np.any(settlement):
                settlement_strikes = strikes[settlement]
                call_payoff = np.maximum(
                    next_spot[:, None, None] - settlement_strikes[None],
                    0.0,
                )
                if instrument_universe == "restricted":
                    put_payoff = np.maximum(
                        settlement_strikes[None] - next_spot[:, None, None],
                        0.0,
                    )
                    put_mask = np.asarray(NORMALIZED_MONEYNESS) > 0.0
                    next_options[:, settlement] = np.where(
                        put_mask[None, None, :],
                        put_payoff,
                        call_payoff,
                    )
                else:
                    next_options[:, settlement] = call_payoff
            live = np.asarray(EXPIRY_DAYS) > day_index + 1
            if np.any(live):
                live_expiries = np.asarray(EXPIRY_DAYS)[live]
                tower_rows = live_expiries - day_index - 2
                live_calls = np.asarray(
                    selected_pricer(
                        support,
                        result.density_tower[:, tower_rows],
                        jnp.asarray(strikes[live][None], dtype=jnp.float64),
                        result.state.background_level,
                        jnp.asarray(live_expiries - day_index - 1, dtype=jnp.float64),
                    )
                )
                if instrument_universe == "restricted":
                    next_options[:, live] = _option_values_from_calls(
                        live_calls,
                        next_spot,
                        strikes[live],
                    )
                else:
                    next_options[:, live] = live_calls

            batch_returns = _spot_normalized_price_increments(
                current_spot,
                next_spot,
                current_options,
                next_options,
                valid_option_mask,
            )
            if np.any(~np.isfinite(batch_returns)):
                raise FloatingPointError("front-spot OCE returns are non-finite")
            returns[completed : completed + valid_count, day_index] = batch_returns[
                :valid_count
            ]

            daily_factor_normal = np.einsum(
                "s,sbf->bf",
                brownian_weights,
                factor_normals[day_index],
            )
            factor_history += math.sqrt(dt) * daily_factor_normal
            q_state = result.state
            current_spot = next_spot
            current_options = next_options

        completed += valid_count
        print(
            json.dumps(
                {
                    "stage": "generate",
                    "completed_paths": completed,
                    "path_count": path_count,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    if np.any(~np.isfinite(features)) or np.any(~np.isfinite(returns)):
        raise FloatingPointError("front-spot OCE population is non-finite")
    if np.any(returns[:, ~active_mask]):
        raise RuntimeError("inactive return coordinates must be zero")
    return FrontSpotOcePopulation(
        features=features,
        returns=returns,
        active_mask=active_mask,
        strikes=strikes,
        instrument_labels=_instrument_labels(
            strike_count,
            instrument_universe=instrument_universe,
        ),
        instrument_universe=instrument_universe,
        initial_date=initial.date,
        generation_seconds=time.perf_counter() - started,
    )


def _feature_normalization(
    features: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    mean = np.mean(features, axis=(0, 1), dtype=np.float64)
    scale = np.std(features, axis=(0, 1), dtype=np.float64)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def _population_transition_gain(
    policy: FrontSpotOcePolicy,
    features: NDArray[np.float32],
    returns: NDArray[np.float32],
    active_mask: torch.Tensor,
    *,
    batch_size: int,
    epsilon: float,
) -> NDArray[np.float64]:
    transition_gain = np.empty(features.shape[:2], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            feature = torch.as_tensor(
                features[start:stop], dtype=policy.fdtype, device=policy.device
            )
            instrument_return = torch.as_tensor(
                returns[start:stop], dtype=policy.fdtype, device=policy.device
            )
            action = policy(feature, active_mask)
            transition_gain[start:stop] = (
                regularized_transition_gain(
                    action,
                    instrument_return,
                    epsilon=epsilon,
                )
                .double()
                .cpu()
                .numpy()
            )
    return transition_gain


def _fit_policy(
    features: NDArray[np.float32],
    returns: NDArray[np.float32],
    active_mask_array: NDArray[np.bool_],
    *,
    label: str,
    train_config: FrontSpotOceTrainConfig,
    output_path: Path,
    device: str = "cpu",
    affine_only: bool = False,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[FrontSpotOcePolicy, list[dict[str, float | int]]]:
    torch.manual_seed(train_config.seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for OCE training but is unavailable")
    if device == "cuda":
        torch.cuda.manual_seed_all(train_config.seed)
    env = BaseEnvironment(precision=32, device=device, debug=False)
    mean, scale = _feature_normalization(features)
    policy_config = FrontSpotOcePolicyConfig(
        label=label,
        feature_count=features.shape[-1],
        instrument_count=returns.shape[-1],
        network=FeedForwardConfig(
            activation="SiLU",
            num_depth=2,
            num_width=32,
        ),
    )
    policy = FrontSpotOcePolicy(
        env,
        config=policy_config,
        feature_mean=torch.as_tensor(mean, dtype=env.fdtype, device=env.device),
        feature_scale=torch.as_tensor(scale, dtype=env.fdtype, device=env.device),
    )
    if initial_state_dict is not None:
        policy.load_state_dict(initial_state_dict)
    active_mask = torch.as_tensor(
        active_mask_array,
        dtype=torch.bool,
        device=env.device,
    )
    parameters = (
        tuple(policy.affine.parameters())
        if affine_only
        else tuple(policy.parameters())
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(train_config.maximum_epochs):
        transition_gain = _population_transition_gain(
            policy,
            features,
            returns,
            active_mask,
            batch_size=train_config.batch_size,
            epsilon=train_config.epsilon,
        )
        log_weight = -transition_gain
        maximum = float(np.max(log_weight))
        weights = np.exp(log_weight - maximum)
        weights /= np.sum(weights, dtype=np.float64)
        log_loss = maximum + math.log(float(np.mean(np.exp(log_weight - maximum))))
        if log_loss < best_loss - train_config.minimum_improvement:
            best_loss = log_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            }

        policy.zero_grad(set_to_none=True)
        for start in range(0, len(features), train_config.batch_size):
            stop = min(start + train_config.batch_size, len(features))
            feature = torch.as_tensor(
                features[start:stop], dtype=policy.fdtype, device=policy.device
            )
            instrument_return = torch.as_tensor(
                returns[start:stop], dtype=policy.fdtype, device=policy.device
            )
            action = policy(feature, active_mask)
            batch_gain = regularized_transition_gain(
                action,
                instrument_return,
                epsilon=train_config.epsilon,
            )
            probability = torch.as_tensor(
                weights[start:stop],
                dtype=batch_gain.dtype,
                device=batch_gain.device,
            )
            torch.sum(-probability * batch_gain).backward()
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
            candidate_gain = _population_transition_gain(
                policy,
                features,
                returns,
                active_mask,
                batch_size=train_config.batch_size,
                epsilon=train_config.epsilon,
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
            "negative_expectation": -math.exp(log_loss),
            "candidate_log_expectation": candidate_loss,
            "gradient_l2_norm_before_clipping": gradient_norm,
            "gradient_l2_norm_after_clipping": math.sqrt(direction_norm_square),
            "accepted_step_size": accepted_step,
            "line_search_evaluations": line_search_evaluations,
            "tilted_effective_sample_size": float(1.0 / np.sum(weights**2)),
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps({"stage": "train", "label": label, **record}), flush=True)
        if accepted_step == 0.0:
            break

    final_gain = _population_transition_gain(
        policy,
        features,
        returns,
        active_mask,
        batch_size=train_config.batch_size,
        epsilon=train_config.epsilon,
    )
    final_log_loss = negative_exponential_statistics(final_gain)["log_expectation"]
    if final_log_loss < best_loss:
        best_loss = final_log_loss
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        }
    if best_state is None:
        raise RuntimeError("OCE optimizer did not produce a finite checkpoint")
    policy.load_state_dict(best_state)
    _atomic_torch(
        {
            "schema": "front-spot-local-oce-policy-v1",
            "policy_code_version": FrontSpotOcePolicyConfig.version.unique_id48,
            "policy_user_version": policy_config.user_version,
            "policy_config": asdict(policy_config),
            "train_config": asdict(train_config),
            "affine_only": affine_only,
            "state_dict": best_state,
            "epochs": len(history),
            "best_log_expectation": best_loss,
            "torch_rng_state": torch.get_rng_state(),
            "history": history,
        },
        output_path,
    )
    return policy, history


def _load_policy_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
    expected_train_config: FrontSpotOceTrainConfig,
) -> tuple[FrontSpotOcePolicy, dict[str, Any]]:
    """Reload a frozen policy without refitting or touching held-out data."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "front-spot-local-oce-policy-v1":
        raise RuntimeError(f"unexpected OCE checkpoint schema in {checkpoint_path}")
    if checkpoint.get("policy_code_version") != FrontSpotOcePolicyConfig.version.unique_id48:
        raise RuntimeError(f"outdated OCE policy checkpoint {checkpoint_path}")
    if checkpoint.get("train_config") != asdict(expected_train_config):
        raise RuntimeError(
            f"saved OCE training configuration differs from the fixed cross-fit setup: "
            f"{checkpoint_path}"
        )
    policy_dict = checkpoint["policy_config"]
    network_dict = policy_dict["network"]
    policy_config = FrontSpotOcePolicyConfig(
        label=policy_dict["label"],
        group=policy_dict["group"],
        user_version=policy_dict["user_version"],
        feature_count=policy_dict["feature_count"],
        instrument_count=policy_dict["instrument_count"],
        network=FeedForwardConfig(**network_dict),
    )
    if checkpoint.get("policy_user_version") != policy_config.user_version:
        raise RuntimeError(f"outdated OCE user checkpoint {checkpoint_path}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for OCE evaluation but is unavailable")
    env = BaseEnvironment(precision=32, device=device, debug=False)
    state_dict = checkpoint["state_dict"]
    policy = FrontSpotOcePolicy(
        env,
        config=policy_config,
        feature_mean=state_dict["feature_mean"].to(env.device),
        feature_scale=state_dict["feature_scale"].to(env.device),
    )
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, checkpoint


def _evaluate_policy(
    policy: FrontSpotOcePolicy,
    features: NDArray[np.float32],
    returns: NDArray[np.float32],
    active_mask_array: NDArray[np.bool_],
    *,
    batch_size: int,
    epsilon: float,
) -> tuple[dict[str, Any], NDArray[np.float64], NDArray[np.float64]]:
    active_mask = torch.as_tensor(
        active_mask_array,
        dtype=torch.bool,
        device=policy.device,
    )
    transition_gain = np.empty(features.shape[:2], dtype=np.float64)
    raw_path_average = np.empty(len(features), dtype=np.float64)
    outer_sum = np.zeros(
        (returns.shape[-1], returns.shape[-1]),
        dtype=np.float64,
    )
    path_energy_sum = 0.0
    path_energy_square_sum = 0.0
    action_square_sum = 0.0
    maximum_absolute_action = 0.0
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            feature = torch.as_tensor(
                features[start:stop], dtype=policy.fdtype, device=policy.device
            )
            instrument_return = torch.as_tensor(
                returns[start:stop], dtype=policy.fdtype, device=policy.device
            )
            action = policy(feature, active_mask)
            transition_gain[start:stop] = (
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
            raw_path_average[start:stop] = np.mean(
                np.sum(
                    action_array * returns[start:stop].astype(np.float64),
                    axis=-1,
                ),
                axis=1,
                dtype=np.float64,
            )
            path_energy = np.sum(action_array**2, axis=(1, 2), dtype=np.float64)
            path_energy_sum += float(np.sum(path_energy, dtype=np.float64))
            path_energy_square_sum += float(np.sum(path_energy**2, dtype=np.float64))
            action_square_sum += float(np.sum(action_array**2, dtype=np.float64))
            maximum_absolute_action = max(
                maximum_absolute_action,
                float(np.max(np.abs(action_array))),
            )
            outer_sum += np.einsum(
                "pti,ptj->ij",
                action_array,
                action_array,
                dtype=np.float64,
                optimize=True,
            )
    pointwise_count = len(features) * features.shape[1]
    outer = outer_sum / pointwise_count
    expected_path_energy = path_energy_sum / len(features)
    path_energy_variance = max(
        (
            path_energy_square_sum
            - path_energy_sum**2 / len(features)
        )
        / max(len(features) - 1, 1),
        0.0,
    )
    pointwise_energy = action_square_sum / pointwise_count
    if not np.isclose(np.trace(outer), pointwise_energy, rtol=2.0e-6, atol=1.0e-10):
        raise RuntimeError("streamed pointwise action moment failed its trace identity")
    summary: dict[str, Any] = negative_exponential_statistics(transition_gain)
    summary["path_count"] = len(features)
    summary["transition_count"] = int(np.prod(features.shape[:2]))
    summary["zero_policy_log_expectation"] = 0.0
    summary["log_expectation_improvement_from_zero"] = -summary["log_expectation"]
    summary["expectation_improvement_from_zero"] = 1.0 - summary["expectation"]
    summary["mean_path_average_raw_action_return"] = float(
        np.mean(raw_path_average, dtype=np.float64)
    )
    paired_drift, _ = paired_path_average_statistics(raw_path_average)
    summary["paired_conditional_drift"] = paired_drift
    summary["expected_path_action_energy"] = expected_path_energy
    summary["standard_error_path_action_energy"] = math.sqrt(
        path_energy_variance / len(features)
    )
    summary["expected_pointwise_action_energy"] = pointwise_energy
    summary["maximum_absolute_action"] = maximum_absolute_action
    summary["rms_action"] = math.sqrt(
        action_square_sum / (pointwise_count * returns.shape[-1])
    )
    summary["pointwise_outer_trace"] = float(np.trace(outer))
    summary["pointwise_outer_diagonal"] = np.diag(outer).tolist()
    return summary, outer, raw_path_average


def run(arguments: argparse.Namespace) -> None:
    """Fit on antithetic sample A and audit frozen actions on sample B."""

    if arguments.paths != 100_000:
        raise ValueError("this requested experiment is defined on exactly 100,000 paths")
    if arguments.factor_rank not in (8, 16):
        raise ValueError("this conditional-drift experiment requires rank 8 or 16")
    if arguments.dlv_substeps < 1:
        raise ValueError("at least one DLV substep is required")
    torch.set_num_threads(arguments.torch_threads)
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    artifact_stem = _artifact_stem(
        arguments.factor_rank,
        arguments.dlv_substeps,
        arguments.instrument_universe,
    )
    policy_label = artifact_stem.replace("_", "-")
    lock = FileLock(output / f"{policy_label}.lock")
    with lock:
        population = _load_population_cache(
            arguments.population_cache,
            path_count=arguments.paths,
            factor_rank=arguments.factor_rank,
            dlv_substeps=arguments.dlv_substeps,
            instrument_universe=arguments.instrument_universe,
            seed=arguments.seed,
        )
        if population is None:
            calibration = calibrate_front_spot_model(
                fit_label=arguments.fit_label,
                factor_rank=arguments.factor_rank,
                batch_size=arguments.calibration_batch_size,
                smoothing_sigma=(arguments.expiry_smoothing, arguments.strike_smoothing),
            )
            population = generate_front_spot_oce_population(
                calibration,
                path_count=arguments.paths,
                batch_size=arguments.generation_batch_size,
                fit_label=arguments.fit_label,
                date_index=arguments.date_index,
                dlv_substeps=arguments.dlv_substeps,
                time_power=arguments.time_power,
                seed=arguments.seed,
                instrument_universe=arguments.instrument_universe,
            )
            _save_population_cache(
                population,
                arguments.population_cache,
                factor_rank=arguments.factor_rank,
                dlv_substeps=arguments.dlv_substeps,
                instrument_universe=arguments.instrument_universe,
                seed=arguments.seed,
            )
        else:
            print(
                json.dumps(
                    {
                        "stage": "load-population-cache",
                        "path": str(arguments.population_cache),
                        "path_count": len(population.features),
                    }
                ),
                flush=True,
            )
        train_config = FrontSpotOceTrainConfig(
            batch_size=arguments.training_batch_size,
            maximum_epochs=arguments.maximum_epochs,
            learning_rate=arguments.learning_rate,
            minimum_improvement=arguments.minimum_improvement,
            maximum_gradient_norm=arguments.maximum_gradient_norm,
            epsilon=arguments.epsilon,
            seed=arguments.policy_seed,
        )
        split = arguments.paths // 2
        affine_A_path = output / f"{artifact_stem}_affine_A.pt"
        policy_A_path = output / f"{artifact_stem}_A.pt"
        if arguments.reuse_forward_checkpoints:
            affine_policy, affine_checkpoint = _load_policy_checkpoint(
                affine_A_path,
                device=arguments.device,
                expected_train_config=train_config,
            )
            policy, policy_checkpoint = _load_policy_checkpoint(
                policy_A_path,
                device=arguments.device,
                expected_train_config=train_config,
            )
            affine_history = affine_checkpoint["history"]
            history = policy_checkpoint["history"]
            forward_checkpoint_source = "reloaded frozen v5 A-only checkpoints"
        else:
            affine_policy, affine_history = _fit_policy(
                population.features[:split],
                population.returns[:split],
                population.active_mask,
                label=f"{policy_label}-affine-A",
                train_config=train_config,
                output_path=affine_A_path,
                device=arguments.device,
                affine_only=True,
            )
            affine_initial_state = {
                name: value.detach().cpu().clone()
                for name, value in affine_policy.state_dict().items()
            }
            policy, history = _fit_policy(
                population.features[:split],
                population.returns[:split],
                population.active_mask,
                label=f"{policy_label}-A",
                train_config=train_config,
                output_path=policy_A_path,
                device=arguments.device,
                initial_state_dict=affine_initial_state,
            )
            forward_checkpoint_source = "fitted in this invocation on sample A only"
        affine_training_summary, _, _ = _evaluate_policy(
            affine_policy,
            population.features[:split],
            population.returns[:split],
            population.active_mask,
            batch_size=arguments.training_batch_size,
            epsilon=arguments.epsilon,
        )
        affine_heldout_summary, _, _ = _evaluate_policy(
            affine_policy,
            population.features[split:],
            population.returns[split:],
            population.active_mask,
            batch_size=arguments.training_batch_size,
            epsilon=arguments.epsilon,
        )
        training_summary, training_outer, _ = _evaluate_policy(
            policy,
            population.features[:split],
            population.returns[:split],
            population.active_mask,
            batch_size=arguments.training_batch_size,
            epsilon=arguments.epsilon,
        )
        heldout_summary, heldout_outer, heldout_path_scores = _evaluate_policy(
            policy,
            population.features[split:],
            population.returns[split:],
            population.active_mask,
            batch_size=arguments.training_batch_size,
            epsilon=arguments.epsilon,
        )
        matrix_values: dict[str, NDArray[Any]] = {
            "training_A": training_outer,
            "heldout_B": heldout_outer,
            "labels": np.asarray(population.instrument_labels),
        }
        crossfit_report: dict[str, Any] | None = None
        if arguments.reverse_crossfit:
            affine_B_policy, affine_B_history = _fit_policy(
                population.features[split:],
                population.returns[split:],
                population.active_mask,
                label=f"{policy_label}-affine-B",
                train_config=train_config,
                output_path=output / f"{artifact_stem}_affine_B.pt",
                device=arguments.device,
                affine_only=True,
            )
            affine_B_initial_state = {
                name: value.detach().cpu().clone()
                for name, value in affine_B_policy.state_dict().items()
            }
            affine_B_training_summary, _, _ = _evaluate_policy(
                affine_B_policy,
                population.features[split:],
                population.returns[split:],
                population.active_mask,
                batch_size=arguments.training_batch_size,
                epsilon=arguments.epsilon,
            )
            affine_B_heldout_summary, _, _ = _evaluate_policy(
                affine_B_policy,
                population.features[:split],
                population.returns[:split],
                population.active_mask,
                batch_size=arguments.training_batch_size,
                epsilon=arguments.epsilon,
            )
            policy_B, history_B = _fit_policy(
                population.features[split:],
                population.returns[split:],
                population.active_mask,
                label=f"{policy_label}-B",
                train_config=train_config,
                output_path=output / f"{artifact_stem}_B.pt",
                device=arguments.device,
                initial_state_dict=affine_B_initial_state,
            )
            reverse_training_summary, reverse_training_outer, _ = _evaluate_policy(
                policy_B,
                population.features[split:],
                population.returns[split:],
                population.active_mask,
                batch_size=arguments.training_batch_size,
                epsilon=arguments.epsilon,
            )
            reverse_heldout_summary, reverse_heldout_outer, reverse_path_scores = (
                _evaluate_policy(
                    policy_B,
                    population.features[:split],
                    population.returns[:split],
                    population.active_mask,
                    batch_size=arguments.training_batch_size,
                    epsilon=arguments.epsilon,
                )
            )
            out_of_fold_path_scores = np.concatenate(
                (reverse_path_scores, heldout_path_scores)
            )
            out_of_fold_summary, out_of_fold_pair_scores = (
                paired_path_average_statistics(out_of_fold_path_scores)
            )
            out_of_fold_outer = 0.5 * (reverse_heldout_outer + heldout_outer)
            out_of_fold_pointwise_energy = float(np.trace(out_of_fold_outer))
            matrix_values.update(
                {
                    "training_B": reverse_training_outer,
                    "heldout_A": reverse_heldout_outer,
                    "out_of_fold": out_of_fold_outer,
                    "out_of_fold_pair_scores": out_of_fold_pair_scores,
                }
            )
            crossfit_report = {
                "fixed_hyperparameters": (
                    "The B-to-A fold uses the identical architecture, optimizer, "
                    "epoch count, and line-search configuration selected before "
                    "examining either held-out fold."
                ),
                "normalization": (
                    "B-to-A feature normalization is fitted on B only and frozen "
                    "before evaluating A."
                ),
                "affine_only_baseline_B_to_A": {
                    "training_B": affine_B_training_summary,
                    "heldout_A": affine_B_heldout_summary,
                    "training_epochs": len(affine_B_history),
                },
                "B_to_A": {
                    "training_B": reverse_training_summary,
                    "heldout_A": reverse_heldout_summary,
                    "training_epochs": len(history_B),
                },
                "combined_out_of_fold": {
                    "paired_conditional_drift": out_of_fold_summary,
                    "expected_pointwise_action_energy": out_of_fold_pointwise_energy,
                    "pointwise_outer_trace": out_of_fold_pointwise_energy,
                    "out_of_fold_path_count": arguments.paths,
                    "out_of_fold_pair_count": arguments.paths // 2,
                },
            }
        matrix_path = output / f"{artifact_stem}_action_moments.npz"
        np.savez_compressed(matrix_path, **matrix_values)
        unconditional_audit = unconditional_spot_scaled_martingale_audit(
            population.returns,
            population.active_mask,
            population.instrument_labels,
        )
        report = {
            "schema": (
                f"front-spot-rank{arguments.factor_rank}-"
                f"{_substep_name(arguments.dlv_substeps).replace('_', '-')}-"
                "spot-scaled-local-oce-v2"
            ),
            "objective": {
                "requested": "sup_theta -E[exp(-a(theta)'R+0.5*a(theta)'a(theta))]",
                "pooled_transition_form": (
                    "sup_theta -mean_{(path,time) in A} exp(-a_theta(X)'R+"
                    "epsilon*a_theta(X)'a_theta(X)/2)"
                ),
                "optimized_equivalent": (
                    "min_theta log mean_{(path,time) in A} exp(-a_theta(X)'R+"
                    "epsilon*a_theta(X)'a_theta(X)/2)"
                ),
                "epsilon": arguments.epsilon,
                "sign_note": (
                    "Without the leading minus sign on the expectation, the stated "
                    "uncapped supremum is infinite even for constant actions."
                ),
            },
            "population": {
                "initial_date": population.initial_date,
                "measure": "Q",
                "path_count": arguments.paths,
                "antithetic_pair_count": arguments.paths // 2,
                "factor_rank": arguments.factor_rank,
                "dlv_substeps": arguments.dlv_substeps,
                "horizon_days": population.features.shape[1],
                "feature_count": population.features.shape[2],
                "instrument_count": population.returns.shape[2],
                "instrument_universe": population.instrument_universe,
                "book_expiry_days": list(EXPIRY_DAYS),
                "fixed_cash_strikes": population.strikes.tolist(),
                "instrument_labels": list(population.instrument_labels),
                "return_definition": (
                    "one-day spot and fixed-cash-strike option price increments "
                    "divided by pre-step spot; expiry-day option increments use "
                    "settlement payoff"
                ),
                "policy_features": (
                    f"{arguments.factor_rank} cumulative Brownian factor histories, "
                    "log spot, live log option-price/spot ratios, calendar time, "
                    "and four times to expiry"
                ),
                "split": (
                    "adjacent paths are antithetic pairs; complete pairs [0,25000) "
                    "form 50000-path training sample A and complete pairs [25000,50000) "
                    "form 50000-path held-out sample B"
                ),
                "split_integrity": (
                    "all path-time observations remain with their complete path and "
                    "both members of every antithetic pair remain in the same sample"
                ),
                "generation_seconds": population.generation_seconds,
            },
            "policy": {
                "architecture": "uncapped signed affine plus two-layer width-32 SiLU network",
                "affine_baseline_training_epochs": len(affine_history),
                "training_epochs": len(history),
                "forward_checkpoint_source": forward_checkpoint_source,
                "training_device": arguments.device,
                "train_config": asdict(train_config),
                "normalization": "fitted on sample A only and frozen on sample B",
                "heldout_use": (
                    "sample B is used once after training; it is not used for early "
                    "stopping, tuning, normalization, or checkpoint selection"
                ),
            },
            "affine_only_baseline": {
                "training_A": affine_training_summary,
                "heldout_B": affine_heldout_summary,
                "role": (
                    "convex affine A-only baseline and initialization for the full "
                    "affine-plus-deep policy"
                ),
            },
            "training_A": training_summary,
            "heldout_B": heldout_summary,
            "headline_heldout_conditional_drift": heldout_summary[
                "paired_conditional_drift"
            ],
            "train_vs_heldout": {
                "log_expectation_B_minus_A": (
                    heldout_summary["log_expectation"]
                    - training_summary["log_expectation"]
                ),
                "expectation_B_minus_A": (
                    heldout_summary["expectation"] - training_summary["expectation"]
                ),
                "log_improvement_B_minus_A": (
                    heldout_summary["log_expectation_improvement_from_zero"]
                    - training_summary["log_expectation_improvement_from_zero"]
                ),
            },
            "heldout_tail_diagnostic": {
                "interpretation": (
                    "Spot-normalized increments avoid dividing by tiny option "
                    "prices. Tilted effective sample size and maximum log payoff "
                    "diagnose any remaining concentration in the exponential loss."
                ),
                "training_tilted_effective_sample_size": training_summary[
                    "tilted_effective_sample_size"
                ],
                "heldout_tilted_effective_sample_size": heldout_summary[
                    "tilted_effective_sample_size"
                ],
                "training_maximum_log_payoff": training_summary[
                    "maximum_log_payoff"
                ],
                "heldout_maximum_log_payoff": heldout_summary[
                    "maximum_log_payoff"
                ],
            },
            "unconditional_spot_scaled_martingale_audit": unconditional_audit,
            "reverse_crossfit": crossfit_report,
            "action_outer_moment": {
                "definition": "E_{path,time}[a_t a_t'] with expired coordinates zero",
                "trace_definition": "E_{path,time}[a_t'a_t]",
                "artifact": str(matrix_path),
                "keys": list(matrix_values),
            },
        }
        _atomic_json(report, output / f"{artifact_stem}.json")
        print(json.dumps(report, indent=2), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "quantdata/sdlv/daily/front_spot/results/"
            "oce_rank8_eight_step_spot_scaled"
        ),
    )
    parser.add_argument("--fit-label", default="full")
    parser.add_argument("--date-index", type=int, default=-1)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--factor-rank", type=int, default=8)
    parser.add_argument(
        "--instrument-universe",
        choices=("restricted", "all_nodes"),
        default="restricted",
    )
    parser.add_argument("--dlv-substeps", type=int, default=8)
    parser.add_argument("--time-power", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--policy-seed", type=int, default=20261006)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--calibration-batch-size", type=int, default=64)
    parser.add_argument("--expiry-smoothing", type=float, default=1.0)
    parser.add_argument("--strike-smoothing", type=float, default=2.0)
    parser.add_argument("--training-batch-size", type=int, default=2_048)
    parser.add_argument("--maximum-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--minimum-improvement", type=float, default=1.0e-7)
    parser.add_argument("--maximum-gradient-norm", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reuse-forward-checkpoints", action="store_true")
    parser.add_argument("--reverse-crossfit", action="store_true")
    parser.add_argument(
        "--population-cache",
        type=Path,
        default=Path(
            "/tmp/front_spot_rank8_eight_step_antithetic_spot_scaled_100k"
        ),
    )
    return parser


def main() -> None:
    """Run the requested rank-8, eight-substep OCE experiment."""

    run(_parser().parse_args())


if __name__ == "__main__":
    main()
