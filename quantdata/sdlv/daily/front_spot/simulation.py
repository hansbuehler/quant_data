"""Path drivers and stability diagnostics for the daily front-spot model."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.daily import DATASET_PATH, load_raw_grid
from quantdata.sdlv.daily.front_spot.calibration import FrontSpotCalibration
from quantdata.sdlv.daily.front_spot.estimation import decode_dlvsq
from quantdata.sdlv.daily.front_spot.jax_model import (
    RiskNeutralState,
    decode_density_tower,
    one_day_physical_substepped,
    one_day_risk_neutral_substepped,
    substep_brownian_weights,
    terminal_front_step,
)


@dataclass(frozen=True)
class InitialModelState:
    """One observed 25-expiry state used to start simulations."""

    date: str
    date_index: int
    support: NDArray[np.float64]
    dlvsq: NDArray[np.float64]
    density_tower: NDArray[np.float64]


@dataclass(frozen=True)
class PilotReport:
    """Daily stability summaries for coupled Q and P pilot paths."""

    initial_date: str
    path_count: int
    factor_rank: int
    dlv_substeps: int
    time_power: float
    days: tuple[dict[str, float | int], ...]


def load_initial_model_state(
    *,
    fit_label: str = "full",
    date_index: int = -1,
    dt: float = 1.0 / DAYS_PER_YEAR,
) -> InitialModelState:
    """Load and decode one fitted 25-by-200 market state."""
    if not jax.config.x64_enabled:
        raise RuntimeError("Front-spot simulation requires JAX 64-bit mode")
    with np.load(DATASET_PATH, allow_pickle=False) as dataset:
        dates = tuple(str(value) for value in dataset["dates"])
        supports = np.asarray(dataset["supports"], dtype=np.float64)
    resolved_index = date_index if date_index >= 0 else len(dates) + date_index
    if not 0 <= resolved_index < len(dates):
        raise IndexError(f"date_index {date_index} is outside {len(dates)} states")
    support = supports[resolved_index]
    dlvsq = decode_dlvsq(load_raw_grid(fit_label, resolved_index))
    density_tower = np.asarray(
        decode_density_tower(
            jnp.asarray(support, dtype=jnp.float64),
            jnp.asarray(dlvsq, dtype=jnp.float64),
            dt=dt,
        )
    )
    return InitialModelState(
        date=dates[resolved_index],
        date_index=resolved_index,
        support=support,
        dlvsq=dlvsq,
        density_tower=density_tower,
    )


def _finite_maximum(value: jax.Array) -> float:
    array = np.asarray(value)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else float("nan")


def _finite_minimum(value: jax.Array) -> float:
    array = np.asarray(value)
    finite = array[np.isfinite(array)]
    return float(np.min(finite)) if finite.size else float("nan")


def run_cpu_pilot(
    calibration: FrontSpotCalibration,
    *,
    fit_label: str = "full",
    date_index: int = -1,
    path_count: int = 16,
    days: int = 25,
    seed: int = 20261004,
    dlv_substeps: int = 8,
    time_power: float = 1.0,
    dt: float = 1.0 / DAYS_PER_YEAR,
) -> PilotReport:
    """Run coupled Q/P paths and report numerical stability day by day."""
    if path_count < 1:
        raise ValueError(f"path_count must be positive, found {path_count}")
    if not 1 <= days <= 25:
        raise ValueError(f"days must lie in [1,25], found {days}")
    if dlv_substeps < 1:
        raise ValueError(f"dlv_substeps must be positive, found {dlv_substeps}")
    if time_power < 1.0 or not np.isfinite(time_power):
        raise ValueError(f"time_power must be finite and at least one, found {time_power}")
    initial = load_initial_model_state(
        fit_label=fit_label,
        date_index=date_index,
        dt=dt,
    )
    factor_rank = calibration.innovations.factor_rank
    generator = np.random.default_rng(seed)
    factor_normals = jnp.asarray(
        generator.standard_normal((days, dlv_substeps, path_count, factor_rank)),
        dtype=jnp.float64,
    )
    background_normals = jnp.asarray(
        generator.standard_normal((days, path_count)),
        dtype=jnp.float64,
    )
    support = jnp.asarray(initial.support, dtype=jnp.float64)
    initial_front = jnp.asarray(initial.density_tower[0], dtype=jnp.float64)
    initial_future = jnp.asarray(initial.dlvsq[1:], dtype=jnp.float64)
    q_state = RiskNeutralState(initial_front, initial_future, jnp.asarray(1.0))
    p_state = RiskNeutralState(initial_front, initial_future, jnp.asarray(1.0))
    loadings = jnp.asarray(
        calibration.innovations.annualized_loadings,
        dtype=jnp.float64,
    )
    physical_risk_price = jnp.asarray(
        calibration.physical.constant_market_price_of_risk,
        dtype=jnp.float64,
    )
    reports: list[dict[str, float | int]] = []
    q_daily_step = jax.jit(
        one_day_risk_neutral_substepped,
        static_argnames=("expiry_dt", "time_power"),
    )
    p_daily_step = jax.jit(
        one_day_physical_substepped,
        static_argnames=("expiry_dt", "time_power"),
    )
    brownian_weights = substep_brownian_weights(
        dlv_substeps,
        time_power=time_power,
        dtype=jnp.float64,
    )
    final_step = jax.jit(
        terminal_front_step,
        static_argnames=("dt",),
    )

    for day_index in range(days):
        future_count = q_state.future_dlvsq.shape[-2]
        if future_count == 0:
            q_terminal = final_step(
                support,
                q_state.front_density,
                q_state.background_level,
                jnp.einsum(
                    "s,sp->p",
                    brownian_weights,
                    factor_normals[day_index, :, :, 0],
                ),
                background_normals[day_index],
                dt=dt,
            )
            p_terminal = final_step(
                support,
                p_state.front_density,
                p_state.background_level,
                jnp.einsum(
                    "s,sp->p",
                    brownian_weights,
                    factor_normals[day_index, :, :, 0],
                )
                + physical_risk_price[..., 0] * np.sqrt(dt),
                background_normals[day_index],
                dt=dt,
            )
            reports.append(
                {
                    "day": day_index + 1,
                    "remaining_future_expiries": 0,
                    "q_spot_min": _finite_minimum(q_terminal.spot),
                    "q_spot_max": _finite_maximum(q_terminal.spot),
                    "p_spot_min": _finite_minimum(p_terminal.spot),
                    "p_spot_max": _finite_maximum(p_terminal.spot),
                    "nonfinite_count": int(
                        np.sum(~np.isfinite(np.asarray(q_terminal.spot)))
                        + np.sum(~np.isfinite(np.asarray(p_terminal.spot)))
                    ),
                }
            )
            break

        active_loadings = loadings[:future_count]
        q_result = q_daily_step(
            support,
            q_state,
            active_loadings,
            factor_normals[day_index],
            background_normals[day_index],
            expiry_dt=dt,
            time_power=time_power,
        )
        p_result = p_daily_step(
            support,
            p_state,
            active_loadings,
            physical_risk_price,
            factor_normals[day_index],
            background_normals[day_index],
            expiry_dt=dt,
            time_power=time_power,
        )
        q_state = q_result.state
        p_state = p_result.state
        q_mass_error = jnp.max(jnp.abs(jnp.sum(q_result.density_tower, axis=-1) - 1.0))
        p_mass_error = jnp.max(jnp.abs(jnp.sum(p_result.density_tower, axis=-1) - 1.0))
        q_forward_error = jnp.max(
            jnp.abs(q_result.density_tower @ support - q_result.latent_spot[..., None])
        )
        p_forward_error = jnp.max(
            jnp.abs(p_result.density_tower @ support - p_result.latent_spot[..., None])
        )
        diagnostic_values = (
            q_result.density_tower,
            p_result.density_tower,
            q_state.future_dlvsq,
            p_state.future_dlvsq,
            q_result.spot,
            p_result.spot,
        )
        nonfinite_count = sum(
            int(np.sum(~np.isfinite(np.asarray(value))))
            for value in diagnostic_values
        )
        reports.append(
            {
                "day": day_index + 1,
                "remaining_future_expiries": future_count - 1,
                "q_mass_max_error": float(q_mass_error),
                "p_mass_max_error": float(p_mass_error),
                "q_forward_max_error": float(q_forward_error),
                "p_forward_max_error": float(p_forward_error),
                "q_dlvsq_min": _finite_minimum(q_state.future_dlvsq),
                "q_dlvsq_max": _finite_maximum(q_state.future_dlvsq),
                "p_dlvsq_min": _finite_minimum(p_state.future_dlvsq),
                "p_dlvsq_max": _finite_maximum(p_state.future_dlvsq),
                "q_spot_min": _finite_minimum(q_result.spot),
                "q_spot_max": _finite_maximum(q_result.spot),
                "p_spot_min": _finite_minimum(p_result.spot),
                "p_spot_max": _finite_maximum(p_result.spot),
                "p_risk_price_abs_max": _finite_maximum(
                    jnp.abs(physical_risk_price)
                ),
                "nonfinite_count": nonfinite_count,
            }
        )
        if nonfinite_count:
            break

    return PilotReport(
        initial_date=initial.date,
        path_count=path_count,
        factor_rank=factor_rank,
        dlv_substeps=dlv_substeps,
        time_power=time_power,
        days=tuple(reports),
    )
