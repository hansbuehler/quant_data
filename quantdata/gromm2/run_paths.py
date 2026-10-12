"""Generate resumable six-month GROMM2 risk-neutral path panels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from quantdata.gromm2.dlv import decode_density_tower
from quantdata.gromm2.dynamics import (
    floating_level_surface,
    interpolate_loading_expiries,
    load_estimated_dynamics,
    selective_factor_scales,
)
from quantdata.gromm2.pricing import price_active_calls
from quantdata.gromm2.simulation import SimulationState, one_day_risk_neutral_step

AUDIT_BUSINESS_DTES = np.arange(5, 126, 5, dtype=np.int64)
AUDIT_NMS = np.asarray(
    [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5],
    dtype=np.float64,
)


def _antithetic_normals(
    *,
    seed: int,
    chunk_start: int,
    substeps: int,
    path_count: int,
    factor_count: int,
    days: int,
) -> tuple[np.ndarray, np.ndarray]:
    if path_count % 2:
        raise ValueError("each simulation chunk must contain complete antithetic pairs")
    generator = np.random.default_rng(np.random.SeedSequence([seed, chunk_start]))
    half = path_count // 2
    factor_half = generator.standard_normal(
        (days, substeps, half, factor_count)
    )
    background_half = generator.standard_normal((days, half))
    factor = np.stack((factor_half, -factor_half), axis=3).reshape(
        days, substeps, path_count, factor_count
    )
    background = np.stack((background_half, -background_half), axis=2).reshape(
        days, path_count
    )
    return factor, background


def _selected_strikes(
    *,
    business_dtes: np.ndarray,
    projected_total_variances: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(business_dtes, AUDIT_BUSINESS_DTES)
    if np.any(indices == len(business_dtes)) or np.any(
        business_dtes[indices] != AUDIT_BUSINESS_DTES
    ):
        raise ValueError("model expiry grid does not contain every five-day audit expiry")
    return np.exp(
        np.sqrt(projected_total_variances[indices])[:, None] * AUDIT_NMS[None, :]
    )


def _to_otm_prices(calls: jax.Array, spot: jax.Array, strikes: jax.Array) -> jax.Array:
    return jnp.where(
        strikes >= 1.0,
        calls,
        calls - spot[..., None, None] + strikes,
    )


def _create_path_file(
    path: Path,
    *,
    model: str,
    path_count: int,
    factor_count: int,
    substeps: int,
    seed: int,
    initial_date: str,
    audit_strikes: np.ndarray,
    initial_prices: np.ndarray,
    simulation_days: int,
    batch_size: int,
    calibrated_history: Path,
    estimate_path: Path,
    density_loading_floor: float,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = h5py.File(path, "w")
    stream.attrs["schema"] = "gromm2-q-paths-v2"
    stream.attrs["model"] = model
    stream.attrs["path_count"] = path_count
    stream.attrs["factor_count"] = factor_count
    stream.attrs["substeps"] = substeps
    stream.attrs["seed"] = seed
    stream.attrs["initial_date"] = initial_date
    stream.attrs["completed_paths"] = 0
    stream.attrs["simulation_days"] = simulation_days
    stream.attrs["batch_size"] = batch_size
    stream.attrs["calibrated_history"] = str(calibrated_history.resolve())
    stream.attrs["estimate_path"] = str(estimate_path.resolve())
    stream.attrs["density_loading_floor"] = density_loading_floor
    stream.create_dataset("audit_business_dtes", data=AUDIT_BUSINESS_DTES)
    stream.create_dataset("audit_nms", data=AUDIT_NMS)
    stream.create_dataset("audit_strikes", data=audit_strikes)
    stream.create_dataset("initial_option_values", data=initial_prices)
    chunk_count = math.ceil(path_count / batch_size)
    stream.create_dataset(
        "drift_taming_fraction",
        shape=(chunk_count, simulation_days),
        dtype=np.float32,
    )
    stream.create_dataset(
        "dlvsq_minimum",
        shape=(chunk_count, simulation_days),
        dtype=np.float32,
    )
    stream.create_dataset(
        "dlvsq_maximum",
        shape=(chunk_count, simulation_days),
        dtype=np.float32,
    )
    stream.create_dataset(
        "spot",
        shape=(path_count, simulation_days + 1),
        dtype=np.float32,
        chunks=(min(path_count, 4096), 1),
    )
    stream.create_dataset(
        "factor_scores",
        shape=(path_count, simulation_days, factor_count),
        dtype=np.float32,
        chunks=(min(path_count, 1024), 1, factor_count),
    )
    gains = stream.create_group("daily_option_gains_over_spot")
    for day in range(1, simulation_days + 1):
        active_count = int(np.sum(AUDIT_BUSINESS_DTES >= day))
        dataset = gains.create_dataset(
            f"day_{day:03d}",
            shape=(path_count, active_count, len(AUDIT_NMS)),
            dtype=np.float32,
            chunks=(min(path_count, 1024), 1, len(AUDIT_NMS)),
            compression="lzf",
        )
        dataset.attrs["audit_expiry_indices"] = np.flatnonzero(
            AUDIT_BUSINESS_DTES >= day
        )
    stream.flush()
    return stream


def simulate_paths(
    *,
    calibrated_history: Path,
    source_history: Path,
    estimate_path: Path,
    output: Path,
    model: str,
    path_count: int,
    batch_size: int,
    substeps: int,
    seed: int,
    simulation_days: int = 125,
    density_loading_floor: float = 1.0e-4,
) -> None:
    """Simulate one constant or selective Q model and retain audit paths."""
    if model not in {"constant", "selective"}:
        raise ValueError("model must be 'constant' or 'selective'")
    if path_count <= 0 or path_count % 2 or batch_size <= 0 or batch_size % 2:
        raise ValueError("path_count and batch_size must be positive and even")
    if not 1 <= simulation_days <= 125:
        raise ValueError("simulation_days must lie in [1, 125]")
    if not 0.0 < density_loading_floor < 1.0:
        raise ValueError("density_loading_floor must lie in (0, 1)")
    if not jax.config.x64_enabled:
        raise RuntimeError("risk-neutral simulation requires JAX_ENABLE_X64=1")
    with h5py.File(calibrated_history, "r") as stream:
        dates = np.asarray(stream["dates"]).astype("U10")
        supports = np.asarray(stream["supports"], dtype=np.float64)
        transports = np.asarray(stream["transports"], dtype=np.float64)
        expiries = np.asarray(stream["model_expiries_y"], dtype=np.float64)
        initial_dlvsq = np.asarray(stream["dlvsq"][-1], dtype=np.float64)
    with h5py.File(source_history, "r") as stream:
        business_dtes = np.asarray(stream["model_business_dtes"], dtype=np.float64)
        projected_total_variances = np.asarray(
            stream["reference_projected_total_variances"], dtype=np.float64
        )
    dynamics = load_estimated_dynamics(estimate_path)
    factor_count = dynamics.factor_count
    reference_factor_dtes = business_dtes[1:]
    intervals = np.diff(expiries, prepend=0.0)
    initial_tower = np.asarray(
        decode_density_tower(
            jnp.asarray(supports),
            jnp.asarray(transports),
            jnp.asarray(initial_dlvsq),
            jnp.asarray(intervals),
        )
    )
    audit_strikes = _selected_strikes(
        business_dtes=business_dtes,
        projected_total_variances=projected_total_variances,
    )
    audit_model_indices = np.searchsorted(business_dtes, AUDIT_BUSINESS_DTES)
    initial_calls = np.asarray(
        price_active_calls(
            jnp.asarray(supports[audit_model_indices]),
            jnp.asarray(initial_tower[audit_model_indices])[None],
            jnp.ones(1),
            jnp.asarray(AUDIT_BUSINESS_DTES / 252.0),
            jnp.asarray(audit_strikes),
        )
    )[0]
    initial_prices = np.where(
        audit_strikes >= 1.0,
        initial_calls,
        initial_calls - 1.0 + audit_strikes,
    )

    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists():
        raise FileExistsError(f"completed path artifact already exists: {output}")
    if partial.exists():
        stream = h5py.File(partial, "r+")
        if (
            stream.attrs.get("schema") != "gromm2-q-paths-v2"
            or stream.attrs.get("model") != model
            or int(stream.attrs.get("path_count", -1)) != path_count
            or int(stream.attrs.get("factor_count", -1)) != factor_count
            or int(stream.attrs.get("substeps", -1)) != substeps
            or int(stream.attrs.get("seed", -1)) != seed
            or int(stream.attrs.get("simulation_days", -1)) != simulation_days
            or int(stream.attrs.get("batch_size", -1)) != batch_size
            or float(stream.attrs.get("density_loading_floor", -1.0))
            != density_loading_floor
            or stream.attrs.get("calibrated_history")
            != str(calibrated_history.resolve())
            or stream.attrs.get("estimate_path") != str(estimate_path.resolve())
        ):
            stream.close()
            raise ValueError(f"incompatible partial path artifact: {partial}")
    else:
        stream = _create_path_file(
            partial,
            model=model,
            path_count=path_count,
            factor_count=factor_count,
            substeps=substeps,
            seed=seed,
            initial_date=str(dates[-1]),
            audit_strikes=audit_strikes,
            initial_prices=initial_prices,
            simulation_days=simulation_days,
            batch_size=batch_size,
            calibrated_history=calibrated_history,
            estimate_path=estimate_path,
            density_loading_floor=density_loading_floor,
        )
    completed = int(stream.attrs["completed_paths"])
    if completed % 2:
        stream.close()
        raise ValueError("partial path count does not end on an antithetic pair")

    daily_step = jax.jit(
        one_day_risk_neutral_step,
        static_argnames=("substeps",),
    )
    selective_step = jax.jit(
        lambda hidden, active_dlvsq, active_dtes, previous_score: (
            lambda floating: selective_factor_scales(
                dynamics=dynamics,
                hidden=hidden,
                floating_log_dlvsq=floating,
                previous_factor_score=previous_score,
            )
        )(
            floating_level_surface(
                active_dlvsq,
                active_dtes,
                jnp.asarray(reference_factor_dtes),
            )
        )
    )
    started = perf_counter()
    try:
        for start in range(completed, path_count, batch_size):
            stop = min(start + batch_size, path_count)
            count = stop - start
            if count % 2:
                raise ValueError("final simulation chunk splits an antithetic pair")
            factor_normals, background_normals = _antithetic_normals(
                seed=seed,
                chunk_start=start,
                substeps=substeps,
                path_count=count,
                factor_count=factor_count,
                days=simulation_days,
            )
            state = SimulationState(
                front_density=jnp.broadcast_to(
                    jnp.asarray(initial_tower[0]), (count, supports.shape[1])
                ),
                dlvsq=jnp.broadcast_to(
                    jnp.asarray(initial_dlvsq), (count,) + initial_dlvsq.shape
                ),
                background_level=jnp.ones(count),
            )
            hidden = jnp.broadcast_to(
                jnp.asarray(dynamics.ssm_terminal_hidden),
                (count, len(dynamics.ssm_terminal_hidden)),
            )
            previous_factor_score = jnp.zeros((count, factor_count))
            previous_spot = jnp.ones(count)
            previous_values = jnp.broadcast_to(
                jnp.asarray(initial_prices), (count,) + initial_prices.shape
            )
            stream["spot"][start:stop, 0] = 1.0
            chunk_index = start // batch_size
            front_index = 0
            for day in range(1, simulation_days + 1):
                while business_dtes[front_index] < day:
                    front_index += 1
                remaining_front_days = int(business_dtes[front_index] - (day - 1))
                active_supports = supports[front_index:]
                active_transports = transports[front_index:]
                active_intervals = intervals[front_index + 1 :]
                future_remaining_dtes = business_dtes[front_index + 1 :] - (day - 1)
                base_loadings = interpolate_loading_expiries(
                    dynamics.log_dlvsq_loadings,
                    reference_factor_dtes,
                    future_remaining_dtes,
                )
                if model == "selective" and len(future_remaining_dtes) > 0:
                    hidden, factor_scales = selective_step(
                        hidden,
                        state.dlvsq[..., front_index + 1 :, :],
                        jnp.asarray(future_remaining_dtes),
                        previous_factor_score,
                    )
                else:
                    factor_scales = jnp.ones((count, factor_count))
                active_loadings = (
                    jnp.asarray(base_loadings)[None]
                    * factor_scales[:, None, None, :]
                )
                result = daily_step(
                    jnp.asarray(active_supports),
                    jnp.asarray(active_transports),
                    jnp.asarray(active_intervals),
                    SimulationState(
                        front_density=state.front_density,
                        dlvsq=state.dlvsq[..., front_index:, :],
                        background_level=state.background_level,
                    ),
                    active_loadings,
                    jnp.asarray(factor_normals[day - 1]),
                    jnp.asarray(background_normals[day - 1]),
                    remaining_front_days=jnp.asarray(remaining_front_days),
                    substeps=substeps,
                    density_loading_floor=density_loading_floor,
                )
                eligible = np.flatnonzero(AUDIT_BUSINESS_DTES >= day)
                selected_model_indices = audit_model_indices[eligible]
                relative_indices = selected_model_indices - front_index
                selected_calls = price_active_calls(
                    jnp.asarray(supports[selected_model_indices]),
                    result.density_tower[..., relative_indices, :],
                    result.state.background_level,
                    jnp.asarray((AUDIT_BUSINESS_DTES[eligible] - day) / 252.0),
                    jnp.asarray(audit_strikes[eligible]),
                )
                next_values = _to_otm_prices(
                    selected_calls,
                    result.spot,
                    jnp.asarray(audit_strikes[eligible]),
                )
                gains = (next_values - previous_values[..., eligible, :]) / previous_spot[
                    ..., None, None
                ]
                diagnostic_values = (
                    result.spot,
                    result.state.dlvsq,
                    result.density_tower,
                    gains,
                )
                if any(
                    not np.all(np.isfinite(np.asarray(value)))
                    for value in diagnostic_values
                ):
                    raise FloatingPointError(
                        f"non-finite simulation state in paths {start}:{stop} on day {day}"
                    )
                stream["daily_option_gains_over_spot"][f"day_{day:03d}"][
                    start:stop
                ] = np.asarray(gains, dtype=np.float32)
                stream["spot"][start:stop, day] = np.asarray(
                    result.spot, dtype=np.float32
                )
                stream["factor_scores"][start:stop, day - 1] = np.asarray(
                    result.daily_factor_score, dtype=np.float32
                )
                stream["drift_taming_fraction"][chunk_index, day - 1] = float(
                    result.drift_taming_fraction
                )
                stream["dlvsq_minimum"][chunk_index, day - 1] = float(
                    jnp.min(result.state.dlvsq)
                )
                stream["dlvsq_maximum"][chunk_index, day - 1] = float(
                    jnp.max(result.state.dlvsq)
                )
                previous_values = previous_values.at[..., eligible, :].set(next_values)
                previous_spot = result.spot
                previous_factor_score = result.daily_factor_score
                state = SimulationState(
                    front_density=result.state.front_density,
                    dlvsq=state.dlvsq.at[..., front_index:, :].set(
                        result.state.dlvsq
                    ),
                    background_level=result.state.background_level,
                )
                if day == 1 or day % 5 == 0:
                    elapsed = perf_counter() - started
                    print(
                        f"paths model={model} batch={start}:{stop} "
                        f"day={day}/{simulation_days} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
            stream.attrs["completed_paths"] = stop
            stream.flush()
            elapsed = perf_counter() - started
            rate = (stop - completed) / max(elapsed, 1.0e-9)
            eta = (path_count - stop) / max(rate, 1.0e-9)
            print(
                f"paths model={model} completed={stop}/{path_count} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    finally:
        stream.close()
    partial.replace(output)
    print(f"wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-history", type=Path, required=True)
    parser.add_argument("--source-history", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("constant", "selective"), required=True)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--days", type=int, default=125)
    parser.add_argument("--density-loading-floor", type=float, default=1.0e-4)
    arguments = parser.parse_args()
    print(
        json.dumps(
            {
                "model": arguments.model,
                "paths": arguments.paths,
                "batch_size": arguments.batch_size,
                "substeps": arguments.substeps,
                "seed": arguments.seed,
                "days": arguments.days,
                "density_loading_floor": arguments.density_loading_floor,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    simulate_paths(
        calibrated_history=arguments.calibrated_history,
        source_history=arguments.source_history,
        estimate_path=arguments.estimate,
        output=arguments.output,
        model=arguments.model,
        path_count=arguments.paths,
        batch_size=arguments.batch_size,
        substeps=arguments.substeps,
        seed=arguments.seed,
        simulation_days=arguments.days,
        density_loading_floor=arguments.density_loading_floor,
    )


if __name__ == "__main__":
    main()
