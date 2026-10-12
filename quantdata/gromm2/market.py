"""Reconstruct GROMM2 states exclusively from the published SANOS dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.bs import bs
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray
from scipy.linalg import solve_banded
from scipy.special import ndtr

from quantdata.gromm2.dlv import density0_from_support
from quantdata.gromm2.geometry import (
    Gromm2GeometryConfig,
    project_total_variance,
    transport_tower,
)
from quantdata.hedging.data.pricing import price_sanos_surface
from quantdata.hedging.data.public_dataset import PublicHedgingDataset

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class PublicSanosDay:
    """One published SANOS state with its documented pricing convention."""

    trading_date: date
    model_expiries_y: FloatArray
    kernel_total_variances: FloatArray
    support_strikes: tuple[FloatArray, ...]
    densities: tuple[FloatArray, ...]
    smoothness: float
    one_minute_y: float


@dataclass(frozen=True, kw_only=True)
class ReconstructedDay:
    """One positive six-month DLV tower implied by a public SANOS state."""

    trading_date: date
    raw_atm_total_variances: FloatArray
    projected_atm_total_variances: FloatArray
    target_densities: FloatArray
    model_densities: FloatArray
    dlvsq: FloatArray
    floored_raw_dlvsq_fraction: float
    capped_raw_dlvsq_fraction: float
    clipped_projection_mass: float
    maximum_density_error: float
    root_mean_square_price_error: float
    maximum_price_error: float


def load_public_sanos_days(dataset_path: Path) -> list[PublicSanosDay]:
    """Load the published SANOS parameters without refitting market data."""
    dataset = PublicHedgingDataset(Path(dataset_path))
    geometry = dataset.geometry
    geometry_rows = dataset.read("sanos_geometry.parquet")
    state_rows = dataset.read("sanos_state.parquet")
    if geometry_rows.height != state_rows.height:
        raise ValueError("published SANOS geometry and state rows do not align")
    days: list[PublicSanosDay] = []
    for geometry_row, state_row in zip(
        geometry_rows.iter_rows(named=True),
        state_rows.iter_rows(named=True),
        strict=True,
    ):
        if geometry_row["date"] != state_row["date"]:
            raise ValueError("published SANOS row dates do not align")
        days.append(
            PublicSanosDay(
                trading_date=geometry_row["date"],
                model_expiries_y=np.asarray(
                    geometry_row["model_expiries_y"], dtype=np.float64
                ),
                kernel_total_variances=np.asarray(
                    state_row["kernel_total_variances"], dtype=np.float64
                ),
                support_strikes=tuple(
                    np.asarray(value, dtype=np.float64)
                    for value in geometry_row["support_strikes"]
                ),
                densities=tuple(
                    np.asarray(value, dtype=np.float64)
                    for value in state_row["densities"]
                ),
                smoothness=float(geometry["sanos_smoothness"]),
                one_minute_y=float(geometry["one_minute_y"]),
            )
        )
    return days


def public_sanos_prices(
    day: PublicSanosDay,
    *,
    expiries_y: FloatArray,
    strikes: FloatArray,
) -> FloatArray:
    """Price calls using exactly the public SANOS interpolation/extrapolation."""
    expiry_array, strike_array = np.broadcast_arrays(
        np.asarray(expiries_y, dtype=np.float64),
        np.asarray(strikes, dtype=np.float64),
    )
    shape = expiry_array.shape
    result = price_sanos_surface(
        query_expiries_y=expiry_array.reshape(-1),
        query_strikes=strike_array.reshape(-1),
        is_call=np.ones(expiry_array.size, dtype=bool),
        model_expiries_y=day.model_expiries_y,
        atm_variances=day.kernel_total_variances,
        support_strikes=day.support_strikes,
        densities=day.densities,
        smoothness=day.smoothness,
        one_minute_y=day.one_minute_y,
    )
    return result.reshape(shape)


def public_sanos_atm_total_variances(
    day: PublicSanosDay,
    expiries_y: FloatArray,
) -> FloatArray:
    """Infer the full SANOS ATM variance, including terminal extrapolation."""
    expiries = np.asarray(expiries_y, dtype=np.float64)
    prices = public_sanos_prices(
        day,
        expiries_y=expiries,
        strikes=np.ones_like(expiries),
    )
    volatilities = np.asarray(
        bs.implied(
            k=np.ones_like(expiries),
            prices=prices,
            sqrtT=np.sqrt(expiries),
            eps=1.0e-10,
        ),
        dtype=np.float64,
    )
    if np.any(~np.isfinite(volatilities)) or np.any(volatilities <= 0.0):
        raise RuntimeError(f"{day.trading_date}: invalid public SANOS ATM volatility")
    return expiries * volatilities**2


def _latent_mixture(
    day: PublicSanosDay,
    expiry_y: float,
) -> list[tuple[float, FloatArray, FloatArray]]:
    upper = int(np.searchsorted(day.model_expiries_y, expiry_y, side="left"))
    if upper == 0:
        return [(1.0, day.support_strikes[0], day.densities[0])]
    if upper == len(day.model_expiries_y):
        return [(1.0, day.support_strikes[-1], day.densities[-1])]
    lower = upper - 1
    lower_distance = abs(expiry_y - float(day.model_expiries_y[lower]))
    upper_distance = abs(float(day.model_expiries_y[upper]) - expiry_y)
    if min(lower_distance, upper_distance) < day.one_minute_y:
        index = upper if upper_distance < lower_distance else lower
        return [(1.0, day.support_strikes[index], day.densities[index])]
    upper_weight = (expiry_y - float(day.model_expiries_y[lower])) / float(
        day.model_expiries_y[upper] - day.model_expiries_y[lower]
    )
    return [
        (1.0 - upper_weight, day.support_strikes[lower], day.densities[lower]),
        (upper_weight, day.support_strikes[upper], day.densities[upper]),
    ]


def _source_kernel_volatility(day: PublicSanosDay, expiry_y: float) -> float:
    clipped_expiry = float(
        np.clip(expiry_y, day.model_expiries_y[0], day.model_expiries_y[-1])
    )
    variance = float(
        np.interp(
            clipped_expiry,
            day.model_expiries_y,
            day.kernel_total_variances,
        )
    )
    return math.sqrt(variance / clipped_expiry) * math.sqrt(day.smoothness)


def _project_points_to_support(
    values: FloatArray,
    masses: FloatArray,
    support: FloatArray,
) -> tuple[FloatArray, float]:
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, support[-1])
    clipped = (values < support[0] - tolerance) | (values > support[-1] + tolerance)
    clipped_mass = float(np.sum(masses[clipped]))
    values = np.clip(values, support[0], support[-1])
    lower = np.searchsorted(support, values, side="right") - 1
    lower = np.clip(lower, 0, len(support) - 2)
    upper = lower + 1
    upper_weight = (values - support[lower]) / (support[upper] - support[lower])
    exact_last = values == support[-1]
    lower[exact_last] = len(support) - 2
    upper[exact_last] = len(support) - 1
    upper_weight[exact_last] = 1.0
    density = np.zeros_like(support)
    np.add.at(density, lower, masses * (1.0 - upper_weight))
    np.add.at(density, upper, masses * upper_weight)
    return density, clipped_mass


def _enforce_unit_mass_and_mean(density: FloatArray, support: FloatArray) -> FloatArray:
    repaired = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    repaired /= np.sum(repaired)
    mean_error = float(repaired @ support - 1.0)
    tolerance = 4.0e-14
    if abs(mean_error) <= tolerance:
        return repaired
    receiver = 0 if mean_error > 0.0 else len(support) - 1
    donor_order = (
        range(len(support) - 1, 0, -1)
        if mean_error > 0.0
        else range(0, len(support) - 1)
    )
    remaining = abs(mean_error)
    for donor in donor_order:
        distance = abs(float(support[donor] - support[receiver]))
        if distance == 0.0 or repaired[donor] == 0.0:
            continue
        transfer = min(float(repaired[donor]), remaining / distance)
        repaired[donor] -= transfer
        repaired[receiver] += transfer
        remaining -= transfer * distance
        if remaining <= tolerance:
            break
    if remaining > 2.0e-13:
        raise RuntimeError(f"Unable to repair density first moment; residual={remaining:.3g}")
    return repaired / np.sum(repaired)


def project_public_sanos_density(
    day: PublicSanosDay,
    *,
    expiry_y: float,
    support: FloatArray,
    backbone_annual_volatility: float,
    quadrature_order: int = 20,
) -> tuple[FloatArray, float]:
    """Factor the public SANOS law into the fixed 5% Black backbone."""
    source_kernel_volatility = _source_kernel_volatility(day, expiry_y)
    residual_annual_variance = max(
        source_kernel_volatility**2 - backbone_annual_volatility**2,
        0.0,
    )
    residual_total_variance = residual_annual_variance * expiry_y
    nodes, raw_weights = hermgauss(quadrature_order)
    weights = raw_weights / math.sqrt(math.pi)
    if residual_total_variance == 0.0:
        multipliers = np.ones(1, dtype=np.float64)
        weights = np.ones(1, dtype=np.float64)
    else:
        multipliers = np.exp(
            -0.5 * residual_total_variance
            + math.sqrt(2.0 * residual_total_variance) * nodes
        )
    density = np.zeros_like(support)
    clipped_mass = 0.0
    for mixture_weight, latent_support, latent_density in _latent_mixture(day, expiry_y):
        source_mass = mixture_weight * latent_density
        active = source_mass > 1.0e-16
        values = (latent_support[active, None] * multipliers[None, :]).reshape(-1)
        masses = (source_mass[active, None] * weights[None, :]).reshape(-1)
        projected, clipped = _project_points_to_support(values, masses, support)
        density += projected
        clipped_mass += clipped
    return _enforce_unit_mass_and_mean(density, support), clipped_mass


def solve_sanos_step_numpy(
    support: FloatArray,
    previous: FloatArray,
    dlvsq_dt: FloatArray,
) -> FloatArray:
    """Solve one tridiagonal DLV step in NumPy/SciPy."""
    box = support[1:-1] ** 2 / (support[2:] - support[:-2])
    w_plus = dlvsq_dt * box / (support[2:] - support[1:-1])
    w_minus = dlvsq_dt * box / (support[1:-1] - support[:-2])
    bands = np.zeros((3, len(support)), dtype=np.float64)
    bands[1] = np.concatenate(([1.0], 1.0 + w_minus + w_plus, [1.0]))
    bands[0, 1:] = np.concatenate((-w_minus, [0.0]))
    bands[2, :-1] = np.concatenate(([0.0], -w_plus))
    return solve_banded((1, 1), bands, previous, check_finite=False)


def _intrinsic_calls_from_density(support: FloatArray, density: FloatArray) -> FloatArray:
    return np.maximum(support[:, None] - support[None, :], 0.0).T @ density


def _black_call_kernel(
    support: FloatArray,
    strikes: FloatArray,
    *,
    annual_volatility: float,
    expiry_y: float,
) -> FloatArray:
    standard_deviation = annual_volatility * math.sqrt(expiry_y)
    if standard_deviation == 0.0:
        return np.maximum(support[None, :] - strikes[:, None], 0.0)
    log_ratio = np.log(support[None, :] / strikes[:, None])
    d1 = log_ratio / standard_deviation + 0.5 * standard_deviation
    d2 = d1 - standard_deviation
    return support[None, :] * ndtr(d1) - strikes[:, None] * ndtr(d2)


def _fit_dlv_interval(
    *,
    support: FloatArray,
    previous_density: FloatArray,
    target_calls: FloatArray,
    interval_years: float,
    nms_nodes: FloatArray,
    fit_nms_range: tuple[float, float],
    minimum_dlvsq: float,
    maximum_dlvsq: float,
    iterations: int = 40,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    previous_calls = _intrinsic_calls_from_density(support, previous_density)
    envelope_calls = np.maximum(previous_calls, target_calls)
    envelope_calls[0] = 1.0 - support[0]
    envelope_calls[-1] = 0.0
    call_increment = np.maximum(envelope_calls[1:-1] - previous_calls[1:-1], 0.0)
    cell_width = support[2:] - support[:-2]
    fit_mask = (nms_nodes >= fit_nms_range[0]) & (nms_nodes <= fit_nms_range[1])
    first_fit = int(np.flatnonzero(fit_mask)[0])
    last_fit = int(np.flatnonzero(fit_mask)[-1])
    dlvsq = np.full_like(call_increment, 0.20**2)
    density = solve_sanos_step_numpy(support, previous_density, dlvsq * interval_years)
    raw = dlvsq.copy()
    for _ in range(iterations):
        denominator = (
            interval_years
            * support[1:-1] ** 2
            * np.maximum(density[1:-1], 1.0e-18)
        )
        raw = cell_width * call_increment / denominator
        raw[:first_fit] = raw[first_fit]
        raw[last_fit + 1 :] = raw[last_fit]
        target_dlvsq = np.clip(raw, minimum_dlvsq, maximum_dlvsq)
        dlvsq = np.exp(0.25 * np.log(dlvsq) + 0.75 * np.log(target_dlvsq))
        density = solve_sanos_step_numpy(
            support, previous_density, dlvsq * interval_years
        )
    return dlvsq, density, raw


def reconstruct_public_day(
    day: PublicSanosDay,
    *,
    supports: FloatArray,
    model_expiries_y: FloatArray,
    geometry: Gromm2GeometryConfig,
    minimum_dlvsq: float = 0.01**2,
    maximum_dlvsq: float = 3.0**2,
) -> ReconstructedDay:
    """Construct a six-month positive DLV tower from one public SANOS state."""
    expiries = np.asarray(model_expiries_y, dtype=np.float64)
    raw_total_variances = public_sanos_atm_total_variances(day, expiries)
    projected_variances = project_total_variance(
        expiries,
        raw_total_variances,
        forward_volatility_floor=geometry.forward_volatility_floor,
    )
    target_densities = np.empty_like(supports)
    clipped_mass = 0.0
    for index, expiry_y in enumerate(expiries):
        target_densities[index], clipped = project_public_sanos_density(
            day,
            expiry_y=float(expiry_y),
            support=supports[index],
            backbone_annual_volatility=geometry.backbone_annual_volatility,
        )
        clipped_mass = max(clipped_mass, clipped)

    transports = transport_tower(supports)
    intervals = np.diff(expiries, prepend=0.0)
    dlvsq = np.empty((len(expiries), supports.shape[1] - 2), dtype=np.float64)
    model_densities = np.empty_like(supports)
    raw_floored = 0
    raw_capped = 0
    previous = np.asarray(density0_from_support(np.asarray(supports[0])))
    for index in range(len(expiries)):
        if index > 0:
            previous = transports[index - 1] @ model_densities[index - 1]
        target_calls = _intrinsic_calls_from_density(
            supports[index], target_densities[index]
        )
        current_dlvsq, current_density, raw = _fit_dlv_interval(
            support=supports[index],
            previous_density=previous,
            target_calls=target_calls,
            interval_years=float(intervals[index]),
            nms_nodes=geometry.nms_nodes,
            fit_nms_range=geometry.fit_nms_range,
            minimum_dlvsq=minimum_dlvsq,
            maximum_dlvsq=maximum_dlvsq,
        )
        raw_floored += int(np.sum(raw < minimum_dlvsq))
        raw_capped += int(np.sum(raw > maximum_dlvsq))
        dlvsq[index] = current_dlvsq
        model_densities[index] = current_density

    fit_strikes = np.exp(
        np.sqrt(projected_variances)[:, None]
        * np.linspace(geometry.fit_nms_range[0], geometry.fit_nms_range[1], 33)[
            None, :
        ]
    )
    target_prices = public_sanos_prices(
        day,
        expiries_y=np.broadcast_to(expiries[:, None], fit_strikes.shape),
        strikes=fit_strikes,
    )
    model_prices = np.stack(
        [
            _black_call_kernel(
                supports[index],
                fit_strikes[index],
                annual_volatility=geometry.backbone_annual_volatility,
                expiry_y=float(expiries[index]),
            )
            @ model_densities[index]
            for index in range(len(expiries))
        ]
    )
    price_errors = model_prices - target_prices
    density_errors = model_densities - target_densities
    coordinate_count = dlvsq.size
    return ReconstructedDay(
        trading_date=day.trading_date,
        raw_atm_total_variances=raw_total_variances,
        projected_atm_total_variances=projected_variances,
        target_densities=target_densities,
        model_densities=model_densities,
        dlvsq=dlvsq,
        floored_raw_dlvsq_fraction=raw_floored / coordinate_count,
        capped_raw_dlvsq_fraction=raw_capped / coordinate_count,
        clipped_projection_mass=clipped_mass,
        maximum_density_error=float(np.max(np.abs(density_errors))),
        root_mean_square_price_error=float(np.sqrt(np.mean(price_errors**2))),
        maximum_price_error=float(np.max(np.abs(price_errors))),
    )
