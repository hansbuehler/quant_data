"""Standalone reconstruction of published SANOS and DLV model prices."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from cdxcore.bs import bs
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _as_finite_vector(value: NDArray[np.floating], *, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a non-empty finite vector, found {vector.shape}")
    return vector


def _call_to_side_price(
    calls: FloatArray,
    strikes: FloatArray,
    is_call: NDArray[np.bool_],
) -> FloatArray:
    sides = np.asarray(is_call, dtype=bool)
    if sides.shape != calls.shape:
        raise ValueError(f"is_call shape {sides.shape} does not match prices {calls.shape}")
    return np.where(sides, calls, calls - 1.0 + strikes)


def _black_kernel(
    latent_spots: FloatArray,
    strikes: FloatArray,
    *,
    annual_volatility: float,
    expiry_y: float,
) -> FloatArray:
    if expiry_y <= 0.0:
        return np.maximum(latent_spots[None, :] - strikes[:, None], 0.0)
    relative_strikes = strikes[:, None] / latent_spots[None, :]
    return np.asarray(
        bs.price(
            k=relative_strikes,
            vol=annual_volatility,
            sqrtT=math.sqrt(expiry_y),
            eps=1.0e-10,
        ),
        dtype=np.float64,
    ) * latent_spots[None, :]


def price_sanos_surface(
    *,
    query_expiries_y: NDArray[np.floating],
    query_strikes: NDArray[np.floating],
    is_call: NDArray[np.bool_],
    model_expiries_y: NDArray[np.floating],
    atm_variances: NDArray[np.floating],
    support_strikes: Sequence[NDArray[np.floating]],
    densities: Sequence[NDArray[np.floating]],
    smoothness: float,
    one_minute_y: float,
) -> FloatArray:
    """Price normalized options from the public SANOS surface representation."""
    expiries = _as_finite_vector(query_expiries_y, name="query_expiries_y")
    strikes = _as_finite_vector(query_strikes, name="query_strikes")
    model_expiries = _as_finite_vector(model_expiries_y, name="model_expiries_y")
    variances = _as_finite_vector(atm_variances, name="atm_variances")
    if expiries.shape != strikes.shape:
        raise ValueError(f"expiry shape {expiries.shape} does not match strikes {strikes.shape}")
    if variances.shape != model_expiries.shape:
        raise ValueError(
            f"variance shape {variances.shape} does not match expiries {model_expiries.shape}"
        )
    if len(support_strikes) != len(model_expiries) or len(densities) != len(model_expiries):
        raise ValueError("SANOS support and density lists must match model expiries")
    if np.any(expiries < 0.0) or np.any(strikes <= 0.0):
        raise ValueError("SANOS query expiries must be non-negative and strikes positive")
    if np.any(np.diff(model_expiries) <= 0.0) or np.any(variances <= 0.0):
        raise ValueError("SANOS expiries and variances must be strictly positive")
    if smoothness < 0.0 or one_minute_y <= 0.0:
        raise ValueError("SANOS smoothness must be non-negative and one_minute_y positive")

    model_supports: list[FloatArray] = []
    model_densities: list[FloatArray] = []
    for index, (support_value, density_value) in enumerate(
        zip(support_strikes, densities, strict=True)
    ):
        support = _as_finite_vector(np.asarray(support_value), name=f"support[{index}]")
        density = _as_finite_vector(np.asarray(density_value), name=f"density[{index}]")
        if support.shape != density.shape or np.any(np.diff(support) <= 0.0):
            raise ValueError(f"Invalid SANOS support-density geometry at expiry {index}")
        if np.min(density) < -1.0e-10:
            raise ValueError(f"Negative SANOS density at expiry {index}")
        if not np.isclose(np.sum(density), 1.0, rtol=0.0, atol=2.0e-6):
            raise ValueError(f"SANOS density at expiry {index} does not have unit mass")
        if not np.isclose(density @ support, 1.0, rtol=0.0, atol=2.0e-6):
            raise ValueError(f"SANOS density at expiry {index} does not integrate to forward one")
        model_supports.append(support)
        model_densities.append(density)

    def node_price(node_index: int, expiry_y: float, query: FloatArray) -> FloatArray:
        if expiry_y < one_minute_y or smoothness == 0.0:
            kernel = np.maximum(
                model_supports[node_index][None, :] - query[:, None],
                0.0,
            )
        else:
            clipped_expiry = float(
                np.clip(expiry_y, model_expiries[0], model_expiries[-1])
            )
            variance = float(np.interp(clipped_expiry, model_expiries, variances))
            # Match ArbFreePricer.interpolate_atm_vols exactly: outside the
            # node range it holds the clipped-node annualized kernel volatility
            # fixed, so kernel total variance continues to grow with query time.
            annual_volatility = math.sqrt(variance / clipped_expiry) * math.sqrt(smoothness)
            kernel = _black_kernel(
                model_supports[node_index],
                query,
                annual_volatility=annual_volatility,
                expiry_y=expiry_y,
            )
        return kernel @ model_densities[node_index]

    calls = np.empty_like(strikes)
    for expiry_y in np.unique(expiries):
        selection = expiries == expiry_y
        query = strikes[selection]
        if expiry_y < one_minute_y:
            calls[selection] = np.maximum(1.0 - query, 0.0)
            continue
        upper = int(np.searchsorted(model_expiries, expiry_y, side="left"))
        if upper == 0:
            calls[selection] = node_price(0, float(expiry_y), query)
            continue
        if upper == len(model_expiries):
            calls[selection] = node_price(-1, float(expiry_y), query)
            continue
        lower = upper - 1
        lower_distance = abs(float(expiry_y) - float(model_expiries[lower]))
        upper_distance = abs(float(model_expiries[upper]) - float(expiry_y))
        if min(lower_distance, upper_distance) < one_minute_y:
            node_index = upper if upper_distance < lower_distance else lower
            calls[selection] = node_price(node_index, float(expiry_y), query)
            continue
        lower_prices = node_price(lower, float(expiry_y), query)
        upper_prices = node_price(upper, float(expiry_y), query)
        interval = float(model_expiries[upper] - model_expiries[lower])
        upper_weight = float(expiry_y - model_expiries[lower]) / interval
        calls[selection] = (1.0 - upper_weight) * lower_prices + upper_weight * upper_prices
    return _call_to_side_price(calls, strikes, is_call)


def _density0_from_support(support: FloatArray) -> FloatArray:
    calls = np.maximum(1.0 - support, 0.0)
    slopes = np.diff(calls, prepend=1.0) / np.diff(support, prepend=0.0)
    density = np.diff(slopes, append=0.0)
    if np.min(density) < -1.0e-10:
        raise ValueError("DLV initial density is negative")
    if not np.isclose(np.sum(density), 1.0, rtol=0.0, atol=2.0e-12):
        raise ValueError("DLV initial density does not have unit mass")
    if not np.isclose(density @ support, 1.0, rtol=0.0, atol=2.0e-12):
        raise ValueError("DLV initial density does not integrate to forward one")
    return density


def _solve_dlv_step(
    support: FloatArray,
    previous_density: FloatArray,
    dlvsq_dt: FloatArray,
) -> FloatArray:
    box = support[1:-1] ** 2 / (support[2:] - support[:-2])
    weight_plus = dlvsq_dt * box / (support[2:] - support[1:-1])
    weight_minus = dlvsq_dt * box / (support[1:-1] - support[:-2])
    lower = np.concatenate((np.zeros(2), -weight_plus))
    diagonal = np.concatenate(
        ((1.0,), 1.0 + weight_minus + weight_plus, (1.0,))
    )
    upper = np.concatenate((-weight_minus, np.zeros(2)))
    size = len(previous_density)
    modified_upper = np.empty(size, dtype=np.float64)
    modified_rhs = np.empty(size, dtype=np.float64)
    modified_upper[0] = upper[0] / diagonal[0]
    modified_rhs[0] = previous_density[0] / diagonal[0]
    for index in range(1, size):
        denominator = diagonal[index] - lower[index] * modified_upper[index - 1]
        modified_upper[index] = upper[index] / denominator
        modified_rhs[index] = (
            previous_density[index] - lower[index] * modified_rhs[index - 1]
        ) / denominator
    result = np.empty(size, dtype=np.float64)
    result[-1] = modified_rhs[-1]
    for index in range(size - 2, -1, -1):
        result[index] = modified_rhs[index] - modified_upper[index] * result[index + 1]
    return result


def decode_dlv_density_tower(
    *,
    support_strikes: NDArray[np.floating],
    raw_log_excess_dlvsq: NDArray[np.floating],
    model_expiries_y: NDArray[np.floating],
    minimum_dlvsq: float,
    maximum_dlvsq: float,
) -> FloatArray:
    """Decode a published raw DLV state into its martingale density tower."""
    support = _as_finite_vector(support_strikes, name="support_strikes")
    expiries = _as_finite_vector(model_expiries_y, name="model_expiries_y")
    raw = np.asarray(raw_log_excess_dlvsq, dtype=np.float64)
    if raw.shape != (len(expiries), len(support) - 2):
        raise ValueError(
            f"raw DLV shape {raw.shape} does not match {(len(expiries), len(support) - 2)}"
        )
    if not np.all(np.isfinite(raw)) or np.any(np.diff(support) <= 0.0):
        raise ValueError("DLV state and support must be finite with increasing support")
    if not 0.0 < minimum_dlvsq < maximum_dlvsq:
        raise ValueError("DLV bounds must satisfy 0 < minimum < maximum")
    if np.any(np.diff(expiries) <= 0.0):
        raise ValueError("DLV model expiries must be strictly increasing")

    maximum_raw = math.log(maximum_dlvsq - minimum_dlvsq)
    dlvsq = minimum_dlvsq + np.exp(np.minimum(raw, maximum_raw))
    density = _density0_from_support(support)
    tower = np.empty((len(expiries), len(support)), dtype=np.float64)
    previous_expiry = 0.0
    for index, expiry_y in enumerate(expiries):
        density = _solve_dlv_step(
            support,
            density,
            dlvsq[index] * (float(expiry_y) - previous_expiry),
        )
        if np.min(density) < -1.0e-10:
            raise FloatingPointError(f"Negative DLV density at expiry {index}")
        if not np.isclose(np.sum(density), 1.0, rtol=0.0, atol=2.0e-10):
            raise FloatingPointError(f"DLV density at expiry {index} does not have unit mass")
        if not np.isclose(density @ support, 1.0, rtol=0.0, atol=2.0e-10):
            raise FloatingPointError(
                f"DLV density at expiry {index} does not integrate to forward one"
            )
        tower[index] = density
        previous_expiry = float(expiry_y)
    return tower


def price_dlv_surface(
    *,
    query_expiries_y: NDArray[np.floating],
    query_strikes: NDArray[np.floating],
    is_call: NDArray[np.bool_],
    support_strikes: NDArray[np.floating],
    raw_log_excess_dlvsq: NDArray[np.floating],
    model_expiries_y: NDArray[np.floating],
    minimum_dlvsq: float,
    maximum_dlvsq: float,
    backbone_annual_volatility: float,
) -> FloatArray:
    """Price normalized options from the public discrete-DLV representation."""
    query_expiries = _as_finite_vector(query_expiries_y, name="query_expiries_y")
    query_strikes_array = _as_finite_vector(query_strikes, name="query_strikes")
    support = _as_finite_vector(support_strikes, name="support_strikes")
    model_expiries = _as_finite_vector(model_expiries_y, name="model_expiries_y")
    if query_expiries.shape != query_strikes_array.shape:
        raise ValueError("DLV query expiries and strikes must have equal shape")
    if np.any(query_expiries <= 0.0) or np.any(query_strikes_array <= 0.0):
        raise ValueError("DLV query expiries and strikes must be positive")
    if backbone_annual_volatility < 0.0:
        raise ValueError("DLV backbone volatility must be non-negative")
    tower = decode_dlv_density_tower(
        support_strikes=support,
        raw_log_excess_dlvsq=raw_log_excess_dlvsq,
        model_expiries_y=model_expiries,
        minimum_dlvsq=minimum_dlvsq,
        maximum_dlvsq=maximum_dlvsq,
    )
    density0 = _density0_from_support(support)
    calls = np.empty_like(query_strikes_array)
    for expiry_y in np.unique(query_expiries):
        selection = query_expiries == expiry_y
        strikes = query_strikes_array[selection]
        raw_upper = int(np.searchsorted(model_expiries, expiry_y, side="left"))
        if raw_upper >= len(model_expiries):
            if expiry_y > model_expiries[-1] + 1.0e-12:
                raise ValueError(f"DLV query expiry {expiry_y} exceeds the model horizon")
            raw_upper = len(model_expiries) - 1
        upper = raw_upper
        before_first = upper == 0
        lower = max(upper - 1, 0)
        previous_expiry = 0.0 if before_first else float(model_expiries[lower])
        interval = float(model_expiries[upper]) - previous_expiry
        alpha = float(np.clip((float(expiry_y) - previous_expiry) / interval, 0.0, 1.0))
        lower_density = density0 if before_first else tower[lower]
        upper_density = tower[upper]
        background_time = (
            (1.0 - alpha) * float(model_expiries[lower])
            + alpha * float(model_expiries[upper])
        )
        kernel = _black_kernel(
            support,
            strikes,
            annual_volatility=backbone_annual_volatility,
            expiry_y=background_time,
        )
        calls[selection] = (
            (1.0 - alpha) * (kernel @ lower_density)
            + alpha * (kernel @ upper_density)
        )
    return _call_to_side_price(calls, query_strikes_array, is_call)


def implied_volatilities(
    *,
    expiries_y: NDArray[np.floating],
    normalized_strikes: NDArray[np.floating],
    normalized_prices: NDArray[np.floating],
    is_call: NDArray[np.bool_],
) -> FloatArray:
    """Compute annualized Black implied volatility with the repository solver."""
    expiries = _as_finite_vector(expiries_y, name="expiries_y")
    strikes = _as_finite_vector(normalized_strikes, name="normalized_strikes")
    prices = _as_finite_vector(normalized_prices, name="normalized_prices")
    if expiries.shape != strikes.shape or prices.shape != strikes.shape:
        raise ValueError("Expiry, strike, and price arrays must have equal shape")
    result, _ = implied_volatilities_with_validity(
        expiries_y=expiries,
        normalized_strikes=strikes,
        normalized_prices=prices,
        is_call=is_call,
    )
    return result


def implied_volatilities_with_validity(
    *,
    expiries_y: NDArray[np.floating],
    normalized_strikes: NDArray[np.floating],
    normalized_prices: NDArray[np.floating],
    is_call: NDArray[np.bool_],
) -> tuple[FloatArray, BoolArray]:
    """Return annualized Black volatility and a numerical-interiority mask."""
    expiries = _as_finite_vector(expiries_y, name="expiries_y")
    strikes = _as_finite_vector(normalized_strikes, name="normalized_strikes")
    prices = _as_finite_vector(normalized_prices, name="normalized_prices")
    if expiries.shape != strikes.shape or prices.shape != strikes.shape:
        raise ValueError("Expiry, strike, and price arrays must have equal shape")
    sides = np.asarray(is_call, dtype=bool)
    if sides.shape != prices.shape:
        raise ValueError("Option sides must align with prices")
    intrinsic = np.where(
        sides,
        np.maximum(1.0 - strikes, 0.0),
        np.maximum(strikes - 1.0, 0.0),
    )
    upper_bound = np.where(sides, 1.0, strikes)
    tolerance = 1.0e-8
    bounded_prices = np.clip(prices, intrinsic, upper_bound)
    diagnostic = bs.implied(
        k=strikes,
        prices=bounded_prices,
        is_call=sides,
        sqrtT=np.sqrt(expiries),
        price_tol=tolerance,
        eps=1.0e-10,
        on_exceed_bounds="quiet",
        ret_only_vols=False,
    )
    result = np.asarray(diagnostic.vols, dtype=np.float64)
    valid = (
        ~np.asarray(diagnostic.failed, dtype=bool)
        & (prices >= intrinsic - tolerance)
        & (prices <= upper_bound + tolerance)
        & (bounded_prices > intrinsic + tolerance)
        & (bounded_prices < upper_bound - tolerance)
    )
    if result.shape != prices.shape or not np.all(np.isfinite(result) & (result > 0.0)):
        raise FloatingPointError("Black implied-volatility calculation failed")
    return result, valid
