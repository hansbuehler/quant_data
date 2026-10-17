"""Black Greeks and distance-weighted interpolation of executable vanillas."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.contracts import OptionQuote, OptionRight

FloatArray = NDArray[np.float64]
_NORMAL_DENSITY = 1.0 / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True, kw_only=True)
class BlackGreeks:
    """Origin-spot normalized value, spot delta, and annual-volatility vega."""

    price: float
    delta: float
    vega: float


def black_digital_greeks(
    *,
    strike_over_forward: float,
    annual_vol: float,
    business_dte: float,
    discount_factor: float,
    right: OptionRight,
) -> BlackGreeks:
    """Return fixed-cash Black digital Greeks at origin spot one."""
    _validate_black_inputs(strike_over_forward, annual_vol, business_dte, discount_factor)
    root_time = math.sqrt(business_dte / DAYS_PER_YEAR)
    total_vol = annual_vol * root_time
    d2 = -math.log(strike_over_forward) / total_vol - 0.5 * total_vol
    density = _NORMAL_DENSITY * math.exp(-0.5 * d2 * d2)
    sign = 1.0 if right is OptionRight.CALL else -1.0
    price = discount_factor * float(ndtr(sign * d2))
    delta = sign * discount_factor * density / total_vol
    vega = sign * discount_factor * density * (
        math.log(strike_over_forward) / (annual_vol * annual_vol * root_time)
        - 0.5 * root_time
    )
    return BlackGreeks(price=price, delta=delta, vega=vega)


def black_option_greeks(quote: OptionQuote, *, forward_over_spot: float, discount_factor: float) -> BlackGreeks:
    """Return Black Greeks of a quoted vanilla using its expiry ATM volatility."""
    _validate_black_inputs(
        quote.strike_over_forward,
        quote.atm_annual_vol,
        quote.business_dte,
        discount_factor,
    )
    if not math.isfinite(forward_over_spot) or forward_over_spot <= 0.0:
        raise ValueError("forward_over_spot must be finite-positive")
    root_time = math.sqrt(quote.business_dte / DAYS_PER_YEAR)
    total_vol = quote.atm_annual_vol * root_time
    log_strike = math.log(quote.strike_over_forward)
    d1 = -log_strike / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    bridge = discount_factor * forward_over_spot
    call_price = bridge * (
        float(ndtr(d1)) - quote.strike_over_forward * float(ndtr(d2))
    )
    is_call = quote.contract.right is OptionRight.CALL
    price = call_price if is_call else call_price + bridge * (quote.strike_over_forward - 1.0)
    delta = bridge * (float(ndtr(d1)) - (0.0 if is_call else 1.0))
    vega = bridge * root_time * _NORMAL_DENSITY * math.exp(-0.5 * d1 * d1)
    return BlackGreeks(price=price, delta=delta, vega=vega)


def strike_interpolation(
    quotes: tuple[OptionQuote, ...],
    *,
    fixed_strike: float,
    target_business_dte: float,
) -> tuple[tuple[int, float], ...]:
    """Return executable distance weights at the matching (or next) expiry."""
    if not quotes or not math.isfinite(fixed_strike) or fixed_strike <= 0.0:
        raise ValueError("quotes and fixed_strike must be non-empty and positive")
    if not math.isfinite(target_business_dte) or target_business_dte <= 0.0:
        raise ValueError("target_business_dte must be finite-positive")
    expiries = np.asarray([quote.business_dte for quote in quotes], dtype=np.float64)
    next_expiries = np.unique(expiries[expiries >= target_business_dte - 1.0e-8])
    for expiry in next_expiries:
        indices = np.flatnonzero(np.isclose(expiries, expiry, rtol=0.0, atol=1.0e-8))
        by_strike = sorted(
            ((float(quotes[index].contract.fixed_strike), int(index)) for index in indices),
            key=lambda item: item[0],
        )
        below = [item for item in by_strike if item[0] <= fixed_strike + 1.0e-12]
        above = [item for item in by_strike if item[0] >= fixed_strike - 1.0e-12]
        if not below or not above:
            continue
        lower_strike, lower_index = below[-1]
        upper_strike, upper_index = above[0]
        if math.isclose(lower_strike, upper_strike, rel_tol=0.0, abs_tol=1.0e-12):
            return ((lower_index, 1.0),)
        upper_weight = (fixed_strike - lower_strike) / (upper_strike - lower_strike)
        return ((lower_index, 1.0 - upper_weight), (upper_index, upper_weight))
    raise ValueError("target strike is outside every executable option bracket")


def interpolate_vega_hedge(
    *,
    target: BlackGreeks,
    quote_greeks: tuple[BlackGreeks, ...],
    bracket: tuple[tuple[int, float], ...],
    claim_quantity: float,
    position_cap: float,
    minimum_vega: float,
) -> tuple[float, FloatArray, float, float, int]:
    """Trade bracket weights in their strike-distance ratio, then hedge spot delta."""
    if not math.isfinite(claim_quantity) or claim_quantity <= 0.0:
        raise ValueError("claim_quantity must be finite-positive")
    interpolated_vega = sum(weight * quote_greeks[index].vega for index, weight in bracket)
    if abs(interpolated_vega) <= minimum_vega:
        raise FloatingPointError("strike-interpolated Black option vega is too small")
    option_actions = np.zeros(len(quote_greeks), dtype=np.float64)
    synthetic_action = -claim_quantity * target.vega / interpolated_vega
    for index, weight in bracket:
        option_actions[index] = np.clip(weight * synthetic_action, -position_cap, position_cap)
    spot_unclipped = -claim_quantity * target.delta - sum(
        option_actions[index] * greeks.delta
        for index, greeks in enumerate(quote_greeks)
    )
    spot_action = float(np.clip(spot_unclipped, -position_cap, position_cap))
    delta_residual = float(
        claim_quantity * target.delta
        + spot_action
        + sum(option_actions[index] * greeks.delta for index, greeks in enumerate(quote_greeks))
    )
    vega_residual = float(
        claim_quantity * target.vega
        + sum(option_actions[index] * greeks.vega for index, greeks in enumerate(quote_greeks))
    )
    actions = np.append(option_actions, spot_action)
    cap_hits = int(np.count_nonzero(np.abs(actions) >= position_cap - 1.0e-9))
    return spot_action, option_actions, delta_residual, vega_residual, cap_hits


def _validate_black_inputs(strike: float, vol: float, dte: float, discount: float) -> None:
    if any(not math.isfinite(value) or value <= 0.0 for value in (strike, vol, dte, discount)):
        raise ValueError("Black strike, volatility, DTE, and discount must be finite-positive")
