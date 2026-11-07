"""Black digital delta/vega hedges in bracketing Europeans and free spot."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedge2.cash import CashCurve, zero_strike_mask
from quantdata.hedge2.claims import Claim


def black_digital_exposures(*, claim: Claim, curve: CashCurve, target_vol: float,
                            elapsed_day: int = 0) -> tuple[float, float]:
    """Cash-payout spot delta and vega per unit annual volatility, at fixed carry."""
    remaining = claim.terminal_day - elapsed_day
    if claim.kind != "digital" or remaining <= 0 or not np.isfinite(target_vol) or target_vol <= 0:
        raise ValueError("Black digital exposures require a live digital and positive finite volatility")
    years = remaining / DAYS_PER_YEAR
    forward, _, discount = curve.at(float(remaining))
    total_vol = target_vol * np.sqrt(years)
    log_ratio = np.log(forward / claim.strike)
    d2 = log_ratio / total_vol - .5 * total_vol
    density = np.exp(-.5 * d2**2) / np.sqrt(2. * np.pi)
    sign = 1. if claim.is_call else -1.
    delta = sign * discount * density / (curve.spot * total_vol)
    vega = sign * discount * density * (-log_ratio / total_vol**2 - .5) * np.sqrt(years)
    return float(delta), float(vega)


def black_digital_hedge(*, claim: Claim, curve: CashCurve, book: pd.DataFrame,
                        target_vol: float, option_vols: np.ndarray, target_side: int,
                        elapsed_day: int = 0) -> np.ndarray:
    """Weight the two cash-strike neighbors by distance, size by vega, offset all delta."""
    remaining = claim.terminal_day - elapsed_day
    if (claim.kind != "digital" or remaining <= 0 or target_side not in (-1, 1)
            or option_vols.shape != (len(book),) or not np.isfinite(option_vols).all()
            or np.any(option_vols < 0) or not np.isfinite(target_vol) or target_vol <= 0):
        raise ValueError("Black digital hedge needs live digitals, positive vols and a long/short side")
    target_delta, target_vega = black_digital_exposures(claim=claim, curve=curve,
        target_vol=target_vol, elapsed_day=elapsed_day)
    positions = np.zeros(1 + len(book))
    ordinary = ~zero_strike_mask(book)
    available = np.flatnonzero(ordinary & (book["bdte"].to_numpy(dtype=float) >= remaining - 1e-6))
    if len(available) == 0:
        raise ValueError("The Black vega hedge has no expiry at or after the target")
    tenor = float(book.iloc[available]["bdte"].min())
    indices = np.flatnonzero(ordinary & np.isclose(book["bdte"].to_numpy(dtype=float), tenor, atol=1e-6, rtol=0.))
    strikes = book.iloc[indices]["strike"].to_numpy(dtype=float)
    lower = np.flatnonzero(strikes <= claim.strike)
    upper = np.flatnonzero(strikes >= claim.strike)
    if len(lower) == 0 or len(upper) == 0:
        raise ValueError("The Black vega hedge needs two surrounding observed option strikes")
    left, right = int(lower[-1]), int(upper[0])
    if left == right:
        selected = np.array([indices[left]])
        weights = np.array([1.])
    else:
        selected = indices[[left, right]]
        weight_left = (strikes[right] - claim.strike) / (strikes[right] - strikes[left])
        weights = np.array([weight_left, 1. - weight_left])
    if np.any(option_vols[selected] <= 0.):
        raise ValueError("The bracketing option combination has zero Black vega")
    hedge_forward, _, hedge_discount = curve.at(tenor)
    hedge_years = tenor / DAYS_PER_YEAR
    hedge_total_vol = option_vols[selected] * np.sqrt(hedge_years)
    d1 = np.log(hedge_forward / book.iloc[selected]["strike"].to_numpy(dtype=float)) / hedge_total_vol + .5 * hedge_total_vol
    hedge_density = np.exp(-.5 * d1**2) / np.sqrt(2. * np.pi)
    hedge_vega = hedge_discount * hedge_forward * hedge_density * np.sqrt(hedge_years) / claim.initial_spot
    effective_vega = float(weights @ hedge_vega)
    if effective_vega <= np.finfo(float).tiny:
        raise ValueError("The bracketing option combination has zero Black vega")
    positions[1 + selected] = -target_side * target_vega * weights / effective_vega
    calls = (book.iloc[selected]["option_type"] == "call").to_numpy()
    option_delta = hedge_discount * hedge_forward / curve.spot * (ndtr(d1) - (~calls).astype(float))
    positions[0] = (-target_side * target_delta * claim.initial_spot
                    - positions[1 + selected] @ option_delta)
    return positions
