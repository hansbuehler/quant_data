"""Fixed-state spot and rolling SANOS-state factor price exposures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quantdata.hedging.data.rolling_sanos_factors import RollingSanosFactors

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class SanosStateGreeks:
    """Price changes per unit spot return or one-standard-deviation state shock."""

    prices: FloatArray
    fixed_state_spot: FloatArray
    factor_exposures: FloatArray
    spot_correlated_state: FloatArray

    @property
    def orthogonalized_spot_exposure(self) -> FloatArray:
        """Joint spot shock exposure, distinct from the fixed-state spot partial."""
        return self.fixed_state_spot + self.spot_correlated_state


def finite_difference_sanos_state_greeks(
    *,
    price: Callable[[float, FloatArray], FloatArray],
    spot: float,
    raw_state: NDArray[np.floating],
    factors: RollingSanosFactors,
    relative_spot_step: float = 0.01,
) -> SanosStateGreeks:
    """Reprice fixed contracts after separate spot and state shocks."""
    state = np.asarray(raw_state, dtype=np.float64)
    if (
        not np.isfinite(spot)
        or spot <= 0.0
        or not np.isfinite(relative_spot_step)
        or not 0.0 < relative_spot_step < 1.0
        or state.ndim != 1
        or not np.all(np.isfinite(state))
        or factors.state_loadings.shape[0] != len(state)
    ):
        raise ValueError("Invalid SANOS Greek state, spot, or bump")

    def checked_price(shifted_spot: float, shifted_state: FloatArray) -> FloatArray:
        values = np.asarray(price(shifted_spot, shifted_state), dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("SANOS pricer must return a finite price vector")
        return values

    prices = checked_price(spot, state)
    spot_up = checked_price(spot * (1.0 + relative_spot_step), state)
    spot_down = checked_price(spot * (1.0 - relative_spot_step), state)
    if spot_up.shape != prices.shape or spot_down.shape != prices.shape:
        raise ValueError("SANOS spot-bump price vectors must align")
    fixed_state_spot = (spot_up - spot_down) / (2.0 * relative_spot_step)

    factor_exposures = np.empty((len(prices), factors.rank), dtype=np.float64)
    for factor_index in range(factors.rank):
        loading = factors.state_loadings[:, factor_index]
        price_up = checked_price(spot, state + loading)
        price_down = checked_price(spot, state - loading)
        if price_up.shape != prices.shape or price_down.shape != prices.shape:
            raise ValueError("SANOS factor-bump price vectors must align")
        factor_exposures[:, factor_index] = (price_up - price_down) / 2.0

    correlated_state = np.zeros_like(prices)
    if factors.method == "spot_orthogonal":
        loading = relative_spot_step * factors.spot_correlated_state_loading
        price_up = checked_price(spot, state + loading)
        price_down = checked_price(spot, state - loading)
        if price_up.shape != prices.shape or price_down.shape != prices.shape:
            raise ValueError("SANOS spot-correlated-state prices must align")
        correlated_state = (price_up - price_down) / (2.0 * relative_spot_step)

    return SanosStateGreeks(
        prices=prices,
        fixed_state_spot=fixed_state_spot,
        factor_exposures=factor_exposures,
        spot_correlated_state=correlated_state,
    )
