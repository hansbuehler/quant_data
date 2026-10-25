"""Fixed-grid SANOS states for public-data generative model training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import NDArray

from quantdata.biz_calendar import NMS
from quantdata.hedging.data.pricing import (
    implied_volatilities_with_validity,
    price_sanos_surface,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    PublicMarketDay,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@version("1.1")
@dataclass(frozen=True, kw_only=True)
class GenerativeSurfaceGrid:
    """Forward-strike or normalized-moneyness grid shared by generators."""

    strikes_over_forward: tuple[float, ...] = (
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        1.00,
        1.05,
        1.10,
        1.20,
        1.30,
        1.40,
    )
    nms_nodes: tuple[float, ...] | None = None
    expiries_y: tuple[float, ...] = (
        1.0 / 252.0,
        1.0 / 52.0,
        2.0 / 52.0,
        1.0 / 12.0,
        1.0 / 6.0,
        1.0 / 4.0,
        1.0 / 2.0,
        3.0 / 4.0,
        1.0,
    )

    def __post_init__(self) -> None:
        strikes = np.asarray(self.strikes_over_forward, dtype=np.float64)
        expiries = np.asarray(self.expiries_y, dtype=np.float64)
        verify_inp(
            (self.nms_nodes is not None or len(strikes) >= 3)
            and np.all(np.isfinite(strikes) & (strikes > 0.0))
            and np.all(np.diff(strikes) > 0.0),
            lambda: "strikes_over_forward must be finite-positive and increasing",
        )
        if self.nms_nodes is not None:
            nodes = np.asarray(self.nms_nodes, dtype=np.float64)
            verify_inp(
                len(nodes) >= 3
                and np.all(np.isfinite(nodes))
                and np.all(np.diff(nodes) > 0.0)
                and np.any(nodes == 0.0),
                lambda: "nms_nodes must be increasing, finite, and contain ATM",
            )
        verify_inp(
            len(expiries) >= 2
            and np.all(np.isfinite(expiries) & (expiries > 0.0))
            and np.all(np.diff(expiries) > 0.0),
            lambda: "expiries_y must be finite-positive and increasing",
        )

    @property
    def strike_count(self) -> int:
        """Number of coordinates along either strike representation."""
        return len(self.nms_nodes) if self.nms_nodes is not None else len(self.strikes_over_forward)

    @classmethod
    def short_end_nms(cls) -> GenerativeSurfaceGrid:
        """One-to-ten-session competitor grid with ATM-dense NMS nodes."""
        return cls(
            strikes_over_forward=(),
            nms_nodes=(
                -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
                0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
            ),
            expiries_y=tuple(days / 252.0 for days in (1, 2, 3, 4, 5, 7, 10)),
        )


@dataclass(frozen=True, kw_only=True)
class SanosFixedGridDay:
    """One public SANOS surface represented on a fixed model-training grid."""

    trading_date: date
    log_implied_volatilities: FloatArray
    implied_vol_is_interior: BoolArray

    def __post_init__(self) -> None:
        if self.log_implied_volatilities.ndim != 2:
            raise ValueError("fixed-grid log implied volatility must be a matrix")
        if self.implied_vol_is_interior.shape != self.log_implied_volatilities.shape:
            raise ValueError("fixed-grid implied-volatility mask does not align")
        if not np.all(np.isfinite(self.log_implied_volatilities)):
            raise ValueError("fixed-grid log implied volatility must be finite")

    @property
    def implied_volatilities(self) -> FloatArray:
        """Return annualized implied volatilities."""
        return np.exp(self.log_implied_volatilities)


@dataclass(frozen=True, kw_only=True)
class SanosFixedGridHistory:
    """Dated fixed-grid SANOS history with public normalized spot returns."""

    dates: tuple[date, ...]
    log_implied_volatilities: FloatArray
    implied_vol_is_interior: BoolArray
    forward_log_returns_into_date: FloatArray
    sessions_from_previous_observation: IntArray

    def __post_init__(self) -> None:
        observation_count = len(self.dates)
        if self.log_implied_volatilities.ndim != 3:
            raise ValueError("surface history must have date-expiry-strike axes")
        if self.log_implied_volatilities.shape[0] != observation_count:
            raise ValueError("surface-history dates do not align")
        if self.implied_vol_is_interior.shape != self.log_implied_volatilities.shape:
            raise ValueError("surface-history masks do not align")
        if self.forward_log_returns_into_date.shape != (observation_count,):
            raise ValueError("surface-history returns do not align")
        if self.sessions_from_previous_observation.shape != (observation_count,):
            raise ValueError("surface-history session counts do not align")
        if not np.all(np.isfinite(self.log_implied_volatilities)):
            raise ValueError("surface-history log implied volatilities must be finite")
        if not np.all(np.isfinite(self.forward_log_returns_into_date[1:])):
            raise ValueError("surface-history returns after the first date must be finite")
        if not np.isnan(self.forward_log_returns_into_date[0]):
            raise ValueError("the first surface-history return must be unavailable")
        if self.sessions_from_previous_observation[0] != 0:
            raise ValueError("the first surface-history session count must be zero")

    @property
    def linked_into_date(self) -> BoolArray:
        """Identify dates linked to their predecessor by exactly one session."""
        linked = self.sessions_from_previous_observation == 1
        linked[0] = False
        return linked


def fixed_grid_sanos_day(
    day: PublicMarketDay,
    *,
    grid: GenerativeSurfaceGrid = GenerativeSurfaceGrid(),
) -> SanosFixedGridDay:
    """Recover a fixed-grid IV surface solely from the released SANOS state.

    Prices at exact Black bounds do not identify finite implied volatility.
    Those nodes retain ``False`` in ``implied_vol_is_interior`` and are filled
    by linear interpolation in log strike, with flat tail extrapolation, from
    the same expiry's numerically interior SANOS nodes. A short expiry with
    fewer than two interior coarse nodes is repriced on a denser ATM-centered
    strike grid before interpolation.
    """
    expiries = np.asarray(grid.expiries_y, dtype=np.float64)
    if grid.nms_nodes is None:
        strike_mesh = np.broadcast_to(
            np.asarray(grid.strikes_over_forward, dtype=np.float64),
            (len(expiries), grid.strike_count),
        )
    else:
        atm_prices = price_sanos_surface(
            query_expiries_y=expiries,
            query_strikes=np.ones_like(expiries),
            is_call=np.ones_like(expiries, dtype=bool),
            model_expiries_y=day.sanos_model_expiries_y,
            atm_variances=day.sanos_kernel_total_variances,
            support_strikes=day.sanos_support_strikes,
            densities=day.sanos_densities,
            smoothness=day.sanos_smoothness,
            one_minute_y=day.one_minute_y,
        )
        atm_volatilities, atm_valid = implied_volatilities_with_validity(
            expiries_y=expiries,
            normalized_strikes=np.ones_like(expiries),
            normalized_prices=atm_prices,
            is_call=np.ones_like(expiries, dtype=bool),
        )
        if not np.all(atm_valid & np.isfinite(atm_volatilities) & (atm_volatilities > 0.0)):
            raise FloatingPointError("SANOS ATM volatilities do not define an NMS grid")
        strike_mesh = NMS.from_nms(
            np.asarray(grid.nms_nodes, dtype=np.float64)[None, :],
            expiries[:, None],
            vol=atm_volatilities[:, None],
        )
    expiry_mesh = np.broadcast_to(expiries[:, None], strike_mesh.shape)
    sides = strike_mesh >= 1.0
    normalized_prices = price_sanos_surface(
        query_expiries_y=expiry_mesh.ravel(),
        query_strikes=strike_mesh.ravel(),
        is_call=sides.ravel(),
        model_expiries_y=day.sanos_model_expiries_y,
        atm_variances=day.sanos_kernel_total_variances,
        support_strikes=day.sanos_support_strikes,
        densities=day.sanos_densities,
        smoothness=day.sanos_smoothness,
        one_minute_y=day.one_minute_y,
    )
    implied_volatilities, interior = implied_volatilities_with_validity(
        expiries_y=expiry_mesh.ravel(),
        normalized_strikes=strike_mesh.ravel(),
        normalized_prices=normalized_prices,
        is_call=sides.ravel(),
    )
    implied_volatilities = implied_volatilities.reshape(expiry_mesh.shape)
    interior = interior.reshape(expiry_mesh.shape)
    repaired = np.empty_like(implied_volatilities)
    for expiry_index in range(len(expiries)):
        log_strikes = np.log(strike_mesh[expiry_index])
        valid = interior[expiry_index]
        if np.count_nonzero(valid) < 2:
            expiry = expiries[expiry_index]
            atm_total_variance = np.interp(
                expiry,
                day.sanos_model_expiries_y,
                day.sanos_kernel_total_variances,
            )
            local_strikes = np.exp(
                np.linspace(-5.0, 5.0, 41)
                * np.sqrt(max(atm_total_variance, 1.0e-12))
            )
            local_prices = price_sanos_surface(
                query_expiries_y=np.full_like(local_strikes, expiry),
                query_strikes=local_strikes,
                is_call=local_strikes >= 1.0,
                model_expiries_y=day.sanos_model_expiries_y,
                atm_variances=day.sanos_kernel_total_variances,
                support_strikes=day.sanos_support_strikes,
                densities=day.sanos_densities,
                smoothness=day.sanos_smoothness,
                one_minute_y=day.one_minute_y,
            )
            local_volatilities, local_interior = implied_volatilities_with_validity(
                expiries_y=np.full_like(local_strikes, expiry),
                normalized_strikes=local_strikes,
                normalized_prices=local_prices,
                is_call=local_strikes >= 1.0,
            )
            if np.count_nonzero(local_interior) < 2:
                raise FloatingPointError(
                    f"{day.trading_date}: SANOS expiry {expiry:g} "
                    "has fewer than two interior implied-volatility nodes"
                )
            valid_log_strikes = np.log(local_strikes[local_interior])
            valid_volatilities = local_volatilities[local_interior]
        else:
            valid_log_strikes = log_strikes[valid]
            valid_volatilities = implied_volatilities[expiry_index, valid]
        if np.any(~np.isfinite(valid_volatilities) | (valid_volatilities <= 0.0)):
            raise FloatingPointError("interior SANOS implied volatilities are invalid")
        repaired[expiry_index] = np.exp(
            np.interp(
                log_strikes,
                valid_log_strikes,
                np.log(valid_volatilities),
            )
        )
    return SanosFixedGridDay(
        trading_date=day.trading_date,
        log_implied_volatilities=np.log(repaired),
        implied_vol_is_interior=interior,
    )


def build_sanos_fixed_grid_history(
    panel: PublicHedgingPanel,
    *,
    end_date: date,
    observation_count: int,
    grid: GenerativeSurfaceGrid = GenerativeSurfaceGrid(),
) -> SanosFixedGridHistory:
    """Build a trailing public SANOS history ending no later than ``end_date``."""
    require_public_hedging_panel(panel)
    verify_inp(
        observation_count >= 2,
        lambda: "observation_count must be at least two",
    )
    try:
        end_index = panel.dates.index(end_date)
    except ValueError as error:
        raise KeyError(f"public panel does not contain {end_date}") from error
    start_index = end_index + 1 - observation_count
    if start_index < 0:
        raise ValueError(
            f"{end_date} has only {end_index + 1} observations including itself"
        )
    dates = panel.dates[start_index : end_index + 1]
    days = tuple(fixed_grid_sanos_day(panel.market_day(value), grid=grid) for value in dates)
    observation_rows = tuple(
        panel.observations.row(index, named=True)
        for index in range(start_index, end_index + 1)
    )
    returns = np.asarray(
        [
            math.nan
            if offset == 0
            else math.log1p(float(row["forward_adjusted_spot_return_into_date"]))
            for offset, row in enumerate(observation_rows)
        ],
        dtype=np.float64,
    )
    sessions = np.asarray(
        [
            0
            if offset == 0
            else int(row["business_sessions_from_previous_observation"])
            for offset, row in enumerate(observation_rows)
        ],
        dtype=np.int64,
    )
    return SanosFixedGridHistory(
        dates=dates,
        log_implied_volatilities=np.stack(
            [day.log_implied_volatilities for day in days]
        ),
        implied_vol_is_interior=np.stack(
            [day.implied_vol_is_interior for day in days]
        ),
        forward_log_returns_into_date=returns,
        sessions_from_previous_observation=sessions,
    )
