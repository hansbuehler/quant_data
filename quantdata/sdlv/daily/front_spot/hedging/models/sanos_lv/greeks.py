"""Local-vol price and rolling DLV-factor Greeks for the SANOS-LV baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.contracts import OptionQuote, OptionRight
from quantdata.hedging.data.pricing import price_dlv_digitals, price_dlv_surface
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicMarketDay
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    RollingDlvFactorFit,
    raw_dlv_state_with_factor_bump,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class DlvGreeks:
    """Local-vol prices and exposures in current spot-one units."""

    model_prices: FloatArray
    spot_deltas: FloatArray
    factor_vegas: FloatArray

    def __post_init__(self) -> None:
        instrument_count = len(self.model_prices)
        if self.spot_deltas.shape != (instrument_count,):
            raise ValueError("spot deltas must align with model prices")
        if self.factor_vegas.ndim != 2 or self.factor_vegas.shape[0] != instrument_count:
            raise ValueError("factor vegas must have one row per instrument")
        arrays = (self.model_prices, self.spot_deltas, self.factor_vegas)
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("DLV Greeks must be finite")

    @property
    def factor_rank(self) -> int:
        """Return the factor-vega dimension."""
        return self.factor_vegas.shape[1]


@dataclass(frozen=True, kw_only=True)
class DeltaFactorHedge:
    """Spot and option action neutralizing local-vol first-order exposures."""

    spot_action: float
    option_actions: FloatArray
    factor_residual: FloatArray
    delta_residual: float
    cap_hits: int

    def __post_init__(self) -> None:
        if self.option_actions.ndim != 1 or self.factor_residual.ndim != 1:
            raise ValueError("hedge action and residual must be vectors")
        if (
            not np.isfinite(self.spot_action)
            or not np.all(np.isfinite(self.option_actions))
            or not np.all(np.isfinite(self.factor_residual))
            or not np.isfinite(self.delta_residual)
        ):
            raise ValueError("hedge action must be finite")
        if self.cap_hits < 0:
            raise ValueError("cap_hits cannot be negative")


def dlv_digital_greeks(
    day: PublicMarketDay,
    *,
    business_dtes: NDArray[np.floating],
    strikes_over_forward: NDArray[np.floating],
    is_call: NDArray[np.bool_],
    discount_factors: NDArray[np.floating],
    factors: RollingDlvFactorFit,
    relative_spot_step: float = 1.0e-4,
) -> DlvGreeks:
    """Price fixed-cash digitals and compute homogeneous spot/factor Greeks."""
    dtes, strikes, sides, discounts = _validate_queries(
        business_dtes=business_dtes,
        strikes_over_forward=strikes_over_forward,
        is_call=is_call,
        price_bridges=discount_factors,
        relative_spot_step=relative_spot_step,
    )
    expiries_y = dtes / DAYS_PER_YEAR
    normalized_prices = _price_digitals(
        day,
        expiries_y=expiries_y,
        strikes=strikes,
        sides=sides,
        raw_state=day.dlv_raw_log_excess_dlvsq,
    )
    price_up = _price_digitals(
        day,
        expiries_y=expiries_y,
        strikes=strikes / (1.0 + relative_spot_step),
        sides=sides,
        raw_state=day.dlv_raw_log_excess_dlvsq,
    )
    price_down = _price_digitals(
        day,
        expiries_y=expiries_y,
        strikes=strikes / (1.0 - relative_spot_step),
        sides=sides,
        raw_state=day.dlv_raw_log_excess_dlvsq,
    )
    deltas = discounts * (price_up - price_down) / (2.0 * relative_spot_step)
    vegas = _factor_vegas(
        day,
        factors=factors,
        price=lambda raw: discounts
        * _price_digitals(
            day,
            expiries_y=expiries_y,
            strikes=strikes,
            sides=sides,
            raw_state=raw,
        ),
        instrument_count=len(strikes),
    )
    return DlvGreeks(
        model_prices=discounts * normalized_prices,
        spot_deltas=deltas,
        factor_vegas=vegas,
    )


def dlv_option_greeks(
    day: PublicMarketDay,
    *,
    quotes: tuple[OptionQuote, ...],
    factors: RollingDlvFactorFit,
    relative_spot_step: float = 1.0e-4,
) -> DlvGreeks:
    """Price fixed-contract vanilla options and compute spot/factor Greeks."""
    if not quotes:
        raise ValueError("quotes must be non-empty")
    if any(quote.trading_date != day.trading_date for quote in quotes):
        raise ValueError("every quote must belong to the supplied market day")
    expiry_lookup = {expiry: index for index, expiry in enumerate(day.expiry_dates)}
    try:
        expiry_indices = np.asarray(
            [expiry_lookup[quote.contract.expiry_date] for quote in quotes],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("a quote expiry is absent from the supplied market day") from error
    dtes = np.asarray([quote.business_dte for quote in quotes])
    strikes = np.asarray([quote.strike_over_forward for quote in quotes])
    sides = np.asarray(
        [quote.contract.right is OptionRight.CALL for quote in quotes],
        dtype=bool,
    )
    bridges = (
        day.discount_factors[expiry_indices]
        * day.forward_over_spot[expiry_indices]
    )
    dtes, strikes, sides, bridges = _validate_queries(
        business_dtes=dtes,
        strikes_over_forward=strikes,
        is_call=sides,
        price_bridges=bridges,
        relative_spot_step=relative_spot_step,
    )
    expiries_y = dtes / DAYS_PER_YEAR
    normalized_prices = _price_options(
        day,
        expiries_y=expiries_y,
        strikes=strikes,
        sides=sides,
        raw_state=day.dlv_raw_log_excess_dlvsq,
    )
    price_up = (1.0 + relative_spot_step) * _price_options(
        day,
        expiries_y=expiries_y,
        strikes=strikes / (1.0 + relative_spot_step),
        sides=sides,
        raw_state=day.dlv_raw_log_excess_dlvsq,
    )
    price_down = (1.0 - relative_spot_step) * _price_options(
        day,
        expiries_y=expiries_y,
        strikes=strikes / (1.0 - relative_spot_step),
        sides=sides,
        raw_state=day.dlv_raw_log_excess_dlvsq,
    )
    deltas = bridges * (price_up - price_down) / (2.0 * relative_spot_step)
    vegas = _factor_vegas(
        day,
        factors=factors,
        price=lambda raw: bridges
        * _price_options(
            day,
            expiries_y=expiries_y,
            strikes=strikes,
            sides=sides,
            raw_state=raw,
        ),
        instrument_count=len(strikes),
    )
    return DlvGreeks(
        model_prices=bridges * normalized_prices,
        spot_deltas=deltas,
        factor_vegas=vegas,
    )


def solve_delta_factor_hedge(
    *,
    target: DlvGreeks,
    hedge_options: DlvGreeks,
    option_cost_weights: NDArray[np.floating],
    residual_tolerance: float,
    position_cap: float,
    previous_option_actions: NDArray[np.floating] | None = None,
) -> DeltaFactorHedge:
    """Return the minimum predictable-turnover exact delta/factor hedge."""
    if len(target.model_prices) != 1:
        raise ValueError("the target Greek block must contain exactly one claim")
    if target.factor_rank != hedge_options.factor_rank:
        raise ValueError("target and hedge factor dimensions do not match")
    costs = np.asarray(option_cost_weights, dtype=np.float64)
    if costs.shape != hedge_options.model_prices.shape or np.any(
        ~np.isfinite(costs) | (costs <= 0.0)
    ):
        raise ValueError("option costs must be a positive vector aligned to options")
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    if not np.isfinite(position_cap) or position_cap <= 0.0:
        raise ValueError("position_cap must be finite and positive")

    instrument_count = len(costs)
    if previous_option_actions is None:
        previous_actions = np.zeros(instrument_count, dtype=np.float64)
    else:
        previous_actions = np.asarray(previous_option_actions, dtype=np.float64)
        if previous_actions.shape != (instrument_count,) or np.any(
            ~np.isfinite(previous_actions)
        ):
            raise ValueError(
                "previous option actions must be a finite vector aligned to options"
            )
        if np.any(np.abs(previous_actions) > position_cap + 1.0e-9):
            raise ValueError("previous option actions exceed the position cap")
    target_vega = target.factor_vegas[0]
    option_vegas = hedge_options.factor_vegas
    if (
        np.linalg.norm(target_vega) <= residual_tolerance
        and np.linalg.norm(previous_actions) <= residual_tolerance
    ):
        option_actions = np.zeros(instrument_count, dtype=np.float64)
    else:
        exposure = option_vegas.T
        equality_matrix = np.concatenate(
            (exposure, np.zeros_like(exposure)),
            axis=1,
        )
        row_scale = np.maximum(
            np.max(np.abs(exposure), axis=1),
            np.abs(target_vega),
        )
        row_scale = np.where(row_scale > 0.0, row_scale, 1.0)
        identity = np.eye(instrument_count, dtype=np.float64)
        turnover_constraints = np.block(
            [
                [identity, -identity],
                [-identity, -identity],
            ]
        )
        turnover_bounds = np.concatenate(
            (previous_actions, -previous_actions)
        )
        solution = linprog(
            np.concatenate((np.zeros(instrument_count), costs)),
            A_ub=turnover_constraints,
            b_ub=turnover_bounds,
            A_eq=equality_matrix / row_scale[:, None],
            b_eq=-target_vega / row_scale,
            bounds=[
                *[(-position_cap, position_cap)] * instrument_count,
                *[(0.0, None)] * instrument_count,
            ],
            method="highs",
            options={
                "dual_feasibility_tolerance": 1.0e-9,
                "primal_feasibility_tolerance": 1.0e-9,
            },
        )
        if not solution.success:
            raise FloatingPointError(
                f"factor-vega hedge linear program failed: {solution.message}"
            )
        option_actions = solution.x[:instrument_count]
    spot_unclipped = -float(
        target.spot_deltas[0] + option_actions @ hedge_options.spot_deltas
    )
    spot_action = float(np.clip(spot_unclipped, -position_cap, position_cap))
    residual = target_vega + option_actions @ option_vegas
    delta_residual = float(
        target.spot_deltas[0]
        + option_actions @ hedge_options.spot_deltas
        + spot_action
    )
    factor_scale = max(1.0, float(np.linalg.norm(target_vega)))
    if float(np.linalg.norm(residual)) > residual_tolerance * factor_scale:
        raise FloatingPointError("factor hedge residual exceeds tolerance")
    all_actions = np.concatenate((option_actions, [spot_action]))
    cap_hits = int(
        np.count_nonzero(
            np.isclose(np.abs(all_actions), position_cap, rtol=0.0, atol=1.0e-9)
        )
    )
    if cap_hits == 0 and abs(delta_residual) > residual_tolerance:
        raise FloatingPointError("uncapped spot hedge did not neutralize delta")
    return DeltaFactorHedge(
        spot_action=spot_action,
        option_actions=option_actions,
        factor_residual=residual,
        delta_residual=delta_residual,
        cap_hits=cap_hits,
    )


def _validate_queries(
    *,
    business_dtes: NDArray[np.floating],
    strikes_over_forward: NDArray[np.floating],
    is_call: NDArray[np.bool_],
    price_bridges: NDArray[np.floating],
    relative_spot_step: float,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], FloatArray]:
    dtes = np.asarray(business_dtes, dtype=np.float64)
    strikes = np.asarray(strikes_over_forward, dtype=np.float64)
    sides = np.asarray(is_call, dtype=bool)
    bridges = np.asarray(price_bridges, dtype=np.float64)
    if not (
        dtes.ndim == 1
        and len(dtes) > 0
        and strikes.shape == dtes.shape
        and sides.shape == dtes.shape
        and bridges.shape == dtes.shape
    ):
        raise ValueError("DLV Greek query arrays must be aligned non-empty vectors")
    if np.any(~np.isfinite(dtes) | (dtes <= 0.0)):
        raise ValueError("business_dtes must be finite and positive")
    if np.any(~np.isfinite(strikes) | (strikes <= 0.0)):
        raise ValueError("strikes_over_forward must be finite and positive")
    if np.any(~np.isfinite(bridges) | (bridges <= 0.0)):
        raise ValueError("price bridges must be finite and positive")
    if not np.isfinite(relative_spot_step) or not 0.0 < relative_spot_step < 1.0:
        raise ValueError("relative_spot_step must lie strictly between zero and one")
    return dtes, strikes, sides, bridges


def _price_digitals(
    day: PublicMarketDay,
    *,
    expiries_y: FloatArray,
    strikes: FloatArray,
    sides: NDArray[np.bool_],
    raw_state: FloatArray,
) -> FloatArray:
    return price_dlv_digitals(
        query_expiries_y=expiries_y,
        query_strikes=strikes,
        is_call=sides,
        support_strikes=day.dlv_support_strikes,
        raw_log_excess_dlvsq=raw_state,
        model_expiries_y=day.dlv_model_expiries_y,
        minimum_dlvsq=day.minimum_dlvsq,
        maximum_dlvsq=day.maximum_dlvsq,
        backbone_annual_volatility=day.backbone_annual_volatility,
    )


def _price_options(
    day: PublicMarketDay,
    *,
    expiries_y: FloatArray,
    strikes: FloatArray,
    sides: NDArray[np.bool_],
    raw_state: FloatArray,
) -> FloatArray:
    return price_dlv_surface(
        query_expiries_y=expiries_y,
        query_strikes=strikes,
        is_call=sides,
        support_strikes=day.dlv_support_strikes,
        raw_log_excess_dlvsq=raw_state,
        model_expiries_y=day.dlv_model_expiries_y,
        minimum_dlvsq=day.minimum_dlvsq,
        maximum_dlvsq=day.maximum_dlvsq,
        backbone_annual_volatility=day.backbone_annual_volatility,
    )


def _factor_vegas(
    day: PublicMarketDay,
    *,
    factors: RollingDlvFactorFit,
    price: Callable[[FloatArray], FloatArray],
    instrument_count: int,
) -> FloatArray:
    vegas = np.empty((instrument_count, factors.factor_rank), dtype=np.float64)
    for factor_index in range(factors.factor_rank):
        raw_up = raw_dlv_state_with_factor_bump(
            day,
            factors,
            factor_index=factor_index,
            standard_deviations=1.0,
        )
        raw_down = raw_dlv_state_with_factor_bump(
            day,
            factors,
            factor_index=factor_index,
            standard_deviations=-1.0,
        )
        values_up = price(raw_up)
        values_down = price(raw_down)
        vegas[:, factor_index] = 0.5 * (values_up - values_down)
    return vegas
