"""DLV-reconstructed same-coordinate returns for the covariance baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.hedging.contracts import OptionQuote, OptionRight
from quantdata.hedging.data.pricing import (
    price_dlv_digitals,
    price_dlv_surface,
    price_sanos_digitals,
    price_sanos_surface,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalTarget,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
    PublicHedgingPanel,
    PublicMarketDay,
)

FloatArray = NDArray[np.float64]
COVARIANCE_RETURN_GEOMETRY_IDENTITY = "released-dlv-state-v1"
ReturnPriceSource = Literal["dlv", "sanos"]


@dataclass(frozen=True, kw_only=True)
class RelativeOptionCoordinate:
    """One option identified by initial DTE, NMS, and side."""

    instrument_id: str
    business_dte: int
    nms: float
    right: OptionRight

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("instrument_id must be non-empty")
        if self.business_dte <= 0 or not math.isfinite(self.nms):
            raise ValueError("relative option coordinate must have valid DTE and NMS")

    @classmethod
    def from_quote(cls, quote: OptionQuote) -> RelativeOptionCoordinate:
        """Construct a historical coordinate from a current executable quote."""
        business_dte = int(round(quote.business_dte))
        if not math.isclose(
            quote.business_dte,
            business_dte,
            rel_tol=0.0,
            abs_tol=1.0e-5,
        ):
            raise ValueError("covariance replay requires integer business DTE")
        return cls(
            instrument_id=quote.contract.contract_id,
            business_dte=business_dte,
            nms=quote.nms,
            right=quote.contract.right,
        )


@dataclass(frozen=True, kw_only=True)
class RelativeReturnObservation:
    """One realized spot/option vector and target-return vector."""

    origin_date: date
    terminal_date: date
    instrument_returns: FloatArray
    target_returns: FloatArray

    def __post_init__(self) -> None:
        if self.terminal_date <= self.origin_date:
            raise ValueError("return observation terminal date must follow origin")
        for name, values in (
            ("instrument_returns", self.instrument_returns),
            ("target_returns", self.target_returns),
        ):
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite vector")


@dataclass(frozen=True, kw_only=True)
class HistoricalReturnPanel:
    """Rolling historical returns aligned to one current hedge universe."""

    training_origin_dates: tuple[date, ...]
    training_terminal_dates: tuple[date, ...]
    instrument_ids: tuple[str, ...]
    target_labels: tuple[str, ...]
    instrument_returns: FloatArray
    target_returns: FloatArray

    def __post_init__(self) -> None:
        observations = len(self.training_origin_dates)
        if len(self.training_terminal_dates) != observations or any(
            terminal <= origin
            for origin, terminal in zip(
                self.training_origin_dates,
                self.training_terminal_dates,
                strict=True,
            )
        ):
            raise ValueError("historical training date pairs are inconsistent")
        if self.instrument_returns.shape != (observations, len(self.instrument_ids)):
            raise ValueError("historical instrument-return panel shape is inconsistent")
        if self.target_returns.shape != (observations, len(self.target_labels)):
            raise ValueError("historical target-return panel shape is inconsistent")
        if len(set(self.instrument_ids)) != len(self.instrument_ids):
            raise ValueError("historical instrument identifiers must be unique")
        if len(set(self.target_labels)) != len(self.target_labels):
            raise ValueError("historical target labels must be unique")
        if not (
            np.all(np.isfinite(self.instrument_returns))
            and np.all(np.isfinite(self.target_returns))
        ):
            raise ValueError("historical return panels must be finite")


def build_historical_return_panel(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    option_quotes: tuple[OptionQuote, ...],
    targets: tuple[DailyDigitalTarget, ...],
    count: int,
    price_source: ReturnPriceSource = "dlv",
) -> HistoricalReturnPanel:
    """Replay a rolling window at the current relative option coordinates.

    The current option identities determine only DTE, NMS, and side. Each
    historical transition constructs a new fixed-strike option at that same
    relative coordinate and follows it for one session using the selected
    released model state. No current or future return enters the training panel.
    """
    if count < 2:
        raise ValueError("historical covariance panel requires at least two rows")
    if not option_quotes or not targets:
        raise ValueError("historical covariance panel requires options and targets")
    if price_source not in ("dlv", "sanos"):
        raise ValueError("historical return price source must be DLV or SANOS")
    coordinates = tuple(
        RelativeOptionCoordinate.from_quote(quote) for quote in option_quotes
    )
    training_origins = panel.prior_one_day_origins(origin_date, count=count)
    observations: list[RelativeReturnObservation] = []
    for training_origin in training_origins:
        transition = panel.one_day_transition(training_origin)
        observations.append(
            relative_return_observation(
                transition,
                endpoint=panel.market_day(transition.next_date),
                coordinates=coordinates,
                targets=targets,
                price_source=price_source,
            )
        )
    return HistoricalReturnPanel(
        training_origin_dates=training_origins,
        training_terminal_dates=tuple(
            observation.terminal_date for observation in observations
        ),
        instrument_ids=("spot", *(coordinate.instrument_id for coordinate in coordinates)),
        target_labels=tuple(target.label for target in targets),
        instrument_returns=np.stack(
            [observation.instrument_returns for observation in observations]
        ),
        target_returns=np.stack(
            [observation.target_returns for observation in observations]
        ),
    )


def relative_return_observation(
    transition: OneDayTransition,
    *,
    endpoint: PublicMarketDay,
    coordinates: tuple[RelativeOptionCoordinate, ...],
    targets: tuple[DailyDigitalTarget, ...],
    price_source: ReturnPriceSource = "dlv",
) -> RelativeReturnObservation:
    """Evaluate one historical transition at fixed relative coordinates."""
    if endpoint.trading_date != transition.next_date:
        raise ValueError("relative-return endpoint does not match transition")
    option_returns = _relative_option_returns(
        transition,
        endpoint=endpoint,
        coordinates=coordinates,
        price_source=price_source,
    )
    target_returns = _relative_digital_returns(
        transition,
        endpoint=endpoint,
        targets=targets,
        price_source=price_source,
    )
    return RelativeReturnObservation(
        origin_date=transition.origin.trading_date,
        terminal_date=transition.next_date,
        instrument_returns=np.concatenate(
            (np.asarray([transition.discounted_spot_gain]), option_returns)
        ),
        target_returns=target_returns,
    )


def _relative_option_returns(
    transition: OneDayTransition,
    *,
    endpoint: PublicMarketDay,
    coordinates: tuple[RelativeOptionCoordinate, ...],
    price_source: ReturnPriceSource,
) -> FloatArray:
    if not coordinates:
        return np.empty(0, dtype=np.float64)
    origin = transition.origin
    business_dtes = np.asarray(
        [coordinate.business_dte for coordinate in coordinates], dtype=np.int64
    )
    nms = np.asarray([coordinate.nms for coordinate in coordinates], dtype=np.float64)
    is_call = np.asarray(
        [coordinate.right is OptionRight.CALL for coordinate in coordinates],
        dtype=bool,
    )
    origin_atm_vols = _curve_values(
        origin,
        business_dtes,
        method="atm_vol_at_business_dte",
    )
    strikes_over_forward = np.asarray(
        NMS.from_nms(
            nms,
            business_dtes / DAYS_PER_YEAR,
            vol=origin_atm_vols,
            eps_vol=origin.book_selection_epsilon_atm_vol,
        ),
        dtype=np.float64,
    )
    origin_forwards = _curve_values(
        origin,
        business_dtes,
        method="forward_at_business_dte",
    )
    origin_discounts = _curve_values(
        origin,
        business_dtes,
        method="discount_at_business_dte",
    )
    fixed_strikes = origin_forwards * strikes_over_forward
    origin_prices = origin_discounts * origin_forwards * _option_prices(
        origin,
        business_dtes=business_dtes,
        strikes_over_forward=strikes_over_forward,
        is_call=is_call,
        price_source=price_source,
    )

    gross_spot_return = (
        transition.forward_over_spot * transition.terminal_normalized_spot
    )
    terminal_values = np.empty(len(coordinates), dtype=np.float64)
    settles = business_dtes == 1
    terminal_values[settles] = transition.discount_factor * np.where(
        is_call[settles],
        np.maximum(gross_spot_return - fixed_strikes[settles], 0.0),
        np.maximum(fixed_strikes[settles] - gross_spot_return, 0.0),
    )
    remains = ~settles
    if np.any(remains):
        remaining_dtes = business_dtes[remains] - 1
        endpoint_forwards = _curve_values(
            endpoint,
            remaining_dtes,
            method="forward_at_business_dte",
        )
        endpoint_discounts = _curve_values(
            endpoint,
            remaining_dtes,
            method="discount_at_business_dte",
        )
        endpoint_strikes = fixed_strikes[remains] / (
            gross_spot_return * endpoint_forwards
        )
        endpoint_prices = _option_prices(
            endpoint,
            business_dtes=remaining_dtes,
            strikes_over_forward=endpoint_strikes,
            is_call=is_call[remains],
            price_source=price_source,
        )
        terminal_values[remains] = (
            transition.discount_factor
            * gross_spot_return
            * endpoint_discounts
            * endpoint_forwards
            * endpoint_prices
        )
    return terminal_values - origin_prices


def _relative_digital_returns(
    transition: OneDayTransition,
    *,
    endpoint: PublicMarketDay,
    targets: tuple[DailyDigitalTarget, ...],
    price_source: ReturnPriceSource,
) -> FloatArray:
    if not targets:
        return np.empty(0, dtype=np.float64)
    origin = transition.origin
    business_dtes = np.asarray(
        [target.maturity_business_days for target in targets], dtype=np.int64
    )
    nms = np.asarray([target.nms for target in targets], dtype=np.float64)
    is_call = np.asarray(
        [target.right is OptionRight.CALL for target in targets], dtype=bool
    )
    origin_atm_vols = _curve_values(
        origin,
        business_dtes,
        method="atm_vol_at_business_dte",
    )
    strikes_over_forward = np.asarray(
        NMS.from_nms(
            nms,
            business_dtes / DAYS_PER_YEAR,
            vol=origin_atm_vols,
            eps_vol=origin.book_selection_epsilon_atm_vol,
        ),
        dtype=np.float64,
    )
    origin_forwards = _curve_values(
        origin,
        business_dtes,
        method="forward_at_business_dte",
    )
    origin_discounts = _curve_values(
        origin,
        business_dtes,
        method="discount_at_business_dte",
    )
    fixed_strikes = origin_forwards * strikes_over_forward
    origin_prices = origin_discounts * _digital_prices(
        origin,
        business_dtes=business_dtes,
        strikes_over_forward=strikes_over_forward,
        is_call=is_call,
        price_source=price_source,
    )

    gross_spot_return = (
        transition.forward_over_spot * transition.terminal_normalized_spot
    )
    terminal_values = np.empty(len(targets), dtype=np.float64)
    settles = business_dtes == 1
    terminal_values[settles] = transition.discount_factor * np.where(
        is_call[settles],
        gross_spot_return > fixed_strikes[settles],
        gross_spot_return <= fixed_strikes[settles],
    )
    remains = ~settles
    if np.any(remains):
        remaining_dtes = business_dtes[remains] - 1
        endpoint_forwards = _curve_values(
            endpoint,
            remaining_dtes,
            method="forward_at_business_dte",
        )
        endpoint_strikes = fixed_strikes[remains] / (
            gross_spot_return * endpoint_forwards
        )
        endpoint_probabilities = _digital_prices(
            endpoint,
            business_dtes=remaining_dtes,
            strikes_over_forward=endpoint_strikes,
            is_call=is_call[remains],
            price_source=price_source,
        )
        endpoint_discounts = _curve_values(
            endpoint,
            remaining_dtes,
            method="discount_at_business_dte",
        )
        terminal_values[remains] = (
            transition.discount_factor
            * endpoint_discounts
            * endpoint_probabilities
        )
    return terminal_values - origin_prices


def _curve_values(
    day: PublicMarketDay,
    business_dtes: NDArray[np.integer],
    *,
    method: str,
) -> FloatArray:
    evaluator = getattr(day, method)
    return np.asarray(
        [evaluator(int(business_dte)) for business_dte in business_dtes],
        dtype=np.float64,
    )


def _option_prices(
    day: PublicMarketDay,
    *,
    business_dtes: NDArray[np.integer],
    strikes_over_forward: FloatArray,
    is_call: NDArray[np.bool_],
    price_source: ReturnPriceSource,
) -> FloatArray:
    if price_source == "sanos":
        return price_sanos_surface(
            query_expiries_y=np.asarray(business_dtes, dtype=np.float64)
            / DAYS_PER_YEAR,
            query_strikes=strikes_over_forward,
            is_call=is_call,
            model_expiries_y=day.sanos_model_expiries_y,
            atm_variances=day.sanos_kernel_total_variances,
            support_strikes=day.sanos_support_strikes,
            densities=day.sanos_densities,
            smoothness=day.sanos_smoothness,
            one_minute_y=day.one_minute_y,
        )
    return _dlv_option_prices(
        day,
        business_dtes=business_dtes,
        strikes_over_forward=strikes_over_forward,
        is_call=is_call,
    )


def _digital_prices(
    day: PublicMarketDay,
    *,
    business_dtes: NDArray[np.integer],
    strikes_over_forward: FloatArray,
    is_call: NDArray[np.bool_],
    price_source: ReturnPriceSource,
) -> FloatArray:
    if price_source == "sanos":
        return price_sanos_digitals(
            query_expiries_y=np.asarray(business_dtes, dtype=np.float64)
            / DAYS_PER_YEAR,
            query_strikes=strikes_over_forward,
            is_call=is_call,
            model_expiries_y=day.sanos_model_expiries_y,
            atm_variances=day.sanos_kernel_total_variances,
            support_strikes=day.sanos_support_strikes,
            densities=day.sanos_densities,
            smoothness=day.sanos_smoothness,
            one_minute_y=day.one_minute_y,
        )
    return _dlv_digital_prices(
        day,
        business_dtes=business_dtes,
        strikes_over_forward=strikes_over_forward,
        is_call=is_call,
    )


def _dlv_option_prices(
    day: PublicMarketDay,
    *,
    business_dtes: NDArray[np.integer],
    strikes_over_forward: FloatArray,
    is_call: NDArray[np.bool_],
) -> FloatArray:
    return price_dlv_surface(
        query_expiries_y=np.asarray(business_dtes, dtype=np.float64) / DAYS_PER_YEAR,
        query_strikes=strikes_over_forward,
        is_call=is_call,
        model_expiries_y=day.dlv_model_expiries_y,
        support_strikes=day.dlv_support_strikes,
        raw_log_excess_dlvsq=day.dlv_raw_log_excess_dlvsq,
        minimum_dlvsq=day.minimum_dlvsq,
        maximum_dlvsq=day.maximum_dlvsq,
        backbone_annual_volatility=day.backbone_annual_volatility,
    )


def _dlv_digital_prices(
    day: PublicMarketDay,
    *,
    business_dtes: NDArray[np.integer],
    strikes_over_forward: FloatArray,
    is_call: NDArray[np.bool_],
) -> FloatArray:
    return price_dlv_digitals(
        query_expiries_y=np.asarray(business_dtes, dtype=np.float64) / DAYS_PER_YEAR,
        query_strikes=strikes_over_forward,
        is_call=is_call,
        model_expiries_y=day.dlv_model_expiries_y,
        support_strikes=day.dlv_support_strikes,
        raw_log_excess_dlvsq=day.dlv_raw_log_excess_dlvsq,
        minimum_dlvsq=day.minimum_dlvsq,
        maximum_dlvsq=day.maximum_dlvsq,
        backbone_annual_volatility=day.backbone_annual_volatility,
    )
