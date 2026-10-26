"""Daily fit evaluation against SANOS and retained SPX quote intervals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from cdxcore.err import verify

from .config import StaticFitConfig
from .double_exponential import call_prices
from .markets import TangentLevyMarketDay
from .metrics import (
    IntervalErrorMetrics,
    PriceErrorMetrics,
    implied_volatility_error,
    interval_error_metrics,
    price_error_metrics,
)
from .static_fit import DayFit


@dataclass(frozen=True)
class DailyFitEvaluation:
    """Outright and executable-interval metrics for one calibrated day."""

    market_date: date
    maturities: int
    optimizer_success_fraction: float
    tangent_vs_sanos_price: PriceErrorMetrics
    tangent_vs_sanos_iv: PriceErrorMetrics
    tangent_pure_interval: IntervalErrorMetrics | None
    sanos_pure_interval: IntervalErrorMetrics | None
    tangent_call_interval: IntervalErrorMetrics | None
    tangent_put_interval: IntervalErrorMetrics | None


def _optional_interval_metrics(
    model: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    mask: np.ndarray,
    *,
    half_spread_floor: float,
) -> IntervalErrorMetrics | None:
    active = (
        mask
        & np.isfinite(model)
        & np.isfinite(bids)
        & np.isfinite(asks)
        & (asks > bids)
    )
    if not np.any(active):
        return None
    return interval_error_metrics(
        model,
        bids,
        asks,
        mask=active,
        half_spread_floor=half_spread_floor,
    )


def evaluate_day_fit(
    market: TangentLevyMarketDay,
    day_fit: DayFit,
    *,
    config: StaticFitConfig = StaticFitConfig(),
) -> DailyFitEvaluation:
    """Score a calibrated DETL day at the original retained market strikes."""

    verify(
        market.market_date == day_fit.market_date,
        lambda: (
            f"market date {market.market_date} does not match fit date {day_fit.market_date}"
        ),
    )
    model_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    strike_parts: list[np.ndarray] = []
    tau_parts: list[np.ndarray] = []
    pure_bid_parts: list[np.ndarray] = []
    pure_ask_parts: list[np.ndarray] = []
    model_mask_parts: list[np.ndarray] = []
    call_bid_parts: list[np.ndarray] = []
    call_ask_parts: list[np.ndarray] = []
    put_bid_parts: list[np.ndarray] = []
    put_ask_parts: list[np.ndarray] = []
    call_active_parts: list[np.ndarray] = []
    put_active_parts: list[np.ndarray] = []
    fit_weight_parts: list[np.ndarray] = []

    for fit in day_fit.fits:
        matches = np.flatnonzero(
            np.isclose(market.expiries, fit.tau, rtol=0.0, atol=1.0e-8)
        )
        verify(
            matches.size == 1,
            lambda fit=fit: (
                f"fit maturity {fit.tau} does not uniquely match the market grid"
            ),
        )
        index = int(matches[0])
        strikes = market.strikes[index]
        model = call_prices(
            tau=fit.tau,
            log_strikes=np.log(strikes),
            parameters=fit.parameters,
            config=config.pricing,
        )
        model_parts.append(model)
        target_parts.append(market.sanos_calls[index])
        strike_parts.append(strikes)
        tau_parts.append(np.full(strikes.shape, fit.tau))
        pure_bid_parts.append(market.pure_bids[index])
        pure_ask_parts.append(market.pure_asks[index])
        model_mask_parts.append(market.model_masks[index])
        call_bid_parts.append(market.call_bids[index])
        call_ask_parts.append(market.call_asks[index])
        put_bid_parts.append(market.put_bids[index])
        put_ask_parts.append(market.put_asks[index])
        call_active_parts.append(market.call_active[index])
        put_active_parts.append(market.put_active[index])
        if market.fit_weights is not None:
            fit_weight_parts.append(market.fit_weights[index])

    model = np.concatenate(model_parts)
    target = np.concatenate(target_parts)
    strikes = np.concatenate(strike_parts)
    tau = np.concatenate(tau_parts)
    pure_bids = np.concatenate(pure_bid_parts)
    pure_asks = np.concatenate(pure_ask_parts)
    model_mask = np.concatenate(model_mask_parts)
    if market.fit_weights is None:
        half_spread = np.maximum(
            0.5 * (pure_asks - pure_bids),
            config.half_spread_floor,
        )
        residual_multiplier = None
    else:
        half_spread = None
        residual_multiplier = np.concatenate(fit_weight_parts)
    call_bids = np.concatenate(call_bid_parts)
    call_asks = np.concatenate(call_ask_parts)
    put_bids = np.concatenate(put_bid_parts)
    put_asks = np.concatenate(put_ask_parts)
    call_active = np.concatenate(call_active_parts)
    put_active = np.concatenate(put_active_parts)
    model_puts = model - 1.0 + strikes

    return DailyFitEvaluation(
        market_date=market.market_date,
        maturities=len(day_fit.fits),
        optimizer_success_fraction=float(
            np.mean([fit.success for fit in day_fit.fits])
        ),
        tangent_vs_sanos_price=price_error_metrics(
            model,
            target,
            half_spread=half_spread,
            residual_multiplier=residual_multiplier,
            mask=model_mask,
        ),
        tangent_vs_sanos_iv=implied_volatility_error(
            tau=tau,
            strikes=strikes,
            model_calls=model,
            target_calls=target,
            mask=model_mask,
        ),
        tangent_pure_interval=(
            None
            if market.fit_weights is not None
            else interval_error_metrics(
                model,
                pure_bids,
                pure_asks,
                mask=model_mask,
                half_spread_floor=config.half_spread_floor,
            )
        ),
        sanos_pure_interval=(
            None
            if market.fit_weights is not None
            else interval_error_metrics(
                target,
                pure_bids,
                pure_asks,
                mask=model_mask,
                half_spread_floor=config.half_spread_floor,
            )
        ),
        tangent_call_interval=_optional_interval_metrics(
            model,
            call_bids,
            call_asks,
            call_active,
            half_spread_floor=config.half_spread_floor,
        ),
        tangent_put_interval=_optional_interval_metrics(
            model_puts,
            put_bids,
            put_asks,
            put_active,
            half_spread_floor=config.half_spread_floor,
        ),
    )
