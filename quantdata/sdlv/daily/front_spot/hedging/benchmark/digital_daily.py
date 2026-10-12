"""Common one-session accounting for the daily digital benchmark family."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.accounting import (
    claim_pnl_breakdown,
    execute_option_trade,
    execute_spot_trade,
)
from quantdata.hedging.claims import DigitalClaim, normalized_strike_from_nms
from quantdata.hedging.contracts import OptionQuote, OptionRight
from quantdata.hedging.data.pricing import price_sanos_digitals
from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
    PublicMarketDay,
)

FloatArray = NDArray[np.float64]


class DailyMarketPanel(Protocol):
    """Minimal dated-panel interface required by the daily benchmark."""

    def one_day_transition(self, origin_date: date) -> OneDayTransition: ...

    def market_day(self, trading_date: date) -> PublicMarketDay: ...


@dataclass(frozen=True, kw_only=True)
class DailyDigitalTarget:
    """One inception-defined target in the agreed daily digital family."""

    maturity_business_days: int
    right: OptionRight
    nms: float

    def __post_init__(self) -> None:
        if self.maturity_business_days not in {1, 3}:
            raise ValueError("daily digital maturity must be one or three business days")
        if not math.isfinite(self.nms):
            raise ValueError("digital target NMS must be finite")
        if self.right is OptionRight.PUT and self.nms > 0.0:
            raise ValueError("daily OTM puts cannot have positive NMS")
        if self.right is OptionRight.CALL and self.nms < 0.0:
            raise ValueError("daily OTM calls cannot have negative NMS")

    @property
    def label(self) -> str:
        """Return a stable human-readable target label."""
        nms_label = f"nms{self.nms:+.2f}" if self.nms != 0.0 else "atm"
        return (
            f"digital-{self.right.value}-{self.maturity_business_days}d-"
            f"{nms_label}"
        )


def daily_digital_targets() -> tuple[DailyDigitalTarget, ...]:
    """Return the 20 agreed one-day and three-day digital targets."""
    one_maturity = (
        DailyDigitalTarget(
            maturity_business_days=1,
            right=OptionRight.PUT,
            nms=0.0,
        ),
        *(
            DailyDigitalTarget(
                maturity_business_days=1,
                right=OptionRight.PUT,
                nms=nms,
            )
            for nms in (-1.0, -0.75, -0.5, -0.25)
        ),
        DailyDigitalTarget(
            maturity_business_days=1,
            right=OptionRight.CALL,
            nms=0.0,
        ),
        *(
            DailyDigitalTarget(
                maturity_business_days=1,
                right=OptionRight.CALL,
                nms=nms,
            )
            for nms in (0.25, 0.5, 0.75, 1.0)
        ),
    )
    return tuple(
        DailyDigitalTarget(
            maturity_business_days=maturity,
            right=target.right,
            nms=target.nms,
        )
        for maturity in (1, 3)
        for target in one_maturity
    )


def daily_claim_quantity_vector(
    targets: tuple[DailyDigitalTarget, ...],
    claim_quantities: Mapping[str, float] | None,
) -> FloatArray:
    """Return finite-positive claim quantities aligned to ``targets``."""
    if claim_quantities is None:
        return np.ones(len(targets), dtype=np.float64)
    labels = tuple(target.label for target in targets)
    if set(claim_quantities) != set(labels):
        raise ValueError("claim quantities must exactly match the daily target labels")
    quantities = np.asarray(
        [claim_quantities[label] for label in labels],
        dtype=np.float64,
    )
    if quantities.shape != (len(targets),) or np.any(
        ~np.isfinite(quantities) | (quantities <= 0.0)
    ):
        raise ValueError("daily claim quantities must be finite-positive")
    return quantities


@dataclass(frozen=True, kw_only=True)
class DailyDigitalSample:
    """One target and fixed inception hedge universe over one market session."""

    transition: OneDayTransition
    endpoint: PublicMarketDay
    target: DailyDigitalTarget
    claim: DigitalClaim
    target_strike_over_forward: float
    target_discount_factor: float
    market_price: float
    terminal_value: float
    claim_return: float
    hedge_quotes: tuple[OptionQuote, ...]
    hedge_returns: FloatArray
    closing_half_spreads_pv: FloatArray

    def __post_init__(self) -> None:
        if self.endpoint.trading_date != self.transition.next_date:
            raise ValueError("daily sample endpoint does not match transition")
        if self.hedge_returns.shape != (len(self.hedge_quotes),):
            raise ValueError("hedge returns do not align with quotes")
        if self.closing_half_spreads_pv.shape != self.hedge_returns.shape:
            raise ValueError("closing spreads do not align with hedge returns")
        numeric = (
            self.target_strike_over_forward,
            self.target_discount_factor,
            self.market_price,
            self.terminal_value,
            self.claim_return,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("daily digital sample must be finite")
        if (
            self.target_strike_over_forward <= 0.0
            or self.target_discount_factor <= 0.0
            or self.market_price < 0.0
            or self.terminal_value < 0.0
            or not np.all(np.isfinite(self.hedge_returns))
            or not np.all(
                np.isfinite(self.closing_half_spreads_pv)
                & (self.closing_half_spreads_pv >= 0.0)
            )
        ):
            raise ValueError("daily digital sample contains invalid prices")


def build_daily_digital_sample(
    panel: DailyMarketPanel,
    *,
    origin_date: date,
    target: DailyDigitalTarget,
) -> DailyDigitalSample:
    """Build one target and its one-session realized market transition."""
    transition = panel.one_day_transition(origin_date)
    origin = transition.origin
    endpoint = panel.market_day(transition.next_date)
    maturity = target.maturity_business_days
    atm_vol = origin.atm_vol_at_business_dte(maturity)
    strike_over_forward = normalized_strike_from_nms(
        nms=target.nms,
        business_dte=float(maturity),
        atm_annual_vol=atm_vol,
        epsilon_atm_annual_vol=origin.book_selection_epsilon_atm_vol,
    )
    forward_over_spot = origin.forward_at_business_dte(maturity)
    discount_factor = origin.discount_at_business_dte(maturity)
    fixed_strike = strike_over_forward * forward_over_spot
    claim = DigitalClaim(
        right=target.right,
        strike=fixed_strike,
        maturity_business_days=maturity,
    )
    market_price = discount_factor * _sanos_digital_probability(
        origin,
        business_dte=maturity,
        strike_over_forward=strike_over_forward,
        right=target.right,
    )
    gross_spot_return = (
        transition.forward_over_spot * transition.terminal_normalized_spot
    )
    if maturity == 1:
        terminal_value = transition.discount_factor * claim.payoff(
            [1.0, gross_spot_return]
        )
    else:
        remaining = maturity - 1
        endpoint_forward = endpoint.forward_at_business_dte(remaining)
        endpoint_strike = fixed_strike / (gross_spot_return * endpoint_forward)
        endpoint_value = endpoint.discount_at_business_dte(
            remaining
        ) * _sanos_digital_probability(
            endpoint,
            business_dte=remaining,
            strike_over_forward=endpoint_strike,
            right=target.right,
        )
        terminal_value = transition.discount_factor * endpoint_value

    hedge_quotes = origin.hedge_snapshot().option_quotes
    if not hedge_quotes:
        raise ValueError(f"{origin_date} has no eligible hedge options")
    endpoint_snapshot = endpoint.reprice_contracts(
        tuple(quote.contract for quote in hedge_quotes),
        spot_over_episode_origin=gross_spot_return,
    )
    endpoint_quotes = endpoint_snapshot.quote_by_id()
    hedge_returns = np.empty(len(hedge_quotes), dtype=np.float64)
    closing_spreads = np.zeros(len(hedge_quotes), dtype=np.float64)
    for index, quote in enumerate(hedge_quotes):
        if quote.contract.expiry_date == transition.next_date:
            terminal_payoff = transition.discount_factor * quote.contract.payoff(
                gross_spot_return
            )
            hedge_returns[index] = terminal_payoff - quote.mark
            continue
        endpoint_quote = endpoint_quotes.get(quote.contract.contract_id)
        if endpoint_quote is None:
            raise ValueError(
                f"{quote.contract.contract_id}: no endpoint synthetic mark"
            )
        hedge_returns[index] = (
            transition.discount_factor * endpoint_quote.mark - quote.mark
        )
        closing_spreads[index] = (
            transition.discount_factor * endpoint_quote.half_spread
        )
    return DailyDigitalSample(
        transition=transition,
        endpoint=endpoint,
        target=target,
        claim=claim,
        target_strike_over_forward=strike_over_forward,
        target_discount_factor=discount_factor,
        market_price=market_price,
        terminal_value=terminal_value,
        claim_return=terminal_value - market_price,
        hedge_quotes=hedge_quotes,
        hedge_returns=hedge_returns,
        closing_half_spreads_pv=closing_spreads,
    )


def evaluate_daily_digital_hedge(
    sample: DailyDigitalSample,
    *,
    model: str,
    measure: str,
    spot_action: float,
    option_actions: NDArray[np.floating],
    model_price: float,
    objective_value: float,
    converged: bool,
    cap_hits: int,
    claim_quantity: float = 1.0,
    diagnostics: dict[str, object] | None = None,
) -> HedgeSmokeResult:
    """Apply common entry/exit costs and emit one model-neutral record."""
    if not math.isfinite(claim_quantity) or claim_quantity == 0.0:
        raise ValueError("claim_quantity must be finite and nonzero")
    actions = np.asarray(option_actions, dtype=np.float64)
    if actions.shape != (len(sample.hedge_quotes),) or not np.all(
        np.isfinite(actions)
    ):
        raise ValueError("option actions must align with the hedge universe")
    spot_open = execute_spot_trade(
        quantity_change=spot_action,
        normalized_spot=1.0,
    )
    terminal_spot_pv = (
        sample.transition.discounted_forward_over_spot
        * sample.transition.terminal_normalized_spot
    )
    spot_close = execute_spot_trade(
        quantity_change=-spot_action,
        normalized_spot=terminal_spot_pv,
        instrument_id="spot-close",
    )
    opening_option_cost = sum(
        execute_option_trade(quote, float(action)).transaction_cost
        for quote, action in zip(sample.hedge_quotes, actions, strict=True)
    )
    closing_option_cost = float(
        np.sum(np.abs(actions) * sample.closing_half_spreads_pv)
    )
    transaction_cost = (
        spot_open.transaction_cost
        + spot_close.transaction_cost
        + opening_option_cost
        + closing_option_cost
    )
    hedge_gain = float(
        spot_action * sample.transition.discounted_spot_gain
        + actions @ sample.hedge_returns
    )
    breakdown = claim_pnl_breakdown(
        claim_quantity=claim_quantity,
        market_price=sample.market_price,
        model_price=model_price,
        payoff=sample.terminal_value,
        hedge_gain=hedge_gain,
        transaction_cost=transaction_cost,
    )
    action_map = {"spot": spot_action}
    action_map.update(
        {
            quote.contract.contract_id: float(action)
            for quote, action in zip(sample.hedge_quotes, actions, strict=True)
            if abs(action) > 1.0e-14
        }
    )
    return HedgeSmokeResult(
        model=model,
        measure=measure,
        origin_date=sample.transition.origin.trading_date.isoformat(),
        terminal_date=sample.transition.next_date.isoformat(),
        claim=sample.target.label,
        market_price=sample.market_price,
        model_price=model_price,
        payoff=sample.terminal_value,
        unhedged_pnl=claim_quantity * sample.claim_return,
        hedged_pnl=breakdown.economic_pnl,
        transaction_cost=transaction_cost,
        actions=action_map,
        pricing_difference=breakdown.pricing_contribution,
        pure_hedge_error=breakdown.pure_hedge_error,
        objective_value=objective_value,
        converged=converged,
        cap_hits=cap_hits,
        claim_quantity=claim_quantity,
        diagnostics={} if diagnostics is None else diagnostics,
    )


def _sanos_digital_probability(
    day: PublicMarketDay,
    *,
    business_dte: int,
    strike_over_forward: float,
    right: OptionRight,
) -> float:
    return float(
        price_sanos_digitals(
            query_expiries_y=np.asarray([business_dte / DAYS_PER_YEAR]),
            query_strikes=np.asarray([strike_over_forward]),
            is_call=np.asarray([right is OptionRight.CALL]),
            model_expiries_y=day.sanos_model_expiries_y,
            atm_variances=day.sanos_kernel_total_variances,
            support_strikes=day.sanos_support_strikes,
            densities=day.sanos_densities,
            smoothness=day.sanos_smoothness,
            one_minute_y=day.one_minute_y,
        )[0]
    )
