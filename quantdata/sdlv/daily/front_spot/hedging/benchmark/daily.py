"""Common one-day target, instrument, and accounting path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.accounting import (
    claim_pnl_breakdown,
    execute_option_trade,
    execute_spot_trade,
)
from quantdata.hedging.claims import DigitalClaim
from quantdata.hedging.contracts import OptionQuote, OptionRight
from quantdata.hedging.data.pricing import price_sanos_digitals
from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import OneDayTransition


@dataclass(frozen=True, kw_only=True)
class OneDayMarketSample:
    """One ATM digital and one executable option over a historical transition."""

    transition: OneDayTransition
    claim: DigitalClaim
    market_price: float
    target_spot_delta: float
    hedge_option: OptionQuote
    spot_return: float
    option_return: float
    claim_return: float
    payoff: float

    def __post_init__(self) -> None:
        values = (
            self.market_price,
            self.target_spot_delta,
            self.spot_return,
            self.option_return,
            self.claim_return,
            self.payoff,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("one-day sample values must be finite")


def one_day_market_sample(transition: OneDayTransition) -> OneDayMarketSample:
    """Build the common ATM digital and nearest-ATM one-day call sample."""
    claim = DigitalClaim(
        right=OptionRight.CALL,
        strike=1.0,
        maturity_business_days=1,
    )
    market_price = _sanos_digital_price(transition, strike=claim.strike)
    target_delta = _sanos_digital_spot_delta(transition, strike=claim.strike)
    snapshot = transition.origin.hedge_snapshot()
    candidates = tuple(
        quote
        for quote in snapshot.option_quotes
        if quote.contract.expiry_date == transition.next_date
        and quote.contract.right is OptionRight.CALL
    )
    if not candidates:
        raise ValueError(
            f"{transition.origin.trading_date} has no eligible one-day PM call hedge"
        )
    hedge_option = min(
        candidates,
        key=lambda quote: (abs(quote.nms), quote.half_spread, quote.contract.contract_id),
    )
    terminal_forward_normalized_spot = transition.terminal_normalized_spot
    payoff = transition.discount_factor * claim.payoff(
        [1.0, terminal_forward_normalized_spot]
    )
    terminal_spot_over_origin = (
        transition.forward_over_spot * terminal_forward_normalized_spot
    )
    option_payoff = transition.discount_factor * hedge_option.contract.payoff(
        terminal_spot_over_origin
    )
    option_return = option_payoff - hedge_option.mark
    claim_return = payoff - market_price
    return OneDayMarketSample(
        transition=transition,
        claim=claim,
        market_price=market_price,
        target_spot_delta=target_delta,
        hedge_option=hedge_option,
        spot_return=transition.discounted_spot_gain,
        option_return=option_return,
        claim_return=claim_return,
        payoff=payoff,
    )


def evaluate_one_day_hedge(
    sample: OneDayMarketSample,
    *,
    model: str,
    measure: str,
    spot_action: float,
    option_action: float,
    model_price: float,
    objective_value: float,
    converged: bool,
    cap_hits: int = 0,
    diagnostics: dict[str, object] | None = None,
) -> HedgeSmokeResult:
    """Charge option entry costs and report formal exit cost separately.

    Options are cash settled and therefore incur only their opening half-spread.
    The normalized spot hedge trades at mid without a transaction cost.
    """
    spot_trade = execute_spot_trade(
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
    option_trade = execute_option_trade(sample.hedge_option, option_action)
    transaction_cost = (
        spot_trade.transaction_cost
        + option_trade.transaction_cost
    )
    hedge_gain = (
        spot_action * sample.spot_return + option_action * sample.option_return
    )
    breakdown = claim_pnl_breakdown(
        claim_quantity=1.0,
        market_price=sample.market_price,
        model_price=model_price,
        payoff=sample.payoff,
        hedge_gain=hedge_gain,
        transaction_cost=transaction_cost,
    )
    return HedgeSmokeResult(
        model=model,
        measure=measure,
        origin_date=sample.transition.origin.trading_date.isoformat(),
        terminal_date=sample.transition.next_date.isoformat(),
        claim="atm-1d-digital-call",
        market_price=sample.market_price,
        model_price=model_price,
        payoff=sample.payoff,
        unhedged_pnl=sample.claim_return,
        hedged_pnl=breakdown.economic_pnl,
        transaction_cost=transaction_cost,
        actions={
            "spot": spot_action,
            sample.hedge_option.contract.contract_id: option_action,
        },
        pricing_difference=breakdown.pricing_contribution,
        pure_hedge_error=breakdown.pure_hedge_error,
        objective_value=objective_value,
        converged=converged,
        cap_hits=cap_hits,
        diagnostics={
            **({} if diagnostics is None else diagnostics),
            "formal_exit_cost": spot_close.transaction_cost,
            "formal_spot_exit_cost": spot_close.transaction_cost,
            "formal_option_exit_cost": 0.0,
            "cost_convention": "entry-only; formal exit excluded from P&L",
        },
    )


def _sanos_digital_price(transition: OneDayTransition, *, strike: float) -> float:
    day = transition.origin
    forward_unit_price = float(
        price_sanos_digitals(
            query_expiries_y=np.asarray([1.0 / DAYS_PER_YEAR]),
            query_strikes=np.asarray([strike]),
            is_call=np.asarray([True]),
            model_expiries_y=day.sanos_model_expiries_y,
            atm_variances=day.sanos_kernel_total_variances,
            support_strikes=day.sanos_support_strikes,
            densities=day.sanos_densities,
            smoothness=day.sanos_smoothness,
            one_minute_y=day.one_minute_y,
        )[0]
    )
    return transition.discount_factor * forward_unit_price


def _sanos_digital_spot_delta(
    transition: OneDayTransition,
    *,
    strike: float,
    relative_step: float = 1.0e-4,
) -> float:
    if not 0.0 < relative_step < 1.0:
        raise ValueError("relative_step must lie strictly between zero and one")
    value_up = _sanos_digital_price(
        transition,
        strike=strike / (1.0 + relative_step),
    )
    value_down = _sanos_digital_price(
        transition,
        strike=strike / (1.0 - relative_step),
    )
    return (value_up - value_down) / (2.0 * relative_step)
