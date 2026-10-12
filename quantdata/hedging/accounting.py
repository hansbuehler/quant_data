"""Common self-financing execution and claim P&L accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantdata.hedging.contracts import OptionQuote


@dataclass(frozen=True, kw_only=True)
class TradeExecution:
    """One trade expressed relative to its common fitted-model mark."""

    instrument_id: str
    quantity_change: float
    reference_price: float
    execution_price: float
    cashflow: float
    transaction_cost: float

    def __post_init__(self) -> None:
        values = (
            self.quantity_change,
            self.reference_price,
            self.execution_price,
            self.cashflow,
            self.transaction_cost,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("trade execution values must be finite")
        if self.reference_price < 0.0 or self.execution_price < 0.0:
            raise ValueError("execution prices must be non-negative")
        if self.transaction_cost < 0.0:
            raise ValueError("transaction_cost must be non-negative")
        expected_cashflow = (
            -self.quantity_change * self.reference_price - self.transaction_cost
        )
        if not math.isclose(self.cashflow, expected_cashflow, abs_tol=1.0e-12):
            raise ValueError("cashflow does not reconcile to midpoint less costs")


@dataclass(frozen=True, kw_only=True)
class ClaimPnlBreakdown:
    """Pricing and pure-hedging components of one signed claim position."""

    claim_quantity: float
    market_price: float
    model_price: float
    payoff: float
    hedge_gain: float
    transaction_cost: float
    pricing_contribution: float
    pure_hedge_error: float
    economic_pnl: float

    def __post_init__(self) -> None:
        values = (
            self.claim_quantity,
            self.market_price,
            self.model_price,
            self.payoff,
            self.hedge_gain,
            self.transaction_cost,
            self.pricing_contribution,
            self.pure_hedge_error,
            self.economic_pnl,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("claim P&L values must be finite")
        if self.transaction_cost < 0.0:
            raise ValueError("transaction_cost must be non-negative")
        if not math.isclose(
            self.pricing_contribution + self.pure_hedge_error,
            self.economic_pnl,
            abs_tol=1.0e-12,
        ):
            raise ValueError("pricing and hedging terms do not reconcile")


def execute_option_trade(
    quote: OptionQuote,
    quantity_change: float,
) -> TradeExecution:
    """Execute around the SANOS mark using the normalized half-spread."""
    if not math.isfinite(quantity_change):
        raise ValueError("quantity_change must be finite")
    if quantity_change > 0.0:
        execution_price = quote.synthetic_ask
    elif quantity_change < 0.0:
        execution_price = quote.synthetic_bid
    else:
        execution_price = quote.mark
    transaction_cost = abs(quantity_change) * quote.half_spread
    cashflow = -quantity_change * quote.mark - transaction_cost
    return TradeExecution(
        instrument_id=quote.contract.contract_id,
        quantity_change=quantity_change,
        reference_price=quote.mark,
        execution_price=execution_price,
        cashflow=cashflow,
        transaction_cost=transaction_cost,
    )


def execute_spot_trade(
    *,
    quantity_change: float,
    normalized_spot: float,
    cost_bps: float = 1.0,
    instrument_id: str = "spot",
) -> TradeExecution:
    """Execute normalized spot with an explicit cost on traded notional."""
    if not math.isfinite(quantity_change):
        raise ValueError("quantity_change must be finite")
    if not math.isfinite(normalized_spot) or normalized_spot <= 0.0:
        raise ValueError("normalized_spot must be finite and positive")
    if not math.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("cost_bps must be finite and non-negative")
    transaction_cost = abs(quantity_change) * normalized_spot * cost_bps * 1.0e-4
    cashflow = -quantity_change * normalized_spot - transaction_cost
    return TradeExecution(
        instrument_id=instrument_id,
        quantity_change=quantity_change,
        reference_price=normalized_spot,
        execution_price=normalized_spot,
        cashflow=cashflow,
        transaction_cost=transaction_cost,
    )


def settlement_cashflow(*, position: float, payoff: float) -> float:
    """Return cash settlement with no transaction cost."""
    if not math.isfinite(position) or not math.isfinite(payoff) or payoff < 0.0:
        raise ValueError("position and non-negative payoff must be finite")
    return position * payoff


def claim_pnl_breakdown(
    *,
    claim_quantity: float,
    market_price: float,
    model_price: float,
    payoff: float,
    hedge_gain: float,
    transaction_cost: float,
) -> ClaimPnlBreakdown:
    """Decompose common-market-price P&L into pricing and hedging terms."""
    inputs = (
        claim_quantity,
        market_price,
        model_price,
        payoff,
        hedge_gain,
        transaction_cost,
    )
    if any(not math.isfinite(value) for value in inputs):
        raise ValueError("claim P&L inputs must be finite")
    if transaction_cost < 0.0:
        raise ValueError("transaction_cost must be non-negative")
    pricing_contribution = claim_quantity * (model_price - market_price)
    pure_hedge_error = (
        -claim_quantity * model_price
        + claim_quantity * payoff
        + hedge_gain
        - transaction_cost
    )
    economic_pnl = (
        -claim_quantity * market_price
        + claim_quantity * payoff
        + hedge_gain
        - transaction_cost
    )
    return ClaimPnlBreakdown(
        claim_quantity=claim_quantity,
        market_price=market_price,
        model_price=model_price,
        payoff=payoff,
        hedge_gain=hedge_gain,
        transaction_cost=transaction_cost,
        pricing_contribution=pricing_contribution,
        pure_hedge_error=pure_hedge_error,
        economic_pnl=economic_pnl,
    )
