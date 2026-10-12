from __future__ import annotations

from datetime import date

import pytest

from quantdata.hedging.accounting import (
    claim_pnl_breakdown,
    execute_option_trade,
    execute_spot_trade,
    settlement_cashflow,
)
from quantdata.hedging.contracts import OptionContract, OptionQuote, OptionRight


def _quote() -> OptionQuote:
    return OptionQuote(
        trading_date=date(2025, 1, 2),
        contract=OptionContract(
            contract_id="SPX-2025-01-03-C-1.0",
            expiry_date=date(2025, 1, 3),
            fixed_strike=1.0,
            right=OptionRight.CALL,
        ),
        business_dte=1.0,
        atm_annual_vol=0.2,
        strike_over_forward=1.0,
        standardized_log_moneyness=0.0,
        mark=0.05,
        half_spread=0.01,
        display_size=50.0,
    )


def test_option_trade_uses_full_entry_book_and_explicit_half_spread() -> None:
    bought = execute_option_trade(_quote(), 3.0)
    sold = execute_option_trade(_quote(), -3.0)

    assert bought.execution_price == pytest.approx(0.06)
    assert bought.cashflow == pytest.approx(-0.18)
    assert bought.transaction_cost == pytest.approx(0.03)
    assert sold.execution_price == pytest.approx(0.04)
    assert sold.cashflow == pytest.approx(0.12)
    assert sold.transaction_cost == pytest.approx(0.03)


def test_spot_cost_is_one_basis_point_of_traded_notional() -> None:
    trade = execute_spot_trade(quantity_change=-2.0, normalized_spot=1.25)
    assert trade.transaction_cost == pytest.approx(2.0 * 1.25e-4)
    assert trade.cashflow == pytest.approx(2.5 - trade.transaction_cost)


def test_cash_settlement_has_no_transaction_cost() -> None:
    assert settlement_cashflow(position=-4.0, payoff=0.25) == pytest.approx(-1.0)


@pytest.mark.parametrize("claim_quantity", [-2.0, 3.0])
def test_claim_pnl_decomposition_reconciles_for_long_and_short(
    claim_quantity: float,
) -> None:
    result = claim_pnl_breakdown(
        claim_quantity=claim_quantity,
        market_price=0.42,
        model_price=0.39,
        payoff=1.0,
        hedge_gain=-0.17,
        transaction_cost=0.02,
    )
    direct = (
        -claim_quantity * 0.42
        + claim_quantity * 1.0
        - 0.17
        - 0.02
    )
    assert result.economic_pnl == pytest.approx(direct)
    assert result.pricing_contribution + result.pure_hedge_error == pytest.approx(
        direct
    )
