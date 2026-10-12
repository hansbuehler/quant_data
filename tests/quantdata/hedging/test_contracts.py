from __future__ import annotations

from datetime import date, timedelta

from quantdata.hedging.contracts import (
    OptionContract,
    OptionQuote,
    OptionRight,
    eligible_option_quotes,
)


def _quote(
    *,
    contract_id: str,
    strike: float,
    right: OptionRight,
    business_dte: float,
    spread: float = 0.02,
    exterior: bool = False,
) -> OptionQuote:
    trading_date = date(2025, 1, 2)
    return OptionQuote(
        trading_date=trading_date,
        contract=OptionContract(
            contract_id=contract_id,
            expiry_date=trading_date + timedelta(days=max(1, round(business_dte))),
            normalized_strike=strike,
            right=right,
            exterior_boundary=exterior,
        ),
        business_dte=business_dte,
        atm_annual_vol=0.2,
        bid=0.10 - 0.5 * spread,
        ask=0.10 + 0.5 * spread,
        display_size=10.0,
    )


def test_universe_uses_otm_book_strict_nms_and_first_expiry_after_five() -> None:
    quotes = (
        _quote(
            contract_id="3d-put",
            strike=0.99,
            right=OptionRight.PUT,
            business_dte=3.0,
        ),
        _quote(
            contract_id="3d-call-itm",
            strike=0.99,
            right=OptionRight.CALL,
            business_dte=3.0,
        ),
        _quote(
            contract_id="6d-call",
            strike=1.01,
            right=OptionRight.CALL,
            business_dte=6.0,
        ),
        _quote(
            contract_id="7d-call",
            strike=1.01,
            right=OptionRight.CALL,
            business_dte=7.0,
        ),
        _quote(
            contract_id="boundary",
            strike=1.02,
            right=OptionRight.CALL,
            business_dte=3.0,
            exterior=True,
        ),
    )
    selected = eligible_option_quotes(quotes)
    assert [quote.contract.contract_id for quote in selected] == ["3d-put", "6d-call"]


def test_exact_five_day_expiry_excludes_later_expiry() -> None:
    quotes = (
        _quote(
            contract_id="5d-call",
            strike=1.01,
            right=OptionRight.CALL,
            business_dte=5.0,
        ),
        _quote(
            contract_id="6d-call",
            strike=1.01,
            right=OptionRight.CALL,
            business_dte=6.0,
        ),
    )
    selected = eligible_option_quotes(quotes)
    assert [quote.contract.contract_id for quote in selected] == ["5d-call"]
