"""Immutable normalized contracts, quotes, and hedge-universe selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import numpy as np


class OptionRight(StrEnum):
    """Cash-settled European option right."""

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, kw_only=True)
class OptionContract:
    """One cash-settled option with strike fixed per unit episode-origin spot."""

    contract_id: str
    expiry_date: date
    fixed_strike: float
    right: OptionRight
    exterior_boundary: bool = False

    def __post_init__(self) -> None:
        if not self.contract_id.strip():
            raise ValueError("contract_id must be non-empty")
        if not math.isfinite(self.fixed_strike) or self.fixed_strike <= 0.0:
            raise ValueError("fixed_strike must be finite and positive")

    def payoff(self, normalized_spot: float) -> float:
        """Return settlement per unit episode-origin spot."""
        if not math.isfinite(normalized_spot) or normalized_spot <= 0.0:
            raise ValueError("normalized_spot must be finite and positive")
        if self.right is OptionRight.CALL:
            return max(normalized_spot - self.fixed_strike, 0.0)
        return max(self.fixed_strike - normalized_spot, 0.0)


@dataclass(frozen=True, kw_only=True)
class OptionQuote:
    """Synthetic executable book around the common SANOS model mark.

    Prices use spot-one units: the mark is ``C/S=DF*(F/S)*C/(DF*F)`` and the
    half-spread is the published ``h/S``. They are fitted-model quantities and
    execution costs, not an observed market midpoint and bid/ask pair.
    """

    trading_date: date
    contract: OptionContract
    business_dte: float
    atm_annual_vol: float
    strike_over_forward: float
    standardized_log_moneyness: float
    mark: float
    half_spread: float
    is_am_settlement: bool = False
    display_size: float | None = None

    def __post_init__(self) -> None:
        finite_positive = (
            self.business_dte > 0.0
            and self.atm_annual_vol > 0.0
            and self.strike_over_forward > 0.0
            and self.mark >= self.half_spread > 0.0
            and all(
                math.isfinite(value)
                for value in (
                    self.business_dte,
                    self.atm_annual_vol,
                    self.strike_over_forward,
                    self.standardized_log_moneyness,
                    self.mark,
                    self.half_spread,
                )
            )
        )
        if not finite_positive:
            raise ValueError("option quote contains invalid geometry, mark, or spread")
        if self.contract.expiry_date <= self.trading_date:
            raise ValueError("a tradable option must expire after its quote date")
        if self.display_size is not None and not (
            math.isfinite(self.display_size) and self.display_size > 0.0
        ):
            raise ValueError("display_size must be finite and positive when supplied")

    @property
    def nms(self) -> float:
        """Return current standardized log-moneyness from the public geometry."""
        return self.standardized_log_moneyness

    @property
    def synthetic_bid(self) -> float:
        """Return the executable sale price around the common mark."""
        return self.mark - self.half_spread

    @property
    def synthetic_ask(self) -> float:
        """Return the executable purchase price around the common mark."""
        return self.mark + self.half_spread

    @property
    def is_otm_book(self) -> bool:
        """Return the deterministic OTM representative convention."""
        if self.strike_over_forward < 1.0:
            return self.contract.right is OptionRight.PUT
        return self.contract.right is OptionRight.CALL


@dataclass(frozen=True, kw_only=True)
class MarketSnapshot:
    """Normalized spot and option books available at one decision time."""

    trading_date: date
    normalized_spot: float
    option_quotes: tuple[OptionQuote, ...]
    reference_spot: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.normalized_spot) or self.normalized_spot <= 0.0:
            raise ValueError("normalized_spot must be finite and positive")
        if self.reference_spot is not None and not (
            math.isfinite(self.reference_spot) and self.reference_spot > 0.0
        ):
            raise ValueError("reference_spot must be finite and positive when supplied")
        if any(quote.trading_date != self.trading_date for quote in self.option_quotes):
            raise ValueError("every option quote must belong to the snapshot date")
        contract_ids = [quote.contract.contract_id for quote in self.option_quotes]
        if len(contract_ids) != len(set(contract_ids)):
            raise ValueError("market snapshot contains duplicate contract identifiers")

    def quote_by_id(self) -> dict[str, OptionQuote]:
        """Return quotes keyed by their stable contract identifier."""
        return {quote.contract.contract_id: quote for quote in self.option_quotes}


def eligible_option_quotes(
    quotes: tuple[OptionQuote, ...],
    *,
    maximum_business_dte: float = 5.0,
    nms_range: tuple[float, float] = (-2.0, 2.0),
) -> tuple[OptionQuote, ...]:
    """Select the common liquid OTM hedge universe.

    All expiries no later than ``maximum_business_dte`` are retained. If there
    is no expiry exactly at that horizon, the first later expiry is also
    retained. Duplicate books for the same expiry and strike are resolved by
    the narrowest half-spread, then by the largest displayed size.
    """
    if not math.isfinite(maximum_business_dte) or maximum_business_dte <= 0.0:
        raise ValueError("maximum_business_dte must be finite and positive")
    lower_nms, upper_nms = nms_range
    if not (
        math.isfinite(lower_nms)
        and math.isfinite(upper_nms)
        and lower_nms < upper_nms
    ):
        raise ValueError("nms_range must contain increasing finite bounds")
    if not quotes:
        return ()
    trading_dates = {quote.trading_date for quote in quotes}
    if len(trading_dates) != 1:
        raise ValueError("eligible quotes must come from one trading date")

    dtes = np.asarray([quote.business_dte for quote in quotes], dtype=np.float64)
    exact_horizon = np.any(np.isclose(dtes, maximum_business_dte, atol=1.0e-8))
    later = dtes[dtes > maximum_business_dte + 1.0e-8]
    first_later = float(np.min(later)) if len(later) and not exact_horizon else None

    candidates: dict[tuple[date, float], OptionQuote] = {}
    for quote in quotes:
        at_or_before_horizon = quote.business_dte <= maximum_business_dte + 1.0e-8
        is_first_later = first_later is not None and np.isclose(
            quote.business_dte,
            first_later,
            atol=1.0e-8,
        )
        if not (at_or_before_horizon or is_first_later):
            continue
        if (
            quote.is_am_settlement
            or quote.contract.exterior_boundary
            or not quote.is_otm_book
        ):
            continue
        if not lower_nms < quote.nms < upper_nms:
            continue
        key = (quote.contract.expiry_date, quote.contract.fixed_strike)
        existing = candidates.get(key)
        if existing is None or _quote_priority(quote) < _quote_priority(existing):
            candidates[key] = quote
    return tuple(
        sorted(
            candidates.values(),
            key=lambda quote: (
                quote.contract.expiry_date,
                quote.contract.fixed_strike,
                quote.contract.contract_id,
            ),
        )
    )


def _quote_priority(quote: OptionQuote) -> tuple[float, float, str]:
    display_size = quote.display_size if quote.display_size is not None else 0.0
    return (quote.half_spread, -display_size, quote.contract.contract_id)
