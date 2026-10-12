"""Deterministic normalized target claims used by the hedging benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.hedging.contracts import OptionRight


class BarrierDirection(StrEnum):
    """Direction of an end-of-day knock-out barrier."""

    DOWN = "down"
    UP = "up"


def normalized_strike_from_nms(
    *,
    nms: float,
    business_dte: float,
    atm_annual_vol: float,
    epsilon_atm_annual_vol: float | None = None,
) -> float:
    """Convert inception NMS into a fixed normalized strike."""
    if epsilon_atm_annual_vol is None:
        epsilon_atm_annual_vol = atm_annual_vol
    if not all(
        math.isfinite(value)
        for value in (
            nms,
            business_dte,
            atm_annual_vol,
            epsilon_atm_annual_vol,
        )
    ):
        raise ValueError("NMS conversion inputs must be finite")
    if (
        business_dte < 0.0
        or atm_annual_vol <= 0.0
        or epsilon_atm_annual_vol <= 0.0
    ):
        raise ValueError("business_dte must be non-negative and volatilities positive")
    return float(
        NMS.from_nms(
            nms,
            business_dte / DAYS_PER_YEAR,
            vol=atm_annual_vol,
            eps_vol=epsilon_atm_annual_vol,
        )
    )


@dataclass(frozen=True, kw_only=True)
class DigitalClaim:
    """Unit-cash European digital on normalized terminal spot."""

    right: OptionRight
    strike: float
    maturity_business_days: int
    cash_payoff: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.strike) or self.strike <= 0.0:
            raise ValueError("strike must be finite and positive")
        if self.maturity_business_days <= 0:
            raise ValueError("maturity_business_days must be positive")
        if not math.isfinite(self.cash_payoff) or self.cash_payoff <= 0.0:
            raise ValueError("cash_payoff must be finite and positive")

    def payoff(self, normalized_spot_path: ArrayLike) -> float:
        """Return the terminal digital payoff."""
        path = _validated_path(normalized_spot_path, self.maturity_business_days)
        terminal = float(path[-1])
        exercised = (
            terminal > self.strike
            if self.right is OptionRight.CALL
            else terminal <= self.strike
        )
        return self.cash_payoff if exercised else 0.0


@dataclass(frozen=True, kw_only=True)
class RealizedVarianceClaim:
    """Average squared normalized simple return over a fixed horizon."""

    maturity_business_days: int = 5

    def __post_init__(self) -> None:
        if self.maturity_business_days <= 0:
            raise ValueError("maturity_business_days must be positive")

    def payoff(self, normalized_spot_path: ArrayLike) -> float:
        """Return unannualized average realized variance."""
        path = _validated_path(normalized_spot_path, self.maturity_business_days)
        returns = np.diff(path) / path[:-1]
        return float(np.mean(np.square(returns)))


@dataclass(frozen=True, kw_only=True)
class BarrierOptionClaim:
    """European option knocked out by future business-day closing levels."""

    right: OptionRight
    strike: float
    barrier: float
    direction: BarrierDirection
    maturity_business_days: int = 5

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) and value > 0.0 for value in (self.strike, self.barrier)):
            raise ValueError("strike and barrier must be finite and positive")
        if self.maturity_business_days <= 0:
            raise ValueError("maturity_business_days must be positive")
        if self.direction is BarrierDirection.DOWN and self.right is not OptionRight.PUT:
            raise ValueError("the benchmark down-and-out claim must be a put")
        if self.direction is BarrierDirection.UP and self.right is not OptionRight.CALL:
            raise ValueError("the benchmark up-and-out claim must be a call")

    def payoff(self, normalized_spot_path: ArrayLike) -> float:
        """Return terminal intrinsic value if no EOD barrier was breached."""
        path = _validated_path(normalized_spot_path, self.maturity_business_days)
        future_closes = path[1:]
        knocked_out = (
            bool(np.any(future_closes <= self.barrier))
            if self.direction is BarrierDirection.DOWN
            else bool(np.any(future_closes >= self.barrier))
        )
        if knocked_out:
            return 0.0
        terminal = float(path[-1])
        if self.right is OptionRight.CALL:
            return max(terminal - self.strike, 0.0)
        return max(self.strike - terminal, 0.0)


def _validated_path(path: ArrayLike, maturity_business_days: int) -> np.ndarray:
    values = np.asarray(path, dtype=np.float64)
    expected = maturity_business_days + 1
    if values.shape != (expected,):
        raise ValueError(f"expected a path of shape {(expected,)}, found {values.shape}")
    if np.any(~np.isfinite(values) | (values <= 0.0)):
        raise ValueError("normalized spot path must be finite and positive")
    return values
