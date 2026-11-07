"""Fixed-cash-strike digital, single-observation KO and ratio-straddle contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from cdxcore.err import verify_inp

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.pricing import DlvPricer

NMS_LEVELS = (-1., -.75, -.5, -.25, 0., .25, .5, .75, 1.)


@dataclass(frozen=True, kw_only=True)
class Claim:
    label: str
    kind: str
    terminal_day: int
    initial_spot: float
    strike: float
    is_call: bool = True
    observation_day: int = 0
    barrier: float = 0.
    reset_spot: float | None = None
    knocked_out: bool = False
    observation_done: bool = False

    def __post_init__(self) -> None:
        verify_inp(self.kind in ("digital", "ko", "ratio_straddle"), lambda: f"Unknown claim kind {self.kind}")
        verify_inp(np.isfinite(self.initial_spot) and self.initial_spot > 0
                   and np.isfinite(self.strike) and self.strike > 0 and self.terminal_day > 0,
                   lambda: "Claim notional scale, strike and terminal day must be positive")
        verify_inp(self.kind == "digital" or 0 < self.observation_day < self.terminal_day,
                   lambda: "Path observation must precede terminal expiry")
        verify_inp(self.kind != "ko" or np.isfinite(self.barrier) and self.barrier > 0,
                   lambda: "A KO barrier must be finite and positive")

    def observe(self, *, elapsed_day: int, spot: float) -> Claim:
        if spot <= 0 or not np.isfinite(spot):
            raise ValueError("Observation spot must be finite and positive")
        if elapsed_day != self.observation_day:
            return self
        if self.kind == "ko":
            survives = spot > self.barrier if self.is_call else spot < self.barrier
            return replace(self, knocked_out=not survives, observation_done=True)
        if self.kind == "ratio_straddle":
            return replace(self, reset_spot=spot, observation_done=True)
        return self

    def payoff(self, terminal_spot: float) -> float:
        if terminal_spot <= 0 or not np.isfinite(terminal_spot):
            raise ValueError("Terminal spot must be finite and positive")
        if self.kind == "digital":
            return float(terminal_spot > self.strike if self.is_call else terminal_spot <= self.strike)
        if self.kind == "ko":
            if not self.observation_done:
                raise ValueError("The KO requires its actual barrier-date observation")
            if self.knocked_out:
                return 0.
            return max(terminal_spot - self.strike if self.is_call else self.strike - terminal_spot, 0.) / self.initial_spot
        if self.reset_spot is None:
            raise ValueError("The forward-start straddle requires its actual reset observation")
        return abs(terminal_spot / self.reset_spot - self.strike)

    def price(self, pricer: DlvPricer, curve: CashCurve, *, elapsed_day: int) -> float:
        remaining = self.terminal_day - elapsed_day
        if remaining < 0:
            raise ValueError("A settled claim cannot be repriced")
        if remaining == 0:
            return self.payoff(curve.spot)
        forward, _, discount = curve.at(float(remaining))
        if self.kind == "digital":
            return float(discount * pricer.european_at(expiry_y=remaining / DAYS_PER_YEAR,
                         strikes=np.array([self.strike / forward]), is_call=self.is_call, digital=True)[0])
        if self.kind == "ko" and self.knocked_out:
            return 0.
        observation_remaining = self.observation_day - elapsed_day
        if observation_remaining <= 0:
            if self.kind == "ko":
                if not self.observation_done:
                    raise ValueError("The KO requires its actual barrier-date observation")
                value = pricer.european_at(expiry_y=remaining / DAYS_PER_YEAR,
                                          strikes=np.array([self.strike / forward]), is_call=self.is_call)[0]
                return float(discount * forward * value / self.initial_spot)
            if self.reset_spot is None:
                raise ValueError("The forward-start straddle requires its actual reset observation")
            difference = forward * pricer.support / self.reset_spot - self.strike
            return float(discount * (pricer.density_at(remaining / DAYS_PER_YEAR) @ np.abs(difference)))
        terminal_index = _expiry_index(pricer, remaining)
        observation_index = _expiry_index(pricer, observation_remaining)
        observation_forward, _, _ = curve.at(float(observation_remaining))
        if self.kind == "ko":
            return float(discount * forward / self.initial_spot * pricer.single_observation_ko(
                ko_index=observation_index, terminal_index=terminal_index,
                strike=self.strike / forward, barrier=self.barrier / observation_forward, is_call=self.is_call))
        return float(discount * pricer.forward_start_straddle(
            reset_index=observation_index, terminal_index=terminal_index, strike_ratio=self.strike,
            forward_ratio=float(forward / observation_forward)))


def _expiry_index(pricer: DlvPricer, business_dte: int) -> int:
    matches = np.flatnonzero(np.isclose(pricer.expiries_y * DAYS_PER_YEAR, business_dte, atol=1e-6, rtol=0.))
    if len(matches) != 1:
        raise ValueError(f"The path-dependent observation requires model tenor {business_dte}D")
    return int(matches[0])


def make_claims(*, initial_spot: float, atm_vols_by_day: dict[int, float]) -> tuple[Claim, ...]:
    """ATM cash strike is inception spot; other strikes use tenor-specific NMS."""
    output = []
    for tenor in (1, 3, 5):
        for level in NMS_LEVELS:
            strike = initial_spot * NMS.from_nms(level, tenor / DAYS_PER_YEAR,
                                                vol=atm_vols_by_day[tenor], eps_vol=atm_vols_by_day[tenor])
            for is_call in ((False, True) if level == 0 else (level > 0,)):
                side = "call" if is_call else "put"
                output.append(Claim(label=f"digital-{tenor}D-{side}-{level:+.2f}", kind="digital",
                                    terminal_day=tenor, initial_spot=initial_spot,
                                    strike=float(strike), is_call=is_call))
    for level in NMS_LEVELS:
        ratio = float(NMS.from_nms(level, 2. / DAYS_PER_YEAR, vol=atm_vols_by_day[2], eps_vol=atm_vols_by_day[2]))
        is_call = level < 0
        output.append(Claim(label=f"ko-3D-5D-{'call' if is_call else 'put'}-{level:+.2f}", kind="ko",
                            terminal_day=5, observation_day=3, barrier=initial_spot,
                            initial_spot=initial_spot, strike=initial_spot * ratio, is_call=is_call))
        output.append(Claim(label=f"ratio-straddle-3D-5D-{level:+.2f}", kind="ratio_straddle",
                            terminal_day=5, observation_day=3, initial_spot=initial_spot, strike=ratio))
    return tuple(output)
