"""Cboe's discrete-strike VIX formula, independent of a pricing model.

Reference: https://cdn.cboe.com/resources/vix/VIX_Methodology.pdf
Prices are discounted currency prices; ``discount_factor`` is exp(-R*T).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify_inp
from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class VixStrip:
    """Selected OTM contracts, including the averaged call/put at K0."""

    indices: NDArray[np.int64]
    strikes: NDArray[np.float64]
    delta_strikes: NDArray[np.float64]
    prices: NDArray[np.float64]
    forward: float
    k0: float
    put_zero_bid_stop: float | None
    call_zero_bid_stop: float | None


def select_vix_strip(
    *,
    strikes: NDArray[np.float64],
    call_bids: NDArray[np.float64],
    call_asks: NDArray[np.float64],
    put_bids: NDArray[np.float64],
    put_asks: NDArray[np.float64],
    discount_factor: float,
) -> VixStrip:
    """Infer F by put-call parity and apply the two-consecutive-zero-bid rule.

    Absent contracts are NaN, not zero bids. Invalid books break a consecutive
    zero sequence; an absent contract is not a strike in that side's chain.
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    books = [np.asarray(x, dtype=np.float64) for x in (call_bids, call_asks, put_bids, put_asks)]
    verify_inp(strikes.ndim == 1 and len(strikes) >= 3,
               lambda: "At least three strikes are required")
    verify_inp(all(x.shape == strikes.shape for x in books), lambda: "Book shapes must match strikes")
    verify_inp(np.all(np.isfinite(strikes)) and np.all(strikes > 0) and np.all(np.diff(strikes) > 0),
               lambda: "Strikes must be finite, positive and strictly increasing")
    verify_inp(np.isfinite(discount_factor) and discount_factor > 0, lambda: "Invalid discount factor")
    cb, ca, pb, pa = books
    valid_c = np.isfinite(cb) & np.isfinite(ca) & (cb >= 0) & (ca >= cb)
    valid_p = np.isfinite(pb) & np.isfinite(pa) & (pb >= 0) & (pa >= pb)
    calls, puts = (cb + ca) / 2, (pb + pa) / 2
    paired = np.flatnonzero(valid_c & valid_p)
    if not len(paired):
        raise ValueError("No matched call/put book for the forward")
    parity_index = paired[np.argmin(np.abs(calls[paired] - puts[paired]))]
    forward = float(strikes[parity_index] + (calls[parity_index] - puts[parity_index]) / discount_factor)
    below = np.flatnonzero(strikes <= forward)
    if not len(below) or below[-1] == len(strikes) - 1:
        raise ValueError("Available strikes do not bracket the inferred forward")
    k0_index = int(below[-1])
    if not (valid_c[k0_index] and valid_p[k0_index]):
        raise ValueError("K0 does not have both a valid call and a valid put")

    def wing(indices: range, bids: NDArray[np.float64], asks: NDArray[np.float64], valid: NDArray[np.bool_]) -> tuple[list[int], float | None]:
        selected, zero_count = [], 0
        for index in indices:
            if not valid[index]:
                if not (np.isnan(bids[index]) and np.isnan(asks[index])):
                    zero_count = 0
                continue
            if bids[index] == 0:
                zero_count += 1
                if zero_count == 2:
                    return selected, float(strikes[index])
            else:
                zero_count = 0
                selected.append(index)
        return selected, None

    left, put_stop = wing(range(k0_index - 1, -1, -1), pb, pa, valid_p)
    right, call_stop = wing(range(k0_index + 1, len(strikes)), cb, ca, valid_c)
    indices = np.asarray(sorted(left + [k0_index] + right), dtype=np.int64)
    if not left or not right:
        raise ValueError("The selected strip must contain both OTM wings")
    k = strikes[indices]
    delta = np.empty_like(k)
    delta[0], delta[-1] = k[1] - k[0], k[-1] - k[-2]
    delta[1:-1] = (k[2:] - k[:-2]) / 2
    q = np.where(indices < k0_index, puts[indices], calls[indices])
    q[indices == k0_index] = (puts[k0_index] + calls[k0_index]) / 2
    return VixStrip(indices=indices, strikes=k, delta_strikes=delta, prices=q,
                    forward=forward, k0=float(strikes[k0_index]),
                    put_zero_bid_stop=put_stop, call_zero_bid_stop=call_stop)


def discrete_total_variance(
    strip: VixStrip, *, discount_factor: float, prices: NDArray[np.float64] | None = None,
) -> float:
    """Return T*sigma² = 2*sum(deltaK*Q/(DF*K²)) - (F/K0-1)²."""
    q = strip.prices if prices is None else np.asarray(prices, dtype=np.float64)
    verify_inp(q.shape == strip.strikes.shape and np.all(np.isfinite(q)) and np.all(q >= 0),
               lambda: "Selected prices must be finite, non-negative and match the strip")
    verify_inp(np.isfinite(discount_factor) and discount_factor > 0, lambda: "Invalid discount factor")
    value = 2 * np.sum(strip.delta_strikes * q / (discount_factor * strip.strikes**2))
    value -= (strip.forward / strip.k0 - 1)**2
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid discrete total variance: {value}")
    return float(value)


def constant_maturity_vix(
    *, near_days: float, next_days: float, near_total_variance: float,
    next_total_variance: float, target_days: float = 30.0,
) -> float:
    """Calendar-time interpolation of total variance, annualized ACT/365."""
    verify_inp(0 < near_days <= target_days <= next_days and near_days < next_days,
               lambda: "Expiries must bracket the target without extrapolation")
    verify_inp(np.isfinite(near_total_variance) and np.isfinite(next_total_variance)
               and min(near_total_variance, next_total_variance) >= 0,
               lambda: "Invalid term variance")
    alpha = (target_days - near_days) / (next_days - near_days)
    return float(100 * np.sqrt(((1 - alpha) * near_total_variance + alpha * next_total_variance) * 365 / target_days))
