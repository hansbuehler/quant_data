"""Unchanged floating contracts, with cheap Black continuation after absorption.

All market arrays remain outside Git. This module changes sampling, not the
market model, policy inputs, instrument universe, or expiry-gain convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr, ndtri

from quantdata.sdlv.daily.front_spot.estimation import BACKGROUND_ANNUAL_VOL
from quantdata.sdlv.daily.front_spot.jax_model import black_mixture_call_prices
from quantdata.sdlv.dhgromm1.feedback import contract_expiries
from quantdata.sdlv.dhgromm1.observable import market_geometry

jax.config.update("jax_enable_x64", True)
EXPIRIES = contract_expiries()
OFFSETS = market_geometry()[0]
TENORS = np.stack((np.ones(20), np.full(20, 5), np.minimum(10, 25-np.arange(20))), -1)
STRIKE_RATIOS = np.c_[np.zeros(20), np.ones(20),
    np.exp(.20*np.sqrt(TENORS[:, :, None]/252)*np.array([-2., -1.5, -1., -.5, -.25])).reshape(20, 15),
    np.zeros((20, 4))]


def black_calls(spot: np.ndarray, strike: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Undiscounted calls under the exact existing independent Black background."""
    spot, strike, days = np.broadcast_arrays(spot, strike, days)
    vol = BACKGROUND_ANNUAL_VOL*np.sqrt(np.maximum(days, 0)/252)
    safe_vol = np.maximum(vol, 1e-12)
    d1 = np.log(spot/np.maximum(strike, np.finfo(float).tiny))/safe_vol+.5*safe_vol
    value = spot*ndtr(d1)-strike*ndtr(d1-safe_vol)
    return np.where(days > 0, np.maximum(value, 0), np.maximum(spot-strike, 0))


def boundary_probabilities(front: np.ndarray) -> tuple[np.ndarray, ...]:
    """Effective endpoint/interior masses of the ORIGINAL Gaussian-CDF sampler.

    The final cell receives all probability above the preceding cumulative
    sum, as in sample_front_index; no tail probability is discarded or floored.
    """
    front = np.asarray(front, dtype=np.float64)
    if front.ndim != 2 or not np.isfinite(front).all() or front.min() < -1e-12:
        raise ValueError("expected a finite nonnegative batch of front densities")
    np.testing.assert_allclose(front.sum(-1), 1, atol=1e-10, rtol=0)
    lower = np.clip(front[:, 0], 0, 1)
    upper_cdf = np.clip(front[:, :-1].sum(-1), lower, 1)
    return lower, 1-upper_cdf, upper_cdf-lower


def interior_normals(normals: np.ndarray, lower: np.ndarray, interior: np.ndarray) -> np.ndarray:
    """Condition factor zero on interior survival; preserve the other 7 factors.

    Factor zero ALSO drives the DLV update, so changing only the sampled spot
    while leaving its Gaussian shock unconditioned would change the Q law.
    """
    out = np.asarray(normals, dtype=np.float64).copy()
    u = lower+interior*ndtr(out[:, 0])
    active = interior > 0
    safe = np.clip(u, np.nextafter(0., 1.), np.nextafter(1., 0.))
    out[:, 0] = np.where(active, ndtri(safe), 0.)
    return out


@jax.jit
def mixture_calls(support, tower, background, strikes, remaining):
    density = tower[:, remaining-1, :]
    return black_mixture_call_prices(support, density, strikes[..., None], background, remaining)[..., 0]


@dataclass
class MarketPaths:
    """Current quotes and causal historical marks, never a future policy input."""

    prices: np.ndarray
    packed: np.ndarray

    @classmethod
    def empty(cls, count: int) -> MarketPaths:
        return cls(np.zeros((count, 21, 21)), np.zeros((count, 3990), np.float32))

    def fork(self, indices: np.ndarray) -> MarketPaths:
        return MarketPaths(self.prices[indices].copy(), self.packed[indices].copy())

    def strikes(self, day: int) -> np.ndarray:
        return self.prices[:, :day, :1]*STRIKE_RATIOS[None, :day]

    def marks(self, day: int, spot: np.ndarray, *, support=None, tower=None, background=None) -> np.ndarray:
        """All previously opened contracts, live marks or already observed cash."""
        count = len(spot)
        strikes = self.strikes(day)
        result = np.empty((count, day, 21))
        result[:, :, 0] = spot[:, None]
        remaining = EXPIRIES[:day]-day
        for instrument in range(1, 21):
            dates = EXPIRIES[:day, instrument]
            settled_spot = self.prices[:, np.minimum(dates, day), 0]
            if instrument < 17:
                result[:, :, instrument] = np.maximum(settled_spot-strikes[:, :, instrument], 0)
            else:
                result[:, :, instrument] = settled_spot
        cohort, instrument = np.nonzero((remaining[:, 1:17] > 0))
        instrument = instrument+1
        if len(cohort):
            k, dte = strikes[:, cohort, instrument], remaining[cohort, instrument]
            live = (black_calls(spot[:, None], k, dte[None]) if tower is None else
                    np.asarray(mixture_calls(support, tower, background, jnp.asarray(k), jnp.asarray(dte))))
            result[:, cohort, instrument] = live
        for instrument in range(17, 21):
            alive = np.flatnonzero(remaining[:, instrument] > 0)
            if len(alive):
                live = (spot[:, None] if tower is None else
                    np.asarray(background)[:, None]*np.einsum("bci,i->bc",
                        np.asarray(tower)[:, remaining[alive, instrument]-1], np.asarray(support)))
                result[:, alive, instrument] = live
        return result

    def observe(self, day: int, spot: np.ndarray, *, support=None, tower=None, background=None) -> None:
        """Observe before the day's shock; all settlements use dates <= day."""
        spot = np.asarray(spot)
        self.prices[:, day, 0] = spot
        if day:
            changes = self.marks(day, spot, support=support, tower=tower, background=background)-self.prices[:, :day]
            if day < 20:
                self.packed[:, OFFSETS[day]:OFFSETS[day+1]] = changes.reshape(len(spot), -1)
        if day == 20:
            return
        k = spot[:, None]*STRIKE_RATIOS[day, 1:17]
        dte = EXPIRIES[day, 1:17]-day
        calls = (black_calls(spot[:, None], k, dte[None]) if tower is None else
                 np.asarray(mixture_calls(support, tower, background, jnp.asarray(k), jnp.asarray(dte))))
        self.prices[:, day, 1:17] = calls
        forward_dte = EXPIRIES[day, 17:]-day
        self.prices[:, day, 17:] = (spot[:, None] if tower is None else
            np.asarray(background)[:, None]*np.einsum("bci,i->bc",
                np.asarray(tower)[:, forward_dte-1], np.asarray(support)))

    def arrays(self, *, support=None, tower=None, background=None) -> tuple[np.ndarray, ...]:
        """Realized floating payoff-minus-price; no division by option premium."""
        spot = self.prices[:, 20, 0]
        close = self.marks(20, spot, support=support, tower=tower, background=background)
        returns = (close-self.prices[:, :20])/self.prices[:, :20, :1]
        payoff = np.abs(spot/self.prices[:, 10, 0]-1)
        if not np.isfinite(returns).all() or not np.isfinite(self.packed).all():
            raise FloatingPointError("nonfinite conditional market arrays")
        return self.prices, returns, payoff


def absorbed_paths(prefix: MarketPaths, *, day: int, background: np.ndarray,
                   endpoints: np.ndarray, normals: np.ndarray) -> MarketPaths:
    """Fork after today's action into both endpoints and M Black continuations.

    Layout [outer, endpoint, continuation]. Shared Black normals retain
    covariance between endpoints and all frozen policies. Endpoint DLV shocks
    are irrelevant because every future density is the same endpoint atom.
    """
    count, inner, steps = normals.shape
    if steps != 20-day or len(background) != count:
        raise ValueError("Black continuation has incompatible horizon")
    indices = np.repeat(np.arange(count), 2*inner)
    result = prefix.fork(indices)
    increments = -.5*BACKGROUND_ANNUAL_VOL**2/252+BACKGROUND_ANNUAL_VOL/np.sqrt(252)*normals
    levels = background[:, None, None]*np.exp(np.cumsum(increments, axis=-1))
    spots = (levels[:, None]*endpoints[None, :, None, None]).reshape(count*2*inner, steps)
    for step, date in enumerate(range(day+1, 21)):
        result.observe(date, spots[:, step])
    return result
