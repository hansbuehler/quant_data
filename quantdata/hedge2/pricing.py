"""European and path-dependent payoffs under the recovered implicit DLV chain."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.linalg import solve_banded

from quantdata.hedge2.dlv import decode_densities


@dataclass(frozen=True, kw_only=True)
class DlvPricer:
    support: np.ndarray
    expiries_y: np.ndarray
    dlvsqs: np.ndarray

    def __post_init__(self) -> None:
        if (self.dlvsqs.shape != (len(self.expiries_y), len(self.support) - 2)
                or not np.isfinite(self.dlvsqs).all() or np.min(self.dlvsqs) < 0
                or not np.isfinite(self.support).all() or np.any(self.support <= 0)
                or not np.all(np.diff(self.support) > 0)
                or not np.all(np.diff(self.expiries_y, prepend=0.) > 0)):
            raise ValueError("Invalid DLV pricing geometry")
        if np.sum(self.support == 1.) != 1:
            raise ValueError("DLV pricing requires one exact ATM node")

    def backward_interval(self, values: np.ndarray, expiry_index: int) -> np.ndarray:
        """Solve the adjoint implicit step for one or many payoff columns."""
        if values.shape[0] != len(self.support) or not np.isfinite(values).all():
            raise ValueError("Payoff values must be finite and start with the state axis")
        step = self.expiries_y[expiry_index] - (self.expiries_y[expiry_index - 1] if expiry_index else 0.)
        box = self.dlvsqs[expiry_index] * self.support[1:-1] ** 2 * step / (self.support[2:] - self.support[:-2])
        to_lower = box / (self.support[1:-1] - self.support[:-2])
        to_upper = box / (self.support[2:] - self.support[1:-1])
        bands = np.zeros((3, len(self.support)))
        bands[1] = np.r_[1., 1. + to_lower + to_upper, 1.]
        bands[0, 2:] = -to_upper
        bands[2, :-2] = -to_lower
        return solve_banded((1, 1), bands, values, check_finite=False)

    def backward_to(self, values: np.ndarray, *, from_index: int, to_index: int = -1) -> np.ndarray:
        for expiry_index in range(from_index, to_index, -1):
            values = self.backward_interval(values, expiry_index)
        return values

    def initial_value(self, values: np.ndarray) -> np.ndarray | float:
        return values[int(np.flatnonzero(self.support == 1.)[0])]

    @cached_property
    def marginal_densities(self) -> np.ndarray:
        density = decode_densities(self.support, self.dlvsqs, self.expiries_y)
        if (not np.isfinite(density).all() or np.min(density) < -1e-12
                or np.max(np.abs(density.sum(axis=1) - 1.)) > 1e-8
                or np.max(np.abs(density @ self.support - 1.)) > 1e-8):
            raise ValueError("Decoded DLV marginals do not preserve positivity, mass and mean")
        return density

    def density_at(self, expiry_y: float) -> np.ndarray:
        """Interpolate European marginals linearly in time, including the initial Dirac."""
        if not np.isfinite(expiry_y) or not 0 <= expiry_y <= self.expiries_y[-1] + 1e-12:
            raise ValueError("Requested expiry is outside the DLV pricing horizon")
        initial = np.zeros(len(self.support))
        initial[int(np.flatnonzero(self.support == 1.)[0])] = 1.
        times = np.r_[0., self.expiries_y]
        index = min(int(np.searchsorted(times, expiry_y, side="right")), len(times) - 1)
        weight = (expiry_y - times[index - 1]) / (times[index] - times[index - 1])
        previous = initial if index == 1 else self.marginal_densities[index - 2]
        return (1. - weight) * previous + weight * self.marginal_densities[index - 1]

    def european_at(self, *, expiry_y: float, strikes: np.ndarray,
                    is_call: np.ndarray | bool = True, digital: bool = False) -> np.ndarray:
        """Use exact piecewise-linear call interpolation; digitals take its right derivative."""
        difference = self.support[:, None] - np.atleast_1d(strikes)[None, :]
        if digital:
            payoff = np.where(is_call, difference > 0., difference <= 0.).astype(float)
        else:
            payoff = np.maximum(np.where(is_call, difference, -difference), 0.)
        return self.density_at(expiry_y) @ payoff

    def european(self, *, expiry_index: int, strikes: np.ndarray,
                 is_call: bool = True, digital: bool = False) -> np.ndarray:
        difference = self.support[:, None] - np.asarray(strikes)[None, :]
        if digital:
            payoff = (difference > 0).astype(float) if is_call else (difference <= 0).astype(float)
        else:
            payoff = np.maximum(difference if is_call else -difference, 0.)
        return np.asarray(self.initial_value(self.backward_to(payoff, from_index=expiry_index)))

    def single_observation_ko(self, *, ko_index: int, terminal_index: int,
                              strike: float, barrier: float, is_call: bool) -> float:
        if not 0 <= ko_index <= terminal_index < len(self.expiries_y):
            raise ValueError("KO and terminal indices must be ordered model expiries")
        payoff = np.maximum(self.support - strike if is_call else strike - self.support, 0.)
        at_ko = self.backward_to(payoff, from_index=terminal_index, to_index=ko_index)
        survival = self.support > barrier if is_call else self.support < barrier
        initial = self.backward_to(at_ko * survival, from_index=ko_index)
        return float(self.initial_value(initial))

    def forward_start_straddle(self, *, reset_index: int, terminal_index: int,
                               strike_ratio: float, forward_ratio: float = 1.) -> float:
        if (not 0 <= reset_index <= terminal_index < len(self.expiries_y)
                or not np.isfinite(strike_ratio) or not np.isfinite(forward_ratio)
                or strike_ratio <= 0 or forward_ratio <= 0):
            raise ValueError("Invalid reset, terminal or strike-ratio inputs")
        payoff_columns = np.abs(forward_ratio * self.support[:, None] / self.support[None, :] - strike_ratio)
        at_reset = self.backward_to(payoff_columns, from_index=terminal_index, to_index=reset_index)
        conditional_values = np.diag(at_reset)
        return float(self.initial_value(self.backward_to(conditional_values, from_index=reset_index)))
