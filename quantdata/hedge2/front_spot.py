"""Calibrated one-day spot quantiles and causal historical Gaussian scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedge2.pricing import DlvPricer
from quantdata.hedge2.source import surface_digitals
from quantdata.sanos.arbfree import ArbFreePricer


@dataclass(frozen=True, kw_only=True)
class DiscreteFrontLaw:
    """Cash-spot distribution of the current zero-backbone pricing density.

    This preserves the recovered DLV marginal, not the smoother historical
    SANOS distribution. Its forward is the parity forward used for option
    pricing, not the separate reference forward used to finance spot.
    A normal driver is transformed through its CDF, never multiplied by a
    historical spot standard deviation.
    """

    cash_support: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        support = np.asarray(self.cash_support, dtype=float)
        probabilities = np.asarray(self.probabilities, dtype=float)
        if (support.ndim != 1 or not len(support) or probabilities.shape != support.shape
                or not np.isfinite(support).all() or not np.isfinite(probabilities).all()
                or np.any(support <= 0.) or np.any(np.diff(support) <= 0.) or np.any(probabilities < 0.)):
            raise ValueError("Front law requires sorted positive cash support and nonnegative finite probabilities")
        total = float(probabilities.sum())
        if not np.isfinite(total) or abs(total - 1.) > 1e-10:
            raise ValueError("The front density must already have unit probability mass")
        positive = probabilities > 0.
        support = support[positive].copy()
        probabilities = (probabilities[positive] / total).copy()
        support.setflags(write=False)
        probabilities.setflags(write=False)
        object.__setattr__(self, "cash_support", support)
        object.__setattr__(self, "probabilities", probabilities)

    @classmethod
    def from_pricer(cls, *, pricer: DlvPricer, parity_forward: float) -> DiscreteFrontLaw:
        if not np.isfinite(parity_forward) or parity_forward <= 0.:
            raise ValueError("The option parity forward must be finite and positive")
        return cls(cash_support=parity_forward * pricer.support,
            probabilities=pricer.density_at(1. / DAYS_PER_YEAR))

    @property
    def mean(self) -> float:
        return float(self.probabilities @ self.cash_support)

    @property
    def variance(self) -> float:
        return float(self.probabilities @ (self.cash_support - self.mean)**2)

    def cdf(self, cash_spots: np.ndarray | float) -> np.ndarray:
        """Return P(S_next <= s), consistent with strict-call digital settlement."""
        values = np.asarray(cash_spots, dtype=float)
        if np.isnan(values).any():
            raise ValueError("Front CDF queries must not contain NaN")
        cumulative = np.r_[0., np.cumsum(self.probabilities)]
        cumulative[-1] = 1.
        return cumulative[np.searchsorted(self.cash_support, values, side="right")]

    def quantile(self, probabilities: np.ndarray | float) -> np.ndarray:
        """Invert the atomic CDF, allowing endpoint queries without zero-mass atoms."""
        values = np.asarray(probabilities, dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0.) | (values > 1.)):
            raise ValueError("Front quantile probabilities must lie in [0,1]")
        cumulative = np.cumsum(self.probabilities)
        cumulative[-1] = 1.
        indices = np.searchsorted(cumulative, values, side="left")
        return self.cash_support[np.minimum(indices, len(self.cash_support) - 1)]

    def from_normal(self, normal: np.ndarray | float) -> np.ndarray:
        """Invert using the smaller normal tail, without rounding Phi(z) to one."""
        values = np.asarray(normal, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Spot Gaussian drivers must be finite")
        lower = np.cumsum(self.probabilities)[:-1]
        upper = np.cumsum(self.probabilities[::-1])[::-1][1:]
        quantiles = ndtri(np.minimum(lower, upper))
        thresholds = np.where(lower <= upper, quantiles, -quantiles)
        return self.cash_support[np.searchsorted(thresholds, values, side="left")]

    def financed_spot_gains(self, *, reference_forward: float, discount: float,
                           initial_spot: float) -> np.ndarray:
        """Raw financed gains; do not silently remove parity/reference-forward basis."""
        if (not np.isfinite([reference_forward, discount, initial_spot]).all()
                or min(reference_forward, discount, initial_spot) <= 0.):
            raise ValueError("Financing forward, discount and episode spot must be positive")
        return discount * (self.cash_support - reference_forward) / initial_spot

    def centered_physical_returns(self, *, reference_forward: float) -> np.ndarray:
        """Centered S_next/F_reference-1, in the physical units of the fitted spot loading."""
        if not np.isfinite(reference_forward) or reference_forward <= 0.:
            raise ValueError("The spot-loading reference forward must be positive")
        return (self.cash_support - self.mean) / reference_forward


@dataclass(frozen=True, kw_only=True)
class GaussianScores:
    scores: np.ndarray
    cdf_probabilities: np.ndarray
    probability_floor: float
    lower_clipped: np.ndarray
    upper_clipped: np.ndarray


def gaussian_scores(cdf_probabilities: np.ndarray, *, probability_floor: float = 1e-7) -> GaussianScores:
    """Record raw PIT values and all finite-tail clipping; do not standardize their empirical law."""
    values = np.asarray(cdf_probabilities, dtype=float)
    if (not np.isfinite(values).all() or np.any((values < 0.) | (values > 1.))
            or not np.isfinite(probability_floor) or not 0. < probability_floor < .5):
        raise ValueError("Gaussian scores need CDF probabilities and a tail floor strictly inside (0,.5)")
    clipped = np.clip(values, probability_floor, 1. - probability_floor)
    return GaussianScores(scores=ndtri(clipped), cdf_probabilities=values.copy(), probability_floor=probability_floor,
        lower_clipped=values < probability_floor, upper_clipped=values > 1. - probability_floor)


def historical_gaussian_score(*, previous_source: ArbFreePricer, realized_spot: float,
                              previous_parity_forward: float, probability_floor: float = 1e-7) -> GaussianScores:
    """Use the PRECEDING fitted SANOS CDF at the next exact-session spot observation.

    The caller must enforce date/session alignment and retain source/carry
    hashes and any short-expiry extrapolation diagnostics. No next-date
    surface, fitted volatility or return normalization enters the CDF.
    """
    if (not np.isfinite([realized_spot, previous_parity_forward]).all()
            or min(realized_spot, previous_parity_forward) <= 0.):
        raise ValueError("Historical spot and its preceding option parity forward must be positive")
    cumulative = surface_digitals(previous_source, expiry_y=1. / DAYS_PER_YEAR,
        strikes=np.array([realized_spot / previous_parity_forward]), is_call=False)
    return gaussian_scores(cumulative, probability_floor=probability_floor)
