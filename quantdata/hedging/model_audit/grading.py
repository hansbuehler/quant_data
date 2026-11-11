"""Explicit retrospective review tolerances, not universal model-validity tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class Rule:
    name: str
    pass_limit: float
    warn_limit: float
    transform: str = "upper"
    origin: str


RULES = {
    "cov_frob_rel": Rule(name="Relative covariance Frobenius error", pass_limit=.25, warn_limit=.50,
        origin="Working tolerances of 25% and 50% of observed covariance norm."),
    "trace_ratio": Rule(name="Model/data surface variance ratio", pass_limit=1.25, warn_limit=2., transform="factor",
        origin="Within a factor 1.25, or within a factor two, of observed total variance."),
    "subspace_angle_max": Rule(name="Maximum principal-subspace angle (degrees)", pass_limit=15., warn_limit=30.,
        origin="Working angular tolerances of 15 and 30 degrees, not confidence bounds."),
    "eig_share_err_k": Rule(name="Leading eigenvalue-share L1 error", pass_limit=.10, warn_limit=.25,
        origin="Total absolute leading-share discrepancies of 10 and 25 percentage points."),
    "spot_corr_abs_error": Rule(name="Absolute spot/PC correlation error", pass_limit=.10, warn_limit=.20,
        origin="Working correlation tolerances of 0.10 and 0.20, not significance cutoffs."),
    "forecast_ratio": Rule(name="Forecast loss / persistence loss", pass_limit=1., warn_limit=1.10,
        origin="No worse than persistence; warning allows at most 10% deterioration."),
    "fit_in_spread": Rule(name="Same-day reconstruction in-spread fraction", pass_limit=.95, warn_limit=.80, transform="lower",
        origin="Working reconstruction goals: at most 1 in 20, or 1 in 5, prices outside the spread."),
    "fit_outside_mean": Rule(name="Same-day mean outside-spread distance", pass_limit=.25, warn_limit=1.,
        origin="Working distances of one-quarter and one full quoted spread width."),
    "fit_outside_q95": Rule(name="Same-day 95th-percentile outside distance", pass_limit=1., warn_limit=2.,
        origin="Working tail distances of one and two full quoted spread widths."),
    "window_deterioration": Rule(name="Relative forecast-skill deterioration", pass_limit=1.10, warn_limit=1.25,
        origin="Working deterioration tolerances of 10% and 25%, after scaling by contemporaneous persistence risk."),
    "acf_abs_error": Rule(name="Absolute floating-IV PC ACF error", pass_limit=.10, warn_limit=.20,
        origin="Working ACF tolerances of 0.10 and 0.20; fewer than 30 observed lag pairs forces at least warning."),
    "ar_radius": Rule(name="PCA AR spectral radius", pass_limit=.99, warn_limit=1.,
        origin="Stationarity boundary is one. The .99 proximity warning corresponds to a modal half-life of about 69 sessions."),
    "invalid_rate": Rule(name="Invalid decoded path fraction", pass_limit=0., warn_limit=.01,
        origin="No invalid paths is the target; up to 1% is a numerical warning, beyond 1% a review failure."),
    "extreme_rate": Rule(name="Fraction of surfaces with IV above 300%", pass_limit=0., warn_limit=.001,
        origin="300% IV is the stated sanity threshold; any occurrence warns and more than 0.1% fails this finite-horizon screen."),
}


def grade(metric: str, value: float | None) -> str:
    """Classify a scalar under its published rule; missing evidence is warning."""
    if value is None or not np.isfinite(value):
        return "warn"
    rule = RULES[metric]
    if metric == "ar_radius":
        return "pass" if value < rule.pass_limit else "warn" if value < rule.warn_limit else "fail"
    if rule.transform == "lower":
        return "pass" if value >= rule.pass_limit else "warn" if value >= rule.warn_limit else "fail"
    if rule.transform == "factor":
        value = max(value, 1. / value) if value > 0 else np.inf
    return "pass" if value <= rule.pass_limit else "warn" if value <= rule.warn_limit else "fail"


def tail_grade(scores: dict) -> str:
    """Exact-binomial coverage plus sparse-tail caution; no false pass from low power."""
    probability = scores.get("exact_binomial_p")
    if probability is None:
        return "warn"
    if probability < .01:
        return "fail"
    if probability < .05 or scores.get("expected_exceptions", 0) < 10:
        return "warn"
    return "pass"
