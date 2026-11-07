"""Deterministic-variance Black baselines for hedge2's path-dependent targets."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from cdxcore.bs import bs
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr, ndtri


def _check_term_structure(
    *, forward_to_ko: float, forward_to_terminal: float,
    variance_to_ko: float, variance_to_terminal: float,
    discount_to_terminal: float,
) -> float:
    values = (
        forward_to_ko, forward_to_terminal, variance_to_ko,
        variance_to_terminal, discount_to_terminal,
    )
    if not all(np.isfinite(values)):
        raise ValueError("Black term-structure inputs must be finite")
    if forward_to_ko <= 0 or forward_to_terminal <= 0 or discount_to_terminal <= 0:
        raise ValueError("Black forwards and discount factor must be positive")
    if variance_to_ko < 0 or variance_to_terminal < variance_to_ko:
        raise ValueError("Forward variance must be nonnegative")
    return variance_to_terminal - variance_to_ko


def _black_price(*, forward: np.ndarray | float, strike: float,
                 variance: float, is_call: bool) -> np.ndarray | float:
    if strike <= 0 or not np.isfinite(strike):
        raise ValueError("Strike must be finite and positive")
    return forward * bs.price(
        k=strike / forward,
        vol=np.sqrt(variance),
        is_call=is_call,
    )


def forward_start_straddle_price(
    *, strike_ratio: float, forward_to_reset: float, forward_to_terminal: float,
    variance_to_reset: float, variance_to_terminal: float,
    discount_to_terminal: float = 1.0,
) -> float:
    """Price the cash payoff ``abs(S_terminal / S_reset - strike_ratio)``.

    This pre-reset formula assumes deterministic carry and independent
    lognormal increments between the reset and terminal observation.
    """
    forward_variance = _check_term_structure(
        forward_to_ko=forward_to_reset,
        forward_to_terminal=forward_to_terminal,
        variance_to_ko=variance_to_reset,
        variance_to_terminal=variance_to_terminal,
        discount_to_terminal=discount_to_terminal,
    )
    forward_ratio = forward_to_terminal / forward_to_reset
    call = _black_price(
        forward=forward_ratio, strike=strike_ratio,
        variance=forward_variance, is_call=True,
    )
    put = _black_price(
        forward=forward_ratio, strike=strike_ratio,
        variance=forward_variance, is_call=False,
    )
    return float(discount_to_terminal * (call + put))


@lru_cache(maxsize=8)
def _quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 8:
        raise ValueError("Quadrature order must be at least eight")
    return leggauss(order)


def single_observation_ko_price(
    *, strike: float, barrier: float, is_call: bool,
    forward_to_ko: float, forward_to_terminal: float,
    variance_to_ko: float, variance_to_terminal: float,
    discount_to_terminal: float = 1.0, quadrature_order: int = 96,
) -> float:
    """Price a call down-and-out or put up-and-out, observed only at the KO date."""
    forward_variance = _check_term_structure(
        forward_to_ko=forward_to_ko,
        forward_to_terminal=forward_to_terminal,
        variance_to_ko=variance_to_ko,
        variance_to_terminal=variance_to_terminal,
        discount_to_terminal=discount_to_terminal,
    )
    if not np.isfinite(barrier) or barrier <= 0:
        raise ValueError("Barrier must be finite and positive")
    if variance_to_ko == 0:
        survives = forward_to_ko > barrier if is_call else forward_to_ko < barrier
        if not survives:
            return 0.0
        return float(discount_to_terminal * _black_price(
            forward=forward_to_terminal, strike=strike,
            variance=forward_variance, is_call=is_call,
        ))

    cutoff = (np.log(barrier / forward_to_ko) + 0.5 * variance_to_ko) / np.sqrt(variance_to_ko)
    cutoff_probability = float(ndtr(cutoff))
    lower, upper = (cutoff_probability, 1.0) if is_call else (0.0, cutoff_probability)
    if upper <= lower:
        return 0.0
    nodes, weights = _quadrature(quadrature_order)
    probabilities = lower + 0.5 * (upper - lower) * (nodes + 1.0)
    ko_spots = forward_to_ko * np.exp(
        -0.5 * variance_to_ko + np.sqrt(variance_to_ko) * ndtri(probabilities)
    )
    conditional_forward = ko_spots * (forward_to_terminal / forward_to_ko)
    conditional_prices = _black_price(
        forward=conditional_forward, strike=strike,
        variance=forward_variance, is_call=is_call,
    )
    return float(discount_to_terminal * 0.5 * (upper - lower) * np.dot(weights, conditional_prices))
