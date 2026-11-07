"""Black spot and forward-variance-bucket hedges for the two exotic families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from cdxcore.bs import bs
from scipy.special import ndtr

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedge2.black_targets import forward_start_straddle_price, single_observation_ko_price
from quantdata.hedge2.cash import CashCurve, zero_strike_mask
from quantdata.hedge2.claims import Claim


@dataclass(frozen=True, kw_only=True)
class BlackExoticHedge:
    positions: np.ndarray
    price: float
    target_exposures: np.ndarray
    residual_exposures: np.ndarray
    variance_buckets: np.ndarray


def black_exotic_price(*, claim: Claim, curve: CashCurve, elapsed_day: int,
                       variance_buckets: np.ndarray, quadrature_order: int = 384) -> float:
    """Buckets are variance to observation and observation-to-terminal before reset.

    After observation only the remaining terminal variance is supplied.
    Cash strikes, barrier and an observed reset spot stay fixed.
    """
    remaining = claim.terminal_day - elapsed_day
    if claim.kind not in ("ko", "ratio_straddle") or remaining <= 0:
        raise ValueError("Black exotic pricing requires a live KO or ratio straddle")
    observation_remaining = claim.observation_day - elapsed_day
    expected = 2 if observation_remaining > 0 else 1
    if (variance_buckets.shape != (expected,) or not np.isfinite(variance_buckets).all()
            or np.any(variance_buckets < 0)):
        raise ValueError("Supply finite nonnegative Black variance buckets")
    forward, _, discount = curve.at(float(remaining))
    if observation_remaining <= 0:
        if not claim.observation_done:
            raise ValueError("Exotic pricing requires the actual observation before post-reset valuation")
        if claim.kind == "ko" and claim.knocked_out:
            return 0.
        strike = claim.strike if claim.kind == "ko" else claim.strike * claim.reset_spot
        call = bs.price(k=strike / forward, vol=np.sqrt(variance_buckets[0]),
                        is_call=claim.is_call if claim.kind == "ko" else True)
        if claim.kind == "ko":
            return float(discount * forward * call / claim.initial_spot)
        put = bs.price(k=strike / forward, vol=np.sqrt(variance_buckets[0]), is_call=False)
        return float(discount * forward * (call + put) / claim.reset_spot)
    observation_forward = float(curve.at(float(observation_remaining))[0])
    arguments = {"forward_to_terminal": float(forward), "discount_to_terminal": float(discount),
                 "variance_to_terminal": float(variance_buckets.sum())}
    if claim.kind == "ko":
        return single_observation_ko_price(strike=claim.strike, barrier=claim.barrier,
            is_call=claim.is_call, forward_to_ko=observation_forward,
            variance_to_ko=float(variance_buckets[0]), quadrature_order=quadrature_order, **arguments) / claim.initial_spot
    return forward_start_straddle_price(strike_ratio=claim.strike, forward_to_reset=observation_forward,
                                        variance_to_reset=float(variance_buckets[0]), **arguments)


def _bracket(book: pd.DataFrame, *, tenor: int, strike: float) -> tuple[np.ndarray, np.ndarray]:
    ordinary = ~zero_strike_mask(book)
    indices = np.flatnonzero(ordinary & np.isclose(book["bdte"].to_numpy(dtype=float), tenor, rtol=0., atol=1e-6))
    order = np.argsort(book.iloc[indices]["strike"].to_numpy(dtype=float))
    indices = indices[order]
    strikes = book.iloc[indices]["strike"].to_numpy(dtype=float)
    lower, upper = np.flatnonzero(strikes <= strike), np.flatnonzero(strikes >= strike)
    if not len(lower) or not len(upper):
        raise ValueError(f"Black exotic hedge needs observed strikes surrounding {strike} at {tenor}D")
    left, right = int(lower[-1]), int(upper[0])
    if left == right:
        return indices[[left]], np.ones(1)
    weight = (strikes[right] - strike) / (strikes[right] - strikes[left])
    return indices[[left, right]], np.array([weight, 1. - weight])


def black_exotic_hedge(*, claim: Claim, curve: CashCurve, book: pd.DataFrame,
                       option_vols: np.ndarray, total_variances: dict[int, float],
                       target_side: int, elapsed_day: int = 0,
                       spot_bump_fraction: float = .01, variance_bump_fraction: float = .001,
                       quadrature_order: int = 384) -> BlackExoticHedge:
    if (target_side not in (-1, 1) or option_vols.shape != (len(book),)
            or not np.isfinite(option_vols).all() or np.any(option_vols < 0)
            or not 0 < spot_bump_fraction < 1 or not 0 < variance_bump_fraction < 1):
        raise ValueError("Black exotic hedges require aligned positive vols, a side and spot bump")
    remaining = claim.terminal_day - elapsed_day
    observation = claim.observation_day - elapsed_day
    terminal_variance = total_variances[remaining]
    if observation > 0:
        first = total_variances[observation]
        buckets = np.array([first, terminal_variance - first])
        tenors = (observation, remaining)
    else:
        buckets = np.array([terminal_variance])
        tenors = (remaining,)
    arguments = dict(claim=claim, elapsed_day=elapsed_day, variance_buckets=buckets,
                     quadrature_order=quadrature_order)
    price = black_exotic_price(curve=curve, **arguments)
    positive = black_exotic_price(curve=curve.at_spot(curve.spot * (1. + spot_bump_fraction)), **arguments)
    negative = black_exotic_price(curve=curve.at_spot(curve.spot * (1. - spot_bump_fraction)), **arguments)
    target = [(positive - negative) / (2. * spot_bump_fraction)]
    for index, variance in enumerate(buckets):
        bump = max(float(variance) * variance_bump_fraction, 1e-9)
        upper, lower = buckets.copy(), buckets.copy()
        upper[index] += bump
        lower[index] = max(float(variance) - bump, 0.)
        high = black_exotic_price(curve=curve, **(arguments | {"variance_buckets": upper}))
        low = black_exotic_price(curve=curve, **(arguments | {"variance_buckets": lower}))
        target.append((high - low) / (upper[index] - lower[index]))
    target = np.asarray(target)
    exposures = np.zeros((1 + len(book), 1 + len(buckets)))
    exposures[0, 0] = curve.spot / claim.initial_spot
    combinations, bucket_columns = [], []
    bracketing_strike = claim.strike if claim.kind == "ko" else (
        claim.strike * (claim.reset_spot if claim.reset_spot is not None else curve.spot))
    for tenor in tenors:
        indices, weights = _bracket(book, tenor=tenor, strike=bracketing_strike)
        if np.any(option_vols[indices] <= 0.):
            raise ValueError("The bracketing Black options have zero implied variance")
        forward, _, discount = curve.at(float(tenor))
        sigma = option_vols[indices] * np.sqrt(tenor / DAYS_PER_YEAR)
        strikes = book.iloc[indices]["strike"].to_numpy(dtype=float)
        d1 = np.log(forward / strikes) / sigma + .5 * sigma
        calls = (book.iloc[indices]["option_type"] == "call").to_numpy()
        delta = discount * forward * (ndtr(d1) - (~calls).astype(float)) / claim.initial_spot
        variance_vega = discount * forward * np.exp(-.5 * d1**2) / (
            np.sqrt(2. * np.pi) * 2. * sigma * claim.initial_spot)
        columns = np.zeros((len(indices), len(buckets)))
        columns[:, 0] = variance_vega
        if len(buckets) == 2 and tenor == remaining:
            columns[:, 1] = variance_vega
        exposures[1 + indices, 0] = delta
        exposures[1 + indices, 1:] = columns
        combinations.append((indices, weights))
        bucket_columns.append(weights @ columns)
    matrix = np.asarray(bucket_columns).T
    if np.linalg.matrix_rank(matrix) != len(buckets):
        raise ValueError("The bracketing Black options do not span the live variance buckets")
    sizes = np.linalg.solve(matrix, -target_side * target[1:])
    positions = np.zeros(1 + len(book))
    for size, (indices, weights) in zip(sizes, combinations):
        positions[1 + indices] += size * weights
    positions[0] = (-target_side * target[0] - positions @ exposures[:, 0]) / exposures[0, 0]
    residual = target_side * target + positions @ exposures
    return BlackExoticHedge(positions=positions, price=price, target_exposures=target,
                            residual_exposures=residual, variance_buckets=buckets)
