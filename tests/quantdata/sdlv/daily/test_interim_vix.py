from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quantdata.sdlv.daily.interim_vix import reprice_saved_strip
from quantdata.sdlv.daily.vix_comparison import log_contract_total_variance, mixture_prices
from quantdata.vixdata.formula import discrete_total_variance, select_vix_strip


def _example() -> tuple:
    support = np.array([.8, 1., 1.2])
    density = np.array([.25, .5, .25])
    strikes = np.array([.8, .95, 1., 1.1, 1.4]) * 6000
    calls, puts = mixture_prices(support, density, strikes / 6000, business_days=20)
    calls, puts = calls * 6000 * .99, puts * 6000 * .99
    strip = select_vix_strip(strikes=strikes, call_bids=calls, call_asks=calls,
                            put_bids=puts, put_asks=puts, discount_factor=.99)
    term = dict(selected_strikes=len(strip.strikes), forward=strip.forward, k0=strip.k0,
                discount_factor=.99, put_zero_bid_stop=None, call_zero_bid_stop=None,
                market_total_variance=discrete_total_variance(strip, discount_factor=.99), business_days=20)
    contracts = pl.DataFrame(dict(strike=strip.strikes, delta_strike=strip.delta_strikes, market_price=strip.prices))
    tower = np.broadcast_to(density, (25, 3))
    return term, contracts, support, tower


def test_saved_strip_repricing_matches_original_formula_and_analytic_integral() -> None:
    term, contracts, support, tower = _example()
    discrete, analytic = reprice_saved_strip(term, contracts, support, tower)
    assert discrete == pytest.approx(term["market_total_variance"], abs=1e-12)
    assert analytic == pytest.approx(log_contract_total_variance(support, tower[0], business_time_years=20 / 252))


def test_saved_strip_rejects_wrong_quadrature_or_market_values() -> None:
    term, contracts, support, tower = _example()
    with pytest.raises(ValueError, match="deltaK"):
        reprice_saved_strip(term, contracts.with_columns(pl.col("delta_strike") * 2), support, tower)
    with pytest.raises(ValueError, match="audited total variance"):
        reprice_saved_strip({**term, "market_total_variance": .1}, contracts, support, tower)


def test_saved_strip_does_not_extend_the_model_horizon() -> None:
    term, contracts, support, tower = _example()
    assert reprice_saved_strip({**term, "business_days": 26}, contracts, support, tower) == (None, None)
