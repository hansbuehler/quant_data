"""Conditional endpoint probabilities, unchanged cash contracts, and outer errors."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from quantdata.sdlv.dhgromm1.conditional_oce import pair_statistic, summarize, weighted_cash
from quantdata.sdlv.dhgromm1.conditional_paths import (
    MarketPaths,
    absorbed_paths,
    black_calls,
    boundary_probabilities,
    interior_normals,
)
from quantdata.sdlv.dhgromm1.training import _economic_utility, optimize_translation


def test_conditioned_gaussian_changes_the_shared_spot_factor_not_the_other_factors():
    front = np.tile([.1,.2,.4,.3],(6,1))
    lower,upper,inside = boundary_probabilities(front)
    np.testing.assert_allclose([lower[0],upper[0],inside[0]],[.1,.3,.6])
    z = np.arange(48).reshape(6,8)/20-1
    shifted = interior_normals(z,lower,inside)
    np.testing.assert_allclose(ndtr(shifted[:,0]),lower+inside*ndtr(z[:,0]),atol=1e-15)
    np.testing.assert_array_equal(shifted[:,1:],z[:,1:])
    assert np.all((ndtr(shifted[:,0])>.1)&(ndtr(shifted[:,0])<.7))


def test_survival_and_both_endpoint_weights_telescope_without_discarding_tiny_tails():
    survival,total = np.ones(4),np.zeros(4)
    for day in range(20):
        lower,upper = np.full(4,1e-14),np.linspace(.001,.01,4)
        total += survival*(lower+upper)
        survival *= 1-lower-upper
    np.testing.assert_allclose(total+survival,1,atol=2e-15)


def test_absorbed_paths_keep_prefix_causal_and_reproduce_floating_payoff_identities():
    prefix = MarketPaths.empty(2)
    # A pure endpoint law is Black at every date, so no DLV simulation needed.
    prefix.observe(0,np.ones(2))
    branch = absorbed_paths(prefix,day=0,background=np.ones(2),endpoints=np.array([.5,1.5]),
        normals=np.zeros((2,2,20)))
    price,returns,h = branch.arrays()
    assert price.shape==(8,21,21)
    np.testing.assert_array_equal(price[:,0],np.repeat(prefix.prices[:,0],4,axis=0))
    np.testing.assert_allclose(returns[:,:,0],price[:,20,0,None]/price[:,:20,0]-1)
    premium = 1+(2*price[:,10,1]-price[:,10,17])/price[:,10,0]
    np.testing.assert_allclose(2*returns[:,10,1]-returns[:,10,17],h-premium,atol=4e-16)
    # Last-day long-tenor options are valued at day 20, NOT fictitiously expired.
    expected = black_calls(price[:,20,0],price[:,19,0]*np.exp(-2*.2*np.sqrt(6/252)),5)
    np.testing.assert_allclose(returns[:,19,12],(expected-price[:,19,12])/price[:,19,0])
    np.testing.assert_allclose(price[:,:20,17:]/price[:,:20,:1],1)


def test_global_weighted_cash_matches_plain_oce_and_does_not_profile_each_branch():
    x = np.array([-.3,-.01,.01,.03,.04,.2])
    assert weighted_cash(x,np.ones(6))==pytest.approx(optimize_translation(x,100.),abs=1e-13)
    assert weighted_cash(np.ones(3)*.123,np.array([.2,.3,.5]))==pytest.approx(.123)
    c = weighted_cash(np.array([-.2,.2]),np.array([.1,.9]))
    score = np.dot([.1,.9],_economic_utility(np.array([-.2,.2]),c,100.))
    assert score < .1*(-.2)+.9*.2  # Jensen: never utility of mean wealth.


def test_oce_error_clusters_all_branches_within_the_outer_antithetic_pair():
    base = np.array([-.01,.01,.02,.04])
    outer = np.repeat(np.arange(4),3)
    net = np.repeat(base,3)[:,None]*np.ones((1,6))
    weights = np.tile([.2,.3,.5],4)
    report,retained = summarize(net,np.zeros_like(net),weights,outer,outer_paths=4,
                               training_cash=np.zeros(6),payoff_mean=.02)
    expected = pair_statistic(_economic_utility(base,0.,100.))
    actual = report["fixed_training_cash"]["dh"]["statistics"]["empty"]
    assert actual["oce"]==pytest.approx(expected["oce"])
    assert actual["three_standard_errors"]==pytest.approx(expected["three_standard_errors"])
    assert actual["independent_outer_pairs"]==2
    assert retained["profiled_cash_outer_utilities"].shape==(4,6)
    assert report["profiled_cash"]["dh"]["long_minus_empty"]["oce"]==pytest.approx(0,abs=1e-14)
