"""Parametric objectives retain covariance, Gaussian tails and their known special cases."""

import numpy as np
import pytest

from quantdata.hedge2.parametric import (
    ParametricOptimizer,
    gaussian_power_risk,
    gaussian_power_risk_quadrature,
)


def test_sigma_points_reproduce_full_covariance() -> None:
    exposures = np.array([[1., .2], [.5, 1.]])
    covariance = np.array([[.04, .015], [.015, .09]])
    optimizer = ParametricOptimizer(exposures=exposures, covariance=covariance, costs=np.zeros(2))
    np.testing.assert_allclose(optimizer.points.T @ optimizer.points / len(optimizer.points), covariance, atol=1e-15)
    target = np.array([2., 1.])
    for objective in ("mean_variance", "power_exponential"):
        result = optimizer.solve(target_exposures=target, objective=objective, risk_aversion=10.)
        np.testing.assert_allclose(exposures.T @ result.hedge.positions, target, atol=1e-5)


def test_power_gaussian_objective_uses_its_own_aversion_and_converges_in_quadrature() -> None:
    optimizer = ParametricOptimizer(exposures=np.ones((1, 1)), covariance=np.array([[.04]]), costs=np.array([.01]))
    result = optimizer.solve(target_exposures=np.array([2.]), objective="power_exponential", risk_aversion=10.)
    assert 0 < result.effective_mv_aversion <= 10.
    assert result.aversion_equation_residual < 1e-5
    assert result.hedge.positions[0] < 2.
    low = gaussian_power_risk_quadrature(.2, risk_aversion=100., order=128)[0]
    high = gaussian_power_risk_quadrature(.2, risk_aversion=100., order=256)[0]
    assert low == pytest.approx(high, abs=3e-4)
    analytic = gaussian_power_risk(.2, risk_aversion=100.)[0]
    assert analytic == pytest.approx(high, abs=3e-4)
    small_risk = gaussian_power_risk(.1, risk_aversion=.01)[0]
    assert small_risk == pytest.approx(.5 * .01 * .1**2, rel=.01)


@pytest.mark.parametrize("scaled_std", [1e-7, 1e-5, .01, .1, 1., 10., 100., 1000.])
def test_analytic_normal_risk_gradient_and_risk_aversion_bound(scaled_std: float) -> None:
    risk_aversion = 10.
    std = scaled_std / risk_aversion
    risk, derivative = gaussian_power_risk(std, risk_aversion=risk_aversion)
    bump = std * 1e-4
    upper = gaussian_power_risk(std + bump, risk_aversion=risk_aversion)[0]
    lower = gaussian_power_risk(std - bump, risk_aversion=risk_aversion)[0]
    assert risk >= 0
    assert 0 < derivative / std <= risk_aversion * (1. + 1e-10)
    assert derivative == pytest.approx((upper - lower) / (2. * bump), rel=3e-4)


def test_power_root_does_not_solve_artificial_tiny_aversion_qps(monkeypatch) -> None:
    optimizer = ParametricOptimizer(exposures=np.array([[1., 0.], [.1, 1.]]),
        covariance=np.diag([.0001, .04]), costs=np.array([0., .01]))
    solve = optimizer.optimizer.solve

    def checked_solve(**kwargs):
        assert kwargs["risk_aversion"] >= 1e-6
        return solve(**kwargs)

    monkeypatch.setattr(optimizer.optimizer, "solve", checked_solve)
    result = optimizer.solve(target_exposures=np.array([2., 1.]), objective="power_exponential",
                              risk_aversion=1.)
    assert result.hedge.convex_first_order_gap_bound < 1e-5
    assert result.aversion_equation_residual < 1e-5
