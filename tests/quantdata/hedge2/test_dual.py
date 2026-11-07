"""Power-OCE dual densities provide genuine objective-unit certificates."""

import numpy as np
import pytest

from quantdata.hedge2.dual import certify_mean_variance, certify_power
from quantdata.hedge2.optimization import ScenarioOptimizer


def test_power_dual_certifies_known_one_asset_optimum() -> None:
    gains = np.array([[-.02], [-.01], [.01], [.02]])
    weights = np.full(4, .25)
    costs = np.array([.0001])
    optimizer = ScenarioOptimizer(instrument_gains=gains, weights=weights, costs=costs)
    hedge = optimizer.solve(claim_gains=2. * gains[:, 0], objective="power_exponential", risk_aversion=1000.)
    certificate = certify_power(instrument_gains=gains, claim_gains=2. * gains[:, 0], weights=weights,
                                 costs=costs, positions=hedge.positions, risk_aversion=1000.,
                                 target_side=-1, gap_tolerance=1e-7)
    assert certificate.objective_gap < 1e-7
    assert certificate.dual_lower_bound <= certificate.primal_upper_bound + 1e-9
    assert abs(weights @ certificate.density - 1.) < 1e-14
    assert certificate.density.min() >= 0.


@pytest.mark.parametrize("risk_aversion", [1., 1000., np.inf])
def test_quadratic_dual_with_turnover_and_nonzero_target_mean(risk_aversion: float) -> None:
    gains = np.array([[-.02], [-.01], [.01], [.02]])
    weights = np.array([.2, .3, .3, .2])
    costs = np.array([.0001])
    previous = np.array([.75])
    target = 2. * gains[:, 0] + .3
    optimizer = ScenarioOptimizer(instrument_gains=gains, weights=weights, costs=costs)
    hedge = optimizer.solve(claim_gains=target, objective="mean_variance", risk_aversion=risk_aversion,
                            previous_positions=previous)
    certificate = certify_mean_variance(instrument_gains=gains, claim_gains=target, weights=weights,
        costs=costs, positions=hedge.positions, risk_aversion=risk_aversion, target_side=-1,
        previous_positions=previous, gap_tolerance=1e-8)
    assert certificate.primal_upper_bound == pytest.approx(hedge.training_objective, abs=1e-12)
    assert certificate.objective_gap < 1e-8
    assert abs(weights @ certificate.dual_variable) < 1e-12
    assert certificate.dual_lower_bound <= certificate.primal_upper_bound + 1e-10


def test_quadratic_certificate_does_not_accept_a_bad_portfolio() -> None:
    gains = np.array([[-.02], [-.01], [.01], [.02]])
    certificate = certify_mean_variance(instrument_gains=gains, claim_gains=2. * gains[:, 0],
        weights=np.full(4, .25), costs=np.array([.0001]), positions=np.zeros(1),
        risk_aversion=1000., target_side=-1, gap_tolerance=1e-8)
    assert certificate.objective_gap > .1
    assert certificate.dual_lower_bound > 0.
