"""Direct OCE refinement preserves feasibility and cannot worsen a candidate."""

import numpy as np

from quantdata.hedge2.polish import polish_power


def test_power_polish_recovers_a_simple_replicating_position() -> None:
    gains = np.array([[-.02], [-.01], [.01], [.02]])
    result = polish_power(instrument_gains=gains, claim_gains=2. * gains[:, 0],
                           weights=np.full(4, .25), costs=np.zeros(1), initial_positions=np.array([1.5]),
                           risk_aversion=1000., target_side=-1)
    assert result.after_objective <= result.before_objective + 1e-12
    assert abs(result.positions[0] - 2.) < 1e-4
    assert result.solver_success


def test_power_polish_uses_turnover_from_prior_instead_of_gross_positions() -> None:
    gains = np.array([[-.02], [-.01], [.01], [.02]])
    result = polish_power(instrument_gains=gains, claim_gains=2. * gains[:, 0],
        weights=np.full(4, .25), costs=np.array([.1]), initial_positions=np.array([0.]),
        previous_positions=np.array([2.]), risk_aversion=1., target_side=-1)
    assert abs(result.positions[0] - 2.) < 1e-6
    assert abs(result.after_objective) < 1e-8


def test_cost_scaling_handles_zero_risk_columns_without_changing_economics() -> None:
    gains = np.c_[np.array([-.02, -.01, .01, .02]), np.zeros(4)]
    result = polish_power(instrument_gains=gains, claim_gains=2. * gains[:, 0],
        weights=np.full(4, .25), costs=np.array([0., 1e-5]), initial_positions=np.array([0., 0.]),
        previous_positions=np.array([0., 400.]), risk_aversion=1000., target_side=-1)
    assert abs(result.positions[0] - 2.) < 1e-4
    assert abs(result.positions[1] - 400.) < 1e-4
    assert result.after_objective < 1e-8
