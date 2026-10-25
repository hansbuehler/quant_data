"""Tests for cost-aware covariance weighting of SANOS Greeks."""

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.greeks import DlvGreeks
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_parametric.hedger import (
    solve_parametric_hedge,
)


def test_spot_action_balances_residual_risk_against_round_trip_cost() -> None:
    target = DlvGreeks(
        model_prices=np.asarray([0.5]),
        spot_deltas=np.asarray([0.4]),
        factor_vegas=np.zeros((1, 2)),
    )
    options = DlvGreeks(
        model_prices=np.asarray([0.1]),
        spot_deltas=np.asarray([0.1]),
        factor_vegas=np.asarray([[0.01, 0.02]]),
    )
    action = solve_parametric_hedge(
        target=target,
        options=options,
        shock_covariance=np.diag([0.001, 1.0, 1.0]),
        option_cost_weights=np.asarray([0.01]),
        spot_cost_weight=0.0002,
        risk_parameter=10.0,
        position_cap=100.0,
    )
    assert action.solver_status == "optimal"
    assert action.cap_hits == 0
    assert np.isclose(action.spot_action, -0.38, atol=1.0e-6)
    assert np.isclose(action.option_actions[0], 0.0, atol=1.0e-6)
    assert np.isclose(action.risk_variance, 4.0e-7, atol=1.0e-10)
