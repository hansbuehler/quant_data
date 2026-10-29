"""Deterministic checks for the SANOS Greek entry-cost cap optimizer."""

from __future__ import annotations

import numpy as np

from quantdata.hedging.sanos_greek_cost_cap import FactorCostCapSolver


def test_reoptimized_factor_hedge_respects_budget_and_spot_delta() -> None:
    solver = FactorCostCapSolver(
        option_factor_exposures=np.eye(2),
        option_spot_deltas=np.asarray([0.5, -0.5]),
        factor_covariance=np.eye(2),
        option_costs_kusd=np.ones(2),
    )
    original = -np.ones(2)
    for budget in (0.0, 1.0, 2.0):
        action, _, risk = solver.solve(
            target_factor_exposures=np.ones(2),
            target_spot_delta=0.2,
            claim_quantity=1.0,
            original_action_per_claim=original,
            budget_kusd=budget,
        )
        assert np.sum(np.abs(action)) <= budget + 1.0e-5
        assert np.isfinite(risk)
        assert abs(-0.2 - np.asarray([0.5, -0.5]) @ action) < 1_000.0
    assert np.allclose(
        solver.solve(
            target_factor_exposures=np.ones(2),
            target_spot_delta=0.2,
            claim_quantity=1.0,
            original_action_per_claim=original,
            budget_kusd=0.0,
        )[0],
        0.0,
    )


def test_partial_budget_beats_proportional_scaling_in_factor_risk() -> None:
    solver = FactorCostCapSolver(
        option_factor_exposures=np.eye(2),
        option_spot_deltas=np.zeros(2),
        factor_covariance=np.eye(2),
        option_costs_kusd=np.asarray([1.0, 2.0]),
    )
    original = -np.ones(2)
    action, status, risk = solver.solve(
        target_factor_exposures=np.ones(2),
        target_spot_delta=0.0,
        claim_quantity=1.0,
        original_action_per_claim=original,
        budget_kusd=1.0,
    )
    scaled_risk = solver.risk(original / 3.0)
    assert status.startswith("CLARABEL:")
    assert risk < scaled_risk
    assert np.asarray([1.0, 2.0]) @ np.abs(action) <= 1.0 + 1.0e-5
