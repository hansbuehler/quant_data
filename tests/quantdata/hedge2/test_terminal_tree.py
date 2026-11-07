"""Full-horizon power optimization and independent node-flow lower bounds."""

from dataclasses import replace

import numpy as np
import pytest

from quantdata.hedge2.terminal import TerminalPaths, terminal_risk
from quantdata.hedge2.terminal_tree import (
    certify_terminal_power,
    solve_complete_tree_power,
    tree_geometry,
)


def two_step_tree() -> TerminalPaths:
    first = np.array([-1., -1., 1., 1.])
    second = np.array([-1., 1., -1., 1.])
    return TerminalPaths(gains=np.stack([first, second], axis=1)[:, :, None],
        costs=np.full((4, 2, 1), .1), features=np.stack([np.zeros(4), first], axis=1)[:, :, None],
        history_ids=np.array([[0, 1], [0, 1], [0, 2], [0, 2]]),
        live_contracts=np.array([[True], [True], [False]]),
        terminate_after_step=np.zeros((4, 2), dtype=bool),
        claim_gain=(first > 0) * second, weights=np.full(4, .25))


@pytest.mark.parametrize("initial", [0., .4, -.3])
@pytest.mark.parametrize("side", [-1, 1])
def test_terminal_power_portfolio_has_independent_small_duality_gap(initial, side) -> None:
    paths = two_step_tree()
    result = solve_complete_tree_power(paths=paths, initial_inventory=np.array([initial]),
        target_side=side, risk_aversion=10., position_cap=2.)
    assert result.certificate.objective_gap < 2e-6
    assert result.certificate.dual_lower_bound <= result.certificate.primal_upper_bound + 1e-9
    assert paths.weights @ result.certificate.density == pytest.approx(1.)
    assert result.evaluation.step_costs[:, 0] == pytest.approx(
        np.repeat(.1 * abs(result.node_positions[0, 0] - initial), 4))
    actual = terminal_risk(result.evaluation.wealth, paths.weights,
        objective="power_exponential", risk_aversion=10.)
    assert actual.value == pytest.approx(result.certificate.primal_upper_bound)


def test_flow_bound_is_valid_for_suboptimal_holdings_and_nonzero_inventory() -> None:
    paths = two_step_tree()
    initial = np.array([.4])
    optimum = solve_complete_tree_power(paths=paths, initial_inventory=initial,
        target_side=1, risk_aversion=10., position_cap=2.)
    bad = np.array([[.7], [-.6], [.3]])
    certificate = certify_terminal_power(paths=paths, initial_inventory=initial,
        node_positions=bad, target_side=1, risk_aversion=10., position_cap=2.)
    assert certificate.dual_lower_bound <= optimum.certificate.primal_upper_bound + 1e-8
    assert certificate.primal_upper_bound > optimum.certificate.primal_upper_bound + .1


def test_settlement_and_knockout_do_not_create_turnover_edges() -> None:
    paths = two_step_tree()
    stops = np.zeros((4, 2), dtype=bool)
    stops[:2, 0] = True
    paths = replace(paths, terminate_after_step=stops)
    geometry = tree_geometry(paths, np.array([.4]))
    np.testing.assert_array_equal(geometry.active[:, 0], [True, False, True])
    np.testing.assert_array_equal(geometry.turnover.toarray(), [[1., 0., 0.], [0., 0., 0.], [-1., 0., 1.]])
    result = solve_complete_tree_power(paths=paths, initial_inventory=np.array([.4]),
        target_side=1, risk_aversion=10., position_cap=2.)
    assert result.node_positions[1, 0] == 0.
    assert result.evaluation.step_costs[:2, 1].sum() == 0.
    assert result.certificate.objective_gap < 2e-6


def test_free_spot_can_replicate_terminal_claim_exactly_without_final_exit_fee() -> None:
    paths = replace(two_step_tree(), costs=np.zeros((4, 2, 1)))
    result = solve_complete_tree_power(paths=paths, initial_inventory=np.zeros(1),
        target_side=1, risk_aversion=10., position_cap=2.)
    np.testing.assert_allclose(result.evaluation.wealth, 0., atol=2e-5)
    assert result.evaluation.entry_cost.sum() == 0.
    assert result.certificate.objective_gap < 2e-6


def test_intermediate_settlement_clears_only_expired_inventory_without_another_fee() -> None:
    original = two_step_tree()
    gains = np.repeat(original.gains, 2, axis=2)
    gains[:, 1, 0] = 0.
    paths = replace(original, gains=gains, costs=np.full_like(gains, .1),
        live_contracts=np.array([[True, True], [False, True], [False, False]]))
    initial = np.array([.4, -.2])
    geometry = tree_geometry(paths, initial)
    assert not geometry.active[1:, 0].any()
    np.testing.assert_array_equal(geometry.turnover.toarray()[2, :], 0.)
    result = solve_complete_tree_power(paths=paths, initial_inventory=initial,
        target_side=1, risk_aversion=10., position_cap=2.)
    np.testing.assert_array_equal(result.evaluation.positions[:, 1, 0], 0.)
    live_turnover = .1 * abs(result.evaluation.positions[:, 1, 1] - result.evaluation.positions[:, 0, 1])
    np.testing.assert_allclose(result.evaluation.step_costs[:, 1], live_turnover)
    assert result.certified


def test_nonuniform_probabilities_and_causal_state_dependent_costs_keep_valid_bound() -> None:
    original = two_step_tree()
    costs = original.costs.copy()
    costs[:2, 1] = .2
    costs[2:, 1] = .03
    paths = replace(original, weights=np.array([.07, .63, .18, .12]), costs=costs)
    result = solve_complete_tree_power(paths=paths, initial_inventory=np.array([.4]),
        target_side=-1, risk_aversion=10., position_cap=2.)
    assert result.certified
    assert result.certificate.objective_gap < 1e-6
    assert paths.weights @ result.certificate.density == pytest.approx(1.)


def test_finite_candidate_with_a_loose_certificate_is_not_labeled_certified(monkeypatch) -> None:
    original = certify_terminal_power

    def weaker_bound(**kwargs):
        certificate = original(**kwargs)
        return replace(certificate, dual_lower_bound=certificate.dual_lower_bound - .1,
            objective_gap=certificate.objective_gap + .1)

    monkeypatch.setattr("quantdata.hedge2.terminal_tree.certify_terminal_power", weaker_bound)
    result = solve_complete_tree_power(paths=two_step_tree(), initial_inventory=np.array([.4]),
        target_side=1, risk_aversion=10., position_cap=2.)
    assert not result.certified
    assert result.certificate.objective_gap >= .1 - 1e-8
    assert np.isfinite(result.node_positions).all()
