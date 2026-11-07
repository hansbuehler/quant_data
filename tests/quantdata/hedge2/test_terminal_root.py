"""Final-step oracle tests preserve sampled means, premium limits and terminal cash."""

import json

import numpy as np
import pytest

from quantdata.hedge2.objectives import power_oce
from quantdata.hedge2.terminal import TerminalPaths, terminal_risk
from quantdata.hedge2.terminal_root import solve_terminal_root


def root_paths(gains, claim_gain, *, costs=None, weights=None, initially_active=True, live=None):
    values = np.asarray(gains, dtype=float)
    count, instruments = values.shape
    return TerminalPaths(gains=values[:, None, :],
        costs=np.broadcast_to(np.zeros(instruments) if costs is None else costs, (count, 1, instruments)).copy(),
        features=np.zeros((count, 1, 1)), history_ids=np.zeros((count, 1), dtype=np.int64),
        live_contracts=np.ones((2, instruments), dtype=bool) if live is None else live,
        terminate_after_step=np.ones((count, 1), dtype=bool), claim_gain=np.asarray(claim_gain, dtype=float),
        weights=np.ones(count) if weights is None else np.asarray(weights, dtype=float), initially_active=initially_active)


def test_raw_sample_means_are_not_removed_from_the_root_objective() -> None:
    paths = root_paths([[0.], [2.]], [0., 0.])
    result = solve_terminal_root(paths=paths, root_normalized_prices=np.ones(1), initial_inventory=np.zeros(1),
        objective="mean_variance", risk_aversion=2., target_side=-1)
    assert result.certified
    assert result.contract_positions[0] == pytest.approx(.5, abs=1e-6)
    assert result.risk.value == pytest.approx(-.25, abs=1e-9)
    assert result.risk.mean_wealth == pytest.approx(.5, abs=1e-6)
    assert result.dual_lower_bound <= result.risk.value
    assert isinstance(result.certified, bool)
    assert json.loads(json.dumps({"certified": result.certified, "gap": result.objective_gap_bound}))["certified"] is True


def test_duplicate_compression_retains_original_nonuniform_sample_weights() -> None:
    duplicate = root_paths([[-1.], [-1.], [2.], [2.]], [-2., -2., 4., 4.], weights=[1., 2., 3., 4.])
    unique = root_paths([[-1.], [2.]], [-2., 4.], weights=[3., 7.])
    arguments = dict(root_normalized_prices=np.ones(1), initial_inventory=np.zeros(1),
        objective="mean_variance", risk_aversion=10., target_side=-1)
    first = solve_terminal_root(paths=duplicate, **arguments)
    second = solve_terminal_root(paths=unique, **arguments)
    assert first.certified and second.certified
    assert first.original_scenarios == 4 and first.optimization_scenarios == 2
    np.testing.assert_allclose(first.contract_positions, second.contract_positions, atol=1e-7)
    assert first.risk.value == pytest.approx(second.risk.value, abs=1e-10)


@pytest.mark.parametrize("side", [-1, 1])
def test_exact_target_replication_preserves_long_and_short_sides(side) -> None:
    paths = root_paths([[-.2], [.2]], [-.4, .4], weights=[2., 2.])
    result = solve_terminal_root(paths=paths, root_normalized_prices=np.array([.5]), initial_inventory=np.zeros(1),
        objective="mean_variance", risk_aversion=10., target_side=side)
    assert result.certified
    assert result.contract_positions[0] == pytest.approx(-2. * side, abs=1e-6)
    assert result.premium_positions[0] == pytest.approx(-side, abs=1e-6)
    np.testing.assert_allclose(result.evaluation.wealth, 0., atol=1e-7)


def test_carried_inventory_outside_today_box_pays_its_mandatory_cost() -> None:
    paths = root_paths([[0.], [0.]], [0., 0.], costs=[.02])
    result = solve_terminal_root(paths=paths, root_normalized_prices=np.array([2.]), initial_inventory=np.array([10.]),
        objective="mean_variance", risk_aversion=1., target_side=-1, premium_cap=10.)
    assert result.certified
    assert result.contract_positions[0] == pytest.approx(5., abs=1e-6)
    assert result.deterministic_mandatory_cost == pytest.approx(.1)
    np.testing.assert_allclose(result.evaluation.entry_cost, .1, atol=1e-8)
    assert result.risk.value == pytest.approx(.1, abs=1e-8)
    assert result.dual_lower_bound == pytest.approx(.1, abs=1e-7)


def test_large_mandatory_fee_cannot_hide_the_requested_small_gap() -> None:
    paths = root_paths([[-.1, 0.], [.1, 0.]], [-.1, .1], costs=[0., 1.])
    result = solve_terminal_root(paths=paths, root_normalized_prices=np.array([1., 0.]),
        initial_inventory=np.array([0., 1e10]), objective="mean_variance",
        risk_aversion=1., target_side=-1, gap_tolerance=1e-8)
    assert result.deterministic_mandatory_cost == 1e10
    assert not result.certified
    assert result.primal_upper_bound >= result.risk.value
    assert result.objective_gap_bound > 1e-8


@pytest.mark.parametrize("objective,risk", [("mean_variance", 1.), ("mean_variance", np.inf), ("power_exponential", 10.)])
def test_live_zero_price_forces_liquidation_but_records_its_fee(objective, risk) -> None:
    paths = root_paths([[-.1, .2], [.1, -.2]], [-.1, .1], costs=[0., .03])
    result = solve_terminal_root(paths=paths, root_normalized_prices=np.array([1., 0.]),
        initial_inventory=np.array([0., 7.]), objective=objective, risk_aversion=risk, target_side=-1)
    assert result.certified and result.contract_positions[1] == 0.
    assert result.deterministic_mandatory_cost == pytest.approx(.21)
    np.testing.assert_allclose(result.evaluation.entry_cost, .21, atol=1e-12)
    assert result.risk.mean_wealth == pytest.approx(-.21, abs=1e-8)
    assert result.risk.value == pytest.approx(0. if np.isinf(risk) else .21, abs=1e-7)


def test_all_zero_price_assets_are_handled_without_an_empty_solver_problem() -> None:
    paths = root_paths([[.1], [-.1]], [0., 1.], costs=[.04])
    result = solve_terminal_root(paths=paths, root_normalized_prices=np.zeros(1), initial_inventory=np.array([3.]),
        objective="mean_variance", risk_aversion=2., target_side=-1)
    assert result.certified and result.solver == "no tradable root assets"
    assert result.deterministic_mandatory_cost == pytest.approx(.12)
    np.testing.assert_allclose(result.evaluation.wealth, [-.12, -1.12])


def test_already_terminated_ko_clears_inventory_at_mid_without_new_fees() -> None:
    paths = root_paths([[.1], [-.1]], [0., 0.], costs=[.4], initially_active=False)
    result = solve_terminal_root(paths=paths, root_normalized_prices=np.ones(1), initial_inventory=np.array([20.]),
        objective="power_exponential", risk_aversion=10., target_side=-1)
    assert result.certified and result.deterministic_mandatory_cost == 0.
    np.testing.assert_array_equal(result.evaluation.entry_cost, np.zeros(2))
    np.testing.assert_array_equal(result.contract_positions, np.zeros(1))


def test_expired_initial_inventory_is_invalid_instead_of_silently_discarded() -> None:
    paths = root_paths([[0.], [0.]], [0., 0.], live=np.zeros((2, 1), dtype=bool))
    with pytest.raises(ValueError, match="inventory"):
        solve_terminal_root(paths=paths, root_normalized_prices=np.ones(1), initial_inventory=np.ones(1),
            objective="mean_variance", risk_aversion=1., target_side=-1)


def test_costs_must_be_exactly_root_deterministic_for_the_convex_oracle() -> None:
    paths = root_paths([[0.], [1.]], [0., 1.], costs=[.1])
    paths.costs[1, 0, 0] += 1e-13
    with pytest.raises(ValueError, match="identical"):
        solve_terminal_root(paths=paths, root_normalized_prices=np.ones(1), initial_inventory=np.zeros(1),
            objective="mean_variance", risk_aversion=1., target_side=-1)


def test_power_oracle_matches_an_independent_scalar_portfolio_search() -> None:
    paths = root_paths([[-.4], [.1], [.7]], [.3, -.1, -.2], costs=[.01], weights=[.2, .5, .3])
    result = solve_terminal_root(paths=paths, root_normalized_prices=np.array([.8]), initial_inventory=np.array([.4]),
        objective="power_exponential", risk_aversion=10., target_side=-1, premium_cap=1.)
    objectives = []
    for position in np.linspace(-1.25, 1.25, 501):
        wealth = -paths.claim_gain + position * paths.gains[:, 0, 0] - .01 * abs(position - .4)
        objectives.append(-power_oce(wealth, paths.weights, risk_aversion=10.).certainty_equivalent)
    assert result.certified and result.risk.value <= min(objectives) + 1e-7
    assert result.objective_gap_bound <= 1e-6
    independently_replayed = terminal_risk(result.evaluation.wealth, paths.weights,
        objective="power_exponential", risk_aversion=10.)
    assert result.risk.value == independently_replayed.value


def test_pure_variance_does_not_quietly_optimize_mean_or_deterministic_cost() -> None:
    paths = root_paths([[0.], [2.]], [0., 0.], costs=[.2])
    result = solve_terminal_root(paths=paths, root_normalized_prices=np.ones(1), initial_inventory=np.ones(1),
        objective="mean_variance", risk_aversion=np.inf, target_side=-1)
    assert result.certified and abs(result.contract_positions[0]) < 1e-5
    assert result.risk.value < 1e-10
    np.testing.assert_allclose(result.evaluation.entry_cost, .2, atol=2e-6)
    assert result.risk.mean_wealth == pytest.approx(-.2, abs=1e-5)


def test_multi_step_problem_cannot_be_substituted_by_a_root_oracle() -> None:
    paths = TerminalPaths(gains=np.zeros((2, 2, 1)), costs=np.zeros((2, 2, 1)),
        features=np.zeros((2, 2, 1)), history_ids=np.zeros((2, 2), dtype=np.int64),
        live_contracts=np.ones((3, 1), dtype=bool), terminate_after_step=np.zeros((2, 2), dtype=bool),
        claim_gain=np.zeros(2), weights=np.ones(2))
    with pytest.raises(ValueError, match="one remaining step"):
        solve_terminal_root(paths=paths, root_normalized_prices=np.ones(1), initial_inventory=np.zeros(1),
            objective="mean_variance", risk_aversion=1., target_side=-1)
