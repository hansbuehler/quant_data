"""Full-terminal wealth differs from local mark risk and preserves predictability."""

from dataclasses import replace

import numpy as np
import pytest

from quantdata.hedge2.terminal import (
    TerminalPaths,
    center_finite_tree_gains,
    evaluate_terminal_policy,
    terminal_risk,
)


def binary_paths() -> TerminalPaths:
    first = np.array([-1., -1., 1., 1.])
    second = np.array([-1., 1., -1., 1.])
    return TerminalPaths(gains=np.stack([first, second], axis=1)[:, :, None],
        costs=np.full((4, 2, 1), .1), features=np.stack([np.zeros(4), first], axis=1)[:, :, None],
        history_ids=np.array([[0, 1], [0, 1], [0, 2], [0, 2]]), live_contracts=np.array([[True], [True], [False]]),
        terminate_after_step=np.zeros((4, 2), dtype=bool), claim_gain=(first > 0) * second, weights=np.full(4, .25))


def test_terminal_optimizer_can_preposition_against_future_cost_risk() -> None:
    paths = binary_paths()
    actions = np.linspace(0., .1, 101)
    objectives = []
    for action in actions:
        def policy(features, inventory, step):
            return np.full_like(inventory, action) if step == 0 else -(features > 0).astype(float)
        result = evaluate_terminal_policy(paths=paths, initial_inventory=np.zeros(1), policy=policy, target_side=1)
        objective = terminal_risk(result.wealth, paths.weights, objective="mean_variance", risk_aversion=10.)
        objectives.append(objective.value)
        expected = .05 + .2 * action + 5. * (action - .05)**2
        assert objective.value == pytest.approx(expected, abs=1e-12)
    assert actions[np.argmin(objectives)] == pytest.approx(.03)
    assert objectives[30] < objectives[0]
    assert np.all(np.einsum("p,ptn->tn", paths.weights, paths.gains) == 0.)


def test_history_conditioned_centering_is_not_global_path_demeaning() -> None:
    paths = binary_paths()
    raw = paths.gains.copy()
    raw[:, 1, 0] += paths.features[:, 1, 0]
    centered, removed = center_finite_tree_gains(raw, paths.weights, paths.history_ids)
    np.testing.assert_allclose(centered, paths.gains)
    np.testing.assert_allclose(removed[:, 1, 0], paths.features[:, 1, 0])
    assert np.mean(raw[:, 1, 0]) == 0.
    assert np.mean(raw[:2, 1, 0]) == -1.


def test_knockout_clears_after_gain_without_exit_fee_or_future_trading() -> None:
    paths = binary_paths()
    stopped = np.zeros((4, 2), dtype=bool)
    stopped[:2, 0] = True
    paths = replace(paths, terminate_after_step=stopped)
    result = evaluate_terminal_policy(paths=paths, initial_inventory=np.zeros(1),
        policy=lambda features, inventory, step: np.ones_like(inventory), target_side=1)
    np.testing.assert_array_equal(result.positions[:2, 1], 0.)
    np.testing.assert_array_equal(result.step_gains[:2, 0], -1.)
    np.testing.assert_array_equal(result.step_gains[:2, 1], 0.)
    np.testing.assert_array_equal(result.step_costs[:, 1], 0.)
    np.testing.assert_allclose(result.entry_cost, .1)


def test_future_information_cannot_enter_features_or_shared_history_actions() -> None:
    paths = binary_paths()
    leaked = paths.features.copy()
    leaked[:, 0, 0] = paths.gains[:, 1, 0]
    with pytest.raises(ValueError, match="Observable features differ"):
        replace(paths, features=leaked)
    with pytest.raises(ValueError, match="anticipates"):
        evaluate_terminal_policy(paths=paths, initial_inventory=np.zeros(1), target_side=1,
            policy=lambda features, inventory, step: np.arange(4.)[:, None])
    future_costs = paths.costs.copy()
    future_costs[:, 0, 0] += .01 * paths.gains[:, 1, 0]
    with pytest.raises(ValueError, match="not yet observed"):
        replace(paths, costs=future_costs)


def test_information_histories_cannot_merge_after_different_observations() -> None:
    paths = binary_paths()
    with pytest.raises(ValueError, match="cannot merge"):
        replace(paths, gains=np.repeat(paths.gains[:, :1], 3, axis=1), costs=np.full((4, 3, 1), .1),
            features=np.zeros((4, 3, 1)), live_contracts=np.ones((4, 1), dtype=bool),
            history_ids=np.c_[paths.history_ids, np.zeros(4, dtype=int)],
            terminate_after_step=np.zeros((4, 3), dtype=bool))


@pytest.mark.parametrize("objective, risk_aversion", [("mean_variance", 10.), ("mean_variance", np.inf), ("power_exponential", 10.)])
def test_terminal_risk_gradient_matches_full_wealth_finite_differences(objective, risk_aversion) -> None:
    wealth = np.array([-.13, -.02, .04, .2])
    weights = np.array([.1, .2, .3, .4])
    result = terminal_risk(wealth, weights, objective=objective, risk_aversion=risk_aversion)
    for index in range(4):
        bump = np.zeros(4)
        bump[index] = 1e-6
        upper = terminal_risk(wealth + bump, weights, objective=objective, risk_aversion=risk_aversion).value
        lower = terminal_risk(wealth - bump, weights, objective=objective, risk_aversion=risk_aversion).value
        assert result.wealth_gradient[index] == pytest.approx((upper - lower) / 2e-6, abs=1e-8)
