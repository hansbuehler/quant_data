"""A locally stable fitted policy cannot be accepted below a feasible reference."""

import json
from dataclasses import replace

import numpy as np
import pytest

from quantdata.hedge2.terminal import terminal_risk
from quantdata.hedge2.terminal_adequacy import assess_zero_action_reference
from tests.quantdata.hedge2.test_terminal_tree import two_step_tree


@pytest.mark.parametrize("objective,aversion", (("mean_variance", 10.), ("mean_variance", np.inf), ("power_exponential", 10.)))
def test_zero_reference_charges_initial_liquidation_once_and_no_terminal_exit(objective: str, aversion: float) -> None:
    paths = two_step_tree()
    paths = replace(paths, costs=np.full_like(paths.costs, .2))
    inventory = np.array([.3])
    reference = terminal_risk(-paths.claim_gain - .06, paths.weights, objective=objective, risk_aversion=aversion)
    result = assess_zero_action_reference(paths=paths, initial_inventory=inventory,
        fitted_risk=reference.value, objective=objective, risk_aversion=aversion, target_side=-1, precision=64)
    assert result["zero_action_mean_entry_cost"] == pytest.approx(.06)
    assert result["zero_action_risk"] == pytest.approx(reference.value)
    assert result["no_worse_than_zero_reference"]


def test_dominated_candidate_is_recorded_without_modifying_cash_arrays() -> None:
    paths = two_step_tree()
    original_costs = paths.costs.copy()
    result = assess_zero_action_reference(paths=paths, initial_inventory=np.zeros(1),
        fitted_risk=100., objective="mean_variance", risk_aversion=10., target_side=-1, precision=32)
    assert not result["no_worse_than_zero_reference"]
    assert result["risk_excess_over_zero"] > result["risk_comparison_tolerance"]
    np.testing.assert_array_equal(paths.costs, original_costs)


def test_precision_tolerance_is_recorded_and_does_not_relax_training_stopping_rules() -> None:
    paths = two_step_tree()
    risk = terminal_risk(-paths.claim_gain, paths.weights, objective="mean_variance", risk_aversion=10.)
    arguments = dict(paths=paths, initial_inventory=np.zeros(1), fitted_risk=risk.value + 1e-7,
        objective="mean_variance", risk_aversion=10., target_side=-1)
    assert assess_zero_action_reference(**arguments, precision=32)["no_worse_than_zero_reference"]
    assert not assess_zero_action_reference(**arguments, precision=64)["no_worse_than_zero_reference"]


@pytest.mark.parametrize("precision", (32, 64))
def test_reference_metadata_is_json_serializable_without_changing_the_precision_rule(precision: int) -> None:
    paths = two_step_tree()
    fitted_risk = np.float32(1.)
    result = assess_zero_action_reference(paths=paths, initial_inventory=np.zeros(1), fitted_risk=fitted_risk,
        objective="mean_variance", risk_aversion=10., target_side=-1, precision=precision)
    decoded = json.loads(json.dumps(result, allow_nan=False))
    assert decoded == result
    expected = 256. * np.finfo(np.float32 if precision == 32 else np.float64).eps * max(
        1., abs(fitted_risk), abs(result["zero_action_risk"]))
    assert result["risk_comparison_tolerance"] == float(expected)
    assert result["no_worse_than_zero_reference"] == bool(fitted_risk <= result["zero_action_risk"] + expected)


@pytest.mark.parametrize("precision,risk", ((16, 1.), (64, np.nan)))
def test_invalid_reference_inputs_are_not_accepted(precision: int, risk: float) -> None:
    with pytest.raises(ValueError, match="finite|precision"):
        assess_zero_action_reference(paths=two_step_tree(), initial_inventory=np.zeros(1),
            fitted_risk=risk, objective="mean_variance", risk_aversion=10., target_side=-1, precision=precision)
