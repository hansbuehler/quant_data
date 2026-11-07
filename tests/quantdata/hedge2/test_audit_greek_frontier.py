"""Recorded infeasibility requires raw support or actual-inventory cost evidence."""

import copy
from dataclasses import asdict

import numpy as np
import pytest

from quantdata.hedge2.audit_greek_frontier import verify_failed_prefix, verify_unavailable_geometry
from quantdata.hedge2.greek_feasibility import certify_greek_box


def _box(side: int = -1):
    prices = np.array([1., .2, .2])
    exposures = np.array([[1., 0.], [0., .5], [0., .5]])
    target = np.array([2., 12.])
    certificate = certify_greek_box(exposures=exposures, target_exposures=target,
        target_side=side, position_cap=10.)
    values = asdict(certificate)
    evidence = {"normalized_premium_prices": prices, "costs": np.array([0., .001, .001]),
        "previous_contract_positions": np.zeros(3), "instrument_exposures": exposures * prices[:, None],
        "target_exposures": target, "normalized_separator": values.pop("normalized_separator"),
        "primal_positions": values.pop("primal_positions")}
    context = {"side": side, "greek_budget": 1., "error_type": "GreekBoxInfeasible",
        "executed_at_failure": False, "terminal_performance_available": False, "box_certificate": values}
    return context, evidence


def _budget(side: int = -1):
    evidence = {"normalized_premium_prices": np.array([1., .1]), "costs": np.array([0., .01]),
        "previous_contract_positions": np.array([0., 200.]),
        "instrument_exposures": np.array([[1., 0.], [.5, .1]]), "target_exposures": np.array([2., 1.])}
    full_cost = 1.9 if side == -1 else 2.1
    context = {"side": side, "greek_budget": .2, "error_type": "GreekBudgetInfeasible",
        "executed_at_failure": False, "terminal_performance_available": False,
        "full_entry_cost": full_cost, "mandatory_turnover_cost": 1., "turnover_budget": .2 * full_cost}
    return context, evidence


@pytest.mark.parametrize("side", [-1, 1])
def test_raw_separator_proves_box_infeasibility_for_both_sides(side) -> None:
    context, evidence = _box(side)
    result = verify_unavailable_geometry(context=context, evidence=evidence, premium_cap=10.)
    assert result["reason"] == "exact_reference_outside_box"
    assert result["minimum_cap_lower_bound"] == pytest.approx(12.)
    assert result["separation_margin"] > result["matching_tolerance_allowance"]


@pytest.mark.parametrize("side", [-1, 1])
def test_actual_inventory_and_independent_cost_witness_prove_budget_infeasibility(side) -> None:
    context, evidence = _budget(side)
    result = verify_unavailable_geometry(context=context, evidence=evidence, premium_cap=10.)
    assert result["reason"] == "mandatory_turnover_exceeds_budget"
    assert result["mandatory_turnover_cost"] == pytest.approx(1.)
    assert result["independent_full_cost_upper_bound"] == pytest.approx(context["full_entry_cost"])
    assert result["budget_shortfall"] > 0.


@pytest.mark.parametrize("field,value", [("mandatory_turnover_cost", .9), ("turnover_budget", .5),
    ("full_entry_cost", float("nan")), ("executed_at_failure", True), ("terminal_performance_available", True)])
def test_invalid_or_fabricated_budget_metadata_is_rejected(field, value) -> None:
    context, evidence = _budget()
    context[field] = value
    with pytest.raises((ValueError, RuntimeError)):
        verify_unavailable_geometry(context=context, evidence=evidence, premium_cap=10.)


def test_feasible_budget_or_solver_string_alone_is_not_an_infeasibility_certificate() -> None:
    context, evidence = _budget()
    context.update(greek_budget=.9, turnover_budget=.9 * context["full_entry_cost"])
    with pytest.raises(RuntimeError, match="does not certify"):
        verify_unavailable_geometry(context=context, evidence=evidence, premium_cap=10.)
    context["error_type"] = "RuntimeError"
    with pytest.raises(ValueError, match="Generic numerical"):
        verify_unavailable_geometry(context=context, evidence=evidence, premium_cap=10.)


def test_bad_separator_changed_cap_and_feasible_box_cannot_pass() -> None:
    context, evidence = _box()
    for changed in (np.zeros_like(evidence["normalized_separator"]), -evidence["normalized_separator"]):
        candidate = copy.deepcopy(evidence)
        candidate["normalized_separator"] = changed
        with pytest.raises(RuntimeError, match="does not independently certify"):
            verify_unavailable_geometry(context=context, evidence=candidate, premium_cap=10.)
    with pytest.raises(RuntimeError):
        verify_unavailable_geometry(context=context, evidence=evidence, premium_cap=20.)
    context["box_certificate"]["declared_cap"] = 9.
    with pytest.raises(RuntimeError, match="disagrees"):
        verify_unavailable_geometry(context=context, evidence=evidence, premium_cap=10.)


@pytest.mark.parametrize("field", ["normalized_premium_prices", "costs", "previous_contract_positions",
    "instrument_exposures", "target_exposures"])
def test_nonfinite_raw_evidence_fails_closed(field) -> None:
    context, evidence = _box()
    evidence[field].flat[0] = np.nan
    with pytest.raises(ValueError):
        verify_unavailable_geometry(context=context, evidence=evidence, premium_cap=10.)


def _first_step_failure(tmp_path):
    episode = tmp_path / "episode"
    failed = episode / "steps" / "step-00"
    failed.mkdir(parents=True)
    np.savez_compressed(failed / "risk_inputs.npz", previous_positions=np.zeros((1, 2)), active_columns=[0, 1])
    return episode, failed


def test_first_step_unavailable_has_no_executed_positions_or_terminal_result(tmp_path) -> None:
    episode, failed = _first_step_failure(tmp_path)
    context = {"elapsed_day": 0, "portfolio_index": 0}
    result = verify_failed_prefix(episode=episode, context=context, premium_cap=10.)
    assert result["executed_prefix_steps"] == 0
    (episode / "summary.parquet").write_bytes(b"fabricated")
    with pytest.raises(ValueError, match="terminal performance"):
        verify_failed_prefix(episode=episode, context=context, premium_cap=10.)


def test_execution_after_failure_and_nonzero_initial_inventory_are_rejected(tmp_path) -> None:
    episode, failed = _first_step_failure(tmp_path)
    context = {"elapsed_day": 0, "portfolio_index": 0}
    np.savez_compressed(failed / "risk_inputs.npz", previous_positions=np.ones((1, 2)), active_columns=[0, 1])
    with pytest.raises(ValueError, match="prior inventory"):
        verify_failed_prefix(episode=episode, context=context, premium_cap=10.)
    (failed / "trades.parquet").write_bytes(b"fabricated")
    with pytest.raises(ValueError, match="incorrectly recorded as executed"):
        verify_failed_prefix(episode=episode, context=context, premium_cap=10.)
