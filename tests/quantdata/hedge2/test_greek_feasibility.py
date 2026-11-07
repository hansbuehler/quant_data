"""A solver status alone never certifies exact-Greek position-box infeasibility."""

from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.hedge2.greek_feasibility import certify_greek_box
from quantdata.hedge2.greek_hedge import GreekBoxInfeasible, solve_greek_hedge


@pytest.mark.parametrize("side", [-1, 1])
def test_separating_certificate_replays_original_exposure_support(side) -> None:
    exposures = np.array([[1., 0.], [0., .5], [0., .5]])
    target = np.array([2., 12.])
    certificate = certify_greek_box(exposures=exposures, target_exposures=target,
        target_side=side, position_cap=10.)
    assert certificate.certified_infeasible
    assert certificate.minimum_maximum_position == pytest.approx(12.)
    assert certificate.dual_lower_bound == pytest.approx(12.)
    assert certificate.separation_margin > certificate.tolerance_allowance
    np.testing.assert_allclose(exposures.T @ certificate.primal_positions, -side * target, atol=1e-10)
    row_scale = np.maximum(np.abs(exposures.T).max(axis=1), np.abs(target))
    original_separator = certificate.normalized_separator / row_scale
    assert float(original_separator @ (-side * target)) == pytest.approx(certificate.separator_target)
    assert 10. * np.abs(exposures @ original_separator).sum() == pytest.approx(certificate.separator_box_support)


def test_exact_boundary_and_within_matching_tolerance_are_not_called_infeasible() -> None:
    for target in (10., 10. + 1e-7):
        certificate = certify_greek_box(exposures=np.ones((1, 1)), target_exposures=np.array([target]),
            position_cap=10.)
        assert not certificate.certified_infeasible


def test_zero_risk_rows_and_arbitrary_risk_units_do_not_change_required_cap() -> None:
    exposures = np.array([[1., 0., 0.], [0., .5, 0.], [0., .5, 0.]])
    target = np.array([2., 12., 0.])
    original = certify_greek_box(exposures=exposures, target_exposures=target, position_cap=10.)
    units = np.array([1e4, 1e-5, 1.])
    scaled = certify_greek_box(exposures=exposures * units, target_exposures=target * units,
        position_cap=10.)
    assert scaled.minimum_maximum_position == pytest.approx(original.minimum_maximum_position)
    assert scaled.certified_infeasible == original.certified_infeasible
    assert scaled.normalized_separator[-1] == 0.


def test_feasible_box_does_not_get_an_infeasibility_label() -> None:
    result = certify_greek_box(exposures=np.eye(2), target_exposures=np.array([2., -3.]), position_cap=10.)
    assert not result.certified_infeasible
    assert result.minimum_maximum_position == pytest.approx(3.)
    assert result.separation_margin < 0.


def test_auxiliary_solver_status_or_invalid_dual_remains_a_review_error(monkeypatch) -> None:
    monkeypatch.setattr("quantdata.hedge2.greek_feasibility.linprog",
        lambda **kwargs: SimpleNamespace(success=False, status=2, message="infeasible", x=None))
    with pytest.raises(RuntimeError, match="needs review"):
        certify_greek_box(exposures=np.eye(2), target_exposures=np.ones(2), position_cap=10.)
    monkeypatch.setattr("quantdata.hedge2.greek_feasibility.linprog", lambda **kwargs:
        SimpleNamespace(success=True, x=np.array([1., 1., 1.]), message="test",
            eqlin=SimpleNamespace(marginals=np.array([-100., -100.]))))
    with pytest.raises(RuntimeError, match="primal/dual"):
        certify_greek_box(exposures=np.eye(2), target_exposures=np.ones(2), position_cap=10.)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0., -1.])
def test_invalid_caps_are_rejected(invalid) -> None:
    with pytest.raises(ValueError):
        certify_greek_box(exposures=np.eye(2), target_exposures=np.ones(2), position_cap=invalid)


def test_unspanned_target_is_not_confused_with_a_certified_finite_cap_problem() -> None:
    with pytest.raises(RuntimeError, match="needs review"):
        certify_greek_box(exposures=np.array([[1., 0.]]), target_exposures=np.ones(2), position_cap=10.)


@pytest.mark.parametrize("side", [-1, 1])
def test_failed_cost_lp_is_classified_only_after_independent_box_certificate(side) -> None:
    with pytest.raises(GreekBoxInfeasible) as captured:
        solve_greek_hedge(exposures=np.eye(2), target_exposures=np.array([12., 1.]),
            target_side=side, costs=np.array([0., .01]), position_cap=10.)
    assert captured.value.certificate.certified_infeasible
    assert captured.value.certificate.dual_lower_bound == pytest.approx(12.)
    assert captured.value.solver_status == 2
    assert "infeasible" in captured.value.solver_message.lower()


@pytest.mark.parametrize("solver_status", [2, 4])
def test_unproved_infeasibility_or_numeric_error_still_stops(monkeypatch, solver_status) -> None:
    monkeypatch.setattr("quantdata.hedge2.greek_hedge.linprog", lambda **kwargs:
        SimpleNamespace(success=False, status=solver_status, message="original exact message"))
    monkeypatch.setattr("quantdata.hedge2.greek_hedge.certify_greek_box", lambda **kwargs:
        SimpleNamespace(certified_infeasible=False))
    with pytest.raises(RuntimeError, match="original exact message") as captured:
        solve_greek_hedge(exposures=np.eye(2), target_exposures=np.ones(2),
            costs=np.array([0., .01]), position_cap=10.)
    assert not isinstance(captured.value, GreekBoxInfeasible)


def test_auxiliary_gap_recovery_keeps_original_matching_and_box_certificate(monkeypatch) -> None:
    from scipy.optimize import linprog as original_linprog

    calls = []

    def inaccurate_first(**kwargs):
        result = original_linprog(**kwargs)
        calls.append(kwargs)
        if len(calls) <= 2:
            result.x[0] += 1e-4
        return result

    monkeypatch.setattr("quantdata.hedge2.greek_feasibility.linprog", inaccurate_first)
    certificate = certify_greek_box(exposures=np.eye(2), target_exposures=np.array([12., 1.]), position_cap=10.)
    assert certificate.certified_infeasible
    assert certificate.dual_lower_bound == pytest.approx(12.)
    assert certificate.minimum_maximum_position == pytest.approx(12.)
    assert certificate.normalized_primal_residual < 1e-12
    assert certificate.duality_gap < 1e-12
    assert len(calls) == 3 and calls[-1]["method"] == "highs-ipm"
    assert "Original numerical certificate failure retained" in certificate.solver_message
    assert "primal/dual check failed" in certificate.solver_message


def test_row_whitening_maps_dual_back_to_original_exposure_units(monkeypatch) -> None:
    from scipy.optimize import linprog as original_linprog

    calls = []

    def inaccurate_first(**kwargs):
        result = original_linprog(**kwargs)
        calls.append(kwargs)
        if len(calls) <= 2:
            result.x[0] += 1e-3
        return result

    monkeypatch.setattr("quantdata.hedge2.greek_feasibility.linprog", inaccurate_first)
    exposures = np.array([[1., 2.], [2., .1], [3., 1.]])
    target = np.array([100., 20.])
    certificate = certify_greek_box(exposures=exposures, target_exposures=target, position_cap=10.)
    matrix = exposures.T
    scale = np.maximum(np.max(np.abs(matrix), axis=1), np.abs(target))
    separator = certificate.normalized_separator
    normalized = matrix / scale[:, None]
    assert certificate.separator_target == pytest.approx(separator @ (target / scale))
    assert certificate.separator_box_support == pytest.approx(10. * np.abs(normalized.T @ separator).sum())
    assert certificate.certified_infeasible
    assert certificate.normalized_primal_residual < 1e-12
