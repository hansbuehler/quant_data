"""Causality and geometry invariants for the controlled DYSANOS rank comparison."""

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from quantdata.hedge2.dynamics import DynamicsConfig, fit_dynamics
from quantdata.hedging.model_audit.rank_comparison import validate_matched_fits
from quantdata.hedging.model_audit.report import _rank_comparison_section
from quantdata.hedging.model_audit.run import retained_dynamics_path


@pytest.mark.parametrize("orthogonal", [False, True])
def test_rank_changes_preserve_window_spot_and_nested_native_span(orthogonal: bool) -> None:
    random = np.random.default_rng(516)
    dates = tuple(date(2023, 1, 1) + timedelta(days=index) for index in range(260))
    states = random.normal(size=(260, 24))
    spot = random.normal(0., .01, size=260)
    arguments = dict(dates=dates, states=states, spot_returns_into_date=spot,
                     session_indices=np.arange(260), as_of_date=dates[-1])
    baseline = fit_dynamics(**arguments, config=DynamicsConfig(rank=8, spot_orthogonal=orthogonal))
    expanded = fit_dynamics(**arguments, config=DynamicsConfig(rank=16, spot_orthogonal=orthogonal))
    validate_matched_fits(baseline, expanded)
    assert expanded.explained_variance_fraction.sum() > baseline.explained_variance_fraction.sum()
    np.testing.assert_allclose(expanded.directions.T @ expanded.directions, np.eye(16), atol=1e-12)
    with pytest.raises(ValueError, match="training windows"):
        validate_matched_fits(baseline, replace(expanded, training_start=dates[0]))
    shifted = expanded.residuals.copy()
    shifted[:, 0] += .001
    with pytest.raises(ValueError, match="paired spot"):
        validate_matched_fits(baseline, replace(expanded, residuals=shifted))


def test_sixteen_factor_loading_never_falls_back_to_eight(tmp_path: Path) -> None:
    day = date(2024, 4, 1)
    path = retained_dynamics_path(tmp_path / "root", "DYSANOS spot-orthogonal (16 factors)", day, output=tmp_path)
    assert path == tmp_path / "fits/orthogonal/days/2024-04-01.npz"
    with pytest.raises(ValueError, match="dedicated"):
        retained_dynamics_path(tmp_path, "DYSANOS ordinary (16 factors)", day)
    assert retained_dynamics_path(tmp_path / "root", "DYSANOS spot-orthogonal (16 factors) (tail)", day,
                                  output=tmp_path) == path


def test_missing_rank_comparison_is_not_reported_as_complete(tmp_path: Path) -> None:
    assert _rank_comparison_section(tmp_path) == []


@pytest.mark.parametrize("reject_all", [False, True])
def test_one_unusable_atm_inversion_does_not_discard_other_paths(monkeypatch, reject_all: bool) -> None:
    from quantdata.hedge2.batch_pricing import BatchDlvPricer
    from quantdata.hedging.model_audit import run

    pricer = BatchDlvPricer(support=np.tile([.8, 1., 1.2], (2, 1)),
        densities=np.tile([.2, .6, .2], (2, len(run.GRID.expiries_y), 1)),
        business_dtes=np.asarray(run.GRID.expiries_y) * 252.,
        lower_clips=np.array([1, 2]), upper_clips=np.array([3, 4]))
    monkeypatch.setattr(run, "decode_batch", lambda *arguments, **keywords: pricer)
    baseline, _ = run._dlv_surface(None, np.zeros((2, 3)))
    original = run.implied_volatilities_with_validity
    calls = 0

    def reject_some_atm(**arguments):
        nonlocal calls
        vols, valid = original(**arguments)
        if calls == 0:
            if reject_all:
                valid[:] = False
            else:
                valid[0] = False
        calls += 1
        return vols, valid

    monkeypatch.setattr(run, "implied_volatilities_with_validity", reject_some_atm)
    surfaces, diagnostics = run._dlv_surface(None, np.zeros((2, 3)))
    assert np.isnan(surfaces[0]).all()
    assert diagnostics["atm_invalid_scenario_count"] == (2 if reject_all else 1)
    assert diagnostics["invalid_path_rate"] == (1. if reject_all else .5)
    assert diagnostics["lower_clips"] == 3
    assert diagnostics["upper_clips"] == 7
    if not reject_all:
        np.testing.assert_array_equal(surfaces[1], baseline[1])
