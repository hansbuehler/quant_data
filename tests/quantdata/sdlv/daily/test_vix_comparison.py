from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantdata.biz_calendar import get_business_days
from quantdata.sdlv.daily.vix_comparison import (
    _model_inputs,
    calendar_business_maturity,
    eligible_expiries,
    interpolate_density,
    log_contract_total_variance,
    mixture_prices,
)


def test_wider_checkpoint_requires_completed_matching_dataset(tmp_path) -> None:
    (tmp_path / "fit").mkdir()
    dataset = tmp_path / "market.npz"
    np.savez(dataset, placeholder=np.zeros(1))
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    checkpoint = tmp_path / "fit/Gromm-v1-4,3.npz"
    np.savez(checkpoint, schema="daily25-jax-weighted-fit-v1", mode="direct", dataset_sha256=digest,
             raw=np.zeros((1, 25, 200)), model_label="Gromm-v1-4,3")
    report = dict(status="complete", dataset_sha256=digest, checkpoint=str(checkpoint))
    (tmp_path / "fit/report.json").write_text(json.dumps(report))
    loaded_dataset, loaded_checkpoint, label, raw = _model_inputs("full", tmp_path)
    assert loaded_dataset == dataset and loaded_checkpoint == checkpoint
    assert label == "Gromm-v1-4,3" and raw.shape == (1, 25, 200)
    report["status"] = "running"
    (tmp_path / "fit/report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not completed"):
        _model_inputs("full", tmp_path)
    report["status"] = "complete"
    (tmp_path / "fit/report.json").write_text(json.dumps(report))
    np.savez(dataset, different=np.ones(1))
    with pytest.raises(ValueError, match="does not belong"):
        _model_inputs("full", tmp_path)


def test_log_contract_includes_background_and_calendar_annualization() -> None:
    total = log_contract_total_variance(np.array([1.0]), np.array([1.0]),
                                       business_time_years=20 / 252, background_volatility=0.2)
    assert total == pytest.approx(0.2**2 * 20 / 252)
    assert 100 * np.sqrt(total * 365 / 30) == pytest.approx(19.65307574511904)


def test_log_contract_matches_broad_otm_option_integral() -> None:
    support, density = np.array([0.8, 1.2]), np.array([0.5, 0.5])
    left = np.geomspace(0.1, 1, 20001)
    right = np.geomspace(1, 3, 20001)
    _, puts = mixture_prices(support, density, left, business_days=21)
    calls, _ = mixture_prices(support, density, right, business_days=21)
    numeric = 2 * (np.trapezoid(puts / left**2, left) + np.trapezoid(calls / right**2, right))
    exact = log_contract_total_variance(support, density, business_time_years=21 / 252)
    assert numeric == pytest.approx(exact, abs=2e-9)
    with pytest.raises(ValueError, match="unit"):
        log_contract_total_variance(support, density * 0.9, business_time_years=21 / 252)


def test_density_interpolation_rejects_extrapolation() -> None:
    tower = np.array([[0.1, 0.8, 0.1], [0.2, 0.6, 0.2]])
    np.testing.assert_allclose(interpolate_density(tower, 1.25), [0.125, 0.75, 0.125])
    with pytest.raises(ValueError, match="outside"):
        interpolate_density(tower, 2.1)


def test_actual_calendar_handles_weekend_and_labor_day() -> None:
    sessions = np.asarray(get_business_days(date(2025, 8, 29), date(2025, 10, 10)), dtype="datetime64[D]")
    assert calendar_business_maturity(date(2025, 8, 29), sessions) == pytest.approx(19 + 2 / 3)


def test_cboe_fridays_do_not_use_other_weekdays_or_third_friday_pm() -> None:
    sessions = np.asarray(get_business_days(date(2025, 8, 18), date(2025, 10, 10)), dtype="datetime64[D]")
    raw = pd.DataFrame({"exdate": ["2025-09-16", "2025-09-19", "2025-09-19", "2025-09-26"],
                        "am_settlement": [0., 0., 1., 0.]})
    entries = eligible_expiries(date(2025, 8, 20), raw, sessions)
    assert entries[0][0] == date(2025, 9, 19)
    assert entries[0][1] is True
    assert entries[0][2] == pytest.approx(30 - 405 / 1440)
    assert all(entry[0].weekday() == 4 for entry in entries)
