"""Spot normalization uses yesterday's carry and rejects missing-session returns."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2 import observations


def test_returns_use_previous_reference_forward_and_exact_sessions(tmp_path, monkeypatch) -> None:
    dates = (date(2024, 4, 1), date(2024, 4, 2), date(2024, 4, 4))
    spots = {dates[0]: 100., dates[1]: 101., dates[2]: 102.}
    def observation(_directory, day):
        return {"date": day, "spot": spots[day], "forward_reference_1d": spots[day] + 0.1}
    monkeypatch.setattr(observations, "daily_observation", observation)
    frame = observations.prepare_observations(SimpleNamespace(dates=lambda: dates), tmp_path / "obs.parquet",
                                               licensed_fit_directory=tmp_path)
    assert frame["normalized_spot_return_into_date"].iloc[1] == pytest.approx(101. / 100.1 - 1.)
    assert np.isnan(frame["normalized_spot_return_into_date"].iloc[2])
    assert frame["exact_previous_session"].tolist() == [False, True, False]


def test_observation_excludes_am_carry_and_keeps_parity_distinct(tmp_path) -> None:
    day = date(2024, 4, 1)
    (tmp_path / "SPX").mkdir()
    frame = pd.DataFrame({"date": [pd.Timestamp(day)] * 3, "spot": [100.] * 3,
        "spot_prev": [99.] * 3, "is_am": [True, False, False], "bdte": [2., 2., 5.],
        "fwd": [130., 102., 105.], "fwd_ref": [120., 101., 103.], "df": [.95, .99, .98]})
    frame.to_parquet(tmp_path / "SPX" / f"{day}.pdq", index=False)
    result = observations.daily_observation(tmp_path, day)
    assert result["forward_parity_1d"] == pytest.approx(np.sqrt(100. * 102.))
    assert result["forward_reference_1d"] == pytest.approx(np.sqrt(100. * 101.))
    assert result["discount_1d"] == pytest.approx(np.sqrt(.99))
    assert result["first_source_business_dte"] == 2.
    assert result["spot_prev_source"] == 99.


def test_preparation_honors_requested_history_range(tmp_path, monkeypatch) -> None:
    dates = (date(2021, 12, 31), date(2022, 1, 3), date(2022, 1, 4), date(2026, 1, 2))
    seen = []

    def observation(_directory, day):
        seen.append(day)
        return {"date": day, "spot": 100., "forward_reference_1d": 100.1}

    monkeypatch.setattr(observations, "daily_observation", observation)
    frame = observations.prepare_observations(SimpleNamespace(dates=lambda: dates), tmp_path / "obs.parquet",
        licensed_fit_directory=tmp_path, start=date(2022, 1, 1), end=date(2025, 12, 31))
    assert seen == [date(2022, 1, 3), date(2022, 1, 4)]
    assert len(frame) == 2


def test_empty_observation_scope_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="No source observations"):
        observations.prepare_observations(SimpleNamespace(dates=lambda: (date(2021, 1, 4),)),
            tmp_path / "obs.parquet", licensed_fit_directory=tmp_path,
            start=date(2022, 1, 1), end=date(2025, 12, 31))
