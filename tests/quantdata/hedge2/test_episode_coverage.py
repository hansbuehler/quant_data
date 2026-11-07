"""Coverage retains the declared ITM book and never fills missing observed mids."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2 import episode_coverage
from quantdata.hedge2.episode_coverage import coverage


def fixture_arguments(monkeypatch, tmp_path):
    model_directory = tmp_path / "models"
    (model_directory / "days").mkdir(parents=True)
    (model_directory / "days" / "2024-04-01.npz").write_bytes(b"model")
    sessions = [date.fromisoformat(value) for value in
        ("2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04", "2024-04-05", "2024-04-08")]
    market = SimpleNamespace(quotes=pd.DataFrame({"bdte": [1.]}))
    source = SimpleNamespace(interpolate_atm_vols=lambda times: np.full_like(times, .2))
    monkeypatch.setattr(episode_coverage, "MarketStore", lambda **kwargs: SimpleNamespace(load=lambda day: market))
    monkeypatch.setattr(episode_coverage, "SourceArchive", lambda **kwargs: SimpleNamespace(load=lambda day: (source, {})))
    monkeypatch.setattr(episode_coverage, "get_business_days", lambda *args: sessions)
    arguments = dict(source_root=tmp_path / "source", market_directory=tmp_path / "market",
        licensed_fit_directory=tmp_path / "licensed", raw_quote_directory=tmp_path / "raw",
        dynamics_directory=model_directory, output_directory=tmp_path / "coverage",
        option_side="itm", include_co_terminal_forwards=True)
    return arguments


def test_one_day_itm_coverage_records_forwards_and_preserves_frozen_contracts(monkeypatch, tmp_path) -> None:
    arguments = fixture_arguments(monkeypatch, tmp_path)
    recorded = []
    book = pd.DataFrame({"expiry": [pd.Timestamp("2024-04-02")] * 2,
        "strike": [99., 0.], "instrument_kind": ["european", "zero_strike_call"]})

    def choose(market, **kwargs):
        recorded.append(kwargs)
        return book.copy()

    marked = []
    monkeypatch.setattr(episode_coverage, "select_book", choose)
    monkeypatch.setattr(episode_coverage, "observed_marks", lambda contracts, market: marked.append(contracts.copy()))
    result = coverage(**arguments)
    assert result["eligible_by_group"]["digital-1D"] == 1
    assert all(entry["option_side"] == "itm" and entry["include_co_terminal_forwards"] for entry in recorded)
    assert recorded[0]["co_terminal_business_dtes"] is None
    pd.testing.assert_frame_equal(marked[0], book)
    pd.testing.assert_frame_equal(marked[1], book)
    rows = pd.read_parquet(arguments["output_directory"] / "coverage.parquet")
    first = rows.loc[rows.group == "digital-1D"].iloc[0]
    assert first.ordinary_options == first.synthetic_forwards == 1
    assert not rows.loc[rows.group != "digital-1D", "eligible"].any()
    assert rows.loc[rows.group != "digital-1D", "reason"].str.contains("Missing rolling model").all()


def test_missing_observed_quotes_are_exclusions_but_numerical_errors_are_not(monkeypatch, tmp_path) -> None:
    arguments = fixture_arguments(monkeypatch, tmp_path)
    book = pd.DataFrame({"expiry": [pd.Timestamp("2024-04-02")], "strike": [99.]})
    monkeypatch.setattr(episode_coverage, "select_book", lambda *args, **kwargs: book)

    def missing(*args):
        raise ValueError("Missing observed mids on 2024-04-02")

    monkeypatch.setattr(episode_coverage, "observed_marks", missing)
    assert coverage(**arguments)["eligible_by_group"]["digital-1D"] == 0

    def numerical_error(*args):
        raise ValueError("Non-finite option price")

    monkeypatch.setattr(episode_coverage, "observed_marks", numerical_error)
    arguments["output_directory"] = tmp_path / "numerical"
    with pytest.raises(ValueError, match="Non-finite"):
        coverage(**arguments)
