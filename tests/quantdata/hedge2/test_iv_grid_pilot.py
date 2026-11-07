"""Checks for declared IV-grid accuracy budgets and dated state integrity."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantdata.hedge2.iv_grid_pilot as pilot
from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.historical import sanos_europeans
from quantdata.hedge2.iv_grid import IVGridConfig, extract_iv_grid
from quantdata.sanos.arbfree import ArbFreePricer


def _source() -> ArbFreePricer:
    source = ArbFreePricer(smoothness=0.1)
    source.expiries = np.array([1., 25.]) / DAYS_PER_YEAR
    source.variances = source.expiries * 0.2**2 / source.smoothness
    source.fitsX = [np.zeros(1), np.zeros(1)]
    source.mstrikes = [np.ones(1), np.ones(1)]
    source.mdensities = [np.ones(1), np.ones(1)]
    return source


def _book() -> tuple[ArbFreePricer, CashCurve, pd.DataFrame]:
    source = _source()
    curve = CashCurve(spot=100., business_dtes=np.array([1., 3., 25.]),
        forwards=np.array([100.1, 100.3, 102.]), reference_forwards=np.array([100.2, 100.6, 104.]),
        discounts=np.array([0.999, 0.997, 0.98]))
    days = np.array([1., 3., 3.])
    strikes = np.array([99., 101., 0.])
    calls = np.array([True, False, True])
    source_prices = sanos_europeans(source=source, curve=curve, business_dtes=days,
                                    cash_strikes=strikes, is_call=calls)
    return source, curve, pd.DataFrame({"bdte": days, "strike": strikes,
        "option_type": ["call", "put", "call"], "instrument_kind": ["european", "european", "zero_strike_call"],
        "observed_mid": source_prices + np.array([0.3, -0.3, 0.]), "observed_half_spread": [0.1, 0.1, 0.]})


def test_accuracy_is_against_source_while_observed_marks_remain_separate() -> None:
    source, curve, book = _book()
    original = book.copy(deep=True)
    extraction = extract_iv_grid(source=source, day=date(2024, 4, 19))
    compared, summary = pilot.compare_book(extraction=extraction, source=source, curve=curve,
                                           book=book, maximum_half_spread_fraction=0.01)
    assert summary["pilot_european_pricing_gate_passed"]
    assert summary["max_abs_source_mid_error_bps_spot"] == pytest.approx(30.)
    assert summary["max_abs_synthetic_error_cash"] == 0.
    np.testing.assert_array_equal(compared["observed_mid"], book["observed_mid"])
    np.testing.assert_allclose(compared["iv_mid_error_cash"],
                               compared["iv_source_error_cash"] + compared["source_mid_error_cash"], atol=1e-14)
    assert np.isnan(compared.iloc[-1]["iv_source_error_fraction_half_spread"])
    pd.testing.assert_frame_equal(book, original)


def test_underresolved_model_is_not_accepted_by_source_mid_fit_quality() -> None:
    source, curve, book = _book()
    extraction = extract_iv_grid(source=source, day=date(2024, 4, 19))
    extraction.grid.log_implied_vols[:] += np.log(1.2)
    _, summary = pilot.compare_book(extraction=extraction, source=source, curve=curve,
                                    book=book, maximum_half_spread_fraction=0.01)
    assert not summary["pilot_european_pricing_gate_passed"]
    assert summary["above_half_spread_budget"] == 2


def test_zero_observed_spread_is_not_a_free_ordinary_asset() -> None:
    source, curve, book = _book()
    book.loc[0, "observed_half_spread"] = 0.
    extraction = extract_iv_grid(source=source, day=date(2024, 4, 19))
    with pytest.raises(ValueError, match="spreads"):
        pilot.compare_book(extraction=extraction, source=source, curve=curve,
                            book=book, maximum_half_spread_fraction=0.01)


def test_resolution_must_pass_every_requested_date() -> None:
    days = (date(2024, 4, 18), date(2024, 4, 19))
    records = [{"date": "2024-04-18", "num_nms_nodes": 100, "pilot_european_pricing_gate_passed": True},
        {"date": "2024-04-19", "num_nms_nodes": 100, "pilot_european_pricing_gate_passed": False},
        {"date": "2024-04-18", "num_nms_nodes": 200, "pilot_european_pricing_gate_passed": True},
        {"date": "2024-04-19", "num_nms_nodes": 200, "pilot_european_pricing_gate_passed": True}]
    assert pilot.select_resolution(records=records, days=days, node_counts=(400, 200, 100)) == 200
    assert pilot.select_resolution(records=records[:-1], days=days, node_counts=(200, 100)) is None


def _archive(monkeypatch: pytest.MonkeyPatch, *, missing_day: date | None = None) -> tuple[date, ...]:
    days = (date(2024, 4, 18), date(2024, 4, 19))

    class Archive:
        def dates(self) -> tuple[date, ...]:
            return days

        def load(self, day: date) -> tuple[ArbFreePricer, dict[str, str]]:
            if day == missing_day:
                raise FileNotFoundError("Missing certified source")
            return _source(), {"source_sha256": "synthetic-source", "source_model": "synthetic"}

    monkeypatch.setattr(pilot, "SourceArchive", lambda **kwargs: Archive())
    return days


def test_dated_runner_preserves_each_state_and_rejects_changed_artifacts(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    days = _archive(monkeypatch)
    directory = tmp_path / "dated"
    config = IVGridConfig(num_nms_nodes=200)
    summary = pilot.run_dated_extraction(source_root=tmp_path / "source", output_directory=directory,
        start=days[0], end=days[-1], config=config)
    assert summary["extracted_dates"] == 2
    assert summary["failed_dates"] == 0
    grid = pilot.load_dated_grid(directory, days[-1])
    assert grid.state.shape == (9 * 200,)
    np.testing.assert_allclose(np.exp(grid.log_implied_vols[:, 110:118]), 0.2, atol=2e-6)
    path = directory / "days" / f"{days[-1].isoformat()}.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hashes"):
        pilot.load_dated_grid(directory, days[-1])


def test_failed_date_is_not_filled_or_presented_as_extracted(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    days = _archive(monkeypatch, missing_day=date(2024, 4, 19))
    directory = tmp_path / "dated"
    summary = pilot.run_dated_extraction(source_root=tmp_path / "source", output_directory=directory,
        start=days[0], end=date(2024, 4, 22), config=IVGridConfig())
    assert summary["status"] == "partial"
    assert summary["failed_dates"] == 1
    assert summary["trailing_source_gap"]
    assert not (directory / "days" / f"{days[-1].isoformat()}.npz").exists()
    with pytest.raises(ValueError, match="incomplete"):
        pilot.load_dated_grid(directory, days[-1])


def test_existing_output_is_never_reused_or_overwritten(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    days = _archive(monkeypatch)
    with pytest.raises(FileExistsError):
        pilot.run_dated_extraction(source_root=tmp_path / "source", output_directory=tmp_path,
            start=days[0], end=days[-1], config=IVGridConfig())


def test_outputs_are_rejected_inside_an_actual_git_worktree(tmp_path: Path) -> None:
    working_tree = tmp_path / "source_worktree"
    (working_tree / ".git").mkdir(parents=True)
    (working_tree / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(ValueError, match="Git working tree"):
        pilot._prepare_run(working_tree / "outputs", tmp_path / "market_archive")


@pytest.mark.parametrize("changes", [{"node_counts": ()}, {"node_counts": (29, 29)},
    {"node_counts": (1,)}, {"maximum_half_spread_fraction": 0.}, {"maximum_half_spread_fraction": np.nan}])
def test_invalid_pilot_policy(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        pilot.IVGridPilotConfig(**changes)
