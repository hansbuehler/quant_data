"""Observed quote joins must not substitute fitted SANOS prices for mids."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quantdata.hedge2.market import BENCHMARK_ID, MarketPaths, load_benchmark_year, load_market_day


def _make_paths(tmp_path: Path) -> MarketPaths:
    benchmark = tmp_path / BENCHMARK_ID
    licensed = tmp_path / "private"
    (benchmark / "SPX").mkdir(parents=True)
    (licensed / "SPX").mkdir(parents=True)
    day = pd.Timestamp("2024-04-01")
    expiry = pd.Timestamp("2024-04-02")
    pd.DataFrame({
        "date": [day, day], "expiry": [expiry, expiry],
        "strike": [100.0, 100.0], "option_type": ["call", "put"],
        "is_am": [False, False], "bdte": [1.0, 1.0],
        "nms": [0.0, 0.0], "fwd": [100.0, 100.0],
        "df": [1.0, 1.0], "fit_price": [1.3, 1.4],
        "fit_vol": [0.2, 0.2],
    }).to_parquet(benchmark / "SPX" / "2024-01-01_2024-12-31x-all-0.10.parquet")
    pd.DataFrame({
        "date": [day], "expiry": [expiry], "strike": [100.0],
        "is_am": [False], "mkt_call_bid": [1.1], "mkt_call_ask": [1.5],
        "mkt_put_bid": [1.2], "mkt_put_ask": [1.8],
    }).to_parquet(licensed / "SPX" / "2024-04-01.pdq")
    return MarketPaths(benchmark_directory=benchmark, licensed_fit_directory=licensed)


def test_observed_market_mid_is_distinct_from_sanos_fit(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    frame = load_market_day(paths, date(2024, 4, 1), load_benchmark_year(paths, 2024))
    assert frame["option_type"].tolist() == ["call", "put"]
    assert frame["observed_mid"].tolist() == pytest.approx([1.3, 1.5])
    assert frame["fit_price"].tolist() == pytest.approx([1.3, 1.4])
    assert frame["observed_half_spread"].tolist() == pytest.approx([0.2, 0.3])


def test_join_rejects_contract_mismatch(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    benchmark = load_benchmark_year(paths, 2024)
    benchmark.loc[benchmark["option_type"] == "put", "strike"] = 101.0
    with pytest.raises(ValueError, match="Benchmark/source mismatch"):
        load_market_day(paths, date(2024, 4, 1), benchmark)
