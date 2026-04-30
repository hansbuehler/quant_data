"""
Tests for FRED rate fetching utilities.

Mirrors the checks done manually in poly/notebooks/test_rates.ipynb.
"""

from __future__ import annotations

import os
from datetime import date

import polars as pl
import pytest

requires_fred = pytest.mark.skipif(
    "FRED_API_KEY" not in os.environ,
    reason="FRED_API_KEY not set",
)


@requires_fred
def test_load_fred_rates() -> None:
    from quantdata.fred.rates import load_fred_rates

    df = load_fred_rates(date(2022, 1, 1), date(2024, 1, 1))

    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    assert set(df.columns) == {"date", "on", "sofr", "effr", "dgs1mo", "dgs3mo", "dgs6mo"}
