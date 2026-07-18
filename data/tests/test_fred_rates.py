"""
Tests for FRED rate fetching utilities.

Mirrors the checks done manually in poly/notebooks/test_rates.ipynb.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl
import pytest

_test_mode = os.environ.get("QUANTDATA_TEST_MODE", "0") == "1"
_stored_fred = Path(__file__).parent / "stored_data" / "raw" / "fred" / "fred"
_has_fred_data = any(_stored_fred.glob("*.pck")) if _stored_fred.exists() else False

requires_fred = pytest.mark.skipif(
    "FRED_API_KEY" not in os.environ or (_test_mode and not _has_fred_data),
    reason="FRED_API_KEY not set"
    if "FRED_API_KEY" not in os.environ
    else "no stored test data in tests/stored_data/",
)


@requires_fred
def test_load_fred_rates() -> None:
    from openqai.data.fred.rates import load_fred_rates

    df = load_fred_rates(date(2022, 1, 1), date(2024, 1, 1))

    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    assert set(df.columns) == {"date", "on", "sofr", "effr", "dgs1mo", "dgs3mo", "dgs6mo"}
