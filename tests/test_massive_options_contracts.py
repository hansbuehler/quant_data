from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quantdata.massive.options.contracts import (
    REJECTED,
    ContractsFilter,
    filter_contracts,
)

_test_mode = os.environ.get("QUANTDATA_TEST_MODE", "0") == "1"
_stored_massive = Path(__file__).parent / "stored_data" / "raw" / "massive"
_has_massive_data = _stored_massive.exists() and any(_stored_massive.iterdir())

requires_key = pytest.mark.skipif(
    "MASSIVE_TOKEN" not in os.environ or (_test_mode and not _has_massive_data),
    reason="MASSIVE_TOKEN not set"
    if "MASSIVE_TOKEN" not in os.environ
    else "no stored test data in tests/stored_data/",
)

_CACHE_FILE = Path(__file__).resolve().parent / "stored_data" / "processing" / "massive" / "contracts" / "max_dte=25,max_mns=3,min_volu_k=1 9cbf6f04" / "SPY" / "contracts" / "2024-01-02.prq"

requires_cached_contracts = pytest.mark.skipif(
    not _CACHE_FILE.exists(),
    reason="stored contract cache file not present",
)

@requires_key
@requires_cached_contracts
def test_filter_contracts_value_ranges() -> None:
    contracts_filter = ContractsFilter(max_dte=25, max_mns=3.0, min_opt_vol_k=1)
    df = filter_contracts("SPY", date(2024, 1, 2), contracts_filter, exclude_half_days=True)
    active = df.filter(pl.col("reject_code") == REJECTED.USED)

    known_codes = {
        REJECTED.USED,
        REJECTED.EXPIRY,
        REJECTED.MONEYNESS,
        REJECTED.VOLUME,
        REJECTED.PREV_VOLUME_MISSING,
        REJECTED.PREV_NO_QUOTES_OR_TRADES,
        REJECTED.NO_TODAY_QUOTES,
    }
    assert set(df["reject_code"].unique().to_list()).issubset(known_codes)

    assert df.filter(pl.col("days_to_exp") < 0).height == 0
    assert df.filter(pl.col("strike_price") <= 0).height == 0
    assert df.filter(pl.col("exp_date") < pl.lit(date(2024, 1, 2))).height == 0

    assert active.height > 0
    assert active.filter(pl.col("days_to_exp") > contracts_filter.max_dte).height == 0
    assert active.filter(pl.col("nms") > contracts_filter.max_mns).height == 0
    assert (
        active.filter(pl.col("approx_cash_vol_prev_k") < contracts_filter.min_opt_vol_k).height == 0
    )
