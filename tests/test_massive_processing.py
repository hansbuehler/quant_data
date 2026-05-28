"""
Testing massive
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from quantdata.massive import client
from quantdata.massive.testing.scope import TICKERS, DATES
from quantdata.utils.testing import hash_polars_df
import numpy as np

def _assert_simple_dict_equal(d1, d2):
    assert list(d1.keys()) == list(d2.keys()), "Expected same columns in day summary"
    for k,v in d1.items():
        assert k in d2, f"Expected column {k} in both day summaries"
        assert np.allclose(v, d2[k], equal_nan=True), f"Expected values in column {k} to be close in day summaries"


def _massive_processing(ticker, d):
    from quantdata.massive.processing import filter_valid_trades, filter_valid_quotes, align_quotes_to_trades, align_trades_to_quotes

    r1 = filter_valid_trades(ticker, d)
    assert hash_polars_df(r1) == "bbb2ae0b0a90d6a3555d3cb847115120", f"Unexpected hash for valid trades of {ticker} on {d}"    

    r1 = filter_valid_quotes(ticker, d)
    assert hash_polars_df(r1) == "52ac768e0f77169e4a8c0de671f08f25", f"Unexpected hash for valid quotes of {ticker} on {d}"

    r1 = align_quotes_to_trades(ticker, d, drop_unmatched=True)
    assert hash_polars_df(r1) == "7d0de72560895a970cb4029277cb6886", f"Unexpected hash for aligned quotes to trades of {ticker} on {d} drop_unmatched=True"

    r1 = align_trades_to_quotes(ticker, d, drop_unmatched=True)
    assert hash_polars_df(r1) == "26be8df16a7008ea70279c66be07e153", f"Unexpected hash for aligned trades to quotes of {ticker} on {d} drop_unmatched=True"

    r1 = align_quotes_to_trades(ticker, d, drop_unmatched=False)
    assert hash_polars_df(r1) == "8570010830d4daba791b2ece72c7cb66", f"Unexpected hash for aligned quotes to trades of {ticker} on {d} drop_unmatched=False"

    r1 = align_trades_to_quotes(ticker, d, drop_unmatched=False)
    assert hash_polars_df(r1) == "d4f6b85ed98e6e6a558e9e8d55e1533f", f"Unexpected hash for aligned trades to quotes of {ticker} on {d} drop_unmatched=False"

    """
    No out of sesson trades
    r1 = align_quotes_to_trades(ticker, d, drop_unmatched=True, within_session=False)
    assert hash_polars_df(r1) == "7d0de72560895a970cb4029277cb6886", f"Unexpected hash for aligned quotes to trades of {ticker} on {d} drop_unmatched=True within_session=False"

    r1 = align_trades_to_quotes(ticker, d, drop_unmatched=True, within_session=False)
    assert hash_polars_df(r1) == "26be8df16a7008ea70279c66be07e153", f"Unexpected hash for aligned trades to quotes of {ticker} on {d} drop_unmatched=True within_session=False"

    r1 = align_quotes_to_trades(ticker, d, drop_unmatched=False, within_session=False)
    assert hash_polars_df(r1) == "8570010830d4daba791b2ece72c7cb66", f"Unexpected hash for aligned quotes to trades of {ticker} on {d} drop_unmatched=False within_session=False"
    """

    r1 = align_trades_to_quotes(ticker, d, drop_unmatched=False, within_session=False)
    assert hash_polars_df(r1) == "7c14f1d456f94ceea5a7e27f39d4e12c", f"Unexpected hash for aligned trades to quotes of {ticker} on {d} drop_unmatched=False within_session=False"

def test_massive_processing():
    _massive_processing(TICKERS[0], DATES[0])
