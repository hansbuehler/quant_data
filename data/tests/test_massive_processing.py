"""
Testing massive

To generate tests run python -m generate from massive/testing

"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from quantdata.massive import client
from quantdata.massive.testing.scope import TICKERS, DATES
from core.src.openqai.core.utils import hash_polars_df
import numpy as np

def _assert_simple_dict_equal(d1, d2):
    assert list(d1.keys()) == list(d2.keys()), "Expected same columns in day summary"
    for k,v in d1.items():
        assert k in d2, f"Expected column {k} in both day summaries"
        assert np.allclose(v, d2[k], equal_nan=True), f"Expected values in column {k} to be close in day summaries"


def _massive_processing(ticker, d):
    from quantdata.massive.processing import filter_valid_trades, filter_valid_quotes, align_quotes_to_trades, align_trades_to_quotes

    r1 = filter_valid_trades(ticker, d)
    assert hash_polars_df(r1) == "027a67e7d81937acb5ba0cb51763d115", f"Unexpected hash for valid trades of {ticker} on {d}"    

    r1 = filter_valid_quotes(ticker, d)
    assert hash_polars_df(r1) == "1d610ae6ad9aeecd4179e987799f43a2", f"Unexpected hash for valid quotes of {ticker} on {d}"

    r1 = align_quotes_to_trades(ticker, d, drop_unmatched=True)
    assert hash_polars_df(r1) == "d7e68c05131392b26fef7e0b8c382ecd", f"Unexpected hash for aligned quotes to trades of {ticker} on {d} drop_unmatched=True"

    r1 = align_trades_to_quotes(ticker, d, drop_unmatched=True)
    assert hash_polars_df(r1) == "22a2376285c8e4454bb7aa25a0f75791", f"Unexpected hash for aligned trades to quotes of {ticker} on {d} drop_unmatched=True"

    r1 = align_quotes_to_trades(ticker, d, drop_unmatched=False)
    assert hash_polars_df(r1) == "a1f2e887e38f006237a8aa1825739712", f"Unexpected hash for aligned quotes to trades of {ticker} on {d} drop_unmatched=False"

    r1 = align_trades_to_quotes(ticker, d, drop_unmatched=False)
    assert hash_polars_df(r1) == "de964b4642b8dd8a350757e507a5eae0", f"Unexpected hash for aligned trades to quotes of {ticker} on {d} drop_unmatched=False"

    r1 = align_quotes_to_trades(ticker, d, drop_unmatched=True, within_session=False)
    assert hash_polars_df(r1) == "eb9101bf50b31146b372bd587d06815a", f"Unexpected hash for aligned quotes to trades of {ticker} on {d} drop_unmatched=True within_session=False"

    r1 = align_trades_to_quotes(ticker, d, drop_unmatched=True, within_session=False)
    assert hash_polars_df(r1) == "cd44bc9dc01e98e46a7deed8b8e639fd", f"Unexpected hash for aligned trades to quotes of {ticker} on {d} drop_unmatched=True within_session=False"

    r1 = align_quotes_to_trades(ticker, d, drop_unmatched=False, within_session=False)
    assert hash_polars_df(r1) == "40e430d4deb90cb6213ba38b90fde133", f"Unexpected hash for aligned quotes to trades of {ticker} on {d} drop_unmatched=False within_session=False"

    r1 = align_trades_to_quotes(ticker, d, drop_unmatched=False, within_session=False)
    assert hash_polars_df(r1) == "a0ce73a1219e12643284a62e84867fc7", f"Unexpected hash for aligned trades to quotes of {ticker} on {d} drop_unmatched=False within_session=False"

def test_massive_processing():
    _massive_processing(TICKERS[0], DATES[0])
