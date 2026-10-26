from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantdata.widb import options_filter


def test_option_filter_restores_legacy_object_string_schema(monkeypatch) -> None:
    frame = pd.DataFrame({
        "ticker": pd.Series(["SPX"], dtype="str"),
        "value": pd.Series([1.0], dtype="float32"),
    })
    monkeypatch.setattr(options_filter, "SCHEMA_OPTION_DAY", lambda *_: {"ticker": np.dtype("O"), "value": np.dtype("float32")})
    monkeypatch.setattr(options_filter.OptionsFilter, "_filter", lambda self, ticker, day, data: data)
    result = options_filter.filter_options_day("SPX", date(2022, 1, 3), source_data=frame, mk_arb_free=None)
    assert result["ticker"].dtype == np.dtype("O")
    assert result["value"].dtype == np.dtype("float32")
    assert result["ticker"].tolist() == ["SPX"]
    assert isinstance(frame["ticker"].dtype, pd.StringDtype)


def test_option_filter_does_not_coerce_numeric_schema_errors(monkeypatch) -> None:
    frame = pd.DataFrame({"ticker": pd.Series(["SPX"], dtype="str"), "value": pd.Series([1.0], dtype="float64")})
    monkeypatch.setattr(options_filter, "SCHEMA_OPTION_DAY", lambda *_: {"ticker": np.dtype("O"), "value": np.dtype("float32")})
    monkeypatch.setattr(options_filter.OptionsFilter, "_filter", lambda self, ticker, day, data: data)
    with pytest.raises(ValueError, match="schema mismatches.*value"):
        options_filter.filter_options_day("SPX", date(2022, 1, 3), source_data=frame, mk_arb_free=None)
