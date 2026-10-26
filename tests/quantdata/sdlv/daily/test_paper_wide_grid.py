from datetime import date

import polars as pl
import pytest

from quantdata.sdlv.daily.paper_wide_grid import _table, comparison_frame


def _frames():
    base = pl.DataFrame(dict(date=[date(2025, 8, 28), date(2025, 8, 29)], cboe_vix=[15.0, 15.36],
                             market_discrete=[15.1, 15.405], gromm_discrete=[18.0, 18.434],
                             gromm_friday_continuous=[18.1, 18.432]))
    wide = base.with_columns(pl.Series("gromm_discrete", [16.0, 16.314]),
                            pl.Series("gromm_friday_continuous", [16.1, 16.318]))
    filtered = base.select("date", "cboe_vix").with_columns(pl.Series("sanos_input", [14.9, 15.197]))
    return base, wide, filtered


def test_comparison_keeps_exact_dates_and_distinct_model_definitions():
    frame = comparison_frame(*_frames())
    assert frame["wide_discrete"].to_list() == [16.0, 16.314]
    assert frame["wide_analytic"].to_list() == [16.1, 16.318]
    assert frame["v1_discrete"].to_list() == [18.0, 18.434]
    assert frame["cboe_vix"].to_list() == [15.0, 15.36]


def test_comparison_rejects_shifted_dates_and_different_official_values():
    base, wide, filtered = _frames()
    with pytest.raises(ValueError, match="date grids"):
        comparison_frame(base, wide.reverse(), filtered)
    with pytest.raises(AssertionError):
        comparison_frame(base, wide.with_columns(pl.col("cboe_vix") + 0.01), filtered)


def test_comparison_rejects_duplicate_dates():
    base, wide, filtered = _frames()
    with pytest.raises(ValueError, match="Duplicate"):
        comparison_frame(pl.concat([base, base]), wide, filtered)


def test_latex_table_has_valid_row_separators_and_label():
    result = _table(["A", "B"], [["1", "2"]], caption="Example.", label="tab:example", alignment="lr")
    assert "A & B \\\\\n" in result
    assert "1 & 2 \\\\\n" in result
    assert r"\label{tab:example}" in result
