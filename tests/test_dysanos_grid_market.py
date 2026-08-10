from datetime import date

import numpy as np
import pytest
from cdxcore.verbose import Context

from quantdata.dysanos.grid import torch_market


def test_grid_market_load_vectorizes_business_day_count(monkeypatch) -> None:
    dates = np.array(
        ["2024-01-02", "2024-01-03", "2024-01-05"], dtype="datetime64[D]"
    )
    monkeypatch.setattr(
        torch_market,
        "interpolate_time_series",
        lambda *args, **kwargs: {"dates": dates},
    )
    calls: list[tuple[object, np.ndarray]] = []

    def count_days(start, end):
        calls.append((start, np.asarray(end)))
        return np.array([0, 1, 3], dtype=np.int32)

    monkeypatch.setattr(torch_market, "count_business_days_between", count_days)
    definition = torch_market.GridMarketDefinition(
        date_range=torch_market.DateRange(
            start=date(2024, 1, 2), end=date(2024, 1, 5)
        )
    )

    market = definition.load(verbose=Context.quiet)

    assert len(calls) == 1
    assert np.array_equal(calls[0][1], dates)
    assert market.time_y == pytest.approx(np.array([0.0, 1.0, 3.0]) / 252.0)
