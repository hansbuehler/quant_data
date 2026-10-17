"""Checks for the statistical one-day digital comparison."""

from __future__ import annotations

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.reports.statistical_1d_matrix import (
    _best_half,
)


def test_best_half_uses_separate_mean_and_median_rankings() -> None:
    cases = {
        f"ordinary-{index}": np.full((10, 1), float(index))
        for index in range(1, 13)
    }
    cases["rare-tail"] = np.array([[100.0]] + [[0.0]] * 9)

    mean_selected, mean_values = _best_half(cases, statistic="mean")
    median_selected, median_values = _best_half(cases, statistic="median")

    assert len(mean_selected) == len(median_selected) == 7
    assert "rare-tail" not in mean_selected
    assert "rare-tail" in median_selected
    assert np.isclose(mean_values[0], 4.0)
    assert np.isclose(median_values[0], 3.0)
