"""The prior-day fit panel excludes today's fitted percentage."""

import numpy as np
import pandas as pd
import pytest

from quantdata.dysanos.v11.daily_fit_report import daily_comparison


def test_comparison_excludes_today_and_weights_daily_percentages():
    frame = pd.DataFrame({"date": ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
                          "inside_percent": [20., 40., 60., 99.], "quotes": [1, 10000, 2, 100000]})
    result = daily_comparison(frame, 123.)
    weights = np.array([.25, .25**(2/3), .25**(1/3)])
    assert result["previous_weighted_mean_fit_percent"] == pytest.approx(np.average([20, 40, 60], weights=weights))
    assert result["previous_weighted_median_fit_percent"] == 40.
    assert result["today_fit_percent"] == 99.
    assert result["runtime_minutes"] == pytest.approx(2.05)
    frame.loc[3, "inside_percent"] = 1.
    changed = daily_comparison(frame, 123.)
    assert changed["previous_weighted_mean_fit_percent"] == result["previous_weighted_mean_fit_percent"]
    assert changed["previous_weighted_median_fit_percent"] == result["previous_weighted_median_fit_percent"]
