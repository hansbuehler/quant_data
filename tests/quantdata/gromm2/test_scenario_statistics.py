from __future__ import annotations

import numpy as np
import pytest

from quantdata.gromm2.forward_hedging import (
    instrument_book,
    sample_summary,
    unconditional_price_diagnostics,
)
from quantdata.gromm2.scenario_statistics import PairMoments, ScenarioStatistics


def test_incremental_pair_moments_match_full_population_with_three_error_band() -> None:
    values = np.random.default_rng(31).normal(size=28)
    moments = PairMoments()
    for chunk in np.array_split(values, 7):
        moments.update(chunk)
    report = moments.scalar_summary()
    expected = sample_summary(values)
    assert report["mean"] == pytest.approx(expected["mean"])
    assert report["standard_error"] == pytest.approx(expected["standard_error"])
    assert report["three_se_lower"] == pytest.approx(report["mean"] - 3.0 * report["standard_error"])


def test_streaming_audits_match_the_existing_in_memory_screens() -> None:
    generator = np.random.default_rng(42)
    prices = np.exp(generator.normal(scale=0.01, size=(16, 21, 23)))
    payoff = np.abs(prices[:, 20, 0] / prices[:, 10, 0] - 1.0)
    statistics = ScenarioStatistics(book=instrument_book(), days=20)
    for start in range(0, 16, 4):
        statistics.update(prices[start:start + 4], payoff[start:start + 4])
    expected = unconditional_price_diagnostics(prices, instrument_book())
    actual = statistics.audit()
    for name in ("raw_daily_gains", "spot_scaled_daily_gains", "option_to_expiry_gains"):
        for key in ("coordinate_count", "independent_pairs", "maximum_absolute_statistic",
                    "percentile_99_absolute_statistic", "count_above_1_96", "count_above_2_576"):
            assert actual[name][key] == pytest.approx(expected[name][key])
    assert statistics.exploration()["minimum"] == prices[:, :, 0].min()


def test_incomplete_antithetic_chunks_are_rejected() -> None:
    with pytest.raises(ValueError, match="antithetic"):
        PairMoments().update(np.ones(3))
