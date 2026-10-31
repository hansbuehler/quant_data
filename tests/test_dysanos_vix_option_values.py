"""Calendar units and antithetic option-payoff error regressions."""

import numpy as np
import polars as pl
import pytest
import torch

from quantdata.biz_calendar import count_business_days_between
from quantdata.dysanos.v11.price_vix_options import (
    calendar_index_from_total_variance,
    paired_payoff_statistics,
    parity_forward,
)


def test_calendar_annualization_interpolates_total_variance_not_volatility() -> None:
    nodes = np.array([20., 25.])
    tv = torch.tensor([[.003, .008], [.006, .012]], dtype=torch.float64)
    actual = calendar_index_from_total_variance(tv, model_business_days=nodes, residual_business_days=22)
    expected = 100 * torch.sqrt((.6 * tv[:, 0] + .4 * tv[:, 1]) * 365 / 30)
    torch.testing.assert_close(actual, expected)


def test_exact_last_node_and_no_extrapolation() -> None:
    tv = torch.tensor([[.001, .002]], dtype=torch.float64)
    result = calendar_index_from_total_variance(tv, model_business_days=np.array([20., 25.]), residual_business_days=25)
    assert float(result[0]) == pytest.approx(100 * np.sqrt(.002 * 365 / 30))
    with pytest.raises(ValueError):
        calendar_index_from_total_variance(tv, model_business_days=np.array([20., 25.]), residual_business_days=26)


def test_antithetic_errors_use_pair_means() -> None:
    values = np.array([10., 20., 30., 40.])
    result = paired_payoff_statistics(values, strike=20, is_call=True, discount_factor=1.)
    assert result["model_value"] == pytest.approx(7.5)
    assert result["mc_se"] == pytest.approx(2.5)
    assert result["independent_pairs"] == 2
    assert not result["mc_tail_resolved"]


def test_empirical_conditional_put_call_parity() -> None:
    values = np.array([10., 20., 30., 40.])
    call = paired_payoff_statistics(values, strike=22, is_call=True, discount_factor=.98)
    put = paired_payoff_statistics(values, strike=22, is_call=False, discount_factor=.98)
    assert call["model_value"] - put["model_value"] == pytest.approx(.98 * (values.mean() - 22))


def test_zero_sample_payoffs_are_not_marked_tail_resolved() -> None:
    result = paired_payoff_statistics(np.ones(40) * 15, strike=100, is_call=True, discount_factor=.99)
    assert result["model_value"] == result["mc_se"] == 0
    assert not result["mc_tail_resolved"]


def test_market_forward_uses_call_put_parity_not_vix_spot() -> None:
    quotes = pl.DataFrame({"strike_price": [15., 15., 20., 20.], "cp_flag": ["C", "P", "C", "P"],
                           "best_bid": [7.9, .9, 3.9, 1.9], "best_offer": [8.1, 1.1, 4.1, 2.1]})
    assert parity_forward(quotes, discount_factor=1.) == pytest.approx(22.)


@pytest.mark.parametrize("values", [[10., 20., 30.], [10., np.nan, 20., 30.], [-1., 10., 20., 30.]])
def test_invalid_terminal_samples_rejected(values: list[float]) -> None:
    with pytest.raises(ValueError):
        paired_payoff_statistics(np.asarray(values), strike=20, is_call=True, discount_factor=1.)


def test_september_business_horizons_include_labor_day_holiday() -> None:
    from datetime import date

    assert count_business_days_between(date(2025, 8, 29), date(2025, 9, 24)) == 17
    assert count_business_days_between(date(2025, 8, 29), date(2025, 10, 1)) == 22
