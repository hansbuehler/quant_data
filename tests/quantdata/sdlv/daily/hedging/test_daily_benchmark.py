from __future__ import annotations

from quantdata.sdlv.daily.front_spot.hedging.benchmark.daily import (
    evaluate_one_day_hedge,
    one_day_market_sample,
)
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_one_day_transition,
)


def test_one_day_benchmark_reconciles_common_accounting() -> None:
    sample = one_day_market_sample(synthetic_one_day_transition())
    result = evaluate_one_day_hedge(
        sample,
        model="test",
        measure="Q",
        spot_action=-sample.target_spot_delta,
        option_action=0.0,
        model_price=sample.market_price,
        objective_value=0.0,
        converged=True,
    )
    assert sample.payoff == 1.0
    assert result.pricing_difference == 0.0
    assert result.hedged_pnl == result.pure_hedge_error
    expected_round_trip_cost = abs(sample.target_spot_delta) * 1.0e-4 * (
        1.0
        + sample.transition.discounted_forward_over_spot
        * sample.transition.terminal_normalized_spot
    )
    assert result.transaction_cost == expected_round_trip_cost
