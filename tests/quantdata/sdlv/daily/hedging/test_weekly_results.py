from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.hedging.accounting import claim_pnl_breakdown
from quantdata.hedging.inference import EpisodeInferenceConfig
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyHedgeResult,
    weekly_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyModelRun,
    WeeklyPriceRecord,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly_results import (
    WeeklyExposure,
    WeeklyResultsConfig,
    summarize_weekly_results,
    weekly_exposure,
)


def _run(
    origin_date: date,
    *,
    payoff: float,
    spot_positions: np.ndarray | None = None,
    option_positions: np.ndarray | None = None,
) -> WeeklyModelRun:
    market_price = 0.10
    model_price = 0.11
    hedge_gain = 0.01
    transaction_cost = 0.001
    breakdown = claim_pnl_breakdown(
        claim_quantity=0.5,
        market_price=market_price,
        model_price=model_price,
        payoff=payoff,
        hedge_gain=hedge_gain,
        transaction_cost=transaction_cost,
    )
    result = WeeklyHedgeResult(
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        target_label=weekly_targets()[0].label,
        origin_date=origin_date,
        terminal_date=origin_date + timedelta(days=7),
        terminal_value=payoff,
        hedge_gain=hedge_gain,
        transaction_cost=transaction_cost,
        spot_transaction_cost=0.0004,
        option_transaction_cost=0.0006,
        spot_gain=0.004,
        option_gain=0.006,
        step_transaction_costs=np.full(6, transaction_cost / 6.0),
        step_hedge_gains=np.full(5, hedge_gain / 5.0),
        breakdown=breakdown,
    )
    prices = tuple(
        WeeklyPriceRecord(
            common_market_price_per_fit_spot=(market_price if index == 0 else None),
            q_expectation_per_fit_spot=model_price,
            buyer_indifference_per_fit_spot=None,
            seller_indifference_per_fit_spot=None,
            continuation_price_per_fit_spot=model_price,
        )
        for index in range(5)
    )
    return WeeklyModelRun(
        result=result,
        market_fit_identities=("market",) * 5,
        policy_fit_identities=("policy",) * 5,
        market_artifact_fingerprints=(("market:sha256",),) * 5,
        policy_artifact_fingerprints=(("policy:sha256",),) * 5,
        fit_as_of_dates=(origin_date.isoformat(),) * 5,
        fit_price_records=prices,
        fit_long_prices_origin_pv=np.full(5, model_price),
        spot_positions=(
            np.zeros(5) if spot_positions is None else np.asarray(spot_positions)
        ),
        option_positions=(
            np.zeros((5, 1))
            if option_positions is None
            else np.asarray(option_positions)
        ),
    )


def test_weekly_results_report_overlap_aware_risk_and_exposure() -> None:
    origins = (date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5))
    runs = tuple(
        _run(origin, payoff=payoff)
        for origin, payoff in zip(origins, (0.0, 0.2, 0.4), strict=True)
    )
    claim = weekly_targets()[0].label
    exposures = {
        (origin.isoformat(), claim): WeeklyExposure(
            total_turnover_notional=float(index + 1),
            entry_turnover_notional=0.5,
            mean_gross_exposure=2.0,
            mean_net_exposure=-0.25,
            mean_absolute_net_exposure=0.25,
            mean_nonzero_actions_per_decision=3.0,
        )
        for index, origin in enumerate(origins)
    }
    config = WeeklyResultsConfig(
        inference=EpisodeInferenceConfig(
            hac_lags=4,
            bootstrap_block_length=2,
            bootstrap_samples=100,
            bootstrap_seed=7,
        )
    )

    slices = summarize_weekly_results(
        runs,
        volatility_by_origin={
            origins[0].isoformat(): 0.10,
            origins[1].isoformat(): 0.20,
            origins[2].isoformat(): 0.30,
        },
        volatility_thresholds=(0.15, 0.25),
        exposure_by_result=exposures,
        position_cap=1_000.0,
        config=config,
    )

    overall = next(value for value in slices if value.slice_label == "overall")
    assert overall.unhedged_loss.observations == 3
    assert overall.unhedged_loss.hac_lags == 2
    assert overall.unhedged_loss.mean == pytest.approx(-0.05)
    assert overall.mean_total_turnover_notional == pytest.approx(2.0)
    assert overall.mean_entry_turnover_notional == pytest.approx(0.5)
    assert overall.mean_nonzero_actions_per_decision == pytest.approx(3.0)
    assert set(overall.pure_hedge_loss.mean_variance) == {"0.1", "1", "10"}
    assert overall.cap_hits == 0


def test_weekly_exposure_includes_rebalancing_and_terminal_liquidation() -> None:
    origin = date(2023, 1, 3)
    run = _run(
        origin,
        payoff=0.2,
        spot_positions=np.asarray([1.0, 1.0, 0.0, 0.0, 0.0]),
        option_positions=np.asarray([[1.0], [1.0], [0.0], [0.0], [0.0]]),
    )
    episode = SimpleNamespace(
        spec=SimpleNamespace(origin_date=origin),
        discount_to_origin=np.ones(6),
        spot_over_episode_origin=np.ones(6),
        option_marks_pv=np.full((6, 1), 0.2),
        option_alive=np.ones((6, 1), dtype=bool),
        option_settles_next=np.zeros((5, 1), dtype=bool),
    )

    exposure = weekly_exposure(run, episode=episode)

    assert exposure.entry_turnover_notional == pytest.approx(1.2)
    assert exposure.total_turnover_notional == pytest.approx(2.4)
    assert exposure.mean_gross_exposure == pytest.approx(0.48)
    assert exposure.mean_net_exposure == pytest.approx(0.48)
    assert exposure.mean_nonzero_actions_per_decision == pytest.approx(0.8)
