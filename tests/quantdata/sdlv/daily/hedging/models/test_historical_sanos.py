from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from quantdata.hedging.contracts import OptionRight
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalTarget,
    build_daily_digital_sample,
)
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_one_day_transition,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.returns import (
    HistoricalReturnPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.adapter import (
    evaluate_historical_sanos_daily_book,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.hedger import (
    fit_historical_sanos_hedge,
)


@pytest.mark.parametrize("objective", tuple(HistoricalSanosObjective))
def test_historical_sanos_fits_all_objectives(objective: HistoricalSanosObjective) -> None:
    generator = np.random.default_rng(7)
    instruments = generator.normal(size=(120, 3))
    claim = -(0.4 * instruments[:, 0] - 0.2 * instruments[:, 1])
    config = HistoricalSanosConfig(
        rolling_observations=120,
        objective=objective,
        risk_parameter=1.0,
        maximum_iterations=1_000,
    )
    fit = fit_historical_sanos_hedge(
        instrument_returns=instruments,
        claim_returns=claim,
        linear_transaction_costs=np.zeros(3),
        config=config,
    )

    assert fit.converged
    assert fit.action[0] == pytest.approx(0.4, abs=3.0e-3)
    assert fit.action[1] == pytest.approx(-0.2, abs=3.0e-3)
    assert fit.action[2] == pytest.approx(0.0, abs=3.0e-3)


def test_historical_sanos_mean_volatility_solves_cost_lasso_socp() -> None:
    generator = np.random.default_rng(17)
    instruments = generator.normal(scale=0.01, size=(252, 12))
    claim = -(0.7 * instruments[:, 0] - 0.3 * instruments[:, 4])
    costs = np.full(12, 1.0e-5)
    config = HistoricalSanosConfig(
        rolling_observations=252,
        objective=HistoricalSanosObjective.MEAN_VOLATILITY,
        risk_parameter=1.0,
        maximum_iterations=10_000,
        gradient_tolerance=1.0e-9,
    )

    fit = fit_historical_sanos_hedge(
        instrument_returns=instruments,
        claim_returns=claim,
        linear_transaction_costs=costs,
        config=config,
    )

    assert fit.converged
    assert fit.cap_hits == 0
    assert fit.claim_policy.maximum_absolute_gradient < 1.0e-6
    assert fit.zero_policy.maximum_absolute_gradient < 1.0e-6


def test_historical_sanos_mean_volatility_reports_recession_cap() -> None:
    generator = np.random.default_rng(23)
    observations = 252
    instruments = (0.1 + 0.001 * generator.normal(size=observations))[:, None]
    fit = fit_historical_sanos_hedge(
        instrument_returns=instruments,
        claim_returns=np.zeros(observations),
        linear_transaction_costs=np.zeros(1),
        config=HistoricalSanosConfig(
            rolling_observations=observations,
            objective=HistoricalSanosObjective.MEAN_VOLATILITY,
            risk_parameter=0.1,
            maximum_iterations=10_000,
            position_cap=10.0,
            solver_position_cap=100.0,
        ),
    )

    assert fit.claim_policy.action[0] == pytest.approx(100.0, abs=1.0e-5)
    assert fit.zero_policy.action[0] == pytest.approx(100.0, abs=1.0e-5)
    assert fit.internal_cap_hits == 2
    assert fit.executable_cap_hits == 0
    assert fit.action[0] == pytest.approx(0.0, abs=1.0e-5)
    assert not fit.converged


def test_virtual_policy_solver_cap_is_distinct_from_executable_cap() -> None:
    generator = np.random.default_rng(31)
    observations = 252
    centered_instrument = generator.normal(size=observations)
    instruments = (2_000.0 + centered_instrument)[:, None]
    claim = 2.0 * centered_instrument + 0.01 * generator.normal(size=observations)
    fit = fit_historical_sanos_hedge(
        instrument_returns=instruments,
        claim_returns=claim,
        linear_transaction_costs=np.zeros(1),
        config=HistoricalSanosConfig(
            rolling_observations=observations,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=1.0,
            maximum_iterations=10_000,
            position_cap=10.0,
            solver_position_cap=10_000.0,
        ),
    )

    assert abs(fit.claim_policy.action[0]) > 10.0
    assert abs(fit.zero_policy.action[0]) > 10.0
    assert abs(fit.action[0]) < 10.0
    assert fit.internal_cap_hits == 0
    assert fit.executable_cap_hits == 0
    assert fit.cap_hits == 0
    assert fit.converged


def test_executable_cap_is_enforced_after_policy_subtraction() -> None:
    generator = np.random.default_rng(47)
    observations = 252
    instruments = generator.normal(size=(observations, 1))
    claim = 20.0 * instruments[:, 0] + 0.01 * generator.normal(size=observations)
    fit = fit_historical_sanos_hedge(
        instrument_returns=instruments,
        claim_returns=claim,
        linear_transaction_costs=np.zeros(1),
        config=HistoricalSanosConfig(
            rolling_observations=observations,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=1.0,
            maximum_iterations=10_000,
            position_cap=10.0,
            solver_position_cap=10_000.0,
        ),
    )

    assert fit.action[0] == pytest.approx(-10.0)
    assert fit.internal_cap_hits == 0
    assert fit.executable_cap_hits == 1
    assert fit.cap_hits == 1
    assert not fit.converged


class _OneDayPanel:
    def __init__(self) -> None:
        self.transition = synthetic_one_day_transition()
        self.endpoint = replace(
            self.transition.origin,
            trading_date=self.transition.next_date,
        )

    def one_day_transition(self, origin_date):
        assert origin_date == self.transition.origin.trading_date
        return self.transition

    def market_day(self, trading_date):
        assert trading_date == self.transition.next_date
        return self.endpoint


def test_historical_sanos_daily_book_reuses_one_aligned_return_panel() -> None:
    panel = _OneDayPanel()
    targets = (
        DailyDigitalTarget(
            maturity_business_days=1,
            right=OptionRight.PUT,
            nms=0.0,
        ),
        DailyDigitalTarget(
            maturity_business_days=1,
            right=OptionRight.CALL,
            nms=0.0,
        ),
    )
    samples = tuple(
        build_daily_digital_sample(
            panel,
            origin_date=panel.transition.origin.trading_date,
            target=target,
        )
        for target in targets
    )
    generator = np.random.default_rng(11)
    instrument_returns = generator.normal(scale=0.01, size=(120, 2))
    target_returns = np.column_stack(
        (
            -0.3 * instrument_returns[:, 0],
            0.2 * instrument_returns[:, 0] - 0.1 * instrument_returns[:, 1],
        )
    )
    dates = tuple(
        panel.transition.origin.trading_date
        for _ in range(len(instrument_returns))
    )
    terminal_dates = tuple(
        panel.transition.next_date for _ in range(len(instrument_returns))
    )
    returns = HistoricalReturnPanel(
        training_origin_dates=dates,
        training_terminal_dates=terminal_dates,
        instrument_ids=(
            "spot",
            samples[0].hedge_quotes[0].contract.contract_id,
        ),
        target_labels=tuple(target.label for target in targets),
        instrument_returns=instrument_returns,
        target_returns=target_returns,
    )
    results = evaluate_historical_sanos_daily_book(
        samples,
        returns=returns,
        config=HistoricalSanosConfig(
            rolling_observations=120,
            maximum_iterations=2_000,
        ),
    )
    assert tuple(result.claim for result in results) == tuple(
        target.label for target in targets
    )
    assert all(result.diagnostics["target_count"] == 2 for result in results)


def test_historical_sanos_daily_book_rejects_misaligned_targets() -> None:
    panel = _OneDayPanel()
    target = DailyDigitalTarget(
        maturity_business_days=1,
        right=OptionRight.CALL,
        nms=0.0,
    )
    sample = build_daily_digital_sample(
        panel,
        origin_date=panel.transition.origin.trading_date,
        target=target,
    )
    returns = HistoricalReturnPanel(
        training_origin_dates=(panel.transition.origin.trading_date,) * 2,
        training_terminal_dates=(panel.transition.next_date,) * 2,
        instrument_ids=(
            "spot",
            sample.hedge_quotes[0].contract.contract_id,
        ),
        target_labels=("wrong-target",),
        instrument_returns=np.zeros((2, 2)),
        target_returns=np.zeros((2, 1)),
    )
    with pytest.raises(ValueError, match="return targets do not align"):
        evaluate_historical_sanos_daily_book((sample,), returns=returns)
