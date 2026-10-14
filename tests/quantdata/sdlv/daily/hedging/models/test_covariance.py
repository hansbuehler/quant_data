from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from quantdata.hedging.objectives import regularized_tyler_covariance
from quantdata.sdlv.daily.front_spot.hedging.benchmark.daily import one_day_market_sample
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_one_day_transition,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.config import (
    CovarianceHedgeConfig,
    CovarianceObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.hedger import (
    _gaussian_power_risk,
    _robust_target_scatter,
    fit_covariance_action,
    fit_covariance_book_actions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.returns import (
    HistoricalReturnPanel,
)


def test_covariance_model_uses_both_spot_and_option_returns() -> None:
    base = one_day_market_sample(synthetic_one_day_transition())
    generator = np.random.default_rng(17)
    samples = []
    for _ in range(100):
        spot = float(generator.normal(scale=0.01))
        option = float(0.5 * spot + generator.normal(scale=0.005))
        claim = float(2.0 * spot - option + generator.normal(scale=0.002))
        samples.append(
            replace(
                base,
                spot_return=spot,
                option_return=option,
                claim_return=claim,
            )
        )
    action = fit_covariance_action(
        tuple(samples),
        current_option_half_spread=base.hedge_option.half_spread,
        config=CovarianceHedgeConfig(
            rolling_observations=100,
            cost_multiplier=0.0,
        ),
    )
    assert action.fit.converged
    assert abs(action.spot) > 0.1
    assert abs(action.option) > 0.1


def _one_factor_return_panel(*, observations: int = 500) -> HistoricalReturnPanel:
    generator = np.random.default_rng(23)
    instrument = generator.normal(size=observations)
    target = 2.0 * instrument + 0.05 * generator.normal(size=observations)
    first = date(2022, 1, 3)
    return HistoricalReturnPanel(
        training_origin_dates=tuple(
            first + timedelta(days=index) for index in range(observations)
        ),
        training_terminal_dates=tuple(
            first + timedelta(days=index + 1) for index in range(observations)
        ),
        instrument_ids=("spot",),
        target_labels=("synthetic-target",),
        instrument_returns=instrument[:, None],
        target_returns=target[:, None],
    )


def test_book_mean_variance_recovers_pure_factor_hedge() -> None:
    panel = _one_factor_return_panel()
    fit = fit_covariance_book_actions(
        panel,
        linear_transaction_costs=np.zeros(1),
        config=CovarianceHedgeConfig(
            rolling_observations=len(panel.training_origin_dates),
            objective=CovarianceObjective.MEAN_VARIANCE,
            risk_parameter=1.0,
            cost_multiplier=0.0,
        ),
    )
    action = fit.actions[0]
    assert action.converged
    assert action.action is not None
    np.testing.assert_allclose(action.action, [-2.0], atol=0.08)
    assert action.standard_deviation < 0.2 * np.sqrt(fit.estimate.covariance[-1, -1])


def test_virtual_policy_solver_cap_is_distinct_from_executable_cap() -> None:
    generator = np.random.default_rng(31)
    observations = 500
    centered_instrument = generator.normal(size=observations)
    instrument = 2_000.0 + centered_instrument
    target = 2.0 * centered_instrument + 0.01 * generator.normal(size=observations)
    first = date(2022, 1, 3)
    panel = HistoricalReturnPanel(
        training_origin_dates=tuple(
            first + timedelta(days=index) for index in range(observations)
        ),
        training_terminal_dates=tuple(
            first + timedelta(days=index + 1) for index in range(observations)
        ),
        instrument_ids=("spot",),
        target_labels=("synthetic-target",),
        instrument_returns=instrument[:, None],
        target_returns=target[:, None],
    )
    fit = fit_covariance_book_actions(
        panel,
        linear_transaction_costs=np.zeros(1),
        config=CovarianceHedgeConfig(
            rolling_observations=observations,
            objective=CovarianceObjective.MEAN_VARIANCE,
            risk_parameter=1.0,
            cost_multiplier=0.0,
            mean_z_threshold=0.0,
            position_cap=10.0,
            solver_position_cap=10_000.0,
        ),
    )

    action = fit.actions[0]
    assert action.action is not None
    assert abs(action.claim_policy.action[0]) > 10.0
    assert abs(action.zero_policy.action[0]) > 10.0
    assert abs(action.action[0]) < 10.0
    assert action.internal_cap_hits == 0
    assert action.executable_cap_hits == 0
    assert action.cap_hits == 0
    assert action.converged


@pytest.mark.parametrize("risk_parameter", [0.1, 1.0, 2.0, 3.0])
def test_book_mean_volatility_recovers_pure_factor_hedge(
    risk_parameter: float,
) -> None:
    panel = _one_factor_return_panel()
    fit = fit_covariance_book_actions(
        panel,
        linear_transaction_costs=np.zeros(1),
        config=CovarianceHedgeConfig(
            rolling_observations=len(panel.training_origin_dates),
            objective=CovarianceObjective.MEAN_VOLATILITY,
            risk_parameter=risk_parameter,
            cost_multiplier=0.0,
        ),
    )
    action = fit.actions[0]
    assert action.bounded
    assert action.converged
    assert action.action is not None
    np.testing.assert_allclose(action.action, [-2.0], atol=0.08)
    assert action.standard_deviation < 0.2 * np.sqrt(fit.estimate.covariance[-1, -1])


@pytest.mark.parametrize("risk_parameter", [0.1, 1.0, 10.0])
def test_book_gaussian_power_recovers_pure_factor_hedge(
    risk_parameter: float,
) -> None:
    panel = _one_factor_return_panel()
    fit = fit_covariance_book_actions(
        panel,
        linear_transaction_costs=np.zeros(1),
        config=CovarianceHedgeConfig(
            rolling_observations=len(panel.training_origin_dates),
            objective=CovarianceObjective.POWER_EXPONENTIAL,
            risk_parameter=risk_parameter,
            cost_multiplier=0.0,
        ),
    )
    action = fit.actions[0]
    assert action.bounded
    assert action.converged
    assert action.action is not None
    np.testing.assert_allclose(action.action, [-2.0], atol=0.08)
    assert action.standard_deviation < 0.2 * np.sqrt(fit.estimate.covariance[-1, -1])


@pytest.mark.parametrize(
    ("risk_parameter", "standard_deviation"),
    ((0.1, 0.01), (1.0, 0.1), (10.0, 1.0)),
)
def test_gaussian_power_risk_has_correct_slope(
    risk_parameter: float,
    standard_deviation: float,
) -> None:
    value, derivative, _ = _gaussian_power_risk(
        standard_deviation,
        risk_parameter=risk_parameter,
    )
    step = 1.0e-6 * max(1.0, standard_deviation)
    upper = _gaussian_power_risk(
        standard_deviation + step,
        risk_parameter=risk_parameter,
    )[0]
    lower = _gaussian_power_risk(
        standard_deviation - step,
        risk_parameter=risk_parameter,
    )[0]

    assert value > 0.0
    assert derivative == pytest.approx((upper - lower) / (2.0 * step), rel=1.0e-7)


def test_gaussian_power_risk_has_mean_variance_small_risk_limit() -> None:
    risk_parameter = 0.2
    standard_deviation = 1.0e-3
    value, derivative, _ = _gaussian_power_risk(
        standard_deviation,
        risk_parameter=risk_parameter,
    )

    assert value == pytest.approx(
        0.5 * risk_parameter * standard_deviation**2,
        rel=1.0e-4,
    )
    assert derivative == pytest.approx(
        risk_parameter * standard_deviation,
        rel=1.0e-4,
    )


def test_gaussian_power_risk_matches_direct_normal_integration() -> None:
    risk_parameter = 2.0
    standard_deviation = 0.35

    def utility(value: float) -> float:
        if value >= 0.0:
            return -math.expm1(-risk_parameter * value) / risk_parameter
        return value - 0.5 * risk_parameter * value**2

    def normal_density(value: float) -> float:
        return math.exp(-0.5 * value**2) / math.sqrt(2.0 * math.pi)

    def oce(translation: float) -> float:
        split = translation / standard_deviation
        expectation = quad(
            lambda value: utility(
                standard_deviation * value - translation
            )
            * normal_density(value),
            -math.inf,
            split,
            epsabs=1.0e-12,
        )[0] + quad(
            lambda value: utility(
                standard_deviation * value - translation
            )
            * normal_density(value),
            split,
            math.inf,
            epsabs=1.0e-12,
        )[0]
        return translation + expectation

    numerical = minimize_scalar(
        lambda translation: -oce(float(translation)),
        bracket=(-1.0, 1.0),
        tol=1.0e-12,
    )
    value, _, translation = _gaussian_power_risk(
        standard_deviation,
        risk_parameter=risk_parameter,
    )

    assert numerical.success
    assert value == pytest.approx(float(numerical.fun), abs=1.0e-12)
    assert translation == pytest.approx(float(numerical.x), abs=1.0e-8)


def test_mean_volatility_caps_and_flags_unbounded_policy() -> None:
    generator = np.random.default_rng(29)
    observations = 500
    instrument = 0.1 + 0.01 * generator.normal(size=observations)
    target = instrument + 0.01 * generator.normal(size=observations)
    first = date(2022, 1, 3)
    panel = HistoricalReturnPanel(
        training_origin_dates=tuple(
            first + timedelta(days=index) for index in range(observations)
        ),
        training_terminal_dates=tuple(
            first + timedelta(days=index + 1) for index in range(observations)
        ),
        instrument_ids=("spot",),
        target_labels=("synthetic-target",),
        instrument_returns=instrument[:, None],
        target_returns=target[:, None],
    )
    fit = fit_covariance_book_actions(
        panel,
        linear_transaction_costs=np.zeros(1),
        config=CovarianceHedgeConfig(
            rolling_observations=observations,
            objective=CovarianceObjective.MEAN_VOLATILITY,
            risk_parameter=1.0,
            cost_multiplier=0.0,
            mean_z_threshold=0.0,
            position_cap=10.0,
        ),
    )
    action = fit.actions[0]
    assert not action.bounded
    assert not action.zero_policy.bounded
    assert action.action is not None
    assert action.zero_policy.iterations > 0
    assert action.cap_hits > 0
    assert not action.converged


def test_expected_return_uses_mean_not_componentwise_median() -> None:
    observations = 100
    instrument = np.concatenate((np.zeros(90), np.full(10, 10.0)))
    target = 0.5 * instrument
    first = date(2022, 1, 3)
    panel = HistoricalReturnPanel(
        training_origin_dates=tuple(
            first + timedelta(days=index) for index in range(observations)
        ),
        training_terminal_dates=tuple(
            first + timedelta(days=index + 1) for index in range(observations)
        ),
        instrument_ids=("spot",),
        target_labels=("skewed-target",),
        instrument_returns=instrument[:, None],
        target_returns=target[:, None],
    )
    fit = fit_covariance_book_actions(
        panel,
        linear_transaction_costs=np.zeros(1),
        config=CovarianceHedgeConfig(
            rolling_observations=observations,
            objective=CovarianceObjective.MEAN_VARIANCE,
            mean_z_threshold=0.0,
            mean_hac_lags=0,
        ),
    )

    assert np.median(instrument) == 0.0
    assert fit.raw_instrument_center_norm == pytest.approx(np.mean(instrument))
    assert fit.screened_instrument_center_norm == pytest.approx(np.mean(instrument))


def test_target_action_is_invariant_to_unrelated_target_catalog() -> None:
    panel = _one_factor_return_panel()
    unrelated = np.random.default_rng(71).standard_t(
        df=3.0,
        size=len(panel.training_origin_dates),
    )
    catalog = replace(
        panel,
        target_labels=("synthetic-target", "unrelated-target"),
        target_returns=np.column_stack((panel.target_returns[:, 0], unrelated)),
    )
    config = CovarianceHedgeConfig(
        rolling_observations=len(panel.training_origin_dates),
        objective=CovarianceObjective.MEAN_VARIANCE,
        risk_parameter=1.0,
        cost_multiplier=0.0,
    )
    single_fit = fit_covariance_book_actions(
        panel,
        linear_transaction_costs=np.zeros(1),
        config=config,
    )
    catalog_fit = fit_covariance_book_actions(
        catalog,
        linear_transaction_costs=np.zeros(1),
        config=config,
    )

    assert single_fit.estimate.shrinkage == catalog_fit.estimate.shrinkage
    assert single_fit.actions[0].action == pytest.approx(catalog_fit.actions[0].action)
    assert single_fit.actions[0].objective == pytest.approx(
        catalog_fit.actions[0].objective
    )


def test_target_scatter_block_is_positive_definite() -> None:
    panel = _one_factor_return_panel()
    estimate = regularized_tyler_covariance(panel.instrument_returns)
    cross_covariance, target_variance = _robust_target_scatter(
        panel.instrument_returns,
        panel.target_returns[:, 0],
        estimate=estimate,
    )
    joint = np.block(
        [
            [estimate.covariance, cross_covariance[:, None]],
            [cross_covariance[None, :], np.asarray([[target_variance]])],
        ]
    )

    assert np.min(np.linalg.eigvalsh(joint)) > 0.0
