from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from quantdata.hedging.mean_reduction import MeanReduction
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
    _solve_power_exponential,
    fit_covariance_action,
    fit_covariance_book_actions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.pca import (
    spot_residual_pca_covariance,
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


def test_spot_residual_pca_keeps_spot_and_eight_option_factors() -> None:
    generator = np.random.default_rng(41)
    samples = generator.normal(size=(500, 13))
    samples[:, 1:] += 0.4 * samples[:, :1]
    covariance = np.cov(samples, rowvar=False) + 0.01 * np.eye(13)
    fit = spot_residual_pca_covariance(covariance, rank=8)

    np.testing.assert_allclose(fit.covariance[0], covariance[0])
    np.testing.assert_allclose(np.diag(fit.covariance), np.diag(covariance), rtol=1e-6)
    assert np.linalg.eigvalsh(fit.covariance)[0] > 0.0
    residual = (
        fit.covariance[1:, 1:]
        - fit.covariance[0, 0] * np.outer(fit.spot_beta, fit.spot_beta)
        - np.diag(fit.idiosyncratic_variance)
    )
    assert np.linalg.matrix_rank(residual, tol=1e-9) == 8
    assert 0.0 < fit.explained_residual_variance < 1.0

    cross = covariance @ generator.normal(size=13)
    projected = fit.project_claim_cross_covariance(cross)
    option_residual_cross = projected[1:] - fit.spot_beta * projected[0]
    np.testing.assert_allclose(
        option_residual_cross,
        fit.loadings @ (fit.loadings.T @ option_residual_cross),
        atol=1e-12,
    )


def test_book_pca_variant_uses_same_zero_mean_hedge_objective() -> None:
    generator = np.random.default_rng(43)
    observations = 252
    factors = generator.normal(size=(observations, 10))
    spot = factors[:, 0]
    options = 0.5 * spot[:, None] + factors[:, 1:] + 0.1 * generator.normal(
        size=(observations, 9)
    )
    target = 2.0 * spot + options[:, 0] + 0.1 * generator.normal(size=observations)
    first = date(2022, 1, 3)
    panel = HistoricalReturnPanel(
        training_origin_dates=tuple(first + timedelta(days=index) for index in range(observations)),
        training_terminal_dates=tuple(first + timedelta(days=index + 1) for index in range(observations)),
        instrument_ids=("spot", *(f"option-{index}" for index in range(9))),
        target_labels=("synthetic-target",),
        instrument_returns=np.column_stack((spot, options)),
        target_returns=target[:, None],
    )
    fit = fit_covariance_book_actions(
        panel,
        linear_transaction_costs=np.zeros(10),
        config=CovarianceHedgeConfig(
            rolling_observations=observations,
            objective=CovarianceObjective.MEAN_VARIANCE,
            risk_parameter=1.0,
            cost_multiplier=0.0,
            spot_residual_pca_rank=8,
        ),
    )
    assert fit.spot_residual_pca_rank == 8
    assert fit.explained_residual_variance is not None
    assert fit.actions[0].converged
    assert fit.actions[0].action is not None
    assert abs(fit.actions[0].action[0]) > 0.1


def test_covariance_mean_reduction_labels_are_analytically_equivalent() -> None:
    panel = _one_factor_return_panel()
    fits = tuple(
        fit_covariance_book_actions(
            panel,
            linear_transaction_costs=np.zeros(1),
            config=CovarianceHedgeConfig(
                rolling_observations=len(panel.training_origin_dates),
                objective=CovarianceObjective.MEAN_VARIANCE,
                risk_parameter=1.0,
                cost_multiplier=0.0,
                mean_reduction=method,
            ),
        )
        for method in MeanReduction
    )

    np.testing.assert_allclose(fits[0].actions[0].action, fits[1].actions[0].action)
    assert fits[0].actions[0].objective == pytest.approx(
        fits[1].actions[0].objective
    )


def test_virtual_policy_solver_cap_is_distinct_from_executable_cap() -> None:
    generator = np.random.default_rng(31)
    observations = 500
    centered_instrument = generator.normal(size=observations)
    instrument = 2_000.0 + centered_instrument
    target = 20.0 * centered_instrument + 0.01 * generator.normal(size=observations)
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
    assert action.zero_policy.action[0] == pytest.approx(0.0)
    assert abs(action.action[0]) == pytest.approx(10.0)
    assert action.internal_cap_hits == 0
    assert action.executable_cap_hits == 1
    assert action.cap_hits == 1
    assert not action.converged


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


def test_gaussian_entropy_is_mean_variance_with_the_same_risk_parameter() -> None:
    panel = _one_factor_return_panel()
    common = dict(
        rolling_observations=len(panel.training_origin_dates),
        risk_parameter=1.0,
        cost_multiplier=0.0,
    )
    mean_variance = fit_covariance_book_actions(
        panel,
        linear_transaction_costs=np.zeros(1),
        config=CovarianceHedgeConfig(
            objective=CovarianceObjective.MEAN_VARIANCE,
            **common,
        ),
    )
    entropy = fit_covariance_book_actions(
        panel,
        linear_transaction_costs=np.zeros(1),
        config=CovarianceHedgeConfig(
            objective=CovarianceObjective.ENTROPY,
            **common,
        ),
    )

    np.testing.assert_allclose(
        entropy.actions[0].action,
        mean_variance.actions[0].action,
    )
    assert entropy.actions[0].objective == pytest.approx(
        mean_variance.actions[0].objective
    )


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


def test_gaussian_power_risk_is_finite_and_monotone_at_extreme_scales() -> None:
    for risk_parameter in np.logspace(-3, 3, 7):
        previous_value = -1.0
        for standard_deviation in np.concatenate(([0.0], np.logspace(-6, 2, 9))):
            value, derivative, translation = _gaussian_power_risk(
                float(standard_deviation),
                risk_parameter=float(risk_parameter),
            )
            assert np.all(np.isfinite((value, derivative, translation)))
            assert value >= previous_value - 1.0e-10
            assert derivative >= 0.0
            previous_value = value


def test_gaussian_power_solver_matches_direct_scalar_spread_problem() -> None:
    covariance = np.asarray([[0.04]], dtype=np.float64)
    claim_cross_covariance = np.asarray([0.012], dtype=np.float64)
    claim_variance = 0.09
    instrument_center = np.asarray([0.006], dtype=np.float64)
    penalty = np.asarray([0.0015], dtype=np.float64)
    config = CovarianceHedgeConfig(
        objective=CovarianceObjective.POWER_EXPONENTIAL,
        risk_parameter=2.0,
        position_cap=10.0,
        solver_position_cap=10.0,
    )

    action, _, converged, cap_hits = _solve_power_exponential(
        instrument_center=instrument_center,
        covariance=covariance,
        claim_cross_covariance=claim_cross_covariance,
        claim_variance=claim_variance,
        penalty=penalty,
        config=config,
    )

    def objective(action_value: float) -> float:
        variance = (
            claim_variance
            + 2.0 * claim_cross_covariance[0] * action_value
            + covariance[0, 0] * action_value**2
        )
        risk = _gaussian_power_risk(
            math.sqrt(max(variance, 0.0)),
            risk_parameter=config.risk_parameter,
        )[0]
        return (
            -instrument_center[0] * action_value
            + penalty[0] * abs(action_value)
            + risk
        )

    numerical = minimize_scalar(
        objective,
        bounds=(-config.solver_position_cap, config.solver_position_cap),
        method="bounded",
        options={"xatol": 1.0e-12},
    )

    assert numerical.success
    assert converged
    assert cap_hits == 0
    assert action[0] == pytest.approx(float(numerical.x), abs=1.0e-7)
    assert objective(action[0]) == pytest.approx(float(numerical.fun), abs=1.0e-11)


def test_mean_volatility_zero_drift_removes_speculative_recession() -> None:
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
    assert action.bounded
    assert action.zero_policy.bounded
    assert action.action is not None
    assert action.zero_policy.action == pytest.approx(np.zeros(1))
    assert action.zero_policy.expected_profit == pytest.approx(0.0)
    assert action.zero_policy.recession_ratio == pytest.approx(0.0)
    assert action.cap_hits == 0
    assert action.converged


def test_expected_return_sample_mean_is_diagnostic_only() -> None:
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
    assert fit.actions[0].zero_policy.action == pytest.approx(np.zeros(1))
    assert fit.actions[0].zero_policy.expected_profit == pytest.approx(0.0)


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
