from __future__ import annotations

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

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
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos import hedger
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.adapter import (
    evaluate_historical_sanos_daily_book,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.hedger import (
    _fit_policy,
    _objective_and_gradient,
    _optimal_power_cash_translation,
    _refine_mean_variance_coordinate_descent,
    fit_historical_sanos_hedge,
    fit_historical_sanos_hedges,
)


def test_historical_sanos_defaults_use_regularized_one_year_history() -> None:
    config = HistoricalSanosConfig()

    assert config.rolling_observations == 252
    assert config.quadratic_ridge == pytest.approx(1.0e-4)
    assert config.certificate_tolerance == pytest.approx(1.0e-6)


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


@pytest.mark.parametrize(
    "objective,risk_parameter",
    (
        (HistoricalSanosObjective.ENTROPY, 10.0),
        (HistoricalSanosObjective.POWER_EXPONENTIAL, 1.0),
    ),
)
def test_historical_sanos_exact_cost_oce_converges_on_redundant_book(
    objective: HistoricalSanosObjective,
    risk_parameter: float,
) -> None:
    generator = np.random.default_rng(71)
    common = generator.normal(scale=0.01, size=(120, 4))
    instruments = np.column_stack(
        (
            common,
            common[:, 0] + 1.0e-3 * common[:, 1],
            common[:, 2] - 1.0e-3 * common[:, 3],
            generator.normal(scale=1.0e-4, size=(120, 6)),
        )
    )
    claims = np.column_stack(
        (
            -0.4 * common[:, 0] + 0.2 * common[:, 2],
            0.3 * common[:, 1] - 0.1 * common[:, 3],
        )
    )
    config = HistoricalSanosConfig(
        rolling_observations=120,
        objective=objective,
        risk_parameter=risk_parameter,
        maximum_iterations=10_000,
        gradient_tolerance=1.0e-8,
    )

    fits = fit_historical_sanos_hedges(
        instrument_returns=instruments,
        claim_returns=claims,
        linear_transaction_costs=np.full(instruments.shape[1], 1.0e-5),
        config=config,
    )

    assert len(fits) == 2
    assert fits[0].zero_policy is fits[1].zero_policy
    assert all(fit.converged for fit in fits)
    assert all(fit.cap_hits == 0 for fit in fits)
    assert all(
        fit.claim_policy.maximum_absolute_gradient <= config.gradient_tolerance
        for fit in fits
    )
    assert fits[0].zero_policy.maximum_absolute_gradient <= config.gradient_tolerance


@pytest.mark.parametrize("objective", tuple(HistoricalSanosObjective))
def test_historical_sanos_standardized_ridge_is_scale_invariant(
    objective: HistoricalSanosObjective,
) -> None:
    generator = np.random.default_rng(83)
    instruments = generator.normal(size=(160, 3)) * np.asarray([0.01, 0.03, 0.002])
    claim = -(0.35 * instruments[:, 0] - 0.15 * instruments[:, 1])
    costs = np.asarray([2.0e-5, 4.0e-5, 1.0e-5])
    coordinate_rescaling = np.asarray([0.1, 25.0, 4.0])
    config = HistoricalSanosConfig(
        rolling_observations=160,
        objective=objective,
        risk_parameter=2.0,
        maximum_iterations=10_000,
    )

    base = fit_historical_sanos_hedge(
        instrument_returns=instruments,
        claim_returns=claim,
        linear_transaction_costs=costs,
        config=config,
    )
    rescaled = fit_historical_sanos_hedge(
        instrument_returns=instruments * coordinate_rescaling,
        claim_returns=claim,
        linear_transaction_costs=costs * coordinate_rescaling,
        config=config,
    )

    assert base.converged
    assert rescaled.converged
    np.testing.assert_allclose(
        rescaled.action * coordinate_rescaling,
        base.action,
        atol=5.0e-5,
        rtol=5.0e-5,
    )


def test_historical_sanos_flags_statistically_thin_daily_fit() -> None:
    generator = np.random.default_rng(89)
    instruments = generator.normal(scale=0.01, size=(40, 30))
    claim = -0.2 * instruments[:, 0]
    config = HistoricalSanosConfig(
        rolling_observations=40,
        objective=HistoricalSanosObjective.MEAN_VARIANCE,
        risk_parameter=1.0,
        maximum_iterations=2_000,
    )

    with pytest.warns(RuntimeWarning, match="statistically thin"):
        fit = fit_historical_sanos_hedge(
            instrument_returns=instruments,
            claim_returns=claim,
            linear_transaction_costs=np.zeros(instruments.shape[1]),
            config=config,
        )

    assert fit.converged


def test_historical_sanos_exact_cost_has_lasso_zero_threshold() -> None:
    instruments = np.asarray([[-0.01], [0.01], [0.02], [0.04]], dtype=np.float64)
    config = HistoricalSanosConfig(
        rolling_observations=20,
        objective=HistoricalSanosObjective.ENTROPY,
        risk_parameter=1.0,
        gradient_tolerance=1.0e-10,
    )

    fit = _fit_policy(
        instruments,
        np.zeros(len(instruments)),
        np.asarray([0.02]),
        config=config,
    )

    assert fit.converged
    assert fit.action[0] == pytest.approx(0.0, abs=1.0e-12)
    assert fit.maximum_absolute_gradient == pytest.approx(0.0, abs=1.0e-12)


def test_historical_sanos_exact_cost_accepts_independent_kkt_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruments = np.asarray([[-0.01], [0.01], [0.02], [0.04]], dtype=np.float64)
    config = HistoricalSanosConfig(
        rolling_observations=20,
        objective=HistoricalSanosObjective.ENTROPY,
        risk_parameter=1.0,
        gradient_tolerance=1.0e-10,
    )

    def line_search_failure(*args: object, **kwargs: object) -> OptimizeResult:
        initial = np.asarray(args[1] if len(args) > 1 else kwargs["x0"], dtype=np.float64)
        return OptimizeResult(
            x=np.zeros_like(initial),
            nit=1,
            success=False,
            message="synthetic line-search failure",
        )

    monkeypatch.setattr(hedger, "minimize", line_search_failure)
    fit = _fit_policy(
        instruments,
        np.zeros(len(instruments)),
        np.asarray([0.02]),
        config=config,
    )

    assert fit.converged
    assert fit.action[0] == pytest.approx(0.0, abs=1.0e-12)
    assert fit.maximum_absolute_gradient == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.parametrize(
    "objective",
    [
        HistoricalSanosObjective.ENTROPY,
        HistoricalSanosObjective.POWER_EXPONENTIAL,
    ],
)
def test_historical_sanos_exact_cost_restarts_stalled_split_fit(
    monkeypatch: pytest.MonkeyPatch,
    objective: HistoricalSanosObjective,
) -> None:
    instruments = np.asarray([[-0.01], [0.01], [0.02], [0.04]], dtype=np.float64)
    config = HistoricalSanosConfig(
        rolling_observations=20,
        objective=objective,
        risk_parameter=1.0,
        gradient_tolerance=1.0e-10,
    )
    calls = 0

    def stalled_then_converged(*args: object, **kwargs: object) -> OptimizeResult:
        nonlocal calls
        calls += 1
        initial = np.asarray(args[1] if len(args) > 1 else kwargs["x0"], dtype=np.float64)
        solution = np.zeros_like(initial)
        if calls == 1:
            solution[0] = 1.0
        return OptimizeResult(
            x=solution,
            nit=1,
            success=(calls != 1),
            message=("stalled" if calls == 1 else "converged"),
        )

    monkeypatch.setattr(hedger, "minimize", stalled_then_converged)
    fit = _fit_policy(
        instruments,
        np.zeros(len(instruments)),
        np.asarray([0.02]),
        config=config,
    )

    assert calls == 3
    assert fit.converged
    assert fit.iterations == 3
    assert fit.action[0] == pytest.approx(0.0, abs=1.0e-12)


def test_historical_sanos_selection_failure_does_not_reject_final_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruments = np.asarray([[-0.01], [0.01], [0.02], [0.04]], dtype=np.float64)
    config = HistoricalSanosConfig(
        rolling_observations=20,
        objective=HistoricalSanosObjective.ENTROPY,
        risk_parameter=1.0,
        gradient_tolerance=1.0e-10,
    )
    calls = 0

    def failed_selection_then_final(
        initial: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> OptimizeResult:
        nonlocal calls
        calls += 1
        return OptimizeResult(
            x=np.zeros_like(initial),
            nit=1,
            success=(calls == 3),
            optimality=(0.0 if calls == 3 else 1.0e-6),
            message=("converged" if calls == 3 else "selection stalled"),
        )

    monkeypatch.setattr(hedger, "_minimize_policy", failed_selection_then_final)
    fit = _fit_policy(
        instruments,
        np.zeros(len(instruments)),
        np.asarray([0.02]),
        config=config,
    )

    assert calls == 3
    assert not fit.selection_converged
    assert fit.selection_maximum_absolute_gradient == pytest.approx(1.0e-6)
    assert fit.selection_cap_hits == 0
    assert fit.final_converged
    assert fit.converged
    assert fit.maximum_absolute_gradient == pytest.approx(0.0)


def test_power_oce_cash_translation_is_shift_invariant() -> None:
    wealth = np.asarray([-0.4, -0.1, 0.2, 0.8], dtype=np.float64)
    shift = 0.37
    risk_parameter = 2.0
    translation = _optimal_power_cash_translation(
        wealth,
        risk_parameter=risk_parameter,
    )
    shifted_translation = _optimal_power_cash_translation(
        wealth + shift,
        risk_parameter=risk_parameter,
    )
    config = HistoricalSanosConfig(
        rolling_observations=20,
        objective=HistoricalSanosObjective.POWER_EXPONENTIAL,
        risk_parameter=risk_parameter,
    )
    instruments = np.zeros((len(wealth), 1), dtype=np.float64)
    costs = np.zeros(1, dtype=np.float64)
    loss = _objective_and_gradient(
        np.asarray([0.0, translation]),
        instruments,
        wealth,
        costs,
        config=config,
        include_translation=True,
    )[0]
    shifted_loss = _objective_and_gradient(
        np.asarray([0.0, shifted_translation]),
        instruments,
        wealth + shift,
        costs,
        config=config,
        include_translation=True,
    )[0]

    assert shifted_translation == pytest.approx(translation + shift)
    assert shifted_loss == pytest.approx(loss - shift)


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


def test_mean_variance_retries_osqp_failure_with_clarabel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = np.random.default_rng(113)
    instruments = generator.normal(scale=0.01, size=(80, 3))
    claim = -(0.5 * instruments[:, 0] - 0.2 * instruments[:, 1])
    original_solve = cp.Problem.solve

    def fail_osqp(
        problem: cp.Problem,
        *args: object,
        **kwargs: object,
    ) -> float:
        if kwargs.get("solver") == cp.OSQP:
            raise cp.error.SolverError("forced OSQP failure")
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fail_osqp)
    fit = _fit_policy(
        instruments,
        claim,
        np.zeros(instruments.shape[1]),
        config=HistoricalSanosConfig(
            rolling_observations=80,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=2.0,
        ),
    )

    assert fit.converged
    assert "OSQP failed: forced OSQP failure" in fit.solver_status
    assert "CLARABEL=optimal" in fit.solver_status


def test_mean_variance_preserves_all_solver_failures_after_bad_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = np.random.default_rng(119)
    instruments = generator.normal(scale=0.01, size=(80, 3))
    claim = -0.4 * instruments[:, 0] + generator.normal(scale=0.001, size=80)
    costs = np.full(instruments.shape[1], 1.0e-5)
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=HistoricalSanosObjective.MEAN_VARIANCE,
        risk_parameter=2.0,
    )
    action_scale = hedger._action_scale(instruments, config=config)
    original_solve = cp.Problem.solve

    def reject_clarabel(
        problem: cp.Problem,
        *args: object,
        **kwargs: object,
    ) -> float:
        if kwargs.get("solver") == cp.CLARABEL:
            raise cp.error.SolverError("forced CLARABEL failure")
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", reject_clarabel)
    monkeypatch.setattr(
        hedger,
        "_maximum_kkt_violation",
        lambda *args, **kwargs: float("inf"),
    )
    result = hedger._minimize_mean_variance(
        np.zeros(instruments.shape[1]),
        instruments,
        claim,
        costs,
        config=config,
        action_scale=action_scale,
    )

    assert not result.success
    assert "OSQP=optimal" in result.message
    assert "CLARABEL failed: forced CLARABEL failure" in result.message


def test_mean_variance_coordinate_refinement_certifies_lasso_qp() -> None:
    generator = np.random.default_rng(131)
    common = generator.normal(scale=0.01, size=120)
    instruments = np.column_stack(
        (
            common + generator.normal(scale=1.0e-5, size=120),
            common + generator.normal(scale=1.0e-5, size=120),
            generator.normal(scale=0.01, size=120),
            generator.normal(scale=0.01, size=120),
        )
    )
    claim = -0.4 * instruments[:, 0] + 0.1 * instruments[:, 2]
    costs = np.asarray([1.0e-5, 1.0e-5, 1.0e-5, 5.0e-3])
    config = HistoricalSanosConfig(
        rolling_observations=120,
        objective=HistoricalSanosObjective.MEAN_VARIANCE,
        risk_parameter=0.1,
    )
    action_scale = hedger._action_scale(instruments, config=config)
    initial_action = np.asarray([0.2, 0.2, -0.1, 1.0e-3])
    initial_objective = _objective_and_gradient(
        initial_action,
        instruments,
        claim,
        costs,
        config=config,
        include_translation=False,
        action_scale=action_scale,
    )[0]

    refined_action, sweeps, maximum_violation = (
        _refine_mean_variance_coordinate_descent(
            initial_action,
            instruments,
            claim,
            costs,
            config=config,
            action_scale=action_scale,
        )
    )
    refined_objective = _objective_and_gradient(
        refined_action,
        instruments,
        claim,
        costs,
        config=config,
        include_translation=False,
        action_scale=action_scale,
    )[0]

    assert 0 < sweeps <= config.maximum_iterations
    assert maximum_violation <= config.certificate_tolerance
    assert refined_objective <= initial_objective
    assert refined_action[3] == 0.0


def test_mean_volatility_retries_failed_certificate_with_scs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = np.random.default_rng(127)
    instruments = generator.normal(scale=0.01, size=(80, 3))
    claim = -0.4 * instruments[:, 0] + generator.normal(scale=0.001, size=80)
    costs = np.full(instruments.shape[1], 1.0e-5)
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=HistoricalSanosObjective.MEAN_VOLATILITY,
        risk_parameter=2.0,
    )
    action_scale = hedger._action_scale(instruments, config=config)
    original_kkt = hedger._maximum_kkt_violation
    certificate_calls = 0

    def reject_primary_certificate(*args: object, **kwargs: object) -> float:
        nonlocal certificate_calls
        certificate_calls += 1
        if certificate_calls == 1:
            return float("inf")
        return original_kkt(*args, **kwargs)

    monkeypatch.setattr(
        hedger,
        "_maximum_kkt_violation",
        reject_primary_certificate,
    )
    result = hedger._minimize_mean_volatility(
        np.zeros(instruments.shape[1]),
        instruments,
        claim,
        costs,
        config=config,
        action_scale=action_scale,
    )

    assert certificate_calls >= 2
    assert result.success
    assert "SCS=optimal" in result.message


def test_mean_volatility_exact_zero_norm_is_not_a_false_failure() -> None:
    instruments = np.asarray(
        [
            [-0.01],
            [0.01],
        ]
        * 30,
        dtype=np.float64,
    )
    fit = _fit_policy(
        instruments,
        np.zeros(len(instruments)),
        np.zeros(1),
        config=HistoricalSanosConfig(
            rolling_observations=60,
            objective=HistoricalSanosObjective.MEAN_VOLATILITY,
            risk_parameter=10.0,
        ),
    )

    assert fit.converged
    assert fit.action == pytest.approx(np.zeros(1), abs=1.0e-12)
    assert "objective_error=0" in fit.solver_status


def test_mean_volatility_canonicalizes_scaled_solver_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = np.random.default_rng(103)
    instruments = generator.normal(scale=0.01, size=(80, 2))
    claim = -0.4 * instruments[:, 0] + generator.normal(scale=1.0e-4, size=80)
    original_solve = cp.Problem.solve
    original_kkt = hedger._maximum_kkt_violation
    certified_inactive_coordinates: list[float] = []

    def inject_inactive_coordinate_noise(
        problem: cp.Problem,
        *args: object,
        **kwargs: object,
    ) -> float:
        value = original_solve(problem, *args, **kwargs)
        if kwargs.get("solver") == cp.CLARABEL:
            scaled_action = problem.variables()[0]
            fitted = np.asarray(scaled_action.value, dtype=np.float64).copy()
            fitted[1] = 5.0e-10
            scaled_action.value = fitted
        return value

    def record_certified_point(*args: object, **kwargs: object) -> float:
        action = np.asarray(args[0], dtype=np.float64)
        certified_inactive_coordinates.append(float(action[1]))
        return original_kkt(*args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", inject_inactive_coordinate_noise)
    monkeypatch.setattr(
        hedger,
        "_maximum_kkt_violation",
        record_certified_point,
    )
    fit = _fit_policy(
        instruments,
        claim,
        np.asarray([0.0, 0.05]),
        config=HistoricalSanosConfig(
            rolling_observations=80,
            objective=HistoricalSanosObjective.MEAN_VOLATILITY,
            risk_parameter=2.0,
        ),
    )

    assert fit.converged
    assert certified_inactive_coordinates[0] != 0.0
    assert 0.0 in certified_inactive_coordinates[1:]
    assert fit.action[1] == 0.0
    assert fit.maximum_absolute_gradient <= 1.0e-6
    assert "canonicalization_error=" in fit.solver_status


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
    assert fit.claim_policy.recession_direction
    assert fit.zero_policy.recession_direction
    assert "recession_direction=" in fit.zero_policy.solver_status
    assert not fit.converged


def test_mean_volatility_accepts_certified_inaccurate_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = np.random.default_rng(97)
    instruments = generator.normal(scale=0.01, size=(60, 2))
    claim = -0.3 * instruments[:, 0]
    original_solve = cp.Problem.solve

    def force_inaccurate_status(
        problem: cp.Problem,
        *args: object,
        **kwargs: object,
    ) -> float:
        value = original_solve(problem, *args, **kwargs)
        problem._status = cp.OPTIMAL_INACCURATE
        return value

    monkeypatch.setattr(cp.Problem, "solve", force_inaccurate_status)
    fit = _fit_policy(
        instruments,
        claim,
        np.zeros(instruments.shape[1]),
        config=HistoricalSanosConfig(
            rolling_observations=60,
            objective=HistoricalSanosObjective.MEAN_VOLATILITY,
            risk_parameter=1.0,
        ),
    )

    assert fit.final_converged
    assert fit.converged
    assert "SCS=optimal_inaccurate" in fit.solver_status


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
            quadratic_ridge=1.0e-12,
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
    assert all(
        result.diagnostics["claim_policy_final_converged"]
        for result in results
    )
    assert all(
        "claim_policy_selection_maximum_optimality_residual" in result.diagnostics
        for result in results
    )
    assert all(
        result.diagnostics["observation_parameter_ratio"] == pytest.approx(60.0)
        for result in results
    )
    assert all(not result.diagnostics["statistically_thin"] for result in results)
    assert all(
        result.diagnostics["quadratic_ridge"] == pytest.approx(1.0e-4)
        for result in results
    )
    assert all(
        result.diagnostics["regularization_coordinates"]
        == "unit-rms instrument gains"
        for result in results
    )
    assert all(
        result.diagnostics["claim_policy_solver_status"] for result in results
    )


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
