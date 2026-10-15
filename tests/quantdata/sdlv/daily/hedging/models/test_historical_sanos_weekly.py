from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

from quantdata.biz_calendar import add_business_days
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyFitInstruction,
    WeeklyTarget,
    WeeklyTargetKind,
    build_weekly_claim,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyDecisionState,
    WeeklyPriceRecord,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_scenarios import (
    WeeklyRiskObjective,
    WeeklyScenarioBatch,
    weekly_certainty_equivalent,
    weekly_scenario_accounting,
)
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_weekly_market_day,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_adapter import (
    HistoricalSanosWeeklyPolicy,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_hedger import (
    HistoricalWeeklyHedgeFit,
    WeeklyOpenLoopPricingContext,
    _WeeklyOpenLoopGeometry,
    fit_historical_weekly_hedge,
    fit_weekly_open_loop_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_scenarios import (
    HistoricalWeeklyScenarioSet,
    HistoricalWeeklyTransitionHistory,
    attach_historical_weekly_claim,
    build_historical_weekly_scenarios,
)


def _scenario_set(*, scenario_count: int = 80) -> HistoricalWeeklyScenarioSet:
    generator = np.random.default_rng(42)
    step_count = 3
    contract_count = 2
    spot_gains = generator.normal(scale=0.01, size=(scenario_count, step_count))
    option_gains = generator.normal(
        scale=0.008,
        size=(scenario_count, step_count, contract_count),
    )
    option_alive = np.asarray(
        [
            [True, True],
            [False, True],
            [False, True],
            [False, True],
        ]
    )
    option_settles_next = np.asarray(
        [
            [True, False],
            [False, False],
            [False, False],
        ]
    )
    option_gains[:, 1:, 0] = 0.0
    option_spreads = np.full(
        (scenario_count, step_count + 1, contract_count),
        2.0e-4,
    )
    option_spreads[:, 1:, 0] = 0.0
    target = -(
        0.4 * np.sum(spot_gains, axis=1)
        - 0.25 * np.sum(option_gains[:, :, 1], axis=1)
    )
    fit_date = date(2022, 12, 30)
    batch = WeeklyScenarioBatch(
        inception_forward_paths=np.exp(
            np.column_stack(
                (
                    np.zeros(scenario_count),
                    np.cumsum(spot_gains, axis=1),
                )
            )
        ),
        spot_values_pv=np.ones((scenario_count, step_count + 1)),
        spot_step_gains=spot_gains,
        option_step_gains=option_gains,
        option_half_spreads_pv=option_spreads,
        option_alive=option_alive,
        option_settles_next=option_settles_next,
        terminal_claim_values_pv=target - np.min(target) + 0.01,
        contract_ids=("short", "long"),
        target_label="test-claim",
        fit_as_of_date=fit_date,
        model_fit_identity="historical-weekly-test",
        measure="historical-P",
        seed=42,
    )
    origins = tuple(
        fit_date - timedelta(days=scenario_count - index + 3)
        for index in range(scenario_count)
    )
    terminals = tuple(value + timedelta(days=3) for value in origins)
    return HistoricalWeeklyScenarioSet(
        scenarios=batch,
        training_origin_dates=origins,
        training_terminal_dates=terminals,
        terminal_claim_pv_scales=np.ones(scenario_count),
    )


@pytest.mark.parametrize("objective", tuple(HistoricalSanosObjective))
def test_historical_weekly_gradient_matches_finite_differences(
    objective: HistoricalSanosObjective,
) -> None:
    scenario_set = _scenario_set()
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=objective,
        risk_parameter=1.3,
        cost_smoothing=1.0e-4,
        maximum_iterations=500,
    )
    geometry = _WeeklyOpenLoopGeometry(
        scenario_set,
        initial_spot_position=0.15,
        initial_option_positions=np.asarray([-0.2, 0.1]),
        config=config,
    )
    include_translation = objective is HistoricalSanosObjective.POWER_EXPONENTIAL
    generator = np.random.default_rng(17)
    parameters = generator.normal(scale=0.15, size=geometry.parameter_count)
    if include_translation:
        parameters = np.concatenate((parameters, [0.02]))
    indices = np.arange(60)
    loss, gradient = geometry._loss_and_gradient(
        parameters,
        indices,
        signed_claim_quantity=0.7,
        include_translation=include_translation,
    )
    finite_difference = np.empty_like(parameters)
    step = 1.0e-6
    for index in range(len(parameters)):
        up = parameters.copy()
        down = parameters.copy()
        up[index] += step
        down[index] -= step
        up_loss = geometry._loss_and_gradient(
            up,
            indices,
            signed_claim_quantity=0.7,
            include_translation=include_translation,
        )[0]
        down_loss = geometry._loss_and_gradient(
            down,
            indices,
            signed_claim_quantity=0.7,
            include_translation=include_translation,
        )[0]
        finite_difference[index] = (up_loss - down_loss) / (2.0 * step)
    assert np.isfinite(loss)
    np.testing.assert_allclose(gradient, finite_difference, atol=2.0e-7, rtol=2.0e-5)


@pytest.mark.parametrize(
    "objective",
    tuple(HistoricalSanosObjective),
)
def test_historical_weekly_fit_respects_expiry_and_returns_pure_hedge(
    objective: HistoricalSanosObjective,
) -> None:
    scenario_set = _scenario_set()
    fit = fit_historical_weekly_hedge(
        scenario_set,
        claim_quantity=1.0,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=80,
            objective=objective,
            risk_parameter=3.0,
            cost_smoothing=1.0e-5,
            maximum_iterations=2_000,
            gradient_tolerance=1.0e-7,
        ),
    )
    assert fit.spot_positions.shape == (3,)
    assert fit.option_positions.shape == (3, 2)
    assert fit.option_positions[1:, 0] == pytest.approx(np.zeros(2))
    assert np.all(np.isfinite(fit.spot_positions))
    assert np.all(np.isfinite(fit.option_positions))
    assert fit.prices.buyer <= fit.prices.seller


def test_weekly_accepts_certified_inaccurate_status() -> None:
    scenario_set = _scenario_set(scenario_count=60)
    config = HistoricalSanosConfig(
        rolling_observations=60,
        objective=HistoricalSanosObjective.ENTROPY,
        risk_parameter=2.0,
    )
    geometry = _WeeklyOpenLoopGeometry(
        scenario_set,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
    )
    indices = np.arange(scenario_set.scenarios.scenario_count, dtype=np.int64)
    parameters = np.zeros(geometry.parameter_count)
    action_scale = geometry._action_scale(indices)
    exact_objective = geometry._objective_value(
        parameters,
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=(
            scenario_set.scenarios.terminal_claim_values_pv
        ),
        include_translation=False,
        action_scale=action_scale,
    )

    accepted_inaccurate, _, objective_error = geometry._certify_conic_result(
        parameters,
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=(
            scenario_set.scenarios.terminal_claim_values_pv
        ),
        include_translation=False,
        action_scale=action_scale,
        solver_status=cp.OPTIMAL_INACCURATE,
        solver_objective=exact_objective,
        maximum_constraint_violation=0.0,
    )
    rejected_bad_objective, _, rejected_error = geometry._certify_conic_result(
        parameters,
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=(
            scenario_set.scenarios.terminal_claim_values_pv
        ),
        include_translation=False,
        action_scale=action_scale,
        solver_status=cp.OPTIMAL_INACCURATE,
        solver_objective=exact_objective + 1.0,
        maximum_constraint_violation=0.0,
    )

    assert accepted_inaccurate
    assert objective_error == pytest.approx(0.0)
    assert not rejected_bad_objective
    assert rejected_error == pytest.approx(1.0)


def test_weekly_inaccurate_primary_requires_independent_scs_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario_set = _scenario_set(scenario_count=60)
    config = HistoricalSanosConfig(
        rolling_observations=60,
        objective=HistoricalSanosObjective.ENTROPY,
        risk_parameter=2.0,
    )
    geometry = _WeeklyOpenLoopGeometry(
        scenario_set,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
    )
    original_solve = cp.Problem.solve

    def force_primary_inaccurate(
        problem: cp.Problem,
        *args: object,
        **kwargs: object,
    ) -> float:
        value = original_solve(problem, *args, **kwargs)
        if kwargs.get("solver") == cp.CLARABEL:
            problem._status = cp.OPTIMAL_INACCURATE
        return value

    monkeypatch.setattr(cp.Problem, "solve", force_primary_inaccurate)
    result = geometry._minimize_entropy_cost(
        np.zeros(geometry.parameter_count),
        np.arange(scenario_set.scenarios.scenario_count, dtype=np.int64),
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=(
            scenario_set.scenarios.terminal_claim_values_pv
        ),
    )

    assert result.success
    assert (
        "CLARABEL=optimal_inaccurate; CLARABEL=optimal_inaccurate; "
        "SCS=optimal"
    ) in result.message
    assert result.constraint_violation <= config.certificate_tolerance


def test_weekly_conic_trace_keeps_inaccurate_retry_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario_set = _scenario_set(scenario_count=60)
    geometry = _WeeklyOpenLoopGeometry(
        scenario_set,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=60,
            objective=HistoricalSanosObjective.ENTROPY,
            risk_parameter=2.0,
        ),
    )
    original_solve = cp.Problem.solve
    clarabel_calls = 0

    def force_first_primary_inaccurate(
        problem: cp.Problem,
        *args: object,
        **kwargs: object,
    ) -> float:
        nonlocal clarabel_calls
        value = original_solve(problem, *args, **kwargs)
        if kwargs.get("solver") == cp.CLARABEL:
            clarabel_calls += 1
            if clarabel_calls == 1:
                problem._status = cp.OPTIMAL_INACCURATE
        return value

    monkeypatch.setattr(cp.Problem, "solve", force_first_primary_inaccurate)
    result = geometry._minimize_entropy_cost(
        np.zeros(geometry.parameter_count),
        np.arange(scenario_set.scenarios.scenario_count, dtype=np.int64),
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=(
            scenario_set.scenarios.terminal_claim_values_pv
        ),
    )

    assert clarabel_calls == 2
    assert result.success
    assert (
        "CLARABEL=optimal_inaccurate; CLARABEL=optimal"
        in result.message
    )


@pytest.mark.parametrize(
    "objective",
    [
        HistoricalSanosObjective.ENTROPY,
        HistoricalSanosObjective.POWER_EXPONENTIAL,
    ],
)
def test_weekly_optimal_primary_with_failed_certificate_retries_scs(
    monkeypatch: pytest.MonkeyPatch,
    objective: HistoricalSanosObjective,
) -> None:
    scenario_set = _scenario_set(scenario_count=60)
    config = HistoricalSanosConfig(
        rolling_observations=60,
        objective=objective,
        risk_parameter=2.0,
    )
    geometry = _WeeklyOpenLoopGeometry(
        scenario_set,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
    )
    original_certificate = geometry._certify_conic_result
    certificate_calls = 0

    def reject_primary_certificate(*args: object, **kwargs: object):
        nonlocal certificate_calls
        certificate_calls += 1
        accepted, exact_objective, objective_error = original_certificate(
            *args,
            **kwargs,
        )
        if certificate_calls == 1:
            return False, exact_objective, objective_error
        return accepted, exact_objective, objective_error

    def reject_canonicalization(*_: object, **__: object) -> np.ndarray:
        raise AssertionError("raw conic solution must be certified unchanged")

    monkeypatch.setattr(
        geometry,
        "_certify_conic_result",
        reject_primary_certificate,
    )
    monkeypatch.setattr(
        geometry,
        "_canonicalize_positions",
        reject_canonicalization,
    )
    minimize_cost = (
        geometry._minimize_entropy_cost
        if objective is HistoricalSanosObjective.ENTROPY
        else geometry._minimize_power_cost
    )
    result = minimize_cost(
        np.zeros(
            geometry.parameter_count
            + int(objective is HistoricalSanosObjective.POWER_EXPONENTIAL)
        ),
        np.arange(scenario_set.scenarios.scenario_count, dtype=np.int64),
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=(
            scenario_set.scenarios.terminal_claim_values_pv
        ),
    )

    assert certificate_calls == 2
    assert result.success
    assert "CLARABEL=optimal; SCS=optimal" in result.message


@pytest.mark.parametrize(
    "objective",
    [
        HistoricalSanosObjective.ENTROPY,
        HistoricalSanosObjective.POWER_EXPONENTIAL,
    ],
)
def test_weekly_clarabel_exception_retries_same_problem_with_scs(
    monkeypatch: pytest.MonkeyPatch,
    objective: HistoricalSanosObjective,
) -> None:
    scenario_set = _scenario_set(scenario_count=60)
    config = HistoricalSanosConfig(
        rolling_observations=60,
        objective=objective,
        risk_parameter=2.0,
    )
    geometry = _WeeklyOpenLoopGeometry(
        scenario_set,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
    )
    original_solve = cp.Problem.solve

    def fail_clarabel(
        problem: cp.Problem,
        *args: object,
        **kwargs: object,
    ) -> float:
        if kwargs.get("solver") == cp.CLARABEL:
            raise cp.error.SolverError("forced CLARABEL failure")
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fail_clarabel)
    minimize_cost = (
        geometry._minimize_entropy_cost
        if objective is HistoricalSanosObjective.ENTROPY
        else geometry._minimize_power_cost
    )
    result = minimize_cost(
        np.zeros(
            geometry.parameter_count
            + int(objective is HistoricalSanosObjective.POWER_EXPONENTIAL)
        ),
        np.arange(scenario_set.scenarios.scenario_count, dtype=np.int64),
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=(
            scenario_set.scenarios.terminal_claim_values_pv
        ),
    )

    assert result.success
    assert "CLARABEL failed: forced CLARABEL failure; SCS=optimal" in result.message


@pytest.mark.parametrize(
    "objective",
    [
        HistoricalSanosObjective.ENTROPY,
        HistoricalSanosObjective.POWER_EXPONENTIAL,
    ],
)
def test_weekly_failed_scs_fallback_is_not_repeated(
    monkeypatch: pytest.MonkeyPatch,
    objective: HistoricalSanosObjective,
) -> None:
    scenario_set = _scenario_set(scenario_count=60)
    geometry = _WeeklyOpenLoopGeometry(
        scenario_set,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=60,
            objective=objective,
            risk_parameter=2.0,
        ),
    )
    scs_calls = 0

    def fail_conic_solvers(
        problem: cp.Problem,
        *args: object,
        **kwargs: object,
    ) -> float:
        nonlocal scs_calls
        if kwargs.get("solver") == cp.SCS:
            scs_calls += 1
            raise cp.error.SolverError("forced SCS failure")
        if kwargs.get("solver") == cp.CLARABEL:
            raise cp.error.SolverError("forced CLARABEL failure")
        raise AssertionError(f"unexpected solver {kwargs.get('solver')}")

    monkeypatch.setattr(cp.Problem, "solve", fail_conic_solvers)
    minimize_cost = (
        geometry._minimize_entropy_cost
        if objective is HistoricalSanosObjective.ENTROPY
        else geometry._minimize_power_cost
    )
    result = minimize_cost(
        np.zeros(
            geometry.parameter_count
            + int(objective is HistoricalSanosObjective.POWER_EXPONENTIAL)
        ),
        np.arange(scenario_set.scenarios.scenario_count, dtype=np.int64),
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=(
            scenario_set.scenarios.terminal_claim_values_pv
        ),
    )

    assert scs_calls == 1
    assert not result.success
    assert "SCS failed: forced SCS failure" in result.message


def test_weekly_standardized_ridge_is_scale_invariant() -> None:
    scenario_set = _scenario_set(scenario_count=80)
    spot_scale = 3.0
    option_scale = np.asarray([0.4, 2.0])
    rescaled_scenarios = replace(
        scenario_set.scenarios,
        spot_values_pv=scenario_set.scenarios.spot_values_pv * spot_scale,
        spot_step_gains=scenario_set.scenarios.spot_step_gains * spot_scale,
        option_step_gains=(
            scenario_set.scenarios.option_step_gains
            * option_scale[None, None, :]
        ),
        option_half_spreads_pv=(
            scenario_set.scenarios.option_half_spreads_pv
            * option_scale[None, None, :]
        ),
    )
    rescaled_set = replace(scenario_set, scenarios=rescaled_scenarios)
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=HistoricalSanosObjective.ENTROPY,
        risk_parameter=2.0,
        gradient_tolerance=1.0e-10,
    )
    base_geometry = _WeeklyOpenLoopGeometry(
        scenario_set,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
    )
    rescaled_geometry = _WeeklyOpenLoopGeometry(
        rescaled_set,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
    )

    base = base_geometry.fit_policy(signed_claim_quantity=1.0)
    rescaled = rescaled_geometry.fit_policy(signed_claim_quantity=1.0)

    assert base.converged
    assert rescaled.converged
    np.testing.assert_allclose(
        rescaled.spot_positions * spot_scale,
        base.spot_positions,
        atol=2.0e-4,
        rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        rescaled.option_positions * option_scale[None, :],
        base.option_positions,
        atol=2.0e-4,
        rtol=1.0e-3,
    )


def test_weekly_selection_failure_does_not_reject_certified_final_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario_set = _scenario_set(scenario_count=60)
    config = HistoricalSanosConfig(
        rolling_observations=60,
        objective=HistoricalSanosObjective.ENTROPY,
        risk_parameter=2.0,
    )
    geometry = _WeeklyOpenLoopGeometry(
        scenario_set,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
    )
    calls = 0

    def failed_selection_then_final(
        initial: np.ndarray,
        indices: np.ndarray,
        **_: object,
    ) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        result = SimpleNamespace(
            x=np.zeros_like(initial),
            nit=1,
            success=calls == 2,
            optimality=0.0 if calls == 2 else 1.0,
            message="optimal" if calls == 2 else "selection failed",
            objective_error=0.0 if calls == 2 else 1.0,
            action_scale=geometry._action_scale(
                np.asarray(indices, dtype=np.int64)
            ),
        )
        return result

    monkeypatch.setattr(geometry, "_minimize", failed_selection_then_final)
    fit = geometry.fit_policy(signed_claim_quantity=1.0)

    assert calls == 2
    assert fit.selection_solver_status == "selection failed"
    assert fit.solver_status == "optimal"
    assert fit.converged
    assert fit.cap_hits == 0


@pytest.mark.parametrize(
    "objective",
    (
        HistoricalSanosObjective.ENTROPY,
        HistoricalSanosObjective.POWER_EXPONENTIAL,
    ),
)
def test_weekly_oce_cones_use_exact_scenario_transaction_costs(
    objective: HistoricalSanosObjective,
) -> None:
    scenario_set = _scenario_set(scenario_count=60)
    config = HistoricalSanosConfig(
        rolling_observations=60,
        objective=objective,
        risk_parameter=2.0,
        quadratic_ridge=1.0e-6,
        maximum_iterations=2_000,
        gradient_tolerance=1.0e-7,
    )
    fit = fit_historical_weekly_hedge(
        scenario_set,
        claim_quantity=1.0,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
    )

    assert fit.converged
    for policy, signed_claim_quantity in (
        (fit.long_policy, 1.0),
        (fit.short_policy, -1.0),
        (fit.zero_policy, 0.0),
    ):
        accounting = weekly_scenario_accounting(
            scenario_set.scenarios,
            spot_positions=policy.spot_positions,
            option_positions=policy.option_positions,
            signed_claim_quantity=signed_claim_quantity,
        )
        expected = weekly_certainty_equivalent(
            accounting.terminal_wealth,
            objective=WeeklyRiskObjective(objective.value),
            risk_parameter=config.risk_parameter,
        )
        assert policy.certainty_equivalent == pytest.approx(expected, abs=1.0e-10)


def test_weekly_cached_geometry_matches_pathwise_scenario_accounting() -> None:
    scenario_set = _scenario_set(scenario_count=60)
    initial_spot_position = 0.15
    initial_option_positions = np.asarray([-0.2, 0.1])
    geometry = _WeeklyOpenLoopGeometry(
        scenario_set,
        initial_spot_position=initial_spot_position,
        initial_option_positions=initial_option_positions,
        config=HistoricalSanosConfig(
            rolling_observations=60,
            objective=HistoricalSanosObjective.ENTROPY,
            risk_parameter=2.0,
        ),
    )
    generator = np.random.default_rng(20261014)
    parameters = generator.normal(scale=0.2, size=geometry.parameter_count)
    spot_positions, option_positions = geometry.unpack(parameters)
    indices = np.arange(scenario_set.scenarios.scenario_count, dtype=np.int64)

    cached_wealth = geometry._exact_scenario_wealth(
        parameters,
        indices=indices,
        signed_claim_quantity=0.7,
        terminal_claim_values_pv=scenario_set.scenarios.terminal_claim_values_pv,
    )
    accounting = weekly_scenario_accounting(
        scenario_set.scenarios,
        spot_positions=spot_positions,
        option_positions=option_positions,
        signed_claim_quantity=0.7,
        initial_spot_position=initial_spot_position,
        initial_option_positions=initial_option_positions,
    )

    np.testing.assert_allclose(
        cached_wealth,
        accounting.terminal_wealth,
        atol=1.0e-14,
        rtol=1.0e-14,
    )


def test_weekly_power_cone_matches_direct_objective_and_cash_shift() -> None:
    scenario_set = _scenario_set(scenario_count=60)
    config = HistoricalSanosConfig(
        rolling_observations=60,
        objective=HistoricalSanosObjective.POWER_EXPONENTIAL,
        risk_parameter=2.0,
        quadratic_ridge=1.0e-6,
        maximum_iterations=2_000,
        gradient_tolerance=1.0e-7,
    )
    geometry = _WeeklyOpenLoopGeometry(
        scenario_set,
        initial_spot_position=0.15,
        initial_option_positions=np.asarray([-0.2, 0.1]),
        config=config,
    )
    indices = np.arange(45, dtype=np.int64)
    claim_values = scenario_set.scenarios.terminal_claim_values_pv
    initial = np.zeros(geometry.parameter_count + 1, dtype=np.float64)
    fitted = geometry._minimize_power_cost(
        initial,
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=claim_values,
    )

    assert fitted.success
    direct = geometry._objective_value(
        fitted.x,
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=claim_values,
        include_translation=True,
    )
    assert fitted.fun == pytest.approx(direct, abs=2.0e-6)

    cash_shift = 0.03
    shifted_parameters = fitted.x.copy()
    shifted_parameters[-1] += cash_shift
    shifted = geometry._objective_value(
        shifted_parameters,
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=claim_values + cash_shift,
        include_translation=True,
    )
    assert shifted == pytest.approx(direct - cash_shift, abs=1.0e-12)


@pytest.mark.parametrize(
    ("initial_spot_position", "initial_option_positions"),
    (
        (0.0, np.zeros(2)),
        (0.1, np.asarray([0.0, -0.1])),
    ),
)
def test_weekly_shared_pricing_context_matches_fresh_target_fit(
    initial_spot_position: float,
    initial_option_positions: np.ndarray,
) -> None:
    scenarios = _scenario_set().scenarios
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=HistoricalSanosObjective.MEAN_VARIANCE,
        risk_parameter=3.0,
        maximum_iterations=2_000,
        gradient_tolerance=1.0e-7,
    )
    context = WeeklyOpenLoopPricingContext(scenarios, config=config)
    first = fit_weekly_open_loop_hedge(
        scenarios,
        claim_quantity=1.0,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
        pricing_context=context,
    )
    second_scenarios = replace(
        scenarios,
        terminal_claim_values_pv=(
            0.02 + 0.7 * scenarios.terminal_claim_values_pv
        ),
        target_label="second-test-claim",
    )
    shared = fit_weekly_open_loop_hedge(
        second_scenarios,
        claim_quantity=0.8,
        initial_spot_position=initial_spot_position,
        initial_option_positions=initial_option_positions,
        config=config,
        pricing_context=context,
    )
    fresh = fit_weekly_open_loop_hedge(
        second_scenarios,
        claim_quantity=0.8,
        initial_spot_position=initial_spot_position,
        initial_option_positions=initial_option_positions,
        config=config,
    )

    assert first.zero_policy is shared.zero_policy
    np.testing.assert_allclose(shared.spot_positions, fresh.spot_positions, atol=1.0e-8)
    np.testing.assert_allclose(
        shared.option_positions,
        fresh.option_positions,
        atol=1.0e-8,
    )
    assert shared.prices.buyer == pytest.approx(fresh.prices.buyer, abs=1.0e-9)
    assert shared.prices.seller == pytest.approx(fresh.prices.seller, abs=1.0e-9)


def test_historical_weekly_target_attaches_to_exact_cached_geometry() -> None:
    reference = _scenario_set()
    state = SimpleNamespace(
        decision_index=0,
        inception_forward_path=np.asarray([1.0]),
    )
    claim = SimpleNamespace(
        claim=SimpleNamespace(
            payoff=lambda path: float(0.02 + 0.7 * path[-1]),
        ),
        target=SimpleNamespace(label="second-test-claim"),
    )

    shared = attach_historical_weekly_claim(reference, state=state, claim=claim)

    assert shared.scenarios.target_label == "second-test-claim"
    np.testing.assert_allclose(
        shared.scenarios.terminal_claim_values_pv,
        0.02 + 0.7 * reference.scenarios.inception_forward_paths[:, -1],
    )
    for name in (
        "inception_forward_paths",
        "spot_values_pv",
        "spot_step_gains",
        "option_step_gains",
        "option_half_spreads_pv",
        "option_alive",
        "option_settles_next",
    ):
        assert getattr(shared.scenarios, name) is getattr(reference.scenarios, name)


def test_historical_weekly_target_rejects_changed_decision_state() -> None:
    reference = _scenario_set()
    state = SimpleNamespace(
        decision_index=0,
        inception_forward_path=np.asarray([1.01]),
    )
    claim = SimpleNamespace(
        claim=SimpleNamespace(payoff=lambda path: float(path[-1])),
        target=SimpleNamespace(label="changed-state"),
    )

    with pytest.raises(ValueError, match="changed decision state"):
        attach_historical_weekly_claim(reference, state=state, claim=claim)


def test_weekly_mean_variance_osqp_matches_clarabel_reference() -> None:
    scenarios = _scenario_set().scenarios
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=HistoricalSanosObjective.MEAN_VARIANCE,
        risk_parameter=3.0,
        maximum_iterations=2_000,
        gradient_tolerance=1.0e-8,
    )
    geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
    )
    fit = geometry.fit_policy(signed_claim_quantity=1.0)

    reference = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
    )
    indices = np.arange(scenarios.scenario_count, dtype=np.int64)
    claim, instrument_gains = reference._linear_wealth_geometry(
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
    )
    action_scale = reference._action_scale(indices)
    scaled_positions = cp.Variable(reference.parameter_count)
    positions = cp.multiply(action_scale, scaled_positions)
    wealth_before_cost = claim + instrument_gains @ positions
    centered = wealth_before_cost - cp.sum(wealth_before_cost) / len(indices)
    turnover_matrix, turnover_offset, turnover_weights = (
        reference._turnover_geometry(indices)
    )
    expanded_problem = cp.Problem(
        cp.Minimize(
            -cp.sum(wealth_before_cost) / len(indices)
            + 0.5
            * config.risk_parameter
            * cp.sum_squares(centered)
            / len(indices)
            + turnover_weights
            @ cp.abs(turnover_matrix @ positions - turnover_offset)
            + 0.5 * config.quadratic_ridge * cp.sum_squares(scaled_positions)
        ),
        [
            positions >= -config.solver_position_cap,
            positions <= config.solver_position_cap,
        ],
    )
    expanded_problem.solve(
        solver=cp.CLARABEL,
        warm_start=False,
        max_iter=config.maximum_iterations,
        tol_gap_abs=config.gradient_tolerance,
        tol_gap_rel=config.gradient_tolerance,
        tol_feas=config.gradient_tolerance,
    )

    assert expanded_problem.status == cp.OPTIMAL
    assert fit.converged
    assert fit.training_loss == pytest.approx(expanded_problem.value, abs=1.0e-8)


def test_weekly_mean_variance_reuses_compact_target_independent_problem() -> None:
    first = _scenario_set().scenarios
    second = replace(
        first,
        terminal_claim_values_pv=0.01 + 0.4 * first.terminal_claim_values_pv,
        target_label="second-compact-claim",
    )
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=HistoricalSanosObjective.MEAN_VARIANCE,
        risk_parameter=3.0,
        maximum_iterations=2_000,
    )
    context = WeeklyOpenLoopPricingContext(first, config=config)

    context.fit_long(first, claim_quantity=1.0)
    first_problems = {
        key: id(problem.problem)
        for key, problem in context._long_geometry._convex_problems.items()
    }
    context.fit_long(second, claim_quantity=1.0)
    second_problems = {
        key: id(problem.problem)
        for key, problem in context._long_geometry._convex_problems.items()
    }

    assert len(first_problems) == 2
    assert second_problems == first_problems
    assert all(
        problem.problem.is_dpp()
        for problem in context._long_geometry._convex_problems.values()
    )


def test_weekly_mean_variance_certifies_raw_solver_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _scenario_set().scenarios
    geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=80,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=3.0,
            maximum_iterations=2_000,
        ),
    )

    def reject_canonicalization(*_: object, **__: object) -> np.ndarray:
        raise AssertionError("raw convex solution must not be modified")

    monkeypatch.setattr(
        geometry,
        "_canonicalize_positions",
        reject_canonicalization,
    )
    fit = geometry.fit_policy(signed_claim_quantity=1.0)

    assert fit.converged


def test_weekly_mean_variance_polishes_failed_raw_certificate_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _scenario_set().scenarios
    geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=80,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=3.0,
            maximum_iterations=2_000,
        ),
    )
    original_certificate = geometry._certify_compact_mean_variance_solution
    certificate_calls = 0

    def reject_primary_certificate(*args: object, **kwargs: object):
        nonlocal certificate_calls
        certificate_calls += 1
        result = original_certificate(*args, **kwargs)
        if certificate_calls == 1:
            return (False, *result[1:])
        return result

    monkeypatch.setattr(
        geometry,
        "_certify_compact_mean_variance_solution",
        reject_primary_certificate,
    )
    result = geometry._minimize_convex_cost_lasso(
        np.zeros(geometry.parameter_count),
        np.arange(scenarios.scenario_count, dtype=np.int64),
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
    )

    assert certificate_calls == 2
    assert result.success
    assert "active-set KKT polish" in result.message
    assert "fallback HIGHS" not in result.message


def test_weekly_mean_variance_retries_osqp_exception_with_highs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _scenario_set().scenarios
    geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=80,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=3.0,
            maximum_iterations=2_000,
        ),
    )
    original_solve = cp.Problem.solve

    def fail_osqp(
        problem: cp.Problem,
        *args: object,
        solver: str | None = None,
        **kwargs: object,
    ) -> float:
        if solver == cp.OSQP:
            raise cp.error.SolverError("forced OSQP failure")
        return original_solve(problem, *args, solver=solver, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fail_osqp)
    result = geometry._minimize_convex_cost_lasso(
        np.zeros(geometry.parameter_count),
        np.arange(scenarios.scenario_count, dtype=np.int64),
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
    )

    assert result.success
    assert "OSQP failed: forced OSQP failure" in result.message
    assert "fallback HIGHS=optimal" in result.message


def test_weekly_mean_variance_polishes_genuine_osqp_user_limit() -> None:
    scenarios = _scenario_set().scenarios
    geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=80,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=3.0,
        ),
    )
    indices = np.arange(scenarios.scenario_count, dtype=np.int64)
    claim, instrument_gains = geometry._linear_wealth_geometry(
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
    )
    compiled = geometry._convex_cost_lasso_problem(
        indices,
        instrument_gains=instrument_gains,
    )
    geometry._assign_convex_claim(
        compiled,
        claim=claim,
        instrument_gains=instrument_gains,
    )
    compiled.scaled_positions.value = np.zeros(geometry.parameter_count)
    compiled.problem.solve(
        solver=cp.OSQP,
        warm_start=False,
        max_iter=50,
        eps_abs=geometry.config.gradient_tolerance,
        eps_rel=geometry.config.gradient_tolerance,
        polishing=False,
    )

    assert compiled.problem.status == cp.USER_LIMIT
    assert geometry._polish_compact_mean_variance_solution(compiled)
    scaled_fitted = np.asarray(compiled.scaled_positions.value)
    certified = geometry._certify_compact_mean_variance_solution(
        compiled,
        scaled_fitted=scaled_fitted,
        indices=indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
        solver_status=compiled.problem.status,
    )

    assert certified[0]
    assert certified[4] <= geometry.config.certificate_tolerance


def test_weekly_mean_variance_certifies_extreme_scaling_with_zero_cost_rows() -> None:
    base = _scenario_set(scenario_count=20).scenarios
    scenarios = replace(
        base,
        spot_step_gains=np.zeros_like(base.spot_step_gains),
        option_step_gains=1.0e-8 * base.option_step_gains,
        option_half_spreads_pv=np.zeros_like(base.option_half_spreads_pv),
        target_label="extreme-scale-zero-cost-turnover",
    )
    cap = 1_000_000.0
    geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=20,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=3.0,
            position_cap=cap,
            solver_position_cap=cap,
            maximum_iterations=2_000,
        ),
    )
    indices = np.arange(scenarios.scenario_count, dtype=np.int64)
    claim, instrument_gains = geometry._linear_wealth_geometry(
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
    )
    compiled = geometry._convex_cost_lasso_problem(
        indices,
        instrument_gains=instrument_gains,
    )
    certificate = compiled.compact_certificate
    assert certificate is not None
    assert np.any(certificate.turnover_weights == 0.0)
    np.testing.assert_allclose(
        certificate.scaled_position_cap,
        cap / compiled.action_scale,
    )

    result = geometry._minimize_convex_cost_lasso(
        np.zeros(geometry.parameter_count),
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
    )

    assert result.success
    assert result.constraint_violation <= geometry.config.certificate_tolerance
    assert result.optimality <= geometry.config.certificate_tolerance


def test_weekly_mean_variance_compact_objective_includes_all_turnover_costs() -> None:
    scenarios = _scenario_set().scenarios
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=HistoricalSanosObjective.MEAN_VARIANCE,
        risk_parameter=3.0,
    )
    geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.1,
        initial_option_positions=np.asarray([0.0, -0.1]),
        config=config,
    )
    indices = np.arange(scenarios.scenario_count, dtype=np.int64)
    claim, instrument_gains = geometry._linear_wealth_geometry(
        indices,
        signed_claim_quantity=0.7,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
    )
    compiled = geometry._convex_cost_lasso_problem(
        indices,
        instrument_gains=instrument_gains,
    )
    geometry._assign_convex_claim(
        compiled,
        claim=claim,
        instrument_gains=instrument_gains,
    )
    certificate = compiled.compact_certificate
    assert certificate is not None
    assert compiled.linear is not None
    assert compiled.objective_constant is not None
    scaled_positions = np.linspace(-0.002, 0.003, geometry.parameter_count)
    positions = compiled.action_scale * scaled_positions
    turnover = (
        certificate.turnover_matrix @ scaled_positions
        - certificate.turnover_offset
    )
    compact_objective = float(
        0.5
        * scaled_positions
        @ certificate.hessian
        @ scaled_positions
        + np.asarray(compiled.linear.value) @ scaled_positions
        + float(compiled.objective_constant.value)
        + certificate.turnover_weights @ np.abs(turnover)
    )
    exact_objective = geometry._objective_value(
        positions,
        indices,
        signed_claim_quantity=0.7,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
        include_translation=False,
        action_scale=compiled.action_scale,
    )

    assert compact_objective == pytest.approx(exact_objective, abs=1.0e-12)
    assert certificate.turnover_offset[0] == pytest.approx(0.1)
    assert certificate.turnover_offset[1] == pytest.approx(0.0)
    assert certificate.turnover_offset[2] == pytest.approx(-0.1)
    assert np.count_nonzero(certificate.turnover_matrix[-2:]) > 0


def test_weekly_mean_variance_compact_size_is_scenario_independent() -> None:
    scenarios = _scenario_set().scenarios
    geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=80,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=3.0,
        ),
    )
    sizes: list[tuple[int, int, int]] = []
    for scenario_count in (40, scenarios.scenario_count):
        indices = np.arange(scenario_count, dtype=np.int64)
        _, instrument_gains = geometry._linear_wealth_geometry(
            indices,
            signed_claim_quantity=1.0,
            terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
        )
        compiled = geometry._convex_cost_lasso_problem(
            indices,
            instrument_gains=instrument_gains,
        )
        certificate = compiled.compact_certificate
        assert certificate is not None
        metrics = compiled.problem.size_metrics
        sizes.append(
            (
                metrics.num_scalar_variables,
                metrics.num_scalar_eq_constr,
                metrics.num_scalar_leq_constr,
            )
        )
        assert metrics.num_scalar_variables == (
            geometry.parameter_count + len(certificate.turnover_weights)
        )
        assert metrics.num_scalar_eq_constr == 0

    assert sizes[0] == sizes[1]


def test_weekly_mean_variance_rejects_feasible_non_kkt_point() -> None:
    scenarios = _scenario_set().scenarios
    geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=80,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=3.0,
        ),
    )
    indices = np.arange(scenarios.scenario_count, dtype=np.int64)
    claim, instrument_gains = geometry._linear_wealth_geometry(
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
    )
    compiled = geometry._convex_cost_lasso_problem(
        indices,
        instrument_gains=instrument_gains,
    )
    geometry._assign_convex_claim(
        compiled,
        claim=claim,
        instrument_gains=instrument_gains,
    )
    compiled.problem.solve(solver=cp.OSQP, polishing=True)
    certificate = compiled.compact_certificate
    assert certificate is not None
    scaled_positions = np.asarray(compiled.scaled_positions.value).copy()
    accepted = geometry._certify_compact_mean_variance_solution(
        compiled,
        scaled_fitted=scaled_positions,
        indices=indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
        solver_status=cp.OPTIMAL_INACCURATE,
    )[0]
    assert accepted

    scaled_positions[0] += 1.0e-3
    turnover = (
        certificate.turnover_matrix @ scaled_positions
        - certificate.turnover_offset
    )
    compiled.scaled_positions.value = scaled_positions
    certificate.turnover_epigraph.value = np.abs(turnover)
    accepted = geometry._certify_compact_mean_variance_solution(
        compiled,
        scaled_fitted=scaled_positions,
        indices=indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
        solver_status=cp.OPTIMAL_INACCURATE,
    )[0]

    assert not accepted


def test_weekly_mean_variance_certifies_boundary_optimum_as_cap_hit() -> None:
    scenarios = _scenario_set().scenarios
    cap = 0.01
    geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=80,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=3.0,
            position_cap=cap,
            solver_position_cap=cap,
        ),
    )
    result = geometry._minimize_convex_cost_lasso(
        np.zeros(geometry.parameter_count),
        np.arange(scenarios.scenario_count, dtype=np.int64),
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
    )

    assert result.success
    assert np.any(np.abs(result.x) >= 0.999 * cap)
    fit = geometry.fit_policy(signed_claim_quantity=1.0)
    assert fit.cap_hits > 0
    assert not fit.converged


def test_weekly_mean_variance_certifies_rank_deficient_zero_ridge() -> None:
    base = _scenario_set().scenarios
    scenarios = replace(
        base,
        spot_step_gains=np.zeros_like(base.spot_step_gains),
        option_step_gains=np.zeros_like(base.option_step_gains),
        target_label="rank-deficient-zero-ridge",
    )
    geometry = _WeeklyOpenLoopGeometry(
        scenarios,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=HistoricalSanosConfig(
            rolling_observations=80,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=3.0,
            quadratic_ridge=0.0,
        ),
    )
    indices = np.arange(scenarios.scenario_count, dtype=np.int64)
    _, instrument_gains = geometry._linear_wealth_geometry(
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
    )
    compiled = geometry._convex_cost_lasso_problem(
        indices,
        instrument_gains=instrument_gains,
    )
    certificate = compiled.compact_certificate
    assert certificate is not None
    assert np.linalg.matrix_rank(certificate.hessian) < geometry.parameter_count
    result = geometry._minimize_convex_cost_lasso(
        np.zeros(geometry.parameter_count),
        indices,
        signed_claim_quantity=1.0,
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
    )

    assert result.success
    assert result.optimality <= geometry.config.certificate_tolerance


def test_weekly_shared_pricing_context_is_target_order_independent() -> None:
    first_scenarios = _scenario_set().scenarios
    second_scenarios = replace(
        first_scenarios,
        terminal_claim_values_pv=(
            0.015 + 0.6 * first_scenarios.terminal_claim_values_pv
        ),
        target_label="second-order-test-claim",
    )
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=HistoricalSanosObjective.MEAN_VARIANCE,
        risk_parameter=3.0,
        maximum_iterations=2_000,
        gradient_tolerance=1.0e-7,
    )

    def fit_in_order(
        ordered: tuple[WeeklyScenarioBatch, WeeklyScenarioBatch],
    ) -> dict[str, HistoricalWeeklyHedgeFit]:
        context = WeeklyOpenLoopPricingContext(first_scenarios, config=config)
        return {
            scenarios.target_label: fit_weekly_open_loop_hedge(
                scenarios,
                claim_quantity=1.0,
                initial_spot_position=0.0,
                initial_option_positions=np.zeros(2),
                config=config,
                pricing_context=context,
            )
            for scenarios in ordered
        }

    forward = fit_in_order((first_scenarios, second_scenarios))
    reverse = fit_in_order((second_scenarios, first_scenarios))
    for target_label in forward:
        np.testing.assert_allclose(
            forward[target_label].long_policy.spot_positions,
            reverse[target_label].long_policy.spot_positions,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            forward[target_label].short_policy.option_positions,
            reverse[target_label].short_policy.option_positions,
            atol=1.0e-10,
        )
        assert forward[target_label].prices.buyer == pytest.approx(
            reverse[target_label].prices.buyer,
            abs=1.0e-11,
        )
        assert forward[target_label].prices.seller == pytest.approx(
            reverse[target_label].prices.seller,
            abs=1.0e-11,
        )


def test_weekly_shared_pricing_context_reuses_only_matching_execution_state() -> None:
    scenarios = _scenario_set().scenarios
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=HistoricalSanosObjective.MEAN_VARIANCE,
        risk_parameter=3.0,
    )
    context = WeeklyOpenLoopPricingContext(scenarios, config=config)
    first = context.execution_geometry(
        scenarios,
        initial_spot_position=0.1,
        initial_option_positions=np.asarray([0.0, -0.1]),
    )
    matching = context.execution_geometry(
        scenarios,
        initial_spot_position=0.1,
        initial_option_positions=np.asarray([0.0, -0.1]),
    )
    changed = context.execution_geometry(
        scenarios,
        initial_spot_position=0.2,
        initial_option_positions=np.asarray([0.0, -0.1]),
    )

    assert matching is first
    assert changed is not first


def test_weekly_shared_pricing_context_rejects_concurrent_use() -> None:
    scenarios = _scenario_set().scenarios
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=HistoricalSanosObjective.MEAN_VARIANCE,
        risk_parameter=3.0,
    )
    context = WeeklyOpenLoopPricingContext(scenarios, config=config)

    with context.exclusive_use():
        with pytest.raises(RuntimeError, match="already in use"):
            fit_weekly_open_loop_hedge(
                scenarios,
                claim_quantity=1.0,
                initial_spot_position=0.0,
                initial_option_positions=np.zeros(2),
                config=config,
                pricing_context=context,
            )


def test_historical_weekly_records_virtual_recession_without_executable_cap() -> None:
    generator = np.random.default_rng(109)
    scenario_count = 80
    fit_date = date(2022, 12, 30)
    spot_gains = (0.1 + 0.001 * generator.normal(size=scenario_count))[:, None]
    scenarios = WeeklyScenarioBatch(
        inception_forward_paths=np.column_stack(
            (np.ones(scenario_count), np.exp(spot_gains[:, 0]))
        ),
        spot_values_pv=np.ones((scenario_count, 2)),
        spot_step_gains=spot_gains,
        option_step_gains=np.zeros((scenario_count, 1, 1)),
        option_half_spreads_pv=np.zeros((scenario_count, 2, 1)),
        option_alive=np.asarray([[True], [False]]),
        option_settles_next=np.asarray([[True]]),
        terminal_claim_values_pv=np.zeros(scenario_count),
        contract_ids=("zero-gain-option",),
        target_label="virtual-recession",
        fit_as_of_date=fit_date,
        model_fit_identity="virtual-recession",
        measure="historical-P",
        seed=109,
    )
    scenario_set = HistoricalWeeklyScenarioSet(
        scenarios=scenarios,
        training_origin_dates=tuple(
            fit_date - timedelta(days=scenario_count - index + 2)
            for index in range(scenario_count)
        ),
        training_terminal_dates=tuple(
            fit_date - timedelta(days=scenario_count - index + 1)
            for index in range(scenario_count)
        ),
        terminal_claim_pv_scales=np.ones(scenario_count),
    )
    fit = fit_historical_weekly_hedge(
        scenario_set,
        claim_quantity=1.0,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(1),
        config=HistoricalSanosConfig(
            rolling_observations=scenario_count,
            objective=HistoricalSanosObjective.MEAN_VOLATILITY,
            risk_parameter=0.1,
            position_cap=10.0,
            solver_position_cap=100.0,
            maximum_iterations=10_000,
        ),
    )

    assert fit.long_policy.spot_positions[0] == pytest.approx(100.0, abs=1.0e-5)
    assert fit.short_policy.spot_positions[0] == pytest.approx(100.0, abs=1.0e-5)
    assert fit.zero_policy.spot_positions[0] == pytest.approx(100.0, abs=1.0e-5)
    assert fit.zero_policy.recession_direction
    assert "recession_direction=" in fit.zero_policy.solver_status
    assert fit.recession_direction
    assert fit.internal_cap_hits >= 3
    assert fit.executable_cap_hits == 0
    assert fit.spot_positions[0] == pytest.approx(0.0, abs=1.0e-5)
    assert not fit.converged

    diagnostic_fit = replace(
        fit,
        internal_cap_hits=2,
        executable_cap_hits=1,
        cap_hits=3,
    )
    policy = HistoricalSanosWeeklyPolicy(
        market_fit_identity="market-fit",
        policy_fit_identity="policy-fit",
        prices=WeeklyPriceRecord(
            common_market_price_per_fit_spot=0.0,
            q_expectation_per_fit_spot=None,
            buyer_indifference_per_fit_spot=0.0,
            seller_indifference_per_fit_spot=0.0,
            continuation_price_per_fit_spot=0.0,
        ),
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        fit_decision_index=0,
        scenarios=scenario_set,
        hedge=diagnostic_fit,
    )
    assert policy.cap_hits == 2
    assert policy.recession_direction
    assert policy.converged is False


def test_historical_weekly_scenarios_use_complete_pre_asof_blocks() -> None:
    day = synthetic_weekly_market_day()
    snapshot = day.hedge_snapshot(
        maximum_business_dte=5.0,
        episode_origin_date=day.trading_date,
    )
    contracts = tuple(quote.contract for quote in snapshot.option_quotes)
    instruction = WeeklyFitInstruction(
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        model="historical-sanos",
        measure="historical-P",
        target_label="realized-variance",
        claim_quantity=1.0,
        model_config_identity="model",
        policy_config_identity="policy",
        episode_origin_date=day.trading_date,
        decision_date=day.trading_date,
        fit_as_of_date=day.trading_date,
        training_dates=(date(2024, 1, 2), date(2024, 1, 3)),
        frozen_contract_ids=tuple(contract.contract_id for contract in contracts),
    )
    state = WeeklyDecisionState(
        instruction=instruction,
        decision_index=0,
        next_session_date=day.expiry_dates[0],
        market_day=day,
        contracts=contracts,
        spot_over_episode_origin=1.0,
        inception_forward_over_episode_spot=np.ones(6),
        inception_forward_path=np.ones(1),
        rolling_forward_path=np.ones(1),
        discount_to_origin=1.0,
        option_marks_pv=np.asarray([quote.mark for quote in snapshot.option_quotes]),
        option_half_spreads_pv=np.asarray(
            [quote.half_spread for quote in snapshot.option_quotes]
        ),
        option_alive=np.ones(len(contracts), dtype=bool),
        previous_spot_position=0.0,
        previous_option_positions=np.zeros(len(contracts)),
    )
    historical_dates = tuple(
        add_business_days(date(2024, 1, 2), offset)
        for offset in range(11)
    )
    historical_days = tuple(
        replace(day, trading_date=trading_date)
        for trading_date in historical_dates
    )
    transitions = tuple(
        OneDayTransition(
            origin=historical_days[index],
            next_date=historical_dates[index + 1],
            forward_adjusted_spot_return=0.002 * ((index % 3) - 1),
            discount_factor=0.999,
            forward_over_spot=1.001,
        )
        for index in range(10)
    )
    history = HistoricalWeeklyTransitionHistory(
        as_of_date=day.trading_date,
        transitions=transitions,
        endpoints=historical_days[1:],
    )
    scenario_set = build_historical_weekly_scenarios(
        history,
        state=state,
        claim=build_weekly_claim(
            day,
            target=WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
        ),
        model_fit_identity="historical-test",
    )
    assert scenario_set.scenarios.scenario_count == 6
    assert scenario_set.scenarios.step_count == 5
    assert scenario_set.training_origin_dates[0] == historical_dates[0]
    assert scenario_set.training_terminal_dates[-1] == historical_dates[-1]
    assert max(scenario_set.training_terminal_dates) <= day.trading_date
    assert np.all(np.isfinite(scenario_set.scenarios.option_step_gains))
    assert np.all(scenario_set.scenarios.terminal_claim_values_pv >= 0.0)
    second_claim = build_weekly_claim(
        day,
        target=WeeklyTarget(kind=WeeklyTargetKind.DOWN_AND_OUT_PUT),
    )
    attached = attach_historical_weekly_claim(
        scenario_set,
        state=state,
        claim=second_claim,
    )
    rebuilt = build_historical_weekly_scenarios(
        history,
        state=state,
        claim=second_claim,
        model_fit_identity="historical-test",
    )
    np.testing.assert_allclose(
        attached.scenarios.terminal_claim_values_pv,
        rebuilt.scenarios.terminal_claim_values_pv,
        atol=1.0e-14,
        rtol=1.0e-14,
    )


def test_restarted_scenarios_preserve_fixed_inception_forward_normalization() -> None:
    day = replace(
        synthetic_weekly_market_day(),
        forward_over_spot=np.asarray([1.01, 1.025, 1.04, 1.055, 1.07]),
    )
    spot_over_episode_origin = 1.08
    snapshot = day.hedge_snapshot(
        maximum_business_dte=5.0,
        spot_over_episode_origin=spot_over_episode_origin,
        episode_origin_date=date(2025, 1, 3),
    )
    contracts = tuple(quote.contract for quote in snapshot.option_quotes)
    inception_forward_curve = np.asarray([1.0, 1.02, 1.045, 1.075, 1.11, 1.15])
    current_normalized_spot = (
        spot_over_episode_origin / inception_forward_curve[1]
    )
    instruction = WeeklyFitInstruction(
        schedule=HedgeSchedule.RESTARTED_WEEKLY,
        model="historical-sanos",
        measure="historical-P",
        target_label="realized-variance",
        claim_quantity=1.0,
        model_config_identity="model",
        policy_config_identity="policy",
        episode_origin_date=date(2025, 1, 3),
        decision_date=day.trading_date,
        fit_as_of_date=day.trading_date,
        training_dates=(date(2024, 1, 2),),
        frozen_contract_ids=tuple(contract.contract_id for contract in contracts),
    )
    state = WeeklyDecisionState(
        instruction=instruction,
        decision_index=1,
        next_session_date=day.expiry_dates[0],
        market_day=day,
        contracts=contracts,
        spot_over_episode_origin=spot_over_episode_origin,
        inception_forward_over_episode_spot=inception_forward_curve,
        inception_forward_path=np.asarray([1.0, current_normalized_spot]),
        rolling_forward_path=np.asarray([1.0, 0.99]),
        discount_to_origin=1.0,
        option_marks_pv=np.asarray([quote.mark for quote in snapshot.option_quotes]),
        option_half_spreads_pv=np.asarray(
            [quote.half_spread for quote in snapshot.option_quotes]
        ),
        option_alive=np.ones(len(contracts), dtype=bool),
        previous_spot_position=0.0,
        previous_option_positions=np.zeros(len(contracts)),
    )
    historical_dates = tuple(
        add_business_days(date(2024, 1, 2), offset) for offset in range(5)
    )
    historical_forward_curve = np.asarray([1.003, 1.009, 1.018, 1.03, 1.045])
    historical_days = tuple(
        replace(
            day,
            trading_date=trading_date,
            forward_over_spot=historical_forward_curve,
        )
        for trading_date in historical_dates
    )
    returns = np.asarray([0.01, -0.004, 0.007, -0.002])
    one_day_forwards = np.asarray([1.001, 1.002, 1.003, 1.004])
    transitions = tuple(
        OneDayTransition(
            origin=historical_days[index],
            next_date=historical_dates[index + 1],
            forward_adjusted_spot_return=float(returns[index]),
            discount_factor=1.0,
            forward_over_spot=float(one_day_forwards[index]),
        )
        for index in range(4)
    )
    scenario_set = build_historical_weekly_scenarios(
        HistoricalWeeklyTransitionHistory(
            as_of_date=day.trading_date,
            transitions=transitions,
            endpoints=historical_days[1:],
        ),
        state=state,
        claim=build_weekly_claim(
            day,
            target=WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
        ),
        model_fit_identity="historical-forward-rescaling-test",
    )
    historical_spot_path = np.cumprod(one_day_forwards * (1.0 + returns))
    historical_forward_normalized_shocks = historical_spot_path / np.asarray(
        [
            historical_days[0].forward_at_business_dte(step)
            for step in range(1, 5)
        ]
    )
    expected_suffix = (
        spot_over_episode_origin
        * np.asarray([day.forward_at_business_dte(step) for step in range(1, 5)])
        * historical_forward_normalized_shocks
        / inception_forward_curve[2:]
    )
    np.testing.assert_allclose(
        scenario_set.scenarios.inception_forward_paths[0],
        np.concatenate(([current_normalized_spot], expected_suffix)),
        atol=1.0e-14,
        rtol=1.0e-14,
    )


def test_restarted_pure_hedge_respects_the_actual_carried_book() -> None:
    scenario_count = 20
    fit_date = date(2022, 12, 30)
    scenarios = WeeklyScenarioBatch(
        inception_forward_paths=np.ones((scenario_count, 2)),
        spot_values_pv=np.ones((scenario_count, 2)),
        spot_step_gains=np.zeros((scenario_count, 1)),
        option_step_gains=np.zeros((scenario_count, 1, 1)),
        option_half_spreads_pv=np.broadcast_to(
            np.asarray([[[0.1], [0.0]]]),
            (scenario_count, 2, 1),
        ).copy(),
        option_alive=np.asarray([[True], [False]]),
        option_settles_next=np.asarray([[True]]),
        terminal_claim_values_pv=np.linspace(0.0, 0.1, scenario_count),
        contract_ids=("settling",),
        target_label="carried-book-test",
        fit_as_of_date=fit_date,
        model_fit_identity="carried-book-test",
        measure="historical-P",
        seed=None,
    )
    scenario_set = HistoricalWeeklyScenarioSet(
        scenarios=scenarios,
        training_origin_dates=tuple(
            fit_date - timedelta(days=scenario_count - index + 2)
            for index in range(scenario_count)
        ),
        training_terminal_dates=tuple(
            fit_date - timedelta(days=scenario_count - index + 1)
            for index in range(scenario_count)
        ),
        terminal_claim_pv_scales=np.ones(scenario_count),
    )
    fit = fit_historical_weekly_hedge(
        scenario_set,
        claim_quantity=1.0,
        initial_spot_position=0.0,
        initial_option_positions=np.asarray([5.0]),
        config=HistoricalSanosConfig(
            rolling_observations=20,
            objective=HistoricalSanosObjective.MEAN_VARIANCE,
            risk_parameter=1.0,
            quadratic_ridge=0.0,
            maximum_iterations=2_000,
            gradient_tolerance=1.0e-9,
        ),
    )
    assert fit.long_policy.option_positions[0, 0] == pytest.approx(0.0, abs=1.0e-8)
    assert fit.zero_policy.option_positions[0, 0] == pytest.approx(0.0, abs=1.0e-8)
    assert fit.option_positions[0, 0] == pytest.approx(5.0, abs=1.0e-7)
    accounting = weekly_scenario_accounting(
        scenarios,
        spot_positions=fit.spot_positions,
        option_positions=fit.option_positions,
        signed_claim_quantity=1.0,
        initial_spot_position=0.0,
        initial_option_positions=np.asarray([5.0]),
    )
    np.testing.assert_allclose(accounting.transaction_costs, 0.0, atol=1.0e-8)


def test_mean_variance_prices_use_the_fitted_expected_cost_functional() -> None:
    base = _scenario_set()
    scenario_cost_multipliers = np.linspace(
        0.01,
        1.99,
        base.scenarios.scenario_count,
    )
    scenarios = replace(
        base.scenarios,
        spot_values_pv=(
            base.scenarios.spot_values_pv * scenario_cost_multipliers[:, None]
        ),
        option_half_spreads_pv=(
            base.scenarios.option_half_spreads_pv
            * scenario_cost_multipliers[:, None, None]
        ),
    )
    scenario_set = replace(base, scenarios=scenarios)
    config = HistoricalSanosConfig(
        rolling_observations=80,
        objective=HistoricalSanosObjective.MEAN_VARIANCE,
        risk_parameter=3.0,
        quadratic_ridge=1.0e-6,
        maximum_iterations=2_000,
        gradient_tolerance=1.0e-8,
    )
    fit = fit_historical_weekly_hedge(
        scenario_set,
        claim_quantity=1.0,
        initial_spot_position=0.0,
        initial_option_positions=np.zeros(2),
        config=config,
    )
    assert fit.prices.buyer == pytest.approx(
        fit.long_policy.certainty_equivalent
        - fit.zero_policy.certainty_equivalent,
        abs=1.0e-12,
    )
    exact_long = weekly_scenario_accounting(
        scenarios,
        spot_positions=fit.long_policy.spot_positions,
        option_positions=fit.long_policy.option_positions,
        signed_claim_quantity=1.0,
    )
    exact_long_ce = weekly_certainty_equivalent(
        exact_long.terminal_wealth,
        objective=WeeklyRiskObjective.MEAN_VARIANCE,
        risk_parameter=config.risk_parameter,
    )
    assert abs(exact_long_ce - fit.long_policy.certainty_equivalent) > 1.0e-7


def test_historical_sanos_config_rejects_invalid_fallback_time_limit() -> None:
    with pytest.raises(ValueError, match="fallback_time_limit_seconds"):
        HistoricalSanosConfig(fallback_time_limit_seconds=0.0)
