"""Open-loop multistep historical SANOS policy optimization."""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Iterator

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp

from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_scenarios import (
    WeeklyIndifferencePrices,
    WeeklyRiskObjective,
    WeeklyScenarioBatch,
    weekly_certainty_equivalent,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_scenarios import (
    HistoricalWeeklyScenarioSet,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class HistoricalWeeklyOpenLoopFit:
    """One signed-claim open-loop schedule and diagnostics."""

    spot_positions: FloatArray
    option_positions: FloatArray
    cash_translation: float | None
    training_loss: float
    validation_loss: float
    certainty_equivalent: float
    iterations: int
    converged: bool
    maximum_absolute_gradient: float
    selection_solver_status: str
    solver_status: str
    selection_objective_error: float
    objective_error: float
    selection_maximum_constraint_violation: float
    maximum_constraint_violation: float
    scenario_count: int
    parameter_count: int
    observation_parameter_ratio: float
    statistically_thin: bool
    selection_recession_direction: bool
    recession_direction: bool
    cap_hits: int


@dataclass(frozen=True, kw_only=True)
class HistoricalWeeklyHedgeFit:
    """Pricing policies and the cost-consistent pure hedge schedule."""

    spot_positions: FloatArray
    option_positions: FloatArray
    long_policy: HistoricalWeeklyOpenLoopFit
    short_policy: HistoricalWeeklyOpenLoopFit
    zero_policy: HistoricalWeeklyOpenLoopFit
    pure_policy: HistoricalWeeklyOpenLoopFit
    prices: WeeklyIndifferencePrices
    converged: bool
    recession_direction: bool
    internal_cap_hits: int
    executable_cap_hits: int
    cap_hits: int


@dataclass(frozen=True, kw_only=True)
class _ConvexCostLassoProblem:
    scaled_positions: cp.Variable
    claim: cp.Parameter
    action_scale: FloatArray
    problem: cp.Problem


@dataclass(frozen=True, kw_only=True)
class _EntropyCostProblem:
    scaled_positions: cp.Variable
    claim: cp.Parameter
    action_scale: FloatArray
    problem: cp.Problem


@dataclass(frozen=True, kw_only=True)
class _PowerCostProblem:
    scaled_positions: cp.Variable
    cash_translation: cp.Variable
    claim: cp.Parameter
    action_scale: FloatArray
    problem: cp.Problem


class WeeklyOpenLoopPricingContext:
    """Reusable target-independent pricing geometry for one scenario state."""

    def __init__(
        self,
        scenarios: WeeklyScenarioBatch,
        *,
        config: HistoricalSanosConfig,
    ) -> None:
        self._scenarios = scenarios
        self._config = config
        self._long_geometry = _WeeklyOpenLoopGeometry(
            scenarios,
            initial_spot_position=0.0,
            initial_option_positions=np.zeros(scenarios.contract_count),
            config=config,
        )
        self._short_geometry = _WeeklyOpenLoopGeometry(
            scenarios,
            initial_spot_position=0.0,
            initial_option_positions=np.zeros(scenarios.contract_count),
            config=config,
        )
        zero_geometry = _WeeklyOpenLoopGeometry(
            scenarios,
            initial_spot_position=0.0,
            initial_option_positions=np.zeros(scenarios.contract_count),
            config=config,
        )
        self._zero_policy = zero_geometry.fit_policy(
            signed_claim_quantity=0.0,
            terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
        )
        self._execution_geometry_key: bytes | None = None
        self._execution_geometry: _WeeklyOpenLoopGeometry | None = None
        self._use_lock = Lock()

    @contextmanager
    def exclusive_use(self) -> Iterator[None]:
        """Reject concurrent target fits on the mutable solver geometries."""
        if not self._use_lock.acquire(blocking=False):
            raise RuntimeError("weekly pricing context is already in use")
        try:
            yield
        finally:
            self._use_lock.release()

    def validate(
        self,
        scenarios: WeeklyScenarioBatch,
        *,
        config: HistoricalSanosConfig,
    ) -> None:
        """Reject use with any different generated instrument geometry."""
        self._validate(scenarios, config=config)

    def fit_long(
        self,
        scenarios: WeeklyScenarioBatch,
        *,
        claim_quantity: float,
    ) -> HistoricalWeeklyOpenLoopFit:
        """Fit the long claim on its reusable solver geometry."""
        self._long_geometry.clear_compiled_problems()
        return self._long_geometry.fit_policy(
            signed_claim_quantity=claim_quantity,
            terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
        )

    @property
    def zero_policy(self) -> HistoricalWeeklyOpenLoopFit:
        """Return the cold-start zero-claim fit shared by every target."""
        return self._zero_policy

    def fit_short(
        self,
        scenarios: WeeklyScenarioBatch,
        *,
        claim_quantity: float,
    ) -> HistoricalWeeklyOpenLoopFit:
        """Fit the short claim on an independent reusable solver geometry."""
        self._short_geometry.clear_compiled_problems()
        return self._short_geometry.fit_policy(
            signed_claim_quantity=-claim_quantity,
            terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
        )

    def execution_geometry(
        self,
        scenarios: WeeklyScenarioBatch,
        *,
        initial_spot_position: float,
        initial_option_positions: NDArray[np.floating],
    ) -> _WeeklyOpenLoopGeometry:
        """Reuse one exact pure-hedge geometry when prior positions agree."""
        self._validate(scenarios, config=self._config)
        initial_options = np.ascontiguousarray(
            initial_option_positions,
            dtype=np.float64,
        )
        key = (
            np.asarray([initial_spot_position], dtype=np.float64).tobytes()
            + initial_options.tobytes()
        )
        if key != self._execution_geometry_key:
            self._execution_geometry = _WeeklyOpenLoopGeometry(
                scenarios,
                initial_spot_position=initial_spot_position,
                initial_option_positions=initial_options,
                config=self._config,
                center_instrument_gains=True,
            )
            self._execution_geometry_key = key
        if self._execution_geometry is None:
            raise RuntimeError("weekly execution geometry was not initialized")
        return self._execution_geometry

    def _validate(
        self,
        scenarios: WeeklyScenarioBatch,
        *,
        config: HistoricalSanosConfig,
    ) -> None:
        if config != self._config:
            raise ValueError("weekly pricing context uses another configuration")
        shared_arrays = (
            "spot_values_pv",
            "spot_step_gains",
            "option_step_gains",
            "option_half_spreads_pv",
            "option_alive",
        )
        if any(
            getattr(scenarios, name) is not getattr(self._scenarios, name)
            for name in shared_arrays
        ):
            raise ValueError("weekly pricing context uses another scenario geometry")
        if (
            scenarios.contract_ids != self._scenarios.contract_ids
            or scenarios.fit_as_of_date != self._scenarios.fit_as_of_date
            or scenarios.model_fit_identity != self._scenarios.model_fit_identity
            or scenarios.measure != self._scenarios.measure
            or scenarios.seed != self._scenarios.seed
        ):
            raise ValueError("weekly pricing context identity changed")


def fit_historical_weekly_hedge(
    scenario_set: HistoricalWeeklyScenarioSet,
    *,
    claim_quantity: float,
    initial_spot_position: float,
    initial_option_positions: NDArray[np.floating],
    config: HistoricalSanosConfig = HistoricalSanosConfig(),
    pricing_context: WeeklyOpenLoopPricingContext | None = None,
) -> HistoricalWeeklyHedgeFit:
    """Fit pricing policies and a cost-consistent pure hedge schedule."""
    return fit_weekly_open_loop_hedge(
        scenario_set.scenarios,
        claim_quantity=claim_quantity,
        initial_spot_position=initial_spot_position,
        initial_option_positions=initial_option_positions,
        config=config,
        pricing_context=pricing_context,
    )


def fit_weekly_open_loop_hedge(
    scenarios: WeeklyScenarioBatch,
    *,
    claim_quantity: float,
    initial_spot_position: float,
    initial_option_positions: NDArray[np.floating],
    config: HistoricalSanosConfig = HistoricalSanosConfig(),
    pricing_context: WeeklyOpenLoopPricingContext | None = None,
) -> HistoricalWeeklyHedgeFit:
    """Fit pricing policies and a drift-neutral pure hedge on weekly paths."""
    if not math.isfinite(claim_quantity) or claim_quantity <= 0.0:
        raise ValueError("weekly claim quantity must be finite-positive")
    if pricing_context is None:
        pricing_context = WeeklyOpenLoopPricingContext(scenarios, config=config)
    with pricing_context.exclusive_use():
        pricing_context.validate(scenarios, config=config)
        execution_geometry = pricing_context.execution_geometry(
            scenarios,
            initial_spot_position=initial_spot_position,
            initial_option_positions=initial_option_positions,
        )
        execution_geometry.clear_compiled_problems()
        with ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="weekly-open-loop",
        ) as executor:
            long_future = executor.submit(
                pricing_context.fit_long,
                scenarios,
                claim_quantity=claim_quantity,
            )
            short_future = executor.submit(
                pricing_context.fit_short,
                scenarios,
                claim_quantity=claim_quantity,
            )
            pure_future = executor.submit(
                execution_geometry.fit_policy,
                signed_claim_quantity=claim_quantity,
                terminal_claim_values_pv=scenarios.terminal_claim_values_pv,
            )
            long_policy = long_future.result()
            short_policy = short_future.result()
            pure_policy = pure_future.result()
        zero_policy = pricing_context.zero_policy
    prices = WeeklyIndifferencePrices(
        buyer=long_policy.certainty_equivalent - zero_policy.certainty_equivalent,
        seller=zero_policy.certainty_equivalent - short_policy.certainty_equivalent,
    )
    raw_positions = np.concatenate(
        (pure_policy.spot_positions[:, None], pure_policy.option_positions),
        axis=1,
    )
    executable_positions = np.clip(
        raw_positions,
        -config.position_cap,
        config.position_cap,
    )
    executable_cap_hits = int(np.count_nonzero(executable_positions != raw_positions))
    internal_cap_hits = (
        long_policy.cap_hits
        + short_policy.cap_hits
        + zero_policy.cap_hits
        + pure_policy.cap_hits
    )
    cap_hits = internal_cap_hits + executable_cap_hits
    recession_direction = any(
        policy.recession_direction
        for policy in (long_policy, short_policy, zero_policy, pure_policy)
    )
    return HistoricalWeeklyHedgeFit(
        spot_positions=executable_positions[:, 0],
        option_positions=executable_positions[:, 1:],
        long_policy=long_policy,
        short_policy=short_policy,
        zero_policy=zero_policy,
        pure_policy=pure_policy,
        prices=prices,
        converged=(
            long_policy.converged
            and short_policy.converged
            and zero_policy.converged
            and pure_policy.converged
            and cap_hits == 0
            and not recession_direction
        ),
        recession_direction=recession_direction,
        internal_cap_hits=internal_cap_hits,
        executable_cap_hits=executable_cap_hits,
        cap_hits=cap_hits,
    )


class _WeeklyOpenLoopGeometry:
    def __init__(
        self,
        scenarios: WeeklyScenarioBatch | HistoricalWeeklyScenarioSet,
        *,
        initial_spot_position: float,
        initial_option_positions: NDArray[np.floating],
        config: HistoricalSanosConfig,
        center_instrument_gains: bool = False,
    ) -> None:
        self.scenarios = (
            scenarios.scenarios
            if isinstance(scenarios, HistoricalWeeklyScenarioSet)
            else scenarios
        )
        self.initial_spot = float(initial_spot_position)
        self.initial_options = np.asarray(initial_option_positions, dtype=np.float64)
        self.config = config
        self.center_instrument_gains = center_instrument_gains
        if not math.isfinite(self.initial_spot):
            raise ValueError("historical weekly initial spot position must be finite")
        if self.initial_options.shape != (self.scenarios.contract_count,) or not np.all(
            np.isfinite(self.initial_options)
        ):
            raise ValueError("historical weekly initial option positions do not align")
        if np.any(
            np.abs(self.initial_options[~self.scenarios.option_alive[0]]) > 1.0e-14
        ):
            raise ValueError("expired historical weekly options cannot retain positions")
        self.option_parameter_pairs = tuple(
            (step, contract)
            for step in range(self.scenarios.step_count)
            for contract in np.flatnonzero(self.scenarios.option_alive[step])
        )
        self.parameter_count = self.scenarios.step_count + len(
            self.option_parameter_pairs
        )
        observation_parameter_ratio = (
            self.scenarios.scenario_count / self.parameter_count
        )
        if observation_parameter_ratio < self.config.thin_sample_ratio:
            warnings.warn(
                "historical weekly policy is statistically thin: "
                f"observations={self.scenarios.scenario_count} "
                f"parameters={self.parameter_count} "
                f"ratio={observation_parameter_ratio:.3f}; the fitted policy is "
                "an explicitly regularized estimate, not an unrestricted empirical "
                "optimum",
                RuntimeWarning,
                stacklevel=2,
            )
        self._option_parameter_index = {
            pair: self.scenarios.step_count + index
            for index, pair in enumerate(self.option_parameter_pairs)
        }
        self._spot_gain_means = np.mean(
            self.scenarios.spot_step_gains,
            axis=0,
        )
        self._option_gain_means = np.mean(
            self.scenarios.option_step_gains,
            axis=0,
        )
        self._instrument_gains = self._build_instrument_gains()
        (
            self._turnover_matrix,
            self._turnover_offset,
            self._scenario_turnover_weights,
        ) = self._build_scenario_turnover_geometry()
        self._convex_problems: dict[bytes, _ConvexCostLassoProblem] = {}
        self._entropy_problems: dict[bytes, _EntropyCostProblem] = {}
        self._power_problems: dict[bytes, _PowerCostProblem] = {}

    def clear_compiled_problems(self) -> None:
        """Discard target-specific canonicalizations before another claim."""
        self._convex_problems.clear()
        self._entropy_problems.clear()
        self._power_problems.clear()

    def fit_policy(
        self,
        *,
        signed_claim_quantity: float,
        terminal_claim_values_pv: NDArray[np.floating] | None = None,
    ) -> HistoricalWeeklyOpenLoopFit:
        claim_values = self._claim_values(terminal_claim_values_pv)
        observation_count = self.scenarios.scenario_count
        validation_count = max(
            1,
            int(round(observation_count * self.config.validation_fraction)),
        )
        split = observation_count - validation_count
        if split < 2:
            raise ValueError("historical weekly training split has fewer than two rows")
        include_translation = (
            self.config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL
        )
        initial = np.zeros(self.parameter_count + int(include_translation))
        bounds = [
            (-self.config.solver_position_cap, self.config.solver_position_cap)
        ] * self.parameter_count
        if include_translation:
            bounds.append((None, None))
        selection = self._minimize(
            initial,
            np.arange(split),
            signed_claim_quantity=signed_claim_quantity,
            terminal_claim_values_pv=claim_values,
            include_translation=include_translation,
            bounds=bounds,
        )
        validation_loss = self._objective_value(
            selection.x,
            np.arange(split, observation_count),
            signed_claim_quantity=signed_claim_quantity,
            terminal_claim_values_pv=claim_values,
            include_translation=include_translation,
            action_scale=np.asarray(selection.action_scale, dtype=np.float64),
        )
        final = self._minimize(
            np.asarray(selection.x, dtype=np.float64),
            np.arange(observation_count),
            signed_claim_quantity=signed_claim_quantity,
            terminal_claim_values_pv=claim_values,
            include_translation=include_translation,
            bounds=bounds,
        )
        training_loss = self._objective_value(
            final.x,
            np.arange(observation_count),
            signed_claim_quantity=signed_claim_quantity,
            terminal_claim_values_pv=claim_values,
            include_translation=include_translation,
            action_scale=np.asarray(final.action_scale, dtype=np.float64),
        )
        if self.config.objective in (
            HistoricalSanosObjective.MEAN_VARIANCE,
            HistoricalSanosObjective.MEAN_VOLATILITY,
            HistoricalSanosObjective.ENTROPY,
            HistoricalSanosObjective.POWER_EXPONENTIAL,
        ):
            optimality = float(getattr(final, "optimality", math.inf))
        else:
            _, gradient = self._loss_and_gradient(
                final.x,
                np.arange(observation_count),
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=claim_values,
                include_translation=include_translation,
            )
            optimality = float(np.max(np.abs(gradient)))
        spot, options = self.unpack(final.x[: self.parameter_count])
        certainty_equivalent = self._policy_certainty_equivalent(
            final.x,
            signed_claim_quantity=signed_claim_quantity,
            terminal_claim_values_pv=claim_values,
            include_translation=include_translation,
        )
        positions = np.concatenate((spot[:, None], options), axis=1)
        cap_hits = int(
            np.count_nonzero(
                np.abs(positions) >= 0.999 * self.config.solver_position_cap
            )
        )
        selection_recession_direction = bool(
            getattr(selection, "recession_direction", False)
        )
        recession_direction = bool(getattr(final, "recession_direction", False))
        ratio = observation_count / self.parameter_count
        return HistoricalWeeklyOpenLoopFit(
            spot_positions=spot,
            option_positions=options,
            cash_translation=(
                float(final.x[-1]) if include_translation else None
            ),
            training_loss=float(training_loss),
            validation_loss=float(validation_loss),
            certainty_equivalent=certainty_equivalent,
            iterations=int(selection.nit) + int(final.nit),
            converged=bool(
                final.success and cap_hits == 0 and not recession_direction
            ),
            maximum_absolute_gradient=optimality,
            selection_solver_status=str(selection.message),
            solver_status=str(final.message),
            selection_objective_error=float(
                getattr(selection, "objective_error", math.inf)
            ),
            objective_error=float(getattr(final, "objective_error", math.inf)),
            selection_maximum_constraint_violation=float(
                getattr(selection, "constraint_violation", math.inf)
            ),
            maximum_constraint_violation=float(
                getattr(final, "constraint_violation", math.inf)
            ),
            scenario_count=observation_count,
            parameter_count=self.parameter_count,
            observation_parameter_ratio=ratio,
            statistically_thin=ratio < self.config.thin_sample_ratio,
            selection_recession_direction=selection_recession_direction,
            recession_direction=recession_direction,
            cap_hits=cap_hits,
        )

    def _policy_certainty_equivalent(
        self,
        parameters: NDArray[np.floating],
        *,
        signed_claim_quantity: float,
        terminal_claim_values_pv: FloatArray,
        include_translation: bool,
    ) -> float:
        indices = np.arange(self.scenarios.scenario_count, dtype=np.int64)
        values = np.asarray(parameters, dtype=np.float64)
        if self.config.objective in (
            HistoricalSanosObjective.MEAN_VARIANCE,
            HistoricalSanosObjective.MEAN_VOLATILITY,
        ):
            claim, instrument_gains = self._linear_wealth_geometry(
                indices,
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
            )
            turnover_matrix, turnover_offset, turnover_weights = (
                self._turnover_geometry(indices)
            )
            positions = values[: self.parameter_count]
            deterministic_cost = float(
                turnover_weights
                @ np.abs(turnover_matrix @ positions - turnover_offset)
            )
            wealth = claim + instrument_gains @ positions - deterministic_cost
        elif self.config.objective in (
            HistoricalSanosObjective.ENTROPY,
            HistoricalSanosObjective.POWER_EXPONENTIAL,
        ):
            wealth = self._exact_scenario_wealth(
                values[: self.parameter_count],
                indices=indices,
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
            )
        else:
            wealth, _ = self._smooth_wealth_and_jacobian(
                values[: self.parameter_count],
                indices=indices,
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
            )
            if include_translation and len(values) != self.parameter_count + 1:
                raise ValueError("weekly translated policy has the wrong shape")
        return _weekly_certainty_equivalent(wealth, config=self.config)

    def _minimize(
        self,
        initial: FloatArray,
        indices: NDArray[np.integer],
        *,
        signed_claim_quantity: float,
        terminal_claim_values_pv: FloatArray,
        include_translation: bool,
        bounds: list[tuple[float | None, float | None]],
    ) -> OptimizeResult:
        if self.config.objective in (
            HistoricalSanosObjective.MEAN_VARIANCE,
            HistoricalSanosObjective.MEAN_VOLATILITY,
        ):
            return self._minimize_convex_cost_lasso(
                initial,
                np.asarray(indices, dtype=np.int64),
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
            )
        if self.config.objective is HistoricalSanosObjective.ENTROPY:
            return self._minimize_entropy_cost(
                initial,
                np.asarray(indices, dtype=np.int64),
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
            )
        if self.config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL:
            return self._minimize_power_cost(
                initial,
                np.asarray(indices, dtype=np.int64),
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
            )
        return minimize(
            lambda parameters: self._loss_and_gradient(
                parameters,
                indices,
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
                include_translation=include_translation,
            ),
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": self.config.maximum_iterations,
                "maxfun": 4 * self.config.maximum_iterations,
                "gtol": self.config.gradient_tolerance,
                "ftol": 0.0,
                "maxls": 50,
            },
        )

    def _minimize_entropy_cost(
        self,
        initial: FloatArray,
        indices: NDArray[np.int64],
        *,
        signed_claim_quantity: float,
        terminal_claim_values_pv: FloatArray,
    ) -> OptimizeResult:
        claim, instrument_gains = self._linear_wealth_geometry(
            indices,
            signed_claim_quantity=signed_claim_quantity,
            terminal_claim_values_pv=terminal_claim_values_pv,
        )
        compiled = self._entropy_cost_problem(
            indices,
            instrument_gains=instrument_gains,
        )
        compiled.claim.value = claim
        compiled.scaled_positions.value = (
            np.asarray(initial[: self.parameter_count], dtype=np.float64)
            / compiled.action_scale
        )
        iterations = 0
        retry_tolerance = max(
            5.0 * self.config.gradient_tolerance,
            5.0e-8,
        )
        primary_traces: list[str] = []
        try:
            for attempt, tolerance in enumerate(
                (self.config.gradient_tolerance, retry_tolerance)
            ):
                compiled.problem.solve(
                    solver=cp.CLARABEL,
                    warm_start=attempt == 0,
                    max_iter=self.config.maximum_iterations,
                    tol_gap_abs=tolerance,
                    tol_gap_rel=tolerance,
                    tol_feas=tolerance,
                )
                iterations += int(compiled.problem.solver_stats.num_iters or 0)
                primary_solver = str(
                    compiled.problem.solver_stats.solver_name or "CLARABEL"
                )
                primary_traces.append(
                    f"{primary_solver}={compiled.problem.status}"
                )
                if compiled.problem.status != cp.OPTIMAL_INACCURATE:
                    break
        except cp.error.SolverError as error:
            primary_trace = "; ".join(
                (*primary_traces, f"CLARABEL failed: {error}")
            )
            solver_status, solver_trace, independent_iterations = (
                self._solve_conic_with_scs(
                    compiled.problem,
                    primary_trace=primary_trace,
                )
            )
        else:
            solver_status = compiled.problem.status
            solver_trace = "; ".join(primary_traces)
            independent_iterations = 0
            if solver_status == cp.OPTIMAL_INACCURATE:
                solver_status, solver_trace, independent_iterations = (
                    self._solve_conic_with_scs(
                        compiled.problem,
                        primary_trace=solver_trace,
                    )
                )
        iterations += independent_iterations

        def certify_current_solution() -> tuple[
            FloatArray,
            int,
            bool,
            float,
            float,
            float,
        ]:
            if compiled.scaled_positions.value is None:
                fitted = np.asarray(
                    initial[: self.parameter_count],
                    dtype=np.float64,
                )
            else:
                fitted = compiled.action_scale * np.asarray(
                    compiled.scaled_positions.value,
                    dtype=np.float64,
                )
            cap_hits = int(
                np.count_nonzero(
                    np.abs(fitted) >= 0.999 * self.config.solver_position_cap
                )
            )
            constraint_violation = _maximum_constraint_violation(compiled.problem)
            certified, exact_objective, objective_error = self._certify_conic_result(
                fitted,
                indices,
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
                include_translation=False,
                action_scale=compiled.action_scale,
                solver_status=solver_status,
                solver_objective=compiled.problem.value,
                maximum_constraint_violation=constraint_violation,
            )
            return (
                fitted,
                cap_hits,
                certified,
                exact_objective,
                objective_error,
                constraint_violation,
            )

        (
            fitted,
            cap_hits,
            certified,
            exact_objective,
            objective_error,
            constraint_violation,
        ) = certify_current_solution()
        if (
            not certified
            and "SCS=" not in solver_trace
            and "SCS failed:" not in solver_trace
        ):
            solver_status, solver_trace, fallback_iterations = (
                self._solve_conic_with_scs(
                    compiled.problem,
                    primary_trace=solver_trace,
                )
            )
            iterations += fallback_iterations
            (
                fitted,
                cap_hits,
                certified,
                exact_objective,
                objective_error,
                constraint_violation,
            ) = certify_current_solution()
        result = OptimizeResult(
            x=fitted,
            nit=iterations,
            success=certified and cap_hits == 0,
            optimality=math.inf,
            message=(
                f"{solver_trace}; exact-objective-error={objective_error:.6g}; "
                f"constraint-violation={constraint_violation:.6g}"
            ),
            fun=exact_objective,
            objective_error=objective_error,
            constraint_violation=constraint_violation,
        )
        result.action_scale = compiled.action_scale
        return result

    def _entropy_cost_problem(
        self,
        indices: NDArray[np.int64],
        *,
        instrument_gains: FloatArray,
    ) -> _EntropyCostProblem:
        key = np.ascontiguousarray(indices).tobytes()
        cached = self._entropy_problems.get(key)
        if cached is not None:
            return cached
        action_scale = self._action_scale(indices)
        scaled_positions = cp.Variable(self.parameter_count)
        positions = cp.multiply(action_scale, scaled_positions)
        claim = cp.Parameter(len(indices))
        turnover = self._turnover_matrix @ positions - self._turnover_offset
        scenario_costs = self._scenario_turnover_weights[indices] @ cp.abs(
            turnover
        )
        wealth = claim + instrument_gains @ positions - scenario_costs
        risk = self.config.risk_parameter
        objective = cp.Minimize(
            (cp.log_sum_exp(-risk * wealth) - math.log(len(indices))) / risk
            + 0.5
            * self.config.quadratic_ridge
            * cp.sum_squares(scaled_positions)
        )
        problem = cp.Problem(objective)
        if not problem.is_dpp():
            raise RuntimeError("weekly entropy problem is not parameter-cacheable")
        cached = _EntropyCostProblem(
            scaled_positions=scaled_positions,
            claim=claim,
            action_scale=np.asarray(action_scale, dtype=np.float64),
            problem=problem,
        )
        self._entropy_problems[key] = cached
        return cached

    def _minimize_power_cost(
        self,
        initial: FloatArray,
        indices: NDArray[np.int64],
        *,
        signed_claim_quantity: float,
        terminal_claim_values_pv: FloatArray,
    ) -> OptimizeResult:
        claim, instrument_gains = self._linear_wealth_geometry(
            indices,
            signed_claim_quantity=signed_claim_quantity,
            terminal_claim_values_pv=terminal_claim_values_pv,
        )
        compiled = self._power_cost_problem(
            indices,
            instrument_gains=instrument_gains,
        )
        compiled.claim.value = claim
        compiled.scaled_positions.value = (
            np.asarray(initial[: self.parameter_count], dtype=np.float64)
            / compiled.action_scale
        )
        compiled.cash_translation.value = float(initial[-1])
        iterations = 0
        retry_tolerance = max(
            5.0 * self.config.gradient_tolerance,
            5.0e-8,
        )
        primary_traces: list[str] = []
        try:
            for attempt, tolerance in enumerate(
                (self.config.gradient_tolerance, retry_tolerance)
            ):
                compiled.problem.solve(
                    solver=cp.CLARABEL,
                    warm_start=attempt == 0,
                    max_iter=self.config.maximum_iterations,
                    tol_gap_abs=tolerance,
                    tol_gap_rel=tolerance,
                    tol_feas=tolerance,
                )
                iterations += int(compiled.problem.solver_stats.num_iters or 0)
                primary_solver = str(
                    compiled.problem.solver_stats.solver_name or "CLARABEL"
                )
                primary_traces.append(
                    f"{primary_solver}={compiled.problem.status}"
                )
                if compiled.problem.status != cp.OPTIMAL_INACCURATE:
                    break
        except cp.error.SolverError as error:
            primary_trace = "; ".join(
                (*primary_traces, f"CLARABEL failed: {error}")
            )
            solver_status, solver_trace, independent_iterations = (
                self._solve_conic_with_scs(
                    compiled.problem,
                    primary_trace=primary_trace,
                )
            )
        else:
            solver_status = compiled.problem.status
            solver_trace = "; ".join(primary_traces)
            independent_iterations = 0
            if solver_status == cp.OPTIMAL_INACCURATE:
                solver_status, solver_trace, independent_iterations = (
                    self._solve_conic_with_scs(
                        compiled.problem,
                        primary_trace=solver_trace,
                    )
                )
        iterations += independent_iterations

        def certify_current_solution() -> tuple[
            FloatArray,
            int,
            float,
            bool,
            float,
            float,
            float,
        ]:
            if compiled.scaled_positions.value is None:
                fitted = np.asarray(
                    initial[: self.parameter_count],
                    dtype=np.float64,
                )
            else:
                fitted = compiled.action_scale * np.asarray(
                    compiled.scaled_positions.value,
                    dtype=np.float64,
                )
            cash_translation = (
                float(initial[-1])
                if compiled.cash_translation.value is None
                else float(compiled.cash_translation.value)
            )
            cap_hits = int(
                np.count_nonzero(
                    np.abs(fitted) >= 0.999 * self.config.solver_position_cap
                )
            )
            parameters = np.concatenate((fitted, [cash_translation]))
            constraint_violation = _maximum_constraint_violation(compiled.problem)
            certified, exact_objective, objective_error = self._certify_conic_result(
                parameters,
                indices,
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
                include_translation=True,
                action_scale=compiled.action_scale,
                solver_status=solver_status,
                solver_objective=compiled.problem.value,
                maximum_constraint_violation=constraint_violation,
            )
            return (
                parameters,
                cap_hits,
                cash_translation,
                certified,
                exact_objective,
                objective_error,
                constraint_violation,
            )

        (
            parameters,
            cap_hits,
            cash_translation,
            certified,
            exact_objective,
            objective_error,
            constraint_violation,
        ) = certify_current_solution()
        if (
            not certified
            and "SCS=" not in solver_trace
            and "SCS failed:" not in solver_trace
        ):
            solver_status, solver_trace, fallback_iterations = (
                self._solve_conic_with_scs(
                    compiled.problem,
                    primary_trace=solver_trace,
                )
            )
            iterations += fallback_iterations
            (
                parameters,
                cap_hits,
                cash_translation,
                certified,
                exact_objective,
                objective_error,
                constraint_violation,
            ) = certify_current_solution()
        result = OptimizeResult(
            x=parameters,
            nit=iterations,
            success=certified and cap_hits == 0,
            optimality=math.inf,
            message=(
                f"{solver_trace}; exact-objective-error={objective_error:.6g}; "
                f"constraint-violation={constraint_violation:.6g}"
            ),
            fun=exact_objective,
            objective_error=objective_error,
            constraint_violation=constraint_violation,
        )
        result.action_scale = compiled.action_scale
        return result

    def _power_cost_problem(
        self,
        indices: NDArray[np.int64],
        *,
        instrument_gains: FloatArray,
    ) -> _PowerCostProblem:
        key = np.ascontiguousarray(indices).tobytes()
        cached = self._power_problems.get(key)
        if cached is not None:
            return cached
        risk = self.config.risk_parameter
        action_scale = self._action_scale(indices)
        scaled_positions = cp.Variable(self.parameter_count)
        positions = cp.multiply(action_scale, scaled_positions)
        cash_translation = cp.Variable()
        claim = cp.Parameter(len(indices))
        turnover = self._turnover_matrix @ positions - self._turnover_offset
        scenario_costs = self._scenario_turnover_weights[indices] @ cp.abs(
            turnover
        )
        wealth = claim + instrument_gains @ positions - scenario_costs
        loss_argument = cp.Variable(len(indices))
        negative_part = cp.Variable(len(indices))
        positive_part = cp.Variable(len(indices))
        scenario_loss = cp.Variable(len(indices))
        constraints = [
            loss_argument >= cash_translation - wealth,
            negative_part <= 0.0,
            positive_part >= 0.0,
            negative_part + positive_part == loss_argument,
            scenario_loss
            >= (cp.exp(risk * negative_part) - 1.0) / risk + positive_part,
            scenario_loss
            >= loss_argument
            + 0.5 * risk * cp.square(cp.pos(loss_argument)),
        ]
        objective = cp.Minimize(
            -cash_translation
            + cp.sum(scenario_loss) / len(indices)
            + 0.5
            * self.config.quadratic_ridge
            * cp.sum_squares(scaled_positions)
        )
        problem = cp.Problem(objective, constraints)
        if not problem.is_dpp():
            raise RuntimeError("weekly power problem is not parameter-cacheable")
        cached = _PowerCostProblem(
            scaled_positions=scaled_positions,
            cash_translation=cash_translation,
            claim=claim,
            action_scale=np.asarray(action_scale, dtype=np.float64),
            problem=problem,
        )
        self._power_problems[key] = cached
        return cached

    def _minimize_convex_cost_lasso(
        self,
        initial: FloatArray,
        indices: NDArray[np.int64],
        *,
        signed_claim_quantity: float,
        terminal_claim_values_pv: FloatArray,
    ) -> OptimizeResult:
        claim, instrument_gains = self._linear_wealth_geometry(
            indices,
            signed_claim_quantity=signed_claim_quantity,
            terminal_claim_values_pv=terminal_claim_values_pv,
        )
        compiled = self._convex_cost_lasso_problem(
            indices,
            instrument_gains=instrument_gains,
        )
        compiled.claim.value = claim
        compiled.scaled_positions.value = (
            np.asarray(initial[: self.parameter_count], dtype=np.float64)
            / compiled.action_scale
        )
        iterations = 0
        try:
            if self.config.objective is HistoricalSanosObjective.MEAN_VARIANCE:
                compiled.problem.solve(
                    solver=cp.OSQP,
                    warm_start=False,
                    max_iter=max(10_000, self.config.maximum_iterations),
                    eps_abs=self.config.gradient_tolerance,
                    eps_rel=self.config.gradient_tolerance,
                    polishing=True,
                )
                iterations += int(compiled.problem.solver_stats.num_iters or 0)
            else:
                for tolerance in (
                    self.config.gradient_tolerance,
                    self.config.certificate_tolerance,
                ):
                    compiled.problem.solve(
                        solver=cp.CLARABEL,
                        warm_start=False,
                        max_iter=self.config.maximum_iterations,
                        tol_gap_abs=tolerance,
                        tol_gap_rel=tolerance,
                        tol_feas=tolerance,
                    )
                    iterations += int(compiled.problem.solver_stats.num_iters or 0)
                    if compiled.problem.status != cp.OPTIMAL_INACCURATE:
                        break
        except cp.error.SolverError as error:
            result = OptimizeResult(
                x=np.asarray(initial[: self.parameter_count], dtype=np.float64),
                nit=iterations,
                success=False,
                optimality=math.inf,
                message=f"convex solver failed: {error}",
                objective_error=math.inf,
            )
            result.action_scale = compiled.action_scale
            return result
        solver_status = compiled.problem.status
        solver_name = str(compiled.problem.solver_stats.solver_name or "primary")
        solver_trace = f"{solver_name}={solver_status}"
        if self.config.objective is not HistoricalSanosObjective.MEAN_VARIANCE:
            solver_status, solver_trace, independent_iterations = (
                self._resolve_inaccurate_conic_problem(compiled.problem)
            )
            iterations += independent_iterations
        fitted, certified, exact_objective, objective_error, constraint_violation = (
            self._certify_raw_convex_solution(
                compiled,
                initial,
                indices,
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
                solver_status=solver_status,
            )
        )
        if (
            self.config.objective is HistoricalSanosObjective.MEAN_VARIANCE
            and not certified
        ):
            try:
                compiled.problem.solve(
                    solver=cp.CLARABEL,
                    warm_start=False,
                    max_iter=self.config.maximum_iterations,
                    tol_gap_abs=self.config.certificate_tolerance,
                    tol_gap_rel=self.config.certificate_tolerance,
                    tol_feas=self.config.certificate_tolerance,
                )
                iterations += int(compiled.problem.solver_stats.num_iters or 0)
                fallback_status, fallback_trace, independent_iterations = (
                    self._resolve_inaccurate_conic_problem(compiled.problem)
                )
                iterations += independent_iterations
                solver_trace = f"{solver_trace}; fallback {fallback_trace}"
                (
                    fitted,
                    certified,
                    exact_objective,
                    objective_error,
                    constraint_violation,
                ) = self._certify_raw_convex_solution(
                    compiled,
                    initial,
                    indices,
                    signed_claim_quantity=signed_claim_quantity,
                    terminal_claim_values_pv=terminal_claim_values_pv,
                    solver_status=fallback_status,
                )
            except cp.error.SolverError as error:
                solver_trace = f"{solver_trace}; CLARABEL fallback failed: {error}"
        scaled_fitted = fitted / compiled.action_scale
        unregularized_objective = exact_objective - (
            0.5
            * self.config.quadratic_ridge
            * float(scaled_fitted @ scaled_fitted)
        )
        recession_direction = bool(
            self.config.objective is HistoricalSanosObjective.MEAN_VOLATILITY
            and np.all(claim == 0.0)
            and unregularized_objective < -self.config.certificate_tolerance
        )
        result = OptimizeResult(
            x=fitted,
            nit=iterations,
            success=certified and not recession_direction,
            optimality=math.inf,
            message=(
                f"{solver_trace}; exact-objective-error={objective_error:.6g}; "
                f"constraint-violation={constraint_violation:.6g}"
                + (
                    f"; recession_direction={unregularized_objective:.6g}"
                    if recession_direction
                    else ""
                )
            ),
            fun=exact_objective,
            objective_error=objective_error,
            constraint_violation=constraint_violation,
            recession_direction=recession_direction,
        )
        result.action_scale = compiled.action_scale
        return result

    def _certify_raw_convex_solution(
        self,
        compiled: _ConvexCostLassoProblem,
        initial: FloatArray,
        indices: NDArray[np.int64],
        *,
        signed_claim_quantity: float,
        terminal_claim_values_pv: FloatArray,
        solver_status: str | None,
    ) -> tuple[FloatArray, bool, float, float, float]:
        """Certify the exact point whose objective the convex solver reports."""
        scaled_fitted = (
            np.asarray(initial[: self.parameter_count], dtype=np.float64)
            / compiled.action_scale
            if compiled.scaled_positions.value is None
            else np.asarray(compiled.scaled_positions.value, dtype=np.float64)
        )
        fitted = compiled.action_scale * scaled_fitted
        constraint_violation = _maximum_constraint_violation(compiled.problem)
        certified, exact_objective, objective_error = self._certify_conic_result(
            fitted,
            indices,
            signed_claim_quantity=signed_claim_quantity,
            terminal_claim_values_pv=terminal_claim_values_pv,
            include_translation=False,
            action_scale=compiled.action_scale,
            solver_status=solver_status,
            solver_objective=compiled.problem.value,
            maximum_constraint_violation=constraint_violation,
        )
        return (
            fitted,
            certified,
            exact_objective,
            objective_error,
            constraint_violation,
        )

    def _convex_cost_lasso_problem(
        self,
        indices: NDArray[np.int64],
        *,
        instrument_gains: FloatArray,
    ) -> _ConvexCostLassoProblem:
        key = np.ascontiguousarray(indices).tobytes()
        cached = self._convex_problems.get(key)
        if cached is not None:
            return cached
        turnover_matrix, turnover_offset, turnover_weights = (
            self._turnover_geometry(indices)
        )
        gain_rms = np.sqrt(np.mean(instrument_gains**2, axis=0))
        action_scale = np.clip(
            1.0 / np.maximum(gain_rms, self.config.scale_floor),
            1.0e-6,
            1.0e6,
        )
        scaled_positions = cp.Variable(self.parameter_count)
        positions = cp.multiply(action_scale, scaled_positions)
        claim = cp.Parameter(len(indices))
        wealth_before_cost = claim + instrument_gains @ positions
        centered = wealth_before_cost - cp.sum(wealth_before_cost) / len(indices)
        cost = turnover_weights @ cp.abs(
            turnover_matrix @ positions - turnover_offset
        )
        if self.config.objective is HistoricalSanosObjective.MEAN_VARIANCE:
            risk_penalty = (
                0.5
                * self.config.risk_parameter
                * cp.sum_squares(centered)
                / len(indices)
            )
        else:
            risk_penalty = (
                self.config.risk_parameter
                * cp.norm(centered, 2)
                / math.sqrt(len(indices))
            )
        ridge = (
            0.5
            * self.config.quadratic_ridge
            * cp.sum_squares(scaled_positions)
        )
        problem = cp.Problem(
            cp.Minimize(
                -cp.sum(wealth_before_cost) / len(indices)
                + risk_penalty
                + cost
                + ridge
            ),
            [
                positions >= -self.config.solver_position_cap,
                positions <= self.config.solver_position_cap,
            ],
        )
        if not problem.is_dpp():
            raise RuntimeError("weekly convex policy problem is not parameter-cacheable")
        cached = _ConvexCostLassoProblem(
            scaled_positions=scaled_positions,
            claim=claim,
            action_scale=np.asarray(action_scale, dtype=np.float64),
            problem=problem,
        )
        self._convex_problems[key] = cached
        return cached

    def _objective_value(
        self,
        parameters: NDArray[np.floating],
        indices: NDArray[np.integer],
        *,
        signed_claim_quantity: float,
        terminal_claim_values_pv: FloatArray,
        include_translation: bool,
        action_scale: FloatArray | None = None,
    ) -> float:
        if self.config.objective not in (
            HistoricalSanosObjective.MEAN_VARIANCE,
            HistoricalSanosObjective.MEAN_VOLATILITY,
            HistoricalSanosObjective.ENTROPY,
            HistoricalSanosObjective.POWER_EXPONENTIAL,
        ):
            return self._loss_and_gradient(
                parameters,
                indices,
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
                include_translation=include_translation,
            )[0]
        values = np.asarray(parameters, dtype=np.float64)
        index_values = np.asarray(indices, dtype=np.int64)
        if action_scale is None:
            action_scale = self._action_scale(index_values)
        if self.config.objective is HistoricalSanosObjective.ENTROPY:
            positions = values[: self.parameter_count]
            wealth = self._exact_scenario_wealth(
                positions,
                indices=np.asarray(indices, dtype=np.int64),
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
            )
            risk = self.config.risk_parameter
            return float(
                (logsumexp(-risk * wealth) - math.log(len(wealth))) / risk
                + 0.5
                * self.config.quadratic_ridge
                * ((positions / action_scale) @ (positions / action_scale))
            )
        if self.config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL:
            positions = values[: self.parameter_count]
            wealth = self._exact_scenario_wealth(
                positions,
                indices=np.asarray(indices, dtype=np.int64),
                signed_claim_quantity=signed_claim_quantity,
                terminal_claim_values_pv=terminal_claim_values_pv,
            )
            translation = float(values[-1])
            residual = wealth - translation
            risk = self.config.risk_parameter
            utility = np.where(
                residual >= 0.0,
                -np.expm1(-risk * np.maximum(residual, 0.0)) / risk,
                residual - 0.5 * risk * residual**2,
            )
            return float(
                -(translation + np.mean(utility))
                + 0.5
                * self.config.quadratic_ridge
                * ((positions / action_scale) @ (positions / action_scale))
            )
        claim, instrument_gains = self._linear_wealth_geometry(
            np.asarray(indices, dtype=np.int64),
            signed_claim_quantity=signed_claim_quantity,
            terminal_claim_values_pv=terminal_claim_values_pv,
        )
        turnover_matrix, turnover_offset, turnover_weights = (
            self._turnover_geometry(np.asarray(indices, dtype=np.int64))
        )
        positions = values[: self.parameter_count]
        wealth = claim + instrument_gains @ positions
        deterministic_cost = float(
            turnover_weights
            @ np.abs(turnover_matrix @ positions - turnover_offset)
        )
        centered = wealth - np.mean(wealth)
        if self.config.objective is HistoricalSanosObjective.MEAN_VARIANCE:
            risk_penalty = (
                0.5
                * self.config.risk_parameter
                * float(np.mean(centered**2))
            )
        else:
            risk_penalty = self.config.risk_parameter * float(
                np.sqrt(np.mean(centered**2))
            )
        scaled_positions = positions / action_scale
        ridge = 0.5 * self.config.quadratic_ridge * float(
            scaled_positions @ scaled_positions
        )
        return -float(np.mean(wealth)) + risk_penalty + deterministic_cost + ridge

    def _certify_conic_result(
        self,
        parameters: NDArray[np.floating],
        indices: NDArray[np.int64],
        *,
        signed_claim_quantity: float,
        terminal_claim_values_pv: FloatArray,
        include_translation: bool,
        action_scale: FloatArray,
        solver_status: str | None,
        solver_objective: float | None,
        maximum_constraint_violation: float,
    ) -> tuple[bool, float, float]:
        """Verify an independently certified optimum and reconstruct its objective."""
        values = np.asarray(parameters, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            return False, math.inf, math.inf
        exact_objective = self._objective_value(
            values,
            indices,
            signed_claim_quantity=signed_claim_quantity,
            terminal_claim_values_pv=terminal_claim_values_pv,
            include_translation=include_translation,
            action_scale=action_scale,
        )
        if solver_objective is None or not math.isfinite(float(solver_objective)):
            return False, exact_objective, math.inf
        objective_error = abs(float(solver_objective) - exact_objective)
        objective_limit = self.config.objective_agreement_tolerance * (
            1.0 + abs(exact_objective)
        )
        accepted_status = solver_status in {
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        }
        accepted = bool(
            accepted_status
            and objective_error <= objective_limit
            and maximum_constraint_violation <= self.config.certificate_tolerance
        )
        if solver_status != cp.OPTIMAL or not accepted:
            print(
                "historical weekly solver certificate "
                f"status={solver_status} accepted={accepted} "
                f"objective_error={objective_error:.6g} "
                f"objective_limit={objective_limit:.6g} "
                f"constraint_violation={maximum_constraint_violation:.6g} "
                f"scenarios={len(indices)} parameters={self.parameter_count}",
                flush=True,
            )
        return (
            accepted,
            exact_objective,
            objective_error,
        )

    def _resolve_inaccurate_conic_problem(
        self,
        problem: cp.Problem,
    ) -> tuple[str | None, str, int]:
        """Resolve an inaccurate primary result with independent SCS."""
        primary_status = problem.status
        primary_solver = str(problem.solver_stats.solver_name or "primary")
        if primary_status != cp.OPTIMAL_INACCURATE:
            return primary_status, f"{primary_solver}={primary_status}", 0
        return self._solve_conic_with_scs(
            problem,
            primary_trace=f"{primary_solver}={primary_status}",
        )

    def _solve_conic_with_scs(
        self,
        problem: cp.Problem,
        *,
        primary_trace: str,
    ) -> tuple[str | None, str, int]:
        """Independently solve the unchanged conic objective with SCS."""
        try:
            problem.solve(
                solver=cp.SCS,
                warm_start=False,
                max_iters=max(50_000, self.config.maximum_iterations),
                eps=min(
                    self.config.gradient_tolerance,
                    0.1 * self.config.certificate_tolerance,
                ),
                normalize=True,
                verbose=False,
            )
        except cp.error.SolverError as error:
            return (
                None,
                f"{primary_trace}; SCS failed: {error}",
                0,
            )
        independent_status = problem.status
        iterations = int(problem.solver_stats.num_iters or 0)
        return (
            independent_status,
            f"{primary_trace}; SCS={independent_status}",
            iterations,
        )

    def _canonicalize_positions(
        self,
        positions: NDArray[np.floating],
        initial_positions: NDArray[np.floating],
        *,
        action_scale: FloatArray,
    ) -> FloatArray:
        """Canonicalize only in standardized coordinates."""
        fitted = np.clip(
            np.asarray(positions, dtype=np.float64),
            -self.config.solver_position_cap,
            self.config.solver_position_cap,
        )
        scaled = fitted / action_scale
        scaled[np.abs(scaled) <= self.config.certificate_tolerance] = 0.0
        scaled_initial = (
            np.asarray(initial_positions, dtype=np.float64) / action_scale
        )
        retain_initial = (
            np.abs(scaled - scaled_initial) <= self.config.certificate_tolerance
        )
        scaled[retain_initial] = scaled_initial[retain_initial]
        return np.clip(
            action_scale * scaled,
            -self.config.solver_position_cap,
            self.config.solver_position_cap,
        )

    def _action_scale(self, indices: NDArray[np.int64]) -> FloatArray:
        gain_rms = np.sqrt(np.mean(self._instrument_gains[indices] ** 2, axis=0))
        return np.clip(
            1.0 / np.maximum(gain_rms, self.config.scale_floor),
            1.0e-6,
            1.0e6,
        )
    def _linear_wealth_geometry(
        self,
        indices: NDArray[np.int64],
        *,
        signed_claim_quantity: float,
        terminal_claim_values_pv: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        claim = signed_claim_quantity * terminal_claim_values_pv[indices]
        return (
            np.asarray(claim, dtype=np.float64),
            self._instrument_gains[indices],
        )

    def _build_instrument_gains(self) -> FloatArray:
        scenarios = self.scenarios
        gains = np.zeros(
            (scenarios.scenario_count, self.parameter_count),
            dtype=np.float64,
        )
        for step in range(scenarios.step_count):
            gains[:, step] = scenarios.spot_step_gains[:, step]
            if self.center_instrument_gains:
                gains[:, step] -= self._spot_gain_means[step]
        for pair, parameter_index in self._option_parameter_index.items():
            step, contract = pair
            gains[:, parameter_index] = scenarios.option_step_gains[
                :,
                step,
                contract,
            ]
            if self.center_instrument_gains:
                gains[:, parameter_index] -= self._option_gain_means[step, contract]
        return np.ascontiguousarray(gains)

    def _build_scenario_turnover_geometry(
        self,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        scenarios = self.scenarios
        rows: list[FloatArray] = []
        offsets: list[float] = []
        weights: list[FloatArray] = []

        for step in range(scenarios.step_count):
            row = np.zeros(self.parameter_count, dtype=np.float64)
            row[step] = 1.0
            if step == 0:
                offset = self.initial_spot
            else:
                row[step - 1] = -1.0
                offset = 0.0
            rows.append(row)
            offsets.append(offset)
            weights.append(scenarios.spot_values_pv[:, step] * 1.0e-4)

            for contract in np.flatnonzero(scenarios.option_alive[step]):
                parameter_index = self._option_parameter_index[(step, int(contract))]
                row = np.zeros(self.parameter_count, dtype=np.float64)
                row[parameter_index] = 1.0
                if step == 0:
                    offset = float(self.initial_options[contract])
                else:
                    previous_index = self._option_parameter_index[
                        (step - 1, int(contract))
                    ]
                    row[previous_index] = -1.0
                    offset = 0.0
                rows.append(row)
                offsets.append(offset)
                weights.append(
                    scenarios.option_half_spreads_pv[:, step, contract]
                )

        terminal_spot = np.zeros(self.parameter_count, dtype=np.float64)
        terminal_spot[scenarios.step_count - 1] = 1.0
        rows.append(terminal_spot)
        offsets.append(0.0)
        weights.append(scenarios.spot_values_pv[:, -1] * 1.0e-4)
        for contract in np.flatnonzero(scenarios.option_alive[-1]):
            row = np.zeros(self.parameter_count, dtype=np.float64)
            row[
                self._option_parameter_index[
                    (scenarios.step_count - 1, int(contract))
                ]
            ] = 1.0
            rows.append(row)
            offsets.append(0.0)
            weights.append(scenarios.option_half_spreads_pv[:, -1, contract])
        return (
            np.stack(rows),
            np.asarray(offsets, dtype=np.float64),
            np.ascontiguousarray(np.column_stack(weights)),
        )

    def _turnover_geometry(
        self,
        indices: NDArray[np.int64],
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        return (
            self._turnover_matrix,
            self._turnover_offset,
            np.mean(self._scenario_turnover_weights[indices], axis=0),
        )

    def _loss_and_gradient(
        self,
        parameters: NDArray[np.floating],
        indices: NDArray[np.integer],
        *,
        signed_claim_quantity: float,
        terminal_claim_values_pv: FloatArray | None = None,
        include_translation: bool,
    ) -> tuple[float, FloatArray]:
        claim_values = self._claim_values(terminal_claim_values_pv)
        values = np.asarray(parameters, dtype=np.float64)
        wealth, jacobian = self._smooth_wealth_and_jacobian(
            values[: self.parameter_count],
            indices=np.asarray(indices, dtype=np.int64),
            signed_claim_quantity=signed_claim_quantity,
            terminal_claim_values_pv=claim_values,
        )
        sample_count = len(wealth)
        risk = self.config.risk_parameter
        if self.config.objective is HistoricalSanosObjective.MEAN_VARIANCE:
            centered = wealth - np.mean(wealth)
            loss = -np.mean(wealth) + 0.5 * risk * np.mean(centered**2)
            path_gradient = (-1.0 + risk * centered) / sample_count
        elif self.config.objective is HistoricalSanosObjective.MEAN_VOLATILITY:
            centered = wealth - np.mean(wealth)
            standard_deviation = math.sqrt(
                float(np.mean(centered**2)) + np.finfo(np.float64).eps
            )
            loss = -np.mean(wealth) + risk * standard_deviation
            path_gradient = (
                -1.0 + risk * centered / standard_deviation
            ) / sample_count
        elif self.config.objective is HistoricalSanosObjective.ENTROPY:
            exponent = -risk * wealth
            loss = (
                logsumexp(exponent) - math.log(sample_count)
            ) / risk
            path_gradient = -np.exp(exponent - logsumexp(exponent))
        elif self.config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL:
            if not include_translation or len(values) != self.parameter_count + 1:
                raise ValueError("weekly power-exponential policy requires translation")
            translation = float(values[-1])
            residual = wealth - translation
            utility = np.where(
                residual >= 0.0,
                -np.expm1(-risk * np.maximum(residual, 0.0)) / risk,
                residual - 0.5 * risk * residual**2,
            )
            marginal = np.where(
                residual >= 0.0,
                np.exp(-risk * np.maximum(residual, 0.0)),
                1.0 - risk * residual,
            )
            loss = -(translation + float(np.mean(utility)))
            path_gradient = -marginal / sample_count
        else:
            raise ValueError(
                f"unsupported historical weekly objective {self.config.objective!r}"
            )
        gradient = jacobian.T @ path_gradient
        if include_translation:
            translation_gradient = -1.0 - float(np.sum(path_gradient))
            gradient = np.concatenate((gradient, [translation_gradient]))
        if not math.isfinite(float(loss)) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("historical weekly objective is non-finite")
        return float(loss), np.asarray(gradient, dtype=np.float64)

    def _smooth_wealth_and_jacobian(
        self,
        positions: FloatArray,
        *,
        indices: NDArray[np.int64],
        signed_claim_quantity: float,
        terminal_claim_values_pv: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(positions, dtype=np.float64)
        if values.shape != (self.parameter_count,):
            raise ValueError("historical weekly parameter vector has wrong shape")
        epsilon = self.config.cost_smoothing
        gains = self._instrument_gains[indices]
        turnover = self._turnover_matrix @ values - self._turnover_offset
        root = np.sqrt(turnover**2 + epsilon**2)
        smooth_absolute = root - epsilon
        turnover_derivative = turnover / root
        weights = self._scenario_turnover_weights[indices]
        wealth = (
            signed_claim_quantity * terminal_claim_values_pv[indices]
            + gains @ values
            - weights @ smooth_absolute
        )
        jacobian = gains - (weights * turnover_derivative) @ self._turnover_matrix
        return wealth, jacobian

    def _exact_scenario_wealth(
        self,
        positions: FloatArray,
        *,
        indices: NDArray[np.int64],
        signed_claim_quantity: float,
        terminal_claim_values_pv: FloatArray,
    ) -> FloatArray:
        values = np.asarray(positions, dtype=np.float64)
        if values.shape != (self.parameter_count,):
            raise ValueError("historical weekly parameter vector has wrong shape")
        gains = self._instrument_gains[indices]
        turnover = self._turnover_matrix @ values - self._turnover_offset
        weights = self._scenario_turnover_weights[indices]
        return (
            signed_claim_quantity * terminal_claim_values_pv[indices]
            + gains @ values
            - weights @ np.abs(turnover)
        )

    def _claim_values(
        self,
        values: NDArray[np.floating] | None,
    ) -> FloatArray:
        claim_values = np.asarray(
            (
                self.scenarios.terminal_claim_values_pv
                if values is None
                else values
            ),
            dtype=np.float64,
        )
        if claim_values.shape != (self.scenarios.scenario_count,):
            raise ValueError("weekly terminal claim values do not align")
        if not np.all(np.isfinite(claim_values)) or np.any(claim_values < 0.0):
            raise ValueError("weekly terminal claim values must be finite non-negative")
        return claim_values

    def unpack(self, parameters: NDArray[np.floating]) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (self.parameter_count,):
            raise ValueError("historical weekly parameter vector has wrong shape")
        spot = values[: self.scenarios.step_count].copy()
        options = np.zeros(
            (self.scenarios.step_count, self.scenarios.contract_count),
            dtype=np.float64,
        )
        for index, pair in enumerate(self.option_parameter_pairs):
            options[pair] = values[self.scenarios.step_count + index]
        return spot, options


def _maximum_constraint_violation(problem: cp.Problem) -> float:
    """Return the largest finite primal constraint violation."""
    maximum = 0.0
    for constraint in problem.constraints:
        try:
            violation = np.asarray(constraint.violation(), dtype=np.float64)
        except (ValueError, TypeError):
            return math.inf
        if not np.all(np.isfinite(violation)):
            return math.inf
        if violation.size:
            maximum = max(maximum, float(np.max(np.abs(violation))))
    return maximum


def _weekly_objective(objective: HistoricalSanosObjective) -> WeeklyRiskObjective:
    return WeeklyRiskObjective(objective.value)


def _weekly_certainty_equivalent(
    wealth: FloatArray,
    *,
    config: HistoricalSanosConfig,
) -> float:
    return weekly_certainty_equivalent(
        wealth,
        objective=_weekly_objective(config.objective),
        risk_parameter=config.risk_parameter,
    )
