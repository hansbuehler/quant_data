"""Empirical four-objective training on historical public SANOS returns."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, brentq, minimize

from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class HistoricalSanosPolicyFit:
    """One fitted static action and held-out training diagnostics."""

    action: FloatArray
    cash_translation: float | None
    training_loss: float
    validation_loss: float
    iterations: int
    selection_converged: bool
    final_converged: bool
    converged: bool
    selection_maximum_absolute_gradient: float
    maximum_absolute_gradient: float
    selection_solver_status: str
    solver_status: str
    selection_recession_direction: bool
    recession_direction: bool
    selection_cap_hits: int
    cap_hits: int


@dataclass(frozen=True, kw_only=True)
class HistoricalSanosHedgeFit:
    """Claim, zero-claim, and differenced empirical policies."""

    action: FloatArray
    claim_policy: HistoricalSanosPolicyFit
    zero_policy: HistoricalSanosPolicyFit
    objective: float
    converged: bool
    internal_cap_hits: int
    executable_cap_hits: int
    cap_hits: int


def fit_historical_sanos_hedge(
    *,
    instrument_returns: NDArray[np.floating],
    claim_returns: NDArray[np.floating],
    linear_transaction_costs: NDArray[np.floating],
    config: HistoricalSanosConfig = HistoricalSanosConfig(),
) -> HistoricalSanosHedgeFit:
    """Fit claim and zero empirical policies and return their difference."""
    instruments, claim, costs = _validated_inputs(
        instrument_returns,
        claim_returns,
        linear_transaction_costs,
    )
    return fit_historical_sanos_hedges(
        instrument_returns=instruments,
        claim_returns=claim[:, None],
        linear_transaction_costs=costs,
        config=config,
    )[0]


def fit_historical_sanos_hedges(
    *,
    instrument_returns: NDArray[np.floating],
    claim_returns: NDArray[np.floating],
    linear_transaction_costs: NDArray[np.floating],
    config: HistoricalSanosConfig = HistoricalSanosConfig(),
) -> tuple[HistoricalSanosHedgeFit, ...]:
    """Fit a claim book while reusing its common zero-claim policy."""
    instruments = np.asarray(instrument_returns, dtype=np.float64)
    claims = np.asarray(claim_returns, dtype=np.float64)
    costs = np.asarray(linear_transaction_costs, dtype=np.float64)
    if claims.ndim != 2 or claims.shape[1] == 0:
        raise ValueError("claim_returns must be a non-empty matrix")
    _validated_inputs(instruments, claims[:, 0], costs)
    if claims.shape[0] != len(instruments) or not np.all(np.isfinite(claims)):
        raise ValueError("historical SANOS claim book must align and be finite")
    observation_parameter_ratio = len(instruments) / instruments.shape[1]
    if observation_parameter_ratio < config.thin_sample_ratio:
        warnings.warn(
            "historical daily policy is statistically thin: "
            f"observations={len(instruments)} parameters={instruments.shape[1]} "
            f"ratio={observation_parameter_ratio:.3f}; the fitted policy is an "
            "explicitly regularized estimate, not an unrestricted empirical optimum",
            RuntimeWarning,
            stacklevel=2,
        )
    zero_policy = _fit_policy(
        instruments,
        np.zeros(len(instruments), dtype=np.float64),
        costs,
        config=config,
    )
    return tuple(
        _compose_hedge_fit(
            instruments,
            claims[:, index],
            costs,
            claim_policy=_fit_policy(
                instruments,
                claims[:, index],
                costs,
                config=config,
            ),
            zero_policy=zero_policy,
            config=config,
        )
        for index in range(claims.shape[1])
    )


def _compose_hedge_fit(
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    claim_policy: HistoricalSanosPolicyFit,
    zero_policy: HistoricalSanosPolicyFit,
    config: HistoricalSanosConfig,
) -> HistoricalSanosHedgeFit:
    raw_action = claim_policy.action - zero_policy.action
    action = np.clip(raw_action, -config.position_cap, config.position_cap)
    executable_cap_hits = int(np.count_nonzero(action != raw_action))
    internal_cap_hits = claim_policy.cap_hits + zero_policy.cap_hits
    cap_hits = internal_cap_hits + executable_cap_hits
    include_translation = (
        config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL
    )
    translation = 0.0
    if include_translation:
        wealth = claim + instruments @ action - float(costs @ np.abs(action))
        translation = _optimal_power_cash_translation(
            wealth,
            risk_parameter=config.risk_parameter,
        )
    objective = _objective_and_gradient(
        np.concatenate(
            (
                action,
                np.asarray([translation]) if include_translation else np.empty(0),
            )
        ),
        instruments,
        claim,
        costs,
        config=config,
        include_translation=include_translation,
        action_scale=_action_scale(instruments, config=config),
    )[0]
    return HistoricalSanosHedgeFit(
        action=action,
        claim_policy=claim_policy,
        zero_policy=zero_policy,
        objective=objective,
        converged=claim_policy.converged and zero_policy.converged and cap_hits == 0,
        internal_cap_hits=internal_cap_hits,
        executable_cap_hits=executable_cap_hits,
        cap_hits=cap_hits,
    )


def _fit_policy(
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
) -> HistoricalSanosPolicyFit:
    validation_count = max(1, int(round(len(instruments) * config.validation_fraction)))
    split = len(instruments) - validation_count
    if split < 2:
        raise ValueError("historical SANOS training split has fewer than two rows")
    include_translation = config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL
    parameter_count = instruments.shape[1] + int(include_translation)
    initial = np.zeros(parameter_count, dtype=np.float64)
    selection_action_scale = _action_scale(instruments[:split], config=config)
    selection = _minimize_policy(
        initial,
        instruments[:split],
        claim[:split],
        costs,
        config=config,
        include_translation=include_translation,
        action_scale=selection_action_scale,
    )
    selection = _retry_exact_cost_policy(
        selection,
        instruments[:split],
        claim[:split],
        costs,
        config=config,
        include_translation=include_translation,
        action_scale=selection_action_scale,
    )
    validation_loss = _objective_and_gradient(
        selection.x,
        instruments[split:],
        claim[split:],
        costs,
        config=config,
        include_translation=include_translation,
        action_scale=selection_action_scale,
    )[0]
    final_action_scale = _action_scale(instruments, config=config)
    final = _minimize_policy(
        selection.x,
        instruments,
        claim,
        costs,
        config=config,
        include_translation=include_translation,
        action_scale=final_action_scale,
    )
    final = _retry_exact_cost_policy(
        final,
        instruments,
        claim,
        costs,
        config=config,
        include_translation=include_translation,
        action_scale=final_action_scale,
    )
    final_loss = _objective_and_gradient(
        final.x,
        instruments,
        claim,
        costs,
        config=config,
        include_translation=include_translation,
        action_scale=final_action_scale,
    )[0]
    action = np.asarray(final.x[: instruments.shape[1]], dtype=np.float64)
    selection_action = np.asarray(
        selection.x[: instruments.shape[1]],
        dtype=np.float64,
    )
    selection_cap_hits = int(
        np.count_nonzero(
            np.abs(selection_action) >= 0.999 * config.solver_position_cap
        )
    )
    cap_hits = int(
        np.count_nonzero(
            np.abs(action) >= 0.999 * config.solver_position_cap
        )
    )
    selection_optimality = float(getattr(selection, "optimality", math.inf))
    optimality = float(getattr(final, "optimality", math.inf))
    selection_recession_direction = bool(
        getattr(selection, "recession_direction", False)
    )
    recession_direction = bool(getattr(final, "recession_direction", False))
    selection_converged = bool(selection.success and selection_cap_hits == 0)
    # The split fit only supplies a warm start and held-out diagnostic. The
    # executable policy and its numerical gate come from the full-window refit.
    final_converged = bool(final.success and cap_hits == 0)
    return HistoricalSanosPolicyFit(
        action=action,
        cash_translation=float(final.x[-1]) if include_translation else None,
        training_loss=float(final_loss),
        validation_loss=float(validation_loss),
        iterations=int(selection.nit) + int(final.nit),
        selection_converged=selection_converged,
        final_converged=final_converged,
        converged=final_converged,
        selection_maximum_absolute_gradient=selection_optimality,
        maximum_absolute_gradient=optimality,
        selection_solver_status=str(selection.message),
        solver_status=str(final.message),
        selection_recession_direction=selection_recession_direction,
        recession_direction=recession_direction,
        selection_cap_hits=selection_cap_hits,
        cap_hits=cap_hits,
    )


def _minimize_policy(
    initial: FloatArray,
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
    include_translation: bool,
    action_scale: FloatArray,
) -> OptimizeResult:
    if config.objective is HistoricalSanosObjective.MEAN_VARIANCE:
        return _minimize_mean_variance(
            initial,
            instruments,
            claim,
            costs,
            config=config,
            action_scale=action_scale,
        )
    if config.objective is HistoricalSanosObjective.MEAN_VOLATILITY:
        return _minimize_mean_volatility(
            initial,
            instruments,
            claim,
            costs,
            config=config,
            action_scale=action_scale,
        )
    if config.objective in (
        HistoricalSanosObjective.ENTROPY,
        HistoricalSanosObjective.POWER_EXPONENTIAL,
    ):
        return _minimize_exact_cost_policy(
            initial,
            instruments,
            claim,
            costs,
            config=config,
            include_translation=include_translation,
            action_scale=action_scale,
        )
    raise ValueError(f"unsupported historical SANOS objective {config.objective!r}")


def _retry_exact_cost_policy(
    result: OptimizeResult,
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
    include_translation: bool,
    action_scale: FloatArray,
) -> OptimizeResult:
    """Restart a stalled split solve from its canonical signed action."""
    if result.success or config.objective not in (
        HistoricalSanosObjective.ENTROPY,
        HistoricalSanosObjective.POWER_EXPONENTIAL,
    ):
        return result
    retry = _minimize_policy(
        np.asarray(result.x, dtype=np.float64),
        instruments,
        claim,
        costs,
        config=config,
        include_translation=include_translation,
        action_scale=action_scale,
    )
    retry.nit = int(result.nit) + int(retry.nit)
    retry.message = f"canonical restart after {result.message}; {retry.message}"
    return retry


def _minimize_exact_cost_policy(
    initial: FloatArray,
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
    include_translation: bool,
    action_scale: FloatArray,
) -> OptimizeResult:
    """Solve a cash-additive OCE with exact linear spread costs."""
    instrument_count = instruments.shape[1]
    initial_action = np.asarray(initial[:instrument_count], dtype=np.float64)
    cost_free_translation = np.empty(0, dtype=np.float64)
    if include_translation:
        cost_free_translation = np.asarray(
            [float(initial[-1]) + float(costs @ np.abs(initial_action))],
            dtype=np.float64,
        )
    initial_scaled_action = initial_action / action_scale
    split_initial = np.concatenate(
        (
            np.maximum(initial_scaled_action, 0.0),
            np.maximum(-initial_scaled_action, 0.0),
            cost_free_translation,
        )
    )
    scaled_caps = config.solver_position_cap / action_scale
    split_bounds: list[tuple[float | None, float | None]] = [
        *((0.0, float(cap)) for cap in scaled_caps),
        *((0.0, float(cap)) for cap in scaled_caps),
    ]
    if include_translation:
        split_bounds.append((None, None))
    zero_costs = np.zeros_like(costs)

    def value_and_gradient(parameters: NDArray[np.floating]) -> tuple[float, FloatArray]:
        values = np.asarray(parameters, dtype=np.float64)
        positive = values[:instrument_count]
        negative = values[instrument_count : 2 * instrument_count]
        scaled_action = positive - negative
        action = action_scale * scaled_action
        risk_parameters = np.concatenate(
            (
                action,
                np.asarray([values[-1]]) if include_translation else np.empty(0),
            )
        )
        risk_value, risk_gradient = _objective_and_gradient(
            risk_parameters,
            instruments,
            claim,
            zero_costs,
            config=config,
            include_translation=include_translation,
            action_scale=action_scale,
        )
        action_gradient = action_scale * risk_gradient[:instrument_count]
        scaled_costs = action_scale * costs
        gradient = np.concatenate(
            (
                action_gradient + scaled_costs,
                -action_gradient + scaled_costs,
                np.asarray([risk_gradient[-1]])
                if include_translation
                else np.empty(0),
            )
        )
        return (
            float(risk_value + scaled_costs @ (positive + negative)),
            np.asarray(gradient, dtype=np.float64),
        )

    result = minimize(
        value_and_gradient,
        split_initial,
        method="L-BFGS-B",
        jac=True,
        bounds=split_bounds,
        options={
            "maxiter": config.maximum_iterations,
            "maxfun": 4 * config.maximum_iterations,
            "gtol": config.gradient_tolerance,
            "ftol": 0.0,
            "maxls": 50,
        },
    )
    split_solution = np.asarray(result.x, dtype=np.float64)
    action = action_scale * (
        split_solution[:instrument_count]
        - split_solution[instrument_count : 2 * instrument_count]
    )
    internal_parameters = np.concatenate(
        (
            action,
            np.asarray([split_solution[-1]])
            if include_translation
            else np.empty(0),
        )
    )
    risk_value, risk_gradient = _objective_and_gradient(
        internal_parameters,
        instruments,
        claim,
        zero_costs,
        config=config,
        include_translation=include_translation,
        action_scale=action_scale,
    )
    optimality = _maximum_kkt_violation(
        action,
        risk_gradient[:instrument_count],
        costs,
        position_cap=config.solver_position_cap,
        coordinate_scale=action_scale,
    )
    if include_translation:
        optimality = max(optimality, abs(float(risk_gradient[-1])))
    translation = np.empty(0, dtype=np.float64)
    if include_translation:
        translation = np.asarray(
            [split_solution[-1] - float(costs @ np.abs(action))],
            dtype=np.float64,
        )
    result.x = np.concatenate((action, translation))
    result.fun = float(risk_value + costs @ np.abs(action))
    result.optimality = float(optimality)
    optimizer_success = bool(result.success)
    result.success = bool(optimality <= config.certificate_tolerance)
    if not result.success:
        result.message = (
            f"{result.message}; KKT residual {optimality:.6g} exceeds "
            f"{config.certificate_tolerance:.6g}"
        )
    elif not optimizer_success:
        result.message = (
            f"{result.message}; accepted by independent KKT certificate "
            f"{optimality:.6g}"
        )
    return result


def _minimize_mean_volatility(
    initial: FloatArray,
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
    action_scale: FloatArray,
) -> OptimizeResult:
    """Solve the empirical mean-volatility cost-lasso as an exact SOCP."""
    observations, instrument_count = instruments.shape
    instrument_center = np.mean(instruments, axis=0)
    claim_center = float(np.mean(claim))
    centered_instruments = instruments - instrument_center
    centered_claim = claim - claim_center
    scaled_action = cp.Variable(instrument_count)
    action = cp.multiply(action_scale, scaled_action)
    expected_wealth = claim_center + instrument_center @ action - costs @ cp.abs(action)
    standard_deviation = cp.norm(
        centered_claim + centered_instruments @ action,
        2,
    ) / math.sqrt(observations)
    objective = cp.Minimize(
        -expected_wealth
        + config.risk_parameter * standard_deviation
        + 0.5 * config.quadratic_ridge * cp.sum_squares(scaled_action)
    )
    problem = cp.Problem(objective)
    scaled_action.value = (
        np.asarray(initial[:instrument_count], dtype=np.float64) / action_scale
    )
    primary_iterations = 0
    try:
        problem.solve(
            solver=cp.CLARABEL,
            warm_start=True,
            max_iter=config.maximum_iterations,
            tol_gap_abs=config.gradient_tolerance,
            tol_gap_rel=config.gradient_tolerance,
            tol_feas=config.gradient_tolerance,
        )
        if problem.status in {cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE}:
            problem = cp.Problem(
                objective,
                [
                    action >= -config.solver_position_cap,
                    action <= config.solver_position_cap,
                ],
            )
            scaled_action.value = (
                np.asarray(initial[:instrument_count], dtype=np.float64) / action_scale
            )
            problem.solve(
                solver=cp.CLARABEL,
                warm_start=True,
                max_iter=config.maximum_iterations,
                tol_gap_abs=config.gradient_tolerance,
                tol_gap_rel=config.gradient_tolerance,
                tol_feas=config.gradient_tolerance,
            )
    except cp.error.SolverError as error:
        solver_status, solver_trace, independent_iterations = (
            _solve_conic_with_scs(
                problem,
                config=config,
                primary_trace=f"CLARABEL failed: {error}",
            )
        )
    else:
        primary_iterations = int(problem.solver_stats.num_iters or 0)
        solver_status, solver_trace, independent_iterations = (
            _resolve_inaccurate_conic_problem(problem, config=config)
        )
    total_iterations = primary_iterations + independent_iterations

    def certify_action(
        candidate: FloatArray,
        *,
        canonicalization_error: float | None = None,
    ) -> OptimizeResult:
        candidate_scaled = candidate / action_scale
        feasibility = float(
            np.max(
                np.maximum(
                    np.abs(candidate) - config.solver_position_cap,
                    0.0,
                )
            )
        )
        residual = centered_claim + centered_instruments @ candidate
        residual_standard_deviation = math.sqrt(float(np.mean(residual**2)))
        zero_residual_tolerance = max(
            config.gradient_tolerance,
            np.finfo(np.float64).eps,
        ) * max(1.0, float(np.std(claim, ddof=0)))
        if residual_standard_deviation > zero_residual_tolerance:
            smooth_gradient = (
                -instrument_center
                + config.risk_parameter
                * (centered_instruments.T @ residual)
                / (observations * residual_standard_deviation)
                + config.quadratic_ridge * candidate / action_scale**2
            )
            optimality = _maximum_kkt_violation(
                candidate,
                smooth_gradient,
                costs,
                position_cap=config.solver_position_cap,
                coordinate_scale=action_scale,
            )
        else:
            # At zero residual volatility the Euclidean norm is nondifferentiable.
            # Require an independently accepted conic solver status.
            optimality = (
                0.0
                if solver_status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                else math.inf
            )
        exact_objective = _objective_and_gradient(
            candidate,
            instruments,
            claim,
            costs,
            config=config,
            include_translation=False,
            action_scale=action_scale,
        )[0]
        objective_error = (
            math.inf
            if problem.value is None or not math.isfinite(float(problem.value))
            else abs(float(problem.value) - exact_objective)
        )
        objective_limit = config.objective_agreement_tolerance * (
            1.0 + abs(exact_objective)
        )
        unregularized_objective = exact_objective - (
            0.5
            * config.quadratic_ridge
            * float(candidate_scaled @ candidate_scaled)
        )
        recession_direction = bool(
            np.all(claim == 0.0)
            and unregularized_objective < -config.certificate_tolerance
        )
        success = bool(
            solver_status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
            and np.all(np.isfinite(candidate))
            and feasibility <= config.certificate_tolerance
            and optimality <= config.certificate_tolerance
            and objective_error <= objective_limit
            and (
                canonicalization_error is None
                or canonicalization_error <= objective_limit
            )
            and not recession_direction
        )
        canonicalization_trace = (
            ""
            if canonicalization_error is None
            else f" canonicalization_error={canonicalization_error:.6g}"
        )
        return OptimizeResult(
            x=candidate,
            nit=total_iterations,
            success=success,
            optimality=optimality,
            message=(
                f"{solver_trace}; certificate kkt={optimality:.6g} "
                f"objective_error={objective_error:.6g} "
                f"constraint_violation={feasibility:.6g}"
                f"{canonicalization_trace}"
                + (
                    f"; recession_direction={unregularized_objective:.6g}"
                    if recession_direction
                    else ""
                )
            ),
            fun=exact_objective,
            recession_direction=recession_direction,
        )

    def certify_current_solution() -> OptimizeResult:
        raw_scaled_action = (
            np.asarray(initial[:instrument_count], dtype=np.float64) / action_scale
            if scaled_action.value is None
            else np.asarray(scaled_action.value, dtype=np.float64).copy()
        )
        raw_action = action_scale * raw_scaled_action
        raw_result = certify_action(raw_action)
        raw_feasible = bool(
            np.all(
                np.abs(raw_action)
                <= config.solver_position_cap + config.certificate_tolerance
            )
        )
        if raw_result.success or (
            raw_result.recession_direction and raw_feasible
        ):
            return raw_result
        canonical_scaled_action = raw_scaled_action.copy()
        canonical_scaled_action[
            np.abs(canonical_scaled_action) <= config.certificate_tolerance
        ] = 0.0
        canonical_action = np.clip(
            action_scale * canonical_scaled_action,
            -config.solver_position_cap,
            config.solver_position_cap,
        )
        if np.array_equal(canonical_action, raw_action):
            return raw_result
        canonical_objective = _objective_and_gradient(
            canonical_action,
            instruments,
            claim,
            costs,
            config=config,
            include_translation=False,
            action_scale=action_scale,
        )[0]
        return certify_action(
            canonical_action,
            canonicalization_error=abs(
                canonical_objective - float(raw_result.fun)
            ),
        )

    conic_result = certify_current_solution()
    if (
        not conic_result.success
        and not conic_result.recession_direction
        and "SCS=" not in solver_trace
        and "SCS failed:" not in solver_trace
    ):
        solver_status, solver_trace, fallback_iterations = _solve_conic_with_scs(
            problem,
            config=config,
            primary_trace=solver_trace,
        )
        total_iterations += fallback_iterations
        conic_result = certify_current_solution()
    if conic_result.success or conic_result.recession_direction:
        return conic_result
    refined = _minimize_exact_cost_policy(
        np.asarray(conic_result.x, dtype=np.float64),
        instruments,
        claim,
        costs,
        config=config,
        include_translation=False,
        action_scale=action_scale,
    )
    refined.nit = int(conic_result.nit) + int(refined.nit)
    refined.message = f"{conic_result.message}; exact refinement {refined.message}"
    refined.recession_direction = False
    return refined


def _resolve_inaccurate_conic_problem(
    problem: cp.Problem,
    *,
    config: HistoricalSanosConfig,
) -> tuple[str | None, str, int]:
    """Resolve an inaccurate CLARABEL status with independent SCS."""
    primary_status = problem.status
    primary_solver = str(problem.solver_stats.solver_name or "CLARABEL")
    if primary_status != cp.OPTIMAL_INACCURATE:
        return primary_status, f"{primary_solver}={primary_status}", 0
    return _solve_conic_with_scs(
        problem,
        config=config,
        primary_trace=f"{primary_solver}={primary_status}",
    )


def _solve_conic_with_scs(
    problem: cp.Problem,
    *,
    config: HistoricalSanosConfig,
    primary_trace: str,
) -> tuple[str | None, str, int]:
    """Independently solve the unchanged conic objective with SCS."""
    try:
        problem.solve(
            solver=cp.SCS,
            warm_start=False,
            max_iters=max(50_000, config.maximum_iterations),
            eps=min(
                config.gradient_tolerance,
                0.1 * config.certificate_tolerance,
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


def _minimize_mean_variance(
    initial: FloatArray,
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
    action_scale: FloatArray,
) -> OptimizeResult:
    """Solve the quadratic spread-cost problem and certify original actions."""
    observations, instrument_count = instruments.shape
    centered_instruments = instruments - np.mean(instruments, axis=0)
    centered_claim = claim - np.mean(claim)
    instrument_center = np.mean(instruments, axis=0)
    scaled_action = cp.Variable(instrument_count)
    action = cp.multiply(action_scale, scaled_action)
    residual = centered_claim + centered_instruments @ action
    objective = cp.Minimize(
        -float(np.mean(claim))
        - instrument_center @ action
        + 0.5
        * config.risk_parameter
        * cp.sum_squares(residual)
        / observations
        + costs @ cp.abs(action)
        + 0.5 * config.quadratic_ridge * cp.sum_squares(scaled_action)
    )
    problem = cp.Problem(
        objective,
        [
            action >= -config.solver_position_cap,
            action <= config.solver_position_cap,
        ],
    )
    scaled_action.value = (
        np.asarray(initial[:instrument_count], dtype=np.float64) / action_scale
    )
    iterations = 0
    traces: list[str] = []
    last_result: OptimizeResult | None = None
    for solver in (cp.OSQP, cp.CLARABEL):
        try:
            if solver == cp.OSQP:
                problem.solve(
                    solver=solver,
                    warm_start=False,
                    max_iter=max(10_000, config.maximum_iterations),
                    eps_abs=config.gradient_tolerance,
                    eps_rel=config.gradient_tolerance,
                    polishing=True,
                )
            else:
                problem.solve(
                    solver=solver,
                    warm_start=False,
                    max_iter=config.maximum_iterations,
                    tol_gap_abs=config.certificate_tolerance,
                    tol_gap_rel=config.certificate_tolerance,
                    tol_feas=config.certificate_tolerance,
                )
        except cp.error.SolverError as error:
            traces.append(f"{solver} failed: {error}")
            continue
        iterations += int(problem.solver_stats.num_iters or 0)
        solver_name = str(problem.solver_stats.solver_name or solver)
        solver_status = problem.status
        traces.append(f"{solver_name}={solver_status}")
        raw_scaled_action = (
            np.asarray(initial[:instrument_count], dtype=np.float64) / action_scale
            if scaled_action.value is None
            else np.asarray(scaled_action.value, dtype=np.float64).copy()
        )
        raw_action = action_scale * raw_scaled_action

        def certify_action(
            candidate: FloatArray,
            *,
            canonicalization_error: float | None = None,
        ) -> OptimizeResult:
            feasibility = float(
                np.max(
                    np.maximum(
                        np.abs(candidate) - config.solver_position_cap,
                        0.0,
                    )
                )
            )
            residual_value = centered_claim + centered_instruments @ candidate
            smooth_gradient = (
                -instrument_center
                + config.risk_parameter
                * (centered_instruments.T @ residual_value)
                / observations
                + config.quadratic_ridge * candidate / action_scale**2
            )
            optimality = _maximum_kkt_violation(
                candidate,
                smooth_gradient,
                costs,
                position_cap=config.solver_position_cap,
                coordinate_scale=action_scale,
            )
            exact_objective = _objective_and_gradient(
                candidate,
                instruments,
                claim,
                costs,
                config=config,
                include_translation=False,
                action_scale=action_scale,
            )[0]
            objective_error = (
                math.inf
                if problem.value is None or not math.isfinite(float(problem.value))
                else abs(float(problem.value) - exact_objective)
            )
            objective_limit = config.objective_agreement_tolerance * (
                1.0 + abs(exact_objective)
            )
            success = bool(
                solver_status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                and feasibility <= config.certificate_tolerance
                and optimality <= config.certificate_tolerance
                and objective_error <= objective_limit
                and (
                    canonicalization_error is None
                    or canonicalization_error <= objective_limit
                )
            )
            canonicalization_trace = (
                ""
                if canonicalization_error is None
                else f" canonicalization_error={canonicalization_error:.6g}"
            )
            return OptimizeResult(
                x=candidate,
                nit=iterations,
                success=success,
                optimality=optimality,
                message=(
                    "; ".join(traces)
                    + f"; certificate kkt={optimality:.6g} "
                    f"objective_error={objective_error:.6g} "
                    f"constraint_violation={feasibility:.6g}"
                    f"{canonicalization_trace}"
                ),
                fun=exact_objective,
            )

        raw_result = certify_action(raw_action)
        if raw_result.success:
            return raw_result
        canonical_scaled_action = raw_scaled_action.copy()
        canonical_scaled_action[
            np.abs(canonical_scaled_action) <= config.certificate_tolerance
        ] = 0.0
        canonical_action = np.clip(
            action_scale * canonical_scaled_action,
            -config.solver_position_cap,
            config.solver_position_cap,
        )
        if not np.array_equal(canonical_action, raw_action):
            canonical_objective = _objective_and_gradient(
                canonical_action,
                instruments,
                claim,
                costs,
                config=config,
                include_translation=False,
                action_scale=action_scale,
            )[0]
            canonicalization_error = abs(
                canonical_objective - float(raw_result.fun)
            )
            canonical_result = certify_action(
                canonical_action,
                canonicalization_error=canonicalization_error,
            )
            if canonical_result.success:
                return canonical_result
            last_result = canonical_result
        else:
            last_result = raw_result
    if last_result is not None:
        refined_action, refinement_sweeps, refinement_optimality = (
            _refine_mean_variance_coordinate_descent(
                np.asarray(last_result.x, dtype=np.float64),
                instruments,
                claim,
                costs,
                config=config,
                action_scale=action_scale,
            )
        )
        refined_result = certify_action(refined_action)
        refined_result.nit = iterations + refinement_sweeps
        refined_result.message = (
            f"{refined_result.message}; coordinate_refinement_sweeps="
            f"{refinement_sweeps} coordinate_refinement_kkt="
            f"{refinement_optimality:.6g}"
        )
        if refined_result.success:
            return refined_result
        last_result = refined_result
        _, separator, diagnostics = str(last_result.message).partition(
            "; certificate"
        )
        last_result.message = "; ".join(traces) + (
            f"; certificate{diagnostics}" if separator else ""
        )
        return last_result
    return OptimizeResult(
        x=np.asarray(initial[:instrument_count], dtype=np.float64),
        nit=iterations,
        success=False,
        optimality=math.inf,
        message="; ".join(traces) or "no mean-variance solver completed",
    )


def _refine_mean_variance_coordinate_descent(
    initial_action: FloatArray,
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
    action_scale: FloatArray,
) -> tuple[FloatArray, int, float]:
    """Refine the exact box-constrained lasso QP in scaled coordinates."""
    observations, instrument_count = instruments.shape
    centered_instruments = instruments - np.mean(instruments, axis=0)
    centered_claim = claim - np.mean(claim)
    scaled_instruments = centered_instruments * action_scale[None, :]
    hessian = (
        config.risk_parameter
        * (scaled_instruments.T @ scaled_instruments)
        / observations
        + config.quadratic_ridge * np.eye(instrument_count)
    )
    linear = (
        -action_scale * np.mean(instruments, axis=0)
        + config.risk_parameter
        * (scaled_instruments.T @ centered_claim)
        / observations
    )
    penalty = costs * action_scale
    bound = config.solver_position_cap / action_scale
    scaled_action = np.clip(initial_action / action_scale, -bound, bound)
    gradient = hessian @ scaled_action + linear
    maximum_violation = math.inf
    for sweep in range(1, config.maximum_iterations + 1):
        for index in range(instrument_count):
            diagonal = hessian[index, index]
            partial = gradient[index] - diagonal * scaled_action[index]
            if partial < -penalty[index]:
                replacement = (-partial - penalty[index]) / diagonal
            elif partial > penalty[index]:
                replacement = (-partial + penalty[index]) / diagonal
            else:
                replacement = 0.0
            replacement = float(np.clip(replacement, -bound[index], bound[index]))
            change = replacement - scaled_action[index]
            if change != 0.0:
                scaled_action[index] = replacement
                gradient += hessian[:, index] * change
        derivative = gradient + penalty * np.sign(scaled_action)
        zero = scaled_action == 0.0
        derivative[zero] = np.maximum(
            np.abs(gradient[zero]) - penalty[zero],
            0.0,
        )
        upper = scaled_action >= bound
        lower = scaled_action <= -bound
        derivative[upper] = np.maximum(derivative[upper], 0.0)
        derivative[lower] = np.minimum(derivative[lower], 0.0)
        maximum_violation = float(np.max(np.abs(derivative)))
        if maximum_violation <= config.certificate_tolerance:
            break
    return (
        np.asarray(action_scale * scaled_action, dtype=np.float64),
        sweep,
        maximum_violation,
    )


def _maximum_kkt_violation(
    action: FloatArray,
    smooth_gradient: FloatArray,
    costs: FloatArray,
    *,
    position_cap: float,
    coordinate_scale: FloatArray | None = None,
) -> float:
    derivative = smooth_gradient + costs * np.sign(action)
    zero = action == 0.0
    derivative[zero] = np.maximum(np.abs(smooth_gradient[zero]) - costs[zero], 0.0)
    upper = action >= position_cap
    lower = action <= -position_cap
    derivative[upper] = np.maximum(derivative[upper], 0.0)
    derivative[lower] = np.minimum(derivative[lower], 0.0)
    if coordinate_scale is not None:
        derivative *= coordinate_scale
    return float(np.max(np.abs(derivative)))


def _objective_and_gradient(
    parameters: NDArray[np.floating],
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
    include_translation: bool,
    action_scale: FloatArray | None = None,
) -> tuple[float, FloatArray]:
    values = np.asarray(parameters, dtype=np.float64)
    instrument_count = instruments.shape[1]
    action = values[:instrument_count]
    if action_scale is None:
        action_scale = _action_scale(instruments, config=config)
    cost = float(costs @ np.abs(action))
    cost_gradient = costs * np.sign(action)
    wealth = claim + instruments @ action - cost
    jacobian = instruments - cost_gradient[None, :]
    sample_count = len(wealth)
    risk = config.risk_parameter
    if config.objective is HistoricalSanosObjective.MEAN_VARIANCE:
        centered = wealth - np.mean(wealth)
        loss = -np.mean(wealth) + 0.5 * risk * np.mean(centered**2)
        path_gradient = (-1.0 + risk * centered) / sample_count
    elif config.objective is HistoricalSanosObjective.MEAN_VOLATILITY:
        centered = wealth - np.mean(wealth)
        standard_deviation = math.sqrt(float(np.mean(centered**2)))
        loss = -np.mean(wealth) + risk * standard_deviation
        if standard_deviation > 0.0:
            path_gradient = (
                -1.0 + risk * centered / standard_deviation
            ) / sample_count
        else:
            # Zero is an admissible subgradient of the Euclidean norm at its
            # origin; unlike an epsilon floor, it leaves the exact objective
            # unchanged for the independent conic certificate.
            path_gradient = np.full(sample_count, -1.0 / sample_count)
    elif config.objective is HistoricalSanosObjective.ENTROPY:
        exponent = -risk * wealth
        maximum = float(np.max(exponent))
        weights = np.exp(exponent - maximum)
        weights /= np.sum(weights)
        loss = (maximum + math.log(float(np.mean(np.exp(exponent - maximum))))) / risk
        path_gradient = -weights
    elif config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL:
        if not include_translation or len(values) != instrument_count + 1:
            raise ValueError("power-exponential objective requires cash translation")
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
        raise ValueError(f"unsupported historical SANOS objective {config.objective!r}")
    gradient = jacobian.T @ path_gradient
    scaled_action = action / action_scale
    loss += 0.5 * config.quadratic_ridge * float(scaled_action @ scaled_action)
    gradient += config.quadratic_ridge * action / action_scale**2
    if include_translation:
        translation_gradient = -1.0 - float(np.sum(path_gradient))
        gradient = np.concatenate((gradient, np.asarray([translation_gradient])))
    if not math.isfinite(float(loss)) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("historical SANOS objective is non-finite")
    return float(loss), np.asarray(gradient, dtype=np.float64)


def _action_scale(
    instruments: FloatArray,
    *,
    config: HistoricalSanosConfig,
) -> FloatArray:
    """Return unit-RMS action coordinates with bounded numerical scaling."""
    gain_rms = np.sqrt(np.mean(np.asarray(instruments, dtype=np.float64) ** 2, axis=0))
    return np.clip(
        1.0 / np.maximum(gain_rms, config.scale_floor),
        1.0e-6,
        1.0e6,
    )


def _optimal_power_cash_translation(
    wealth: FloatArray,
    *,
    risk_parameter: float,
) -> float:
    width = 50.0 / risk_parameter
    lower = float(np.min(wealth) - width)
    upper = float(np.max(wealth) + width)

    def first_order_condition(translation: float) -> float:
        residual = wealth - translation
        marginal = np.where(
            residual >= 0.0,
            np.exp(-risk_parameter * np.maximum(residual, 0.0)),
            1.0 - risk_parameter * residual,
        )
        return 1.0 - float(np.mean(marginal))

    return float(brentq(first_order_condition, lower, upper))


def _validated_inputs(
    instrument_returns: NDArray[np.floating],
    claim_returns: NDArray[np.floating],
    linear_transaction_costs: NDArray[np.floating],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    instruments = np.asarray(instrument_returns, dtype=np.float64)
    claim = np.asarray(claim_returns, dtype=np.float64)
    costs = np.asarray(linear_transaction_costs, dtype=np.float64)
    if instruments.ndim != 2 or min(instruments.shape) == 0:
        raise ValueError("instrument_returns must be a non-empty matrix")
    if claim.shape != (len(instruments),) or costs.shape != (instruments.shape[1],):
        raise ValueError("historical SANOS arrays do not align")
    if not np.all(np.isfinite(instruments)) or not np.all(np.isfinite(claim)):
        raise ValueError("historical SANOS returns must be finite")
    if np.any(~np.isfinite(costs) | (costs < 0.0)):
        raise ValueError("historical SANOS costs must be finite and non-negative")
    return instruments, claim, costs
