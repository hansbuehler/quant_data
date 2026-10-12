"""Rolling robust covariance and spread-weighted cost-lasso action."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from quantdata.hedging.objectives import (
    CovarianceEstimate,
    LinearHedge,
    cost_lasso_minimum_variance_hedge,
    regularized_tyler_covariance,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.daily import OneDayMarketSample
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.config import (
    CovarianceHedgeConfig,
    CovarianceObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.returns import (
    HistoricalReturnPanel,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class CovarianceAction:
    """Capped action and underlying unconstrained fit."""

    spot: float
    option: float
    cap_hits: int
    fit: LinearHedge


@dataclass(frozen=True, kw_only=True)
class RobustPolicyFit:
    """One direct robust policy before claim-minus-zero differencing."""

    action: FloatArray
    objective: float
    expected_profit: float
    standard_deviation: float
    iterations: int
    converged: bool
    cap_hits: int
    bounded: bool
    recession_ratio: float


@dataclass(frozen=True, kw_only=True)
class CovarianceBookAction:
    """One pure hedge action and its claim/zero-policy diagnostics."""

    action: FloatArray | None
    claim_policy: RobustPolicyFit
    zero_policy: RobustPolicyFit
    objective: float
    expected_profit: float
    standard_deviation: float
    cap_hits: int
    converged: bool
    bounded: bool


@dataclass(frozen=True, kw_only=True)
class CovarianceBookFit:
    """All target policies from one target-catalog-invariant robust fit."""

    actions: tuple[CovarianceBookAction, ...]
    estimate: CovarianceEstimate
    instrument_count: int
    target_count: int
    mean_z_threshold: float
    mean_hac_lags: int
    retained_mean_directions: int
    maximum_absolute_mean_t: float
    raw_instrument_center_norm: float
    screened_instrument_center_norm: float


def fit_covariance_action(
    training_samples: tuple[OneDayMarketSample, ...],
    *,
    current_option_half_spread: float,
    config: CovarianceHedgeConfig = CovarianceHedgeConfig(),
) -> CovarianceAction:
    """Fit the independent rolling hedge without a future-bearing input.

    The one-day spot position pays one basis point when opened and one basis
    point when liquidated. The expiring option is cash settled and pays only
    its opening half-spread.
    """
    if len(training_samples) != config.rolling_observations:
        raise ValueError(
            f"expected {config.rolling_observations} training samples, "
            f"found {len(training_samples)}"
        )
    instrument_returns = np.asarray(
        [[sample.spot_return, sample.option_return] for sample in training_samples],
        dtype=np.float64,
    )
    claim_returns = np.asarray(
        [sample.claim_return for sample in training_samples], dtype=np.float64
    )
    if not np.isfinite(current_option_half_spread) or current_option_half_spread < 0.0:
        raise ValueError("current_option_half_spread must be finite and non-negative")
    linear_transaction_costs = np.asarray(
        [2.0e-4, current_option_half_spread], dtype=np.float64
    )
    fit = cost_lasso_minimum_variance_hedge(
        instrument_returns=instrument_returns,
        claim_returns=claim_returns,
        linear_transaction_costs=linear_transaction_costs,
        cost_multiplier=config.cost_multiplier,
    )
    capped = np.clip(fit.action, -config.position_cap, config.position_cap)
    return CovarianceAction(
        spot=float(capped[0]),
        option=float(capped[1]),
        cap_hits=int(np.sum(capped != fit.action)),
        fit=fit,
    )


def fit_covariance_book_actions(
    returns: HistoricalReturnPanel,
    *,
    linear_transaction_costs: NDArray[np.floating],
    config: CovarianceHedgeConfig = CovarianceHedgeConfig(),
) -> CovarianceBookFit:
    """Fit pure-hedge actions without dependence on the target catalog.

    Tyler scatter is fitted once to instruments only. Each claim receives a
    separate robust cross-scatter estimate using fixed instrument-derived row
    weights. Expected returns are sample means, not Tyler's robust marginal
    center, and the instrument mean is soft-screened using Newey--West errors
    of the observed covariance-eigenportfolio returns.
    """
    instruments = np.asarray(returns.instrument_returns, dtype=np.float64)
    targets = np.asarray(returns.target_returns, dtype=np.float64)
    if len(instruments) != config.rolling_observations:
        raise ValueError(
            f"expected {config.rolling_observations} return rows, found "
            f"{len(instruments)}"
        )
    costs = np.asarray(linear_transaction_costs, dtype=np.float64)
    if costs.shape != (instruments.shape[1],) or np.any(
        ~np.isfinite(costs) | (costs < 0.0)
    ):
        raise ValueError("transaction costs must align with the instrument panel")

    estimate = regularized_tyler_covariance(instruments)
    instrument_count = instruments.shape[1]
    raw_instrument_center = np.mean(instruments, axis=0)
    covariance = estimate.covariance
    (
        instrument_center,
        mean_z_threshold,
        mean_hac_lags,
        retained_mean_directions,
        maximum_absolute_mean_t,
    ) = _screen_instrument_mean(
            instruments,
            covariance=covariance,
            configured_threshold=config.mean_z_threshold,
            configured_hac_lags=config.mean_hac_lags,
        )
    penalty = config.cost_multiplier * costs
    recession_ratio = 0.0
    if config.objective is CovarianceObjective.MEAN_VOLATILITY:
        recession_ratio = _mean_volatility_recession_ratio(
            instrument_center=instrument_center,
            covariance=covariance,
            penalty=penalty,
        )
    bounded = (
        config.objective is CovarianceObjective.MEAN_VARIANCE
        or recession_ratio <= config.risk_parameter + config.recession_tolerance
    )
    zero_policy = _fit_policy(
        instrument_center=instrument_center,
        covariance=covariance,
        claim_center=0.0,
        claim_cross_covariance=np.zeros(instrument_count, dtype=np.float64),
        claim_variance=0.0,
        penalty=penalty,
        config=config,
        bounded=bounded,
        recession_ratio=recession_ratio,
    )
    actions: list[CovarianceBookAction] = []
    for target_index in range(targets.shape[1]):
        target = targets[:, target_index]
        claim_center = float(np.mean(target))
        claim_cross_covariance, claim_variance = _robust_target_scatter(
            instruments,
            target,
            estimate=estimate,
        )
        claim_policy = _fit_policy(
            instrument_center=instrument_center,
            covariance=covariance,
            claim_center=claim_center,
            claim_cross_covariance=claim_cross_covariance,
            claim_variance=claim_variance,
            penalty=penalty,
            config=config,
            bounded=bounded,
            recession_ratio=recession_ratio,
        )
        if not bounded:
            actions.append(
                CovarianceBookAction(
                    action=None,
                    claim_policy=claim_policy,
                    zero_policy=zero_policy,
                    objective=claim_policy.objective,
                    expected_profit=claim_center,
                    standard_deviation=math.sqrt(claim_variance),
                    cap_hits=0,
                    converged=False,
                    bounded=False,
                )
            )
            continue
        raw_action = claim_policy.action - zero_policy.action
        action = np.clip(raw_action, -config.position_cap, config.position_cap)
        final_cap_hits = int(np.count_nonzero(action != raw_action))
        expected_profit, standard_deviation, objective = _policy_statistics(
            action=action,
            instrument_center=instrument_center,
            covariance=covariance,
            claim_center=claim_center,
            claim_cross_covariance=claim_cross_covariance,
            claim_variance=claim_variance,
            penalty=penalty,
            config=config,
        )
        cap_hits = claim_policy.cap_hits + zero_policy.cap_hits + final_cap_hits
        actions.append(
            CovarianceBookAction(
                action=action,
                claim_policy=claim_policy,
                zero_policy=zero_policy,
                objective=objective,
                expected_profit=expected_profit,
                standard_deviation=standard_deviation,
                cap_hits=cap_hits,
                converged=(
                    claim_policy.converged
                    and zero_policy.converged
                    and cap_hits == 0
                    and final_cap_hits == 0
                ),
                bounded=True,
            )
        )
    return CovarianceBookFit(
        actions=tuple(actions),
        estimate=estimate,
        instrument_count=instrument_count,
        target_count=targets.shape[1],
        mean_z_threshold=mean_z_threshold,
        mean_hac_lags=mean_hac_lags,
        retained_mean_directions=retained_mean_directions,
        maximum_absolute_mean_t=maximum_absolute_mean_t,
        raw_instrument_center_norm=float(np.linalg.norm(raw_instrument_center)),
        screened_instrument_center_norm=float(np.linalg.norm(instrument_center)),
    )


def _screen_instrument_mean(
    instrument_returns: FloatArray,
    *,
    covariance: FloatArray,
    configured_threshold: float | None,
    configured_hac_lags: int | None,
) -> tuple[FloatArray, float, int, int, float]:
    """Soft-threshold sample means using eigenportfolio Newey--West errors."""
    observations, dimension = instrument_returns.shape
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, np.finfo(np.float64).tiny)
    del eigenvalues
    eigenportfolio_returns = instrument_returns @ eigenvectors
    projected_center = np.mean(eigenportfolio_returns, axis=0)
    hac_lags = (
        int(math.floor(4.0 * (observations / 100.0) ** (2.0 / 9.0)))
        if configured_hac_lags is None
        else configured_hac_lags
    )
    hac_lags = min(hac_lags, observations - 1)
    standard_errors = _newey_west_mean_standard_errors(
        eigenportfolio_returns,
        lags=hac_lags,
    )
    z_threshold = (
        math.sqrt(2.0 * math.log(max(dimension, 2)))
        if configured_threshold is None
        else configured_threshold
    )
    screened_projection = np.sign(projected_center) * np.maximum(
        np.abs(projected_center) - z_threshold * standard_errors,
        0.0,
    )
    return (
        eigenvectors @ screened_projection,
        z_threshold,
        hac_lags,
        int(np.count_nonzero(screened_projection)),
        float(
            np.max(
                np.abs(projected_center)
                / np.maximum(standard_errors, np.finfo(np.float64).tiny)
            )
        ),
    )


def _newey_west_mean_standard_errors(
    returns: FloatArray,
    *,
    lags: int,
) -> FloatArray:
    """Return Bartlett-kernel HAC standard errors of column sample means."""
    observations = len(returns)
    if not 0 <= lags < observations:
        raise ValueError("Newey--West lags must lie in [0, observations)")
    centered = returns - np.mean(returns, axis=0, keepdims=True)
    long_run_variance = np.sum(centered * centered, axis=0) / observations
    for lag in range(1, lags + 1):
        autocovariance = np.sum(
            centered[lag:] * centered[:-lag],
            axis=0,
        ) / observations
        long_run_variance += 2.0 * (1.0 - lag / (lags + 1.0)) * autocovariance
    long_run_variance = np.maximum(long_run_variance, 0.0)
    return np.sqrt(long_run_variance / observations)


def _robust_target_scatter(
    instruments: FloatArray,
    target: FloatArray,
    *,
    estimate: CovarianceEstimate,
) -> tuple[FloatArray, float]:
    """Estimate one target block using fixed instrument-derived Tyler weights."""
    centered_instruments = instruments - estimate.center
    centered_target = target - np.median(target)
    solved_instruments = np.linalg.solve(
        estimate.covariance,
        centered_instruments.T,
    ).T
    radial = np.sum(centered_instruments * solved_instruments, axis=1)
    positive_radial = radial[radial > np.finfo(np.float64).eps]
    if len(positive_radial) < 2:
        raise FloatingPointError("target scatter has too few nonzero instrument rows")
    radial_floor = max(
        float(np.quantile(positive_radial, 0.1)),
        np.finfo(np.float64).eps * max(instruments.shape[1], 1),
    )
    row_weights = np.zeros_like(radial)
    nonzero = radial > np.finfo(np.float64).eps
    row_weights[nonzero] = 1.0 / np.maximum(radial[nonzero], radial_floor)
    weighted_instruments = row_weights[:, None] * centered_instruments
    regression_gram = centered_instruments.T @ weighted_instruments
    average_diagonal = max(
        float(np.trace(regression_gram)) / instruments.shape[1],
        np.finfo(np.float64).eps,
    )
    ridge_scale = average_diagonal * max(
        estimate.shrinkage / max(1.0 - estimate.shrinkage, 1.0e-12),
        1.0e-8,
    )
    regression = np.linalg.solve(
        regression_gram + ridge_scale * np.eye(instruments.shape[1]),
        centered_instruments.T @ (row_weights * centered_target),
    )
    cross_covariance = estimate.covariance @ regression
    residual = centered_target - centered_instruments @ regression
    residual -= np.median(residual)
    robust_residual_scale = 1.4826 * float(np.median(np.abs(residual)))
    target_scale = 1.4826 * float(np.median(np.abs(centered_target)))
    residual_variance = max(
        robust_residual_scale**2,
        (1.0e-6 * target_scale) ** 2,
        np.finfo(np.float64).eps**2,
    )
    explained_variance = float(cross_covariance @ regression)
    target_variance = explained_variance + residual_variance
    return cross_covariance, target_variance


def _mean_volatility_recession_ratio(
    *,
    instrument_center: FloatArray,
    covariance: FloatArray,
    penalty: FloatArray,
) -> float:
    """Return the exact recession ratio after the linear spread penalty.

    Boundedness is equivalent to ``mu`` belonging to the sum of the spread
    box ``[-penalty, penalty]`` and the covariance ellipsoid of radius
    ``kappa``. The returned value is the covariance-metric distance from
    ``mu`` to that box.
    """
    inverse_covariance = np.linalg.inv(covariance)
    if np.all(penalty == 0.0):
        return float(
            math.sqrt(
                max(instrument_center @ inverse_covariance @ instrument_center, 0.0)
            )
        )

    def objective(box_point: FloatArray) -> float:
        residual = instrument_center - box_point
        return float(residual @ inverse_covariance @ residual)

    def gradient(box_point: FloatArray) -> FloatArray:
        return 2.0 * inverse_covariance @ (box_point - instrument_center)

    initial = np.clip(instrument_center, -penalty, penalty)
    result = minimize(
        objective,
        initial,
        jac=gradient,
        bounds=tuple((-float(cost), float(cost)) for cost in penalty),
        method="L-BFGS-B",
        options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 10_000},
    )
    if not result.success:
        raise RuntimeError(f"mean-volatility recession check failed: {result.message}")
    return float(math.sqrt(max(result.fun, 0.0)))


def _fit_policy(
    *,
    instrument_center: FloatArray,
    covariance: FloatArray,
    claim_center: float,
    claim_cross_covariance: FloatArray,
    claim_variance: float,
    penalty: FloatArray,
    config: CovarianceHedgeConfig,
    bounded: bool,
    recession_ratio: float,
) -> RobustPolicyFit:
    if not bounded:
        action = np.zeros_like(instrument_center)
        iterations = 0
        converged = False
        cap_hits = 0
    elif config.objective is CovarianceObjective.MEAN_VARIANCE:
        action, iterations, converged, cap_hits = _solve_quadratic_lasso(
            instrument_center=instrument_center,
            covariance=covariance,
            claim_cross_covariance=claim_cross_covariance,
            penalty=penalty,
            variance_aversion=config.risk_parameter,
            initial_action=np.zeros_like(instrument_center),
            config=config,
        )
    else:
        action, iterations, converged, cap_hits = _solve_mean_volatility(
            instrument_center=instrument_center,
            covariance=covariance,
            claim_cross_covariance=claim_cross_covariance,
            claim_variance=claim_variance,
            penalty=penalty,
            config=config,
        )
    expected_profit, standard_deviation, objective = _policy_statistics(
        action=action,
        instrument_center=instrument_center,
        covariance=covariance,
        claim_center=claim_center,
        claim_cross_covariance=claim_cross_covariance,
        claim_variance=claim_variance,
        penalty=penalty,
        config=config,
    )
    return RobustPolicyFit(
        action=action,
        objective=objective,
        expected_profit=expected_profit,
        standard_deviation=standard_deviation,
        iterations=iterations,
        converged=converged,
        cap_hits=cap_hits,
        bounded=bounded,
        recession_ratio=recession_ratio,
    )


def _solve_mean_volatility(
    *,
    instrument_center: FloatArray,
    covariance: FloatArray,
    claim_cross_covariance: FloatArray,
    claim_variance: float,
    penalty: FloatArray,
    config: CovarianceHedgeConfig,
) -> tuple[FloatArray, int, bool, int]:
    action = np.zeros_like(instrument_center)
    initial_standard_deviation = math.sqrt(max(claim_variance, 0.0))
    variance_aversion = config.risk_parameter / max(
        initial_standard_deviation,
        config.minimum_standard_deviation,
    )
    total_iterations = 0
    converged = False
    cap_hits = 0
    for _ in range(config.maximum_reweighting_iterations):
        updated, iterations, inner_converged, inner_cap_hits = _solve_quadratic_lasso(
            instrument_center=instrument_center,
            covariance=covariance,
            claim_cross_covariance=claim_cross_covariance,
            penalty=penalty,
            variance_aversion=variance_aversion,
            initial_action=action,
            config=config,
        )
        total_iterations += iterations
        cap_hits = max(cap_hits, inner_cap_hits)
        variance = _residual_variance(
            action=updated,
            covariance=covariance,
            claim_cross_covariance=claim_cross_covariance,
            claim_variance=claim_variance,
        )
        updated_aversion = config.risk_parameter / max(
            math.sqrt(variance),
            config.minimum_standard_deviation,
        )
        action_change = float(np.max(np.abs(updated - action)))
        aversion_change = abs(updated_aversion - variance_aversion) / max(
            1.0,
            abs(variance_aversion),
        )
        action = updated
        variance_aversion = updated_aversion
        if (
            inner_converged
            and action_change <= config.action_tolerance
            and aversion_change <= config.action_tolerance
        ):
            converged = True
            break
    return action, total_iterations, converged, cap_hits


def _solve_quadratic_lasso(
    *,
    instrument_center: FloatArray,
    covariance: FloatArray,
    claim_cross_covariance: FloatArray,
    penalty: FloatArray,
    variance_aversion: float,
    initial_action: FloatArray,
    config: CovarianceHedgeConfig,
) -> tuple[FloatArray, int, bool, int]:
    quadratic = variance_aversion * covariance.copy()
    quadratic.flat[:: len(quadratic) + 1] += 1.0e-12
    linear = variance_aversion * claim_cross_covariance - instrument_center
    action = np.asarray(initial_action, dtype=np.float64).copy()
    converged = False
    iterations = 0
    for iterations in range(1, config.maximum_solver_iterations + 1):
        previous = action.copy()
        for coordinate in range(len(action)):
            residual = -linear[coordinate] - (
                quadratic[coordinate] @ action
                - quadratic[coordinate, coordinate] * action[coordinate]
            )
            unconstrained = _soft_threshold(
                residual,
                penalty[coordinate],
            ) / quadratic[coordinate, coordinate]
            action[coordinate] = np.clip(
                unconstrained,
                -config.position_cap,
                config.position_cap,
            )
        if np.max(np.abs(action - previous)) <= config.action_tolerance:
            converged = True
            break
    cap_hits = int(
        np.count_nonzero(
            np.isclose(np.abs(action), config.position_cap, rtol=0.0, atol=1.0e-10)
        )
    )
    return action, iterations, converged, cap_hits


def _policy_statistics(
    *,
    action: FloatArray,
    instrument_center: FloatArray,
    covariance: FloatArray,
    claim_center: float,
    claim_cross_covariance: FloatArray,
    claim_variance: float,
    penalty: FloatArray,
    config: CovarianceHedgeConfig,
) -> tuple[float, float, float]:
    expected_profit = float(claim_center + instrument_center @ action)
    variance = _residual_variance(
        action=action,
        covariance=covariance,
        claim_cross_covariance=claim_cross_covariance,
        claim_variance=claim_variance,
    )
    standard_deviation = math.sqrt(variance)
    risk = (
        0.5 * config.risk_parameter * variance
        if config.objective is CovarianceObjective.MEAN_VARIANCE
        else config.risk_parameter * standard_deviation
    )
    objective = float(-expected_profit + risk + penalty @ np.abs(action))
    return expected_profit, standard_deviation, objective


def _residual_variance(
    *,
    action: FloatArray,
    covariance: FloatArray,
    claim_cross_covariance: FloatArray,
    claim_variance: float,
) -> float:
    variance = float(
        claim_variance
        + 2.0 * claim_cross_covariance @ action
        + action @ covariance @ action
    )
    if variance < -1.0e-10:
        raise FloatingPointError(f"negative residual variance {variance}")
    return max(variance, 0.0)


def _soft_threshold(value: float, threshold: float) -> float:
    return math.copysign(max(abs(value) - threshold, 0.0), value)
