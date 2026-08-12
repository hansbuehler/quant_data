"""One-day portfolio audit from fitted historical DYSANOS states.

At each historical observation, this module resets spot to one, simulates
independent one-day continuations from the fitted DYSANOS state, and solves a
maximum-mean portfolio problem.  The frozen portfolio is then
marked from the current observed market surface to the next observed market
surface.  The next observation is used only after the portfolio has been
fitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from cdxcore.bs import bs
from cdxcore.err import verify_inp

from quantdata.biz_calendar import DAYS_PER_YEAR

from .conditional_static_arbitrage import weighted_mean_standard_error
from .multi_expiry_arbitrage import observation_numerical_band


@dataclass(frozen=True, kw_only=True)
class HistoricalStateAuditConfig:
    """Numerical settings for the historical-state one-day audit."""

    training_paths: int = 1_024
    validation_paths: int = 4_096
    cost_rate: float = 0.0
    seed: int = 731_991
    decoder_block_size: int = 8_192

    def __post_init__(self) -> None:
        verify_inp(
            self.training_paths > 1,
            lambda: f"'training_paths' must exceed one, got {self.training_paths}",
        )
        verify_inp(
            self.validation_paths > 1,
            lambda: (
                "'validation_paths' must exceed one, got "
                f"{self.validation_paths}"
            ),
        )
        verify_inp(
            self.cost_rate in {0.0, 1.0e-4, 1.0e-3},
            lambda: f"unsupported transaction-cost rate {self.cost_rate}",
        )
        verify_inp(
            self.decoder_block_size > 0,
            lambda: (
                "'decoder_block_size' must be positive, got "
                f"{self.decoder_block_size}"
            ),
        )


def _strike_interpolated_call(
    *,
    extended_strikes: np.ndarray,
    inner_calls: np.ndarray,
    query_strikes: np.ndarray,
) -> np.ndarray:
    """Interpolate calls in strike with the SANOS boundary prices included."""

    extended_strikes = np.asarray(extended_strikes, dtype=np.float64)
    inner_calls = np.asarray(inner_calls, dtype=np.float64)
    query_strikes = np.asarray(query_strikes, dtype=np.float64)
    verify_inp(
        extended_strikes.ndim == 1
        and inner_calls.shape == (extended_strikes.size - 2,)
        and np.all(np.diff(extended_strikes) > 0.0),
        "invalid observed strike grid",
    )
    boundary_calls = np.concatenate(
        (
            np.asarray([1.0 - extended_strikes[0]]),
            inner_calls,
            np.asarray([0.0]),
        )
    )
    result = np.interp(
        query_strikes,
        extended_strikes,
        boundary_calls,
        left=np.nan,
        right=np.nan,
    )
    below = query_strikes < extended_strikes[0]
    above = query_strikes > extended_strikes[-1]
    result = np.where(below, np.maximum(1.0 - query_strikes, 0.0), result)
    result = np.where(above, 0.0, result)
    if not np.all(np.isfinite(result)) or np.any(result < -1.0e-12):
        raise FloatingPointError("observed strike interpolation is invalid")
    return np.maximum(result, 0.0)


def observed_surface_calls(
    *,
    grid_expiries: np.ndarray,
    extended_strikes: np.ndarray,
    inner_calls: np.ndarray,
    query_expiries: np.ndarray,
    query_strikes: np.ndarray,
) -> np.ndarray:
    """Mark normalized calls on one observed surface.

    Strike interpolation uses the observed normalized call prices and their
    SANOS boundary values.  Maturity interpolation is linear in normalized
    call price.  Between time zero and the first grid expiry it interpolates
    from intrinsic value.  The experiment queries only maturities no later
    than the last grid expiry.
    """

    grid_expiries = np.asarray(grid_expiries, dtype=np.float64)
    extended_strikes = np.asarray(extended_strikes, dtype=np.float64)
    inner_calls = np.asarray(inner_calls, dtype=np.float64)
    query_expiries = np.asarray(query_expiries, dtype=np.float64)
    query_strikes = np.asarray(query_strikes, dtype=np.float64)
    verify_inp(
        grid_expiries.ndim == 1
        and extended_strikes.ndim == 2
        and inner_calls.ndim == 2
        and extended_strikes.shape[0] == inner_calls.shape[0]
        == grid_expiries.size
        and extended_strikes.shape[1] == inner_calls.shape[1] + 2,
        "observed surface arrays have incompatible shapes",
    )
    verify_inp(
        query_strikes.ndim == 2
        and query_expiries.shape == (query_strikes.shape[0],)
        and query_strikes.shape[0] == grid_expiries.size,
        "query surface arrays have incompatible shapes",
    )
    verify_inp(
        np.all(np.diff(grid_expiries) > 0.0)
        and np.all(query_expiries >= 0.0)
        and float(np.max(query_expiries)) <= grid_expiries[-1] + 1.0e-12,
        "query maturities lie outside the observed surface",
    )

    result = np.empty_like(query_strikes, dtype=np.float64)
    for row, (expiry, strikes) in enumerate(
        zip(query_expiries, query_strikes, strict=True)
    ):
        upper = int(np.searchsorted(grid_expiries, expiry, side="left"))
        if upper == 0:
            high = _strike_interpolated_call(
                extended_strikes=extended_strikes[0],
                inner_calls=inner_calls[0],
                query_strikes=strikes,
            )
            alpha = expiry / grid_expiries[0]
            result[row] = (
                alpha * high
                + (1.0 - alpha) * np.maximum(1.0 - strikes, 0.0)
            )
            continue
        if upper == grid_expiries.size:
            upper -= 1
        if abs(expiry - grid_expiries[upper]) <= 1.0e-14:
            result[row] = _strike_interpolated_call(
                extended_strikes=extended_strikes[upper],
                inner_calls=inner_calls[upper],
                query_strikes=strikes,
            )
            continue
        lower = upper - 1
        low = _strike_interpolated_call(
            extended_strikes=extended_strikes[lower],
            inner_calls=inner_calls[lower],
            query_strikes=strikes,
        )
        high = _strike_interpolated_call(
            extended_strikes=extended_strikes[upper],
            inner_calls=inner_calls[upper],
            query_strikes=strikes,
        )
        alpha = (
            (expiry - grid_expiries[lower])
            / (grid_expiries[upper] - grid_expiries[lower])
        )
        result[row] = (1.0 - alpha) * low + alpha * high

    if not np.all(np.isfinite(result)) or np.any(result < -1.0e-12):
        raise FloatingPointError("observed surface marking produced invalid calls")
    return np.maximum(result, 0.0)


def execute_frozen_action(
    action: np.ndarray,
    *,
    spot_ratio: float,
    entry_calls: np.ndarray,
    exit_calls: np.ndarray,
    cost_rate: float,
) -> dict[str, float]:
    """Execute one spot-plus-options action on an observed one-day move."""

    action = np.asarray(action, dtype=np.float64).reshape(-1)
    entry_calls = np.asarray(entry_calls, dtype=np.float64)
    exit_calls = np.asarray(exit_calls, dtype=np.float64)
    verify_inp(
        action.shape == (entry_calls.size + 1,)
        and exit_calls.shape == entry_calls.shape,
        "frozen action and observed calls have incompatible shapes",
    )
    verify_inp(
        math.isfinite(spot_ratio)
        and spot_ratio > 0.0
        and np.all(np.isfinite(entry_calls))
        and np.all(np.isfinite(exit_calls)),
        "observed execution contains invalid prices",
    )
    returns = np.concatenate(
        (
            np.asarray([spot_ratio - 1.0]),
            (spot_ratio * exit_calls - entry_calls).reshape(-1),
        )
    )
    costs = cost_rate * np.concatenate(
        (np.asarray([1.0]), entry_calls.reshape(-1))
    )
    gross = float(returns @ action)
    transaction_cost = float(costs @ np.abs(action))
    net = gross - transaction_cost
    band = float(observation_numerical_band(action, returns[None], costs)[0])
    return {
        "gross_pnl": gross,
        "transaction_cost": transaction_cost,
        "net_pnl": net,
        "numerical_band": band,
        "net_pnl_in_bps_of_l1_budget": 10_000.0 * net,
    }


def summarize_fit_errors(
    *,
    fitted_calls: np.ndarray,
    market_calls: np.ndarray,
    relative_strikes: np.ndarray,
    expiries: np.ndarray,
    market_vols: np.ndarray,
    weights: np.ndarray,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Return per-state and aggregate DYSANOS surface-fit errors."""

    fitted_calls = np.asarray(fitted_calls, dtype=np.float64)
    market_calls = np.asarray(market_calls, dtype=np.float64)
    relative_strikes = np.asarray(relative_strikes, dtype=np.float64)
    expiries = np.asarray(expiries, dtype=np.float64)
    market_vols = np.asarray(market_vols, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    verify_inp(
        fitted_calls.shape
        == market_calls.shape
        == relative_strikes.shape
        == market_vols.shape
        == weights.shape
        and expiries.shape == fitted_calls.shape[:2],
        "surface-fit arrays have incompatible shapes",
    )
    fitted_vols = bs.implied(
        k=relative_strikes,
        prices=fitted_calls,
        sqrtT=np.sqrt(expiries)[:, :, None],
        eps=1.0e-7,
    )
    price_error = fitted_calls - market_calls
    vol_error = fitted_vols - market_vols
    if not np.all(np.isfinite(price_error)) or not np.all(np.isfinite(vol_error)):
        raise FloatingPointError("DYSANOS fit diagnostics contain NaN or infinity")
    positive_weight = np.maximum(weights, 0.0)
    weight_sum = positive_weight.sum(axis=(1, 2))
    verify_inp(np.all(weight_sum > 0.0), "surface-fit weights have zero mass")
    weighted_price_rmse = np.sqrt(
        np.sum(positive_weight * np.square(price_error), axis=(1, 2))
        / weight_sum
    )
    per_state = []
    for index in range(fitted_calls.shape[0]):
        per_state.append(
            {
                "normalized_price_mae": float(np.mean(np.abs(price_error[index]))),
                "normalized_price_rmse": float(
                    np.sqrt(np.mean(np.square(price_error[index])))
                ),
                "normalized_price_max_abs": float(
                    np.max(np.abs(price_error[index]))
                ),
                "weighted_normalized_price_rmse": float(
                    weighted_price_rmse[index]
                ),
                "implied_vol_mae_bps": float(
                    10_000.0 * np.mean(np.abs(vol_error[index]))
                ),
                "implied_vol_rmse_bps": float(
                    10_000.0 * np.sqrt(np.mean(np.square(vol_error[index])))
                ),
                "implied_vol_max_abs_bps": float(
                    10_000.0 * np.max(np.abs(vol_error[index]))
                ),
            }
        )

    def distribution(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "p10": float(np.quantile(values, 0.10)),
            "median": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "maximum": float(np.max(values)),
        }

    aggregate = {
        name: distribution(np.asarray([row[name] for row in per_state]))
        for name in per_state[0]
    }
    return per_state, aggregate


def _pnl_summary(
    *,
    action: np.ndarray,
    scenario_return: np.ndarray,
    costs: np.ndarray,
) -> dict[str, Any]:
    """Summarize a frozen action on one conditional scenario sample."""

    action = np.asarray(action, dtype=np.float64)
    scenario_return = np.asarray(scenario_return, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    pnl = scenario_return @ action - costs @ np.abs(action)
    band = observation_numerical_band(action, scenario_return, costs)
    statistics = weighted_mean_standard_error(pnl)
    mean = float(statistics["mean"])
    standard_error = float(statistics["standard_error"])
    loss = pnl < -band
    gain = pnl > band
    return {
        "paths": int(pnl.size),
        "mean_pnl": mean,
        "mean_pnl_standard_error": standard_error,
        "mean_lower_three_standard_errors": mean - 3.0 * standard_error,
        "minimum_pnl": float(np.min(pnl)),
        "minimum_pnl_plus_band": float(np.min(pnl + band)),
        "loss_count_beyond_float32_band": int(np.count_nonzero(loss)),
        "loss_frequency_beyond_float32_band": float(np.mean(loss)),
        "gain_count_beyond_float32_band": int(np.count_nonzero(gain)),
        "pnl_p01": float(np.quantile(pnl, 0.01)),
        "pnl_p10": float(np.quantile(pnl, 0.10)),
        "pnl_median": float(np.quantile(pnl, 0.50)),
        "pnl_p90": float(np.quantile(pnl, 0.90)),
        "pnl_p99": float(np.quantile(pnl, 0.99)),
    }


def solve_optimal_expected_return(
    scenario_return: np.ndarray,
    costs: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    r"""Solve the expected-P&L problem under a global unit L1 budget.

    For mean return ``mu`` and symmetric linear costs ``c``, the problem is

    ``max_a mu @ a - c @ abs(a)`` subject to ``sum(abs(a)) <= 1``.

    Its exact solution puts the entire budget into the signed coordinate with
    the largest positive score ``abs(mu_j) - c_j``.  If every score is
    non-positive, the zero action is optimal.
    """

    scenario_return = np.asarray(scenario_return, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    verify_inp(
        scenario_return.ndim == 2
        and scenario_return.shape[0] > 1
        and scenario_return.shape[1] == costs.size,
        "expected-return optimization arrays have incompatible shapes",
    )
    if not np.all(np.isfinite(scenario_return)) or not np.all(np.isfinite(costs)):
        raise FloatingPointError("expected-return optimization contains NaN/infinity")
    verify_inp(np.all(costs >= 0.0), "transaction costs must be nonnegative")
    mean_return = np.mean(scenario_return, axis=0)
    score = np.abs(mean_return) - costs
    instrument = int(np.argmax(score))
    action = np.zeros(costs.size, dtype=np.float64)
    if score[instrument] > 0.0:
        action[instrument] = math.copysign(1.0, mean_return[instrument])
    return action, {
        "selected_instrument": instrument,
        "selected_side": (
            "long"
            if action[instrument] > 0.0
            else "short"
            if action[instrument] < 0.0
            else "none"
        ),
        "selected_mean_return_before_cost": float(mean_return[instrument]),
        "selected_cost": float(costs[instrument]),
        "selected_expected_net_pnl": float(max(score[instrument], 0.0)),
        "runner_up_expected_net_pnl": float(
            max(np.partition(score, -2)[-2], 0.0) if score.size > 1 else 0.0
        ),
    }


def project_l1_ball(values: np.ndarray, *, radius: float = 1.0) -> np.ndarray:
    """Return the Euclidean projection onto an L1 ball."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    verify_inp(
        math.isfinite(radius) and radius > 0.0,
        lambda: f"L1 radius must be finite-positive, got {radius}",
    )
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("cannot project NaN or infinity onto an L1 ball")
    if float(np.abs(values).sum()) <= radius:
        return values.copy()
    absolute = np.sort(np.abs(values))[::-1]
    cumulative = np.cumsum(absolute)
    indices = np.arange(1, values.size + 1, dtype=np.float64)
    active = np.flatnonzero(absolute - (cumulative - radius) / indices > 0.0)
    verify_inp(active.size > 0, "L1 projection failed to find an active set")
    last = int(active[-1])
    threshold = (cumulative[last] - radius) / float(last + 1)
    projected = np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)
    if not np.all(np.isfinite(projected)):
        raise FloatingPointError("L1 projection produced NaN or infinity")
    return projected


def solve_mean_variance_portfolio(
    scenario_return: np.ndarray,
    *,
    risk_aversion: float,
    radius: float = 1.0,
    maximum_iterations: int = 5_000,
    tolerance: float = 2.0e-7,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    r"""Solve a conditional mean-variance portfolio on an L1 ball.

    The maximized training objective is

    ``mu @ a - 0.5 * risk_aversion * a @ covariance @ a``.

    Accelerated projected gradient uses the exact Euclidean projection onto
    the L1 ball.  The returned projected-gradient residual provides a direct
    first-order convergence diagnostic.
    """

    scenario_return = np.asarray(scenario_return, dtype=np.float64)
    verify_inp(
        scenario_return.ndim == 2 and scenario_return.shape[0] > 1,
        "mean-variance returns must have shape (paths > 1, instruments)",
    )
    verify_inp(
        math.isfinite(risk_aversion) and risk_aversion > 0.0,
        lambda: f"risk aversion must be finite-positive, got {risk_aversion}",
    )
    verify_inp(
        maximum_iterations > 0 and math.isfinite(tolerance) and tolerance > 0.0,
        "invalid mean-variance convergence settings",
    )
    if not np.all(np.isfinite(scenario_return)):
        raise FloatingPointError("mean-variance returns contain NaN/infinity")
    mean = np.mean(scenario_return, axis=0)
    centered = scenario_return - mean[None]
    covariance = centered.T @ centered / float(scenario_return.shape[0])
    covariance = 0.5 * (covariance + covariance.T)
    maximum_eigenvalue = float(np.linalg.eigvalsh(covariance)[-1])
    if maximum_eigenvalue <= np.finfo(np.float64).eps:
        action, _ = solve_optimal_expected_return(
            scenario_return,
            np.zeros(scenario_return.shape[1], dtype=np.float64),
        )
        return action, {
            "risk_aversion": risk_aversion,
            "iterations": 0,
            "converged": True,
            "projected_gradient_residual": 0.0,
            "training_mean_pnl": float(mean @ action),
            "training_variance": 0.0,
            "training_objective": float(mean @ action),
            "l1_norm": float(np.abs(action).sum()),
        }

    lipschitz = risk_aversion * maximum_eigenvalue
    action = np.zeros(mean.size, dtype=np.float64)
    accelerated = action.copy()
    momentum = 1.0
    converged = False
    residual = math.inf
    iteration = 0
    for iteration in range(1, maximum_iterations + 1):
        gradient = risk_aversion * (covariance @ accelerated) - mean
        next_action = project_l1_ball(
            accelerated - gradient / lipschitz,
            radius=radius,
        )
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum**2))
        accelerated = next_action + (
            (momentum - 1.0) / next_momentum
        ) * (next_action - action)
        action = next_action
        momentum = next_momentum
        if iteration == 1 or iteration % 20 == 0:
            action_gradient = risk_aversion * (covariance @ action) - mean
            projected = project_l1_ball(
                action - action_gradient / lipschitz,
                radius=radius,
            )
            residual = float(np.max(np.abs(projected - action)))
            if residual <= tolerance:
                converged = True
                break
    training_mean = float(mean @ action)
    training_variance = float(action @ covariance @ action)
    objective = training_mean - 0.5 * risk_aversion * training_variance
    if not np.all(np.isfinite(action)) or not all(
        math.isfinite(value)
        for value in (training_mean, training_variance, objective, residual)
    ):
        raise FloatingPointError("mean-variance optimizer produced NaN/infinity")
    verify_inp(
        float(np.abs(action).sum()) <= radius + 1.0e-9,
        "mean-variance optimizer violated its L1 budget",
    )
    return action, {
        "risk_aversion": risk_aversion,
        "iterations": iteration,
        "converged": converged,
        "projected_gradient_residual": residual,
        "training_mean_pnl": training_mean,
        "training_variance": training_variance,
        "training_objective": objective,
        "l1_norm": float(np.abs(action).sum()),
    }


def _top_positions(
    action: np.ndarray,
    *,
    relative_strikes: np.ndarray,
    expiries: np.ndarray,
    count: int = 12,
) -> list[dict[str, Any]]:
    """Describe the largest absolute positions of a flattened action."""

    action = np.asarray(action, dtype=np.float64)
    relative_strikes = np.asarray(relative_strikes, dtype=np.float64)
    expiries = np.asarray(expiries, dtype=np.float64)
    positions: list[dict[str, Any]] = []
    for instrument in np.argsort(-np.abs(action))[:count]:
        notional = float(action[instrument])
        if abs(notional) <= 1.0e-12:
            continue
        if instrument == 0:
            positions.append(
                {
                    "kind": "spot",
                    "notional": notional,
                }
            )
            continue
        expiry_index, strike_index = divmod(
            int(instrument - 1), relative_strikes.shape[1]
        )
        positions.append(
            {
                "kind": "call",
                "expiry_index": expiry_index,
                "expiry_days": float(expiries[expiry_index] * DAYS_PER_YEAR),
                "strike_index": strike_index,
                "relative_strike": float(
                    relative_strikes[expiry_index, strike_index]
                ),
                "notional": notional,
            }
        )
    return positions


def _decode_current_surface(model, *, state: np.ndarray, expiries: np.ndarray):
    """Decode one model state on the model's traded strike grid."""

    import torch

    torch_state = torch.as_tensor(
        state[None], dtype=model.env.fdtype, device=model.env.device
    )
    torch_expiries = torch.as_tensor(
        expiries[None], dtype=model.env.fdtype, device=model.env.device
    )
    with torch.no_grad():
        pricer = model.decoder(
            state=torch_state,
            expiries=torch_expiries,
            training=False,
        )
        calls = pricer(pricer.strikes, training=False)
    return (
        pricer.strikes.detach().cpu().numpy()[0].astype(np.float64),
        calls.detach().cpu().numpy()[0].astype(np.float64),
    )


def conditional_period_returns(
    model,
    sampler,
    *,
    surface_state: np.ndarray,
    expiries: np.ndarray,
    paths: int,
    seed: int,
    decoder_block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw model-consistent one-day spot and fixed-call returns."""

    import torch

    surface_state = np.asarray(surface_state, dtype=np.float64)
    expiries = np.asarray(expiries, dtype=np.float64)
    initial_state = np.concatenate((np.asarray([0.0]), surface_state))
    relative_strikes, initial_calls = _decode_current_surface(
        model,
        state=surface_state,
        expiries=expiries,
    )
    sampler.rng = np.random.default_rng(seed)
    state_paths = sampler.sample_conditional_xstates_at_offsets(
        initial_state=initial_state,
        step_offsets=np.asarray([0, 1], dtype=np.int64),
        num_samples=paths,
        dt=1.0 / DAYS_PER_YEAR,
    )
    endpoint_state = state_paths[:, 1]
    spot_ratio = np.exp(endpoint_state[:, 0])
    scenario_return = np.empty(
        (paths, 1 + initial_calls.size), dtype=np.float64
    )
    scenario_return[:, 0] = spot_ratio - 1.0
    torch_expiries = torch.as_tensor(
        expiries[None], dtype=model.env.fdtype, device=model.env.device
    )
    for start in range(0, paths, decoder_block_size):
        stop = min(start + decoder_block_size, paths)
        block_spot = spot_ratio[start:stop]
        block_state = torch.as_tensor(
            endpoint_state[start:stop, 1:],
            dtype=model.env.fdtype,
            device=model.env.device,
        )
        endpoint_strikes = relative_strikes[None] / block_spot[:, None, None]
        with torch.no_grad():
            endpoint_pricer = model.decoder(
                state=block_state,
                expiries=torch_expiries,
                training=False,
            )
            endpoint_calls = endpoint_pricer(
                torch.as_tensor(
                    endpoint_strikes,
                    dtype=model.env.fdtype,
                    device=model.env.device,
                ),
                dt=1.0 / DAYS_PER_YEAR,
                training=False,
            )
        endpoint_cash_calls = (
            block_spot[:, None, None]
            * endpoint_calls.detach().cpu().numpy().astype(np.float64)
        )
        scenario_return[start:stop, 1:] = (
            endpoint_cash_calls - initial_calls[None]
        ).reshape(stop - start, -1)
    if not np.all(np.isfinite(scenario_return)):
        raise FloatingPointError("conditional DYSANOS returns contain NaN/infinity")
    return scenario_return, relative_strikes, initial_calls


def decode_historical_fit(model, sampler, market: Mapping[str, Any]) -> np.ndarray:
    """Decode every historical state at the original 100-strike market grid."""

    import torch

    states = np.asarray(sampler.xstate[:, 1:], dtype=np.float64)
    expiries = np.asarray(market["expiries"], dtype=np.float64)
    strikes = np.asarray(market["xstrikes"], dtype=np.float64)[:, :, 1:-1]
    block_size = 256
    result = np.empty_like(strikes)
    for start in range(0, states.shape[0], block_size):
        stop = min(start + block_size, states.shape[0])
        with torch.no_grad():
            pricer = model.decoder(
                state=torch.as_tensor(
                    states[start:stop],
                    dtype=model.env.fdtype,
                    device=model.env.device,
                ),
                expiries=torch.as_tensor(
                    expiries[start:stop],
                    dtype=model.env.fdtype,
                    device=model.env.device,
                ),
                training=False,
            )
            calls = pricer(
                torch.as_tensor(
                    strikes[start:stop],
                    dtype=model.env.fdtype,
                    device=model.env.device,
                ),
                training=False,
            )
        result[start:stop] = calls.detach().cpu().numpy()
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("historical DYSANOS fit contains NaN/infinity")
    return result


def audit_historical_state(
    model,
    sampler,
    market: Mapping[str, Any],
    *,
    index: int,
    fit_error: Mapping[str, float],
    config: HistoricalStateAuditConfig,
) -> dict[str, Any]:
    """Fit and execute one historical state's one-day portfolio."""

    verify_inp(
        0 <= index < len(market["dates"]) - 1,
        lambda: f"historical state index {index} has no next observation",
    )
    expiries = np.asarray(market["expiries"][index], dtype=np.float64)
    training_return, relative_strikes, initial_model_calls = (
        conditional_period_returns(
            model,
            sampler,
            surface_state=sampler.xstate[index, 1:],
            expiries=expiries,
            paths=config.training_paths,
            seed=config.seed + 2_000_003 * index,
            decoder_block_size=config.decoder_block_size,
        )
    )
    costs = config.cost_rate * np.concatenate(
        (np.asarray([1.0]), initial_model_calls.reshape(-1))
    )
    action, optimum = solve_optimal_expected_return(training_return, costs)
    validation_return, validation_strikes, validation_initial_calls = (
        conditional_period_returns(
            model,
            sampler,
            surface_state=sampler.xstate[index, 1:],
            expiries=expiries,
            paths=config.validation_paths,
            seed=config.seed + 2_000_003 * index + 1,
            decoder_block_size=config.decoder_block_size,
        )
    )
    if not np.allclose(validation_strikes, relative_strikes) or not np.allclose(
        validation_initial_calls, initial_model_calls
    ):
        raise RuntimeError("conditional train/validation contract grids differ")
    training = _pnl_summary(
        action=action,
        scenario_return=training_return,
        costs=costs,
    )
    validation = _pnl_summary(
        action=action,
        scenario_return=validation_return,
        costs=costs,
    )

    entry_observed_calls = observed_surface_calls(
        grid_expiries=np.asarray(market["expiries"][index]),
        extended_strikes=np.asarray(market["xstrikes"][index]),
        inner_calls=np.asarray(market["mids"][index]),
        query_expiries=expiries,
        query_strikes=relative_strikes,
    )
    spot_ratio = float(
        market["spot_ref"][index + 1] / market["spot_ref"][index]
    )
    remaining_expiries = np.maximum(expiries - 1.0 / DAYS_PER_YEAR, 0.0)
    exit_observed_calls = observed_surface_calls(
        grid_expiries=np.asarray(market["expiries"][index + 1]),
        extended_strikes=np.asarray(market["xstrikes"][index + 1]),
        inner_calls=np.asarray(market["mids"][index + 1]),
        query_expiries=remaining_expiries,
        query_strikes=relative_strikes / spot_ratio,
    )
    execution = execute_frozen_action(
        action,
        spot_ratio=spot_ratio,
        entry_calls=entry_observed_calls,
        exit_calls=exit_observed_calls,
        cost_rate=config.cost_rate,
    )
    action_l1 = float(np.abs(action).sum())
    return {
        "state_index": index,
        "date": str(np.asarray(market["dates"])[index]),
        "next_market_date": str(np.asarray(market["dates"])[index + 1]),
        "spot_is_reset_to_one_for_model_fit": True,
        "realized_next_spot_ratio": spot_ratio,
        "surface_fit_error": dict(fit_error),
        "portfolio": {
            "l1_norm": action_l1,
            "nonzero_positions": int(
                np.count_nonzero(np.abs(action) > 1.0e-12)
            ),
            "training_optimum": optimum,
            "top_positions": _top_positions(
                action,
                relative_strikes=relative_strikes,
                expiries=expiries,
            ),
            "action": action.tolist(),
        },
        "model_training": training,
        "model_validation": validation,
        "next_market_execution": execution,
        "entry_market_minus_model_call_error": {
            "mae": float(
                np.mean(np.abs(entry_observed_calls - initial_model_calls))
            ),
            "maximum_abs": float(
                np.max(np.abs(entry_observed_calls - initial_model_calls))
            ),
        },
    }


def aggregate_historical_audits(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate historical-state optimization and execution outcomes."""

    verify_inp(len(rows) > 0, "no historical-state audit rows were supplied")
    action_l1 = np.asarray(
        [float(row["portfolio"]["l1_norm"]) for row in rows]
    )
    realized = np.asarray(
        [float(row["next_market_execution"]["net_pnl"]) for row in rows]
    )
    realized_band = np.asarray(
        [float(row["next_market_execution"]["numerical_band"]) for row in rows]
    )
    validation_mean = np.asarray(
        [float(row["model_validation"]["mean_pnl"]) for row in rows]
    )
    validation_losses = np.asarray(
        [
            int(row["model_validation"]["loss_count_beyond_float32_band"])
            for row in rows
        ]
    )
    active = action_l1 > 1.0e-12
    actual_loss = realized < -realized_band
    validation_mean_positive = validation_mean > 0.0

    def pnl_distribution(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "standard_error": float(
                np.std(values, ddof=0) / math.sqrt(values.size)
            ),
            "minimum": float(np.min(values)),
            "p01": float(np.quantile(values, 0.01)),
            "p10": float(np.quantile(values, 0.10)),
            "median": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p99": float(np.quantile(values, 0.99)),
            "maximum": float(np.max(values)),
        }

    active_realized = realized[active]
    aggregate: dict[str, Any] = {
        "historical_states_executed": len(rows),
        "active_portfolios": int(np.count_nonzero(active)),
        "active_portfolio_frequency": float(np.mean(active)),
        "positive_model_validation_mean_count": int(
            np.count_nonzero(validation_mean_positive)
        ),
        "positive_model_validation_mean_frequency": float(
            np.mean(validation_mean_positive)
        ),
        "validation_portfolios_with_any_loss": int(
            np.count_nonzero(validation_losses > 0)
        ),
        "realized_next_market_losses_beyond_band": int(
            np.count_nonzero(actual_loss)
        ),
        "realized_next_market_loss_frequency": float(np.mean(actual_loss)),
        "realized_next_market_pnl": pnl_distribution(realized),
        "model_validation_mean_pnl": pnl_distribution(validation_mean),
        "correlation_model_validation_mean_to_realized_pnl": (
            float(np.corrcoef(validation_mean, realized)[0, 1])
            if np.std(validation_mean) > 0.0 and np.std(realized) > 0.0
            else None
        ),
    }
    aggregate["active_portfolio_realized_next_market_pnl"] = (
        pnl_distribution(active_realized) if active_realized.size else None
    )
    return aggregate


__all__ = [
    "HistoricalStateAuditConfig",
    "aggregate_historical_audits",
    "audit_historical_state",
    "conditional_period_returns",
    "decode_historical_fit",
    "execute_frozen_action",
    "observed_surface_calls",
    "solve_optimal_expected_return",
    "solve_mean_variance_portfolio",
    "project_l1_ball",
    "summarize_fit_errors",
]
