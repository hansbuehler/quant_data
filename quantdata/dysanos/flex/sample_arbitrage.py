"""Finite-scenario strict-arbitrage separation for weekly option rolls."""

from __future__ import annotations

from typing import Any

import numpy as np
from cdxcore.err import verify_inp
from scipy.optimize import linprog


def solve_strict_sample_arbitrage(
    returns: np.ndarray,
    *,
    constraints_per_iteration: int = 128,
    maximum_iterations: int = 128,
    feasibility_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    r"""Maximize worst-scenario P&L over signed unit-L1 portfolios.

    ``returns[p, i]`` is the one-period value change of instrument ``i`` in
    scenario ``p``. Cash financing is implicit: buying action ``a`` at the
    current instrument prices and borrowing its cost gives terminal P&L
    ``returns @ a``. A strictly positive optimum separates the current price
    vector from the convex hull of sampled successor values.
    """
    values = np.asarray(returns, dtype=np.float64)
    verify_inp(
        values.ndim == 2 and values.shape[0] >= 2 and values.shape[1] >= 1,
        lambda: f"returns must have shape (paths,instruments), got {values.shape}",
    )
    verify_inp(
        np.all(np.isfinite(values)),
        lambda: "sample-arbitrage returns contain NaN or infinity",
    )
    verify_inp(
        constraints_per_iteration > 0 and maximum_iterations > 0,
        lambda: "constraint-generation controls must be positive",
    )
    verify_inp(
        feasibility_tolerance > 0.0,
        lambda: "feasibility_tolerance must be positive",
    )

    paths, instruments = values.shape
    rms = np.sqrt(np.mean(np.square(values), axis=0))
    active = rms > 64.0 * np.finfo(np.float64).eps
    if not np.any(active):
        return {
            "success": True,
            "strict_sample_arbitrage": False,
            "action": np.zeros(instruments, dtype=np.float64).tolist(),
            "active_instruments": 0,
            "strict_margin": 0.0,
            "strict_margin_bps_spot": 0.0,
            "numerical_band": 0.0,
            "numerical_band_bps_spot": 0.0,
            "selected_constraints": 0,
            "iterations": 0,
            "solver_status": 0,
            "solver_message": "all sampled returns are identically zero",
        }

    normalized = values[:, active] / rms[active][None]
    dimension = normalized.shape[1]
    # Variables are signed action, its absolute-value envelope, and the worst
    # scenario margin. This avoids the a_plus/a_minus null directions.
    objective = np.zeros(2 * dimension + 1, dtype=np.float64)
    objective[-1] = -1.0
    identity = np.eye(dimension, dtype=np.float64)
    absolute_envelope = np.concatenate(
        (
            np.concatenate((identity, -identity, np.zeros((dimension, 1))), axis=1),
            np.concatenate((-identity, -identity, np.zeros((dimension, 1))), axis=1),
        ),
        axis=0,
    )
    gross_constraint = np.concatenate(
        (np.zeros(dimension), np.ones(dimension), np.zeros(1))
    )[None]
    fixed_inequality = np.concatenate((absolute_envelope, gross_constraint), axis=0)
    fixed_rhs = np.concatenate((np.zeros(2 * dimension), np.ones(1)))
    bounds = (
        [(None, None)] * dimension
        + [(0.0, None)] * dimension
        + [(None, None)]
    )

    selected = np.unique(
        np.concatenate(
            (np.argmin(normalized, axis=0), np.argmax(normalized, axis=0))
        )
    )
    optimized = None
    solver_attempts: list[dict[str, object]] = []
    solver_method = ""
    for iteration in range(maximum_iterations):
        scenario_constraints = np.concatenate(
            (
                -normalized[selected],
                np.zeros((len(selected), dimension), dtype=np.float64),
                np.ones((len(selected), 1), dtype=np.float64),
            ),
            axis=1,
        )
        inequality = np.concatenate((fixed_inequality, scenario_constraints), axis=0)
        right_hand_side = np.concatenate((fixed_rhs, np.zeros(len(selected))))
        optimized = None
        iteration_attempts: list[dict[str, object]] = []
        for method in ("highs", "highs-ds", "highs-ipm"):
            candidate = linprog(
                objective,
                A_ub=inequality,
                b_ub=right_hand_side,
                bounds=bounds,
                method=method,
                options={
                    "presolve": True,
                    "primal_feasibility_tolerance": feasibility_tolerance,
                    "dual_feasibility_tolerance": feasibility_tolerance,
                    "ipm_optimality_tolerance": feasibility_tolerance,
                },
            )
            attempt = {
                "iteration": iteration + 1,
                "method": method,
                "success": bool(candidate.success),
                "status": int(candidate.status),
                "message": str(candidate.message),
            }
            iteration_attempts.append(attempt)
            if candidate.success and candidate.x is not None:
                primal_violation = float(
                    np.max(inequality @ candidate.x - right_hand_side)
                )
                attempt["maximum_primal_violation"] = primal_violation
                if (
                    np.all(np.isfinite(candidate.x))
                    and primal_violation <= 4.0 * feasibility_tolerance
                ):
                    optimized = candidate
                    solver_method = method
                    break
                attempt["success"] = False
                attempt["message"] = "solver success failed explicit primal audit"
        solver_attempts.extend(iteration_attempts)
        if optimized is None or optimized.x is None:
            raise RuntimeError(
                "strict sample-arbitrage LP failed all deterministic methods: "
                f"{iteration_attempts}"
            )
        normalized_action = optimized.x[:dimension]
        pnl = normalized @ normalized_action
        target_margin = float(optimized.x[-1])
        violation = target_margin - pnl
        violating = np.flatnonzero(violation > feasibility_tolerance)
        if violating.size == 0:
            break
        worst = violating[
            np.argsort(-violation[violating], kind="stable")[:constraints_per_iteration]
        ]
        expanded = np.unique(np.concatenate((selected, worst)))
        if len(expanded) == len(selected):
            raise RuntimeError("strict sample-arbitrage constraint generation stalled")
        selected = expanded
    else:
        raise RuntimeError(
            f"strict sample-arbitrage LP did not converge in {maximum_iterations} iterations"
        )
    assert optimized is not None and optimized.x is not None

    action = np.zeros(instruments, dtype=np.float64)
    action[active] = optimized.x[:dimension] / rms[active]
    gross = float(np.sum(np.abs(action)))
    if gross > 0.0:
        action /= gross
    economic_pnl = values @ action
    strict_margin = float(np.min(economic_pnl))
    # The path values were decoded in float32. This conservative band scales
    # with portfolio gross exposure and the largest observed instrument move.
    numerical_band = float(
        64.0
        * np.finfo(np.float32).eps
        * np.dot(np.abs(action), 1.0 + np.max(np.abs(values), axis=0))
    )
    return {
        "success": True,
        "strict_sample_arbitrage": strict_margin > numerical_band,
        "action": action.tolist(),
        "active_instruments": int(np.count_nonzero(active)),
        "strict_margin": strict_margin,
        "strict_margin_bps_spot": strict_margin * 1.0e4,
        "maximum_pnl": float(np.max(economic_pnl)),
        "maximum_pnl_bps_spot": float(np.max(economic_pnl)) * 1.0e4,
        "numerical_band": numerical_band,
        "numerical_band_bps_spot": numerical_band * 1.0e4,
        "selected_constraints": int(len(selected)),
        "iterations": iteration + 1,
        "solver_status": int(optimized.status),
        "solver_message": str(optimized.message),
        "solver_method": solver_method,
        "solver_attempts": solver_attempts,
    }


__all__ = ["solve_strict_sample_arbitrage"]
