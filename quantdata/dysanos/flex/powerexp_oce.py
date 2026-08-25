"""Power-exponential OCE diagnostics for frozen portfolio directions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from cdxcore.err import verify_inp


def powerexp_utility(value: np.ndarray) -> np.ndarray:
    r"""Evaluate the concave power-exponential utility with unit slope at zero.

    The utility is exponential on gains and quadratic on losses:

    .. math::

        u(z)=1-e^{-z}\quad(z>0),\qquad
        u(z)=z-z^2/2\quad(z\leq0).
    """

    argument = np.asarray(value, dtype=np.float64)
    result = np.empty_like(argument)
    positive = argument > 0.0
    result[positive] = -np.expm1(-argument[positive])
    negative = argument[~positive]
    result[~positive] = negative - 0.5 * np.square(negative)
    return result


def powerexp_marginal(value: np.ndarray) -> np.ndarray:
    """Evaluate the strictly positive marginal utility without branch overflow."""

    argument = np.asarray(value, dtype=np.float64)
    result = np.empty_like(argument)
    positive = argument > 0.0
    result[positive] = np.exp(-argument[positive])
    result[~positive] = 1.0 - argument[~positive]
    return result


def _translation_for_notional(pnl: np.ndarray, notional: float) -> np.ndarray:
    r"""Solve ``mean(u'(y + notional * pnl)) = 1`` per direction."""

    scaled = notional * pnl
    lower = -np.max(scaled, axis=0) - 1.0
    upper = -np.min(scaled, axis=0) + 1.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        condition = np.mean(powerexp_marginal(scaled + middle[None]), axis=0) - 1.0
        lower = np.where(condition > 0.0, middle, lower)
        upper = np.where(condition > 0.0, upper, middle)
    return 0.5 * (lower + upper)


def evaluate_powerexp_oce_grid(
    pnl: np.ndarray,
    notionals: Sequence[float],
) -> list[dict[str, Any]]:
    r"""Evaluate frozen-direction OCEs on an externally fixed notional grid.

    For every P&L column and notional ``H`` this evaluates

    .. math::

        \sup_y E[u(y+H\pi)]-y.

    Only the scalar translation required by the definition of OCE is solved.
    The portfolio direction and notional are never optimized here.
    """

    values = np.asarray(pnl, dtype=np.float64)
    grid = tuple(float(value) for value in notionals)
    verify_inp(
        values.ndim == 2 and values.shape[0] >= 2 and values.shape[1] >= 1,
        lambda: f"pnl must have shape (paths,directions), got {values.shape}",
    )
    verify_inp(np.all(np.isfinite(values)), lambda: "power-exp P&L contains NaN or infinity")
    verify_inp(
        bool(grid)
        and all(np.isfinite(value) and value >= 0.0 for value in grid)
        and all(right > left for left, right in zip(grid, grid[1:], strict=False)),
        lambda: f"notionals must be finite, nonnegative, and increasing: {grid}",
    )

    result: list[dict[str, Any]] = []
    root_paths = np.sqrt(values.shape[0])
    for notional in grid:
        translation = _translation_for_notional(values, notional)
        shifted = translation[None] + notional * values
        utility = powerexp_utility(shifted)
        marginal = powerexp_marginal(shifted)
        objective = np.mean(utility, axis=0) - translation
        derivative_paths = marginal * values
        derivative = np.mean(derivative_paths, axis=0)
        derivative_standard_error = np.std(derivative_paths, axis=0, ddof=1) / root_paths
        objective_standard_error = np.std(utility, axis=0, ddof=1) / root_paths
        marginal_mass = np.sum(marginal, axis=0)
        marginal_probability = marginal / marginal_mass[None]
        effective_sample_size = 1.0 / np.sum(np.square(marginal_probability), axis=0)
        if not all(
            np.all(np.isfinite(value))
            for value in (
                translation,
                objective,
                derivative,
                derivative_standard_error,
                objective_standard_error,
                effective_sample_size,
            )
        ):
            raise FloatingPointError("power-exp OCE evaluation is non-finite")
        result.append(
            {
                "notional": notional,
                "translation": translation.tolist(),
                "oce": objective.tolist(),
                "oce_standard_error": objective_standard_error.tolist(),
                "notional_derivative": derivative.tolist(),
                "notional_derivative_standard_error": (derivative_standard_error.tolist()),
                "marginal_effective_sample_size": effective_sample_size.tolist(),
                "maximum_translation_first_order_residual": float(
                    np.max(np.abs(np.mean(marginal, axis=0) - 1.0))
                ),
            }
        )
    return result


__all__ = [
    "evaluate_powerexp_oce_grid",
    "powerexp_marginal",
    "powerexp_utility",
]
