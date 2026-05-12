from __future__ import annotations

import cvxpy as cp
import numpy as np

from quantdata.utils.arbfree import repair_density_nonnegative


def _solve_projection_with_cvxpy(
    p: np.ndarray,
    x: np.ndarray,
    *,
    target_sum: float = 1.0,
    target_xmoment: float = 1.0,
    floor: float = 0.0,
) -> np.ndarray:
    y = cp.Variable(p.size)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(y - p)),
        [
            y >= floor,
            cp.sum(y) == target_sum,
            x @ y == target_xmoment,
        ],
    )
    problem.solve(solver="CLARABEL")

    assert problem.status in {"optimal", "optimal_inaccurate"}
    assert y.value is not None
    return np.asarray(y.value, dtype=np.float64)


def test_repair_density_nonnegative_keeps_feasible_density_unchanged() -> None:
    x = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    p = np.array([0.25, 0.5, 0.25], dtype=np.float64)

    repaired = repair_density_nonnegative(p=p, x=x)

    np.testing.assert_allclose(repaired, p, rtol=0.0, atol=1e-12)
    assert np.all(repaired >= 0.0)
    assert abs(np.sum(repaired) - 1.0) <= 1e-12
    assert abs(repaired @ x - 1.0) <= 1e-12


def test_repair_density_nonnegative_matches_cvxpy_projection() -> None:
    x = np.array([0.45, 0.8, 1.1, 1.65], dtype=np.float64)
    p = np.array([-0.08, 0.52, 0.41, 0.16], dtype=np.float64)

    repaired = repair_density_nonnegative(p=p, x=x)
    expected = _solve_projection_with_cvxpy(p=p, x=x)

    np.testing.assert_allclose(repaired, expected, rtol=0.0, atol=1e-7)
    assert np.all(repaired >= -1e-10)
    assert abs(np.sum(repaired) - 1.0) <= 1e-10
    assert abs(repaired @ x - 1.0) <= 1e-10


def test_repair_density_nonnegative_respects_floor_and_custom_targets() -> None:
    x = np.array([0.7, 0.95, 1.25, 1.8], dtype=np.float64)
    p = np.array([0.01, 0.72, 0.18, 0.09], dtype=np.float64)
    floor = 0.03
    target_sum = 1.2
    target_xmoment = 1.32

    repaired = repair_density_nonnegative(
        p=p,
        x=x,
        target_sum=target_sum,
        target_xmoment=target_xmoment,
        floor=floor,
    )
    expected = _solve_projection_with_cvxpy(
        p=p,
        x=x,
        target_sum=target_sum,
        target_xmoment=target_xmoment,
        floor=floor,
    )

    np.testing.assert_allclose(repaired, expected, rtol=0.0, atol=1e-7)
    assert np.min(repaired) >= floor - 1e-10
    assert abs(np.sum(repaired) - target_sum) <= 1e-10
    assert abs(repaired @ x - target_xmoment) <= 1e-10