"""Continuous-density consistency constraints and HJM drift."""

from __future__ import annotations

import numpy as np
from cdxcore.err import verify, verify_inp

from .coordinates import trapezoid_weights


def constraint_matrix(
    jump_x: np.ndarray,
    *,
    include_mass: bool = False,
) -> np.ndarray:
    """Return discrete linear functionals for factor-mode constraints.

    Continuous DETL factors require the exponential-moment constraint RA3.
    Discrete Tangent Lévy factors additionally require the zero-mass condition.
    """

    jump_x = np.asarray(jump_x, dtype=np.float64)
    weights = trapezoid_weights(jump_x)
    rows = [weights * np.expm1(jump_x)]
    if include_mass:
        rows.insert(0, weights)
    return np.stack(rows, axis=0)


def project_factor_constraints(
    values: np.ndarray,
    *,
    jump_x: np.ndarray,
    include_mass: bool = False,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Euclidean-project every trailing jump vector into the nullspace."""

    values = np.asarray(values, dtype=np.float64)
    jump_x = np.asarray(jump_x, dtype=np.float64)
    verify_inp(
        values.ndim >= 1
        and values.shape[-1] == jump_x.size
        and np.all(np.isfinite(values)),
        lambda: (
            "values must be finite and have trailing jump dimension "
            f"{jump_x.size}, got {values.shape}"
        ),
    )
    constraints = constraint_matrix(jump_x, include_mass=include_mass)
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)
        verify_inp(
            values.ndim >= 2 and mask.shape == values.shape[-2:],
            lambda: (
                "active mask must match the trailing maturity/jump shape "
                f"{values.shape[-2:]}, got {mask.shape}"
            ),
        )
        projected = np.zeros_like(values)
        for maturity_index, active in enumerate(mask):
            restricted = constraints[:, active]
            if (
                np.count_nonzero(active) <= restricted.shape[0]
                or np.linalg.matrix_rank(restricted) < restricted.shape[0]
            ):
                # This slice has no nonzero constrained degree of freedom and
                # therefore remains deterministic.
                continue
            selected = values[..., maturity_index, active]
            gram = restricted @ restricted.T
            coefficients = (selected @ restricted.T) @ np.linalg.inv(gram)
            projected[..., maturity_index, active] = (
                selected - coefficients @ restricted
            )
        return projected
    gram = constraints @ constraints.T
    correction_coefficients = (values @ constraints.T) @ np.linalg.inv(gram)
    return values - correction_coefficients @ constraints


def constraint_residuals(
    values: np.ndarray,
    *,
    jump_x: np.ndarray,
    include_mass: bool = False,
) -> np.ndarray:
    """Evaluate each discrete constraint on the trailing jump dimension."""

    values = np.asarray(values, dtype=np.float64)
    constraints = constraint_matrix(jump_x, include_mass=include_mass)
    return values @ constraints.T


def maturity_integral(tau: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return ``integral_0^tau values(s, x) ds`` on the supplied grid."""

    tau = np.asarray(tau, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    verify_inp(
        tau.ndim == 1
        and tau.size >= 2
        and tau[0] == 0.0
        and np.all(np.diff(tau) > 0.0),
        lambda: "maturity grid must start at zero and be strictly increasing",
    )
    verify_inp(
        values.ndim >= 2
        and values.shape[-2] == tau.size
        and np.all(np.isfinite(values)),
        lambda: (
            "values must be finite with trailing maturity size "
            f"{tau.size}, got {values.shape}"
        ),
    )
    increments = 0.5 * (values[..., 1:, :] + values[..., :-1, :])
    increments *= np.diff(tau).reshape(
        *((1,) * (values.ndim - 2)), -1, 1
    )
    return np.concatenate(
        [np.zeros_like(values[..., :1, :]), np.cumsum(increments, axis=-2)],
        axis=-2,
    )


def hjm_drift(
    *,
    tau: np.ndarray,
    jump_x: np.ndarray,
    beta: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Discretize the convolutional HJM drift restriction (2.7).

    ``beta`` has shape ``(F, T, X)`` and is in annualized units. Values outside
    the finite symmetric jump grid are treated as zero.
    """

    tau = np.asarray(tau, dtype=np.float64)
    jump_x = np.asarray(jump_x, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    verify_inp(
        beta.ndim == 3
        and beta.shape[1:] == (tau.size, jump_x.size)
        and np.all(np.isfinite(beta)),
        lambda: (
            "'beta' must be finite with shape (F,T,X), trailing shape "
            f"{(tau.size, jump_x.size)}, got {beta.shape}"
        ),
    )
    verify_inp(
        jump_x.size % 2 == 1
        and np.allclose(jump_x, -jump_x[::-1])
        and np.allclose(np.diff(jump_x), np.diff(jump_x)[0]),
        "jump grid must be odd, symmetric, and uniform for convolution",
    )
    beta_bar = maturity_integral(tau, beta)
    x_weights = trapezoid_weights(jump_x)
    beta_integral = beta @ x_weights
    beta_bar_integral = beta_bar @ x_weights
    drift = np.zeros((tau.size, jump_x.size), dtype=np.float64)
    dx = float(jump_x[1] - jump_x[0])
    for factor in range(beta.shape[0]):
        for maturity in range(tau.size):
            convolution = np.convolve(
                beta_bar[factor, maturity],
                beta[factor, maturity],
                mode="same",
            ) * dx
            drift[maturity] -= (
                convolution
                - beta_bar[factor, maturity] * beta_integral[factor, maturity]
                - beta[factor, maturity] * beta_bar_integral[factor, maturity]
            )
    # Cropping the full convolution back to finite jump support introduces a
    # small exponential-moment residual.  Remove that quadrature artifact so
    # the numerical drift remains tangent to the configured state manifold.
    drift = project_factor_constraints(
        drift,
        jump_x=jump_x,
        active_mask=active_mask,
    )
    verify(
        np.all(np.isfinite(drift)),
        lambda: "HJM drift calculation produced non-finite values",
    )
    return drift


def musiela_drift(
    *,
    tau: np.ndarray,
    forward_density: np.ndarray,
    hjm_alpha: np.ndarray,
) -> np.ndarray:
    """Add the maturity-transport derivative from equation (2.30)."""

    tau = np.asarray(tau, dtype=np.float64)
    kappa = np.asarray(forward_density, dtype=np.float64)
    alpha = np.asarray(hjm_alpha, dtype=np.float64)
    verify_inp(
        alpha.ndim == 2
        and kappa.ndim >= 2
        and alpha.shape[0] == tau.size
        and kappa.shape[-2:] == alpha.shape
        and np.all(np.isfinite(kappa))
        and np.all(np.isfinite(alpha)),
        lambda: (
            "kappa and HJM alpha must be finite with common trailing shape, got "
            f"{kappa.shape} and {alpha.shape}"
        ),
    )
    maturity_derivative = np.gradient(kappa, tau, axis=-2, edge_order=2)
    return maturity_derivative + alpha
