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
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)
        verify_inp(
            mask.shape == drift.shape,
            lambda: (
                "active mask must match the HJM drift shape "
                f"{drift.shape}, got {mask.shape}"
            ),
        )
    # The factor support does not constrain the drift support: convolution can
    # create non-zero alpha at jump locations where every beta is zero.  Do not
    # project or mask alpha after applying the HJM formula, since either action
    # destroys the characteristic-generator identity.  Callers must instead
    # choose factor support whose convolution fits inside the configured grid.
    verify(
        np.all(np.isfinite(drift)),
        lambda: "HJM drift calculation produced non-finite values",
    )
    return drift


def levy_characteristic_operator(
    values: np.ndarray,
    *,
    jump_x: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Apply the compensated exponential-Levy operator on the jump lattice.

    The quadrature and compensator match the finite-grid pricing convention.
    ``values`` may have arbitrary leading dimensions and a trailing jump axis;
    the result replaces that axis by the frequency axis.
    """

    values = np.asarray(values, dtype=np.float64)
    jump_x = np.asarray(jump_x, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    verify_inp(
        values.ndim >= 1
        and values.shape[-1] == jump_x.size
        and np.all(np.isfinite(values)),
        lambda: (
            "values must be finite with trailing jump dimension "
            f"{jump_x.size}, got {values.shape}"
        ),
    )
    verify_inp(
        frequencies.ndim == 1 and np.all(np.isfinite(frequencies)),
        lambda: "frequencies must be a finite one-dimensional array",
    )
    kernel = (
        np.exp(1j * frequencies[:, None] * jump_x[None])
        - 1.0
        - 1j * frequencies[:, None] * np.expm1(jump_x)[None]
    )
    kernel *= trapezoid_weights(jump_x)[None]
    return np.einsum("...x,ux->...u", values, kernel)


def hjm_characteristic_residual(
    *,
    tau: np.ndarray,
    jump_x: np.ndarray,
    beta: np.ndarray,
    alpha: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Return the finite-grid Tangent Levy HJM generator residual.

    A dynamically consistent lattice satisfies this identity at every
    maturity and Fourier frequency, not merely the exponential-moment
    constraint on each factor.
    """

    beta = np.asarray(beta, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    verify_inp(
        beta.ndim == 3 and alpha.shape == beta.shape[1:],
        lambda: (
            "beta and alpha must have shapes (F,T,X) and (T,X), got "
            f"{beta.shape} and {alpha.shape}"
        ),
    )
    beta_bar = maturity_integral(np.asarray(tau, dtype=np.float64), beta)
    alpha_operator = levy_characteristic_operator(
        alpha,
        jump_x=jump_x,
        frequencies=frequencies,
    )
    beta_operator = levy_characteristic_operator(
        beta,
        jump_x=jump_x,
        frequencies=frequencies,
    )
    beta_bar_operator = levy_characteristic_operator(
        beta_bar,
        jump_x=jump_x,
        frequencies=frequencies,
    )
    return alpha_operator + np.sum(
        beta_operator * beta_bar_operator,
        axis=0,
    )


def hjm_cell_drift(
    beta_cells: np.ndarray,
    *,
    jump_x: np.ndarray,
) -> np.ndarray:
    """Return cell-integrated HJM drift from cell-integrated factors.

    Cell measures make maturity transport an exact index shift.  Constructing
    cumulative alpha from the quadratic HJM identity also avoids imposing a
    false trapezoidal product rule on the maturity grid.
    """

    beta_cells = np.asarray(beta_cells, dtype=np.float64)
    jump_x = np.asarray(jump_x, dtype=np.float64)
    verify_inp(
        beta_cells.ndim == 3
        and beta_cells.shape[-1] == jump_x.size
        and np.all(np.isfinite(beta_cells)),
        lambda: (
            "beta cells must be finite with shape (F,C,X), got "
            f"{beta_cells.shape}"
        ),
    )
    verify_inp(
        jump_x.size >= 3
        and jump_x.size % 2 == 1
        and np.allclose(jump_x, -jump_x[::-1])
        and np.allclose(np.diff(jump_x), np.diff(jump_x)[0]),
        "jump grid must be odd, symmetric, and uniform for convolution",
    )
    center = jump_x.size // 2
    tolerance = 1.0e-14 * max(
        1.0,
        float(np.max(np.abs(beta_cells), initial=0.0)),
    )
    for factor in range(beta_cells.shape[0]):
        support = np.flatnonzero(
            np.any(np.abs(beta_cells[factor]) > tolerance, axis=0)
        )
        if support.size == 0:
            continue
        verify_inp(
            2 * int(support[0]) - center >= 0
            and 2 * int(support[-1]) - center < jump_x.size,
            lambda: (
                "factor support is not closed under self-convolution on the "
                f"configured jump grid (factor {factor}, support indices "
                f"{support[0]}:{support[-1]}, centre {center})"
            ),
        )
    cumulative_beta = np.concatenate(
        (
            np.zeros_like(beta_cells[:, :1]),
            np.cumsum(beta_cells, axis=1),
        ),
        axis=1,
    )
    weights = trapezoid_weights(jump_x)
    dx = float(jump_x[1] - jump_x[0])
    cumulative_alpha = np.zeros(cumulative_beta.shape[1:], dtype=np.float64)
    for factor in range(beta_cells.shape[0]):
        for boundary in range(cumulative_beta.shape[1]):
            integrated = cumulative_beta[factor, boundary]
            convolution = np.convolve(integrated, integrated, mode="same") * dx
            mass = integrated @ weights
            cumulative_alpha[boundary] -= 0.5 * (
                convolution - 2.0 * integrated * mass
            )
    return np.diff(cumulative_alpha, axis=0)


def hjm_integrated_characteristic_residual(
    *,
    beta_cells: np.ndarray,
    alpha_cells: np.ndarray,
    jump_x: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Return the cumulative lattice HJM residual at every cell boundary."""

    beta_cells = np.asarray(beta_cells, dtype=np.float64)
    alpha_cells = np.asarray(alpha_cells, dtype=np.float64)
    verify_inp(
        beta_cells.ndim == 3 and alpha_cells.shape == beta_cells.shape[1:],
        lambda: (
            "beta and alpha cells must have shapes (F,C,X) and (C,X), got "
            f"{beta_cells.shape} and {alpha_cells.shape}"
        ),
    )
    cumulative_beta = np.concatenate(
        (np.zeros_like(beta_cells[:, :1]), np.cumsum(beta_cells, axis=1)),
        axis=1,
    )
    cumulative_alpha = np.concatenate(
        (np.zeros_like(alpha_cells[:1]), np.cumsum(alpha_cells, axis=0)),
        axis=0,
    )
    alpha_operator = levy_characteristic_operator(
        cumulative_alpha,
        jump_x=jump_x,
        frequencies=frequencies,
    )
    beta_operator = levy_characteristic_operator(
        cumulative_beta,
        jump_x=jump_x,
        frequencies=frequencies,
    )
    return alpha_operator + 0.5 * np.sum(beta_operator * beta_operator, axis=0)


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
