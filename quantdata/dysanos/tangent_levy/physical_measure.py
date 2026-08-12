"""Optional low-dimensional physical-measure drift estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify_inp

from .constraints import hjm_drift, musiela_drift
from .dynamics import DynamicsFit


@dataclass(frozen=True)
class PhysicalMeasureFit:
    """Regularized constant factor market price of risk."""

    theta: np.ndarray
    ridge: float
    residual_rmse: float
    target_drift_norm: float
    explained_drift_fraction: float


def fit_constant_market_price_of_risk(
    forward_density_history: np.ndarray,
    dynamics: DynamicsFit,
    *,
    ridge: float = 1.0e-4,
    consecutive_mask: np.ndarray | None = None,
) -> PhysicalMeasureFit:
    """Fit ``alpha_P = alpha_Q + sum beta_n theta_n`` to historical mean drift.

    This function estimates a physical layer; it does not alter or re-estimate
    the HJM-consistent risk-neutral drift stored by the core model.
    """

    history = np.asarray(forward_density_history, dtype=np.float64)
    verify_inp(
        history.ndim == 3
        and history.shape[1:] == dynamics.beta.shape[1:]
        and history.shape[0] >= 3
        and np.all(np.isfinite(history)),
        lambda: (
            "history must be finite with shape (D,T,X) matching dynamics, got "
            f"{history.shape}"
        ),
    )
    verify_inp(
        math.isfinite(ridge) and ridge >= 0.0,
        lambda: f"'ridge' must be finite and nonnegative, got {ridge!r}",
    )
    changes = np.diff(history, axis=0)
    state_before_change = history[:-1]
    if consecutive_mask is not None:
        consecutive = np.asarray(consecutive_mask, dtype=bool)
        verify_inp(
            consecutive.shape == (changes.shape[0],),
            lambda: (
                "'consecutive_mask' must have one value per date transition, got "
                f"{consecutive.shape}"
            ),
        )
        changes = changes[consecutive]
        state_before_change = state_before_change[consecutive]
    verify_inp(
        changes.shape[0] >= 2,
        lambda: (
            "physical-drift estimation requires at least two consecutive changes, "
            f"got {changes.shape[0]}"
        ),
    )
    empirical_drift = np.mean(changes, axis=0) / dynamics.dt
    mean_state = np.mean(state_before_change, axis=0)
    active_mask = (
        np.ones(mean_state.shape, dtype=bool)
        if dynamics.active_mask is None
        else np.asarray(dynamics.active_mask, dtype=bool)
    )
    alpha = hjm_drift(
        tau=dynamics.tau,
        jump_x=dynamics.jump_x,
        beta=dynamics.beta,
        active_mask=active_mask,
    )
    q_drift = musiela_drift(
        tau=dynamics.tau,
        forward_density=mean_state,
        hjm_alpha=alpha,
    )
    target = (empirical_drift - q_drift)[active_mask]
    factor_matrix = dynamics.beta[:, active_mask].T
    gram = factor_matrix.T @ factor_matrix
    theta = np.linalg.solve(
        gram + ridge * np.eye(dynamics.num_factors),
        factor_matrix.T @ target,
    )
    residual = target - factor_matrix @ theta
    target_norm = float(np.linalg.norm(target))
    explained = (
        1.0 - float(np.dot(residual, residual)) / float(np.dot(target, target))
        if target_norm > 0.0
        else 0.0
    )
    return PhysicalMeasureFit(
        theta=theta,
        ridge=float(ridge),
        residual_rmse=float(np.sqrt(np.mean(np.square(residual)))),
        target_drift_norm=target_norm,
        explained_drift_fraction=explained,
    )
