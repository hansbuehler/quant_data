"""Constrained PCA estimation of Tangent Lévy code-book dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify, verify_inp

from .config import DynamicsConfig
from .constraints import constraint_residuals, project_factor_constraints


@dataclass(frozen=True)
class DynamicsFit:
    """Annualized factor representation of one-session ``kappa`` changes."""

    tau: np.ndarray
    jump_x: np.ndarray
    beta: np.ndarray
    eigenvalues_one_session: np.ndarray
    explained_variance_ratio: np.ndarray
    mean_change: np.ndarray
    observations: int
    maximum_constraint_residual: float
    covariance_shrinkage: float
    dt: float
    active_mask: np.ndarray | None = None

    @property
    def num_factors(self) -> int:
        """Number of retained Brownian factors."""

        return int(self.beta.shape[0])

    def reconstructed_one_session_covariance(self) -> np.ndarray:
        """Return the retained low-rank covariance in flattened coordinates."""

        flat = self.beta.reshape(self.num_factors, -1)
        return self.dt * flat.T @ flat


def _orient_components(components: np.ndarray) -> np.ndarray:
    result = components.copy()
    for index, component in enumerate(result):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0.0:
            result[index] *= -1.0
    return result


def fit_dynamics(
    forward_density_history: np.ndarray,
    *,
    config: DynamicsConfig = DynamicsConfig(),
    consecutive_mask: np.ndarray | None = None,
    reference_density: np.ndarray | None = None,
) -> DynamicsFit:
    """Fit constrained constant factors to consecutive daily code-book changes.

    ``reference_density`` selects coordinates with enough positivity headroom at
    the state where the fitted local diffusion will be initialized.  When it is
    omitted, the historical coordinate-wise minimum retains the conservative
    legacy behavior.
    """

    history = np.asarray(forward_density_history, dtype=np.float64)
    tau = np.asarray(config.tau_grid, dtype=np.float64)
    jump_x = np.asarray(config.jump_x_grid, dtype=np.float64)
    verify_inp(
        history.ndim == 3
        and history.shape[1:] == (tau.size, jump_x.size)
        and history.shape[0] >= 3
        and np.all(np.isfinite(history))
        and np.all(history >= 0.0),
        lambda: (
            "history must be finite, nonnegative, and have shape (D,T,X) with "
            f"at least three dates; got {history.shape}"
        ),
    )
    changes = np.diff(history, axis=0)
    if consecutive_mask is not None:
        consecutive_mask = np.asarray(consecutive_mask, dtype=bool)
        verify_inp(
            consecutive_mask.shape == (changes.shape[0],),
            lambda: (
                "'consecutive_mask' must have one value per date transition, got "
                f"{consecutive_mask.shape}"
            ),
        )
        changes = changes[consecutive_mask]
    verify_inp(
        changes.shape[0] >= config.minimum_observations,
        lambda: (
            "dynamics covariance requires at least "
            f"{config.minimum_observations} consecutive changes, got "
            f"{changes.shape[0]}"
        ),
    )
    empirical_scale = np.std(changes, axis=0, ddof=1)
    if reference_density is None:
        positivity_reference = np.min(history, axis=0)
    else:
        positivity_reference = np.asarray(reference_density, dtype=np.float64)
        verify_inp(
            positivity_reference.shape == (tau.size, jump_x.size)
            and np.all(np.isfinite(positivity_reference))
            and np.all(positivity_reference >= 0.0),
            lambda: (
                "reference density must be finite, nonnegative, and have shape "
                f"{(tau.size, jump_x.size)}, got {positivity_reference.shape}"
            ),
        )
    active_mask = (
        (positivity_reference > config.inactive_density_threshold)
        & (
            positivity_reference
            > config.boundary_standard_deviations * empirical_scale
        )
    )
    for _ in range(5):
        projected = project_factor_constraints(
            changes,
            jump_x=jump_x,
            active_mask=active_mask,
        )
        projected_scale = np.std(projected, axis=0, ddof=1)
        updated_mask = active_mask & (
            positivity_reference
            > config.boundary_standard_deviations * projected_scale
        )
        if np.array_equal(updated_mask, active_mask):
            break
        active_mask = updated_mask
    projected = project_factor_constraints(
        changes,
        jump_x=jump_x,
        active_mask=active_mask,
    )
    mean_change = np.mean(projected, axis=0)
    centered = projected - mean_change
    flat = centered.reshape(centered.shape[0], -1)
    _, singular_values, right_vectors = np.linalg.svd(flat, full_matrices=False)
    components = _orient_components(right_vectors)
    eigenvalues = np.square(singular_values) / (flat.shape[0] - 1)
    total_variance = float(np.sum(eigenvalues))
    average_variance = total_variance / flat.shape[1]
    shrunk_eigenvalues = (
        (1.0 - config.covariance_shrinkage) * eigenvalues
        + config.covariance_shrinkage * average_variance
    )
    positive_eigenvalues = eigenvalues > (
        np.finfo(np.float64).eps * max(1.0, float(eigenvalues[0]))
    )
    retained = min(config.num_factors, int(np.count_nonzero(positive_eigenvalues)))
    verify(
        retained > 0,
        lambda: "historical changes have no positive covariance eigenvalue",
    )
    retained_components = components[:retained].reshape(
        retained,
        tau.size,
        jump_x.size,
    )
    retained_components = project_factor_constraints(
        retained_components,
        jump_x=jump_x,
        active_mask=active_mask,
    )
    norms = np.linalg.norm(retained_components.reshape(retained, -1), axis=1)
    verify(
        np.all(norms > 0.0),
        lambda: f"constraint projection annihilated a PCA mode: norms={norms}",
    )
    retained_components /= norms[:, None, None]
    beta = (
        np.sqrt(shrunk_eigenvalues[:retained] / config.dt)[:, None, None]
        * retained_components
    )
    residual = constraint_residuals(beta, jump_x=jump_x)
    explained = (
        shrunk_eigenvalues[:retained] / total_variance
        if total_variance > 0.0
        else np.zeros(retained, dtype=np.float64)
    )
    return DynamicsFit(
        tau=tau,
        jump_x=jump_x,
        beta=beta,
        eigenvalues_one_session=shrunk_eigenvalues[:retained],
        explained_variance_ratio=explained,
        mean_change=mean_change,
        observations=int(changes.shape[0]),
        maximum_constraint_residual=float(np.max(np.abs(residual))),
        covariance_shrinkage=config.covariance_shrinkage,
        dt=config.dt,
        active_mask=active_mask,
    )
