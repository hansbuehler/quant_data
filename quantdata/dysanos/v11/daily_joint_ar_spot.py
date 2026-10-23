"""Daily low-rank joint state/spot dynamics for DYSANOS 1.1.

The surface innovation factor basis explicitly contains the direction most
correlated with the front-SANOS spot score.  This prevents the spot/surface
link from being fitted only after the surface dynamics have been fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class DailyJointArSpotFit:
    """Stable low-rank daily transition and joint front-spot score law."""

    state_mean: np.ndarray
    factor_basis: np.ndarray
    factor_intercept: np.ndarray
    factor_transition: np.ndarray
    innovation_factor: np.ndarray
    spot_score_mean: float
    spot_score_std: float
    gamma: np.ndarray
    spectral_radius: float
    spot_direction_fraction: float

    @property
    def rank(self) -> int:
        return int(self.factor_basis.shape[1])


def _stable_transition(transition: np.ndarray, maximum_radius: float) -> tuple[np.ndarray, float]:
    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if radius >= maximum_radius:
        transition = transition * (maximum_radius / radius)
        radius = maximum_radius
    return transition, radius


def _spot_aware_factor_basis(
    increments: np.ndarray,
    spot_scores: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, float]:
    """Return an orthonormal increment basis with spot direction first."""
    covariance = np.cov(increments, rowvar=False, ddof=1)
    spot_direction = increments.T @ spot_scores / (len(spot_scores) - 1)
    spot_norm = float(np.linalg.norm(spot_direction))
    _, _, vectors_t = np.linalg.svd(covariance, full_matrices=False)
    principal = vectors_t.T

    if spot_norm <= 1e-14:
        candidates = principal[:, :rank]
    else:
        candidates = np.column_stack((spot_direction / spot_norm, principal[:, :rank]))
    basis, _ = np.linalg.qr(candidates, mode="reduced")
    if basis.shape[1] < rank:
        raise ValueError(f"rank {rank} exceeds available state dimension {basis.shape[1]}")

    total_increment_variance = max(float(np.trace(covariance)), 1e-14)
    return basis[:, :rank], min(1.0, spot_norm**2 / total_increment_variance)


def fit_daily_joint_ar_spot(
    states: np.ndarray,
    spot_scores: np.ndarray,
    *,
    rank: int = 16,
    ridge: float = 1e-4,
    covariance_shrinkage: float = 0.05,
    maximum_spectral_radius: float = 0.995,
    transition_structure: str = "dense",
) -> DailyJointArSpotFit:
    """Fit a daily stable low-rank state transition jointly with spot score.

    states holds consecutive daily affine DYSANOS states with shape (n + 1, d).
    spot_scores holds the current-front-law Gaussian scores for n daily returns.
    """
    states = np.asarray(states, dtype=np.float64)
    spot_scores = np.asarray(spot_scores, dtype=np.float64)
    if states.ndim != 2 or spot_scores.ndim != 1 or len(states) != len(spot_scores) + 1:
        raise ValueError("states must be (n + 1, d) and spot_scores must be (n,)")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(spot_scores)):
        raise ValueError("states and spot_scores must be finite")
    if not 1 <= rank <= states.shape[1]:
        raise ValueError(f"rank must be in [1, {states.shape[1]}], got {rank}")
    if len(spot_scores) <= rank + 5:
        raise ValueError("insufficient daily observations for requested rank")
    if not 0.0 <= covariance_shrinkage < 1.0:
        raise ValueError("covariance_shrinkage must be in [0, 1)")
    if transition_structure not in {"dense", "diagonal"}:
        raise ValueError("transition_structure must be dense or diagonal")

    state_mean = states[:-1].mean(axis=0)
    increments = np.diff(states, axis=0)
    score_mean = float(spot_scores.mean())
    score_std = max(float(spot_scores.std(ddof=1)), 1e-12)
    standardized_scores = (spot_scores - score_mean) / score_std
    basis, spot_direction_fraction = _spot_aware_factor_basis(increments, standardized_scores, rank)

    factors_now = (states[:-1] - state_mean) @ basis
    factors_next = (states[1:] - state_mean) @ basis
    if transition_structure == "dense":
        design = np.column_stack((np.ones(len(factors_now)), factors_now))
        penalty = np.eye(rank + 1) * ridge
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ factors_next
        )
        intercept = coefficients[0]
        transition, spectral_radius = _stable_transition(
            coefficients[1:], maximum_spectral_radius
        )
    else:
        centered_now = factors_now - np.mean(factors_now, axis=0)
        centered_next = factors_next - np.mean(factors_next, axis=0)
        diagonal = np.sum(centered_now * centered_next, axis=0) / (
            np.sum(centered_now * centered_now, axis=0) + ridge
        )
        diagonal = np.clip(
            diagonal, -maximum_spectral_radius, maximum_spectral_radius
        )
        transition = np.diag(diagonal)
        spectral_radius = float(np.max(np.abs(diagonal)))
        intercept = np.mean(factors_next, axis=0) - np.mean(
            factors_now, axis=0
        ) * diagonal
    residuals = factors_next - intercept - factors_now @ transition

    covariance = np.cov(residuals, rowvar=False, ddof=1)
    covariance = (1.0 - covariance_shrinkage) * covariance + covariance_shrinkage * np.diag(np.diag(covariance))
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-10 * max(float(eigenvalues.max()), 1.0))
    innovation_factor = eigenvectors * np.sqrt(eigenvalues)[None, :]

    spot_covariance = residuals.T @ standardized_scores / (len(standardized_scores) - 1)
    gamma = np.linalg.solve(innovation_factor, spot_covariance)
    gamma_norm = float(np.linalg.norm(gamma))
    if gamma_norm >= 0.999:
        gamma *= 0.999 / gamma_norm

    return DailyJointArSpotFit(
        state_mean=state_mean,
        factor_basis=basis,
        factor_intercept=intercept,
        factor_transition=transition,
        innovation_factor=innovation_factor,
        spot_score_mean=score_mean,
        spot_score_std=score_std,
        gamma=gamma,
        spectral_radius=spectral_radius,
        spot_direction_fraction=spot_direction_fraction,
    )


def sample_daily_joint_ar_spot(
    fit: DailyJointArSpotFit,
    states: np.ndarray,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate antithetic daily next states and front-SANOS Gaussian scores."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    if states.ndim != 2 or states.shape[1] != fit.factor_basis.shape[0]:
        raise ValueError("states has incompatible shape")
    if count <= 0 or count % 2:
        raise ValueError("count must be a positive even number for antithetic sampling")

    generator = np.random.default_rng(seed)
    half_count = count // 2
    epsilon_half = generator.standard_normal((len(states), half_count, fit.rank))
    eta_half = generator.standard_normal((len(states), half_count))
    epsilon = np.concatenate((epsilon_half, -epsilon_half), axis=1)
    eta = np.concatenate((eta_half, -eta_half), axis=1)

    centered = states - fit.state_mean
    conditional_factor_mean = fit.factor_intercept + (centered @ fit.factor_basis) @ fit.factor_transition
    state_mean = fit.state_mean + conditional_factor_mean @ fit.factor_basis.T
    factor_shock = epsilon @ fit.innovation_factor.T
    next_states = state_mean[:, None, :] + factor_shock @ fit.factor_basis.T

    residual_scale = np.sqrt(max(1.0 - float(fit.gamma @ fit.gamma), 1e-12))
    standardized_scores = epsilon @ fit.gamma + residual_scale * eta
    spot_scores = fit.spot_score_mean + fit.spot_score_std * standardized_scores
    return next_states, spot_scores
