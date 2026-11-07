"""Causal PCA and mean-reverting AR(1) for daily floating DLV coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from scipy.optimize import linear_sum_assignment


@version("0.2")
@dataclass(frozen=True, kw_only=True)
class DynamicsConfig:
    window_transitions: int = 252
    rank: int = 8
    max_spectral_radius: float = 0.999
    regression_ridge: float = 1e-6
    spot_orthogonal: bool = False

    def __post_init__(self) -> None:
        verify_inp(self.window_transitions > self.rank + 2 and self.rank > 0,
                   lambda: "The dynamics window must exceed the factor rank")
        verify_inp(0 < self.max_spectral_radius < 1 and self.regression_ridge > 0,
                   lambda: "AR(1) stability limit must be in (0,1) and ridge positive")


@dataclass(frozen=True, kw_only=True)
class FittedDynamics:
    as_of_date: date
    training_start: date
    training_end: date
    initialisation_date: date | None
    state_mean: np.ndarray
    directions: np.ndarray
    daily_factor_scales: np.ndarray
    explained_variance_fraction: np.ndarray
    ar_matrix: np.ndarray
    ar_intercept: np.ndarray
    equilibrium_scores: np.ndarray
    current_scores: np.ndarray
    residuals: np.ndarray
    joint_innovation_covariance: np.ndarray
    spot_state_loading: np.ndarray
    unconstrained_spectral_radius: float
    constrained_spectral_radius: float
    training_rmse: float
    persistence_rmse: float
    spot_conditioned_training_rmse: float
    spot_conditioned_persistence_rmse: float

    def forecast_state_change(self, current_state: np.ndarray) -> np.ndarray:
        scores = (current_state - self.state_mean) @ self.directions
        return self.directions @ (self.ar_intercept + self.ar_matrix @ scores - scores)

    def empirical_innovation_draws(self, count: int, *, rng: np.random.Generator) -> np.ndarray:
        """Draw centered, jointly paired spot/factor innovations from this window."""
        return self.residuals[rng.integers(0, len(self.residuals), size=count)]


def fit_dynamics(*, dates: tuple[date, ...], states: np.ndarray,
                 spot_returns_into_date: np.ndarray, session_indices: np.ndarray,
                 as_of_date: date, config: DynamicsConfig,
                 previous: FittedDynamics | None = None) -> FittedDynamics:
    """Fit the latest exact one-session pairs ending no later than as_of_date."""
    if (states.ndim != 2 or len(states) != len(dates)
            or spot_returns_into_date.shape != (len(dates),)
            or session_indices.shape != (len(dates),)
            or not np.isfinite(states).all()
            or any(left >= right for left, right in zip(dates, dates[1:]))):
        raise ValueError("Dates, states, spot returns and session indices must align")
    if previous is not None and previous.as_of_date >= as_of_date:
        raise ValueError("Prior factor orientation must come from an earlier date")
    eligible = np.array([index for index in range(1, len(dates))
                         if dates[index] <= as_of_date and session_indices[index] - session_indices[index - 1] == 1])
    selected = eligible[-config.window_transitions:]
    if len(selected) != config.window_transitions:
        raise ValueError("Insufficient exact one-session transitions for the rolling window")
    spot = spot_returns_into_date[selected]
    if not np.isfinite(spot).all():
        raise ValueError("The selected normalized spot returns must be finite")
    before = states[selected - 1]
    after = states[selected]
    changes = after - before
    centered_changes = changes - changes.mean(axis=0)
    centered_spot = spot - spot.mean()
    spot_loading = np.zeros(states.shape[1])
    if config.spot_orthogonal:
        spot_energy = float(centered_spot @ centered_spot)
        if spot_energy <= 0:
            raise ValueError("Spot variance vanishes in this rolling window")
        spot_loading = centered_spot @ centered_changes / spot_energy
        centered_changes -= centered_spot[:, None] * spot_loading

    gram = centered_changes @ centered_changes.T / (len(changes) - 1)
    values, vectors = np.linalg.eigh(gram)
    selected_order = np.argsort(values)[::-1][:config.rank]
    eigenvalues = values[selected_order]
    if np.any(eigenvalues <= np.finfo(float).eps * max(float(values.max()), 1.)):
        raise ValueError("The requested PCA rank exceeds the numerical data rank")
    directions = centered_changes.T @ vectors[:, selected_order]
    directions /= np.sqrt((len(changes) - 1) * eigenvalues)
    scales = np.sqrt(eigenvalues)
    explained = eigenvalues / np.trace(gram)

    if previous is not None:
        if previous.directions.shape != directions.shape:
            raise ValueError("Prior factor geometry is incompatible; do not silently cold-start")
        old_columns, new_columns = linear_sum_assignment(-np.abs(previous.directions.T @ directions))
        permutation = new_columns[np.argsort(old_columns)]
        directions = directions[:, permutation]
        scales = scales[permutation]
        explained = explained[permutation]
        signs = np.sign(np.sum(previous.directions * directions, axis=0))
    else:
        signs = np.sign(directions[np.argmax(np.abs(directions), axis=0), np.arange(config.rank)])
    directions *= np.where(signs == 0., 1., signs)

    state_mean = np.mean(np.vstack((before, after)), axis=0)
    before_scores = (before - state_mean) @ directions
    after_scores = (after - state_mean) @ directions
    before_mean = before_scores.mean(axis=0)
    after_mean = after_scores.mean(axis=0)
    centered_before = before_scores - before_mean
    centered_after = after_scores - after_mean
    projected_spot_loading = spot_loading @ directions
    conditioned_after = centered_after - centered_spot[:, None] * projected_spot_loading
    cross = centered_before.T @ centered_before
    ridge_scale = max(float(np.trace(cross) / config.rank), np.finfo(float).tiny)
    coefficient = np.linalg.solve(cross + config.regression_ridge * ridge_scale * np.eye(config.rank),
                                  centered_before.T @ conditioned_after)
    ar_matrix = coefficient.T
    unconstrained_radius = float(np.max(np.abs(np.linalg.eigvals(ar_matrix))))
    if unconstrained_radius > config.max_spectral_radius:
        ar_matrix *= config.max_spectral_radius / unconstrained_radius
    constrained_radius = float(np.max(np.abs(np.linalg.eigvals(ar_matrix))))
    intercept = after_mean - ar_matrix @ before_mean
    raw_residuals = after_scores - (before_scores @ ar_matrix.T + intercept)
    residuals = raw_residuals - centered_spot[:, None] * projected_spot_loading
    joint_residuals = np.c_[centered_spot, residuals]
    joint_residuals -= joint_residuals.mean(axis=0)
    covariance = joint_residuals.T @ joint_residuals / len(joint_residuals)
    latest_index = max(index for index, day in enumerate(dates) if day <= as_of_date)
    equilibrium = np.linalg.solve(np.eye(config.rank) - ar_matrix, intercept)
    return FittedDynamics(
        as_of_date=as_of_date, training_start=dates[int(selected[0]) - 1],
        training_end=dates[int(selected[-1])],
        initialisation_date=None if previous is None else previous.as_of_date,
        state_mean=state_mean, directions=directions, daily_factor_scales=scales,
        explained_variance_fraction=explained, ar_matrix=ar_matrix,
        ar_intercept=intercept, equilibrium_scores=equilibrium,
        current_scores=(states[latest_index] - state_mean) @ directions,
        residuals=joint_residuals, joint_innovation_covariance=covariance,
        spot_state_loading=spot_loading, unconstrained_spectral_radius=unconstrained_radius,
        constrained_spectral_radius=constrained_radius,
        training_rmse=float(np.sqrt(np.mean(raw_residuals**2))),
        persistence_rmse=float(np.sqrt(np.mean((after_scores - before_scores)**2))),
        spot_conditioned_training_rmse=float(np.sqrt(np.mean(residuals**2))),
        spot_conditioned_persistence_rmse=float(np.sqrt(np.mean(
            (after_scores - before_scores - centered_spot[:, None] * projected_spot_loading)**2))),
    )
