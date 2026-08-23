"""Weekly P spot/front-state copula for the joint hierarchical FLEX model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import torch

from quantdata.dysanos.flex.model import JointHierarchicalDynamicsModel
from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs


@dataclass(frozen=True)
class JointSpotCorrelationFit:
    """P regression of weekly log spot on whitened front-state innovations."""

    weekly_drift: float
    front_beta: np.ndarray
    orthogonal_std: float
    spot_innovation_std: float
    model_explained_fraction: float
    empirical_r_squared: float
    num_weekly_transitions: int

    def __post_init__(self) -> None:
        if (
            self.front_beta.ndim != 1
            or not np.all(np.isfinite(self.front_beta))
            or not math.isfinite(self.weekly_drift)
            or not math.isfinite(self.orthogonal_std)
            or self.orthogonal_std <= 0.0
            or not math.isfinite(self.spot_innovation_std)
            or self.spot_innovation_std <= 0.0
            or not 0.0 <= self.model_explained_fraction < 1.0
            or self.num_weekly_transitions <= 0
        ):
            raise ValueError("invalid joint weekly P spot-correlation fit")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe immutable research record."""
        return {
            "weekly_drift": self.weekly_drift,
            "front_beta": self.front_beta.tolist(),
            "front_beta_norm": float(np.linalg.norm(self.front_beta)),
            "orthogonal_std": self.orthogonal_std,
            "spot_innovation_std": self.spot_innovation_std,
            "model_explained_fraction": self.model_explained_fraction,
            "empirical_r_squared": self.empirical_r_squared,
            "num_weekly_transitions": self.num_weekly_transitions,
            "spot_rank_front_correlations": (
                self.front_beta / self.spot_innovation_std
            ).tolist(),
        }


def _p_innovations(
    model: JointHierarchicalDynamicsModel,
    states: np.ndarray,
    dates: tuple[date, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weekly pair indices and ordered standard-normal P innovations."""
    if states.shape != (len(dates), model.config.state_size):
        raise ValueError("state history does not match joint P model geometry")
    origin, endpoint = exact_weekly_pairs(dates, weeks=1)
    if len(origin) == 0:
        raise ValueError("spot-correlation fit requires weekly state transitions")
    with torch.no_grad():
        transition = model.weekly_transition().detach().cpu().double().numpy()
        equilibrium = model.equilibrium_mean.detach().cpu().double().numpy()
        factor = model.innovation_cholesky().detach().cpu().double().numpy()
    prediction = equilibrium + (states[origin] - equilibrium) @ transition
    residual = states[endpoint] - prediction
    standard_normals = np.linalg.solve(factor, residual.T).T
    if not np.all(np.isfinite(standard_normals)):
        raise ValueError("non-finite whitened state innovations")
    return origin, endpoint, standard_normals


def fit_joint_spot_correlation_under_p(
    model: JointHierarchicalDynamicsModel,
    states: np.ndarray,
    log_spot: np.ndarray,
    dates: tuple[date, ...],
) -> JointSpotCorrelationFit:
    """Fit weekly P spot correlation before applying any implied marginal."""
    if log_spot.shape != (len(dates),) or not np.all(np.isfinite(log_spot)):
        raise ValueError("log-spot history does not match P training dates")
    origin, endpoint, standard_normals = _p_innovations(model, states, dates)
    front_width = model.config.state_sizes_by_expiry[0]
    front = standard_normals[:, :front_width]
    weekly_return = log_spot[endpoint] - log_spot[origin]
    design = np.concatenate(
        (np.ones((len(front), 1), dtype=np.float64), front),
        axis=1,
    )
    coefficients = np.linalg.lstsq(design, weekly_return, rcond=None)[0]
    prediction = design @ coefficients
    residual = weekly_return - prediction
    orthogonal_std = math.sqrt(float(np.mean(np.square(residual))))
    variance = float(np.var(weekly_return))
    empirical_r_squared = (
        1.0 - float(np.mean(np.square(residual))) / variance
        if variance > np.finfo(float).tiny
        else 0.0
    )
    beta = coefficients[1:]
    beta_variance = float(beta @ beta)
    spot_variance = beta_variance + orthogonal_std**2
    if orthogonal_std <= math.sqrt(np.finfo(float).tiny) or spot_variance <= 0.0:
        raise ValueError("weekly P spot regression is singular")
    return JointSpotCorrelationFit(
        weekly_drift=float(coefficients[0]),
        front_beta=beta,
        orthogonal_std=orthogonal_std,
        spot_innovation_std=math.sqrt(spot_variance),
        model_explained_fraction=beta_variance / spot_variance,
        empirical_r_squared=empirical_r_squared,
        num_weekly_transitions=len(origin),
    )


def evaluate_joint_spot_correlation_under_p(
    fit: JointSpotCorrelationFit,
    model: JointHierarchicalDynamicsModel,
    states: np.ndarray,
    log_spot: np.ndarray,
    dates: tuple[date, ...],
) -> dict[str, float | int]:
    """Evaluate the frozen P copula regression on another dated state path."""
    if log_spot.shape != (len(dates),) or not np.all(np.isfinite(log_spot)):
        raise ValueError("log-spot evaluation history does not match dates")
    origin, endpoint, standard_normals = _p_innovations(model, states, dates)
    front = standard_normals[:, : len(fit.front_beta)]
    realized = log_spot[endpoint] - log_spot[origin]
    prediction = fit.weekly_drift + front @ fit.front_beta
    error = realized - prediction
    variance = float(np.var(realized))
    r_squared = (
        1.0 - float(np.mean(np.square(error))) / variance
        if variance > np.finfo(float).tiny
        else 0.0
    )
    standardized_error = error / fit.orthogonal_std
    correlation = (
        float(np.corrcoef(realized, prediction)[0, 1])
        if np.std(realized) > 0.0 and np.std(prediction) > 0.0
        else 0.0
    )
    return {
        "num_weekly_transitions": len(origin),
        "weekly_return_mean": float(np.mean(realized)),
        "weekly_return_std": float(np.std(realized)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "r_squared": r_squared,
        "realized_prediction_correlation": correlation,
        "standardized_residual_mean": float(np.mean(standardized_error)),
        "standardized_residual_std": float(np.std(standardized_error)),
    }


__all__ = [
    "JointSpotCorrelationFit",
    "evaluate_joint_spot_correlation_under_p",
    "fit_joint_spot_correlation_under_p",
]
