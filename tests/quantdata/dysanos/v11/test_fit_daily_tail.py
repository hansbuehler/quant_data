"""Checks for the daily joint Tail-Diffusion residual transform."""

from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.v11.fit_daily_tail import joint_white_innovations


def test_joint_white_innovations_recovers_constructed_shocks() -> None:
    epsilon = np.array([[0.3, -0.5], [-0.8, 0.2]])
    eta = np.array([0.4, -0.7])
    factor = np.array([[1.2, 0.1], [0.2, 0.8]])
    gamma = np.array([0.2, -0.3])
    initial = np.array([0.1, -0.2])
    transition = np.array([[0.8, 0.1], [-0.2, 0.7]])
    intercept = np.array([0.05, -0.03])
    states = [initial]
    for shock in epsilon:
        states.append(intercept + states[-1] @ transition + shock @ factor.T)
    scores = epsilon @ gamma + np.sqrt(1.0 - gamma @ gamma) * eta
    white = joint_white_innovations(
        np.asarray(states),
        scores,
        state_mean=np.zeros(2),
        factor_basis=np.eye(2),
        factor_intercept=intercept,
        factor_transition=transition,
        innovation_factor=factor,
        spot_score_mean=0.0,
        spot_score_std=1.0,
        gamma=gamma,
    )
    np.testing.assert_allclose(white, np.column_stack((epsilon, eta)), atol=1e-14)


def test_joint_white_innovations_rejects_misaligned_dates() -> None:
    with pytest.raises(ValueError, match="adjacent sessions"):
        joint_white_innovations(
            np.zeros((3, 1)),
            np.zeros(3),
            state_mean=np.zeros(1),
            factor_basis=np.eye(1),
            factor_intercept=np.zeros(1),
            factor_transition=np.eye(1),
            innovation_factor=np.eye(1),
            spot_score_mean=0.0,
            spot_score_std=1.0,
            gamma=np.zeros(1),
        )
