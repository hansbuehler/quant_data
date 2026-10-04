from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantdata.dysanos.flex.run_joint_q2_full_cost_oce import (
    _convergence_checks,
    _joint_oce_probabilities,
    _paired_oce_standard_error,
)


def test_joint_probabilities_match_nested_detached_oce_objectives() -> None:
    q1_density = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float64)
    low_cost_pnl = np.array([0.2, -0.3, 0.1, 0.4], dtype=np.float64)
    full_cost_pnl = np.array([-0.1, 0.5, 0.2, -0.2], dtype=np.float64)

    result = _joint_oce_probabilities(
        q1_density=q1_density,
        low_cost_pnl=low_cost_pnl,
        full_cost_pnl=full_cost_pnl,
    )

    q1_probability = q1_density / np.sum(q1_density)
    q2_unnormalized = q1_probability * np.exp(-low_cost_pnl)
    q2_probability = q2_unnormalized / np.sum(q2_unnormalized)
    a3_unnormalized = q2_probability * np.exp(-full_cost_pnl)
    a3_probability = a3_unnormalized / np.sum(a3_unnormalized)
    expected_a2_oce = -np.log(np.sum(q1_probability * np.exp(-low_cost_pnl)))
    expected_a3_oce = -np.log(np.sum(q2_probability * np.exp(-full_cost_pnl)))

    assert result["a2_oce"] == pytest.approx(expected_a2_oce)
    assert result["a3_oce_under_q2"] == pytest.approx(expected_a3_oce)
    assert result["q2_probability"] == pytest.approx(q2_probability)
    assert result["a3_tilted_probability"] == pytest.approx(a3_probability)
    assert np.sum(result["q2_probability"]) == pytest.approx(1.0)
    assert np.sum(result["a3_tilted_probability"]) == pytest.approx(1.0)
    assert np.sum(
        q1_probability * np.exp(result["q2_log_relative_density"])
    ) == pytest.approx(1.0)


def test_detached_q2_density_does_not_add_an_a3_gradient_to_a2() -> None:
    q1_density = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float64)
    low_features = torch.tensor([0.2, -0.4, 0.7, 0.1], dtype=torch.float64)
    full_features = torch.tensor([-0.3, 0.6, 0.2, -0.5], dtype=torch.float64)
    a2 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    a3 = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
    result = _joint_oce_probabilities(
        q1_density=q1_density,
        low_cost_pnl=(a2.detach() * low_features).numpy(),
        full_cost_pnl=(a3.detach() * full_features).numpy(),
    )

    a2_loss = -torch.sum(
        torch.as_tensor(result["q2_probability"]) * a2 * low_features
    )
    a3_loss = -torch.sum(
        torch.as_tensor(result["a3_tilted_probability"]) * a3 * full_features
    )
    (a2_loss + a3_loss).backward()

    q1_probability = torch.as_tensor(q1_density / np.sum(q1_density))
    direct_a2 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    direct_a2_loss = torch.log(
        torch.sum(q1_probability * torch.exp(-direct_a2 * low_features))
    )
    direct_a2_loss.backward()
    detached_q2 = torch.as_tensor(result["q2_probability"])
    direct_a3 = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
    direct_a3_loss = torch.log(
        torch.sum(detached_q2 * torch.exp(-direct_a3 * full_features))
    )
    direct_a3_loss.backward()

    assert float(a2.grad) == pytest.approx(float(direct_a2.grad))
    assert float(a3.grad) == pytest.approx(float(direct_a3.grad))


def test_joint_convergence_requires_both_policies_and_density_stability() -> None:
    arguments = SimpleNamespace(
        full_gradient_rms_tolerance=1.0e-3,
        full_gradient_maximum_tolerance=2.0e-3,
        full_gradient_scale_foc_tolerance=3.0e-3,
        q2_log_density_rms_tolerance=4.0e-3,
        a3_gradient_rms_tolerance=1.0e-3,
        a3_gradient_maximum_tolerance=2.0e-3,
        a3_scale_foc_tolerance=3.0e-3,
    )
    record = {
        "a2_gradient_rms": 0.5e-3,
        "a2_gradient_maximum_absolute": 1.0e-3,
        "a2_tilted_expected_pnl": 2.0e-3,
        "q2_log_density_rms_change": 3.0e-3,
        "a3_enabled": True,
        "a3_gradient_rms": 0.5e-3,
        "a3_gradient_maximum_absolute": 1.0e-3,
        "a3_tilted_expected_pnl": 2.0e-3,
    }

    assert all(_convergence_checks(record, arguments).values())
    record["a3_enabled"] = False
    assert not all(_convergence_checks(record, arguments).values())
    record["a3_enabled"] = True
    record["q2_log_density_rms_change"] = 5.0e-3
    assert not all(_convergence_checks(record, arguments).values())


def test_paired_oce_error_uses_self_normalized_density_influence() -> None:
    probability = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    pnl = np.array([0.2, -0.1, -0.3, 0.4], dtype=np.float64)

    actual = _paired_oce_standard_error(
        q2_probability=probability,
        full_cost_pnl=pnl,
        antithetic_block_size=4,
    )

    weighted_exponential_mean = np.sum(probability * np.exp(-pnl))
    influence = (
        -len(probability) * probability * np.exp(-pnl) / weighted_exponential_mean
        + len(probability) * probability
    )
    paired = 0.5 * (influence[:2] + influence[2:])
    expected = np.std(paired, ddof=1) / math.sqrt(len(paired))
    assert actual == pytest.approx(expected)
