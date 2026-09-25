"""Tests for two-eta statistical-model figure diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.flex.plot_statistical_model_diagnostics import (
    extract_tail_probabilities,
    select_pca_expiries,
    stationary_state_covariance,
)


def test_select_pca_expiries_spans_seven_distinct_model_nodes() -> None:
    model_expiries = np.arange(5.0, 75.0, 5.0)

    selected = select_pca_expiries(model_expiries)

    assert len(selected) == 7
    assert selected[0] == model_expiries[0]
    assert selected[-1] == model_expiries[-1]
    assert np.all(np.diff(selected) > 0.0)


def test_stationary_state_covariance_uses_row_state_orientation() -> None:
    transition = np.asarray(((0.8, 0.1), (0.0, 0.6)))
    innovation = np.asarray(((0.3, 0.04), (0.04, 0.2)))

    invariant = stationary_state_covariance(transition, innovation)

    np.testing.assert_allclose(
        invariant,
        transition.T @ invariant @ transition + innovation,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.min(np.linalg.eigvalsh(invariant)) > 0.0


def test_select_pca_expiries_rejects_too_few_nodes() -> None:
    with pytest.raises(ValueError, match="seven"):
        select_pca_expiries(np.arange(1.0, 7.0))


def test_extract_tail_probabilities_aligns_market_gaussian_and_replay() -> None:
    report = {
        "schema": "flex-score-driven-student-t-v1",
        "market_tail_diagnostics": {
            f"{week}w": {
                "2025_exceedance_of_pre_2025_q95": 0.04 + 0.001 * week,
                "gaussian_exceedance_of_pre_2025_q95": 0.001 * week,
            }
            for week in (1, 2, 3)
        },
        "simulation": {
            "iid_student_t": {
                "results": {
                    f"{week}w": {"exceedance_probability": 0.02 + 0.001 * week}
                    for week in (1, 2, 3)
                }
            },
            "dynamic_student_t": {
                "results": {
                    f"{week}w": {"exceedance_probability": 0.025 + 0.001 * week}
                    for week in (1, 2, 3)
                }
            },
            "historical_replay": {
                "results": {
                    f"{week}w": {"exceedance_probability": 0.03 + 0.001 * week}
                    for week in (1, 2, 3)
                }
            }
        },
    }
    diffusion_report = {
        "schema": "flex-log-variance-diffusion-v1",
        "simulation": {
            "results": {
                f"{week}w": {"exceedance_probability": 0.035 + 0.001 * week}
                for week in (1, 2, 3)
            }
        },
    }

    result = extract_tail_probabilities(report, diffusion_report)

    assert result["market_2025"] == pytest.approx([0.041, 0.042, 0.043])
    assert result["gaussian"] == pytest.approx([0.001, 0.002, 0.003])
    assert result["iid_student_t"] == pytest.approx([0.021, 0.022, 0.023])
    assert result["dynamic_student_t"] == pytest.approx([0.026, 0.027, 0.028])
    assert result["stochastic_volatility"] == pytest.approx([0.036, 0.037, 0.038])
    assert result["historical_replay"] == pytest.approx([0.031, 0.032, 0.033])
