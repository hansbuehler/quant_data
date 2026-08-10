from __future__ import annotations

import numpy as np
import pytest
import torch

from quantdata.dysanos.grid.gendysanos.run_incremental_candidate_adversarial_replay import (
    _raw_state_path_from_shocks,
    first_loss_radius_on_ray,
    project_rows_to_l2_balls_,
)
from quantdata.dysanos.grid.gendysanos.run_incremental_candidate_replay import (
    recover_candidate_state,
    select_strongest_candidates,
    summarize_conditional_pnl,
)
from quantdata.dysanos.grid.gendysanos.run_incremental_pca_candidate_replay import (
    select_available_candidates,
)


def test_select_strongest_candidates_uses_weaker_adjusted_mean_bound() -> None:
    candidates = [
        {
            "adjusted_direct_P_mean_lower": 3.0,
            "adjusted_weighted_P_mean_lower": 1.0,
        },
        {
            "adjusted_direct_P_mean_lower": 2.0,
            "adjusted_weighted_P_mean_lower": 2.0,
        },
    ]
    report = {
        "candidate_certification": {
            "local_conditional_baskets": {"surviving": candidates}
        }
    }

    assert select_strongest_candidates(report, count=1) == [candidates[1]]


def test_select_available_candidates_returns_all_when_fewer_than_limit() -> None:
    candidates = [
        {
            "adjusted_direct_P_mean_lower": 3.0,
            "adjusted_weighted_P_mean_lower": 1.0,
        },
        {
            "adjusted_direct_P_mean_lower": 2.0,
            "adjusted_weighted_P_mean_lower": 2.0,
        },
    ]
    report = {
        "candidate_certification": {
            "local_conditional_baskets": {"surviving": candidates}
        }
    }

    assert select_available_candidates(report, maximum_count=10) == [
        candidates[1],
        candidates[0],
    ]


def test_recover_candidate_state_inverts_training_normalization() -> None:
    report = {
        "conditional": {
            "times": [
                {
                    "state_normalization": {
                        "reference_only_mean": [10.0, -2.0],
                        "reference_only_population_scale": [2.0, 4.0],
                    }
                }
            ]
        }
    }
    candidate = {
        "trading_time_index": 0,
        "state_region": {"center": [0.5, -0.25]},
    }

    assert recover_candidate_state(report, candidate) == pytest.approx([11.0, -3.0])


def test_summarize_conditional_pnl_separates_economic_and_band_losses() -> None:
    result = summarize_conditional_pnl(
        np.asarray([-2.0e-5, -5.0e-7, 3.0e-5]),
        np.asarray([1.0e-6, 1.0e-6, 1.0e-6]),
    )

    assert result["economic_loss_count"] == 2
    assert result["loss_count_beyond_float32_band"] == 1
    assert result["any_path_loses_money"] is True
    assert result["any_path_loses_beyond_float32_band"] is True


def test_project_rows_to_l2_balls_respects_each_radius() -> None:
    values = torch.tensor([[3.0, 4.0], [0.3, 0.4]])
    radii = torch.tensor([2.0, 1.0])

    project_rows_to_l2_balls_(values, radii)

    assert torch.linalg.vector_norm(values, dim=1) == pytest.approx([2.0, 0.5])


def test_raw_state_path_from_shocks_applies_affine_transition() -> None:
    path = _raw_state_path_from_shocks(
        torch.tensor([[[2.0], [-1.0]]]),
        initial_state=torch.tensor([10.0, 2.0]),
        center=torch.tensor([8.0, 0.0]),
        scale=torch.tensor([2.0, 2.0]),
        transition=torch.eye(2),
        intercept=torch.tensor([[1.0, -1.0]]),
        factor=torch.tensor([[0.5], [1.0]]),
    )

    assert torch.allclose(
        path,
        torch.tensor([[[10.0, 2.0], [14.0, 4.0], [15.0, 0.0]]]),
    )


def test_first_loss_radius_on_ray_finds_sign_boundary() -> None:
    def pnl(shocks: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.linalg.vector_norm(shocks.flatten(start_dim=1), dim=1)

    radius, value = first_loss_radius_on_ray(
        pnl, torch.tensor([[[2.0, 0.0]]]), iterations=40
    )

    assert radius == pytest.approx(1.0, abs=1.0e-6)
    assert value < 0.0
