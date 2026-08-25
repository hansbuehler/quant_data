from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantdata.dysanos.flex.analyze_state_allocation import (
    _balanced_allocation,
    _rank_at_fraction,
    _triangular_rrr_residual_fraction,
)
from quantdata.dysanos.flex.config import ForwardMuFlexGeometryConfig
from quantdata.dysanos.flex.evaluate_state_allocations import (
    _decoded_checkpoint_raw_states,
    _nearest_model_grid_statistics,
)


def test_rank_at_fraction_uses_singular_value_energy() -> None:
    singular_values = torch.tensor([3.0, 1.0], dtype=torch.float64)

    assert _rank_at_fraction(singular_values, 0.89) == 1
    assert _rank_at_fraction(singular_values, 0.91) == 2


def test_balanced_allocation_preserves_the_total_budget() -> None:
    allocation = _balanced_allocation(
        20,
        total_size=140,
        num_expiries=14,
    )

    assert allocation == (20, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9)
    assert sum(allocation) == 140


def test_triangular_rrr_recovers_modes_at_their_first_visible_expiry() -> None:
    # The first latent mode is visible in grid 0. The second first appears in
    # grid 1, so one state in each block reconstructs this surface exactly.
    coefficients = torch.tensor(
        [
            [3.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    residual_fraction, selected = _triangular_rrr_residual_fraction(
        coefficients,
        allocation=(1, 1),
        block_width=2,
    )

    assert selected == (1, 1)
    assert residual_fraction == pytest.approx(0.0, abs=1.0e-14)


def test_decoded_checkpoint_applies_the_triangular_weight_mask() -> None:
    checkpoint = {
        "model_state_dict": {
            "daily_state_embedding.weight": torch.tensor([[2.0, 3.0]]),
            "raw_decoder.I0.weight": torch.tensor([[5.0, 7.0]]),
            "raw_decoder.I0.bias": torch.tensor([11.0]),
            "decoder_weight_mask": torch.tensor([[1.0, 0.0]]),
        }
    }

    raw = _decoded_checkpoint_raw_states(checkpoint)

    assert raw.item() == pytest.approx(21.0)


def test_nearest_model_grid_statistics_use_quote_half_spreads() -> None:
    day = SimpleNamespace(
        bids=np.asarray([0.40, 0.65]),
        asks=np.asarray([0.60, 0.75]),
        quote_business_dtes=np.asarray([5.0, 10.0]),
    )

    rows = _nearest_model_grid_statistics(
        (np.asarray([0.50, 0.60]),),
        [day],
        geometry=ForwardMuFlexGeometryConfig(),
    )

    assert rows[0]["num_options"] == 1
    assert rows[0]["percent_zero"] == pytest.approx(100.0)
    assert rows[1]["num_options"] == 1
    assert rows[1]["positive_mean"] == pytest.approx(1.0)
