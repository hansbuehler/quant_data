"""Unit tests for the zero-Black-backbone diagnostic."""

from types import SimpleNamespace

import numpy as np
import torch

from quantdata.dysanos.v11.run_dlv_only_leverage import (
    _cross_tenor_correlation,
    _dlv_only_front_gross,
)


def test_dlv_only_front_quantile_is_discrete_and_mean_normalized() -> None:
    surface = SimpleNamespace(
        support_strikes=torch.tensor([[[0.5, 1.0, 1.5]]], dtype=torch.float64),
        densities=torch.tensor([[[0.25, 0.5, 0.25]]], dtype=torch.float64),
    )
    result = _dlv_only_front_gross(surface, np.array([-10.0, 0.0, 10.0]))
    np.testing.assert_allclose(result, np.array([0.5, 1.0, 1.5]))


def test_cross_tenor_correlation_has_unit_diagonal() -> None:
    changes = np.array(
        [[-2.0, -4.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 4.0]]
    )
    result = _cross_tenor_correlation(changes)
    np.testing.assert_allclose(result, np.ones((2, 2)))
