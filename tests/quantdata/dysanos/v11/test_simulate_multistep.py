"""Path statistics respect path boundaries and compounded state covariance."""

import numpy as np
import torch

from quantdata.dysanos.v11.simulate_multistep import (
    _conditional_scale_mixture_normal_score,
    _path_radius_tails,
    _pooled_path_lag_correlations,
)
from quantdata.dysanos.v11.tail_copula import normal_score_from_scale_mixture


def test_path_autocorrelation_never_joins_different_paths() -> None:
    series = np.stack((np.arange(25.0), np.arange(25.0) + 1000.0))
    result = _pooled_path_lag_correlations(series)
    assert result[1][0] == 1.0
    assert result[20][0] == 1.0


def test_path_radius_compounds_daily_variance() -> None:
    states = np.arange(31.0)[None, :, None] * 0.02
    panels = _path_radius_tails(
        states,
        equilibrium_mean=np.zeros(1),
        transition=np.array([[0.9]]),
        innovation_covariance=np.array([[0.1]]),
        deterministic_shift=np.zeros(1),
    )
    assert panels[1]["observations"] == 26
    assert panels[4]["observations"] == 11
    assert np.isfinite(panels[4]["p99"])


def test_conditional_tail_spot_score_matches_reference_quadrature() -> None:
    nodes, weights = np.polynomial.hermite.hermgauss(48)
    raw = np.array([-2.0, 0.0, 1.0, 2.5])
    conditional_mean = np.array([-0.2, 0.1, 0.3, -0.1])
    result = _conditional_scale_mixture_normal_score(
        torch.tensor(raw, dtype=torch.float64),
        conditional_log_mean=torch.tensor(conditional_mean, dtype=torch.float64),
        conditional_log_variance=0.4,
        stationary_log_variance=0.7,
        unconditional_scale=1.1,
        nodes=torch.tensor(nodes, dtype=torch.float64),
        weights=torch.tensor(weights, dtype=torch.float64),
    )
    expected = normal_score_from_scale_mixture(
        raw,
        stationary_log_variance=0.7,
        unconditional_scale=1.1,
        conditional_log_variance_mean=conditional_mean,
        conditional_log_variance_variance=0.4,
    )
    np.testing.assert_allclose(result.numpy(), expected, rtol=1.0e-12)
