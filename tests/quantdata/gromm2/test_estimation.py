from __future__ import annotations

import numpy as np

from quantdata.gromm2.estimation import fit_constant_diffusion


def test_constant_diffusion_uses_ar_innovations() -> None:
    generator = np.random.default_rng(17)
    observations = 240
    shape = (3, 20)
    spot_scores = generator.normal(size=observations)
    second_scores = generator.normal(size=observations)
    loading0 = np.linspace(-0.015, 0.020, np.prod(shape)).reshape(shape)
    loading1 = np.linspace(0.010, -0.012, np.prod(shape)).reshape(shape)
    slope = np.full(shape, 0.92)
    intercept = np.full(shape, -0.01)
    states = np.empty((observations + 1,) + shape)
    states[0] = -3.0
    for index in range(observations):
        innovation = loading0 * spot_scores[index] + loading1 * second_scores[index]
        states[index + 1] = intercept + slope * states[index] + innovation
    panel = {
        "log_dlvsq_origins": states[:-1],
        "log_dlvsq_changes": np.diff(states, axis=0),
        "spot_scores": spot_scores,
        "origin_indices": np.arange(observations, dtype=np.int64),
        "nms_nodes": np.linspace(-3.0, 2.0, shape[-1]),
    }
    estimate = fit_constant_diffusion(
        panel,
        factor_count=2,
        level_rank=2,
        smoothing_expiry_sigma=0.0,
        smoothing_strike_sigma=0.0,
    )
    assert np.allclose(estimate.physical_ar_slope, slope, atol=0.03)
    assert np.all(np.isfinite(estimate.log_dlvsq_loadings))
    assert estimate.explained_variance_ratio > 0.99
    fitted_spot = estimate.log_dlvsq_loadings[..., 0] / np.sqrt(252.0)
    core = (panel["nms_nodes"] >= -2.5) & (panel["nms_nodes"] <= 1.5)
    assert np.corrcoef(fitted_spot[:, core].reshape(-1), loading0[:, core].reshape(-1))[
        0, 1
    ] > 0.99
