from __future__ import annotations

import torch

from quantdata.dysanos.flex.few_instrument_oce import (
    entropic_oce_values,
    fit_batched_entropic_oce,
)


def test_entropic_oce_recovers_gaussian_tilt_for_multiple_origins() -> None:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(915)
    half = torch.randn((2, 50_000, 3), dtype=dtype, generator=generator)
    innovations = torch.cat((half, -half), dim=1)
    drift = torch.tensor(
        ((0.08, -0.04, 0.02), (-0.03, 0.06, 0.01)), dtype=dtype
    )
    returns = innovations + drift[:, None, :]

    fit = fit_batched_entropic_oce(returns, mean_tolerance=2.0e-8)

    assert bool(torch.all(fit.converged))
    torch.testing.assert_close(fit.action, drift, atol=2.0e-3, rtol=0.0)
    assert float(torch.max(torch.abs(fit.weighted_mean))) < 2.0e-8
    torch.testing.assert_close(
        torch.mean(fit.weights, dim=1), torch.ones(2, dtype=dtype), atol=1.0e-12, rtol=0.0
    )


def test_entropic_values_do_not_center_the_return_drift() -> None:
    returns = torch.tensor(
        (((-1.0, 0.0), (1.0, 0.0), (0.5, 0.2), (-0.5, 0.2)),),
        dtype=torch.float64,
    )
    action = torch.zeros((1, 2), dtype=torch.float64)

    _, weights, weighted_mean = entropic_oce_values(returns, action)

    torch.testing.assert_close(weights, torch.ones_like(weights))
    torch.testing.assert_close(weighted_mean, torch.mean(returns, dim=1))
