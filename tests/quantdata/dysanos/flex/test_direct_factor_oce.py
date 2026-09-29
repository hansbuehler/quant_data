from __future__ import annotations

import torch

from quantdata.dysanos.flex.direct_factor_oce import (
    antithetic_gaussian_factor_returns,
    exponential_oce_values,
    fit_exponential_oce_action,
)


def test_direct_factor_oce_recovers_gaussian_market_price_of_risk() -> None:
    dtype = torch.float64
    target = torch.tensor((0.4, -0.2, 0.1), dtype=dtype)
    generator = torch.Generator().manual_seed(71)
    returns = antithetic_gaussian_factor_returns(
        num_paths=40_000,
        factor_drift=target,
        time_increment_y=0.02,
        generator=generator,
    )

    fit = fit_exponential_oce_action(
        returns,
        return_variance=0.02,
        maximum_iterations=30,
        mean_tolerance=1.0e-10,
    )

    assert fit.converged
    torch.testing.assert_close(fit.action, target, atol=0.025, rtol=0.0)
    assert float(torch.max(torch.abs(fit.values.weighted_return_mean))) < 1.0e-10


def test_exponential_density_has_unit_mass() -> None:
    returns = torch.tensor(((0.1, -0.2), (-0.1, 0.2)), dtype=torch.float64)
    action = torch.tensor((0.3, -0.4), dtype=torch.float64)

    values = exponential_oce_values(returns, action)

    torch.testing.assert_close(torch.mean(values.weights), torch.tensor(1.0, dtype=torch.float64))
