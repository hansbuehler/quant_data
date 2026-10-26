from __future__ import annotations

import numpy as np

from quantdata.gromm2.vix_comparison import (
    marginal_variance_from_density,
    model_free_variance_from_density,
)


def test_model_free_variance_is_zero_for_a_deterministic_forward() -> None:
    variance = model_free_variance_from_density(
        support=np.asarray([0.5, 1.0, 1.5]),
        density=np.asarray([0.0, 1.0, 0.0]),
        expiry_y=30.0 / 365.0,
        background_annual_volatility=0.0,
        num_strikes_per_side=101,
    )
    assert variance == 0.0


def test_marginal_variance_is_the_background_for_a_deterministic_forward() -> None:
    annual_variance = marginal_variance_from_density(
        support=np.asarray([0.5, 1.0, 1.5]),
        density=np.asarray([0.0, 1.0, 0.0]),
        expiry_y=30.0 / 365.0,
        background_annual_volatility=0.05,
    )
    assert annual_variance == (np.exp(0.05**2 * 30.0 / 365.0) - 1.0) / (30.0 / 365.0)
