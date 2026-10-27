from __future__ import annotations

import numpy as np
import pytest

from quantdata.sdlv.daily.vix_comparison import log_contract_total_variance
from quantdata.sdlv.daily.vix_options import log_variance_level, payoff_statistics


def test_log_variance_matches_existing_formula_and_is_scale_invariant() -> None:
    support = np.array([.8, 1.2])
    density = np.array([[.5, .5]])
    business, calendar = 22.8 / 252, (30 + 390 / 1440) / 365
    expected = 100 * np.sqrt(log_contract_total_variance(support, density[0], business_time_years=business) / calendar)
    level = log_variance_level(support, density, business_years=business, calendar_years=calendar)
    np.testing.assert_allclose(level, expected)
    np.testing.assert_allclose(log_variance_level(support * 1.4, density, business_years=business, calendar_years=calendar), level)
    with pytest.raises(ValueError, match='normalized'):
        log_variance_level(support, density * .9, business_years=business, calendar_years=calendar)


def test_put_call_parity_and_independent_pair_standard_errors() -> None:
    levels = np.array([10., 20., 12., 22.])
    mean, se = payoff_statistics(levels, np.array([15., 15.]), np.array([True, False]), discount=.99)
    np.testing.assert_allclose(mean, .99 * np.array([3., 2.]))
    np.testing.assert_allclose(se, [.495, .495])
    assert mean[0] - mean[1] == pytest.approx(.99 * (levels.mean() - 15))
    with pytest.raises(ValueError, match='pairs'):
        payoff_statistics(levels[:3], np.array([15.]), np.array([True]), discount=1.)


def test_partial_bridge_preserves_mass_and_conditional_forward() -> None:
    import jax
    import jax.numpy as jnp

    from quantdata.sdlv.daily.front_spot.jax_model import RiskNeutralState, advance_front_density
    from quantdata.sdlv.daily.vix_options import _partial_day

    with jax.enable_x64():
        support = jnp.array([.5, .8, 1., 1.2, 1.5])
        front = jnp.array([[.05, .2, .5, .2, .05]])
        state = RiskNeutralState(front, jnp.full((1, 2, 3), .04), jnp.ones(1))
        normals = jnp.array([[[.3, -.2]]])
        result = _partial_day(support, state, jnp.zeros((2, 3, 2)), normals, fraction=.2)
        revealed = advance_front_density(front, normals[0, :, 0], remaining_time=1/252,
                                         step_size=.2/252, final_step=False)
        np.testing.assert_allclose(np.asarray(result.sum(axis=-1)), 1., atol=1e-12)
        np.testing.assert_allclose(np.asarray(result @ support), np.broadcast_to(np.asarray(revealed @ support)[:, None], (1, 2)), atol=1e-12)
        assert np.all(np.asarray(result) >= 0)
