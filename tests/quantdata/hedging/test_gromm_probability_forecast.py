"""The floating observable must keep independent Black Y and normalize by spot."""

from types import SimpleNamespace

import numpy as np

from quantdata.hedging.model_audit.gromm_probability_forecast import observable


def test_absorbed_x_still_has_unchanged_independent_black_iv() -> None:
    support = np.tile([.9, 1., 1.1], (25, 1))
    geometry = SimpleNamespace(expiries=np.arange(1, 26) / 252., supports=support,
        variance_at=lambda time, backbone: .05**2 * np.asarray(time))
    dynamics = SimpleNamespace(geometry=geometry)
    densities = np.zeros((3, 25, 3))
    for position in range(3):
        densities[position, :, position] = 1.
    log_background = np.array([-.1, 0., .1])
    spots = np.exp(log_background) * support[0]
    values = observable(dynamics, densities, log_background, spots, elapsed=1)
    assert values.shape == (3, 49)
    np.testing.assert_allclose(values, np.log(.05), atol=3e-5, rtol=0.)
