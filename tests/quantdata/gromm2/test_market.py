from __future__ import annotations

import numpy as np

from quantdata.gromm2.market import _black_call_kernel, _project_points_to_support


def test_point_projection_preserves_mass_and_mean_without_clipping() -> None:
    support = np.asarray([0.5, 0.8, 1.0, 1.3, 1.8])
    values = np.asarray([0.7, 0.95, 1.4])
    masses = np.asarray([0.2, 0.5, 0.3])
    density, clipped_mass = _project_points_to_support(values, masses, support)
    assert clipped_mass == 0.0
    assert np.isclose(np.sum(density), np.sum(masses), atol=2.0e-15)
    assert np.isclose(density @ support, masses @ values, atol=2.0e-15)


def test_black_kernel_zero_volatility_is_intrinsic() -> None:
    support = np.asarray([0.8, 1.0, 1.2])
    strikes = np.asarray([0.9, 1.1])
    actual = _black_call_kernel(
        support,
        strikes,
        annual_volatility=0.0,
        expiry_y=0.2,
    )
    expected = np.maximum(support[None, :] - strikes[:, None], 0.0)
    assert np.array_equal(actual, expected)
