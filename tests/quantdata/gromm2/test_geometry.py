from __future__ import annotations

import numpy as np

from quantdata.gromm2.geometry import (
    Gromm2GeometryConfig,
    barycentric_transport,
    expanding_supports,
    project_total_variance,
    transport_tower,
)


def test_total_variance_projection_enforces_forward_floor() -> None:
    expiries = np.asarray([1.0, 2.0, 5.0]) / 252.0
    raw = np.asarray([0.04, 0.03, 0.031]) ** 2 * expiries
    projected = project_total_variance(
        expiries,
        raw,
        forward_volatility_floor=0.01,
    )
    increments = np.diff(projected, prepend=0.0) / np.diff(expiries, prepend=0.0)
    assert np.min(increments) >= 0.01**2 - 1.0e-14


def test_expanding_support_and_transport_preserve_affine_payoffs() -> None:
    config = Gromm2GeometryConfig(num_inner_strikes=24)
    variances = np.asarray([0.15**2 / 252.0, 0.15**2 * 5.0 / 252.0])
    supports = expanding_supports(variances, config=config)
    transports = transport_tower(supports)
    transport = transports[0]
    assert np.allclose(np.sum(transport, axis=0), 1.0, atol=2.0e-14, rtol=0.0)
    assert np.allclose(supports[1] @ transport, supports[0], atol=2.0e-13, rtol=0.0)
    assert np.any(config.nms_nodes == 0.0)


def test_barycentric_transport_rejects_extrapolation() -> None:
    source = np.asarray([0.5, 1.0, 1.5])
    target = np.asarray([0.6, 1.0, 1.4])
    try:
        barycentric_transport(source, target)
    except ValueError as error:
        assert "contain" in str(error)
    else:
        raise AssertionError("expected non-containing support to be rejected")
