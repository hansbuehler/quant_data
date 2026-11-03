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


def test_wide_default_grid_has_a_downweighted_outer_fit_band() -> None:
    config = Gromm2GeometryConfig()
    assert config.grid_nms_range == (-4.0, 3.0)
    assert config.fit_nms_range == (-4.0, 3.0)
    assert config.core_fit_nms_range == (-3.5, 2.5)
    assert config.outer_fit_weight == 0.10
    assert config.nms_nodes[0] == -4.0
    assert config.nms_nodes[-1] == 3.0


def test_barycentric_transport_rejects_extrapolation() -> None:
    source = np.asarray([0.5, 1.0, 1.5])
    target = np.asarray([0.6, 1.0, 1.4])
    try:
        barycentric_transport(source, target)
    except ValueError as error:
        assert "contain" in str(error)
    else:
        raise AssertionError("expected non-containing support to be rejected")


def test_tail_fill_is_off_by_default_and_geometric_when_on() -> None:
    variances = np.asarray([0.15**2 / 252.0, 0.15**2 * 5.0 / 252.0])
    base_config = Gromm2GeometryConfig(num_inner_strikes=24)
    base = expanding_supports(variances, config=base_config)
    assert base.shape == (2, 26)
    assert np.array_equal(base_config.support_nms_nodes, base_config.nms_nodes)
    config = Gromm2GeometryConfig(num_inner_strikes=24, lower_tail_nodes=14, upper_tail_nodes=8)
    supports = expanding_supports(variances, config=config)
    assert supports.shape == (2, 24 + 14 + 8 + 2)
    assert config.support_nms_nodes.shape == (config.num_interior_nodes,)
    assert np.allclose(supports[:, [0, *range(15, 39), -1]], base, rtol=0.0, atol=1.0e-15)
    ratios = supports[:, 1:16] / supports[:, :15]
    assert np.allclose(ratios, 2.0 ** (1.0 / 15.0), rtol=1.0e-12)
    assert np.allclose(supports[:, -9:] / supports[:, -10:-1], 1.5 ** (1.0 / 9.0), rtol=1.0e-12)
    transports = transport_tower(supports)
    assert np.allclose(supports[1] @ transports[0], supports[0], atol=2.0e-13, rtol=0.0)
