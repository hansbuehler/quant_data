"""Tests for representative fixed relative-strike construction."""

from __future__ import annotations

import numpy as np

from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)


def test_four_week_fixed_geometry_has_one_immutable_support() -> None:
    inner = tuple(np.exp(np.linspace(np.log(0.75), np.log(1.25), 200)))
    atm = int(np.argmin(np.abs(np.asarray(inner) - 1.0)))
    values = list(inner)
    values[atm] = 1.0
    geometry = FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_model_expiries=4,
        explicit_model_business_dtes=(5.0, 10.0, 15.0, 20.0),
        inner_relative_strikes=tuple(values),
    )

    np.testing.assert_array_equal(
        geometry.model_business_dtes,
        np.array((5.0, 10.0, 15.0, 20.0)),
    )
    assert geometry.native_state_size == 4 * 202
    assert sum(value == 1.0 for value in geometry.inner_relative_strikes) == 1
