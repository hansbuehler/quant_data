"""Tests for the fixed-grid drift-adjusted model entry point."""

from dataclasses import asdict

import numpy as np

from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.run_all_history_joint_direct import (
    _geometry_from_completion,
)


def test_requested_expiry_grid_is_restored_as_fixed_relative_geometry() -> None:
    strikes = (0.8, 0.9, 1.0, 1.1, 1.2)
    model_business_dtes = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    source = FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=len(strikes),
        num_model_expiries=len(model_business_dtes),
        explicit_model_business_dtes=model_business_dtes,
        inner_relative_strikes=strikes,
    )

    restored = _geometry_from_completion(
        {"geometry": asdict(source)},
        requested_model_business_dtes=model_business_dtes,
    )

    assert isinstance(
        restored,
        FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    )
    assert np.array_equal(restored.model_business_dtes, model_business_dtes)
    assert np.array_equal(restored.inner_relative_strikes, strikes)
