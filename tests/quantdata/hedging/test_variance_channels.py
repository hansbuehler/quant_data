"""Observable-reference identity and background-scale invariance for variance diagnostics."""

import numpy as np
import pytest

from quantdata.hedging.model_audit.gromm import kernel_prices
from quantdata.hedging.model_audit.variance_channels import same_observed_changes


def test_observed_variance_reference_must_match_exactly() -> None:
    changes = np.arange(12, dtype=float).reshape(4, 3)
    np.testing.assert_array_equal(same_observed_changes({"first": changes, "second": changes.copy()}), changes)
    shifted = changes.copy()
    shifted[0, 0] += 1e-12
    with pytest.raises(ValueError, match="reference differs"):
        same_observed_changes({"first": changes, "second": shifted})


def test_gromm_floating_price_cancels_common_background_level() -> None:
    support = np.array([.8, 1., 1.2])
    density = np.array([[.1, .6, .3], [.3, .6, .1]])
    spot = density @ support
    strikes = np.tile([.99, 1., 1.01], (2, 1))
    reference = kernel_prices(support, density, np.ones(2), spot, strikes, expiry=5. / 252.)
    background = np.array([.9, 1.15])
    shifted = kernel_prices(support, density, background, spot * background, strikes, expiry=5. / 252.)
    np.testing.assert_allclose(shifted, reference, atol=1e-14, rtol=1e-13)
