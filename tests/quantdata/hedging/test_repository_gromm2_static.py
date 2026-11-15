from types import SimpleNamespace

import numpy as np
import pytest
from cdxcore.bs import bs

from quantdata.hedging.model_audit.repository_gromm2_static import calls


def test_gromm_static_prices_use_same_cash_strikes_and_backbone_variance():
    simulator = SimpleNamespace(M=3, strikes=np.array([[0.8, 1.2]] * 3), yvar=np.array([0., 0.02, 0.05, 0.09]))
    densities = np.array([[[0.5, 0.5]] * 3])
    strikes = np.array([0., 0.9, 1.1])
    actual = calls(simulator, densities, np.array([0.]), np.array([1.]), day=1, tenor=1, strikes=strikes)
    expected = sum(0.5 * node * bs.price(k=strikes / node, vol=np.sqrt(0.03), sqrtT=1.) for node in (0.8, 1.2))
    np.testing.assert_allclose(actual[0], expected)
    with pytest.raises(ValueError, match="fixed GROMM expiry tower"):
        calls(simulator, densities, np.array([0.]), np.array([1.]), day=2, tenor=2, strikes=strikes)
