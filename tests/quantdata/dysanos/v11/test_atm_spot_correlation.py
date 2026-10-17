from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from quantdata.dysanos.v11.atm_spot_correlation import (
    _antithetic_correlation_standard_errors,
    _atm_iv_from_normalized_calls,
)


def test_atm_call_inversion_recovers_black_volatility() -> None:
    expiries = np.asarray((1.0 / 252.0, 20.0 / 252.0, 1.0))
    volatility = np.asarray((0.12, 0.25, 0.40))
    calls = 2.0 * ndtr(0.5 * volatility * np.sqrt(expiries)) - 1.0
    recovered = _atm_iv_from_normalized_calls(calls[None, :], expiries)
    assert recovered[0] == pytest.approx(volatility)


def test_antithetic_correlation_error_is_finite() -> None:
    spot = np.asarray((-2.0, -1.0, 2.0, 1.0))
    volatility = np.column_stack((0.7 * spot, -0.2 * spot + (1.0, -1.0, -1.0, 1.0)))
    errors = _antithetic_correlation_standard_errors(spot, volatility)
    assert errors.shape == (2,)
    assert np.all(np.isfinite(errors))
    assert np.all(errors >= 0.0)
