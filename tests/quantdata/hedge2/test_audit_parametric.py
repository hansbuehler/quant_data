"""Adaptive utility integration independently checks the production Gaussian risk."""

import pytest

from quantdata.hedge2.audit_parametric import integrated_normal_power
from quantdata.hedge2.parametric import gaussian_power_risk


@pytest.mark.parametrize("scaled_std", [1e-5, .01, 1., 100., 1000.])
def test_integrated_normal_power_agrees_with_truncated_normal_formula(scaled_std: float) -> None:
    std = scaled_std / 10.
    integral = integrated_normal_power(std, risk_aversion=10.)
    production = gaussian_power_risk(std, risk_aversion=10.)
    assert integral[0] == pytest.approx(production[0], rel=1e-6, abs=1e-10)
    assert integral[1] == pytest.approx(production[1], rel=1e-6, abs=1e-10)
