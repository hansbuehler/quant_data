import numpy as np
import pytest

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.sdlv.daily.paper_wide_factors import (
    model_iv_surface,
    nms_strikes,
    saved_factor_spectrum,
)


def test_spectrum_counts_spot_and_residual_power_once():
    loadings = np.array([[[2.0, 3.0]]]) * np.sqrt(DAYS_PER_YEAR)
    fractions = saved_factor_spectrum(loadings, np.array([9.0, 3.0]), expected_fraction=13 / 16)
    np.testing.assert_allclose(fractions, [4 / 16, 9 / 16, 3 / 16])
    with pytest.raises(AssertionError):
        saved_factor_spectrum(loadings, np.array([9.0, 3.0]), expected_fraction=0.9)


def test_spectrum_rejects_mismatched_loadings_and_eigenvalues():
    loadings = np.array([[[2.0, 3.0]]]) * np.sqrt(DAYS_PER_YEAR)
    with pytest.raises(AssertionError):
        saved_factor_spectrum(loadings, np.array([8.0, 3.0]), expected_fraction=13 / 16)


def test_mean_surface_coordinates_use_each_expiry_atm_scale():
    atm = np.array([0.2, 0.3])
    coordinates = np.array([-4.0, 0.0, 3.0])
    strikes = nms_strikes(atm, coordinates)
    np.testing.assert_allclose(NMS.to_nms(strikes, np.arange(1, 3)[:, None] / DAYS_PER_YEAR,
                                         vol=atm[:, None], eps_vol=atm[0]),
                               np.broadcast_to(coordinates, strikes.shape))


def test_otm_surface_inversion_recovers_black_in_both_wings():
    atm = np.full(25, 0.05)
    strikes = nms_strikes(atm, np.array([-4.0, -2.0, 0.0, 1.0, 3.0]))
    iv = model_iv_surface(np.ones(1), np.ones((25, 1)), strikes)
    np.testing.assert_allclose(iv, 0.05, rtol=1e-9, atol=1e-10)


def test_surface_rejects_non_martingale_density():
    with pytest.raises(AssertionError):
        model_iv_surface(np.array([1.1]), np.ones((2, 1)), np.ones((2, 3)))
