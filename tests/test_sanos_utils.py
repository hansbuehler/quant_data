from __future__ import annotations

import numpy as np

from quantdata.sanos.old.utils import extract_atm_fwd_vols


def test_extract_atm_fwd_vols_interpolates_atm_and_forward_variance():
    expiries = np.asarray([1.0, 2.0], dtype=np.float64)
    strikes = [
        np.asarray([1.1, 0.9], dtype=np.float64),
        np.asarray([1.2, 0.8], dtype=np.float64),
    ]
    vols = [
        np.asarray([0.22, 0.18], dtype=np.float64),
        np.asarray([0.34, 0.26], dtype=np.float64),
    ]

    fwd_vols = extract_atm_fwd_vols(expiries, strikes, vols, fwdvol_min=0.01)

    expected = np.asarray([0.20, np.sqrt(0.30**2 * 2.0 - 0.20**2)], dtype=np.float64)
    assert np.allclose(fwd_vols, expected), f"Expected forward vols {expected}, got {fwd_vols}"
    assert fwd_vols.dtype == np.float64, f"Expected float64 output dtype, got {fwd_vols.dtype}"