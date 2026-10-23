"""Checks for DYSANOS-native ATM observations and date alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantdata.dysanos.v11.native_atm_history import load_atm_history


def test_load_native_atm_history_npz(tmp_path: Path) -> None:
    directory = tmp_path / "quant_data"
    directory.mkdir()
    path = directory / "native-atm.npz"
    np.savez_compressed(
        path,
        dates=np.array(["2024-12-30", "2024-12-31", "2025-01-02"]),
        business_dtes=np.array([1.0, 5.0]),
        atm_iv=np.array([[0.2, 0.21], [0.22, 0.23], [0.24, 0.25]]),
        spot_returns=np.array([-0.01, 0.02]),
    )
    dates, dtes, atm, spot = load_atm_history(
        history_path=path, dataset_path=tmp_path
    )
    assert dates.tolist() == ["2024-12-30", "2024-12-31", "2025-01-02"]
    np.testing.assert_array_equal(dtes, [1.0, 5.0])
    assert atm.shape == (3, 2)
    np.testing.assert_array_equal(spot, [-0.01, 0.02])


def test_load_native_atm_history_rejects_misaligned_rows(tmp_path: Path) -> None:
    directory = tmp_path / "quant_data"
    directory.mkdir()
    path = directory / "native-atm.npz"
    np.savez_compressed(
        path,
        dates=np.array(["2024-12-30", "2024-12-31"]),
        business_dtes=np.array([1.0]),
        atm_iv=np.array([[0.2], [0.21]]),
        spot_returns=np.array([0.01, 0.02]),
    )
    with pytest.raises(ValueError, match="invalid DYSANOS native ATM history"):
        load_atm_history(history_path=path, dataset_path=tmp_path)


def test_load_native_atm_history_rejects_other_calibrations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require native-atm-history"):
        load_atm_history(history_path=tmp_path / "other-sanos.h5", dataset_path=tmp_path)
