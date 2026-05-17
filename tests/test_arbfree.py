from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from cdxcore.verbose import Context

from quantdata.utils.arbfree import MkArbFree


DATA_FILE = Path(__file__).resolve().parent / "stored_data" / "training" / "grid_sanos_four_samples.h5"


def _load_training_data() -> dict[str, np.ndarray]:
    with h5py.File(DATA_FILE, "r") as h5_file:
        root = h5_file["root"]
        return {key: root[key][()] for key in root.keys()}


def _build_market(data: dict[str, np.ndarray], isample: int) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    mask = data["is_valid"][isample].astype(bool)
    expiries = []
    strikes = []
    bids = []
    asks = []

    for expiry in np.unique(data["tte"][isample][mask]):
        group_mask = (data["tte"][isample] == expiry) & mask
        order = np.argsort(data["strikes"][isample][group_mask])
        expiries.append(float(expiry))
        strikes.append(data["strikes"][isample][group_mask][order].astype(np.float64))
        bids.append(data["bid"][isample][group_mask][order].astype(np.float64))
        asks.append(data["ask"][isample][group_mask][order].astype(np.float64))

    return np.array(expiries, dtype=np.float64), strikes, bids, asks


def _assert_surface_result_invariants(
    result,
    input_expiries: np.ndarray,
    input_strikes: list[np.ndarray],
    input_bids: list[np.ndarray],
    input_asks: list[np.ndarray],
) -> None:
    count = len(result.expiries)
    assert count > 0
    assert len(result.xstrikes) == count
    assert len(result.masks) == count
    assert len(result.fits1) == count
    assert len(result.fitsX) == count
    assert len(result.mstrikes) == count
    assert len(result.xix_to_mix) == count
    assert len(result.input_index) == count
    assert np.all(np.diff(result.expiries) > 0.0)

    input_index = np.asarray(result.input_index, dtype=np.int32)
    assert np.all(np.diff(input_index) > 0)
    assert np.all(input_index >= 0)
    assert np.all(input_index < len(input_expiries))
    np.testing.assert_allclose(result.expiries.astype(np.float64), input_expiries[input_index], rtol=0.0, atol=0.0)

    for output_index, (xstrikes_t, mask_t, fit1_t, fitx_t, mstrikes_t, xix_to_mix_t) in enumerate(zip(
        result.xstrikes,
        result.masks,
        result.fits1,
        result.fitsX,
        result.mstrikes,
        result.xix_to_mix,
        strict=True,
    )):
        source_strikes_t = input_strikes[input_index[output_index]]
        source_bids_t = input_bids[input_index[output_index]]
        source_asks_t = input_asks[input_index[output_index]]
        mask_t = np.asarray(mask_t, dtype=bool)

        assert np.all(np.diff(xstrikes_t) > 0.0)
        assert mask_t.shape == source_strikes_t.shape
        assert np.array_equal(source_strikes_t[mask_t], xstrikes_t[1:-1])
        assert np.all(np.diff(mstrikes_t) > 0.0)
        assert fit1_t.shape == fitx_t.shape == source_strikes_t.shape
        assert source_bids_t[mask_t].shape == source_asks_t[mask_t].shape == fit1_t[mask_t].shape == fitx_t[mask_t].shape == xstrikes_t[1:-1].shape
        assert np.all(np.isfinite(fit1_t))
        assert np.all(np.isfinite(fitx_t))
        assert np.all(fit1_t[mask_t] >= source_bids_t[mask_t] - 1e-7)
        assert np.all(fit1_t[mask_t] <= source_asks_t[mask_t] + 1e-7)

        if xix_to_mix_t is None:
            assert np.array_equal(mstrikes_t, xstrikes_t)
        else:
            assert np.array_equal(mstrikes_t[xix_to_mix_t], xstrikes_t)


@pytest.mark.parametrize(
    ("isample", "expect_drop"),
    [
        (0, False),
        (1, True),
    ],
)
def test_mk_arbfree_surface_on_training_samples(isample: int, expect_drop: bool) -> None:
    data = _load_training_data()
    expiries, strikes, bids, asks = _build_market(data, isample)

    result = MkArbFree("surface")(expiries, strikes, bids, asks, verbose=Context.quiet)

    if expect_drop:
        assert len(result.expiries) < len(expiries)
    else:
        assert len(result.expiries) == len(expiries)

    _assert_surface_result_invariants(result, expiries, strikes, bids, asks)


def test_make_extended_strikes_trims_invalid_boundary_slopes() -> None:
    arbfree = MkArbFree(no_arb_mode="off")
    strikes = np.array([0.60, 0.75, 0.90, 1.00, 1.10, 1.20, 1.30, 1.45], dtype=np.float64)
    mids = np.array([0.50, 0.32, 0.20, 0.13, 0.08, 0.04, 0.03995, 0.03990], dtype=np.float64)
    bids = mids - 0.01
    asks = mids + 0.01

    xstrikes, mask, notice = arbfree.make_extended_strikes(strikes, bids, asks)

    expected_mask = np.array([False, True, True, True, True, False, False, False])
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_allclose(xstrikes[1:-1], strikes[mask], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(xstrikes[[0, -1]], np.array([0.2, 1.5]), rtol=0.0, atol=1e-12)
    assert notice is not None
    assert notice() == "removed 4 out of 8 strikes to ensure -1 < dC/dK[0] and dC/dK[-1] < 0."


def test_make_extended_strikes_respects_previous_boundaries() -> None:
    arbfree = MkArbFree(no_arb_mode="off", min_strikes_per_expiry=3)
    strikes = np.array([0.80, 0.90, 1.00, 1.10, 1.20], dtype=np.float64)
    mids = np.array([0.22, 0.13, 0.06, 0.02, 0.005], dtype=np.float64)
    bids = mids - 0.002
    asks = mids + 0.002

    xstrikes, mask, notice = arbfree.make_extended_strikes(
        strikes,
        bids,
        asks,
        prev_lower=0.25,
        prev_upper=1.5,
    )

    np.testing.assert_array_equal(mask, np.ones_like(strikes, dtype=bool))
    np.testing.assert_allclose(xstrikes[1:-1], strikes, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(xstrikes[[0, -1]], np.array([0.25, 1.5]), rtol=0.0, atol=1e-12)
    assert notice is None