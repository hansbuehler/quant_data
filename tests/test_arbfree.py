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
    assert len(result.strikes) == count
    assert len(result.bids) == count
    assert len(result.asks) == count
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

    for output_index, (xstrikes_t, strikes_t, bids_t, asks_t, mask_t, fit1_t, fitx_t, mstrikes_t, xix_to_mix_t) in enumerate(zip(
        result.xstrikes,
        result.strikes,
        result.bids,
        result.asks,
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
        assert np.array_equal(strikes_t, source_strikes_t)
        assert np.array_equal(bids_t, source_bids_t)
        assert np.array_equal(asks_t, source_asks_t)
        assert mask_t.shape == strikes_t.shape
        assert np.array_equal(strikes_t[mask_t], xstrikes_t[1:-1])
        assert np.all(np.diff(mstrikes_t) > 0.0)
        assert fit1_t.shape == fitx_t.shape == strikes_t.shape
        assert bids_t[mask_t].shape == asks_t[mask_t].shape == fit1_t[mask_t].shape == fitx_t[mask_t].shape == xstrikes_t[1:-1].shape
        assert np.all(np.isfinite(fit1_t))
        assert np.all(np.isfinite(fitx_t))
        assert np.all(fit1_t[mask_t] >= bids_t[mask_t] - 1e-7)
        assert np.all(fit1_t[mask_t] <= asks_t[mask_t] + 1e-7)

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


def test_mk_arbfree_off_mode_returns_extended_market_only() -> None:
    data = _load_training_data()
    expiries, strikes, bids, asks = _build_market(data, 0)

    result = MkArbFree(no_arb_mode="off")(expiries, strikes, bids, asks, verbose=Context.quiet)

    assert result.fits1 is None
    assert result.fitsX is None
    assert result.mstrikes is None
    assert result.xix_to_mix is None
    assert len(result.expiries) == len(result.xstrikes) == len(result.bids) == len(result.asks)
    assert len(result.strikes) == len(result.masks) == len(result.expiries)

    input_index = np.asarray(result.input_index, dtype=np.int32)
    assert len(input_index) == len(result.expiries)
    assert np.all(np.diff(input_index) > 0)
    np.testing.assert_allclose(result.expiries.astype(np.float64), expiries[input_index], rtol=0.0, atol=0.0)

    for output_index, (xstrikes_t, strikes_t, bids_t, asks_t, mask_t) in enumerate(zip(result.xstrikes, result.strikes, result.bids, result.asks, result.masks, strict=True)):
        source_strikes_t = strikes[input_index[output_index]]
        source_bids_t = bids[input_index[output_index]]
        source_asks_t = asks[input_index[output_index]]
        mask_t = np.asarray(mask_t, dtype=bool)

        assert np.all(np.diff(xstrikes_t) > 0.0)
        assert np.array_equal(strikes_t, source_strikes_t)
        assert np.array_equal(bids_t, source_bids_t)
        assert np.array_equal(asks_t, source_asks_t)
        assert mask_t.shape == strikes_t.shape
        assert np.array_equal(strikes_t[mask_t], xstrikes_t[1:-1])