from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from quantdata.gromm2.hedging_data import load_mapped_scenarios
from quantdata.gromm2.hedging_policy import feature_normalization


def _write(path, *, chunked=False):
    generator = np.random.default_rng(42)
    with h5py.File(path, "w") as stream:
        stream.attrs["identity"] = json.dumps({"split": "train", "model": {"test": 1},
            "config": {"paths": 8, "fixing_day": 2, "terminal_day": 4, "fixing_hedge_moneyness": [1.0]}})
        stream.attrs["completed_paths"] = 8
        arrays = {"features": generator.normal(size=(8, 4, 3)).astype("f4"), "prices": np.ones((8, 5, 3)),
            "payoff": np.zeros(8), "fixing_option_prices": np.zeros((8, 5, 1)),
            "option_expiry_days": np.array([1, 4]), "option_strikes": np.array([0.99, 1.01]),
            "option_is_call": np.array([False, True])}
        arrays["fixing_option_prices"][:, 2:4] = 0.1
        for name, values in arrays.items():
            stream.create_dataset(name, data=values, chunks=True if chunked else None)


def test_memory_mapping_matches_data_and_both_universes_have_identical_information(tmp_path) -> None:
    path = tmp_path / "train.h5"
    _write(path)
    with_book = load_mapped_scenarios(path, expected_split="train", with_fixing_options=True)
    without_book = load_mapped_scenarios(path, expected_split="train", with_fixing_options=False)
    rows = np.array([6, 0, 3, 1])
    assert with_book.features.shape == (8, 4, 5)
    assert with_book.prices.shape == (8, 5, 4)
    assert without_book.prices.shape == (8, 5, 3)
    np.testing.assert_array_equal(with_book.features[rows], without_book.features[rows])
    np.testing.assert_array_equal(with_book.features[:][:, :2, -2:], 0.0)
    np.testing.assert_array_equal(with_book.prices[:][:, :2, -1], 0.0)
    assert with_book.option_expiry_days == (1, 4, 4)
    assert with_book.sha256 == without_book.sha256
    assert isinstance(without_book.prices, np.memmap)
    mean, scale = feature_normalization(with_book.features, floor=1e-4)
    assert len(with_book.features) == 8
    assert mean.shape == scale.shape == (5,)
    with h5py.File(path, "r") as source:
        np.testing.assert_array_equal(without_book.prices[:], source["prices"][:])


def test_mapper_rejects_partial_or_chunked_sources(tmp_path) -> None:
    path = tmp_path / "train.h5"
    _write(path, chunked=True)
    with pytest.raises(ValueError, match="contiguous"):
        load_mapped_scenarios(path, expected_split="train", with_fixing_options=True)
    with h5py.File(path, "r+") as stream:
        stream.attrs["completed_paths"] = 4
    with pytest.raises(ValueError, match="complete"):
        load_mapped_scenarios(path, expected_split="train", with_fixing_options=True)
