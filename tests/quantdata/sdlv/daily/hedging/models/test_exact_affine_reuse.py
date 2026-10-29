from __future__ import annotations

import h5py
import numpy as np
import pytest

from quantdata.sdlv.daily.front_spot.hedging.runners.reuse_exact_affine_history import (
    _same_dataset_values,
)


def test_identical_inputs_allow_only_path_metadata_to_differ(tmp_path) -> None:
    older = tmp_path / "older.h5"
    newer = tmp_path / "newer.h5"
    for path, source_history in ((older, "old/source.h5"), (newer, "new/source.h5")):
        with h5py.File(path, "w") as stream:
            stream.create_dataset("dates", data=np.array([20240401, 20240402]))
            stream.create_dataset("values", data=np.array([[1.0, 2.0]]))
            stream.attrs["schema"] = "prepared-book"
            stream.attrs["source_history"] = source_history

    assert _same_dataset_values(
        older,
        newer,
        ignored_attributes=frozenset({"source_history"}),
    ) == 2
    with pytest.raises(ValueError, match="metadata changed"):
        _same_dataset_values(older, newer)


def test_changed_model_values_reject_affine_reuse(tmp_path) -> None:
    older = tmp_path / "older.h5"
    newer = tmp_path / "newer.h5"
    with h5py.File(older, "w") as stream:
        stream.create_dataset("values", data=np.array([1.0, 2.0]))
    with h5py.File(newer, "w") as stream:
        stream.create_dataset("values", data=np.array([1.0, 2.1]))

    with pytest.raises(ValueError, match="model input changed"):
        _same_dataset_values(older, newer)
