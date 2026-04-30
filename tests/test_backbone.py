from __future__ import annotations

import numpy as np
import pytest

from quantdata.utils.backbone import BackboneNumpy, are_broadcastable, relu, sigmoid


def test_numpy_backbone_scalar_helpers() -> None:
    values = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)

    assert np.allclose(sigmoid(values), np.asarray([0.26894142, 0.5, 0.73105858]))
    assert np.allclose(relu(values), np.asarray([0.0, 0.0, 1.0]))
    assert are_broadcastable((3, 1), (1, 4))
    assert not are_broadcastable((2, 3), (4, 3))


def test_numpy_backbone_array_roundtrip_and_verify() -> None:
    backbone = BackboneNumpy()
    arr = backbone.to_array([1, 2, 3], dtype=backbone.float32)

    assert arr.dtype == np.float32
    assert np.array_equal(backbone.to_numpy(arr), np.asarray([1, 2, 3], dtype=np.float32))
    assert backbone.stack([arr, arr], axis=0).shape == (2, 3)
    backbone.verify(True, "should not raise")

    with pytest.raises(RuntimeError, match="expected failure"):
        backbone.verify(False, "expected failure")