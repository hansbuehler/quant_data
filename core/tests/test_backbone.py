from __future__ import annotations

import numpy as np
import pytest

from quantdata.torch import BaseEnvironment
from quantdata.torch.bbtorch import BackboneTorch
from core.src.openqai.core.utils import BackboneNumpy, are_broadcastable, relu, sigmoid


def _make_backbones() -> tuple[BackboneNumpy, BackboneTorch]:
    return BackboneNumpy(), BackboneTorch(BaseEnvironment(precision=32, device="cpu"))


def _assert_same_values(expected, actual, *, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    np.testing.assert_allclose(np.asarray(expected), np.asarray(actual), atol=atol, rtol=rtol)


def _geometry_reference(
    *,
    fwd_vols: np.ndarray,
    opt_lix: np.ndarray,
    opt_vweight: np.ndarray,
    opt_vextra: np.ndarray,
    vol_interp: str,
    dsqrtTs: np.ndarray,
    dts: np.ndarray,
    fwdvol_floor: float,
) -> np.ndarray:
    head = opt_lix.shape[:-1]
    target_shape = head + (dsqrtTs.shape[0],)
    if fwd_vols.ndim == 1:
        fwd_vols = fwd_vols.reshape((1,) * len(head) + fwd_vols.shape)
    fwd_vols = np.broadcast_to(fwd_vols, target_shape)

    if vol_interp == "lin_sqrt_time":
        total = np.cumsum((fwd_vols + fwdvol_floor) * dsqrtTs.reshape((1,) * len(head) + dsqrtTs.shape), axis=-1)
        prev = np.where(opt_lix == 0, 0.0, np.take_along_axis(total, np.clip(opt_lix - 1, a_min=0, a_max=None), axis=-1))
        curr = np.take_along_axis(total, opt_lix, axis=-1)
        return opt_vweight * prev + (1.0 - opt_vweight) * curr + opt_vextra * (fwd_vols[..., -1:] + fwdvol_floor)

    total = np.cumsum((fwd_vols + fwdvol_floor) ** 2 * dts.reshape((1,) * len(head) + dts.shape), axis=-1)
    prev = np.where(opt_lix == 0, 0.0, np.take_along_axis(total, np.clip(opt_lix - 1, a_min=0, a_max=None), axis=-1))
    curr = np.take_along_axis(total, opt_lix, axis=-1)
    return np.sqrt(opt_vweight * prev + (1.0 - opt_vweight) * curr + opt_vextra * ((fwd_vols[..., -1:] + fwdvol_floor) ** 2))


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


def test_numpy_and_torch_backbones_share_common_array_semantics() -> None:
    bbnp, bbt = _make_backbones()

    lhs_np = bbnp.to_array([[-1.5, 0.0, 2.0], [3.25, -4.0, 5.5]], dtype=bbnp.float32)
    rhs_np = bbnp.to_array([[0.5, -1.0, 1.5], [2.5, -3.5, 4.0]], dtype=bbnp.float32)
    mask_np = bbnp.to_array([[True, False, True], [False, True, True]], dtype=bbnp.bool_)
    indices_np = bbnp.to_array([[2, 0, 1], [1, 1, 0]], dtype=np.int64)

    lhs_t = bbt.to_array(lhs_np, dtype=bbt.float32)
    rhs_t = bbt.to_array(rhs_np, dtype=bbt.float32)
    mask_t = bbt.to_array(mask_np, dtype=bbt.bool_)
    indices_t = bbt.to_array(indices_np, dtype=bbt.int64)

    assert lhs_np.dtype == np.float32
    assert bbt.to_numpy(lhs_t).dtype == np.float32
    _assert_same_values(bbnp.to_numpy(lhs_np), bbt.to_numpy(lhs_t))
    _assert_same_values(bbnp.stop_gradient(lhs_np), bbt.to_numpy(bbt.stop_gradient(lhs_t)))
    _assert_same_values(bbnp.sigmoid(lhs_np), bbt.to_numpy(bbt.sigmoid(lhs_t)))
    _assert_same_values(bbnp.relu(lhs_np), bbt.to_numpy(bbt.relu(lhs_t)))
    _assert_same_values(bbnp.where(mask_np, lhs_np, rhs_np), bbt.to_numpy(bbt.where(mask_t, lhs_t, rhs_t)))
    _assert_same_values(bbnp.clip(lhs_np, min=-1.0, max=2.5), bbt.to_numpy(bbt.clip(lhs_t, min=-1.0, max=2.5)))
    _assert_same_values(bbnp.minimum(lhs_np, rhs_np), bbt.to_numpy(bbt.minimum(lhs_t, rhs_t)))
    _assert_same_values(bbnp.cumsum(lhs_np, axis=-1), bbt.to_numpy(bbt.cumsum(lhs_t, axis=-1)))
    _assert_same_values(
        bbnp.take_along_axis(lhs_np, indices_np, axis=-1),
        bbt.to_numpy(bbt.take_along_axis(lhs_t, indices_t, axis=-1)),
    )

    stacked_np = bbnp.stack([lhs_np, rhs_np], axis=0, dtype=bbnp.float64)
    stacked_t = bbt.stack([lhs_t, rhs_t], axis=0, dtype=bbt.float64)

    assert stacked_np.dtype == np.float64
    assert bbt.to_numpy(stacked_t).dtype == np.float64
    _assert_same_values(stacked_np, bbt.to_numpy(stacked_t))
    assert bbnp.are_broadcastable(lhs_np[:, :1], rhs_np[:, :1])
    assert bbt.are_broadcastable(lhs_t[:, :1], rhs_t[:, :1])


@pytest.mark.parametrize("high_precision", [False, True])
def test_numpy_and_torch_backbones_share_bs_price_semantics(high_precision: bool) -> None:
    bbnp, bbt = _make_backbones()

    strikes_np = bbnp.to_array([[0.8, 1.0, 1.2], [0.95, 1.05, 1.15]], dtype=bbnp.float32)
    sqrtvars_np = bbnp.to_array([[0.2, 0.3, 0.4], [0.15, 0.25, 0.35]], dtype=bbnp.float32)
    spot_np = bbnp.to_array([[1.0], [1.1]], dtype=bbnp.float32)
    mask_np = bbnp.to_array([[True, False, True], [True, True, False]], dtype=bbnp.bool_)

    prices_np = bbnp.bs_price(
        strikes=strikes_np,
        sqrtVars=sqrtvars_np,
        mask=mask_np,
        spot=spot_np,
        high_precision=high_precision,
        debug_verify_bounds=True,
    )
    prices_t = bbt.bs_price(
        strikes=bbt.to_array(strikes_np, dtype=bbt.float32),
        sqrtVars=bbt.to_array(sqrtvars_np, dtype=bbt.float32),
        mask=bbt.to_array(mask_np, dtype=bbt.bool_),
        spot=bbt.to_array(spot_np, dtype=bbt.float32),
        high_precision=high_precision,
        debug_verify_bounds=True,
    )

    assert prices_np.dtype == np.float32
    assert bbt.to_numpy(prices_t).dtype == np.float32
    _assert_same_values(prices_np, bbt.to_numpy(prices_t), atol=2e-6)
