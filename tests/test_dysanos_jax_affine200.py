"""JAX price parity for the DYSANOS affine calibration."""

import numpy as np
import torch

from quantdata.dysanos.flex.config import (
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    NativeStateConfig,
)
from quantdata.dysanos.flex.model import NativeStateSanosModel, bound_native_raw
from quantdata.dysanos.flex.sanos import sanos_time_interpolation_numpy
from quantdata.dysanos.v11.audit_model_grid import grid_prices
from quantdata.dysanos.v11.jax_affine200 import make_functions
from quantdata.torch import BaseEnvironment


def test_variance_interpolation_matches_torch_pricer() -> None:
    import jax.numpy as jnp

    geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=12, nms_range=(-4., 3.), num_model_expiries=3,
        explicit_model_business_dtes=(1., 5., 10.),
    )
    env = BaseEnvironment(precision=32, device="cpu", debug=False)
    model = NativeStateSanosModel(env, num_days=1, config=NativeStateConfig(label="jax-price-parity", geometry=geometry))
    generator = torch.Generator().manual_seed(123)
    raw = model.daily_raw_states.detach() + 0.15 * torch.randn(model.daily_raw_states.shape, generator=generator)
    params = bound_native_raw(raw, geometry=geometry)
    surface = model.sanos_decoder.forward_batch(dlvsqs=params.dlvsqs, forward_vols=params.forward_vols, eta=params.eta)
    query = np.array([[.5, 2., 5., 7., 10.]], dtype=np.float64)/252
    strikes = np.array([[.95, .98, 1., 1.02, 1.05]], dtype=np.float32)
    expected = model.sanos_decoder.price_calls_batch(surface, expiries_y=torch.tensor(query, dtype=torch.float32), strikes=torch.tensor(strikes)).detach().numpy()
    index = sanos_time_interpolation_numpy(query, geometry=geometry)
    quote = {"upper": jnp.array(index.upper_index), "lower": jnp.array(index.lower_index),
             "alpha": jnp.array(index.upper_weight, dtype=jnp.float32), "before": jnp.array(index.before_first),
             "strikes": jnp.array(strikes), "is_call": jnp.ones_like(jnp.array(strikes), dtype=bool)}
    grad_fn, price_fn = make_functions(geometry, "variance")
    actual, _ = price_fn(jnp.array(raw.numpy()), quote)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-5, rtol=5e-4)
    _, vol_price_fn = make_functions(geometry, "volatility")
    vol_price, _ = vol_price_fn(jnp.array(raw.numpy()), quote)
    np.testing.assert_allclose(np.asarray(vol_price)[:, [2, 4]], np.asarray(actual)[:, [2, 4]], atol=1e-7)
    quote.update(bids=jnp.maximum(actual-.001, 0.), asks=actual+.001, mids=actual,
                 mask=jnp.ones_like(actual, dtype=bool))
    state = {"b": jnp.array(raw.numpy()[0]), "B": jnp.zeros((2, raw.shape[-1])), "h": jnp.ones((1, 2))}
    (loss, _), gradient = grad_fn(state, quote, jnp.array([0]))
    assert np.isfinite(np.asarray(loss)).all()
    assert all(np.isfinite(np.asarray(value)).all() for value in gradient.values())


def test_grid_audit_prices_match_jax() -> None:
    import jax.numpy as jnp

    geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=12, nms_range=(-4., 3.), num_model_expiries=3,
        explicit_model_business_dtes=(1., 5., 10.),
    )
    env = BaseEnvironment(precision=32, device="cpu", debug=False)
    model = NativeStateSanosModel(env, num_days=1, config=NativeStateConfig(label="grid-parity", geometry=geometry))
    raw = model.daily_raw_states.detach().numpy()
    strikes, actual = grid_prices(raw[0], geometry)
    q = {"strikes": jnp.array(np.tile(strikes, 3)[None, :]),
         "upper": jnp.array(np.repeat(np.arange(3), 12)[None, :]),
         "lower": jnp.array(np.repeat(np.maximum(np.arange(3)-1, 0), 12)[None, :]),
         "alpha": jnp.ones((1, 36)), "before": jnp.zeros((1, 36), dtype=bool),
         "is_call": jnp.ones((1, 36), dtype=bool)}
    _, price_fn = make_functions(geometry, "variance")
    expected, _ = price_fn(jnp.array(raw), q)
    np.testing.assert_allclose(actual.ravel(), np.asarray(expected).ravel(), atol=2e-5, rtol=5e-4)
