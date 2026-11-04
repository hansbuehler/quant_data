"""Expiry trade cohorts use their own strikes, fixing dates and terminal marks."""

import jax
import jax.numpy as jnp
import numpy as np

from quantdata.sdlv.daily.front_spot.jax_model import decode_density_tower
from quantdata.sdlv.dhgromm1.rolling_simulation import ITM_LEVELS, generate_chunk
from quantdata.sdlv.dhgromm1.training import day10_straddle_benchmark

jax.config.update("jax_enable_x64", True)


def test_expiry_cashflows_and_post_horizon_marks(tmp_path) -> None:
    support = jnp.asarray([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    dlvsq = jnp.full((25, 4), 0.04)
    tower = decode_density_tower(support, dlvsq, dt=1.0 / 252.0)
    loadings = np.zeros((24, 4, 8), dtype=np.float64)
    loadings[..., 0] = 0.02
    np.savez(tmp_path / "diffusion.npz", support=support, density_tower=tower,
             dlvsq=dlvsq, annualized_loadings=loadings)
    prices, returns, payoff = generate_chunk(calibration=tmp_path, paths=4, batch_size=4,
                                             substeps=1, seed=np.random.SeedSequence(42))
    assert prices.shape == (4, 21, 21)
    assert returns.shape == (4, 20, 21)
    np.testing.assert_allclose(payoff, np.abs(prices[:, 20, 0] / prices[:, 10, 0] - 1))
    _, penalty, replication_error = day10_straddle_benchmark(prices, returns, payoff)
    assert replication_error < 1e-12
    assert np.all(penalty > 0)
    np.testing.assert_allclose(prices[:, :20, 0] * (1 + returns[:, :, 0]),
                               np.broadcast_to(prices[:, 20, 0, None], (4, 20)), atol=1e-13)
    for day in range(20):
        spot = prices[:, day, 0]
        terminal = prices[:, day, 1] * (1 + returns[:, day, 1])
        np.testing.assert_allclose(terminal, np.maximum(prices[:, 20, 0] - spot, 0), atol=1e-13)
        terminal_forward = prices[:, day, 17] * (1 + returns[:, day, 17])
        np.testing.assert_allclose(terminal_forward, prices[:, 20, 0], atol=1e-13)
        for index, dte in enumerate((1, 5, min(10, 25 - day))):
            strikes = spot[:, None] * np.exp(np.array(ITM_LEVELS) * .2 * np.sqrt(dte / 252))
            assert np.all(strikes < spot[:, None])
            marks = prices[:, day, 2 + 5 * index:7 + 5 * index]
            realized = marks * (1 + returns[:, day, 2 + 5 * index:7 + 5 * index])
            if day + dte <= 20:
                expected = np.maximum(prices[:, day + dte, 0, None] - strikes, 0)
                np.testing.assert_allclose(realized, expected, atol=1e-13)
                forward = prices[:, day, 18 + index] * (1 + returns[:, day, 18 + index])
                np.testing.assert_allclose(forward, prices[:, day + dte, 0], atol=1e-13)
            else:
                intrinsic = np.maximum(prices[:, 20, 0, None] - strikes, 0)
                assert np.all(realized >= intrinsic - 1e-12)
                assert np.any(realized > intrinsic + 1e-8)
