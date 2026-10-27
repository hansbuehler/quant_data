from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.gromm2 import forward_hedging
from quantdata.gromm2.dlv import decode_density_tower
from quantdata.gromm2.forward_hedging import (
    ForwardScenarioConfig,
    forward_payoff,
    generate_scenarios,
    independent_normal_block,
    instrument_book,
    price_fixed_book,
    sample_summary,
    scenario_function,
    unconditional_price_diagnostics,
)
from quantdata.gromm2.geometry import Gromm2GeometryConfig, expanding_supports, transport_tower

jax.config.update("jax_enable_x64", True)


def small_model() -> dict[str, np.ndarray | float]:
    expiries = np.arange(1.0, 26.0) / 252.0
    supports = expanding_supports(0.2**2 * expiries, config=Gromm2GeometryConfig(num_inner_strikes=16))
    transports = transport_tower(supports)
    dlvsq = np.full((len(expiries), 16), 0.18**2)
    densities = np.asarray(decode_density_tower(jnp.asarray(supports), jnp.asarray(transports), jnp.asarray(dlvsq), jnp.asarray(np.diff(expiries, prepend=0.0))))
    return {"supports": supports, "transports": transports, "expiries_y": expiries,
            "dlvsq": dlvsq, "densities": densities, "loadings": np.zeros((24, 16, 2)), "background_volatility": 0.05}


def test_fixed_otm_book_has_only_the_requested_two_extra_wings() -> None:
    book = instrument_book()
    assert len(book["option_strikes"]) == 22
    for expiry in (5, 10, 15):
        np.testing.assert_allclose(book["option_strikes"][book["option_expiry_days"] == expiry], [0.98, 0.99, 1.0, 1.01, 1.02])
    np.testing.assert_allclose(book["option_strikes"][book["option_expiry_days"] == 20], [0.95, 0.98, 0.99, 1.0, 1.01, 1.02, 1.05])
    np.testing.assert_array_equal(book["option_is_call"], book["option_strikes"] >= 1.0)


def test_train_and_validation_random_streams_are_independent_and_paired() -> None:
    config = ForwardScenarioConfig(paths=8, batch_size=8)
    training = independent_normal_block(config=config, split="train", completed=0, count=8, factors=3)
    validation = independent_normal_block(config=config, split="validation", completed=0, count=8, factors=3)
    for factors, background in (training, validation):
        np.testing.assert_array_equal(factors[:, :, ::2] + factors[:, :, 1::2], 0.0)
        np.testing.assert_array_equal(background[:, ::2] + background[:, 1::2], 0.0)
    assert not np.array_equal(training[0], validation[0])
    np.testing.assert_array_equal(training[0], independent_normal_block(config=config, split="train", completed=0, count=8, factors=3)[0])


def test_forward_payoff_uses_fixing_not_inception_and_pair_level_se() -> None:
    spots = np.asarray([[1.0, 0.9, 1.1], [1.0, 1.1, 1.0], [1.0, 1.2, 1.1], [1.0, 0.8, 0.9]])
    payoff = forward_payoff(spots, fixing_day=1, terminal_day=2)
    np.testing.assert_allclose(payoff, np.abs(spots[:, 2] / spots[:, 1] - 1.0))
    pairs = 0.5 * (payoff[::2] + payoff[1::2])
    stats = sample_summary(payoff)
    assert stats["mean"] == pytest.approx(np.mean(payoff))
    assert stats["standard_error"] == pytest.approx(np.std(pairs, ddof=1) / np.sqrt(2))


def test_book_keeps_poststep_density_index_before_front_roll() -> None:
    model = small_model()
    book = {name: jnp.asarray(value) for name, value in instrument_book().items()}
    book["background_volatility"] = jnp.asarray(0.05)
    support = jnp.asarray(model["supports"])
    densities = jnp.asarray(model["densities"])[None]
    before_roll = price_fixed_book(supports=support, densities=densities, background=jnp.ones(1), spot=jnp.ones(1), day=jnp.asarray(1), front_day=jnp.asarray(0), book=book)
    after_roll = price_fixed_book(supports=support[1:], densities=densities[:, 1:], background=jnp.ones(1), spot=jnp.ones(1), day=jnp.asarray(1), book=book)
    np.testing.assert_allclose(before_roll, after_roll, rtol=1.0e-12, atol=1.0e-14)


def test_simulation_is_causal_and_settles_fixed_contracts_once() -> None:
    model = small_model()
    config = ForwardScenarioConfig(paths=4, batch_size=4, substeps=1)
    simulate, book = scenario_function(**model, config=config)
    factors, background = independent_normal_block(config=config, split="train", completed=0, count=4, factors=2)
    features, prices, payoff, diagnostics = map(np.asarray, simulate(jnp.asarray(factors), jnp.asarray(background)))
    assert features.shape[:2] == (4, 20)
    assert prices.shape == (4, 21, 23)
    np.testing.assert_allclose(payoff, forward_payoff(prices[:, :, 0], fixing_day=10, terminal_day=20), atol=1.0e-14)
    for contract, expiry in enumerate(book["option_expiry_days"]):
        side = 1.0 if book["option_is_call"][contract] else -1.0
        intrinsic = np.maximum(side * (prices[:, expiry, 0] - book["option_strikes"][contract]), 0.0)
        np.testing.assert_allclose(prices[:, expiry, contract + 1], intrinsic, atol=1.0e-14)
        np.testing.assert_array_equal(prices[:, expiry + 1:, contract + 1], 0.0)
    assert np.max(diagnostics[:, 1:3]) < 1.0e-10
    assert np.min(diagnostics[:, 3]) >= -1.0e-12
    changed = factors.copy()
    changed[10:] *= -1.0
    changed_background = background.copy()
    changed_background[10:] *= -1.0
    other_features, other_prices, _, _ = map(np.asarray, simulate(jnp.asarray(changed), jnp.asarray(changed_background)))
    np.testing.assert_array_equal(features[:, :11], other_features[:, :11])
    np.testing.assert_array_equal(prices[:, :11], other_prices[:, :11])


def test_invalid_sampling_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="antithetic"):
        ForwardScenarioConfig(paths=3)
    with pytest.raises(ValueError, match="forward-start"):
        replace(ForwardScenarioConfig(), fixing_day=20)
    with pytest.raises(ValueError, match="linear_solver"):
        replace(ForwardScenarioConfig(), linear_solver="unknown")


def test_ar1_scales_are_predictable_and_use_saved_log_squared_state() -> None:
    model = small_model()
    ar1 = {"variance_intercept": np.asarray([-0.7, -0.5]), "variance_slope": np.asarray([0.2, 0.1]),
           "fixed_factor_scales": np.asarray([1.1, 0.9]), "scale_calibration": np.asarray([1.4, 1.3]),
           "terminal_log_squared": np.asarray([-0.3, 0.4]), "log_scale_bound": np.log(3.0), "squared_score_floor": 0.0025}
    config = ForwardScenarioConfig(paths=4, batch_size=4, substeps=2)
    simulate, _ = scenario_function(**model, config=config, ar1_parameters=ar1, loading_reference_days=np.arange(2.0, 26.0))
    factors, background = independent_normal_block(config=config, split="train", completed=0, count=4, factors=2)
    observed, prices, _, _ = map(np.asarray, simulate(jnp.asarray(factors), jnp.asarray(background)))
    expected_scale = ar1["fixed_factor_scales"] * ar1["scale_calibration"] * np.exp(
        0.5 * (ar1["variance_intercept"] + ar1["variance_slope"] * ar1["terminal_log_squared"]))
    np.testing.assert_allclose(observed[:, 0, -4:-2], np.broadcast_to(ar1["terminal_log_squared"], (4, 2)))
    np.testing.assert_allclose(observed[:, 0, -2:], np.broadcast_to(expected_scale, (4, 2)))
    expected_state = np.log((expected_scale * np.sum(factors[0], axis=0) / np.sqrt(2.0) / ar1["fixed_factor_scales"]) ** 2 + 0.0025)
    np.testing.assert_allclose(observed[:, 1, -4:-2], expected_state, atol=1.0e-12)
    ordinary, _ = scenario_function(**model, config=config)
    ordinary_prices = np.asarray(ordinary(jnp.asarray(factors), jnp.asarray(background))[1])
    np.testing.assert_array_equal(prices, ordinary_prices)


def test_unconditional_screens_exclude_postsettlement_price_drops() -> None:
    book = {"option_expiry_days": np.asarray([1]), "option_strikes": np.asarray([1.0]), "option_is_call": np.asarray([True])}
    prices = np.ones((4, 3, 2))
    prices[:, :, 0] = [[1.0, 1.1, 1.0], [1.0, 0.9, 1.0], [1.0, 1.2, 1.0], [1.0, 0.8, 1.0]]
    prices[:, 2, 1] = 0.0
    report = unconditional_price_diagnostics(prices, book)
    assert report["raw_daily_gains"]["coordinate_count"] == 3
    assert report["raw_daily_gains"]["nonzero_deterministic_coordinates"] == 0
    assert report["option_to_expiry_gains"]["maximum_absolute_statistic"] == 0.0


def test_scenario_cache_is_read_only_after_completion_and_keeps_split_identity(tmp_path, monkeypatch) -> None:
    model = small_model()
    volatility = model.pop("background_volatility")
    monkeypatch.setattr(forward_hedging, "check_resource_headroom", lambda: None)
    monkeypatch.setattr(forward_hedging, "load_model", lambda *_: (model, {"background_volatility": volatility, "origin": "2025-08-29"}))
    config = ForwardScenarioConfig(paths=8, batch_size=4, substeps=1)
    arguments = dict(calibration_root=tmp_path, origin="2025-08-29", run_directory=tmp_path, config=config)
    training = generate_scenarios(**arguments, split="train")
    checksum = hashlib.sha256(training.read_bytes()).hexdigest()
    assert generate_scenarios(**arguments, split="train") == training
    assert hashlib.sha256(training.read_bytes()).hexdigest() == checksum
    validation = generate_scenarios(**arguments, split="validation")
    with h5py.File(training, "r") as train, h5py.File(validation, "r") as heldout:
        assert int(train.attrs["completed_paths"]) == 8
        assert train["prices"].dtype == np.dtype("float64")
        assert train["features"].dtype == np.dtype("float32")
        assert not np.array_equal(train["prices"][:], heldout["prices"][:])
    progress = json.loads((tmp_path / "train_progress.json").read_text())
    assert progress["completed_paths"] == 8
    assert progress["stage"] == "provisional_path_prefix_not_final_results"
    assert progress["unconditional_martingale_diagnostics"]["raw_daily_gains"]["coordinate_count"] > 0
    with pytest.raises(ValueError, match="sampling plan"):
        generate_scenarios(**{**arguments, "config": replace(config, seed=42)}, split="train")
