from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.gromm2.geometry import Gromm2GeometryConfig, expanding_supports, transport_tower
from quantdata.gromm2.observed_calibration import (
    ObservedFitConfig,
    black_quote_kernels,
    calibration_functions,
    extract_observed_quotes,
    quote_interpolation,
    reference_grid_nms,
)
from quantdata.gromm2.repair_calibration import (
    RepairConfig,
    linear_tower_initializer,
    relaxed_expiry_bound,
    run_reference_repair,
    sequential_tower_initializer,
    total_variance_functions,
)

jax.config.update("jax_enable_x64", True)


def example_data(quote_days=None):
    expiries = np.asarray([1.0, 5.0, 10.0]) / 252.0
    supports = expanding_supports(0.2**2 * expiries, config=Gromm2GeometryConfig(num_inner_strikes=20))
    quote_expiries = np.asarray([5.0, 5.0, 10.0, 10.0] if quote_days is None else quote_days) / 252.0
    lower, upper, alpha, before = quote_interpolation(expiries, quote_expiries)
    strikes = np.asarray([0.99, 1.01, 0.99, 1.01])
    arguments = dict(strikes=strikes, expiries_y=quote_expiries, is_call=np.ones(4, dtype=bool), annual_volatility=0.05)
    return {
        "model_expiries_y": expiries, "supports": supports, "transports": transport_tower(supports),
        "lower": lower, "upper": upper, "alpha": alpha, "before_first": before,
        "lower_kernels": black_quote_kernels(supports=supports[lower], **arguments),
        "upper_kernels": black_quote_kernels(supports=supports[upper], **arguments),
        "front_prices": black_quote_kernels(supports=np.ones((4, 1)), **arguments)[:, 0], "fair_values": np.zeros(4),
        "half_spreads": np.full(4, 0.0001), "quote_weights": np.ones(4),
        "strikes": strikes, "expiries_y": quote_expiries,
    }


def test_total_variance_coordinates_preserve_existing_objective_and_prices() -> None:
    data = example_data()
    config = ObservedFitConfig()
    old_loss, old_decode = calibration_functions(data, config=config)
    new_loss, new_decode = total_variance_functions(data, config=config)
    raw = jnp.linspace(-23.0, 2.0, 60).reshape(3, 20)
    total = config.minimum_dlvsq + jnp.exp(raw)
    assert float(old_loss(raw)) == pytest.approx(float(new_loss(jnp.log(total))), rel=1.0e-12)
    for old, new in zip(old_decode(raw), new_decode(jnp.log(total)), strict=True):
        np.testing.assert_allclose(old, new, rtol=2.0e-12, atol=1.0e-14)


def test_log_total_variance_releases_the_additive_floor_gradient_dead_zone() -> None:
    data = example_data()
    config = ObservedFitConfig()
    old_loss, _ = calibration_functions(data, config=config)
    new_loss, _ = total_variance_functions(data, config=config)
    raw = jnp.full((3, 20), -23.0)
    log_total = jnp.log(config.minimum_dlvsq + jnp.exp(raw))
    old_gradient = np.asarray(jax.grad(old_loss)(raw))
    new_gradient = np.asarray(jax.grad(new_loss)(log_total))
    assert np.all(np.isfinite(new_gradient))
    assert np.linalg.norm(new_gradient) > 100_000.0 * np.linalg.norm(old_gradient)


def test_exact_lower_bound_has_finite_objective_and_gradient() -> None:
    data = example_data()
    config = ObservedFitConfig()
    loss, _ = total_variance_functions(data, config=config)
    point = jnp.full((3, 20), np.log(config.minimum_dlvsq))
    value, gradient = jax.value_and_grad(loss)(point)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_relaxed_density_bound_can_represent_an_existing_model_exactly() -> None:
    data = example_data()
    config = ObservedFitConfig()
    _, decode = total_variance_functions(data, config=config)
    data["fair_values"] = np.asarray(decode(jnp.full((3, 20), np.log(0.04)))[-1])
    bound = relaxed_expiry_bound(data, config=config)
    assert bound["optimistic_objective"] < 1.0e-9
    assert all(expiry["inside_percent"] == 100.0 for expiry in bound["expiries"])


def test_repair_configuration_rejects_invalid_training_controls() -> None:
    with pytest.raises(ValueError, match="positive"):
        RepairConfig(maximum_iterations=0)


def test_reference_grid_coordinate_uses_saved_origin_total_variance() -> None:
    expiries = np.asarray([1.0, 5.0]) / 252.0
    variances = np.asarray([0.0001, 0.0009])
    z_values = np.asarray([-4.0, -3.5, 0.0, 2.5, 3.0])
    query = 3.0 / 252.0
    strikes = np.exp(z_values * np.sqrt(0.0005))
    actual = reference_grid_nms(strikes, expiry=query, reference_nms_curve=(expiries, variances))
    np.testing.assert_allclose(actual, z_values, atol=2.0e-14)


def test_reference_quote_range_selects_full_market_not_history_nms_intersection() -> None:
    expiry = 5.0 / 252.0
    reference_z = np.asarray([-4.1, -3.8, -2.0, 0.0, 2.8, 3.1])
    strikes = np.exp(reference_z * np.sqrt(0.2**2 * expiry))
    fair_values = np.full(len(strikes), 0.02)

    class HistoryPricer:
        def interpolate_atm_vols(self, expiry):
            return 0.03

    market = {"expiries": np.asarray([expiry]), "strikes": [strikes],
              "is_american": [np.zeros(len(strikes), dtype=bool)], "is_am": [False],
              "af_call_mids": [fair_values], "af_put_mids": [fair_values + strikes - 1.0]}
    for side in ("call", "put"):
        market.update({f"{side}_bids": [np.full(len(strikes), 0.001)],
                       f"{side}_asks": [np.full(len(strikes), 0.003)],
                       f"{side}_active": [np.ones(len(strikes), dtype=bool)]})
    payload = {"market": market}
    geometry = Gromm2GeometryConfig()
    old = extract_observed_quotes(payload, geometry=geometry, pricer=HistoryPricer())
    new = extract_observed_quotes(payload, geometry=geometry, pricer=HistoryPricer(),
                                  reference_nms_curve=(np.asarray([expiry]), np.asarray([0.2**2 * expiry])))
    assert len(old["strikes"]) == 2
    np.testing.assert_array_equal(new["strikes"], np.tile(strikes[[1, 2, 3, 4]], 2))
    np.testing.assert_array_equal(new["quote_weights"], np.tile([0.1, 1.0, 1.0, 0.1], 2))


def test_reference_history_is_causal_and_never_emits_failed_quality_model(tmp_path, monkeypatch) -> None:
    original = tmp_path / "original"
    origin = "2025-08-29"
    (original / "own_dates" / origin).mkdir(parents=True)
    history = original / "rolling" / origin / "history"
    for day in ("2025-08-27", "2025-08-28"):
        (history / day).mkdir(parents=True)
        (history / day / "report.json").write_text("{}")
    initializers = []

    def fake_repair(*, original_directory, run_directory, fixed_grid, config, initializer):
        initializers.append(initializer)
        run_directory.mkdir(parents=True, exist_ok=True)
        np.savez(run_directory / "fit.npz", raw=np.zeros((1, 1)))
        return {"date": original_directory.name, "stop_reason": "bounded_lbfgs_0",
                "quote_range_policy": "reference-date-relative-strike-range",
                "fit": {"all": {"inside_percent": 85.0, "quotes": 10}},
                "density": {"minimum_probability": 0.0, "maximum_mass_error": 0.0, "maximum_forward_error": 0.0}}

    monkeypatch.setattr("quantdata.gromm2.repair_calibration.repair_fit", fake_repair)
    output = tmp_path / "candidate"
    with pytest.raises(RuntimeError, match="no Q model"):
        run_reference_repair(original_calibration=original, run_directory=output, origin=origin,
                             config=RepairConfig(reference_date_quote_range=True))
    assert initializers == [None, output / "rolling" / origin / "history" / "2025-08-27"]
    assert not (output / "rolling" / origin / "ar1.npz").exists()
    assert not (output / "rolling" / origin / "report.json").exists()
    summary = json.loads((output / "rolling" / origin / "fit_summary.json").read_text())
    assert not summary["ready"]


def test_linear_tower_initializer_preserves_dlv_bounds_and_decoded_prices() -> None:
    data = example_data()
    config = ObservedFitConfig()
    _, decode = total_variance_functions(data, config=config)
    data["fair_values"] = np.asarray(decode(jnp.full((3, 20), np.log(0.04)))[-1])
    result = linear_tower_initializer(data, config=config, time_limit_seconds=10.0)
    assert np.all(result["dlvsq"] >= config.minimum_dlvsq)
    assert np.all(result["dlvsq"] <= config.maximum_dlvsq)
    np.testing.assert_allclose(result["decoded_densities"].sum(axis=-1), 1.0, atol=1.0e-10)
    np.testing.assert_allclose((result["decoded_densities"] * data["supports"]).sum(axis=-1), 1.0, atol=1.0e-10)
    assert result["lp_maximum_density_decode_error"] < 1.0e-7
    np.testing.assert_allclose(result["prices"], data["fair_values"], atol=1.0e-8)


def test_sequential_initializer_preserves_all_quote_brackets_and_bounds() -> None:
    data = example_data()
    config = ObservedFitConfig()
    data["initial_raw"] = np.full((3, 20), np.log(0.04 - config.minimum_dlvsq))
    _, decode = total_variance_functions(data, config=config)
    data["fair_values"] = np.asarray(decode(jnp.full((3, 20), np.log(0.04)))[-1])
    result = sequential_tower_initializer(data, config=config, time_limit_seconds=30.0)
    assert np.all(result["dlvsq"] >= config.minimum_dlvsq)
    assert np.all(result["dlvsq"] <= config.maximum_dlvsq)
    np.testing.assert_allclose(result["decoded_densities"].sum(axis=-1), 1.0, atol=1.0e-10)
    np.testing.assert_allclose(result["prices"], data["fair_values"], atol=1.0e-7)


def test_full_tower_lp_keeps_front_constants_and_fractional_expiry_interpolation() -> None:
    data = example_data(quote_days=[0.5, 3.0, 7.5, 10.0])
    config = ObservedFitConfig()
    _, decode = total_variance_functions(data, config=config)
    data["fair_values"] = np.asarray(decode(jnp.full((3, 20), np.log(0.04)))[-1])
    result = linear_tower_initializer(data, config=config, time_limit_seconds=10.0)
    np.testing.assert_allclose(result["prices"], data["fair_values"], atol=1.0e-7)
