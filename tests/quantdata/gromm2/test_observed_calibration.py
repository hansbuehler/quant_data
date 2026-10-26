from __future__ import annotations

import hashlib
import json
from datetime import date
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from cdxcore.bs import bs

from quantdata.biz_calendar import NMS
from quantdata.gromm2.observed_calibration import (
    Gromm2GeometryConfig,
    ObservedFitConfig,
    black_quote_kernels,
    extract_observed_quotes,
    load_initial_state,
    load_saved_source,
    prepare_observed_fit,
    price_observed_quotes,
    quote_interpolation,
    regrid_initial_state,
)


def test_observed_quotes_keep_contract_widths_and_fair_values(monkeypatch) -> None:
    monkeypatch.setattr(
        "quantdata.gromm2.observed_calibration.ArbFreePricer.from_dict",
        lambda _: SimpleNamespace(interpolate_atm_vols=lambda _: 0.2),
    )
    nms = np.asarray([-4.1, -3.8, -2.0, 0.0, 2.8, 3.1])
    strikes = NMS.from_nms(nms, t=1.0 / 252.0, vol=0.2, eps_vol=0.2)
    bids = np.arange(len(strikes), dtype=float) * 0.001 + 0.001
    asks = bids + np.arange(1, len(strikes) + 1) * 0.0002
    fair = bids + 0.75 * (asks - bids)
    payload = {
        "af_pricer": {},
        "market": {
            "expiries": np.asarray([1.0 / 252.0]),
            "strikes": [strikes], "is_american": [np.zeros(len(strikes), dtype=bool)],
            "is_am": [False],
            **{f"{side}_{field}": [value] for side in ("call", "put") for field, value in (
                ("bids", bids), ("asks", asks), ("active", np.ones(len(strikes), dtype=bool)),
            )},
            "af_call_mids": [fair], "af_put_mids": [fair + strikes - 1.0],
        },
    }
    actual = extract_observed_quotes(payload, geometry=Gromm2GeometryConfig())
    selected = np.asarray([1, 2, 3, 4])
    np.testing.assert_array_equal(actual["strikes"], np.tile(strikes[selected], 2))
    np.testing.assert_array_equal(actual["half_spreads"], np.tile(0.5 * (asks - bids)[selected], 2))
    np.testing.assert_array_equal(actual["fair_values"][:4], fair[selected])
    np.testing.assert_array_equal(actual["quote_weights"], np.tile([0.1, 1.0, 1.0, 0.1], 2))
    assert not np.allclose(actual["fair_values"][:4], 0.5 * (bids + asks)[selected])


def test_quote_interpolation_retains_am_clock_and_snaps_float32_node() -> None:
    model = np.arange(1.0, 26.0) / 252.0
    queries = np.asarray([0.8, 1.00000006, 13.2, 25.0]) / 252.0
    lower, upper, alpha, before = quote_interpolation(model, queries)
    np.testing.assert_array_equal(lower, [0, 0, 12, 23])
    np.testing.assert_array_equal(upper, [0, 0, 13, 24])
    np.testing.assert_allclose(alpha, [0.8, 1.0, 0.2, 1.0], atol=1.0e-12)
    np.testing.assert_array_equal(before, [True, True, False, False])
    with pytest.raises(ValueError, match="horizon"):
        quote_interpolation(model, np.asarray([26.0 / 252.0]))


def test_black_kernels_preserve_put_call_parity_on_each_support() -> None:
    support = np.asarray([[0.7, 1.0, 1.4], [0.7, 1.0, 1.4]])
    kernels = black_quote_kernels(
        supports=support, strikes=np.asarray([1.1, 1.1]),
        expiries_y=np.asarray([0.2, 0.2]), is_call=np.asarray([True, False]),
        annual_volatility=0.05,
    )
    np.testing.assert_allclose(kernels[0] - kernels[1], support[0] - 1.1, atol=1.0e-15)


def test_quote_prices_mix_laws_on_their_own_supports() -> None:
    density = jnp.asarray([[0.25, 0.5, 0.25], [0.5, 0.25, 0.25]])
    arrays = {
        "lower": jnp.asarray([0, 0]), "upper": jnp.asarray([0, 1]),
        "alpha": jnp.asarray([0.5, 0.25]), "before_first": jnp.asarray([True, False]),
        "front_prices": jnp.asarray([2.0, 0.0]),
        "lower_kernels": jnp.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        "upper_kernels": jnp.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
    }
    actual = np.asarray(price_observed_quotes(density, arrays))
    np.testing.assert_allclose(actual, [2.0, 0.75 * 2.0 + 0.25 * 2.25])


def test_regridding_is_only_a_log_strike_initial_guess() -> None:
    old = np.asarray([[0.5, 0.8, 1.0, 1.2, 1.5]])
    new = np.asarray([[0.5, 0.85, 1.05, 1.15, 1.5]])
    raw = 2.0 + 3.0 * np.log(old[:, 1:-1])
    actual = regrid_initial_state(raw, old, new)
    np.testing.assert_allclose(actual, 2.0 + 3.0 * np.log(new[:, 1:-1]), atol=1.0e-15)


def test_previous_date_weights_do_not_require_saved_quote_kernels(tmp_path) -> None:
    supports = np.asarray([[0.5, 0.8, 1.0, 1.2, 1.5]])
    raw = 2.0 + 3.0 * np.log(supports[:, 1:-1])
    np.savez(tmp_path / "fit.npz", raw=raw, supports=supports, date=np.asarray("2024-01-02"))
    actual, day = load_initial_state(tmp_path, target_date="2024-01-03", target_supports=supports)
    np.testing.assert_array_equal(actual, raw)
    assert day == "2024-01-02"
    assert not (tmp_path / "inputs.npz").exists()
    with pytest.raises(ValueError, match="future"):
        load_initial_state(tmp_path, target_date="2024-01-01", target_supports=supports)


def test_saved_shorter_sanos_model_prices_the_tower_without_source_refitting(tmp_path, monkeypatch) -> None:
    queried_expiries = []

    def saved_prices(*, expiries, strikes, mode):
        queried_expiries.append(np.asarray(expiries))
        return bs.price(k=np.asarray(strikes), vol=0.2, sqrtT=np.sqrt(expiries))

    pricer = SimpleNamespace(expiries=np.asarray([1.0, 60.0]) / 252.0, price=saved_prices)
    payload = {"schema_version": 4, "af_pricer": {}, "d": date(2025, 6, 3), "ticker": "SPX"}
    quotes = {"strikes": np.asarray([1.0, 1.1]), "expiries_y": np.asarray([30.0, 125.0]) / 252.0, "is_call": np.ones(2, dtype=bool)}
    monkeypatch.setattr("quantdata.gromm2.observed_calibration.SubDir.read", lambda *args, **kwargs: payload)
    monkeypatch.setattr("quantdata.gromm2.observed_calibration.ArbFreePricer.from_dict", lambda _: pricer)
    monkeypatch.setattr("quantdata.gromm2.observed_calibration._observed_market", lambda *args: {})
    monkeypatch.setattr("quantdata.gromm2.observed_calibration.extract_observed_quotes", lambda *args, **kwargs: quotes)
    monkeypatch.setattr("quantdata.gromm2.observed_calibration.enrich_day", lambda *args, **kwargs: pd.DataFrame({
        "tte_y": np.asarray([1.0, 125.0]) / 252.0, "fwd": [100.0, 100.0], "spot": [100.0, 100.0],
    }))
    data = prepare_observed_fit(tmp_path / "saved.h5", config=ObservedFitConfig(geometry=Gromm2GeometryConfig(num_inner_strikes=16)))
    assert float(data["source_terminal_business_dte"]) == pytest.approx(60.0)
    assert data["supports"].shape == (45, 18)
    assert np.max(queried_expiries[0]) == pytest.approx(125.0 / 252.0)
    assert np.all(np.isfinite(data["initial_raw"]))


def test_dataset_source_reference_reads_existing_parameters_and_checks_hash(tmp_path, monkeypatch) -> None:
    parameter = tmp_path / "sanos_state.parquet"
    parameter.write_bytes(b"existing-parameters")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({
        "schema": "gromm2-existing-sanos-dataset-reference-v1", "dataset_path": str(tmp_path),
        "date": "2024-08-28", "ticker": "SPX",
        "parameter_file_sha256": {parameter.name: hashlib.sha256(parameter.read_bytes()).hexdigest()},
    }))
    saved_day = SimpleNamespace(model_expiries_y=np.asarray([0.1, 0.2]), smoothness=0.25)
    monkeypatch.setattr("quantdata.gromm2.observed_calibration.load_public_sanos_day", lambda dataset, day: saved_day)
    payload, pricer = load_saved_source(reference)
    assert payload["d"] == date(2024, 8, 28)
    assert pricer.day is saved_day
    parameter.write_bytes(b"changed-parameters")
    with pytest.raises(ValueError, match="dataset changed"):
        load_saved_source(reference)
