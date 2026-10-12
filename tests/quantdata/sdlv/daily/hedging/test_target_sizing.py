from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.hedging.data.public_dataset import HEDGING_PUBLIC_SCHEMA
from quantdata.sdlv.daily.front_spot.hedging.benchmark import sizing
from quantdata.sdlv.daily.front_spot.hedging.benchmark.sizing import (
    DailyTargetNotionalCalibration,
    TargetNotionalConfig,
    calibrate_daily_target_notionals_from_returns,
    daily_target_notional_artifact_path,
    daily_target_notional_file_sha256,
    load_authenticated_daily_target_notionals,
    load_or_calibrate_daily_target_notionals,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.returns import (
    HistoricalReturnPanel,
)


class _Dataset:
    def __init__(self, directory: Path) -> None:
        self.directory = directory


class _CachePanel(PublicHedgingPanel):
    def __init__(self, directory: Path) -> None:
        self.geometry = {"schema": HEDGING_PUBLIC_SCHEMA}
        self.dataset = _Dataset(directory)
        self.dates = (date(2022, 12, 29), date(2022, 12, 30))

    def one_day_transition(self, origin_date: date) -> SimpleNamespace:
        assert origin_date == self.dates[0]
        return SimpleNamespace(next_date=self.dates[1])

    def market_day(self, trading_date: date) -> SimpleNamespace:
        assert trading_date == self.dates[1]
        quote = SimpleNamespace(half_spread=4.0e-4, business_dte=1.0)
        snapshot = SimpleNamespace(option_quotes=(quote,))
        return SimpleNamespace(
            atm_vol_at_business_dte=lambda maturity: 0.02 * np.sqrt(252.0),
            hedge_snapshot=lambda: snapshot,
        )


def _patch_single_cache_target(monkeypatch: pytest.MonkeyPatch) -> None:
    target = SimpleNamespace(label="target", maturity_business_days=1)
    monkeypatch.setattr(sizing, "daily_digital_targets", lambda: (target,))


def test_target_notionals_match_one_atm_spot_risk_unit() -> None:
    generator = np.random.default_rng(31)
    observations = 120
    instruments = generator.normal(scale=0.01, size=(observations, 3))
    instruments -= np.mean(instruments, axis=0, keepdims=True)
    targets = np.column_stack(
        (
            0.4 * instruments[:, 0] + generator.normal(scale=0.01, size=observations),
            -0.2 * instruments[:, 1] + generator.normal(scale=0.02, size=observations),
        )
    )
    targets -= np.mean(targets, axis=0, keepdims=True)
    first = date(2022, 1, 3)
    origins = tuple(first + timedelta(days=index) for index in range(observations))
    terminals = tuple(value + timedelta(days=1) for value in origins)
    returns = HistoricalReturnPanel(
        training_origin_dates=origins,
        training_terminal_dates=terminals,
        instrument_ids=("spot", "put", "call"),
        target_labels=("one-day", "three-day"),
        instrument_returns=instruments,
        target_returns=targets,
    )
    calibration = calibrate_daily_target_notionals_from_returns(
        returns,
        as_of_date=date(2022, 12, 30),
        target_maturities_business_days=np.asarray([1, 3]),
        reference_annual_volatilities=np.asarray([0.2, 0.25]),
        linear_transaction_costs=np.asarray([0.0, 0.0, 0.0]),
        config=TargetNotionalConfig(minimum_observations=100),
    )
    np.testing.assert_allclose(
        calibration.target_quantities * calibration.target_standard_deviations,
        calibration.reference_spot_standard_deviations,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    assert calibration.reference_spot_standard_deviations[0] == pytest.approx(
        0.2 / np.sqrt(252.0)
    )
    assert calibration.reference_spot_standard_deviations[1] == pytest.approx(
        0.25 * np.sqrt(3.0 / 252.0)
    )
    assert calibration.covariance_converged
    assert calibration.covariance_cap_hits == 0
    np.testing.assert_allclose(calibration.transaction_cost_fractions, 0.0)

    restored = type(calibration).from_dict(calibration.to_dict())
    assert restored.as_of_date == calibration.as_of_date
    assert restored.training_origin_dates == calibration.training_origin_dates
    assert restored.target_labels == calibration.target_labels
    np.testing.assert_array_equal(
        restored.target_maturities_business_days,
        calibration.target_maturities_business_days,
    )
    np.testing.assert_allclose(restored.target_quantities, calibration.target_quantities)
    np.testing.assert_allclose(restored.covariance_actions, calibration.covariance_actions)


def test_target_notionals_reject_degenerate_claims() -> None:
    origins = tuple(date(2022, 1, 3) + timedelta(days=index) for index in range(20))
    returns = HistoricalReturnPanel(
        training_origin_dates=origins,
        training_terminal_dates=tuple(value + timedelta(days=1) for value in origins),
        instrument_ids=("spot",),
        target_labels=("constant",),
        instrument_returns=np.linspace(-0.01, 0.01, 20)[:, None],
        target_returns=np.ones((20, 1)),
    )
    with pytest.raises(ValueError, match="standard deviation is too small"):
        calibrate_daily_target_notionals_from_returns(
            returns,
            as_of_date=date(2022, 12, 30),
            target_maturities_business_days=np.asarray([1]),
            reference_annual_volatilities=np.asarray([0.2]),
            linear_transaction_costs=np.asarray([0.0]),
            config=TargetNotionalConfig(minimum_observations=20),
        )


def test_target_notional_cache_is_dataset_and_config_keyed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_cache_target(monkeypatch)
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    panel = _CachePanel(tmp_path)
    calibration = DailyTargetNotionalCalibration(
        as_of_date=date(2022, 12, 30),
        training_origin_dates=(date(2022, 12, 29),),
        target_labels=("target",),
        target_maturities_business_days=np.asarray([1], dtype=np.int64),
        target_standard_deviations=np.asarray([0.1]),
        reference_spot_standard_deviations=np.asarray([0.02]),
        target_quantities=np.asarray([0.2]),
        covariance_actions=np.asarray([[0.5, -0.25]]),
        transaction_cost_fractions=np.asarray([0.01]),
        covariance_converged=True,
        covariance_cap_hits=0,
    )
    calls = 0

    def calibrate(_panel, *, config):
        nonlocal calls
        del config
        assert _panel is panel
        calls += 1
        return calibration

    monkeypatch.setattr(sizing, "calibrate_daily_target_notionals", calibrate)
    cache = tmp_path / "cache"
    first = load_or_calibrate_daily_target_notionals(
        panel,
        cache_directory=cache,
    )
    second = load_or_calibrate_daily_target_notionals(
        panel,
        cache_directory=cache,
    )
    assert calls == 2
    np.testing.assert_allclose(first.target_quantities, calibration.target_quantities)
    np.testing.assert_allclose(second.target_quantities, calibration.target_quantities)
    assert len(tuple(cache.glob("daily-target-notionals-*.json"))) == 1


def test_target_notional_cache_is_return_geometry_keyed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    panel = _CachePanel(tmp_path)
    cache = tmp_path / "cache"
    original = daily_target_notional_artifact_path(
        panel,
        cache_directory=cache,
    )

    monkeypatch.setattr(
        sizing,
        "COVARIANCE_RETURN_GEOMETRY_IDENTITY",
        "test-different-return-geometry",
    )
    changed = daily_target_notional_artifact_path(
        panel,
        cache_directory=cache,
    )

    assert changed != original


def test_authenticated_target_notional_load_uses_pinned_preflight_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_cache_target(monkeypatch)
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    panel = _CachePanel(tmp_path)
    calibration = DailyTargetNotionalCalibration(
        as_of_date=date(2022, 12, 30),
        training_origin_dates=(date(2022, 12, 29),),
        target_labels=("target",),
        target_maturities_business_days=np.asarray([1], dtype=np.int64),
        target_standard_deviations=np.asarray([0.1]),
        reference_spot_standard_deviations=np.asarray([0.02]),
        target_quantities=np.asarray([0.2]),
        covariance_actions=np.asarray([[0.5, -0.25]]),
        transaction_cost_fractions=np.asarray([0.01]),
        covariance_converged=True,
        covariance_cap_hits=0,
    )
    monkeypatch.setattr(
        sizing,
        "calibrate_daily_target_notionals",
        lambda _panel, *, config: calibration,
    )
    cache = tmp_path / "cache"
    load_or_calibrate_daily_target_notionals(panel, cache_directory=cache)
    artifact = daily_target_notional_artifact_path(
        panel,
        cache_directory=cache,
    )
    fingerprint = daily_target_notional_file_sha256(artifact)

    def fail_recomputation(*args, **kwargs):
        del args, kwargs
        raise AssertionError("authenticated load must not repeat calibration")

    monkeypatch.setattr(
        sizing,
        "calibrate_daily_target_notionals",
        fail_recomputation,
    )
    loaded = load_authenticated_daily_target_notionals(
        panel,
        cache_directory=cache,
        expected_artifact_sha256=fingerprint,
    )
    np.testing.assert_allclose(loaded.target_quantities, calibration.target_quantities)
    with pytest.raises(RuntimeError, match="changed after CPU preflight"):
        load_authenticated_daily_target_notionals(
            panel,
            cache_directory=cache,
            expected_artifact_sha256="0" * 64,
        )


def test_target_notional_cache_rejects_hashed_quantity_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_cache_target(monkeypatch)
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    panel = _CachePanel(tmp_path)
    calibration = DailyTargetNotionalCalibration(
        as_of_date=date(2022, 12, 30),
        training_origin_dates=(date(2022, 12, 29),),
        target_labels=("target",),
        target_maturities_business_days=np.asarray([1], dtype=np.int64),
        target_standard_deviations=np.asarray([0.1]),
        reference_spot_standard_deviations=np.asarray([0.02]),
        target_quantities=np.asarray([0.2]),
        covariance_actions=np.asarray([[0.5, -0.25]]),
        transaction_cost_fractions=np.asarray([0.01]),
        covariance_converged=True,
        covariance_cap_hits=0,
    )
    monkeypatch.setattr(
        sizing,
        "calibrate_daily_target_notionals",
        lambda _panel, *, config: calibration,
    )
    cache = tmp_path / "cache"
    load_or_calibrate_daily_target_notionals(panel, cache_directory=cache)
    path = next(cache.glob("daily-target-notionals-*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calibration"]["target_quantities"] = [999.0]
    payload["calibration_sha256"] = sizing._calibration_payload_sha256(
        payload["calibration"]
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent target quantities"):
        load_or_calibrate_daily_target_notionals(panel, cache_directory=cache)


def test_target_notional_cache_rejects_hashed_date_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_cache_target(monkeypatch)
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    panel = _CachePanel(tmp_path)
    calibration = DailyTargetNotionalCalibration(
        as_of_date=date(2022, 12, 30),
        training_origin_dates=(date(2022, 12, 29),),
        target_labels=("target",),
        target_maturities_business_days=np.asarray([1], dtype=np.int64),
        target_standard_deviations=np.asarray([0.1]),
        reference_spot_standard_deviations=np.asarray([0.02]),
        target_quantities=np.asarray([0.2]),
        covariance_actions=np.asarray([[0.5, -0.25]]),
        transaction_cost_fractions=np.asarray([0.01]),
        covariance_converged=True,
        covariance_cap_hits=0,
    )
    monkeypatch.setattr(
        sizing,
        "calibrate_daily_target_notionals",
        lambda _panel, *, config: calibration,
    )
    cache = tmp_path / "cache"
    load_or_calibrate_daily_target_notionals(panel, cache_directory=cache)
    path = next(cache.glob("daily-target-notionals-*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calibration"]["as_of_date"] = "2024-12-31"
    payload["calibration"]["training_origin_dates"] = ["2024-12-30"]
    payload["calibration_sha256"] = sizing._calibration_payload_sha256(
        payload["calibration"]
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected as-of date"):
        load_or_calibrate_daily_target_notionals(panel, cache_directory=cache)


def test_target_notional_cache_rejects_coupled_payload_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_cache_target(monkeypatch)
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    panel = _CachePanel(tmp_path)
    calibration = DailyTargetNotionalCalibration(
        as_of_date=date(2022, 12, 30),
        training_origin_dates=(date(2022, 12, 29),),
        target_labels=("target",),
        target_maturities_business_days=np.asarray([1], dtype=np.int64),
        target_standard_deviations=np.asarray([0.1]),
        reference_spot_standard_deviations=np.asarray([0.02]),
        target_quantities=np.asarray([0.2]),
        covariance_actions=np.asarray([[0.5, -0.25]]),
        transaction_cost_fractions=np.asarray([0.01]),
        covariance_converged=True,
        covariance_cap_hits=0,
    )
    monkeypatch.setattr(
        sizing,
        "calibrate_daily_target_notionals",
        lambda _panel, *, config: calibration,
    )
    cache = tmp_path / "cache"
    load_or_calibrate_daily_target_notionals(panel, cache_directory=cache)
    path = next(cache.glob("daily-target-notionals-*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calibration"]["target_quantities"] = [999.0]
    payload["calibration"]["target_standard_deviations"] = [0.02 / 999.0]
    payload["calibration"]["covariance_actions"] = [[0.0, 0.5]]
    payload["calibration"]["transaction_cost_fractions"] = [0.01]
    payload["calibration_sha256"] = sizing._calibration_payload_sha256(
        payload["calibration"]
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its public-data fit"):
        load_or_calibrate_daily_target_notionals(panel, cache_directory=cache)
