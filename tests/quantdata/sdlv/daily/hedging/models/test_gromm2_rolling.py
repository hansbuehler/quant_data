from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta

import h5py
import numpy as np
import pytest

from quantdata.gromm2.market import PublicSanosDay
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    ROLLING_GROMM2_SCHEMA,
    RollingGromm2Config,
    load_rolling_gromm2,
    rolling_gromm2_paths,
    select_rolling_public_days,
)


def _public_day(trading_date: date) -> PublicSanosDay:
    return PublicSanosDay(
        trading_date=trading_date,
        model_expiries_y=np.asarray([1.0 / 252.0]),
        kernel_total_variances=np.asarray([0.2**2 / 252.0]),
        support_strikes=(np.asarray([0.8, 1.0, 1.2]),),
        densities=(np.asarray([0.1, 0.8, 0.1]),),
        smoothness=1.0,
        one_minute_y=1.0 / (252.0 * 390.0),
    )


def test_rolling_gromm2_selection_ends_at_as_of_without_future_dates() -> None:
    days = [_public_day(date(2023, 1, 2) + timedelta(days=index)) for index in range(7)]

    selected = select_rolling_public_days(
        days,
        as_of_date=date(2023, 1, 6),
        window_transitions=2,
    )

    assert [day.trading_date for day in selected] == [
        date(2023, 1, 4),
        date(2023, 1, 5),
        date(2023, 1, 6),
    ]


def test_rolling_gromm2_selection_requires_exact_as_of_state() -> None:
    days = [_public_day(date(2023, 1, 2) + timedelta(days=index)) for index in range(4)]

    with pytest.raises(ValueError, match="end at as-of"):
        select_rolling_public_days(
            days,
            as_of_date=date(2023, 1, 8),
            window_transitions=2,
        )


def test_complete_rolling_gromm2_loader_rejects_future_history(tmp_path) -> None:
    config = RollingGromm2Config()
    as_of = date(2023, 12, 29)
    paths = rolling_gromm2_paths(tmp_path, as_of_date=as_of)
    paths.directory.mkdir(parents=True)
    paths.source_history.write_bytes(b"source")
    paths.prepared_books.write_bytes(b"books")
    np.savez_compressed(
        paths.dynamics,
        physical_market_prices_of_risk=np.zeros((252, 16)),
    )
    dataset_manifest_sha256 = "a" * 64
    paths.dynamics.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "gromm2-estimate-v1",
                "factor_count": 16,
                "pair_count": 252,
                "explained_variance_ratio": 0.91,
                "physical_projection_fraction": 0.62,
            }
        ),
        encoding="utf-8",
    )
    dates = [
        as_of - timedelta(days=config.window_transitions - index)
        for index in range(config.window_transitions + 1)
    ]
    _write_affine_history(paths.affine_history, dates=dates, config=config)
    manifest = {
        "schema": ROLLING_GROMM2_SCHEMA,
        "status": "complete",
        "as_of_date": as_of.isoformat(),
        "training_start_date": dates[0].isoformat(),
        "training_end_date": as_of.isoformat(),
        "state_count": 253,
        "transition_count": 252,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "config": json.loads(json.dumps(asdict(config), sort_keys=True)),
        "source_history_sha256": file_sha256(paths.source_history),
        "prepared_books_sha256": file_sha256(paths.prepared_books),
        "affine_history_sha256": file_sha256(paths.affine_history),
        "dynamics_sha256": file_sha256(paths.dynamics),
        "dynamics_report_sha256": file_sha256(paths.dynamics.with_suffix(".json")),
    }
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_rolling_gromm2(
        tmp_path,
        as_of_date=as_of,
        config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )

    assert loaded.as_of_date == as_of
    assert loaded.training_dates[-1] == as_of
    assert loaded.percent_inside_spread == pytest.approx(83.5)
    assert loaded.dataset_manifest_sha256 == dataset_manifest_sha256
    assert loaded.maximum_daily_risk_price_norm == 0.0

    with pytest.raises(ValueError, match="stale or incompatible"):
        load_rolling_gromm2(
            tmp_path,
            as_of_date=as_of,
            config=config,
            dataset_manifest_sha256="b" * 64,
        )

    report_path = paths.dynamics.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["physical_projection_fraction"] = 0.1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest["dynamics_report_sha256"] = file_sha256(report_path)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="projection gate"):
        load_rolling_gromm2(
            tmp_path,
            as_of_date=as_of,
            config=config,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
    report["physical_projection_fraction"] = 0.62
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest["dynamics_report_sha256"] = file_sha256(report_path)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    np.savez_compressed(
        paths.dynamics,
        physical_market_prices_of_risk=np.full((252, 16), 16.0),
    )
    manifest["dynamics_sha256"] = file_sha256(paths.dynamics)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="daily risk-price gate"):
        load_rolling_gromm2(
            tmp_path,
            as_of_date=as_of,
            config=config,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
    np.savez_compressed(
        paths.dynamics,
        physical_market_prices_of_risk=np.zeros((252, 16)),
    )
    manifest["dynamics_sha256"] = file_sha256(paths.dynamics)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    dates[-1] = as_of + timedelta(days=1)
    _write_affine_history(paths.affine_history, dates=dates, config=config)
    manifest["affine_history_sha256"] = file_sha256(paths.affine_history)
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="dated geometry"):
        load_rolling_gromm2(
            tmp_path,
            as_of_date=as_of,
            config=config,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )


def _write_affine_history(path, *, dates: list[date], config: RollingGromm2Config) -> None:
    with h5py.File(path, "w") as stream:
        stream.attrs["schema"] = "gromm2-affine-dlv-history-v1"
        stream.attrs["percent_inside_spread"] = 83.5
        stream.create_dataset(
            "dates",
            data=np.asarray([value.isoformat() for value in dates], dtype="S10"),
        )
        stream.create_dataset("nms_nodes", data=config.geometry.nms_nodes)
