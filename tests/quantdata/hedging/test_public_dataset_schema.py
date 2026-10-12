from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quantdata.biz_calendar import DAYS_PER_YEAR, ONE_MINUTE
from quantdata.hedging.data.pricing import (
    implied_volatilities_with_validity,
    price_dlv_surface,
    price_sanos_surface,
)
from quantdata.hedging.data.public_dataset import (
    CHECKSUM_FILES,
    DATA_FILE_SCHEMAS,
    HEDGING_PUBLIC_SCHEMA,
    validate_public_dataset,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_minimal_dataset(directory: Path) -> None:
    trading_date = date(2024, 1, 2)
    expiry_date = date(2024, 1, 3)
    expiry = np.asarray([1.0 / DAYS_PER_YEAR])
    strike = np.asarray([1.0])
    side = np.asarray([True])

    sanos_support = [np.asarray([0.8, 1.0, 1.2])]
    sanos_density = [np.asarray([0.25, 0.5, 0.25])]
    sanos_variance = np.asarray([0.2**2 / DAYS_PER_YEAR])
    sanos_price = price_sanos_surface(
        query_expiries_y=expiry,
        query_strikes=strike,
        is_call=side,
        model_expiries_y=expiry,
        atm_variances=sanos_variance,
        support_strikes=sanos_support,
        densities=sanos_density,
        smoothness=0.25,
        one_minute_y=ONE_MINUTE,
    )
    sanos_vol, sanos_valid = implied_volatilities_with_validity(
        expiries_y=expiry,
        normalized_strikes=strike,
        normalized_prices=sanos_price,
        is_call=side,
    )

    dlv_support = np.asarray([0.6, 0.85, 1.0, 1.2, 1.6])
    raw_dlv = np.log(np.asarray([[0.03, 0.04, 0.05]])).astype(np.float32).astype(np.float64)
    dlv_price = price_dlv_surface(
        query_expiries_y=expiry,
        query_strikes=strike,
        is_call=side,
        support_strikes=dlv_support,
        raw_log_excess_dlvsq=raw_dlv,
        model_expiries_y=expiry,
        minimum_dlvsq=0.01**2,
        maximum_dlvsq=3.0**2,
        backbone_annual_volatility=0.05,
    )
    dlv_vol, dlv_valid = implied_volatilities_with_validity(
        expiries_y=expiry,
        normalized_strikes=strike,
        normalized_prices=dlv_price,
        is_call=side,
    )
    assert sanos_valid.item() and dlv_valid.item()

    rows = {
        "observations.parquet": {
            "date": [trading_date],
            "trading_session_is_half_day": [False],
            "business_sessions_from_previous_observation": [None],
            "forward_carry_from_previous_observation": [None],
            "discount_factor_from_previous_observation": [None],
            "forward_adjusted_spot_return_into_date": [None],
        },
        "expiries.parquet": {
            "date": [trading_date],
            "expiry_dates": [[expiry_date]],
            "expiries_y": [expiry.tolist()],
            "business_dtes": [[1.0]],
            "calendar_dtes": [[1]],
            "is_am_settlement": [[False]],
            "forward_over_spot": [[1.0001]],
            "discount_factors": [[0.9999]],
            "book_selection_epsilon_annualized_atm_vol": [0.2],
            "book_selection_annualized_atm_vols": [[0.2]],
            "sanos_annualized_atm_vols": [sanos_vol.tolist()],
            "dlv_annualized_atm_vols": [dlv_vol.tolist()],
        },
        "instruments.parquet": {
            "date": [trading_date],
            "expiry_indices": [[0]],
            "strikes_over_forward": [strike.tolist()],
            "standardized_log_moneyness": [[0.0]],
            "in_paper_retained_population": [[True]],
            "in_recommended_hedge_nms_population": [[True]],
            "is_call": [side.tolist()],
            "half_spreads_over_spot": [[0.001]],
            "sanos_prices_over_discounted_forward": [sanos_price.tolist()],
            "dlv_prices_over_discounted_forward": [dlv_price.tolist()],
            "sanos_annualized_implied_vols": [sanos_vol.tolist()],
            "dlv_annualized_implied_vols": [dlv_vol.tolist()],
            "sanos_implied_vol_is_valid": [sanos_valid.tolist()],
            "dlv_implied_vol_is_valid": [dlv_valid.tolist()],
        },
        "sanos_geometry.parquet": {
            "date": [trading_date],
            "model_expiries_y": [expiry.tolist()],
            "support_strikes": [[support.tolist() for support in sanos_support]],
        },
        "sanos_state.parquet": {
            "date": [trading_date],
            "kernel_total_variances": [sanos_variance.tolist()],
            "densities": [[density.tolist() for density in sanos_density]],
        },
        "dlv_geometry.parquet": {
            "date": [trading_date],
            "model_expiries_y": [expiry.tolist()],
            "support_strikes": [dlv_support.tolist()],
        },
        "dlv_state.parquet": {
            "date": [trading_date],
            "raw_log_excess_dlvsq": [raw_dlv.astype(np.float32).reshape(-1).tolist()],
        },
    }
    for filename, row in rows.items():
        pl.DataFrame(row, schema=DATA_FILE_SCHEMAS[filename], strict=True).write_parquet(
            directory / filename
        )

    geometry = {
        "schema": HEDGING_PUBLIC_SCHEMA,
        "days_per_year": DAYS_PER_YEAR,
        "normalized_strike_unit": "K/F",
        "model_price_unit": "cash_price/(DF*F)",
        "half_spread_unit": "half_spread/S",
        "normalized_forward_unit": "F/S",
        "dlv_state_shape": [1, 3],
        "minimum_dlvsq": 0.01**2,
        "maximum_dlvsq": 3.0**2,
        "backbone_annual_volatility": 0.05,
        "sanos_smoothness": 0.25,
        "sanos_density_kind": "surface",
        "sanos_terminal_extrapolation": (
            "last density with clipped-node annualized kernel volatility held constant"
        ),
        "one_minute_y": ONE_MINUTE,
        "released_book_nms_range": [-2.5, 2.0],
        "paper_retained_book_nms_range": [-2.5, 1.5],
        "recommended_hedge_nms_range": [-2.0, 2.0],
        "dlv_fit_grid_nms_range": [-2.5, 1.5],
    }
    (directory / "geometry.json").write_text(json.dumps(geometry), encoding="utf-8")
    (directory / "config.json").write_text(
        json.dumps({"schema": HEDGING_PUBLIC_SCHEMA}), encoding="utf-8"
    )
    (directory / "README.md").write_text("fixture\n", encoding="utf-8")
    (directory / "DATA_DICTIONARY.md").write_text(
        "fixture dictionary\n",
        encoding="utf-8",
    )
    checksums = {filename: _sha256(directory / filename) for filename in CHECKSUM_FILES}
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema": HEDGING_PUBLIC_SCHEMA,
                "distribution": "public_candidate",
                "num_dates": 1,
                "num_quote_side_books": 1,
                "num_paper_retained_quote_side_books": 1,
                "num_observation_hedge_nms_quote_side_books": 1,
                "generator_code_sha256": {"fixture.py": "0" * 64},
                "generator_worktree_dirty": True,
                "checksums": checksums,
            }
        ),
        encoding="utf-8",
    )


def test_public_dataset_schema_checksums_and_reconstruction(tmp_path: Path) -> None:
    _write_minimal_dataset(tmp_path)
    manifest = validate_public_dataset(tmp_path)
    assert manifest["num_dates"] == 1


def test_public_dataset_rejects_unexpected_columns(tmp_path: Path) -> None:
    _write_minimal_dataset(tmp_path)
    path = tmp_path / "observations.parquet"
    pl.read_parquet(path).with_columns(pl.lit(1.0).alias("market_mid")).write_parquet(path)
    with pytest.raises(ValueError, match="Unexpected schema"):
        validate_public_dataset(tmp_path, verify_checksums=False)


def test_public_distribution_requires_release_metadata(tmp_path: Path) -> None:
    _write_minimal_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["distribution"] = "public"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="requires release metadata"):
        validate_public_dataset(tmp_path)


def test_public_dataset_detects_model_price_tampering(tmp_path: Path) -> None:
    _write_minimal_dataset(tmp_path)
    path = tmp_path / "instruments.parquet"
    frame = pl.read_parquet(path)
    frame = frame.with_columns(
        pl.col("sanos_prices_over_discounted_forward")
        .list.eval(pl.element() + 1.0e-4)
        .alias("sanos_prices_over_discounted_forward")
    )
    frame.write_parquet(path)
    with pytest.raises(AssertionError):
        validate_public_dataset(tmp_path, verify_checksums=False)


def test_public_dataset_detects_population_mask_tampering(tmp_path: Path) -> None:
    _write_minimal_dataset(tmp_path)
    path = tmp_path / "instruments.parquet"
    frame = pl.read_parquet(path).with_columns(
        pl.Series(
            "in_paper_retained_population",
            [[False]],
            dtype=pl.List(pl.Boolean),
        )
    )
    frame.write_parquet(path)
    with pytest.raises(ValueError, match="num_paper_retained_quote_side_books"):
        validate_public_dataset(tmp_path, verify_checksums=False)
