"""Load and validate the distributable hedging public-candidate dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from cdxcore.err import verify_inp

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS, count_business_days_between
from quantdata.hedging.data.normalization import (
    interpolate_positive_curve_at_business_dte,
)

HEDGING_PUBLIC_SCHEMA = "hedging-public-candidate-v1"

DATA_FILE_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "observations.parquet": {
        "date": pl.Date,
        "trading_session_is_half_day": pl.Boolean,
        "business_sessions_from_previous_observation": pl.Int16,
        "forward_carry_from_previous_observation": pl.Float64,
        "discount_factor_from_previous_observation": pl.Float64,
        "forward_adjusted_spot_return_into_date": pl.Float64,
    },
    "expiries.parquet": {
        "date": pl.Date,
        "expiry_dates": pl.List(pl.Date),
        "expiries_y": pl.List(pl.Float64),
        "business_dtes": pl.List(pl.Float64),
        "calendar_dtes": pl.List(pl.Int16),
        "is_am_settlement": pl.List(pl.Boolean),
        "forward_over_spot": pl.List(pl.Float64),
        "discount_factors": pl.List(pl.Float64),
        "book_selection_epsilon_annualized_atm_vol": pl.Float64,
        "book_selection_annualized_atm_vols": pl.List(pl.Float64),
        "sanos_annualized_atm_vols": pl.List(pl.Float64),
        "dlv_annualized_atm_vols": pl.List(pl.Float64),
    },
    "instruments.parquet": {
        "date": pl.Date,
        "expiry_indices": pl.List(pl.Int16),
        "strikes_over_forward": pl.List(pl.Float64),
        "standardized_log_moneyness": pl.List(pl.Float64),
        "in_paper_retained_population": pl.List(pl.Boolean),
        "in_recommended_hedge_nms_population": pl.List(pl.Boolean),
        "is_call": pl.List(pl.Boolean),
        "half_spreads_over_spot": pl.List(pl.Float64),
        "sanos_prices_over_discounted_forward": pl.List(pl.Float64),
        "dlv_prices_over_discounted_forward": pl.List(pl.Float64),
        "sanos_annualized_implied_vols": pl.List(pl.Float64),
        "dlv_annualized_implied_vols": pl.List(pl.Float64),
        "sanos_implied_vol_is_valid": pl.List(pl.Boolean),
        "dlv_implied_vol_is_valid": pl.List(pl.Boolean),
    },
    "sanos_geometry.parquet": {
        "date": pl.Date,
        "model_expiries_y": pl.List(pl.Float64),
        "support_strikes": pl.List(pl.List(pl.Float64)),
    },
    "sanos_state.parquet": {
        "date": pl.Date,
        "kernel_total_variances": pl.List(pl.Float64),
        "densities": pl.List(pl.List(pl.Float64)),
    },
    "dlv_geometry.parquet": {
        "date": pl.Date,
        "model_expiries_y": pl.List(pl.Float64),
        "support_strikes": pl.List(pl.Float64),
    },
    "dlv_state.parquet": {
        "date": pl.Date,
        "raw_log_excess_dlvsq": pl.List(pl.Float32),
    },
}

DATA_FILE_COLUMNS = {
    filename: tuple(schema) for filename, schema in DATA_FILE_SCHEMAS.items()
}
METADATA_FILES = (
    "config.json",
    "geometry.json",
    "README.md",
    "DATA_DICTIONARY.md",
)
CHECKSUM_FILES = (*DATA_FILE_COLUMNS, *METADATA_FILES)
PUBLIC_DATASET_FILES = (*CHECKSUM_FILES, "manifest.json")

FORBIDDEN_COLUMN_FRAGMENTS = (
    "absolute_spot",
    "ask_price",
    "bid_price",
    "contract_id",
    "market_mid",
    "mid_price",
    "open_interest",
    "option_id",
    "quote_id",
    "source_row",
    "spot_ref",
    "symbol",
    "vendor_greek",
    "vendor_iv",
    "volume",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a non-empty finite vector, found {vector.shape}")
    return vector


def _equal_lengths(row: dict[str, Any], names: tuple[str, ...], *, context: str) -> int:
    lengths = {name: len(row[name]) for name in names}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{context}: nested columns do not align: {lengths}")
    length = next(iter(lengths.values()))
    if length == 0:
        raise ValueError(f"{context}: nested columns must be non-empty")
    return length


@dataclass(frozen=True)
class PublicHedgingDataset:
    """Independently loadable view of one public-candidate dataset directory."""

    directory: Path

    def __post_init__(self) -> None:
        directory = Path(self.directory)
        verify_inp(directory.is_dir(), lambda: f"Dataset directory does not exist: {directory}")
        object.__setattr__(self, "directory", directory)

    @property
    def manifest(self) -> dict[str, object]:
        """Read the immutable dataset manifest."""
        return self.read_json("manifest.json")

    @property
    def manifest_sha256(self) -> str:
        """Return the exact immutable manifest identity."""
        return _sha256(self.directory / "manifest.json")

    @property
    def config(self) -> dict[str, object]:
        """Read the numerical export configuration."""
        return self.read_json("config.json")

    @property
    def geometry(self) -> dict[str, object]:
        """Read the units and reconstruction geometry."""
        return self.read_json("geometry.json")

    def read(self, filename: str) -> pl.DataFrame:
        """Read one allowlisted Parquet member."""
        verify_inp(
            filename in DATA_FILE_COLUMNS,
            lambda: f"Unknown public dataset member {filename!r}",
        )
        return pl.read_parquet(self.directory / filename)

    def read_json(self, filename: str) -> dict[str, object]:
        """Read one allowlisted JSON metadata member."""
        verify_inp(
            filename in {"config.json", "geometry.json", "manifest.json"},
            lambda: f"Unknown public JSON member {filename!r}",
        )
        payload = json.loads((self.directory / filename).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{filename} must contain a JSON object")
        return payload


def _validate_metadata(dataset: PublicHedgingDataset) -> tuple[dict[str, object], dict[str, object]]:
    config = dataset.config
    geometry = dataset.geometry
    if config.get("schema") != HEDGING_PUBLIC_SCHEMA:
        raise ValueError(f"Unexpected config schema {config.get('schema')!r}")
    if geometry.get("schema") != HEDGING_PUBLIC_SCHEMA:
        raise ValueError(f"Unexpected geometry schema {geometry.get('schema')!r}")
    required_geometry = {
        "days_per_year",
        "normalized_strike_unit",
        "model_price_unit",
        "half_spread_unit",
        "normalized_forward_unit",
        "dlv_state_shape",
        "minimum_dlvsq",
        "maximum_dlvsq",
        "backbone_annual_volatility",
        "sanos_smoothness",
        "sanos_density_kind",
        "sanos_terminal_extrapolation",
        "one_minute_y",
        "released_book_nms_range",
        "paper_retained_book_nms_range",
        "recommended_hedge_nms_range",
        "dlv_fit_grid_nms_range",
    }
    missing = required_geometry - set(geometry)
    if missing:
        raise ValueError(f"geometry.json is missing fields {sorted(missing)}")
    if geometry["days_per_year"] != DAYS_PER_YEAR:
        raise ValueError(f"Unexpected business-year basis {geometry['days_per_year']!r}")
    expected_units = {
        "normalized_strike_unit": "K/F",
        "model_price_unit": "cash_price/(DF*F)",
        "half_spread_unit": "half_spread/S",
        "normalized_forward_unit": "F/S",
    }
    for name, expected in expected_units.items():
        if geometry.get(name) != expected:
            raise ValueError(f"Unexpected {name}: {geometry.get(name)!r}")
    return config, geometry


def _validate_tabular_structure(
    dataset: PublicHedgingDataset,
    *,
    verify_checksums: bool,
) -> tuple[dict[str, pl.DataFrame], dict[str, int]]:
    manifest = dataset.manifest
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict) or set(checksums) != set(CHECKSUM_FILES):
        raise ValueError("Manifest checksums must cover every public member except manifest.json")

    visible_members = {
        path.name for path in dataset.directory.iterdir() if not path.name.startswith(".")
    }
    if visible_members != set(PUBLIC_DATASET_FILES):
        raise ValueError(
            "Unexpected public dataset members: "
            f"missing={sorted(set(PUBLIC_DATASET_FILES) - visible_members)}, "
            f"extra={sorted(visible_members - set(PUBLIC_DATASET_FILES))}"
        )
    if verify_checksums:
        for filename in CHECKSUM_FILES:
            if checksums.get(filename) != _sha256(dataset.directory / filename):
                raise ValueError(f"Checksum mismatch for {filename}")

    frames: dict[str, pl.DataFrame] = {}
    row_counts: dict[str, int] = {}
    expected_dates: pl.Series | None = None
    for filename, expected_schema in DATA_FILE_SCHEMAS.items():
        path = dataset.directory / filename
        actual_schema = dict(pl.read_parquet_schema(path))
        if actual_schema != expected_schema:
            raise ValueError(
                f"Unexpected schema for {filename}: expected {expected_schema}, "
                f"found {actual_schema}"
            )
        violations = sorted(
            column
            for column in actual_schema
            if any(fragment in column.lower() for fragment in FORBIDDEN_COLUMN_FRAGMENTS)
        )
        if violations:
            raise ValueError(f"Forbidden public columns in {filename}: {violations}")
        frame = dataset.read(filename)
        dates = frame["date"]
        if dates.n_unique() != frame.height or not dates.is_sorted():
            raise ValueError(f"Dates in {filename} must be unique and sorted")
        if expected_dates is None:
            expected_dates = dates
        elif not expected_dates.equals(dates):
            raise ValueError(f"Dates in {filename} do not align with observations")
        frames[filename] = frame
        row_counts[filename] = frame.height
    return frames, row_counts


def _validate_observations(frame: pl.DataFrame, expiry_frame: pl.DataFrame) -> None:
    for index, row in enumerate(frame.iter_rows(named=True)):
        sessions = row["business_sessions_from_previous_observation"]
        carry = row["forward_carry_from_previous_observation"]
        discount = row["discount_factor_from_previous_observation"]
        forward_return = row["forward_adjusted_spot_return_into_date"]
        values = (sessions, carry, discount, forward_return)
        if index == 0:
            if any(value is not None for value in values):
                raise ValueError("The first observation must have null interval-return fields")
            continue
        previous_date = frame["date"][index - 1]
        expected_sessions = count_business_days_between(previous_date, row["date"])
        if sessions is None or int(sessions) != expected_sessions:
            raise ValueError(f"{row['date']}: invalid previous-observation session count")
        if carry is None or not np.isfinite(carry) or carry <= 0.0:
            raise ValueError(f"{row['date']}: invalid forward carry")
        if discount is None or not np.isfinite(discount) or discount <= 0.0:
            raise ValueError(f"{row['date']}: invalid interval discount factor")
        if forward_return is None or not np.isfinite(forward_return) or forward_return <= -1.0:
            raise ValueError(f"{row['date']}: invalid forward-adjusted spot return")
        previous_expiries = expiry_frame.row(index - 1, named=True)
        expected_carry = interpolate_positive_curve_at_business_dte(
            business_dtes=previous_expiries["business_dtes"],
            values=previous_expiries["forward_over_spot"],
            target_business_dte=expected_sessions,
            name="forward-over-spot",
        )
        expected_discount = interpolate_positive_curve_at_business_dte(
            business_dtes=previous_expiries["business_dtes"],
            values=previous_expiries["discount_factors"],
            target_business_dte=expected_sessions,
            name="discount-factor",
        )
        if not np.isclose(carry, expected_carry, rtol=0.0, atol=2.0e-14):
            raise ValueError(f"{row['date']}: stored forward carry does not match expiry curve")
        if not np.isclose(discount, expected_discount, rtol=0.0, atol=2.0e-14):
            raise ValueError(
                f"{row['date']}: stored interval discount factor does not match expiry curve"
            )


def _validate_day_and_reconstruction(
    *,
    row_index: int,
    frames: dict[str, pl.DataFrame],
    geometry: dict[str, object],
) -> None:
    from quantdata.hedging.data.pricing import (
        decode_dlv_density_tower,
        implied_volatilities_with_validity,
        price_dlv_surface,
        price_sanos_surface,
    )

    expiry_row = frames["expiries.parquet"].row(row_index, named=True)
    instrument_row = frames["instruments.parquet"].row(row_index, named=True)
    sanos_geometry = frames["sanos_geometry.parquet"].row(row_index, named=True)
    sanos_state = frames["sanos_state.parquet"].row(row_index, named=True)
    dlv_geometry = frames["dlv_geometry.parquet"].row(row_index, named=True)
    dlv_state = frames["dlv_state.parquet"].row(row_index, named=True)
    trading_date: date = expiry_row["date"]

    expiry_fields = (
        "expiry_dates",
        "expiries_y",
        "business_dtes",
        "calendar_dtes",
        "is_am_settlement",
        "forward_over_spot",
        "discount_factors",
        "book_selection_annualized_atm_vols",
        "sanos_annualized_atm_vols",
        "dlv_annualized_atm_vols",
    )
    expiry_count = _equal_lengths(expiry_row, expiry_fields, context=str(trading_date))
    expiries = _finite_vector(expiry_row["expiries_y"], name="expiries_y")
    business_dtes = _finite_vector(expiry_row["business_dtes"], name="business_dtes")
    forwards = _finite_vector(expiry_row["forward_over_spot"], name="forward_over_spot")
    discounts = _finite_vector(expiry_row["discount_factors"], name="discount_factors")
    selection_atm_vols = _finite_vector(
        expiry_row["book_selection_annualized_atm_vols"],
        name="book_selection_annualized_atm_vols",
    )
    selection_epsilon_atm_vol = float(
        expiry_row["book_selection_epsilon_annualized_atm_vol"]
    )
    sanos_atm_vols = _finite_vector(
        expiry_row["sanos_annualized_atm_vols"], name="sanos_annualized_atm_vols"
    )
    dlv_atm_vols = _finite_vector(
        expiry_row["dlv_annualized_atm_vols"], name="dlv_annualized_atm_vols"
    )
    if (
        np.any(expiries <= 0.0)
        or np.any(np.diff(expiries) <= 0.0)
        or np.any(np.abs(business_dtes - DAYS_PER_YEAR * expiries) > 2.0e-6)
        or np.any(forwards <= 0.0)
        or np.any(discounts <= 0.0)
        or not np.isfinite(selection_epsilon_atm_vol)
        or selection_epsilon_atm_vol <= 0.0
        or np.any(selection_atm_vols <= 0.0)
        or np.any(sanos_atm_vols <= 0.0)
        or np.any(dlv_atm_vols <= 0.0)
    ):
        raise ValueError(f"{trading_date}: invalid expiry curve")
    expiry_dates = expiry_row["expiry_dates"]
    calendar_dtes = np.asarray(expiry_row["calendar_dtes"], dtype=np.int64)
    if any(expiry_date <= trading_date for expiry_date in expiry_dates):
        raise ValueError(f"{trading_date}: expiry dates must follow the trading date")
    if not np.array_equal(
        calendar_dtes,
        np.asarray([(expiry_date - trading_date).days for expiry_date in expiry_dates]),
    ):
        raise ValueError(f"{trading_date}: calendar DTE values do not match expiry dates")

    instrument_fields = (
        "expiry_indices",
        "strikes_over_forward",
        "standardized_log_moneyness",
        "in_paper_retained_population",
        "in_recommended_hedge_nms_population",
        "is_call",
        "half_spreads_over_spot",
        "sanos_prices_over_discounted_forward",
        "dlv_prices_over_discounted_forward",
        "sanos_annualized_implied_vols",
        "dlv_annualized_implied_vols",
        "sanos_implied_vol_is_valid",
        "dlv_implied_vol_is_valid",
    )
    _equal_lengths(instrument_row, instrument_fields, context=str(trading_date))
    expiry_indices = np.asarray(instrument_row["expiry_indices"], dtype=np.intp)
    strikes = _finite_vector(instrument_row["strikes_over_forward"], name="strikes")
    stored_nms = _finite_vector(
        instrument_row["standardized_log_moneyness"],
        name="standardized_log_moneyness",
    )
    stored_paper_population = np.asarray(
        instrument_row["in_paper_retained_population"], dtype=bool
    )
    stored_hedge_population = np.asarray(
        instrument_row["in_recommended_hedge_nms_population"], dtype=bool
    )
    sides = np.asarray(instrument_row["is_call"], dtype=bool)
    half_spreads = _finite_vector(
        instrument_row["half_spreads_over_spot"], name="half_spreads_over_spot"
    )
    sanos_prices = _finite_vector(
        instrument_row["sanos_prices_over_discounted_forward"], name="sanos_prices"
    )
    dlv_prices = _finite_vector(
        instrument_row["dlv_prices_over_discounted_forward"], name="dlv_prices"
    )
    if (
        np.any((expiry_indices < 0) | (expiry_indices >= expiry_count))
        or np.any(strikes <= 0.0)
        or np.any(half_spreads <= 0.0)
    ):
        raise ValueError(f"{trading_date}: invalid instrument geometry")
    intrinsic = np.where(
        sides,
        np.maximum(1.0 - strikes, 0.0),
        np.maximum(strikes - 1.0, 0.0),
    )
    upper = np.where(sides, 1.0, strikes)
    for name, prices in (("SANOS", sanos_prices), ("DLV", dlv_prices)):
        if np.any(prices < intrinsic - 2.0e-8) or np.any(prices > upper + 2.0e-8):
            raise ValueError(f"{trading_date}: {name} prices violate Black bounds")

    sanos_model_expiries = _finite_vector(
        sanos_geometry["model_expiries_y"], name="SANOS model expiries"
    )
    sanos_variances = _finite_vector(
        sanos_state["kernel_total_variances"], name="SANOS kernel total variances"
    )
    sanos_supports = [
        np.asarray(value, dtype=np.float64) for value in sanos_geometry["support_strikes"]
    ]
    sanos_densities = [
        np.asarray(value, dtype=np.float64) for value in sanos_state["densities"]
    ]
    if not (
        len(sanos_model_expiries)
        == len(sanos_variances)
        == len(sanos_supports)
        == len(sanos_densities)
    ):
        raise ValueError(f"{trading_date}: SANOS state and geometry do not align")

    quote_expiries = expiries[expiry_indices]
    calculated_nms = np.asarray(
        NMS.to_nms(
            strikes,
            t=quote_expiries,
            vol=selection_atm_vols[expiry_indices],
            eps_vol=selection_epsilon_atm_vol,
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(calculated_nms, stored_nms, rtol=0.0, atol=2.0e-10)
    released_range = np.asarray(geometry["released_book_nms_range"], dtype=np.float64)
    paper_range = np.asarray(
        geometry["paper_retained_book_nms_range"], dtype=np.float64
    )
    hedge_range = np.asarray(geometry["recommended_hedge_nms_range"], dtype=np.float64)
    if np.any(stored_nms < released_range[0] - 2.0e-8) or np.any(
        stored_nms > released_range[1] + 2.0e-8
    ):
        raise ValueError(f"{trading_date}: instrument lies outside declared NMS screen")
    expected_paper_population = (stored_nms >= paper_range[0] - 2.0e-10) & (
        stored_nms <= paper_range[1] + 2.0e-10
    )
    expected_hedge_population = (stored_nms > hedge_range[0]) & (
        stored_nms < hedge_range[1]
    )
    if not np.array_equal(stored_paper_population, expected_paper_population):
        raise ValueError(f"{trading_date}: paper-population mask is inconsistent")
    if not np.array_equal(stored_hedge_population, expected_hedge_population):
        raise ValueError(f"{trading_date}: hedge-population mask is inconsistent")
    reconstructed_sanos = price_sanos_surface(
        query_expiries_y=quote_expiries,
        query_strikes=strikes,
        is_call=sides,
        model_expiries_y=sanos_model_expiries,
        atm_variances=sanos_variances,
        support_strikes=sanos_supports,
        densities=sanos_densities,
        smoothness=float(geometry["sanos_smoothness"]),
        one_minute_y=float(geometry["one_minute_y"]),
    )
    np.testing.assert_allclose(reconstructed_sanos, sanos_prices, rtol=0.0, atol=2.0e-10)

    dlv_model_expiries = _finite_vector(
        dlv_geometry["model_expiries_y"], name="DLV model expiries"
    )
    dlv_support = _finite_vector(dlv_geometry["support_strikes"], name="DLV support")
    dlv_shape = tuple(int(value) for value in geometry["dlv_state_shape"])
    raw_dlv = np.asarray(dlv_state["raw_log_excess_dlvsq"], dtype=np.float64)
    if raw_dlv.size != int(np.prod(dlv_shape)) or dlv_shape != (
        len(dlv_model_expiries),
        len(dlv_support) - 2,
    ):
        raise ValueError(f"{trading_date}: DLV state shape does not match its geometry")
    raw_dlv = raw_dlv.reshape(dlv_shape)
    decode_dlv_density_tower(
        support_strikes=dlv_support,
        raw_log_excess_dlvsq=raw_dlv,
        model_expiries_y=dlv_model_expiries,
        minimum_dlvsq=float(geometry["minimum_dlvsq"]),
        maximum_dlvsq=float(geometry["maximum_dlvsq"]),
    )
    reconstructed_dlv = price_dlv_surface(
        query_expiries_y=quote_expiries,
        query_strikes=strikes,
        is_call=sides,
        support_strikes=dlv_support,
        raw_log_excess_dlvsq=raw_dlv,
        model_expiries_y=dlv_model_expiries,
        minimum_dlvsq=float(geometry["minimum_dlvsq"]),
        maximum_dlvsq=float(geometry["maximum_dlvsq"]),
        backbone_annual_volatility=float(geometry["backbone_annual_volatility"]),
    )
    np.testing.assert_allclose(reconstructed_dlv, dlv_prices, rtol=0.0, atol=2.0e-10)

    for prefix, prices, stored_vols, stored_valid in (
        (
            "sanos",
            sanos_prices,
            instrument_row["sanos_annualized_implied_vols"],
            instrument_row["sanos_implied_vol_is_valid"],
        ),
        (
            "dlv",
            dlv_prices,
            instrument_row["dlv_annualized_implied_vols"],
            instrument_row["dlv_implied_vol_is_valid"],
        ),
    ):
        calculated_vols, calculated_valid = implied_volatilities_with_validity(
            expiries_y=quote_expiries,
            normalized_strikes=strikes,
            normalized_prices=prices,
            is_call=sides,
        )
        np.testing.assert_allclose(calculated_vols, stored_vols, rtol=0.0, atol=2.0e-10)
        if not np.array_equal(calculated_valid, np.asarray(stored_valid, dtype=bool)):
            raise ValueError(f"{trading_date}: {prefix} implied-volatility validity differs")

    atm_strikes = np.ones(expiry_count, dtype=np.float64)
    atm_sides = np.ones(expiry_count, dtype=bool)
    sanos_atm_prices = price_sanos_surface(
        query_expiries_y=expiries,
        query_strikes=atm_strikes,
        is_call=atm_sides,
        model_expiries_y=sanos_model_expiries,
        atm_variances=sanos_variances,
        support_strikes=sanos_supports,
        densities=sanos_densities,
        smoothness=float(geometry["sanos_smoothness"]),
        one_minute_y=float(geometry["one_minute_y"]),
    )
    dlv_atm_prices = price_dlv_surface(
        query_expiries_y=expiries,
        query_strikes=atm_strikes,
        is_call=atm_sides,
        support_strikes=dlv_support,
        raw_log_excess_dlvsq=raw_dlv,
        model_expiries_y=dlv_model_expiries,
        minimum_dlvsq=float(geometry["minimum_dlvsq"]),
        maximum_dlvsq=float(geometry["maximum_dlvsq"]),
        backbone_annual_volatility=float(geometry["backbone_annual_volatility"]),
    )
    for name, prices, expected_vols in (
        ("SANOS", sanos_atm_prices, sanos_atm_vols),
        ("DLV", dlv_atm_prices, dlv_atm_vols),
    ):
        calculated_vols, valid = implied_volatilities_with_validity(
            expiries_y=expiries,
            normalized_strikes=atm_strikes,
            normalized_prices=prices,
            is_call=atm_sides,
        )
        if not np.all(valid):
            raise ValueError(f"{trading_date}: {name} ATM implied volatility is not interior")
        np.testing.assert_allclose(calculated_vols, expected_vols, rtol=0.0, atol=2.0e-10)


def validate_public_dataset(
    directory: Path,
    *,
    verify_checksums: bool = True,
    reconstruct_models: bool = True,
) -> dict[str, object]:
    """Validate public schema, leakage controls, invariants, and reconstruction."""
    dataset = PublicHedgingDataset(Path(directory))
    manifest = dataset.manifest
    if manifest.get("schema") != HEDGING_PUBLIC_SCHEMA:
        raise ValueError(f"Unexpected dataset schema {manifest.get('schema')!r}")
    distribution = manifest.get("distribution")
    if distribution not in {"public_candidate", "public"}:
        raise ValueError(f"Invalid distribution status {distribution!r}")
    if distribution == "public":
        required_release_fields = (
            "release_license",
            "rights_basis",
            "approver",
            "approval_date",
            "approval_reference",
        )
        missing = [name for name in required_release_fields if not manifest.get(name)]
        if missing:
            raise ValueError(f"Public distribution requires release metadata {missing}")
        if manifest.get("generator_worktree_dirty") is not False:
            raise ValueError("Public distribution requires a clean generator worktree")
    generator_checksums = manifest.get("generator_code_sha256")
    if not isinstance(generator_checksums, dict) or not generator_checksums:
        raise ValueError("Manifest must contain generator source checksums")
    if any(
        not isinstance(filename, str)
        or not isinstance(checksum, str)
        or len(checksum) != 64
        for filename, checksum in generator_checksums.items()
    ):
        raise ValueError("Manifest contains an invalid generator source checksum")
    if not isinstance(manifest.get("generator_worktree_dirty"), bool):
        raise ValueError("Manifest generator_worktree_dirty must be boolean")

    _, geometry = _validate_metadata(dataset)
    frames, row_counts = _validate_tabular_structure(
        dataset,
        verify_checksums=verify_checksums,
    )
    expected_count = manifest.get("num_dates")
    if not isinstance(expected_count, int) or any(
        count != expected_count for count in row_counts.values()
    ):
        raise ValueError(
            f"Manifest date count {expected_count!r} does not match {row_counts}"
        )
    instrument_frame = frames["instruments.parquet"]
    population_counts = {
        "num_quote_side_books": sum(
            len(values) for values in instrument_frame["expiry_indices"]
        ),
        "num_paper_retained_quote_side_books": sum(
            sum(values) for values in instrument_frame["in_paper_retained_population"]
        ),
        "num_observation_hedge_nms_quote_side_books": sum(
            sum(values)
            for values in instrument_frame["in_recommended_hedge_nms_population"]
        ),
    }
    for name, calculated in population_counts.items():
        if manifest.get(name) != calculated:
            raise ValueError(
                f"Manifest {name}={manifest.get(name)!r} does not match {calculated}"
            )
    _validate_observations(
        frames["observations.parquet"],
        frames["expiries.parquet"],
    )
    if reconstruct_models:
        for row_index in range(int(expected_count)):
            _validate_day_and_reconstruction(
                row_index=row_index,
                frames=frames,
                geometry=geometry,
            )
    return manifest
