"""Export a SANOS-compatible, WRDS-free hedging public-candidate dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
from cdxcore.err import verify_inp
from cdxcore.util import is_filename
from cdxcore.verbose import Context
from cdxcore.version import version
from numpy.typing import NDArray

from quantdata.biz_calendar import (
    DAYS_PER_YEAR,
    NMS,
    ONE_MINUTE,
    count_business_days_between,
    get_half_days,
)
from quantdata.dysanos.flex.config import (
    FlexGeometryConfig,
    FlexMarketConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.market import market_day_from_payload
from quantdata.dysanos.market import smooth_interpolated_market
from quantdata.hedging.data.normalization import (
    interpolate_positive_curve_at_business_dte,
)
from quantdata.hedging.data.pricing import (
    implied_volatilities_with_validity,
    price_dlv_surface,
    price_sanos_surface,
)
from quantdata.hedging.data.public_dataset import (
    DATA_FILE_COLUMNS,
    DATA_FILE_SCHEMAS,
    HEDGING_PUBLIC_SCHEMA,
    validate_public_dataset,
)
from quantdata.sanos.arbfree import ArbFreePricer
from quantdata.sdlv.daily import DATASET_PATH, load_raw_grids
from quantdata.widb.arbfree import MkArbFree, mk_arb_free_day
from quantdata.widb.day import ExpiryFilter, RawDayFilter
from quantdata.widb.options_filter import OptionsFilter

_MODEL_BUSINESS_DTES = tuple(float(value) for value in range(1, 26))
_MINIMUM_DLV_SQ = 0.01**2
_MAXIMUM_DLV_SQ = 3.0**2
_BACKBONE_ANNUAL_VOLATILITY = 0.05
_DLV_FIT_GRID_NMS_RANGE = (-2.5, 1.5)
_RELEASED_BOOK_NMS_RANGE = (-2.5, 2.0)
_PAPER_RETAINED_BOOK_NMS_RANGE = (-2.5, 1.5)
_RECOMMENDED_HEDGE_NMS_RANGE = (-2.0, 2.0)
_GENERATOR_SOURCE_FILES = (
    "quantdata/hedging/data/normalization.py",
    "quantdata/hedging/data/pricing.py",
    "quantdata/hedging/data/public_dataset.py",
    "quantdata/sdlv/daily/front_spot/hedging/data/export_public_candidate.py",
)


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class InterimPublicDatasetConfig:
    """Numerical identity of the interim SANOS-compatible public projection."""

    ticker: str = "SPX"
    fit_label: str = "affine-200"
    smoothness: float = 0.25
    max_business_dte: int = 25
    dataset_name: str = "spx-2022-2025-hedging-public-candidate-v1"

    def __post_init__(self) -> None:
        verify_inp(self.ticker == "SPX", lambda: f"Expected SPX, found {self.ticker!r}")
        verify_inp(
            self.fit_label == "affine-200",
            lambda: f"Expected retained affine-200 fit, found {self.fit_label!r}",
        )
        verify_inp(self.smoothness > 0.0, lambda: "smoothness must be positive")
        verify_inp(
            self.max_business_dte == 25,
            lambda: f"Expected the retained 25-day horizon, found {self.max_business_dte}",
        )
        verify_inp(
            is_filename(self.dataset_name),
            lambda: f"dataset_name must be a valid filename, found {self.dataset_name!r}",
        )


def _market_geometry() -> SharedStrikeCappedUnboundedDlvFlexGeometryConfig:
    return SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=200,
        nms_range=_RELEASED_BOOK_NMS_RANGE,
        num_model_expiries=25,
        first_model_business_dte=1.0,
        model_expiry_step_business_days=1.0,
        explicit_model_business_dtes=_MODEL_BUSINESS_DTES,
    )


def _options_filter() -> OptionsFilter:
    canonical_market_horizon = math.ceil(FlexGeometryConfig().max_business_dte)
    return OptionsFilter(
        max_num_options=100_000,
        max_business_days_to_expiry=canonical_market_horizon,
        max_nms=(-100.0, 100.0),
    )


def _filtered_source_day(
    ticker: str,
    trading_date: date,
    options_filter: OptionsFilter,
) -> Any:
    source = mk_arb_free_day(
        ticker=ticker,
        d=trading_date,
        raw_day_filter=RawDayFilter(),
        expiry_filter=ExpiryFilter(),
        mk_arb_free=MkArbFree(),
        raise_if_bad_day=False,
        verbose=Context.quiet,
    )
    if source is None or source.empty:
        raise RuntimeError(f"No retained source day for {trading_date}")
    # The public exporter needs only normalized F/S and DF. Calling the filter
    # implementation directly avoids an unrelated pandas StringDtype schema
    # mismatch on symbol columns that are deliberately excluded from output.
    return options_filter._filter(ticker, trading_date, source)


def _unique_finite_scalar(values: Any, *, name: str, trading_date: date) -> float:
    array = np.asarray(values, dtype=np.float64)
    unique = np.unique(array)
    if unique.shape != (1,) or not np.isfinite(unique[0]):
        raise ValueError(f"{trading_date}: expected one finite {name}, found {unique}")
    return float(unique[0])


def _expiry_curve(
    *,
    trading_date: date,
    market_day: Any,
    filtered_source: Any,
) -> tuple[list[float], list[float]]:
    forwards: list[float] = []
    discount_factors: list[float] = []
    for expiry_id in market_day.expiry_ids:
        rows = filtered_source[filtered_source["expiry_id"] == int(expiry_id)]
        if rows.empty:
            raise ValueError(f"{trading_date}: missing source curve for expiry {expiry_id}")
        forwards.append(
            _unique_finite_scalar(rows["fwd"], name="normalized forward", trading_date=trading_date)
        )
        discount_factors.append(
            _unique_finite_scalar(rows["df"], name="discount factor", trading_date=trading_date)
        )
    if np.any(np.asarray(forwards) <= 0.0) or np.any(np.asarray(discount_factors) <= 0.0):
        raise ValueError(f"{trading_date}: forwards and discount factors must be positive")
    return forwards, discount_factors


def _finalize_observations(
    observations: list[dict[str, object]],
    expiries: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(observations) != len(expiries) or len(observations) == 0:
        raise ValueError("Observation and expiry records must align and be non-empty")
    finalized: list[dict[str, object]] = []
    for index, observation in enumerate(observations):
        output = {
            "date": observation["date"],
            "trading_session_is_half_day": observation[
                "trading_session_is_half_day"
            ],
            "business_sessions_from_previous_observation": None,
            "forward_carry_from_previous_observation": None,
            "discount_factor_from_previous_observation": None,
            "forward_adjusted_spot_return_into_date": None,
        }
        if index > 0:
            previous_date = observations[index - 1]["date"]
            current_date = observation["date"]
            if not isinstance(previous_date, date) or not isinstance(current_date, date):
                raise TypeError("Observation dates must be datetime.date values")
            sessions = count_business_days_between(previous_date, current_date)
            previous_expiries = expiries[index - 1]
            carry = interpolate_positive_curve_at_business_dte(
                business_dtes=previous_expiries["business_dtes"],
                values=previous_expiries["forward_over_spot"],
                target_business_dte=sessions,
                name="forward-over-spot",
            )
            discount = interpolate_positive_curve_at_business_dte(
                business_dtes=previous_expiries["business_dtes"],
                values=previous_expiries["discount_factors"],
                target_business_dte=sessions,
                name="discount-factor",
            )
            previous_spot = float(observations[index - 1]["_source_spot_reference"])
            current_spot = float(observation["_source_spot_reference"])
            gross_spot_return = current_spot / previous_spot
            output.update(
                {
                    "business_sessions_from_previous_observation": sessions,
                    "forward_carry_from_previous_observation": carry,
                    "discount_factor_from_previous_observation": discount,
                    "forward_adjusted_spot_return_into_date": (
                        gross_spot_return / carry - 1.0
                    ),
                }
            )
        finalized.append(output)
    return finalized


def _last_eligible_origin(
    dates: list[str],
    observations: list[dict[str, object]],
    *,
    horizon_sessions: int,
) -> str | None:
    for origin in range(len(dates) - horizon_sessions - 1, -1, -1):
        interval_rows = observations[origin + 1 : origin + horizon_sessions + 1]
        if all(
            row["business_sessions_from_previous_observation"] == 1
            for row in interval_rows
        ):
            return dates[origin]
    return None


def _one_day(
    task: tuple[int, str, NDArray[np.float32], dict[str, NDArray[Any]], InterimPublicDatasetConfig]
) -> dict[str, dict[str, object]]:
    date_index, date_string, raw_grid, source, config = task
    trading_date = date.fromisoformat(date_string)
    options_filter = _options_filter()
    payload = smooth_interpolated_market(
        config.ticker,
        trading_date,
        smoothness=config.smoothness,
        options_filter=options_filter,
        verbose=Context.quiet,
    )
    if not payload:
        raise RuntimeError(f"No smooth SANOS market for {trading_date}")
    market_day = market_day_from_payload(
        payload,
        ticker=config.ticker,
        geometry=_market_geometry(),
        config=FlexMarketConfig(excluded_business_dtes=(0.2,)),
    )
    if market_day is None:
        raise RuntimeError(f"No eligible 25-day market for {trading_date}")
    filtered_source = _filtered_source_day(config.ticker, trading_date, options_filter)
    forward_over_spot, discount_factors = _expiry_curve(
        trading_date=trading_date,
        market_day=market_day,
        filtered_source=filtered_source,
    )

    spot_reference = _unique_finite_scalar(
        filtered_source["spot_ref"],
        name="spot reference",
        trading_date=trading_date,
    )
    quote_expiries_y = np.asarray(market_day.quote_business_dtes, dtype=np.float64) / DAYS_PER_YEAR
    quote_strikes = np.asarray(market_day.strikes, dtype=np.float64)
    quote_sides = np.asarray(market_day.is_call, dtype=bool)

    pricer = ArbFreePricer.from_dict(payload["af_pricer"])
    selection_epsilon_atm_vol = float(
        np.asarray(
            pricer.interpolate_atm_vols(np.asarray([1.0 / DAYS_PER_YEAR])),
            dtype=np.float64,
        )[0]
    )
    expiry_indices = np.asarray(market_day.expiry_index, dtype=np.intp)
    quote_nms = np.asarray(
        NMS.to_nms(
            quote_strikes,
            t=quote_expiries_y,
            vol=np.asarray(market_day.atm_annual_vols, dtype=np.float64)[expiry_indices],
            eps_vol=selection_epsilon_atm_vol,
        ),
        dtype=np.float64,
    )
    in_paper_population = (
        quote_nms >= _PAPER_RETAINED_BOOK_NMS_RANGE[0] - 2.0e-10
    ) & (quote_nms <= _PAPER_RETAINED_BOOK_NMS_RANGE[1] + 2.0e-10)
    in_hedge_nms_population = (quote_nms > _RECOMMENDED_HEDGE_NMS_RANGE[0]) & (
        quote_nms < _RECOMMENDED_HEDGE_NMS_RANGE[1]
    )
    sanos_prices = np.asarray(
        pricer.price(quote_expiries_y, quote_strikes, mode="surface"),
        dtype=np.float64,
    )
    sanos_prices = np.where(
        quote_sides,
        sanos_prices,
        sanos_prices - 1.0 + quote_strikes,
    )
    if not np.allclose(sanos_prices, market_day.fit_prices, rtol=0.0, atol=5.0e-8):
        maximum_error = float(np.max(np.abs(sanos_prices - market_day.fit_prices)))
        raise ValueError(f"{trading_date}: SANOS price mismatch {maximum_error}")

    dlv_prices = price_dlv_surface(
        query_expiries_y=quote_expiries_y,
        query_strikes=quote_strikes,
        is_call=quote_sides,
        support_strikes=np.asarray(source["supports"][date_index], dtype=np.float64),
        raw_log_excess_dlvsq=np.asarray(raw_grid, dtype=np.float64),
        model_expiries_y=np.asarray(source["model_times_y"], dtype=np.float64),
        minimum_dlvsq=_MINIMUM_DLV_SQ,
        maximum_dlvsq=_MAXIMUM_DLV_SQ,
        backbone_annual_volatility=_BACKBONE_ANNUAL_VOLATILITY,
    )
    sanos_implied_vols, sanos_implied_vol_is_valid = implied_volatilities_with_validity(
        expiries_y=quote_expiries_y,
        normalized_strikes=quote_strikes,
        normalized_prices=sanos_prices,
        is_call=quote_sides,
    )
    dlv_implied_vols, dlv_implied_vol_is_valid = implied_volatilities_with_validity(
        expiries_y=quote_expiries_y,
        normalized_strikes=quote_strikes,
        normalized_prices=dlv_prices,
        is_call=quote_sides,
    )

    expiry_count = len(market_day.expiries_y)
    atm_strikes = np.ones(expiry_count, dtype=np.float64)
    atm_sides = np.ones(expiry_count, dtype=bool)
    sanos_atm_prices = np.asarray(
        pricer.price(
            np.asarray(market_day.expiries_y, dtype=np.float64),
            atm_strikes,
            mode="surface",
        ),
        dtype=np.float64,
    )
    sanos_atm_vols, sanos_atm_vol_is_valid = implied_volatilities_with_validity(
        expiries_y=np.asarray(market_day.expiries_y, dtype=np.float64),
        normalized_strikes=atm_strikes,
        normalized_prices=sanos_atm_prices,
        is_call=atm_sides,
    )
    dlv_atm_prices = price_dlv_surface(
        query_expiries_y=np.asarray(market_day.expiries_y, dtype=np.float64),
        query_strikes=atm_strikes,
        is_call=atm_sides,
        support_strikes=np.asarray(source["supports"][date_index], dtype=np.float64),
        raw_log_excess_dlvsq=np.asarray(raw_grid, dtype=np.float64),
        model_expiries_y=np.asarray(source["model_times_y"], dtype=np.float64),
        minimum_dlvsq=_MINIMUM_DLV_SQ,
        maximum_dlvsq=_MAXIMUM_DLV_SQ,
        backbone_annual_volatility=_BACKBONE_ANNUAL_VOLATILITY,
    )
    dlv_atm_vols, dlv_atm_vol_is_valid = implied_volatilities_with_validity(
        expiries_y=np.asarray(market_day.expiries_y, dtype=np.float64),
        normalized_strikes=atm_strikes,
        normalized_prices=dlv_atm_prices,
        is_call=atm_sides,
    )
    if not np.all(sanos_atm_vol_is_valid) or not np.all(dlv_atm_vol_is_valid):
        raise FloatingPointError(f"{trading_date}: model ATM implied volatility is invalid")

    sanos_horizon = config.max_business_dte / DAYS_PER_YEAR
    retained_sanos = np.asarray(pricer.expiries) <= sanos_horizon + 1.0e-12
    retained_indices = np.flatnonzero(retained_sanos)
    if len(retained_indices) == 0 or not np.array_equal(
        retained_indices,
        np.arange(retained_indices[-1] + 1),
    ):
        raise ValueError(f"{trading_date}: SANOS 25-day state is not a prefix")
    if pricer.mstrikes is None or pricer.mdensities is None:
        raise ValueError(f"{trading_date}: SANOS surface state is unavailable")
    sanos_model_expiries = np.asarray(pricer.expiries)[retained_sanos]
    sanos_variances = np.asarray(pricer.variances)[retained_sanos]
    sanos_supports = [
        np.asarray(pricer.mstrikes[index], dtype=np.float64)
        for index in retained_indices
    ]
    sanos_densities = [
        np.asarray(pricer.mdensities[index], dtype=np.float64)
        for index in retained_indices
    ]
    reconstructed_sanos = price_sanos_surface(
        query_expiries_y=quote_expiries_y,
        query_strikes=quote_strikes,
        is_call=quote_sides,
        model_expiries_y=sanos_model_expiries,
        atm_variances=sanos_variances,
        support_strikes=sanos_supports,
        densities=sanos_densities,
        smoothness=float(pricer.smoothness),
        one_minute_y=ONE_MINUTE,
    )
    if not np.allclose(reconstructed_sanos, sanos_prices, rtol=0.0, atol=2.0e-8):
        maximum_error = float(np.max(np.abs(reconstructed_sanos - sanos_prices)))
        raise ValueError(f"{trading_date}: public SANOS reconstruction error {maximum_error}")

    expiry_dates = [date.fromisoformat(value) for value in market_day.expiry_dates]
    price_unit_bridge = (
        np.asarray(discount_factors, dtype=np.float64)
        * np.asarray(forward_over_spot, dtype=np.float64)
    )
    half_spreads_over_spot = (
        np.asarray(market_day.half_spreads, dtype=np.float64)
        * price_unit_bridge[np.asarray(market_day.expiry_index, dtype=np.intp)]
    )
    return {
        "observations": {
            "date": trading_date,
            "trading_session_is_half_day": trading_date
            in set(get_half_days(trading_date, trading_date)),
            "_source_spot_reference": float(spot_reference),
        },
        "expiries": {
            "date": trading_date,
            "expiry_dates": expiry_dates,
            "expiries_y": np.asarray(market_day.expiries_y, dtype=np.float64).tolist(),
            "business_dtes": np.asarray(market_day.business_dtes, dtype=np.float64).tolist(),
            "calendar_dtes": [int((expiry_date - trading_date).days) for expiry_date in expiry_dates],
            "is_am_settlement": np.asarray(market_day.expiry_is_am, dtype=bool).tolist(),
            "forward_over_spot": np.asarray(forward_over_spot, dtype=np.float64).tolist(),
            "discount_factors": np.asarray(discount_factors, dtype=np.float64).tolist(),
            "book_selection_epsilon_annualized_atm_vol": selection_epsilon_atm_vol,
            "book_selection_annualized_atm_vols": np.asarray(
                market_day.atm_annual_vols, dtype=np.float64
            ).tolist(),
            "sanos_annualized_atm_vols": np.asarray(
                sanos_atm_vols, dtype=np.float64
            ).tolist(),
            "dlv_annualized_atm_vols": np.asarray(dlv_atm_vols, dtype=np.float64).tolist(),
        },
        "instruments": {
            "date": trading_date,
            "expiry_indices": np.asarray(market_day.expiry_index, dtype=np.int16).tolist(),
            "strikes_over_forward": np.asarray(quote_strikes, dtype=np.float64).tolist(),
            "standardized_log_moneyness": quote_nms.tolist(),
            "in_paper_retained_population": in_paper_population.tolist(),
            "in_recommended_hedge_nms_population": in_hedge_nms_population.tolist(),
            "is_call": quote_sides.tolist(),
            "half_spreads_over_spot": np.asarray(
                half_spreads_over_spot, dtype=np.float64
            ).tolist(),
            "sanos_prices_over_discounted_forward": np.asarray(
                sanos_prices, dtype=np.float64
            ).tolist(),
            "dlv_prices_over_discounted_forward": np.asarray(
                dlv_prices, dtype=np.float64
            ).tolist(),
            "sanos_annualized_implied_vols": np.asarray(
                sanos_implied_vols, dtype=np.float64
            ).tolist(),
            "dlv_annualized_implied_vols": np.asarray(
                dlv_implied_vols, dtype=np.float64
            ).tolist(),
            "sanos_implied_vol_is_valid": sanos_implied_vol_is_valid.tolist(),
            "dlv_implied_vol_is_valid": dlv_implied_vol_is_valid.tolist(),
        },
        "sanos_geometry": {
            "date": trading_date,
            "model_expiries_y": np.asarray(sanos_model_expiries, dtype=np.float64).tolist(),
            "support_strikes": [support.tolist() for support in sanos_supports],
        },
        "sanos_state": {
            "date": trading_date,
            "kernel_total_variances": np.asarray(
                sanos_variances, dtype=np.float64
            ).tolist(),
            "densities": [density.tolist() for density in sanos_densities],
        },
        "dlv_geometry": {
            "date": trading_date,
            "model_expiries_y": np.asarray(
                source["model_times_y"], dtype=np.float64
            ).tolist(),
            "support_strikes": np.asarray(
                source["supports"][date_index], dtype=np.float64
            ).tolist(),
        },
        "dlv_state": {
            "date": trading_date,
            "raw_log_excess_dlvsq": np.asarray(raw_grid, dtype=np.float32).reshape(-1).tolist(),
        },
    }


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _generator_code_identity() -> tuple[dict[str, str], bool]:
    repository = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    checksums = {
        filename: _sha256(repository / filename) for filename in _GENERATOR_SOURCE_FILES
    }
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *_GENERATOR_SOURCE_FILES],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return checksums, bool(status.strip())


def _write_readme(path: Path, config: InterimPublicDatasetConfig) -> None:
    content = f"""# Interim hedging public-candidate dataset

This independently loadable dataset is a deliberately lossy, normalized
projection of fitted SPX option-market models from 2022 through August 2025. It has
distribution status `public_candidate`; that status is a technical leakage
classification and does not itself grant redistribution rights.

It contains no absolute spot, market bid, market ask, market midpoint, option
identifier, symbol, volume, open interest, vendor Greek, or vendor implied
volatility. The SANOS and DLV price arrays are fitted model outputs, not market
midpoints. Their annualized Black implied volatilities are computed with
`cdxcore.bs` and stored alongside them, including each model's actual implied
ATM volatility at `K/F=1`. The separate book-selection ATM curve and its
one-day epsilon volatility record the SANOS kernel-input geometry used to
convert NMS to `K/F`.

## Units

- Normalized strike: `K/F`.
- Model option price: `C/(DF*F)`.
- Half-spread: `h/S`.
- Forward: `F/S`, with no absolute spot level retained.
- Discount factor: `DF`.
- Expiry time: repository business years using {DAYS_PER_YEAR} days per year.
- ATM and instrument volatilities: annualized Black volatility.
- DLV state: raw log excess above the squared-DLV floor.

The conversion to spot-one units is `C/S = DF*(F/S)*C/(DF*F)`. Conversely,
`h/(DF*F) = (h/S)/(DF*(F/S))`.

## Returns and intervals

The retained panel contains occasional gaps. For retained dates `t[i-1]` and
`t[i]`, the dataset stores their business-session distance `n[i]`, the
log-linearly interpolated previous-date carry

`c[i] = F_(t[i-1])(t[i]) / S_(t[i-1])`,

the corresponding discount factor, and

`r_fwd[i] = (S[i]/S[i-1]) / c[i] - 1`.

Thus the normalized gross spot return is `c[i]*(1+r_fwd[i])`. The first row has
null interval fields. Daily experiments must require `n[i]=1`; five-day episodes
must require five consecutive one-session intervals.

## Execution and frozen contracts

The released quote-side population spans NMS `[-2.5,2]`, a superset covering
the paper's retained calibration population `[-2.5,1.5]` and the benchmark's
recommended hedge screen `(-2,2)`. Instrument-level Boolean masks identify
both populations. The hedge mask is only the contemporaneous NMS screen;
expiry, settlement, OTM-side, and episode-specific eligibility remain dynamic.

The canonical common market mark is the SANOS model price. A buy executes at
that mark plus `h/S`, and a sale at the mark minus `h/S`, after conversion to a
common spot-one unit. The DLV price is retained as a separately reconstructable
model output. Because source contract identifiers are excluded, frozen-weekly
experiments use synthetic inception contracts identified by expiry date and
cash strike. Their future `K/F` and spread are obtained from the normalized
spot/forward path and cross-sectional spread interpolation; exact listed-contract
replay and display-size limits are not public-dataset capabilities.

The common fitted models can be reconstructed from the separate SANOS and DLV
state/geometry files. `quantdata.hedging.data` supplies standalone decoders and
deep schema and reconstruction validation. See `DATA_DICTIONARY.md` for the
field-by-field schema.

Generated configuration: `{config}`.
"""
    path.write_text(content, encoding="utf-8")


def _write_data_dictionary(path: Path) -> None:
    content = """# Public-candidate data dictionary

Every Parquet file has one row per retained observation date and is ordered by
`date`. Rows at the same position across files refer to the same date. A list
field in `expiries.parquet` has length `E[d]`, the number of retained expiries
on date `d`. A list field in `instruments.parquet` has length `Q[d]`, the number
of retained call/put quote-side books on that date. Calls and puts at one strike
are separate books. `expiry_indices[q]` maps instrument `q` to its expiry row.

## `observations.parquet`

| Field | Type | Meaning |
| --- | --- | --- |
| `date` | date | Retained market observation date. |
| `trading_session_is_half_day` | bool | Whether the observation date is an exchange half-day. |
| `business_sessions_from_previous_observation` | nullable int16 | Number of exchange sessions since the previous retained row; null on the first row. |
| `forward_carry_from_previous_observation` | nullable float64 | Previous-date `F(t_previous,date)/S(t_previous)`, log-linearly interpolated in business DTE; null on the first row. |
| `discount_factor_from_previous_observation` | nullable float64 | Previous-date discount factor to this row's date under the same interpolation; null on the first row. |
| `forward_adjusted_spot_return_into_date` | nullable float64 | `(S[d]/S[d-1])/forward_carry-1`; null on the first row. It contains no absolute spot level. |

## `expiries.parquet`

| Field | Type | Meaning |
| --- | --- | --- |
| `date` | date | Observation date. |
| `expiry_dates` | list[date] | Calendar settlement date of each retained expiry. |
| `expiries_y` | list[float64] | Time to expiry in repository business years (`252` days per year, with the configured AM convention). |
| `business_dtes` | list[float64] | Business-day time to expiry. |
| `calendar_dtes` | list[int16] | Calendar-day difference between observation and expiry dates. |
| `is_am_settlement` | list[bool] | Whether the expiry uses opening/AM settlement. Primary public hedging excludes these rows. |
| `forward_over_spot` | list[float64] | Normalized forward `F/S` for each expiry. |
| `discount_factors` | list[float64] | Discount factor `DF` for each expiry. |
| `book_selection_epsilon_annualized_atm_vol` | float64 | One-day epsilon ATM input used by the normalized-moneyness geometry. |
| `book_selection_annualized_atm_vols` | list[float64] | ATM input curve used to define the released NMS population and reconstruct `K/F` from NMS. |
| `sanos_annualized_atm_vols` | list[float64] | Black ATM implied volatility obtained from the fitted SANOS model at `K/F=1`. |
| `dlv_annualized_atm_vols` | list[float64] | Black ATM implied volatility obtained from the fitted DLV model at `K/F=1`. |

## `instruments.parquet`

| Field | Type | Meaning |
| --- | --- | --- |
| `date` | date | Observation date. |
| `expiry_indices` | list[int16] | Zero-based indices into every expiry-aligned list for the same date. |
| `strikes_over_forward` | list[float64] | Normalized strike `K/F`. No absolute cash strike is retained. |
| `standardized_log_moneyness` | list[float64] | Repository NMS coordinate computed from `K/F`, expiry time, and the book-selection ATM curve. |
| `in_paper_retained_population` | list[bool] | Exact GROMM calibration-population mask, inclusive NMS `[-2.5,1.5]`. |
| `in_recommended_hedge_nms_population` | list[bool] | Contemporaneous strict NMS `(-2,2)` mask. It does not apply expiry, AM/PM, OTM-side, or episode filters. |
| `is_call` | list[bool] | `true` for a call book and `false` for a put book. |
| `half_spreads_over_spot` | list[float64] | Released source half-spread normalized as `h/S`; it is not a bid, ask, or midpoint. |
| `sanos_prices_over_discounted_forward` | list[float64] | Fitted SANOS model price `C/(DF*F)`, not a market midpoint. |
| `dlv_prices_over_discounted_forward` | list[float64] | Fitted DLV model price `C/(DF*F)`, not a market midpoint. |
| `sanos_annualized_implied_vols` | list[float64] | Annualized Black IV inverted from the SANOS model price with `cdxcore.bs`. |
| `dlv_annualized_implied_vols` | list[float64] | Annualized Black IV inverted from the DLV model price with `cdxcore.bs`. |
| `sanos_implied_vol_is_valid` | list[bool] | Whether the SANOS price is numerically interior to Black bounds and its stored IV is an ordinary inversion. |
| `dlv_implied_vol_is_valid` | list[bool] | Corresponding numerical-interiority flag for the DLV price. |

Convert a model price to spot-one units with
`C/S = DF * (F/S) * C/(DF*F)`. Convert the half-spread to discounted-forward
units with `h/(DF*F) = (h/S)/(DF*(F/S))`.

## `sanos_geometry.parquet`

| Field | Type | Meaning |
| --- | --- | --- |
| `date` | date | Observation date. |
| `model_expiries_y` | list[float64] | SANOS model-node expiries in business years. |
| `support_strikes` | list[list[float64]] | Per-model-expiry latent normalized support `X/F`; each inner list aligns with the same-index density. |

## `sanos_state.parquet`

| Field | Type | Meaning |
| --- | --- | --- |
| `date` | date | Observation date. |
| `kernel_total_variances` | list[float64] | Unsmoothed ATM input total variance at each SANOS model expiry. The Black kernel uses `sanos_smoothness` times this value. |
| `densities` | list[list[float64]] | Per-expiry non-negative latent probability masses. Each density has unit mass and unit forward mean on its support. |

## `dlv_geometry.parquet`

| Field | Type | Meaning |
| --- | --- | --- |
| `date` | date | Observation date. |
| `model_expiries_y` | list[float64] | The 25 DLV model-node expiries in business years. |
| `support_strikes` | list[float64] | Shared 202-node `K/F` support: 200 interior nodes plus two exterior boundary nodes. |

## `dlv_state.parquet`

| Field | Type | Meaning |
| --- | --- | --- |
| `date` | date | Observation date. |
| `raw_log_excess_dlvsq` | list[float32] | Expiry-major flattened `[25,200]` state. Decode as `DLVsq = minimum_dlvsq + exp(min(raw, log(maximum_dlvsq-minimum_dlvsq)))`; boundary local variances are zero. |

## Metadata files

- `config.json` records the schema, ticker, retained affine fit, SANOS
  smoothness, maximum business DTE, dataset name, excluded 0.2D population,
  released/paper/hedge/DLV NMS ranges, and model business-DTE grid.
- `geometry.json` records all units and reconstruction constants: DLV bounds,
  state shape, 5% Black backbone, boundary and interpolation conventions,
  SANOS smoothing and interpolation conventions, one-minute threshold, forward
  interpolation, and population ranges.
- `manifest.json` records distribution status and release-approval placeholders,
  source and Git provenance, generator-source hashes and dirty status, date and
  population counts, last eligible daily/weekly origins, return and execution
  conventions, capability flags, and SHA-256 checksums for every payload file.
- `README.md` gives the economic conventions, exclusions, units, return
  construction, execution rule, and frozen-contract limitation.

`distribution="public_candidate"` is a technical leakage classification. It
does not grant redistribution rights. A `public` artifact additionally requires
completed license, rights-basis, approver, approval-date, approval-reference,
and clean-generator provenance fields.
"""
    path.write_text(content, encoding="utf-8")


def export_public_candidate(
    *,
    output_root: Path,
    config: InterimPublicDatasetConfig = InterimPublicDatasetConfig(),
    workers: int = 4,
) -> Path:
    """Materialize the complete interim dataset under ``output_root``."""
    verify_inp(workers > 0, lambda: f"workers must be positive, found {workers}")
    output_root = Path(output_root)
    output_directory = output_root / config.dataset_name
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset {output_directory}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{config.dataset_name}.", dir=output_root)
    )
    started = perf_counter()
    try:
        with np.load(DATASET_PATH, allow_pickle=False) as loaded:
            source = {name: np.asarray(loaded[name]) for name in loaded.files}
        if str(source["schema"].item()) != "daily25-jax-market-v1":
            raise ValueError(f"Unexpected internal source schema {source['schema'].item()!r}")
        dates = [str(value) for value in source["dates"]]
        raw_grids = load_raw_grids(config.fit_label)
        if raw_grids.shape[0] != len(dates):
            raise ValueError("DLV states do not align with source dates")
        task_source = {
            name: source[name]
            for name in (
                "model_times_y",
                "supports",
            )
        }
        tasks = (
            (index, date_string, raw_grids[index], task_source, config)
            for index, date_string in enumerate(dates)
        )
        records = {name: [] for name in (
            "observations",
            "expiries",
            "instruments",
            "sanos_geometry",
            "sanos_state",
            "dlv_geometry",
            "dlv_state",
        )}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for completed, day in enumerate(executor.map(_one_day, tasks), start=1):
                for name, record in day.items():
                    records[name].append(record)
                if completed == 1 or completed % 10 == 0 or completed == len(dates):
                    elapsed = perf_counter() - started
                    rate = completed / elapsed
                    remaining = (len(dates) - completed) / rate if rate > 0.0 else math.inf
                    print(
                        f"public-export elapsed={elapsed:.1f}s completed={completed}/{len(dates)} "
                        f"eta={remaining:.1f}s",
                        flush=True,
                    )

        records["observations"] = _finalize_observations(
            records["observations"],
            records["expiries"],
        )

        file_stems = {
            "observations": "observations.parquet",
            "expiries": "expiries.parquet",
            "instruments": "instruments.parquet",
            "sanos_geometry": "sanos_geometry.parquet",
            "sanos_state": "sanos_state.parquet",
            "dlv_geometry": "dlv_geometry.parquet",
            "dlv_state": "dlv_state.parquet",
        }
        for record_name, filename in file_stems.items():
            frame = pl.DataFrame(
                records[record_name],
                schema=DATA_FILE_SCHEMAS[filename],
                strict=True,
            )
            if tuple(frame.columns) != DATA_FILE_COLUMNS[filename]:
                raise ValueError(
                    f"Generated columns for {filename} are {frame.columns}, "
                    f"expected {DATA_FILE_COLUMNS[filename]}"
                )
            frame.write_parquet(staging / filename, compression="zstd", statistics=True)

        geometry_payload = {
            "schema": HEDGING_PUBLIC_SCHEMA,
            "days_per_year": DAYS_PER_YEAR,
            "normalized_strike_unit": "K/F",
            "model_price_unit": "cash_price/(DF*F)",
            "half_spread_unit": "half_spread/S",
            "normalized_forward_unit": "F/S",
            "model_business_dtes": list(_MODEL_BUSINESS_DTES),
            "dlv_state_shape": [25, 200],
            "minimum_dlvsq": _MINIMUM_DLV_SQ,
            "maximum_dlvsq": _MAXIMUM_DLV_SQ,
            "backbone_annual_volatility": _BACKBONE_ANNUAL_VOLATILITY,
            "dlv_boundary_convention": (
                "two exterior support nodes with zero local variance; "
                "interior state is expiry-major"
            ),
            "dlv_expiry_interpolation": (
                "linear density interpolation between adjacent model expiries "
                "with the fixed Black backbone kernel"
            ),
            "sanos_smoothness": config.smoothness,
            "sanos_density_kind": "surface",
            "sanos_expiry_interpolation": (
                "linear price interpolation between adjacent surface-density nodes"
            ),
            "sanos_terminal_extrapolation": (
                "last density with clipped-node annualized kernel volatility held constant"
            ),
            "sanos_kernel_total_variance_convention": (
                "unsmoothed ATM input total variance; the Black smoothing kernel "
                "uses smoothness times this curve"
            ),
            "one_minute_y": ONE_MINUTE,
            "forward_curve_interpolation": (
                "log-linear in business DTE, anchored at value one at DTE zero"
            ),
            "released_book_nms_range": list(_RELEASED_BOOK_NMS_RANGE),
            "paper_retained_book_nms_range": list(_PAPER_RETAINED_BOOK_NMS_RANGE),
            "recommended_hedge_nms_range": list(_RECOMMENDED_HEDGE_NMS_RANGE),
            "dlv_fit_grid_nms_range": list(_DLV_FIT_GRID_NMS_RANGE),
        }
        _atomic_write_json(geometry_payload, staging / "geometry.json")
        config_payload = {
            "schema": HEDGING_PUBLIC_SCHEMA,
            "ticker": config.ticker,
            "fit_label": config.fit_label,
            "smoothness": config.smoothness,
            "max_business_dte": config.max_business_dte,
            "dataset_name": config.dataset_name,
            "excluded_business_dtes": [0.2],
            "released_book_nms_range": list(_RELEASED_BOOK_NMS_RANGE),
            "paper_retained_book_nms_range": list(_PAPER_RETAINED_BOOK_NMS_RANGE),
            "recommended_hedge_nms_range": list(_RECOMMENDED_HEDGE_NMS_RANGE),
            "dlv_fit_grid_nms_range": list(_DLV_FIT_GRID_NMS_RANGE),
            "model_business_dtes": list(_MODEL_BUSINESS_DTES),
        }
        _atomic_write_json(config_payload, staging / "config.json")
        _write_readme(staging / "README.md", config)
        _write_data_dictionary(staging / "DATA_DICTIONARY.md")
        checksums = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        generator_code_sha256, generator_worktree_dirty = _generator_code_identity()
        paper_counts = np.asarray(
            [
                sum(row["in_paper_retained_population"])
                for row in records["instruments"]
            ],
            dtype=np.int64,
        )
        source_quote_counts = np.asarray(source["quote_counts"], dtype=np.int64)
        if not np.array_equal(paper_counts, source_quote_counts):
            mismatch = np.flatnonzero(paper_counts != source_quote_counts)
            first = int(mismatch[0])
            raise ValueError(
                "Released data do not reproduce the paper population on "
                f"{dates[first]}: {paper_counts[first]} versus {source_quote_counts[first]}"
            )
        manifest = {
            "schema": HEDGING_PUBLIC_SCHEMA,
            "distribution": "public_candidate",
            "dataset_name": config.dataset_name,
            "release_license": None,
            "rights_basis": None,
            "approver": None,
            "approval_date": None,
            "approval_reference": None,
            "git_revision": _git_revision(),
            "generator_code_sha256": generator_code_sha256,
            "generator_worktree_dirty": generator_worktree_dirty,
            "num_dates": len(dates),
            "start_date": dates[0],
            "end_date": dates[-1],
            "last_eligible_daily_origin": _last_eligible_origin(
                dates,
                records["observations"],
                horizon_sessions=1,
            ),
            "last_eligible_frozen_weekly_origin": _last_eligible_origin(
                dates,
                records["observations"],
                horizon_sessions=5,
            ),
            "num_quote_side_books": int(
                sum(len(row["expiry_indices"]) for row in records["instruments"])
            ),
            "num_paper_retained_quote_side_books": int(paper_counts.sum()),
            "num_observation_hedge_nms_quote_side_books": int(
                sum(
                    sum(row["in_recommended_hedge_nms_population"])
                    for row in records["instruments"]
                )
            ),
            "source_model": "retained daily affine-200 DLV and smooth SANOS fits",
            "return_convention": (
                "r_fwd[i]=(S[i]/S[i-1])/(F_[i-1](t[i])/S[i-1])-1; "
                "previous-date F/S is log-linearly interpolated at the retained-date "
                "business-session distance"
            ),
            "execution_convention": (
                "SANOS model mark plus/minus half_spread_over_spot in spot-one units"
            ),
            "capabilities": {
                "normalized_sanos_pricing": True,
                "normalized_dlv_pricing": True,
                "normalized_spread_experiments": True,
                "forward_adjusted_return": True,
                "synthetic_frozen_contracts": True,
                "exact_vendor_book_reconstruction": False,
                "contract_level_replay": False,
                "absolute_dollar_ticks": False,
                "liquidity_or_volume_models": False,
            },
            "checksums": checksums,
        }
        _atomic_write_json(manifest, staging / "manifest.json")
        validate_public_dataset(staging)
        staging.replace(output_directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_directory


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Run the public-candidate exporter."""
    arguments = _parse_arguments()
    output = export_public_candidate(
        output_root=arguments.output_root,
        workers=arguments.workers,
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
