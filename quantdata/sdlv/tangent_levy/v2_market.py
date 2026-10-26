"""Exact 891-date quote population for Tangent Levy v2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.err import verify, verify_inp
from cdxcore.verbose import Context

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.dysanos.market import smooth_interpolated_market
from quantdata.sanos.arbfree import ArbFreePricer
from quantdata.widb.options_filter import OptionsFilter

from .v2_config import TangentLevyV2MarketConfig

DEFAULT_POPULATION_MANIFEST = Path(
    "/home/hans/quantdata_data/training/flex-shared-strikes/capacity/"
    "spx-2022-2025-shared-strikes-dte1-5-weekly-v1/reports/"
    "spx-2022-2025-shared-strikes-dte1-5-weekly-v1-native-complete.json"
)


@dataclass(frozen=True)
class V2MaturityBook:
    """Separate call and put observations in call-equivalent units."""

    tau: float
    expiry_date: str
    strikes: np.ndarray
    bids: np.ndarray
    asks: np.ndarray
    is_call: np.ndarray
    sanos_calls: np.ndarray

    @property
    def quote_count(self) -> int:
        return int(self.strikes.size)


@dataclass(frozen=True)
class V2MarketDay:
    """One date of the exact original-book v2 population."""

    market_date: date
    spot_ref: float
    spot_return: float
    books: tuple[V2MaturityBook, ...]
    market_uid: str

    @property
    def quote_count(self) -> int:
        return sum(book.quote_count for book in self.books)


def _canonical_options_filter(config: TangentLevyV2MarketConfig) -> OptionsFilter:
    return OptionsFilter(
        max_num_options=100_000,
        max_business_days_to_expiry=config.maximum_business_dte,
        max_nms=(-100.0, 100.0),
    )


def market_day_from_payload(
    payload: dict[str, object],
    *,
    config: TangentLevyV2MarketConfig = TangentLevyV2MarketConfig(),
) -> V2MarketDay | None:
    """Apply the exact horizon, AM, and canonical-NMS filters from ``dy21``."""

    if not payload:
        return None
    market = dict(payload["market"])
    expiries = np.asarray(market["expiries"], dtype=np.float64)
    pricer = ArbFreePricer.from_dict(payload["af_pricer"])
    atm_annual_vols = np.asarray(
        pricer.interpolate_atm_vols(expiries),
        dtype=np.float64,
    )
    one_day_vol = float(
        np.asarray(
            pricer.interpolate_atm_vols(
                np.asarray([1.0 / DAYS_PER_YEAR], dtype=np.float64)
            )
        ).item()
    )
    endpoints = NMS.from_nms(
        np.asarray(config.nms_range, dtype=np.float64)[None, :],
        t=expiries[:, None],
        vol=atm_annual_vols[:, None],
        eps_vol=one_day_vol,
    )
    expiry_dates = tuple(str(value) for value in market["expiry_dates"])
    is_am = np.asarray(market["is_am"], dtype=bool)
    books: list[V2MaturityBook] = []
    for expiry_index, tau in enumerate(expiries):
        business_dte = float(tau * DAYS_PER_YEAR)
        if business_dte < config.minimum_business_dte - 1.0e-4:
            continue
        if business_dte > config.maximum_business_dte + 1.0e-4:
            continue
        if any(
            abs(business_dte - excluded) <= 1.0e-4
            for excluded in config.excluded_business_dtes
        ):
            verify(
                bool(is_am[expiry_index]),
                lambda: (
                    f"excluded fractional expiry {business_dte:g} on "
                    f"{payload['d']} is not AM-settled"
                ),
            )
            continue
        strikes = np.asarray(market["strikes"][expiry_index], dtype=np.float64)
        european = ~np.asarray(market["is_american"][expiry_index], dtype=bool)
        fit_available = np.asarray(
            market["af_mid_available"][expiry_index], dtype=bool
        )
        within_nms = (
            (strikes >= endpoints[expiry_index, 0])
            & (strikes <= endpoints[expiry_index, 1])
        )
        selected_strikes: list[np.ndarray] = []
        selected_bids: list[np.ndarray] = []
        selected_asks: list[np.ndarray] = []
        selected_sides: list[np.ndarray] = []
        selected_targets: list[np.ndarray] = []
        for is_call, active_name, bid_name, ask_name, fit_name in (
            (True, "call_active", "call_bids", "call_asks", "af_call_mids"),
            (False, "put_active", "put_bids", "put_asks", "af_put_mids"),
        ):
            active = (
                np.asarray(market[active_name][expiry_index], dtype=bool)
                & european
                & fit_available
                & within_nms
            )
            if not np.any(active):
                continue
            strike = strikes[active]
            parity = 0.0 if is_call else 1.0 - strike
            selected_strikes.append(strike)
            selected_bids.append(
                np.asarray(market[bid_name][expiry_index], dtype=np.float64)[active]
                + parity
            )
            selected_asks.append(
                np.asarray(market[ask_name][expiry_index], dtype=np.float64)[active]
                + parity
            )
            selected_sides.append(np.full(strike.shape, is_call, dtype=bool))
            selected_targets.append(
                np.asarray(market[fit_name][expiry_index], dtype=np.float64)[active]
                + parity
            )
        if not selected_strikes:
            continue
        book = V2MaturityBook(
            tau=float(tau),
            expiry_date=expiry_dates[expiry_index],
            strikes=np.concatenate(selected_strikes),
            bids=np.concatenate(selected_bids),
            asks=np.concatenate(selected_asks),
            is_call=np.concatenate(selected_sides),
            sanos_calls=np.concatenate(selected_targets),
        )
        verify(
            book.quote_count >= 10
            and np.all(np.isfinite(book.bids))
            and np.all(np.isfinite(book.asks))
            and np.all(book.asks > book.bids),
            lambda: (
                f"invalid retained quote book on {payload['d']} expiry "
                f"{book.expiry_date}"
            ),
        )
        books.append(book)
    if not books:
        return None
    verify(
        all(left.tau < right.tau for left, right in zip(books, books[1:])),
        lambda: "retained v2 expiries are not strictly ordered",
    )
    return V2MarketDay(
        market_date=date.fromisoformat(str(payload["d"])),
        spot_ref=float(payload["spot_ref"]),
        spot_return=float(payload["spot_ret"]),
        books=tuple(books),
        market_uid=str(market["uid"]),
    )


def load_market_day(
    market_date: date,
    *,
    config: TangentLevyV2MarketConfig = TangentLevyV2MarketConfig(),
    verbose: Context | None = Context.all,
) -> V2MarketDay | None:
    """Load one cached SPX day without changing the upstream population."""

    verify_inp(
        config.start <= market_date <= config.end,
        lambda: f"date {market_date} lies outside the configured v2 population",
    )
    payload = smooth_interpolated_market(
        config.ticker,
        market_date,
        smoothness=config.smoothness,
        options_filter=_canonical_options_filter(config),
        override_cache_mode="cacheonly",
        verbose=verbose,
    )
    return market_day_from_payload(payload, config=config)


def exact_population_dates(
    *,
    config: TangentLevyV2MarketConfig = TangentLevyV2MarketConfig(),
    manifest_path: Path = DEFAULT_POPULATION_MANIFEST,
) -> tuple[date, ...]:
    """Read and validate the immutable date population used by ``dy21``."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dates = tuple(date.fromisoformat(value) for value in payload["market_dates"])
    verify(
        payload.get("status") == "native_complete"
        and payload.get("ticker") == config.ticker
        and int(payload.get("num_days", -1)) == config.expected_dates
        and int(payload.get("num_quotes", -1)) == config.expected_quotes
        and len(dates) == config.expected_dates
        and len(set(dates)) == len(dates)
        and dates == tuple(sorted(dates))
        and dates[0] == config.start
        and dates[-1] == config.end,
        lambda: f"manifest {manifest_path} is not the exact v2 population",
    )
    return dates


def load_exact_market_population(
    *,
    config: TangentLevyV2MarketConfig = TangentLevyV2MarketConfig(),
    manifest_path: Path = DEFAULT_POPULATION_MANIFEST,
    verbose: Context | None = Context.all,
) -> tuple[V2MarketDay, ...]:
    """Load all 891 dates and assert the exact separate-book quote count."""

    dates = exact_population_dates(config=config, manifest_path=manifest_path)
    days: list[V2MarketDay] = []
    for index, market_date in enumerate(dates):
        day = load_market_day(market_date, config=config, verbose=verbose)
        verify(day is not None, lambda: f"missing exact-population day {market_date}")
        days.append(day)
        if (index + 1) % 25 == 0 or index + 1 == len(dates):
            print(
                f"tangent-levy-v2 market days={index + 1}/{len(dates)} "
                f"quotes={sum(value.quote_count for value in days)}",
                flush=True,
            )
    quote_count = sum(day.quote_count for day in days)
    verify(
        quote_count == config.expected_quotes,
        lambda: (
            f"exact population has {quote_count} quotes; expected "
            f"{config.expected_quotes}"
        ),
    )
    return tuple(days)
