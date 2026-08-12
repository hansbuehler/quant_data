"""Historical SPX market adapter for the Tangent Lévy project."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

import numpy as np
from cdxcore.err import verify, verify_inp
from cdxcore.verbose import Context

from quantdata.biz_calendar import DAYS_PER_YEAR, DateRange, get_half_days, to_date
from quantdata.dysanos.market import MARKET_SCHEMA_VERSION, smooth_interpolated_market
from quantdata.sanos.arbfree import ArbFreePricer

from .config import MarketConfig
from .coordinates import implied_volatility


@dataclass(frozen=True)
class MarketAudit:
    """Coverage and schema diagnostics for one requested market date."""

    market_date: date
    early_close: bool
    available: bool
    input_expiries: int
    retained_expiries: int
    retained_min_tau: float
    retained_max_tau: float
    input_quotes: int
    active_quotes: int
    excluded_american_quotes: int
    option_cap_bound: bool
    reason: str = ""


@dataclass(frozen=True)
class CommonGridCoverage:
    """Training-only maturity coverage for a candidate fixed grid."""

    candidate_tau: np.ndarray
    coverage_fraction: np.ndarray
    selected_tau: np.ndarray
    minimum_fraction: float
    sessions: int
    available_sessions: int
    option_cap_fraction: float


@dataclass(frozen=True)
class TangentLevyMarketDay:
    """One normalized smoothed market with ragged original observations."""

    ticker: str
    market_date: date
    spot_ref: float
    spot_return: float
    expiries: np.ndarray
    strikes: tuple[np.ndarray, ...]
    sanos_calls: tuple[np.ndarray, ...]
    pure_bids: tuple[np.ndarray, ...]
    pure_asks: tuple[np.ndarray, ...]
    model_masks: tuple[np.ndarray, ...]
    call_bids: tuple[np.ndarray, ...]
    call_asks: tuple[np.ndarray, ...]
    put_bids: tuple[np.ndarray, ...]
    put_asks: tuple[np.ndarray, ...]
    call_active: tuple[np.ndarray, ...]
    put_active: tuple[np.ndarray, ...]
    expiry_dates: tuple[str, ...]
    expiry_ids: np.ndarray
    is_am: np.ndarray
    pricer: ArbFreePricer
    audit: MarketAudit

    def dense_surface(
        self,
        *,
        tau: np.ndarray,
        log_moneyness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the stored SANOS surface as calls and implied volatilities."""

        tau = np.asarray(tau, dtype=np.float64)
        x = np.asarray(log_moneyness, dtype=np.float64)
        verify_inp(
            tau.ndim == 1
            and x.ndim == 1
            and tau.size > 0
            and x.size > 0
            and np.all(np.isfinite(tau))
            and np.all(np.isfinite(x)),
            lambda: f"expected finite one-dimensional grids, got {tau.shape}, {x.shape}",
        )
        verify_inp(
            np.all(tau >= self.pricer.expiries[0])
            and np.all(tau <= self.pricer.expiries[-1]),
            lambda: (
                "requested maturities exceed the retained SANOS range "
                f"[{self.pricer.expiries[0]}, {self.pricer.expiries[-1]}]"
            ),
        )
        strikes = np.exp(x)
        calls = np.stack(
            [self.pricer.price(float(maturity), strikes) for maturity in tau],
            axis=0,
        ).astype(np.float64, copy=False)
        vols = implied_volatility(
            tau=tau[:, None],
            strike=strikes[None, :],
            call_price=calls,
        )
        return calls, vols


def market_dates(start: date | str, end: date | str) -> list[date]:
    """Return every exchange session in the requested inclusive range."""

    return DateRange(
        start=to_date(start),
        end=to_date(end),
        exclude_half_days=False,
    ).get_business_days()


def _arrays(values: list[object], *, dtype: np.dtype | type) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(value, dtype=dtype) for value in values)


def market_day_from_payload(
    payload: dict[str, object],
    *,
    config: MarketConfig,
) -> TangentLevyMarketDay | None:
    """Validate and adapt one canonical ``market.py`` payload."""

    if len(payload) == 0:
        return None
    verify(
        int(payload["schema_version"]) == MARKET_SCHEMA_VERSION,
        lambda: (
            f"unsupported market schema {payload['schema_version']}; "
            f"expected {MARKET_SCHEMA_VERSION}"
        ),
    )
    ticker = str(payload["ticker"])
    market_date = to_date(payload["d"])
    market = dict(payload["market"])
    expiries_all = np.asarray(market["expiries"], dtype=np.float64)
    strikes_all = _arrays(market["strikes"], dtype=np.float64)
    calls_all = _arrays(market["af_call_mids"], dtype=np.float64)
    bids_all = _arrays(market["bids"], dtype=np.float64)
    asks_all = _arrays(market["asks"], dtype=np.float64)
    masks_all = _arrays(market["model_masks"], dtype=bool)
    american_all = _arrays(market["is_american"], dtype=bool)
    verify(
        len(expiries_all)
        == len(strikes_all)
        == len(calls_all)
        == len(bids_all)
        == len(asks_all)
        == len(masks_all)
        == len(american_all),
        lambda: "market payload has inconsistent per-expiry list lengths",
    )

    min_tau = config.min_tau
    max_tau = config.max_tau
    selected = np.flatnonzero((expiries_all >= min_tau) & (expiries_all <= max_tau))
    pricer = ArbFreePricer.from_dict(payload["af_pricer"])
    retained_input = set(int(index) for index in np.asarray(pricer.input_index))
    selected = np.asarray(
        [index for index in selected if int(index) in retained_input],
        dtype=np.int64,
    )

    def choose(name: str, dtype: np.dtype | type) -> tuple[np.ndarray, ...]:
        values = _arrays(market[name], dtype=dtype)
        return tuple(values[index] for index in selected)

    def choose_european_active(name: str) -> tuple[np.ndarray, ...]:
        values = _arrays(market[name], dtype=bool)
        return tuple(
            values[index] & ~american_all[index] for index in selected
        )

    expiries = expiries_all[selected]
    strikes = tuple(strikes_all[index] for index in selected)
    sanos_calls = tuple(calls_all[index] for index in selected)
    pure_bids = tuple(bids_all[index] for index in selected)
    pure_asks = tuple(asks_all[index] for index in selected)
    model_masks = tuple(
        masks_all[index] & ~american_all[index] for index in selected
    )
    expiry_dates_all = tuple(str(value) for value in market["expiry_dates"])
    expiry_ids_all = np.asarray(market["expiry_ids"], dtype=np.int64)
    is_am_all = np.asarray(market["is_am"], dtype=bool)
    input_quotes = sum(value.size for value in strikes_all)
    active_quotes = sum(np.count_nonzero(value) for value in model_masks)
    excluded_american = sum(
        np.count_nonzero(masks_all[index] & american_all[index]) for index in selected
    )
    audit = MarketAudit(
        market_date=market_date,
        early_close=market_date in get_half_days(market_date, market_date),
        available=True,
        input_expiries=int(expiries_all.size),
        retained_expiries=int(expiries.size),
        retained_min_tau=float(expiries[0]) if expiries.size else math.nan,
        retained_max_tau=float(expiries[-1]) if expiries.size else math.nan,
        input_quotes=input_quotes,
        active_quotes=int(active_quotes),
        excluded_american_quotes=int(excluded_american),
        option_cap_bound=input_quotes >= config.options_filter.max_num_options,
        reason="" if expiries.size > 0 else "no retained European expiry in configured range",
    )
    if expiries.size == 0:
        return None
    verify(
        np.all(np.diff(expiries) > 0.0),
        lambda: f"retained expiries must be strictly increasing, got {expiries}",
    )
    for index, arrays in enumerate(
        zip(strikes, sanos_calls, pure_bids, pure_asks, model_masks, strict=True)
    ):
        shapes = {array.shape for array in arrays}
        verify(
            len(shapes) == 1,
            lambda index=index, shapes=shapes: (
                f"expiry {index} contains inconsistent quote shapes {shapes}"
            ),
        )
    return TangentLevyMarketDay(
        ticker=ticker,
        market_date=market_date,
        spot_ref=float(payload["spot_ref"]),
        spot_return=float(payload["spot_ret"]),
        expiries=expiries,
        strikes=strikes,
        sanos_calls=sanos_calls,
        pure_bids=pure_bids,
        pure_asks=pure_asks,
        model_masks=model_masks,
        call_bids=choose("call_bids", np.float64),
        call_asks=choose("call_asks", np.float64),
        put_bids=choose("put_bids", np.float64),
        put_asks=choose("put_asks", np.float64),
        call_active=choose_european_active("call_active"),
        put_active=choose_european_active("put_active"),
        expiry_dates=tuple(expiry_dates_all[index] for index in selected),
        expiry_ids=expiry_ids_all[selected],
        is_am=is_am_all[selected],
        pricer=pricer,
        audit=audit,
    )


def common_grid_coverage(
    audits: tuple[MarketAudit, ...] | list[MarketAudit],
    *,
    candidate_tau: np.ndarray,
    minimum_fraction: float = 0.99,
) -> CommonGridCoverage:
    """Select candidate maturities covered on a prespecified session fraction.

    Missing sessions remain in the denominator.  Maturity zero is a model
    boundary and is therefore retained whenever supplied, while every positive
    node must lie inside that day's retained market range.
    """

    tau = np.asarray(candidate_tau, dtype=np.float64)
    verify_inp(
        tau.ndim == 1
        and tau.size >= 3
        and tau[0] == 0.0
        and np.all(np.isfinite(tau))
        and np.all(np.diff(tau) > 0.0),
        lambda: "candidate maturities must start at zero and strictly increase",
    )
    verify_inp(
        0.0 < minimum_fraction <= 1.0,
        lambda: f"invalid minimum coverage fraction {minimum_fraction!r}",
    )
    verify_inp(len(audits) > 0, "at least one market audit is required")
    coverage = np.zeros(tau.size, dtype=np.float64)
    coverage[0] = 1.0
    available = 0
    cap_bound = 0
    maturity_tolerance = 0.5 / DAYS_PER_YEAR
    for audit in audits:
        if not audit.available:
            continue
        available += 1
        cap_bound += audit.option_cap_bound
        coverage[1:] += (
            (tau[1:] >= audit.retained_min_tau - maturity_tolerance)
            & (tau[1:] <= audit.retained_max_tau + maturity_tolerance)
        )
    coverage[1:] /= len(audits)
    selected = tau[coverage >= minimum_fraction]
    verify(
        selected.size >= 3,
        lambda: (
            "candidate grid has fewer than three nodes at minimum coverage "
            f"{minimum_fraction}: fractions={coverage}"
        ),
    )
    return CommonGridCoverage(
        candidate_tau=tau,
        coverage_fraction=coverage,
        selected_tau=selected,
        minimum_fraction=float(minimum_fraction),
        sessions=len(audits),
        available_sessions=available,
        option_cap_fraction=cap_bound / len(audits),
    )


def load_market_day(
    ticker: str,
    market_date: date,
    *,
    config: MarketConfig = MarketConfig(),
    verbose: Context | None = Context.all,
) -> TangentLevyMarketDay | None:
    """Load one smoothed market through the canonical ``market.py`` producer."""

    payload = smooth_interpolated_market(
        ticker,
        market_date,
        smoothness=config.smoothness,
        options_filter=config.options_filter,
        raw_day_filter=config.raw_day_filter,
        expiry_filter=config.expiry_filter,
        mk_arb_free=config.mk_arb_free,
        verbose=verbose,
    )
    return market_day_from_payload(payload, config=config)


def iter_market_days(
    ticker: str,
    start: date | str,
    end: date | str,
    *,
    config: MarketConfig = MarketConfig(),
    verbose: Context | None = Context.all,
) -> Iterator[tuple[date, TangentLevyMarketDay | None]]:
    """Yield every requested session and its market, preserving empty dates."""

    for market_date in market_dates(start, end):
        yield market_date, load_market_day(
            ticker,
            market_date,
            config=config,
            verbose=verbose,
        )
