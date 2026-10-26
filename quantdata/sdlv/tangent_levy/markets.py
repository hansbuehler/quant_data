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
from quantdata.pool import PoolConfig
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
    """One normalized smoothed market from a fixed-grid or quote-level source."""

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
    pricer: ArbFreePricer | None
    audit: MarketAudit
    fit_weights: tuple[np.ndarray, ...] | None = None
    source: str = "smooth_market"

    def dense_surface(
        self,
        *,
        tau: np.ndarray,
        log_moneyness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the stored SANOS surface as calls and implied volatilities."""

        tau = np.asarray(tau, dtype=np.float64)
        x = np.asarray(log_moneyness, dtype=np.float64)
        verify(
            self.pricer is not None,
            lambda: (
                "dense-surface interpolation is unavailable for a market day "
                f"loaded from {self.source!r}"
            ),
        )
        verify_inp(
            tau.ndim == 1
            and x.ndim == 1
            and tau.size > 0
            and x.size > 0
            and np.all(np.isfinite(tau))
            and np.all(np.isfinite(x)),
            lambda: (
                f"expected finite one-dimensional grids, got {tau.shape}, {x.shape}"
            ),
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
        return tuple(values[index] & ~american_all[index] for index in selected)

    expiries = expiries_all[selected]
    strikes = tuple(strikes_all[index] for index in selected)
    sanos_calls = tuple(calls_all[index] for index in selected)
    pure_bids = tuple(bids_all[index] for index in selected)
    pure_asks = tuple(asks_all[index] for index in selected)
    model_masks = tuple(masks_all[index] & ~american_all[index] for index in selected)
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
        reason=""
        if expiries.size > 0
        else "no retained European expiry in configured range",
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
        fit_weights=None,
        source="smooth_market",
    )


def grid_market_days_from_panel(
    panel: object,
    *,
    ticker: str,
    start: date | str,
    end: date | str,
    config: MarketConfig,
) -> dict[date, TangentLevyMarketDay]:
    """Adapt the cached Dysanos/grid panel without fabricating quote intervals."""

    dates = np.asarray(panel["dates"], dtype="datetime64[D]")
    expiries = np.asarray(panel["expiries"], dtype=np.float64)
    xstrikes = np.asarray(panel["xstrikes"], dtype=np.float64)
    calls = np.asarray(panel["mids"], dtype=np.float64)
    weights = np.asarray(panel["weights"], dtype=np.float64)
    spot_ref = np.asarray(panel["spot_ref"], dtype=np.float64)
    spot_return = np.asarray(panel["spot_ret"], dtype=np.float64)
    verify(
        dates.ndim == 1
        and expiries.ndim == 2
        and xstrikes.ndim == 3
        and calls.ndim == 3
        and weights.shape == calls.shape
        and expiries.shape == calls.shape[:2]
        and xstrikes.shape == (calls.shape[0], calls.shape[1], calls.shape[2] + 2)
        and spot_ref.shape == dates.shape
        and spot_return.shape == dates.shape
        and calls.shape[0] == dates.size,
        lambda: (
            "Dysanos/grid cache has an incompatible panel schema: "
            f"dates={dates.shape}, expiries={expiries.shape}, "
            f"xstrikes={xstrikes.shape}, mids={calls.shape}, "
            f"weights={weights.shape}"
        ),
    )
    verify(
        calls.shape[2] == config.grid_num_strikes,
        lambda: (
            "Dysanos/grid strike count does not match MarketConfig: "
            f"{calls.shape[2]} != {config.grid_num_strikes}"
        ),
    )
    start_date = to_date(start)
    end_date = to_date(end)
    result: dict[date, TangentLevyMarketDay] = {}
    for date_index, numpy_date in enumerate(dates):
        market_date = to_date(numpy_date)
        if market_date < start_date or market_date > end_date:
            continue
        selected = np.flatnonzero(
            (expiries[date_index] >= config.min_tau - 1.0e-12)
            & (expiries[date_index] <= config.max_tau + 1.0e-12)
        )
        if selected.size == 0:
            continue
        selected_expiries = expiries[date_index, selected]
        selected_strikes = tuple(
            np.asarray(xstrikes[date_index, maturity_index, 1:-1])
            for maturity_index in selected
        )
        selected_calls = tuple(
            np.asarray(calls[date_index, maturity_index]) for maturity_index in selected
        )
        selected_weights = tuple(
            np.asarray(weights[date_index, maturity_index])
            for maturity_index in selected
        )
        masks = tuple(
            np.isfinite(strike)
            & np.isfinite(call)
            & np.isfinite(weight)
            & (strike > 0.0)
            & (weight > 0.0)
            for strike, call, weight in zip(
                selected_strikes,
                selected_calls,
                selected_weights,
                strict=True,
            )
        )
        missing_prices = tuple(np.full_like(value, np.nan) for value in selected_calls)
        inactive = tuple(np.zeros_like(value, dtype=bool) for value in selected_calls)
        input_quotes = int(sum(value.size for value in selected_calls))
        active_quotes = int(sum(np.count_nonzero(value) for value in masks))
        audit = MarketAudit(
            market_date=market_date,
            early_close=False,
            available=True,
            input_expiries=int(expiries.shape[1]),
            retained_expiries=int(selected.size),
            retained_min_tau=float(selected_expiries[0]),
            retained_max_tau=float(selected_expiries[-1]),
            input_quotes=input_quotes,
            active_quotes=active_quotes,
            excluded_american_quotes=0,
            option_cap_bound=False,
            reason=(
                "source=dysanos/grid cache; original quote intervals and raw "
                "option-cap diagnostics are unavailable"
            ),
        )
        result[market_date] = TangentLevyMarketDay(
            ticker=ticker,
            market_date=market_date,
            spot_ref=float(spot_ref[date_index]),
            spot_return=float(spot_return[date_index]),
            expiries=selected_expiries,
            strikes=selected_strikes,
            sanos_calls=selected_calls,
            pure_bids=missing_prices,
            pure_asks=missing_prices,
            model_masks=masks,
            call_bids=missing_prices,
            call_asks=missing_prices,
            put_bids=missing_prices,
            put_asks=missing_prices,
            call_active=inactive,
            put_active=inactive,
            expiry_dates=tuple("" for _ in selected),
            expiry_ids=np.full(selected.size, -1, dtype=np.int64),
            is_am=np.zeros(selected.size, dtype=bool),
            pricer=None,
            audit=audit,
            fit_weights=selected_weights,
            source="dysanos_grid_cache",
        )
    return result


def load_grid_market_days(
    ticker: str,
    start: date | str,
    end: date | str,
    *,
    config: MarketConfig = MarketConfig(),
    verbose: Context | None = Context.all,
) -> dict[date, TangentLevyMarketDay]:
    """Load the existing Dysanos/grid panel strictly in cache-only mode."""

    from quantdata.dysanos.grid.torch_market import GridConfig, GridMarketDefinition

    verify_inp(
        config.source == "dysanos_grid_cache",
        lambda: f"grid cache loader received source {config.source!r}",
    )
    definition = GridMarketDefinition(
        ticker=ticker,
        date_range=DateRange(
            start=config.grid_cache_start,
            end=config.grid_cache_end,
            exclude_half_days=True,
        ),
        grid_config=GridConfig(
            smoothness=config.smoothness,
            grid_num_strikes=config.grid_num_strikes,
        ),
        raw_day_filter=config.raw_day_filter,
        expiry_filter=config.expiry_filter,
        mk_arb_free=config.mk_arb_free,
        options_filter=config.options_filter,
    )
    panel = definition.load(
        pool_config=PoolConfig.off(),
        override_cache_mode="cacheonly",
        verbose=Context.quiet if verbose is None else verbose,
    )
    return grid_market_days_from_panel(
        panel,
        ticker=ticker,
        start=start,
        end=end,
        config=config,
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
        coverage[1:] += (tau[1:] >= audit.retained_min_tau - maturity_tolerance) & (
            tau[1:] <= audit.retained_max_tau + maturity_tolerance
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
    """Load one market from the configured source.

    The default grid route is deliberately cache-only.  It never falls back to
    the raw-data producer when the shared Dysanos/grid artifact is missing.
    """

    if config.source == "dysanos_grid_cache":
        return load_grid_market_days(
            ticker,
            market_date,
            market_date,
            config=config,
            verbose=verbose,
        ).get(market_date)

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

    dates = market_dates(start, end)
    if config.source == "dysanos_grid_cache":
        cached = load_grid_market_days(
            ticker,
            start,
            end,
            config=config,
            verbose=verbose,
        )
        for market_date in dates:
            yield market_date, cached.get(market_date)
        return

    for market_date in dates:
        yield (
            market_date,
            load_market_day(
                ticker,
                market_date,
                config=config,
                verbose=verbose,
            ),
        )
