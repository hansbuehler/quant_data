"""Adapt ``market.py`` output to the exact fixed-weekly FLEX universe."""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from datetime import date
from functools import cached_property

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.verbose import Context

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.dysanos.flex.config import FlexGeometryConfig, FlexMarketConfig
from quantdata.dysanos.market import smooth_interpolated_market
from quantdata.sanos.arbfree import ArbFreePricer
from quantdata.widb.options_filter import OptionsFilter

MAX_MARKET_LP_PROCESSES = 4


def market_process_count(requested: int) -> int:
    """Apply the project-wide CPU-process ceiling for market-surface LP work."""
    verify_inp(
        isinstance(requested, int) and not isinstance(requested, bool) and requested > 0,
        lambda: f"Expected a positive market process count, found {requested!r}",
    )
    if requested > MAX_MARKET_LP_PROCESSES:
        warnings.warn(
            f"Capping FLEX market processes from {requested} to "
            f"{MAX_MARKET_LP_PROCESSES}",
            RuntimeWarning,
            stacklevel=2,
        )
    return min(requested, MAX_MARKET_LP_PROCESSES)


@dataclass(frozen=True)
class MarketEligibilityAudit:
    """Counts before and after the exact horizon and canonical-NMS filters."""

    active_european_books: int
    horizon_eligible_books: int
    nms_eligible_books: int
    input_expiries: int
    eligible_expiries: int


@dataclass(frozen=True)
class FlexMarketDay:
    """Eligible real call and put books plus ATM curves for one market day."""

    ticker: str
    trading_date: date
    expiries_y: np.ndarray
    expiry_dates: tuple[str, ...]
    expiry_ids: np.ndarray
    expiry_is_am: np.ndarray
    expiry_index: np.ndarray
    strikes: np.ndarray
    is_call: np.ndarray
    bids: np.ndarray
    asks: np.ndarray
    fit_prices: np.ndarray
    log_volumes: np.ndarray
    log_open_interests: np.ndarray
    quote_ids: tuple[str, ...]
    atm_annual_vols: np.ndarray
    model_atm_total_vols: np.ndarray
    strike_lower_bounds: np.ndarray
    strike_upper_bounds: np.ndarray
    market_uid: str
    audit: MarketEligibilityAudit
    spot_ref: float | None = None

    def __post_init__(self) -> None:
        Q = len(self.strikes)
        quote_arrays = (
            self.expiry_index,
            self.is_call,
            self.bids,
            self.asks,
            self.fit_prices,
            self.log_volumes,
            self.log_open_interests,
        )
        if any(array.shape != (Q,) for array in quote_arrays):
            raise ValueError(
                "Every FLEX quote field must be a one-dimensional array of equal length"
            )
        if len(self.quote_ids) != Q:
            raise ValueError("'quote_ids' must have one entry per quote")
        if Q == 0:
            raise ValueError("A FLEX market day must contain at least one eligible quote")

        E = len(self.expiries_y)
        expiry_arrays = (
            self.expiry_ids,
            self.expiry_is_am,
            self.atm_annual_vols,
            self.strike_lower_bounds,
            self.strike_upper_bounds,
        )
        if self.expiries_y.ndim != 1 or E == 0:
            raise ValueError("'expiries_y' must be a non-empty vector")
        if any(array.shape != (E,) for array in expiry_arrays):
            raise ValueError("Every FLEX expiry field must have one value per expiry")
        if len(self.expiry_dates) != E:
            raise ValueError("'expiry_dates' must have one value per expiry")
        if not np.all(np.diff(self.expiries_y) > 0.0):
            raise ValueError("'expiries_y' must be strictly increasing")
        if not np.all(self.expiries_y >= 0.0):
            raise ValueError("Observed FLEX expiries must be non-negative")
        if not np.all(
            (self.expiry_index >= 0) & (self.expiry_index < len(self.expiries_y))
        ):
            raise ValueError("'expiry_index' contains an out-of-range value")
        if not np.all(np.isfinite(self.strikes) & (self.strikes > 0.0)):
            raise ValueError("Every normalized strike must be finite and positive")
        if not np.all(
            np.isfinite(self.bids)
            & np.isfinite(self.asks)
            & np.isfinite(self.fit_prices)
            & (self.bids >= 0.0)
            & (self.asks > self.bids)
        ):
            raise ValueError("Every retained quote must have a valid book and fit price")
        if not np.all(
            np.isfinite(self.atm_annual_vols) & (self.atm_annual_vols > 0.0)
        ):
            raise ValueError("Every observed ATM annual volatility must be positive")
        if not np.all(
            np.isfinite(self.model_atm_total_vols)
            & (self.model_atm_total_vols > 0.0)
        ):
            raise ValueError("Every model-node ATM total volatility must be positive")
        if not np.all(
            (self.strike_lower_bounds > 0.0)
            & (self.strike_lower_bounds < 1.0)
            & (self.strike_upper_bounds > 1.0)
        ):
            raise ValueError("Every canonical-NMS strike interval must straddle ATM")
        if self.spot_ref is not None and not (
            np.isfinite(self.spot_ref) and self.spot_ref > 0.0
        ):
            raise ValueError("'spot_ref' must be finite and positive when supplied")

    @property
    def num_quotes(self) -> int:
        """Number of separate eligible real call/put observations."""
        return len(self.strikes)

    @cached_property
    def business_dtes(self) -> np.ndarray:
        """Observed expiry nodes in the repository's business-day clock."""
        return self.expiries_y * DAYS_PER_YEAR

    @cached_property
    def quote_business_dtes(self) -> np.ndarray:
        """Business DTE for every separate real option book."""
        return self.business_dtes[self.expiry_index]

    @cached_property
    def half_spreads(self) -> np.ndarray:
        """Observed half-spread for every real book."""
        return 0.5 * (self.asks - self.bids)

    @cached_property
    def call_equivalent_fit_prices(self) -> np.ndarray:
        """Smooth fit targets expressed as calls without merging books."""
        return self.fit_prices + np.where(self.is_call, 0.0, 1.0 - self.strikes)

    @cached_property
    def call_equivalent_book_mids(self) -> np.ndarray:
        """Real book mids expressed as calls without merging calls and puts."""
        mids = 0.5 * (self.bids + self.asks)
        return mids + np.where(self.is_call, 0.0, 1.0 - self.strikes)

    @cached_property
    def training_fingerprint(self) -> str:
        """Hash every ordered field that changes a FLEX fitting objective."""
        digest = hashlib.sha256()
        for value in (
            self.ticker,
            self.trading_date.isoformat(),
            self.market_uid,
            *self.expiry_dates,
            *self.quote_ids,
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="little"))
            digest.update(encoded)
        for array in (
            self.expiries_y,
            self.expiry_ids,
            self.expiry_is_am,
            self.expiry_index,
            self.strikes,
            self.is_call,
            self.bids,
            self.asks,
            self.fit_prices,
            self.atm_annual_vols,
            self.model_atm_total_vols,
            self.strike_lower_bounds,
            self.strike_upper_bounds,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()


def _atm_curves_and_bounds(
    payload: dict[str, object],
    market_expiries_y: np.ndarray,
    *,
    geometry: FlexGeometryConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the AF pricer and compute exact canonical-NMS eligibility."""
    if "af_pricer" not in payload:
        raise ValueError("Market payload does not contain its serialized 'af_pricer'")
    pricer = ArbFreePricer.from_dict(payload["af_pricer"])
    model_expiries_y = geometry.model_expiries_y
    model_atm_annual_vols = np.asarray(
        pricer.interpolate_atm_vols(model_expiries_y),
        dtype=np.float64,
    )
    model_atm_total_vols = model_atm_annual_vols * np.sqrt(model_expiries_y)
    atm_annual_vols = np.asarray(
        pricer.interpolate_atm_vols(market_expiries_y),
        dtype=np.float64,
    )
    endpoints = NMS.from_nms(
        np.asarray(geometry.nms_range, dtype=np.float64)[None, :],
        t=market_expiries_y[:, None],
        vol=atm_annual_vols[:, None],
        eps_vol=float(model_atm_annual_vols[0]),
    )
    return (
        atm_annual_vols,
        model_atm_total_vols,
        np.asarray(endpoints[:, 0], dtype=np.float64),
        np.asarray(endpoints[:, 1], dtype=np.float64),
    )


def market_day_from_payload(
    payload: dict[str, object],
    *,
    ticker: str = "UNKNOWN",
    geometry: FlexGeometryConfig = FlexGeometryConfig(),
    config: FlexMarketConfig = FlexMarketConfig(),
) -> FlexMarketDay | None:
    """Extract the exact-horizon, canonical-NMS FLEX universe from a payload."""
    del config  # The payload already records its upstream filtering identity.
    if not payload:
        return None
    market = payload["market"]
    all_expiries_y = np.asarray(market["expiries"], dtype=np.float64)
    if len(all_expiries_y) == 0:
        return None

    (
        all_atm_annual_vols,
        model_atm_total_vols,
        all_lower_bounds,
        all_upper_bounds,
    ) = _atm_curves_and_bounds(payload, all_expiries_y, geometry=geometry)

    expiry_indices: list[np.ndarray] = []
    strikes: list[np.ndarray] = []
    is_calls: list[np.ndarray] = []
    bids: list[np.ndarray] = []
    asks: list[np.ndarray] = []
    fit_prices: list[np.ndarray] = []
    log_volumes: list[np.ndarray] = []
    log_open_interests: list[np.ndarray] = []
    quote_ids: list[str] = []
    active_european_books = 0
    horizon_eligible_books = 0
    nms_eligible_books = 0

    for expiry_index, expiry_y in enumerate(all_expiries_y):
        within_horizon = (
            expiry_y >= 0.0
            and expiry_y <= geometry.model_expiries_y[-1] + 1.0e-12
        )
        strikes_t = np.asarray(market["strikes"][expiry_index], dtype=np.float64)
        european = ~np.asarray(market["is_american"][expiry_index], dtype=bool)
        fit_available = np.asarray(
            market["af_mid_available"][expiry_index],
            dtype=bool,
        )
        within_nms = (
            (strikes_t >= all_lower_bounds[expiry_index])
            & (strikes_t <= all_upper_bounds[expiry_index])
        )
        expiry_id = int(np.asarray(market["expiry_ids"])[expiry_index])

        side_fields = (
            (
                True,
                "C",
                "call_active",
                "call_bids",
                "call_asks",
                "af_call_mids",
                "call_log_volumes",
                "call_log_open_interests",
            ),
            (
                False,
                "P",
                "put_active",
                "put_bids",
                "put_asks",
                "af_put_mids",
                "put_log_volumes",
                "put_log_open_interests",
            ),
        )
        for (
            is_call,
            side_label,
            active_key,
            bid_key,
            ask_key,
            fit_key,
            volume_key,
            open_interest_key,
        ) in side_fields:
            active = (
                np.asarray(market[active_key][expiry_index], dtype=bool)
                & european
                & fit_available
            )
            active_european_books += int(np.sum(active))
            if not within_horizon:
                continue
            horizon_eligible_books += int(np.sum(active))
            active &= within_nms
            nms_eligible_books += int(np.sum(active))
            if not np.any(active):
                continue
            count = int(np.sum(active))
            selected_strikes = strikes_t[active]
            expiry_indices.append(
                np.full(count, expiry_index, dtype=np.int64)
            )
            strikes.append(selected_strikes)
            is_calls.append(np.full(count, is_call, dtype=bool))
            bids.append(
                np.asarray(market[bid_key][expiry_index], dtype=np.float64)[active]
            )
            asks.append(
                np.asarray(market[ask_key][expiry_index], dtype=np.float64)[active]
            )
            fit_prices.append(
                np.asarray(market[fit_key][expiry_index], dtype=np.float64)[active]
            )
            log_volumes.append(
                np.asarray(market[volume_key][expiry_index], dtype=np.float64)[active]
            )
            log_open_interests.append(
                np.asarray(
                    market[open_interest_key][expiry_index],
                    dtype=np.float64,
                )[active]
            )
            quote_ids.extend(
                f"{payload['d']}:{expiry_id}:{strike:.12g}:{side_label}"
                for strike in selected_strikes
            )

    if len(strikes) == 0:
        return None

    used_expiry_indices = np.unique(np.concatenate(expiry_indices))
    expiry_remap = np.full(len(all_expiries_y), -1, dtype=np.int64)
    expiry_remap[used_expiry_indices] = np.arange(len(used_expiry_indices))
    expiry_dates_all = tuple(str(value) for value in market["expiry_dates"])
    expiry_ids_all = np.asarray(market["expiry_ids"], dtype=np.int64)
    expiry_is_am_all = np.asarray(market["is_am"], dtype=bool)
    return FlexMarketDay(
        ticker=ticker,
        trading_date=payload["d"],
        expiries_y=all_expiries_y[used_expiry_indices],
        expiry_dates=tuple(expiry_dates_all[index] for index in used_expiry_indices),
        expiry_ids=expiry_ids_all[used_expiry_indices],
        expiry_is_am=expiry_is_am_all[used_expiry_indices],
        expiry_index=expiry_remap[np.concatenate(expiry_indices)],
        strikes=np.concatenate(strikes),
        is_call=np.concatenate(is_calls),
        bids=np.concatenate(bids),
        asks=np.concatenate(asks),
        fit_prices=np.concatenate(fit_prices),
        log_volumes=np.concatenate(log_volumes),
        log_open_interests=np.concatenate(log_open_interests),
        quote_ids=tuple(quote_ids),
        atm_annual_vols=all_atm_annual_vols[used_expiry_indices],
        model_atm_total_vols=model_atm_total_vols,
        strike_lower_bounds=all_lower_bounds[used_expiry_indices],
        strike_upper_bounds=all_upper_bounds[used_expiry_indices],
        market_uid=str(market["uid"]),
        audit=MarketEligibilityAudit(
            active_european_books=active_european_books,
            horizon_eligible_books=horizon_eligible_books,
            nms_eligible_books=nms_eligible_books,
            input_expiries=len(all_expiries_y),
            eligible_expiries=len(used_expiry_indices),
        ),
        spot_ref=(
            None
            if payload.get("spot_ref") is None
            else float(payload["spot_ref"])
        ),
    )


def load_market_day(
    ticker: str,
    trading_date: date,
    *,
    geometry: FlexGeometryConfig = FlexGeometryConfig(),
    config: FlexMarketConfig = FlexMarketConfig(),
    verbose: Context | None = Context.all,
) -> FlexMarketDay | None:
    """Load one market and apply FLEX's exact floating horizon and NMS range."""
    payload = smooth_interpolated_market(
        ticker,
        trading_date,
        smoothness=config.smoothness,
        options_filter=OptionsFilter(
            max_num_options=config.max_num_options,
            max_business_days_to_expiry=math.ceil(geometry.max_business_dte),
            # The upstream NMS5 filter is deliberately non-binding. FLEX
            # applies its required canonical-NMS interval from the AF ATM curve.
            max_nms=(-100.0, 100.0),
        ),
        verbose=verbose,
    )
    return market_day_from_payload(
        payload,
        ticker=ticker,
        geometry=geometry,
        config=config,
    )
