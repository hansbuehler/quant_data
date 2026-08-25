"""Load filtered smooth Dysanos markets and their arbitrage-free pricers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
from cdxcore.err import verify
from cdxcore.verbose import Context
from cdxcore.version import version

from quantdata.dysanos.market import MARKET_SCHEMA_VERSION, smooth_interpolated_market
from quantdata.sanos.arbfree import ArbFreePricer, MkArbFree
from quantdata.widb.day import ExpiryFilter, RawDayFilter
from quantdata.widb.options_filter import OptionsFilter, filter_options_day


@version("0.1", dependencies=[RawDayFilter, ExpiryFilter, OptionsFilter, MkArbFree])
@dataclass(frozen=True, kw_only=True)
class MarketConfig:
    """Data-functional inputs for a normalized smooth Dysanos market."""

    smoothness: float = 0.25
    raw_day_filter: RawDayFilter = field(default_factory=RawDayFilter)
    expiry_filter: ExpiryFilter = field(default_factory=ExpiryFilter)
    options_filter: OptionsFilter = field(default_factory=OptionsFilter)
    mk_arb_free: MkArbFree = field(default_factory=MkArbFree)

    def __post_init__(self) -> None:
        if not np.isfinite(self.smoothness) or self.smoothness <= 0.01:
            raise ValueError(
                f"'smoothness' must be finite and greater than 0.01, found {self.smoothness!r}"
            )
        if not self.options_filter.normalize:
            raise ValueError("'options_filter.normalize' must be True")


@dataclass(frozen=True)
class ListedOptionBook:
    """Filtered listed contracts with prices normalized by discounted forward."""

    expiries: np.ndarray
    expiry_ids: np.ndarray
    is_am: np.ndarray
    expiry_dates: tuple[date, ...]
    absolute_strikes: tuple[np.ndarray, ...]
    strikes_over_forward: tuple[np.ndarray, ...]
    forwards: np.ndarray
    discount_factors: np.ndarray
    call_bids: tuple[np.ndarray, ...]
    call_asks: tuple[np.ndarray, ...]
    put_bids: tuple[np.ndarray, ...]
    put_asks: tuple[np.ndarray, ...]
    call_active: tuple[np.ndarray, ...]
    put_active: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class DysanosMarketDay:
    """Filtered normalized quotes and the smooth surface that fitted them."""

    ticker: str
    market_date: date
    spot_ref: float
    spot_return: float
    expiries: np.ndarray
    expiry_dates: tuple[date, ...]
    strikes: tuple[np.ndarray, ...]
    bids: tuple[np.ndarray, ...]
    asks: tuple[np.ndarray, ...]
    call_bids: tuple[np.ndarray, ...]
    call_asks: tuple[np.ndarray, ...]
    put_bids: tuple[np.ndarray, ...]
    put_asks: tuple[np.ndarray, ...]
    call_active: tuple[np.ndarray, ...]
    put_active: tuple[np.ndarray, ...]
    model_masks: tuple[np.ndarray, ...]
    input_indices: np.ndarray
    af_pricer: ArbFreePricer
    listed_options: ListedOptionBook | None


def _array_tuple(values: list[object], *, dtype: np.dtype | type) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(value, dtype=dtype) for value in values)


def _listed_option_book(
    ticker: str,
    market_date: date,
    *,
    config: MarketConfig,
) -> ListedOptionBook:
    # The fitted surface deliberately limits its input to the most relevant 1,000
    # rows.  Contract follow-up needs the wider listed book so that a contract
    # selected today can still be found after spot and time-to-expiry change.
    listed_filter = replace(
        config.options_filter,
        max_num_options=100_000,
        max_nms=1.0e6,
        min_vegaSqrtVar=0.0,
        min_log_volume=0.0,
        min_log_open_interest=0.0,
        min_opt_per_expiry=1,
    )
    data = filter_options_day(
        ticker,
        market_date,
        raw_day_filter=config.raw_day_filter,
        expiry_filter=config.expiry_filter,
        options_filter=listed_filter,
        mk_arb_free=config.mk_arb_free,
        raise_if_bad_day=False,
        verbose=Context.quiet,
    )
    verify(data is not None and len(data) > 0, lambda: "listed option book is empty")
    data = data.loc[~data["is_american"]].copy()
    verify(len(data) > 0, lambda: "listed European option book is empty")
    spot_ref = float(data["spot_ref"].unique().item())

    expiry_identity = (
        data[["expiry_id", "tte_y"]]
        .drop_duplicates()
        .sort_values(["tte_y", "expiry_id"], kind="stable")
    )
    expiry_ids = expiry_identity["expiry_id"].to_numpy(dtype=np.int64)
    expiries = expiry_identity["tte_y"].to_numpy(dtype=np.float64)
    expiry_dates: list[date] = []
    is_am: list[bool] = []
    absolute_strikes: list[np.ndarray] = []
    strikes_over_forward: list[np.ndarray] = []
    forwards: list[float] = []
    discounts: list[float] = []
    call_bids: list[np.ndarray] = []
    call_asks: list[np.ndarray] = []
    put_bids: list[np.ndarray] = []
    put_asks: list[np.ndarray] = []
    call_active: list[np.ndarray] = []
    put_active: list[np.ndarray] = []
    for expiry_id, expiry in zip(expiry_ids, expiries, strict=True):
        group = data.loc[data["expiry_id"] == expiry_id].sort_values(
            "pure_strike", kind="stable"
        )
        group_expiries = group["tte_y"].unique()
        group_dates = np.unique(
            group["expiry"].to_numpy().astype("datetime64[D]")
        )
        group_is_am = group["is_am"].unique()
        normalized_forwards = group["fwd"].unique()
        group_discounts = group["df"].unique()
        verify(
            len(group_expiries)
            == len(group_dates)
            == len(group_is_am)
            == len(normalized_forwards)
            == len(group_discounts)
            == 1,
            lambda: f"expiry ID {expiry_id} has inconsistent contract metadata",
        )
        verify(
            float(group_expiries.item()) == float(expiry),
            lambda: f"expiry ID {expiry_id} changed time to expiry",
        )
        discounted_forward = float(
            group_discounts.item() * normalized_forwards.item()
        )
        verify(
            np.isfinite(discounted_forward) and discounted_forward > 0.0,
            lambda: f"expiry ID {expiry_id} has an invalid discounted forward",
        )
        call_bid = group["mkt_call_bid"].to_numpy(dtype=np.float64) / discounted_forward
        call_ask = group["mkt_call_ask"].to_numpy(dtype=np.float64) / discounted_forward
        put_bid = group["mkt_put_bid"].to_numpy(dtype=np.float64) / discounted_forward
        put_ask = group["mkt_put_ask"].to_numpy(dtype=np.float64) / discounted_forward

        expiry_dates.append(date.fromisoformat(str(group_dates.item())))
        is_am.append(bool(group_is_am.item()))
        absolute_strikes.append(
            group["strike"].to_numpy(dtype=np.float64) * spot_ref
        )
        strikes_over_forward.append(
            group["pure_strike"].to_numpy(dtype=np.float64)
        )
        forwards.append(float(normalized_forwards.item() * spot_ref))
        discounts.append(float(group_discounts.item()))
        call_bids.append(call_bid)
        call_asks.append(call_ask)
        put_bids.append(put_bid)
        put_asks.append(put_ask)
        call_active.append(
            np.isfinite(call_bid)
            & np.isfinite(call_ask)
            & (call_bid >= 0.0)
            & (call_ask > call_bid)
        )
        put_active.append(
            np.isfinite(put_bid)
            & np.isfinite(put_ask)
            & (put_bid >= 0.0)
            & (put_ask > put_bid)
        )

    return ListedOptionBook(
        expiries=expiries,
        expiry_ids=expiry_ids,
        is_am=np.asarray(is_am, dtype=bool),
        expiry_dates=tuple(expiry_dates),
        absolute_strikes=tuple(absolute_strikes),
        strikes_over_forward=tuple(strikes_over_forward),
        forwards=np.asarray(forwards, dtype=np.float64),
        discount_factors=np.asarray(discounts, dtype=np.float64),
        call_bids=tuple(call_bids),
        call_asks=tuple(call_asks),
        put_bids=tuple(put_bids),
        put_asks=tuple(put_asks),
        call_active=tuple(call_active),
        put_active=tuple(put_active),
    )


@version("0.3", dependencies=[MarketConfig, smooth_interpolated_market, filter_options_day])
def load_market_day(
    ticker: str,
    market_date: date,
    *,
    config: MarketConfig = MarketConfig(),
    include_contract_details: bool = False,
    verbose: Context | None = Context.all,
) -> DysanosMarketDay | None:
    """Return one filtered market and an AF pricer for its complete surface."""

    payload = smooth_interpolated_market(
        ticker,
        market_date,
        smoothness=config.smoothness,
        raw_day_filter=config.raw_day_filter,
        expiry_filter=config.expiry_filter,
        options_filter=config.options_filter,
        mk_arb_free=config.mk_arb_free,
        verbose=verbose,
    )
    if len(payload) == 0:
        return None
    verify(
        int(payload["schema_version"]) == MARKET_SCHEMA_VERSION,
        lambda: (
            f"unsupported market schema {payload['schema_version']}; "
            f"expected {MARKET_SCHEMA_VERSION}"
        ),
    )
    market = dict(payload["market"])
    pricer = ArbFreePricer.from_dict(payload["af_pricer"])
    input_indices = np.asarray(pricer.input_index, dtype=np.int64)
    all_expiries = np.asarray(market["expiries"], dtype=np.float64)
    all_strikes = _array_tuple(market["strikes"], dtype=np.float64)
    all_bids = _array_tuple(market["bids"], dtype=np.float64)
    all_asks = _array_tuple(market["asks"], dtype=np.float64)
    all_call_bids = _array_tuple(market["call_bids"], dtype=np.float64)
    all_call_asks = _array_tuple(market["call_asks"], dtype=np.float64)
    all_put_bids = _array_tuple(market["put_bids"], dtype=np.float64)
    all_put_asks = _array_tuple(market["put_asks"], dtype=np.float64)
    all_call_active = _array_tuple(market["call_active"], dtype=bool)
    all_put_active = _array_tuple(market["put_active"], dtype=bool)
    all_masks = _array_tuple(market["model_masks"], dtype=bool)
    all_american = _array_tuple(market["is_american"], dtype=bool)

    strikes: list[np.ndarray] = []
    bids: list[np.ndarray] = []
    asks: list[np.ndarray] = []
    call_bids: list[np.ndarray] = []
    call_asks: list[np.ndarray] = []
    put_bids: list[np.ndarray] = []
    put_asks: list[np.ndarray] = []
    call_active: list[np.ndarray] = []
    put_active: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for retained_index, input_index in enumerate(input_indices):
        index = int(input_index)
        mask = all_masks[index] & ~all_american[index] & pricer.masks[retained_index]
        strikes.append(all_strikes[index][mask])
        bids.append(all_bids[index][mask])
        asks.append(all_asks[index][mask])
        call_bids.append(all_call_bids[index][mask])
        call_asks.append(all_call_asks[index][mask])
        put_bids.append(all_put_bids[index][mask])
        put_asks.append(all_put_asks[index][mask])
        call_active.append(all_call_active[index][mask])
        put_active.append(all_put_active[index][mask])
        masks.append(np.ones(np.count_nonzero(mask), dtype=bool))

    retained_expiries = all_expiries[input_indices]
    verify(
        np.array_equal(retained_expiries, np.asarray(pricer.expiries)),
        lambda: "market expiry mapping does not match the reconstructed AF pricer",
    )
    return DysanosMarketDay(
        ticker=str(payload["ticker"]),
        market_date=payload["d"],
        spot_ref=float(payload["spot_ref"]),
        spot_return=float(payload["spot_ret"]),
        expiries=retained_expiries,
        expiry_dates=tuple(
            date.fromisoformat(str(market["expiry_dates"][int(index)]))
            for index in input_indices
        ),
        strikes=tuple(strikes),
        bids=tuple(bids),
        asks=tuple(asks),
        call_bids=tuple(call_bids),
        call_asks=tuple(call_asks),
        put_bids=tuple(put_bids),
        put_asks=tuple(put_asks),
        call_active=tuple(call_active),
        put_active=tuple(put_active),
        model_masks=tuple(masks),
        input_indices=input_indices,
        af_pricer=pricer,
        listed_options=(
            _listed_option_book(ticker, market_date, config=config)
            if include_contract_details
            else None
        ),
    )
