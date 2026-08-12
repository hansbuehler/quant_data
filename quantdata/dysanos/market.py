"""
Standardized market, interpolated by smooth SANOS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from numbers import Real

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.pretty import PrettyObject
from cdxcore.uniquehash import named_unique_filename48_8, unique_hash8
from cdxcore.verbose import Context

from quantdata.dysanos import SubDir, _locations_dysanos
from quantdata.sanos.arbfree import ArbFreePricer, MkArbFree
from quantdata.widb.day import ExpiryFilter, RawDayFilter
from quantdata.widb.options_filter import OptionsFilter, filter_options_day

_cache = _locations_dysanos.training("market")
MARKET_SCHEMA_VERSION = 4
_MIN_SMOOTHNESS = 0.01


def _validated_smoothness(smoothness: object) -> float:
    verify_inp(
        isinstance(smoothness, Real)
        and not isinstance(smoothness, bool)
        and np.isfinite(smoothness)
        and smoothness > _MIN_SMOOTHNESS,
        lambda: (
            f"'smoothness' must be finite and strictly greater than {_MIN_SMOOTHNESS}, "
            f"found {smoothness!r}"
        ),
    )
    return float(smoothness)


def _uid_market(
    ticker: str,
    d: date,
    smoothness: float,
    raw_day_filter: RawDayFilter,
    expiry_filter: ExpiryFilter,
    options_filter: OptionsFilter,
    mk_arb_free: MkArbFree,
) -> str:
    smoothness = _validated_smoothness(smoothness)
    filter_label = f"{raw_day_filter}, {expiry_filter}, {options_filter}, {mk_arb_free}"
    filter_uid = named_unique_filename48_8(
        filter_label,
        raw_day_filter,
        expiry_filter,
        options_filter,
        mk_arb_free,
    )
    smoothness_label = f"sm_{smoothness:.2f}"
    smoothness_uid = named_unique_filename48_8(smoothness_label, smoothness)
    return f"{filter_uid}/{smoothness_uid}/{ticker}/{d}"


@_cache(fmt=SubDir.PYTREE_HDF5).cache(
    "0.15",
    dependencies=[MkArbFree, OptionsFilter, RawDayFilter, ExpiryFilter, filter_options_day],
    uid=_uid_market,
)
def smooth_interpolated_market(
    ticker: str,
    d: date,
    *,
    smoothness: float = 0.25,
    options_filter: OptionsFilter = OptionsFilter(),
    raw_day_filter: RawDayFilter = RawDayFilter(),
    expiry_filter: ExpiryFilter = ExpiryFilter(),
    mk_arb_free: MkArbFree = MkArbFree(),
    message_context: Callable[[], str] | str | None = None,
    verbose: Context | None = Context.all,
) -> dict[str, object]:
    """Fit and serialize a smooth arbitrage-free market for one trading day."""
    smoothness = _validated_smoothness(smoothness)
    verify_inp(
        options_filter.normalize,
        lambda: "'options_filter.normalize' must be True to produce normalized market data",
    )

    verbose = Context.quiet if verbose is None else verbose

    def message(with_cols: bool = True) -> str:
        my_context = f"{ticker} {d}"
        if message_context is None:
            msg = my_context
        else:
            msg = str(message_context() if callable(message_context) else message_context)
            if msg != "":
                msg += " " + my_context
            else:
                msg = my_context
        msg = msg[0].upper() + msg[1:]
        return msg + ": " if with_cols else msg

    # load data, apply filters etc
    # ---------

    data = filter_options_day(
        ticker,
        d,
        mk_arb_free=mk_arb_free,
        options_filter=options_filter,
        raw_day_filter=raw_day_filter,
        expiry_filter=expiry_filter,
        verbose=verbose,
        raise_if_bad_day=False,
    )
    if data is None or len(data) == 0:
        return {}

    spot_ref = float(data["spot_ref"].unique().item())
    spot_ret = float(data["spot_ret"].unique().item())

    required_market_columns = {
        "df",
        "expiry",
        "expiry_id",
        "fwd",
        "is_am",
        "is_american",
        "mask_arb_free",
        "mkt_call_ask",
        "mkt_call_bid",
        "mkt_call_log_open_interest",
        "mkt_call_log_volume",
        "mkt_put_ask",
        "mkt_put_bid",
        "mkt_put_log_open_interest",
        "mkt_put_log_volume",
    }
    verify_inp(
        set(data.columns) >= required_market_columns,
        lambda: (
            "Filtered option data is missing columns required by the smooth market contract: "
            f"{sorted(required_market_columns - set(data.columns))}"
        ),
    )

    strikes = data["pure_strike"].to_numpy()
    ixs = np.argsort(strikes)
    strikes = strikes[ixs]
    tte_y = data["tte_y"].to_numpy()[ixs]
    bid = data["pure_bid_price"].to_numpy()[ixs]
    ask = data["pure_ask_price"].to_numpy()[ixs]

    # Preserve real call and put books in forward units. The WIDB data at this
    # point has prices and forwards divided by spot, so dividing by df*fwd is
    # exactly the dimensionless cash-price transformation C/(DF*F).
    discounted_forward = data["df"].to_numpy()[ixs] * data["fwd"].to_numpy()[ixs]
    verify_inp(
        np.all(np.isfinite(discounted_forward)) and np.all(discounted_forward > 0.0),
        lambda: "Expected every discounted forward to be finite and positive",
    )

    direct = {
        "expiry": data["expiry"].to_numpy()[ixs],
        "expiry_id": data["expiry_id"].to_numpy()[ixs],
        "is_am": data["is_am"].to_numpy()[ixs],
        "is_american": data["is_american"].to_numpy()[ixs],
        "model_mask": data["mask_arb_free"].to_numpy(dtype=bool)[ixs],
        "call_bid": data["mkt_call_bid"].to_numpy()[ixs] / discounted_forward,
        "call_ask": data["mkt_call_ask"].to_numpy()[ixs] / discounted_forward,
        "put_bid": data["mkt_put_bid"].to_numpy()[ixs] / discounted_forward,
        "put_ask": data["mkt_put_ask"].to_numpy()[ixs] / discounted_forward,
        "call_log_volume": data["mkt_call_log_volume"].to_numpy()[ixs],
        "call_log_open_interest": data["mkt_call_log_open_interest"].to_numpy()[ixs],
        "put_log_volume": data["mkt_put_log_volume"].to_numpy()[ixs],
        "put_log_open_interest": data["mkt_put_log_open_interest"].to_numpy()[ixs],
    }
    del discounted_forward, ixs

    # split into per-expiry
    # ---------------------

    market = PrettyObject()
    market.uid = unique_hash8(raw_day_filter, expiry_filter, options_filter, mk_arb_free)
    market.expiries = np.unique(tte_y)
    market.strikes  = []
    market.bids     = []
    market.asks     = []
    market.expiry_dates = []
    market.expiry_ids = []
    market.is_am = []
    market.is_american = []
    market.pre_af_masks = []
    market.model_masks = []
    market.call_bids = []
    market.call_asks = []
    market.put_bids = []
    market.put_asks = []
    market.call_log_volumes = []
    market.call_log_open_interests = []
    market.put_log_volumes = []
    market.put_log_open_interests = []
    assert len(market.expiries) > 0, (
        f"{message()}No expiries found after filtering for ticker {ticker} on date {d} "
        f"with raw_day_filter={raw_day_filter}, expiry_filter={expiry_filter}, "
        f"options_filter={options_filter} and mk_arb_free={mk_arb_free}. {data}"
    )

    for expiry in market.expiries:
        mask = tte_y == expiry
        market.strikes.append(strikes[mask])
        market.bids.append(bid[mask])
        market.asks.append(ask[mask])
        expiry_dates = np.unique(direct["expiry"][mask].astype("datetime64[D]"))
        expiry_ids = np.unique(direct["expiry_id"][mask])
        is_am = np.unique(direct["is_am"][mask])
        assert len(expiry_dates) == len(expiry_ids) == len(is_am) == 1, (
            f"{message()}Expected one contract expiry identity for tte_y={expiry}, found "
            f"dates={expiry_dates}, ids={expiry_ids}, is_am={is_am}"
        )
        market.expiry_dates.append(str(expiry_dates.item()))
        market.expiry_ids.append(int(expiry_ids.item()))
        market.is_am.append(bool(is_am.item()))
        market.is_american.append(direct["is_american"][mask])
        market.pre_af_masks.append(direct["model_mask"][mask])
        market.model_masks.append(direct["model_mask"][mask].copy())
        market.call_bids.append(direct["call_bid"][mask])
        market.call_asks.append(direct["call_ask"][mask])
        market.put_bids.append(direct["put_bid"][mask])
        market.put_asks.append(direct["put_ask"][mask])
        market.call_log_volumes.append(direct["call_log_volume"][mask])
        market.call_log_open_interests.append(direct["call_log_open_interest"][mask])
        market.put_log_volumes.append(direct["put_log_volume"][mask])
        market.put_log_open_interests.append(direct["put_log_open_interest"][mask])

    market.expiry_ids = np.asarray(market.expiry_ids, dtype=np.int64)
    market.is_am = np.asarray(market.is_am, dtype=bool)

    assert set(market) >= {"expiries", "strikes", "bids", "asks"}, (
        f"{message()}Expected market to have keys 'expiries', 'strikes', 'bids', "
        f"'asks' but found {set(market)}"
    )

    str_smth = f"{smoothness:.2f}-smooth"

    # fit a *big* smooth SANOS model to the market
    # --------------------------------------------

    num_strikes = sum(len(strikes_t) for strikes_t in market.strikes)
    with verbose.write_t(
        lambda: f"\r{message()}Fitting {str_smth} SANOS to {num_strikes} strikes... ",
        end="",
    ) as tme:
        sanos_fit = MkArbFree(smoothness=smoothness, surface_expiry_linear=True)
        af_pricer: ArbFreePricer = sanos_fit(
            strikes=market.strikes,
            expiries=market.expiries,
            bids=market.bids,
            asks=market.asks,
            verbose=verbose.quiet,
        )
        fit_result: PrettyObject = af_pricer.compute_market_fit(
            expiries=market.expiries,
            strikes=market.strikes,
            bids=market.bids,
            asks=market.asks,
            and_vols=False,
        )
        verbose.write(
            f"done; median error is {fit_result.median_err:.2f} half-spreads for the "
            f"{fit_result.pcnt_fit * 100:.1f}% of options which were not fitted within "
            f"bid-ask spread. This took {tme}.",
            head=False,
        )

    final_model_masks = [np.zeros_like(mask, dtype=bool) for mask in market.model_masks]
    for retained_index, input_index in enumerate(af_pricer.input_index):
        final_model_masks[int(input_index)] = (
            market.model_masks[int(input_index)] & af_pricer.masks[retained_index]
        )
    market.model_masks = final_model_masks

    market.af_call_mids = fit_result.fits
    market.af_put_mids = [
        call_mid - 1.0 + strikes_t
        for call_mid, strikes_t in zip(market.af_call_mids, market.strikes, strict=True)
    ]
    market.af_mid_available = [
        np.isfinite(call_mid) & np.isfinite(put_mid)
        for call_mid, put_mid in zip(
            market.af_call_mids,
            market.af_put_mids,
            strict=True,
        )
    ]
    market.call_active = []
    market.put_active = []
    for call_bid, call_ask, put_bid, put_ask, pre_af_mask in zip(
        market.call_bids,
        market.call_asks,
        market.put_bids,
        market.put_asks,
        market.pre_af_masks,
        strict=True,
    ):
        market.call_active.append(
            pre_af_mask
            & np.isfinite(call_bid)
            & np.isfinite(call_ask)
            & (call_bid >= 0.0)
            & (call_ask > call_bid)
        )
        market.put_active.append(
            pre_af_mask
            & np.isfinite(put_bid)
            & np.isfinite(put_ask)
            & (put_bid >= 0.0)
            & (put_ask > put_bid)
        )

    return {
        "schema_version": MARKET_SCHEMA_VERSION,
        "ticker": ticker,
        "d": d,
        "spot_ref": spot_ref,
        "spot_ret": spot_ret,
        "market": dict(market),
        "af_pricer": af_pricer.to_dict(),
        "fit_result": dict(fit_result),
    }
