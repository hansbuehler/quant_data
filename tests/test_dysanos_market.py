from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cdxcore.bs import bs
from cdxcore.subdir import SubDir
from cdxcore.verbose import Context

from quantdata.dysanos import market as dysanos_market
from quantdata.dysanos.tangent_levy.config import MarketConfig
from quantdata.dysanos.tangent_levy.markets import market_day_from_payload
from quantdata.sanos.arbfree import ArbFreePricer


@pytest.mark.parametrize(
    "smoothness",
    [None, -0.1, 0.0, 0.01, np.nan, np.inf],
)
def test_smooth_interpolated_market_requires_positive_smoothing(
    smoothness: float | None,
) -> None:
    with pytest.raises(ValueError, match="strictly greater than 0.01"):
        dysanos_market.smooth_interpolated_market(
            "TEST",
            date(2025, 6, 3),
            smoothness=smoothness,
            verbose=Context.quiet,
        )


def _one_day_market() -> pd.DataFrame:
    expiries = np.array([0.1, 0.25], dtype=np.float64)
    strikes = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2], dtype=np.float64)
    expiry_grid, strike_grid = np.meshgrid(expiries, strikes, indexing="ij")
    mids = bs.price(k=strike_grid, vol=0.3, sqrtT=np.sqrt(expiry_grid))
    put_mids = mids - 1.0 + strike_grid
    discounted_forward = 0.99 * 1.01
    real_half_spread = 0.0001
    expiry_dates = np.repeat(
        np.array(["2025-06-13", "2025-06-20"], dtype="datetime64[ms]"),
        len(strikes),
    )
    expiry_ids = np.repeat(np.array([202506131, 202506201], dtype=np.int32), len(strikes))

    return pd.DataFrame(
        {
            "spot_ref": 100.0,
            "spot_ret": 0.01,
            "pure_strike": strike_grid.ravel(),
            "tte_y": expiry_grid.ravel(),
            "pure_bid_price": (mids - 0.001).ravel(),
            "pure_ask_price": (mids + 0.001).ravel(),
            "df": 0.99,
            "fwd": 1.01,
            "expiry": expiry_dates,
            "expiry_id": expiry_ids,
            "is_am": False,
            "is_american": False,
            "mask_arb_free": True,
            "mkt_call_bid": (
                np.maximum(mids - real_half_spread, 0.0) * discounted_forward
            ).ravel(),
            "mkt_call_ask": ((mids + real_half_spread) * discounted_forward).ravel(),
            "mkt_put_bid": (
                np.maximum(put_mids - real_half_spread, 0.0) * discounted_forward
            ).ravel(),
            "mkt_put_ask": ((put_mids + real_half_spread) * discounted_forward).ravel(),
            "mkt_call_log_volume": 2.0,
            "mkt_call_log_open_interest": 3.0,
            "mkt_put_log_volume": 2.5,
            "mkt_put_log_open_interest": 3.5,
        }
    )


def test_smooth_interpolated_market_one_day_reconstructs_arb_free_pricer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    ticker = "TEST"
    market_date = date(2025, 6, 3)
    day = _one_day_market()

    def load_day(requested_ticker, requested_date, **kwargs):
        assert requested_ticker == ticker
        assert requested_date == market_date
        assert kwargs["raise_if_bad_day"] is False
        return day

    monkeypatch.setattr(dysanos_market, "filter_options_day", load_day)

    result = dysanos_market.smooth_interpolated_market.__wrapped__(
        ticker,
        market_date,
        verbose=Context.quiet,
    )

    assert result["ticker"] == ticker
    assert result["schema_version"] == 4
    assert result["d"] == market_date
    assert result["spot_ref"] == 100.0
    assert result["spot_ret"] == 0.01

    cache = SubDir(tmp_path, fmt=SubDir.PYTREE_HDF5)
    assert cache.write("market", result)
    cached_result = cache.read("market", raise_on_error=True)

    market = cached_result["market"]
    restored_pricer = ArbFreePricer.from_dict(cached_result["af_pricer"])
    assert isinstance(restored_pricer, ArbFreePricer)
    np.testing.assert_array_equal(restored_pricer.expiries, market["expiries"])

    for expiry, strikes, expected_prices in zip(
        market["expiries"],
        market["strikes"],
        cached_result["fit_result"]["fits"],
        strict=True,
    ):
        restored_prices = restored_pricer.price(expiry, strikes)
        np.testing.assert_allclose(restored_prices, expected_prices, rtol=1e-12, atol=1e-12)

    assert market["expiry_dates"] == ["2025-06-13", "2025-06-20"]
    for index, strikes in enumerate(market["strikes"]):
        call_mid = 0.5 * (market["call_bids"][index] + market["call_asks"][index])
        put_mid = 0.5 * (market["put_bids"][index] + market["put_asks"][index])
        np.testing.assert_allclose(call_mid - put_mid, 1.0 - strikes, atol=1.0e-4)
        np.testing.assert_array_equal(market["call_active"][index], True)
        np.testing.assert_array_equal(market["put_active"][index], True)
        np.testing.assert_array_equal(market["pre_af_masks"][index], True)
        np.testing.assert_allclose(
            market["af_put_mids"][index],
            market["af_call_mids"][index] - 1.0 + strikes,
        )
        np.testing.assert_array_equal(market["af_mid_available"][index], True)

    # The Tangent Lévy adapter consumes the canonical serialized payload and
    # independently applies its European-only contract to executable masks.
    result["market"]["is_american"][0][0] = True
    adapted = market_day_from_payload(result, config=MarketConfig())
    assert adapted is not None
    assert adapted.expiries.shape == (2,)
    assert not adapted.model_masks[0][0]
    assert not adapted.call_active[0][0]
    assert not adapted.put_active[0][0]
    assert adapted.audit.excluded_american_quotes == 1
