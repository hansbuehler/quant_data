"""
Yahoo finance
"""

from quantdata import env, SubDir, Context, version#NOQA

from datetime import date
import yfinance as yf
import polars as pl
from cdxcore.util import fmt_date
from cdxcore.err import verify

_yfinance_locations = env.locations("yfinance", fmt_raw=SubDir.PICKLE, fmt_processing=SubDir.PANDAS_PARQUET, fmt_training=SubDir.PYTREE_HDF5)
_cache = _yfinance_locations.raw("fetch", fmt=SubDir.PANDAS_PARQUET)

@version("0.1")
class YFClient(object):
    """
    Yahoo Finance Client
    """

    def __init__(self):
        pass

    def validate_status(self):
        """ Check that I am not in test mode """
        verify( not env.test_mode, "Attempted to connect to Yahoo Finance in test mode. Set QUANTDATA_TEST_MODE=0 to enable connections." )

    NO_EARNINGS_QUOTE_TYPES = {
        "ETF",
        "MUTUALFUND",
        "INDEX",
        "CRYPTOCURRENCY",
        "CURRENCY",
        "FUTURE",
        "OPTION",
    }

    SCHEMA_EARNINGS = {
        "date": pl.Date,
        "is_earnings": pl.Boolean,
    }

    @_cache(fmt=SubDir.PANDAS_PARQUET).cache(
        "0.1",
        uid=lambda ticker, start, end : f"has_earnings/{ticker}/{fmt_date(start)}_{fmt_date(end)}",
        exclude_args=['self']) # excluding 'self' is not necessary; should be excluded automatically
    def has_earnings(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        """Use yfinance to get earnings dates and ex-dividend dates for a ticker."""
        self.validate_status()
        t = yf.Ticker(ticker)

        # find quote
        qt = None
        try:
            qt = (t.fast_info or {}).get("quoteType")
        except Exception:
            qt = (t.info or {}).get("quoteType", None)
        qt = str(qt).upper() if qt is not None else None

        if qt in self.NO_EARNINGS_QUOTE_TYPES:
            return pl.DataFrame(schema=self.SCHEMA_EARNINGS)

        earnings_dates = t.earnings_dates
        if earnings_dates is None or earnings_dates.empty:
            return pl.DataFrame(schema=self.SCHEMA_EARNINGS)
        earnings_dates = [dt.date() for dt in earnings_dates.index]
        earnings_dates = [dt for dt in earnings_dates if start <= dt <= end]
        earnings_dates = sorted(earnings_dates)
        df = pl.DataFrame(
            dict(
                date=pl.Series(earnings_dates, dtype=pl.Date),
                is_earnings=pl.Series([True] * len(earnings_dates), dtype=pl.Boolean),
            )
        )
        df = df.sort("date")
        assert set(df.columns) == set(self.SCHEMA_EARNINGS), f"Internal error retrieving earnings for {ticker} between {fmt_date(start)} and {fmt_date(end)}: expected columns {set(self.SCHEMA_EARNINGS)} but got {set(df.columns)}. Please update code. Ref: https://massive.com/docs/rest/stocks/corporate-actions/earnings"
        return df
