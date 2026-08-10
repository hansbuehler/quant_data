"""
Yahoo finance
"""

from quantdata import env, SubDir, Context, version#NOQA
from quantdata.biz_calendar import to_date
from datetime import date
import yfinance as yf
import polars as pl
from cdxcore.util import fmt_date, fmt_list
from cdxcore.err import verify
import pandas as pd
from dataclasses import dataclass
import numpy as np
import time as time

_yfinance_locations = env.locations(
    "yfinance",
    fmt_raw=SubDir.PICKLE,
    fmt_processing=SubDir.PANDAS_PARQUET,
    fmt_training=SubDir.PYTREE_HDF5,
)
_cache = _yfinance_locations.raw("fetch", fmt=SubDir.PANDAS_PARQUET)
_cache_polars = _yfinance_locations.raw("fetch", fmt=SubDir.POLARS_PARQUET)


@dataclass(frozen=True)
class YFConnectionConfig:
    """ Configuration for managing yfinance/yaoo connectivity """
    max_retries : int = 5
    timeout     : float = 1.

    def __post_init__(self):
        verify( self.max_retries >= 0, "max_retries must be non-negative" )
        verify( self.timeout > 0., "timeout must be positive" )

class DataError(Exception):
    """ Exception raised when data retrieved from yahoo is inconsistent or insufficient. """
    pass

@version("0.1")
class YFClient(object):
    """
    Yahoo Finance Client
    """

    locations = _yfinance_locations

    def __init__(self):
        pass

    def validate_status(self):
        """ Check that I am not in test mode """
        verify( not env.test_mode, "Attempted to connect to Yahoo Finance in test mode. Set QUANTDATA_TEST_MODE=0 to enable connections." )

    NO_EARNINGS_TYPES = {
        "ETF",
        "MUTUALFUND",
        "INDEX",
        "CRYPTOCURRENCY",
        "CURRENCY",
        "FUTURE",
        "OPTION",
    }
    """ Instrument types which do not have earnings """

    SCHEMA_HAS_EARNINGS = {
        "date": pl.Date,
        "is_earnings": pl.Boolean,
    }
    """ Schema for the :class:`pandas.DataFrame` returned by :func:`has_earnings` """

    @_cache_polars.cache(
        "0.1",
        uid=lambda ticker, start, end : f"has_earnings/{ticker}/{fmt_date(start)}_{fmt_date(end)}",
        exclude_args=['self']) # 'self' is excluded fromm the arguments as it does not affect the output.
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

        if qt in self.NO_EARNINGS_TYPES:
            return pl.DataFrame(schema=self.SCHEMA_HAS_EARNINGS)

        earnings_dates = t.earnings_dates
        if earnings_dates is None or earnings_dates.empty:
            return pl.DataFrame(schema=self.SCHEMA_HAS_EARNINGS)
        earnings_dates = [dt.date() for dt in earnings_dates.index]
        earnings_dates = [dt for dt in earnings_dates if start <= dt <= end]
        earnings_dates = sorted(earnings_dates)
        df = pl.DataFrame(
            dict(
                date=pl.Series(earnings_dates, dtype=pl.Date),
                is_earnings=pl.Series([True] * len(earnings_dates), dtype=pl.Boolean),
            ),
            infer_schema_length=None
        )
        df = df.sort("date")
        assert set(df.columns) == set(self.SCHEMA_HAS_EARNINGS), f"Internal error retrieving earnings for {ticker} between {fmt_date(start)} and {fmt_date(end)}: expected columns {set(self.SCHEMA_HAS_EARNINGS)} but got {set(df.columns)}. Please update code. Ref: https://massive.com/docs/rest/stocks/corporate-actions/earnings"
        return df

    PRICE_FIELDS = {'Close', 'High', 'Low', 'Open', 'Volume', 'Dividends', 'Stock Splits'}
    """ Fields returned by :func:`prices` and expected to be present in the data frame returned by yahoo. """

    @_cache(fmt=SubDir.PANDAS_PARQUET).cache(
        "0.1",
        uid=lambda ticker, start, end: f"bins/{ticker}/{fmt_date(start)}_{fmt_date(end)}",
        exclude_args=['self']) # 'self' is excluded fromm the arguments as it does not affect the output.
    def prices( self, ticker: str, start: date, end: date, *,
                yfconnection_config : YFConnectionConfig = YFConnectionConfig() ) -> pd.DataFrame:
        """
        Downloads one ticker from yahoo using `yfinance <https://ranaroussi.github.io/yfinance/index.html>`__
        and applies some basic consistency checks.

        The function returns a pandas data frame with the fields ``Close``, ``High``, ``Low``, ``Open``, ``Volume``, ``Dividends`` and ``Stock Splits``,
        see also :attr:`PRICE_FIELDS`. The index of the data frame are the historic dates. All rows with NaNs are removed.
        
        Checks performed:

        * All required fields are present.
        * Close, Low, High, Open are positive
        * High > Low (please note yahoo data does not include Open/Close in Low/High)
                
        Parameters
        ----------
        ticker : str
            Ticker such as ``AAPL``.
            
        start : :class:`datetime.date`
            Start of the period.
        
        end : :class:`datetime.date`
            End of the period.

        yfconnection_config : :class:`YFConnectionConfig`, default ``YFConnectionConfig()``
            Configuration for managing yfinance/yaoo connectivity, such as timeouts and retry counts.

        Returns
        -------
        Result : :class:`pandas.DataFrame`
            Non-empty dataframe with the data from yahoo.
            Its index are the historic dates.

        Raises
        ------
        Inconsistent data returned : :class:`AttributeError`
            Raised if the data returned from yahoo did not contain the expected column headers.
            
        Timeout : :class:`TimeoutError`
            Raised if `yf.download <https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html#yfinance.download>`__
            timed out.

        Data error: :class:`DataError`
            Raised if data validation fails.
        """

        start = to_date(start)
        end = to_date(end)
        
        timeout_cnt = 0
        time_sleep = yfconnection_config.timeout
        while True:
            try:
                df = yf.download( ticker, start=start, end=end, auto_adjust=True, progress=False, actions=True  )
                break
            except yf.exceptions.YFRateLimitError as e:
                raise TimeoutError(f"{ticker} yfinance timeout error: {e}. Retry later")
            except TimeoutError as e:
                if timeout_cnt > yfconnection_config.max_retries:
                    raise TimeoutError(f"{ticker} yfinance timeout error: {e}. Retry later")
                time.sleep(time_sleep)
                time_sleep *= 2
                timeout_cnt += 1
        if len(df) == 0:
            raise DataError(f"{ticker} yahoo returned no data", ticker=ticker, start=start, end=end)

        try:
            df = df.xs(ticker, level=1, axis=1)
        except KeyError as e:
            raise KeyError(f"Cannot find ticker '{ticker}' ... in data set for '{ticker}' ? DF length: {len(df)}, tickers: {df.columns}") from e
            
        if set(df.columns) < self.PRICE_FIELDS:
            raise AttributeError(ticker,f"{ticker} is missing column headers; only found {list(df.columns)}; check code base consistency."+\
                                        f"Fields found: {fmt_list(sorted(df.columns))}; fields expected: {fmt_list(sorted(self.PRICE_FIELDS))}")
        
        df = df.dropna()
        if df.empty:
            raise DataError(f"{ticker} no data left after dropping NaN's", ticker=ticker, start=start, end=end)
        if np.min( df['Close'] ) <= 0.:
            raise DataError(f"{ticker}: Closing prices of zero found", ticker=ticker, start=start, end=end)
        if np.min( df['Open'] ) <= 0.:
            raise DataError(f"{ticker}: Open prices of zero found", ticker=ticker, start=start, end=end)
        if np.min( df['High'] - df['Low'] ) < 0.:
            ix = df['High'].values - df['Low'].values < 0.
            raise DataError(f"{ticker}: High and Low are out of order {len(ix)}; High: {df['High'].values[ix][:4]}, Low: {df['Low'].values[ix][:4]}", ticker=ticker, start=start, end=end)
        if np.min( df['Low'] ) <= 0.:
            raise DataError(f"{ticker}: Low prices of zero found", ticker=ticker, start=start, end=end)

        df.index = df.index.values.astype('datetime64[D]')
        df = df.astype(np.float64, copy=False)
        return df
        