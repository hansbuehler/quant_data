"""
quantdata.massive.client
^^^^^^^^^^^^^^^^^^^^^^^^

A simple client wrapper for the `www.massive.com <https://www.massive.com>`_ API.

Access
======

You will require a `massive.com token <https://www.massive.com/docs/rest/authentication>`_ to access the data.
Set the token in the ``MASSIVE_TOKEN`` environment variable, e.g. by adding the following line to your ``~/.bashrc`` or ``~/.zshrc`` file:

.. code-block:: bash

    export MASSIVE_TOKEN="your_token"

Once installed, use is straightforward, e.g. to fetch trades and quotes for Apple 150 call expiring on 2023-06-16::

    from quantdata.massive import MassiveClient
    client = MassiveClient()

    trades = client.fetch_trades("AAPL230616C00150000", date(2023,6,16))
    quotes = client.fetch_quotes("AAPL230616C00150000", date(2023,6,16))
    ...
"""

from __future__ import annotations

import os as os
import time as pytime
import warnings as warnings
from datetime import date

import polars as pl
import requests as requests
from cdxcore.err import verify
from cdxcore.subdir import SubDir, version
from cdxcore.util import fmt_date, verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from massive import RESTClient
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from quantdata import env, Locations
from quantdata.biz_calendar import TZ_NY, add_business_days
from quantdata.massive.trade_conds import pack_conditions

_massive_locations = env.locations(
    "massive",
    fmt_raw=SubDir.PICKLE,
    fmt_processing=SubDir.PANDAS_PARQUET,
    fmt_training=SubDir.PYTREE_HDF5,
)
_cache = _massive_locations.raw

# --------------------------------------------------------------------
# Low-level HTTP helpers
# --------------------------------------------------------------------

@version("0.1")
class MassiveClient(object):
    """
    Wrapper around massive.com connectivity.

    Standard usage::

        from quantdata.massive import MassiveClient
        client = MassiveClient()

        trades = client.fetch_trades("AAPL230616C00150000", date(2023,6,16))
        quotes = client.fetch_quotes("AAPL230616C00150000", date(2023,6,16))

    **Temporary Caching**

    This client supports in-session caching. Unlike the overall caching framework for :mod:`quantdata`, this caching
    scheme is temporary and is not guaranteed to persist between machine reboots (e.g. in Linux the root temporary directory is ``/tmp/massive``).
    This can be useful for speeding up repeated requests during an active session
    but should not be relied upon for long-term storage of data.

    To enable temporary caching in a, say, ``"work"`` context::

        handle = MassiveClient("work")
    """

    locations = _massive_locations
    """ :class:`quantdata.Locations` for massive data. """

    _MASSIVE_TOKEN = None
    _HTTP_SESSION = None

    BASE_URL = "https://api.massive.com"  #: Base URL for requests.
    START_DATE = date(
        2022, 3, 7
    )
    """ Start date for quote and trade requests for options. `See massive.com <https://massive.com/docs/rest/options/trades-quotes/quotes>`_ """

    TIME_SPANS = {
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "quarter",
        "year",
    }
    """ Standard `massive time spans <https://massive.com/docs/rest/stocks/aggregates/custom-bars>`_ """

    _PREV_CLOSE_CACHE = {}

    """
    Static members
    """

    @staticmethod
    def get_token():
        """Reads massive token from the ``MASSIVE_TOKEN`` environment variable"""
        if MassiveClient._MASSIVE_TOKEN is None:
            verify( not env.test_mode, "Attempted to connect to massive.com in test mode. Set QUANTDATA_TEST_MODE=0 to enable connections." )
            MassiveClient._MASSIVE_TOKEN = os.environ.get("MASSIVE_TOKEN", None)
            verify(
                not MassiveClient._MASSIVE_TOKEN is None,
                "Must specify environment variable 'MASSIVE_TOKEN' to access `massive.com <https://www.massive.com>`_ data",
                exception=RuntimeError,
            )
        return MassiveClient._MASSIVE_TOKEN

    @staticmethod
    def get_rest_client():
        """
        Returns a :class:`massive.RESTClient` instance with the current token, defined by the ``MASSIVE_TOKEN`` environment variable.
        """
        return RESTClient(MassiveClient.get_token())

    @staticmethod
    def get_session():
        """
        Return an HTTP session per process.
        """
        if MassiveClient._HTTP_SESSION is None:
            MassiveClient._HTTP_SESSION = requests.Session()
            MassiveClient._HTTP_SESSION.mount(
                "https://",
                HTTPAdapter(
                    pool_connections=50,
                    pool_maxsize=50,
                    max_retries=Retry(
                        total=5, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504]
                    ),
                ),
            )
            MassiveClient._HTTP_SESSION.mount(
                "http://",
                HTTPAdapter(
                    pool_connections=50,
                    pool_maxsize=50,
                    max_retries=Retry(
                        total=5, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504]
                    ),
                ),
            )
        return MassiveClient._HTTP_SESSION

    """
    Object
    """

    def __init__(self, temp_caching_topic: str | None = None, fmt: SubDir.Format = SubDir.PICKLE):
        """Create client with temporary caching topic"""
        self._session = None
        self._temp_dir = (
            MassiveClient.locations.temp("massive_api")(temp_caching_topic, fmt=fmt)
            if not temp_caching_topic is None
            else None
        )
        self.get = (
            self._temp_dir.cache(label="massive_get", exclude_args=["self"])(self._get)
            if not self._temp_dir is None
            else self._get
        )

    @version("0.1")
    def _get(self, url: str, params: dict | None = None) -> dict:
        """`
        GET wrapper that injects apiKey and raises for HTTP errors.
        Do not call this function, call ``self.get`` which is optionally
        cached in a temporary directory for speed
        """
        MASSIVE_token = MassiveClient.get_token()
        if params is None:
            params = {"apiKey": MASSIVE_token}
        else:
            params = {**params, "apiKey": MASSIVE_token}
        cnt = 1
        while True:
            e_ = None
            try:
                r = self.session.get(url, params=params, timeout=60 * cnt)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RetryError as e:
                if cnt > 5:
                    raise e
                e_ = e
            except requests.exceptions.ReadTimeout as e:
                if cnt > 5:
                    raise e
                e_ = e
            except requests.exceptions.ConnectTimeout as e:
                if cnt > 5:
                    raise e
                e_ = e
            except requests.exceptions.JSONDecodeError as e:
                if cnt > 5:
                    raise e
                e_ = e
            except requests.exceptions.ChunkedEncodingError as e:
                if cnt > 5:
                    raise e
                e_ = e
            except requests.exceptions.ConnectionError as e:
                if cnt > 5:
                    raise e
                e_ = e
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                # transient server-side error → retry
                if status is None or not (500 <= status < 600):
                    raise e
                e_ = e
            warnings.warn(f"**** Massive error: {e_} ...sleeping for {cnt} seconds ***")
            pytime.sleep(cnt)
            cnt += 1

    @version("0.1", dependencies=[_get])
    def __call__(self, url: str, params: dict | None = None) -> dict:
        """Short-cut for ``self.get``"""
        return self.get(url=url, params=params)

    @version("0.1", dependencies=[_get])
    def paginate(self, initial_url: str, params: dict | None = None) -> list[dict]:
        """
        Handle `v3-style pagination <https://massive.com/blog/api-pagination-patterns>`_ (quotes, trades, contracts), c.f.
        """
        results: list[dict] = []
        data = self.get(initial_url, params or {})
        results.extend(data.get("results", []) or [])
        next_url = data.get("next_url")

        while next_url:
            if next_url and next_url.startswith("/"):
                next_url = self.BASE_URL + next_url
            data = self.get(next_url, None)
            results.extend(data.get("results", []) or [])
            next_url = data.get("next_url")
            pytime.sleep(0.02)  # be gentle

        return results

    @property
    def session(self):
        """Return the HTTP session for this MassiveClient. Note that a session is shared within a process"""
        if self._session is None:
            self._session = MassiveClient.get_session()
        return self._session

    def clear_temp(self):
        """Clear the temporary directory for this MassiveClient."""
        if self._temp_dir is not None:
            self._temp_dir.delete_all_content(delete_self=True)

    # =============================================================================
    # Polygon fetch: v3 trades / quotes (SIP). Keep ts_ns in ns.
    # =============================================================================

    @_cache(fmt=SubDir.PICKLE).cache(
        "0.1",
        label=lambda ticker, d: f"has_any_quotes_or_trades/{ticker}/{fmt_date(d)}",
        exclude_args=["self"],
    )
    def has_any_quotes_or_trades(self, ticker: str, d: date) -> tuple[bool, bool]:
        """Whether ``ticker`` has any quotes or trades on date ``d``. Returns tuple (has_quotes, has_trades)."""
        url = f"{MassiveClient.BASE_URL}/v3/quotes/{ticker}"
        response = self.get(
            url,
            params={
                "timestamp": d.strftime("%Y-%m-%d"),  # YYYY-MM-DD
                "limit": 1,
                "order": "asc",
                "sort": "timestamp",
            },
        )
        has_quotes = len(response["results"]) > 0

        url = f"{MassiveClient.BASE_URL}/v3/trades/{ticker}"
        response = self.get(
            url,
            params={
                "timestamp": d.strftime("%Y-%m-%d"),  # YYYY-MM-DD
                "limit": 1,
                "order": "asc",
                "sort": "timestamp",
            },
        )

        has_trades = len(response["results"]) > 0
        return has_quotes, has_trades

    def has_any_quotes(self, ticker: str, d: date) -> bool:
        """Whether ``ticker`` has any quotes on date ``d``."""
        return self.has_any_quotes_or_trades(ticker, d)[0]

    def has_any_trades(self, ticker: str, d: date) -> bool:
        """Whether ``ticker`` has any trades on date ``d``."""
        return self.has_any_quotes_or_trades(ticker, d)[1]

    SCHEMA_TRADES = {
        "ts_ns": pl.Datetime("ns", TZ_NY),
        "price": pl.Float64,
        "size": pl.Float64,
        "trade_code": pl.UInt32,
        "sequence_number": pl.UInt64,
    }
    """ Schema of the :class:`polars.DataFrame` returned by :meth:`fetch_trades`. """

    @_cache(fmt=SubDir.POLARS_PARQUET).cache(
        "0.11", label=lambda ticker, d: f"trades/{ticker}/{fmt_date(d)}", exclude_args=["self"]
    )
    def fetch_trades(self, ticker: str, d: date) -> pl.DataFrame:
        """
        Gets trades from /v3/trades/{ticker} sorted by sip_timestamp asc.
        Used both for options and underlyings.

        The returned :class:`polars.DataFrame` has the following columns:

        * ts_ns: Datetime[ns, tz]: time zone information with nanonsecond resolutuin.
        * price: Float64
        * size: Float64
        * trade_code: UInt32. Compressed & Packed trade conditions, see :func:`quantdata.massive.trade_conds.pack_conditions`.
          It is recommended to remove trades with uncommon conditions, see :attr:`quantdata.massive.trade_conds.NORMAL_TRADE_EXCLUDE_FLAGS`.
        * sequence_number: UInt64. Sequence number of the trade for a given timestamp, useful for ordering trades with the same timestamp.

        Parameters
        ----------
        ticker : str
            Ticker symbol, e.g. "AAPL230616C00150000" for Apple 150 call expiring on 2023-06-16.
        d : date
            Date for which to fetch trades.

        Returns
        -------
        Result : :class:`polars.DataFrame`
            Returns a data frame with schema :attr:`MassiveClient.SCHEMA_TRADES`.
            The dataframe can have length zero (but it will have the correct data schema).

            * ts_ns: Datetime[ns, tz]
            * price: Float64
            * size: Float64
            * trade_code: UInt32. Compressed & Packed trade conditions, see :func:`quantdata.massive.trade_conds.pack_conditions`.
            * sequence_number: UInt64. Sequence number of the trade for a given timestamp. Trades will be sorted by (ts_ns, sequence_number) asc.
        """
        url = f"{self.BASE_URL}/v3/trades/{ticker}"
        params = {
            "timestamp": d.strftime("%Y-%m-%d"),
            "order": "asc",
            "limit": 50000,
            "sort": "sip_timestamp",
        }
        rows = self.paginate(url, params)
        rows = [
            dict(
                price=r["price"],
                size=r["size"],
                sip_timestamp=r["sip_timestamp"],
                trade_code=pack_conditions(r.get("conditions", [])),
                sequence_number=r["sequence_number"],
            )
            for r in rows
        ]

        if not rows:
            return pl.DataFrame(schema=MassiveClient.SCHEMA_TRADES)

        df = pl.DataFrame(rows, infer_schema_length=None)
        out = df.select(
            pl.from_epoch(pl.col("sip_timestamp").cast(pl.Int64), time_unit="ns")
            .dt.convert_time_zone(TZ_NY.key)
            .alias("ts_ns"),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64),
            pl.col("trade_code").cast(pl.UInt32),
            pl.col("sequence_number").cast(pl.UInt64)
        ).sort("ts_ns", "sequence_number")
        assert set(out.columns) == set(MassiveClient.SCHEMA_TRADES), (
            f"Internal error: expected columns {set(MassiveClient.SCHEMA_TRADES)} but got {set(out.columns)}. Sanity check the request and contact the authors if you need help."
        )
        return out

    SCHEMA_QUOTES = {
        "ts_ns": pl.Datetime("ns", TZ_NY),
        "bid_price": pl.Float64,
        "bid_size": pl.Float64,
        "ask_price": pl.Float64,
        "ask_size": pl.Float64,
        "sequence_number": pl.UInt64,
    }
    """ Schema of the :class:`polars.DataFrame` returned by :meth:`fetch_quotes`. """

    @_cache(fmt=SubDir.POLARS_PARQUET).cache(
        "0.11", label=lambda ticker, d: f"quotes/{ticker}/{fmt_date(d)}", exclude_args=["self"]
    )
    def fetch_quotes(self, ticker: str, d: date) -> pl.DataFrame:
        """
        Fetches quotes from "/v3/quotes/{ticker}" sorted by sip_timestamp (and sequence_number) asc.
        Used both for options and underlyings.

        Parameters
        ----------
        ticker : str
            Ticker symbol, e.g. "AAPL230616C00150000" for Apple 150 call expiring on 2023-06-16.
        d : date
            Date for which to fetch quotes.

        Returns
        -------
        Result : :class:`polars.DataFrame`
            Returns a data frame with schema :attr:`MassiveClient.SCHEMA_QUOTES`.
            The dataframe can have length zero (but it will have the correct data schema).

            * ts_ns: Datetime[ns, tz]
            * bid_price: Float64
            * bid_size: Float64
            * ask_price: Float64
            * ask_size: Float64
            * sequence_number: UInt64. Sequence number of the quote for a given timestamp. Quotes will be sorted by (ts_ns, sequence_number) asc.
        """
        url = f"{self.BASE_URL}/v3/quotes/{ticker}"
        params = {
            "timestamp": d.strftime("%Y-%m-%d"),
            "order": "asc",
            "limit": 50000,
            "sort": "sip_timestamp",
        }
        rows = self.paginate(url, params)
        if not rows:
            return pl.DataFrame(schema=MassiveClient.SCHEMA_QUOTES)

        df = pl.DataFrame(rows, infer_schema_length=None)
        out = df.select(
            pl.from_epoch(pl.col("sip_timestamp").cast(pl.Int64), time_unit="ns")
            .dt.convert_time_zone(TZ_NY.key)
            .alias("ts_ns"),
            pl.col("bid_price").cast(pl.Float64),
            pl.col("bid_size").cast(pl.Float64),
            pl.col("ask_price").cast(pl.Float64),
            pl.col("ask_size").cast(pl.Float64),#
            pl.col("sequence_number").cast(pl.UInt64)
        ).sort("ts_ns", "sequence_number")
        assert set(out.columns) == set(MassiveClient.SCHEMA_QUOTES), (
            f"Internal error: expected columns {set(MassiveClient.SCHEMA_QUOTES)} but got {set(out.columns)}. Sanity check the request and contact the authors if you need help."
        )
        return out

    SCHEMA_BARS = {
        "ts_ms": pl.Datetime("ms", TZ_NY),
        "close": pl.Float64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "volume": pl.Float64,
        "vwap": pl.Float64,
        "num_trades": pl.Int64,
    }
    """ Schema of the :class:`polars.DataFrame` returned by :meth:`fetch_bars`. """

    @_cache(fmt=SubDir.POLARS_PARQUET).cache(
        "0.1",
        uid=lambda ticker, start, end, adjusted, timespan: (
            f"bars/{timespan}/{'adj' if adjusted else 'raw'}/{ticker}/{fmt_date(start)}_{fmt_date(end)}_{timespan}"
        ),
        exclude_args=["self"],
    )  # 'self' is excluded fromm the arguments as it does not affect the output.
    def fetch_bars(
        self, ticker: str, start: date, end: date, adjusted: bool, timespan: str
    ) -> pl.DataFrame:
        """
        Download daily bars for a ticker accross a number of dates; convert timestamp to datetime in NY/ns.
        See `custom bars documentation<https://massive.com/docs/rest/stocks/aggregates/custom-bars>`_

        Parameters
        ----------
        ticker : str
            Ticker symbol, e.g. "AAPL" for Apple.
        start : :class:`datetime.date`
            Start date for bars.
        end : :class:`datetime.date`
            End date for bars.
        adjusted : bool
            Whether to fetch adjusted bars.
        timespan : str
            Timespan for the bars, e.g. "day", "week", "month".
            Must be one of 'second', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'.

        Returns
        -------
        Result : :class:`polars.DataFrame`
            Data frame wiwh schema :attr:`MassiveClient.SCHEMA_BARS`. The dataframe can have length zero (but it will have the correct data schema).

            - ts_ms: Datetime[ms, tz]: time zone information with millisecond precision
            - close: Float64
            - open: Float64
            - high: Float64
            - low: Float64
            - volume: Float64
            - vwap: Float64
            - num_trades: Int64
        """
        if timespan not in self.TIME_SPANS:
            raise ValueError(
                f"Cannot retrieve bars for {ticker} between {start} and {end}: invalid timespan '{timespan}'. Must be one of {sorted(self.TIME_SPANS)}"
            )
        if start > end:
            raise ValueError(
                f"Cannot retrieve bars for {ticker} between {start} and {end}: start date must be before end date."
            )

        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
        rows = self.paginate(
            url,
            params={"adjusted": "false" if not adjusted else "true", "limit": 50000, "sort": "asc"},
        )

        if not rows:
            return pl.DataFrame(schema=MassiveClient.SCHEMA_BARS)

        df = pl.DataFrame(rows, infer_schema_length=None)
        out = df.select(
            pl.from_epoch(pl.col("t").cast(pl.Int64), time_unit="ms")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone(TZ_NY.key)
            .alias("ts_ms"),
            pl.col("c").cast(pl.Float64).alias("close"),
            pl.col("o").cast(pl.Float64).alias("open"),
            pl.col("h").cast(pl.Float64).alias("high"),
            pl.col("l").cast(pl.Float64).alias("low"),
            pl.col("v").cast(pl.Float64).alias("volume"),
            pl.col("vw").cast(pl.Float64).alias("vwap"),
            pl.col("n").cast(pl.Int64).alias("num_trades"),
        ).sort("ts_ms")
        assert set(out.columns) == set(MassiveClient.SCHEMA_BARS), (
            f"Internal error: expected columns {set(MassiveClient.SCHEMA_BARS)} but got {set(out.columns)}. Sanity check the request and contact the authors if you need help."
        )
        return out

    @_cache(fmt=SubDir.PICKLE).cache(
        "0.1",
        label=lambda ticker, d, adjusted, can_be_none: (
            f"open-close/{ticker}/{fmt_date(d)} {'a1' if adjusted else 'a0'} {'c0' if can_be_none else 'c1'}"
        ),
        exclude_args=["self"],
    )
    def fetch_day_summary(
        self, ticker: str, d: date, *, adjusted: bool, can_be_none: bool = False
    ) -> dict | None:
        """
        Fetch open/close/volume/high/low for ``ticker`` on date ``d``.
        Returns dict with keys: open, close, low, high, volume, adjusted and optionally afterHours and preMarket if available.

        See `daily ticker summary documentation <https://massive.com/docs/rest/options/aggregates/daily-ticker-summary>`_.

        If no trade was recorded raise an exception except of ``can_be_none`` is True,
        in which case return ``None``.

        Parameters
        ----------
        ticker : str
            Ticker symbol, e.g. "AAPL230616C00150000" for Apple 150 call expiring on 2023-06-16.
        d : date
            Date for which to fetch open/close/volume.
        adjusted : bool
            Whether to fetch adjusted or unadjusted open/close/volume.
        can_be_none : bool, default False
            If True, return None if no trade was recorded for the day. If False, raise an exception if no trade was recorded for the day.

        Returns
        -------
        Result : dict
            Dict with keys:

            * ``open [float]``
            * ``close [float]``
            * ``low [float]``
            * ``high [float]``
            * ``volume [float]``
            * ``adjusted [bool]``
            * ``afterHours [float|None]`` (optional)
            * ``preMarket [float|None]`` (optional)
        """
        url = f"{self.BASE_URL}/v1/open-close/{ticker}/{d:%Y-%m-%d}"
        try:
            data = self.get(url, {"adjusted": "false" if not adjusted else "true"})
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404 and can_be_none:
                return None
            raise requests.exceptions.HTTPError(
                f"No open/close/volume for {ticker} on {d} url {url}: {e.response.text}",
                response=e.response,
                request=e.request,
            ) from e

        # Polygon open-close has "open","close","volume" (and more)
        assert not "next_url" in data, (
            f"Internal error: pagination not expected for open-close endpoint, url '{url}' response '{data}'. Sanity check request & contact the authors."
        )
        open = data.get("open")
        close = data.get("close")
        volume = data.get("volume")
        high = data.get("high")
        low = data.get("low")
        has_data = (
            open is not None
            and close is not None
            and volume is not None
            and high is not None
            and low is not None
        )
        if not has_data:
            if can_be_none:
                return None
            raise RuntimeError(
                f"Cannot get {'adjusted' if adjusted else 'unadjusted'} open/close/volume/high/low for {ticker} on {d}. Massive URL used: '{url}'."
            )

        def get_optional(item, type=float):
            x = data.get(item, None)
            return x if x is None else type(x)

        return {
            "open": float(open),
            "close": float(close),
            "volume": float(volume),
            "high": float(high),
            "low": float(low),
            "afterHours": get_optional("afterHours"),
            "preMarket": get_optional("preMarket"),
            "adjusted": adjusted,
        }

    @_cache(fmt=SubDir.PICKLE).cache(
        "0.1",
        label=lambda underlying, d, max_dte: (
            f"all_contracts/max_dte={max_dte if not max_dte is None else 'all'}/{underlying}/{fmt_date(d)}"
        ),
        exclude_args=["self"],
    )  # 'self' is excluded fromm the arguments as it does not affect the output.
    def list_all_contracts_expiring_after_day(
        self, underlying: str, d: date, max_dte: int = None
    ) -> list[dict]:
        """
        Download ALL option contracts with expiration_date at or past ``d``, with an optional upper bound.

        Parameters
        ----------
        underlying : str
            Underlying ticker symbol, e.g. "AAPL" for Apple.
        d : date
            Date for which to fetch contracts. Only contracts with expiration_date >= d are returned.
        max_dte : int | None, default ``None``
            If not None, only return contracts with ``expiration_date <= d + max_dte`` in business days.
            This is computed using :func:`quantdata.biz_calendar.add_business_days` with the NYSE calendar *including half days*.

        Returns:
        -------
        Result : dict
            See `v3/reference/options/contracts documentation <https://massive.com/docs/rest/options/contracts>`_.
        """
        url = f"{self.BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "as_of": d.strftime("%Y-%m-%d"),
            "expiration_date.gte": d.strftime("%Y-%m-%d"),  # at or after today
            "limit": 1000,
            "order": "asc",
            "sort": "expiration_date",
        }
        if not max_dte is None:
            max_expiry = add_business_days(d, max_dte, exclude_half_days=False)
            verify_inp(
                max_expiry >= d,
                f"'max_dte' {max_dte} leads to max_expiry {max_expiry} before date {d}",
            )
            params["expiration_date.lte"] = max_expiry.strftime("%Y-%m-%d")
        return self.paginate(url, params)

    @_cache(fmt=SubDir.PICKLE).cache(
        "0.1",
        uid=lambda opt_ticker, as_of: f"opt_ref_date/{named_unique_filename48_8(str(opt_ticker) if as_of is None else f'{opt_ticker} {as_of}')}",
    )
    def option_contract_overview( self, opt_ticker: str, as_of : date | None = None ) -> dict:
        """
        Download option `contract overview data <https://massive.com/docs/rest/options/contracts/contract-overview>`_ 
        for a given options ticker.

        Parameters
        ----------
        opt_ticker : str
            Option ticker symbol, e.g. "AAPL230616C00150000" for Apple 150 call expiring on 2023-06-16.
        as_of : :class:`datetime.date` | None, default None
            If not None, fetch the contract overview as of this date. Otherwise, fetch the latest.

        Returns
        -------
        Result : dict
            See `v3/reference/options/contract overview documentation <https://massive.com/docs/rest/options/contracts/contract-overview>`_.
        """
        url = f"{self.BASE_URL}/v3/reference/options/contracts/{opt_ticker}{'' if as_of is None else f'?as_of={as_of.isoformat()}'}"
        r   = self.get(url)
        return r['results']

    @_cache(fmt=SubDir.PICKLE).cache(
        "0.1",
        uid=lambda underlying, d, expiry: f"expiry_contracts/{underlying}/{fmt_date(d)}/{expiry}"
    )
    def list_all_contracts_for_expiry(
        self, underlying: str, d: date, expiry: date
    ) -> list[dict]:
        """
        Download ALL option contracts with expiration_date at or past ``d``, with an optional upper bound.

        Parameters
        ----------
        underlying : str
            Underlying ticker symbol, e.g. "AAPL" for Apple.
        d : date
            Date for which to fetch contracts. Only contracts with expiration_date >= d are returned.
        expiry : date
            Expiration date for which to fetch contracts. Only contracts with expiration_date equal to expiry are returned.

        Returns:
        -------
        Result : dict
            See `v3/reference/options/contracts documentation <https://massive.com/docs/rest/options/contracts>`_.
        """
        url = f"{self.BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "as_of": d.strftime("%Y-%m-%d"),
            "expiration_date.gte": expiry.strftime("%Y-%m-%d"), 
            "expiration_date.lte": expiry.strftime("%Y-%m-%d"), 
            "limit": 1000,
            "order": "asc",
            "sort": "expiration_date",
        }
        return self.paginate(url, params)

    @_cache(fmt=SubDir.POLARS_PARQUET).cache(
        "0.1",
        uid=lambda ticker, start, end: f"adjfactors/{ticker}/{fmt_date(start)}_{fmt_date(end)}",
        exclude_args=["self"],
    )  # 'self' is excluded fromm the arguments as it does not affect the output.
    def adjustment_factors(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        """
        Downloads daily adjusted and unadjusted close, and computes the adjustment factor as ``adj_close / unadj_close``.

        See `custom bars documentation <https://massive.com/docs/rest/stocks/aggregates/custom-bars>`_

        Returns a Polars frame with:

        * date
        * close_unadj
        * close_adj
        * factor_close = close_adj / close_unadj

        Notes:

        * Uses NY timezone to derive the trading date from 't'.
        * Uses only 'close' as requested.
        * If you later want volume/cash-vol invariance checks, also download 'v' and compute factors similarly.
        """
        verify_inp(
            start <= end,
            lambda: (
                f"Cannot find adjustment factors for {ticker}: dates must be in order; found start {fmt_date(start)} vs end {fmt_date(end)}"
            ),
        )

        unadjusted = self.fetch_bars(ticker, start, end, adjusted=False, timespan="day")
        adjusted = self.fetch_bars(ticker, start, end, adjusted=True, timespan="day")

        if len(unadjusted) == 0 or len(adjusted) == 0:
            raise RuntimeError(
                f"Cannot find adjustment factors for {ticker} between {fmt_date(start)} and {fmt_date(end)}: no bars found for {'adjusted' if len(adjusted) == 0 else 'unadjusted'} data. Please check that the ticker is correct and that there were trades for this ticker between {fmt_date(start)} and {fmt_date(end)}. If you need help, please contact the authors."
            )

        unadjusted = (
            unadjusted.with_columns(pl.col("ts_ms").dt.date().alias("date"))
            .rename({"close": "unadj_close"})
            .select(["date", "unadj_close"])
        )
        adjusted = (
            adjusted.with_columns(pl.col("ts_ms").dt.date().alias("date"))
            .rename({"close": "adj_close"})
            .select(["date", "adj_close"])
        )

        # Join + factor
        out = (
            unadjusted.join(adjusted, on="date", how="outer")
            .sort("date")
            .with_columns(
                (pl.col("adj_close") / pl.col("unadj_close")).cast(pl.Float32).alias("adjfactor")
            )
        )
        err = out.filter(pl.col("adjfactor").is_infinite() | pl.col("adjfactor").is_nan())
        if err.height > 0:
            print(err.head())  # print stats to stdout
            raise RuntimeError(
                f"Unexpected data error: found infinite/NaN adjustment factor for ticker {ticker} between {fmt_date(start)} and {fmt_date(end)}. This likely indicates a data error in the underlying bars, e.g. a missing unadjusted close price. Please check the underlying bars and contact the authors if you need help."
            )

        out = out.select(["date", "adjfactor"])
        return out

    @_cache(fmt=SubDir.PICKLE).cache(
        "0.1", uid=lambda ticker: f"reference/{ticker}", exclude_args=["self"]
    )  # 'self' is excluded fromm the arguments as it does not affect the output.
    def ticker_reference_data(self, ticker: str) -> dict:
        """
        Return reference data for a given ticker.

        See `ticker overview documentation <https://massive.com/docs/rest/stocks/tickers/ticker-overview>`_ for details.
        This function converts boolean, integer, floats and dates.

        Parameters
        ----------
        ticker : str
            Ticker symbol, e.g. "AAPL230616C00150000" for Apple 150 call expiring on 2023-06-16.

        Returns
        -------
        Result : dict
            Dict with reference data for the ticker. See `ticker overview documentation <https://massive.com/docs/rest/stocks/tickers/ticker-overview>`_ for details.
            The following conversions will be performed:

            * active [bool|None]
            * list_date [date|None]
            * delisted_utc [date|None]
            * market_cap [float|None]
            * round_lot [int|None]
            * total_employees [int|None]
            * weighted_shares_outstanding [int|None]
            * share_class_shares_outstanding [int|None]

        Raises
        ------
        Ticker not found: :class:`FileNotFoundError`
            If the request for reference data from massive did not yield any results.
        """

        url = f"{self.BASE_URL}/v3/reference/tickers/{ticker}"
        response = self.get(url)
        results = response.get("results", {})

        if len(results) == 0:
            raise FileNotFoundError(
                f"No reference data found for ticker {ticker} on massive.com. URL {url}."
            )

        bool_fields = ["active"]
        date_fields = ["list_date", "delisted_utc"]
        float_fields = ["market_cap"]
        int_fields = [
            "round_lot",
            "total_employees",
            "weighted_shares_outstanding",
            "share_class_shares_outstanding",
            "sic_code",
        ]

        for f in bool_fields:
            if f in results:
                results[f] = bool(results[f])

        for f in date_fields:
            if f in results and results[f] is not None:
                results[f] = date.fromisoformat(results[f])

        for f in float_fields:
            if f in results and results[f] is not None:
                results[f] = float(results[f])

        for f in int_fields:
            if f in results and results[f] is not None:
                results[f] = int(results[f])

        return results

    # ----------------------------
    # corporate events etc
    # ----------------------------

    def _process_massive(
        self,
        what,
        rows: list[dict],
        schema: dict[str, pl.DataType],
        sort_by: list[str] | str,
        drop_ticker: bool,
    ) -> pl.DataFrame:
        """
        Helper function to process corporate events data (dividends, splits) and convert to Polars DataFrame with correct schema.
        """
        if not rows:
            return pl.DataFrame(schema=schema)

        df = pl.DataFrame(rows, infer_schema_length=None)
        if drop_ticker:
            df = df.drop("ticker")

        for k, t in schema.items():
            if k not in df.columns:
                df = df.with_columns(pl.lit(None).cast(t).alias(k))
            elif t == pl.Date:
                df = df.with_columns(pl.col(k).str.strptime(pl.Date, "%Y-%m-%d").alias(k))
            else:
                df = df.with_columns(pl.col(k).cast(t).alias(k))

        df = df.sort(sort_by)
        assert set(df.columns) == set(schema), (
            f"Internal error {what()}: expected columns {set(schema)} but got {set(df.columns)}. Sanity check the request and contact the authors if you need help."
        )
        return df

    SCHEMA_DIVIDENDS = {
        "cash_amount": pl.Float64,
        "currency": pl.String,
        "declaration_date": pl.Date,
        "distribution_type": pl.String,
        "ex_dividend_date": pl.Date,
        "frequency": pl.Int16,
        "historical_adjustment_factor": pl.Float64,
        "id": pl.String,
        "pay_date": pl.Date,
        "record_date": pl.Date,
        "split_adjusted_cash_amount": pl.Float64,
    }
    """ Schema of the :class:`polars.DataFrame` returned by :meth:`get_dividends`. """

    @_cache(fmt=SubDir.POLARS_PARQUET).cache(
        "0.1",
        uid=lambda ticker, start, end: f"dividends/{ticker}/{fmt_date(start)}_{fmt_date(end)}",
        exclude_args=["self"],
    )  # 'self' is excluded fromm the arguments as it does not affect the output.
    def get_dividends(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        """
        Retrieve dividends which go ex div between two dates from `/stocks/v1/dividends < https://massive.com/docs/rest/stocks/corporate-actions/dividends>`_.

        Parameters
        ----------
        ticker : str
            Ticker symbol, e.g. "AAPL" for Apple.
        start : :class:`datetime.date`
            Start date for dividends.
        end : :class:`datetime.date`
            End date for dividends.

        Returns
        -------
        :class:`polars.DataFrame`
            DataFrame with dividend information using schema :class:`MassiveClient.SCHEMA_DIVIDENDS`.
            The dataframe can have length zero (but it will have the correct data schema).
        """

        verify_inp(
            start <= end,
            lambda: (
                f"Cannot find adjustment factors for {ticker}: dates must be in order; found start {fmt_date(start)} vs end {fmt_date(end)}"
            ),
        )

        url = f"{self.BASE_URL}/stocks/v1/dividends"
        params = {
            "ticker": ticker,
            "ex_dividend_date.gte": start.strftime("%Y-%m-%d"),
            "ex_dividend_date.lte": end.strftime("%Y-%m-%d"),
            "limit": 1000,
            "order": "asc",
            "sort": "ex_dividend_date",
        }
        rows = self.paginate(url, params)
        return self._process_massive(
            what=lambda: f"dividends for {ticker} between {fmt_date(start)} and {fmt_date(end)}",
            rows=rows,
            schema=MassiveClient.SCHEMA_DIVIDENDS,
            sort_by="ex_dividend_date",
            drop_ticker=True,
        )

    SCHEMA_SPLITS = {
        "adjustment_type": pl.String,
        "execution_date": pl.Date,
        "historical_adjustment_factor": pl.Float64,
        "id": pl.String,
        "split_from": pl.Int32,
        "split_to": pl.Int32,
    }
    """ Schema of the :class:`polars.DataFrame` returned by :meth:`get_splits`. """

    @_cache(fmt=SubDir.POLARS_PARQUET).cache(
        "0.1",
        uid=lambda ticker, start, end: f"splits/{ticker}/{fmt_date(start)}_{fmt_date(end)}",
        exclude_args=["self"],
    )  # 'self' is excluded fromm the arguments as it does not affect the output.
    def get_splits(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        """Retrieve stock splits between two dates from `/stocks/v1/splits <https://massive.com/docs/rest/stocks/corporate-actions/splits>`_."""
        url = f"{self.BASE_URL}/stocks/v1/splits"
        params = {
            "ticker": ticker,
            "execution_date.gte": start.strftime("%Y-%m-%d"),
            "execution_date.lte": end.strftime("%Y-%m-%d"),
            "limit": 1000,
            "order": "asc",
            "sort": "execution_date",
        }
        rows = self.paginate(url, params)
        return self._process_massive(
            what=lambda: f"splits for {ticker} between {fmt_date(start)} and {fmt_date(end)}",
            rows=rows,
            schema=MassiveClient.SCHEMA_SPLITS,
            sort_by="execution_date",
            drop_ticker=True,
        )

    SCHEMA_HAS_TICKER_CHANGES = {
        "date": pl.Date,
        "has_ticker_change": pl.Boolean,
    }
    """ Schema of the :class:`polars.DataFrame` returned by :meth:`has_ticker_changes`. """

    @_cache(fmt=SubDir.POLARS_PARQUET).cache(
        "0.1",
        uid=lambda ticker, start, end: f"ticker_changes/{ticker}/{fmt_date(start)}_{fmt_date(end)}",
        exclude_args=["self"],
    )  # 'self' is excluded fromm the arguments as it does not affect the output.
    def has_ticker_changes(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        """
        Find dates at which an underlying had ticker changes.

        If the endpoint is not available return an empty frame.
        At the moment ticker changes are `the only events massive.com supports .
        """

        client = self.get_rest_client()
        events = client.get_ticker_events(ticker)
        events = events.events
        if len(events) == 0:
            return pl.DataFrame(schema=MassiveClient.SCHEMA_HAS_TICKER_CHANGES)
        df = pl.DataFrame(events)
        event_types = set(df.select(pl.col("type").unique()).get_column("type").to_list())
        assert event_types == {"ticker_change"}, (
            f"Internal error retrieving ticker changes for {ticker} between {fmt_date(start)} and {fmt_date(end)}: expected event type 'ticker_change' but got {sorted(event_types)}. Please update code. Ref: https://massive.com/docs/rest/stocks/corporate-actions/ticker-events"
        )

        df = df.select(
            pl.col("date").str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d").alias("date")
        )

        # filter by requested window
        df = df.filter((pl.col("date") >= start) & (pl.col("date") <= end))
        if df.is_empty():
            return pl.DataFrame(schema=MassiveClient.SCHEMA_HAS_TICKER_CHANGES)

        df = (
            df.unique(subset=["date"])
            .sort("date")
            .with_columns(pl.lit(True).alias("has_ticker_change"))
        )
        df = df.sort("date")
        assert df.height > 0
        assert set(df.columns) == set(MassiveClient.SCHEMA_HAS_TICKER_CHANGES), (
            f"Internal error retrieving ticker changes for {ticker} between {fmt_date(start)} and {fmt_date(end)}: expected columns {set(MassiveClient.SCHEMA_HAS_TICKER_CHANGES)} but got {set(df.columns)}. Please update code. Ref: https://massive.com/docs/rest/stocks/corporate-actions/ticker-events"
        )
        return df

    @version("0.1")
    def consistent_prev_close( self, ticker : str, d : date, *, prev_d : date|None = None, exclude_half_days: bool = True ) -> float:
        r"""
        Computes the previous day close consistent with today's unadjusted prices.

        It computes
        $$
            S_{\mathrm{t-1}} := S^{\mathrm{adjusted}}_{\mathrm{t-1}} \cdot \frac{S^{\mathrm{unadjusted}}_{\mathrm{t}}}{S^{\mathrm{adjusted}}_{\mathrm{t}}}
        $$

        Parameters
        ----------
        ticker : str
            Ticker symbol, e.g. "AAPL".
        d : :class:`datetime.date`
            Date for which to compute the previous close.
        prev_d : :class:`datetime.date` | None, default None
            If not None, use this date as the previous day instead of computing it as the previous business day. 
        exclude_half_days : bool, default ``True``
            Whether to exclude half days when computing the previous business day. 
            This is used if ``prev_d`` is ``None``.

        Returns
        -------
        Close : ``float``
            The previous day's close, consistent with today's unadjusted prices.
        """
        prev_d = prev_d if not prev_d is None else add_business_days( d, -1, exclude_half_days=exclude_half_days )  # previous business day (NB: if d is a Monday then this is the previous Friday, so we correctly skip the weekend)
        r = self._PREV_CLOSE_CACHE.get((ticker, d, prev_d), None)
        if r is not None:
            return r

        close_adj_prev = self.fetch_day_summary( ticker, prev_d, adjusted=True, can_be_none=False )["close"]
        close_unadj = self.fetch_day_summary( ticker, d, adjusted=False, can_be_none=False )["close"]
        close_adj = self.fetch_day_summary(ticker, d, adjusted=True, can_be_none=False)[ "close" ]
        prev_adj_close = close_adj_prev / close_adj * close_unadj
        self._PREV_CLOSE_CACHE[(ticker, d, prev_d)] = prev_adj_close
        return prev_adj_close
