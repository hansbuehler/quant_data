"""
WIDB connection and OptionMetrics query helpers.

Wraps the ``wrds`` Python library to execute SQL queries against
the OptionMetrics (IvyDB) dataset into a :class:`quantdata.widb.WIDBClient` class
which provides access via the following functions:

* :meth:`quantdata.widb.WIDBClient.lookup_secid(ticker)`: look up the OptionMetrics secid for a given ticker symbol.
* :meth:`quantdata.widb.WIDBClient.zero_curve(date)`: fetch the zero-coupon yield curve for a given date.
* :meth:`quantdata.widb.WIDBClient.spot_price(secid, date)`: fetch the spot price of the underlying for a given secid and date.
* :meth:`quantdata.widb.WIDBClient.option_prices(secid, date)`: fetch raw option price records for a given secid and date.
* :meth:`quantdata.widb.WIDBClient.forward_prices(secid, date)`: fetch forward prices for a given secid and date
"""

from __future__ import annotations

from quantdata import env
from quantdata.biz_calendar import to_date
from cdxcore.subdir import SubDir, verify, version
from cdxcore.verbose import Context
from cdxcore.util import fmt_date, verify_inp
import datetime as dt
import pandas as pd
import wrds as wrds
from dataclasses import dataclass
import warnings as warnings
import os as os
import numpy as np

# Cache available table names per (library, username)
_TABLES_CACHE: dict[tuple[str, str], set[str]] = {}

def _pick_optionm_library(conn: wrds.Connection) -> str:
    """Pick an OptionMetrics library that exists for this WIDB account."""
    libs = conn.list_libraries()
    for lib in ("optionm_all", "optionm"):
        if lib in libs:
            return lib
    raise RuntimeError(
        "WIDB libraries do not include 'optionm_all' or 'optionm'. "
        "Your WIDB account may not have OptionMetrics/IvyDB access."
    )

def _tables_in_library(conn: wrds.Connection, library: str) -> set[str]:
    username = WIDBClient.credentials().username
    key = (library, username)
    if key not in _TABLES_CACHE:
        _TABLES_CACHE[key] = set(conn.list_tables(library))
    return _TABLES_CACHE[key]

def _resolve_table(
    conn: wrds.Connection,
    *,
    logical: str,
    candidates: list[str],
    date: dt.date | None = None,
) -> str:
    """Resolve a logical table name to an existing qualified table.

    OptionMetrics on WIDB is often year-partitioned (e.g. opprcd2010) and/or
    uses base names without the "_all" suffix.
    """
    lib = _pick_optionm_library(conn)
    tables = _tables_in_library(conn, lib)

    expanded: list[str] = []
    for cand in candidates:
        if "{year}" in cand:
            if date is None:
                raise ValueError(f"Table candidate '{cand}' requires a date/year")
            expanded.append(cand.format(year=date.year))
        else:
            expanded.append(cand)

    for name in expanded:
        if name in tables:
            return f"{lib}.{name}"

    raise RuntimeError(
        f"Could not resolve OptionMetrics table for '{logical}'. Tried: {expanded} in library '{lib}'. "
        "Your WIDB account may not have the required dataset access, or the table names differ."
    )

@version("0.1")
@dataclass(frozen=True)
class WRDSCredentials:
    """ WIDB login credentials read from environment variables. """
    username: str #: User account name for `WIDB <https://wrds-www.wharton.upenn.edu/>`__. For live access, set the environment variable ``WRDS_USERNAME`` to your account name.
    wrds_off: bool = False #: If ``True``, WIDB access is disabled. To disable WIDB access, set the environment variable ``WRDS_OFF`` to ``1``.

    @staticmethod
    def create() -> WRDSCredentials:
        """ Returns :class:`dys.core.Environment.WRDSCredentials` with WIDB credentials read from ``WRDS_USERNAME`` """
        username = os.environ.get("WRDS_USERNAME", None)
        wrds_off = os.environ.get("WRDS_OFF", None) == "1"

        if username is None:
            raise RuntimeError("For live access to WIDB, set the environment variable WRDS_USERNAME: currently " + ("not set" if username is None else f"'{username}'")+".")
        if wrds_off:
            warnings.warn("WIDB is disabled")
        return WRDSCredentials(username=username,wrds_off=wrds_off)

@version("1.0", dependencies=[WRDSCredentials])
class WIDBClient(object):
    """
    WIDB connectivity.
    See `the IvyDB reference manual <https://wrds-www.wharton.upenn.edu/documents/1995/IvyDB_US_v6.0_Reference_Manual.pdf>`_ for details on the OptionMetrics dataset and table schemas.

    Paramemers
    ----------
    connect : bool, default ``False``
        If True, establish the connection immediately. Otherwise, delay until the first query is executed.
    """

    locations = env.locations("widb", default_fmt=SubDir.PANDAS_PARQUET)

    _conn : wrds.Connection | None = None #: Active WIDB connection, or None if not connected.
    _credentials : WRDSCredentials | None = None # Credentials for WIDB connection, read from environment variables when first needed.

    OPTION_COLS = {
        'symbol',
        'optionid',
        'am_settlement',
        'cp_flag',
        'exercise_style',
        'strike_price',
        'exdate',
        'best_bid',
        'best_offer',
        'volume',
        'open_interest',
        'impl_volatility',
        'delta',
        'gamma',
        'vega',
        'theta',
        'ss_flag',
        'expiry_indicator',
        'contract_size'}
    """ Columns returned by option_prices() queries. """

    def __init__(self, connect : bool = False):
        """ Initialize """
        if connect:
            self.connect()

    @staticmethod   
    def credentials() -> WRDSCredentials:
        """
        Returns WIDB credentials read from environment variables. 

        At the moment this is just the ``username``.
        """
        if WIDBClient._credentials is None:
            WIDBClient._credentials = WRDSCredentials.create()
        return WIDBClient._credentials
    
    @staticmethod
    def connect() -> wrds.Connection:
        """
        Establish a WIDB connection if not already connected.
        Returns a :class:`wrds.Connection` object.
        
        Users usually do not have to call this function directly when using any of the
        prepared cached functions below as those will obtain a connection
        lazily themselves.

        This function is provided for native WRDS access.
        """
        if WIDBClient._conn is None:
            verify( not env.test_mode, "Attempted to connect to WRDS IvyDB in test mode. Set QUANTDATA_TEST_MODE=0 to enable connections." )
            credentials = WIDBClient.credentials()
            verify( not credentials.wrds_off, "Attempted to connect to WRDS IvyDB while WRDS IvyDB access is disabled. Set QUANTDATA_WRDS_OFF=0 to enable connections." )
            try:
               WIDBClient._conn  = wrds.Connection(wrds_username=credentials.username) 
            except Exception as e:
                raise RuntimeError(f"Failed to connect to WRDS with username '{credentials.username}': {str(e)}") from e
        return WIDBClient._conn
    
    @locations.raw(fmt=SubDir.PICKLE).cache(
                    "0.2",
                    uid = "ticker_to_secid_map",
                    exclude_args=['self'])
    def ticker_to_secid_map(self) -> dict[str, int]:
        """
        Fetch a mapping of all tickers to secids from the securd table.
        Use :meth:`quantdata.widb.client.WIDBClient.lookup_secid` for individual lookups, which is cached separately.
        """
        conn = self.connect()
        assert not conn is None, "WRDS connection is not established."
        securd = _resolve_table(conn, logical="securd", candidates=["securd", "securd1"])
        query = f"""
            SELECT ticker, secid
            FROM {securd}
        """
        df = conn.raw_sql(query)
        return dict(zip(df["ticker"], df["secid"]))

    @version("0.3", dependencies=["ticker_to_secid_map"])
    def lookup_secid(self, ticker: str) -> int:
        """Look up the OptionMetrics (IvyDB) secid for a given ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol, e.g. 'SPX'.

        Returns
        -------
        secid : int
            The secid.

        Raises
        ------
        Not found: :class:`LookupError`
            If the ticker is not found.
        """
        ticker_map = self.ticker_to_secid_map()
        secid      = ticker_map.get(ticker, None) 
        if secid is None:
            raise LookupError(f"Ticker '{ticker}' not found in OptionMetrics securd table.")
        return int(secid)
    
    @version("0.1", dependencies=[ticker_to_secid_map])
    def lookup_ticker(self, secid: int) -> str:
        """Look up the ticker symbol for a given OptionMetrics (IvyDB) secid.

        Parameters
        ----------
        secid : int
            The secid.

        Returns
        -------
        ticker : str
            Ticker symbol, e.g. 'SPX'.

        Raises
        ------
        Not found: :class:`LookupError`
            If the secid is not found.
        """
        verify_inp( isinstance(secid, int), f"'secid' must be an integer; found '{secid}' of type {type(secid)}")
        ticker_map = self.ticker_to_secid_map()
        for ticker, tsecod in ticker_map.items():
            if int(tsecod) == secid:
                return ticker
        raise LookupError(f"SecID '{secid}' not found in OptionMetrics securd table.")  

    @locations.raw("options", fmt=SubDir.PANDAS_PARQUET).cache(
                    "0.4", 
                    uid = lambda secid, date : f"{int(secid)}/{fmt_date(date)}", 
                    exclude_args=['self', 'verbose'])
    def options(
        self,
        secid: int,
        date: dt.date,
        *,
        verbose: Context = Context.quiet
    ) -> pd.DataFrame:
        """Fetch all option price records for a given secid on a single date.

        Columns returned are contained in :attr:`quantdata.widb.client.WIDBClient.OPTION_COLS`, which includes:
        'symbol', 'optionid', 'am_settlement', 'cp_flag', 'exercise_style', 'strike_price', 'exdate', 'best_bid', 'best_offer', 'volume', 'open_interest', 'impl_volatility', 'delta', 'gamma', 'vega', 'theta', 'ss_flag', 'expiry_indicator', 'contract_size'.
        See `the IvyDB reference manual <https://wrds-www.wharton.upenn.edu/documents/1995/IvyDB_US_v6.0_Reference_Manual.pdf>`_ page 20.

        Parameters
        ----------
        secid : int
            OptionMetrics security ID. Use :meth:`quantdata.widb.client.WIDBClient.lookup_secid` to obtain the secid for a given ticker.
        date : :class:`dt.date`
            Observation date.
        verbose : :class:`cdxcore.verbose.Context`, default :attr:`cdxcore.verbose.Context.quiet`
            Channel to print verbosity information, if any.

        Returns
        -------
        pd.DataFrame
            Raw option prices for the date.
        """
        verify_inp( isinstance(secid, int), f"'secid' must be an integer; found '{secid}'")
        date  = to_date(date)
        secid = int(secid)
        conn = self.connect()
        assert not conn is None, "WIDB connection is not established."

        opprcd = _resolve_table(
            conn,
            logical="opprcd",
            candidates=["opprcd{year}", "opprcd"],
            date=date,
        )
        query = f"""
            SELECT
                symbol,
                optionid,
                am_settlement,
                cp_flag,
                'E' AS exercise_style,
                strike_price / 1000.0 AS strike_price,
                exdate,
                best_bid,
                best_offer,
                volume,
                open_interest,
                impl_volatility,
                ss_flag,
                delta,
                gamma,
                vega,
                theta,
                expiry_indicator,
                contract_size
            FROM {opprcd}
            WHERE secid = {secid}
            AND date = '{date}'
        """
        verbose.write(lambda : f"Fetching option prices for secid {secid} on {date}.")
        df = conn.raw_sql(query)
        assert set(df.columns) <= self.OPTION_COLS, f"{secid} {date}: missing columns {sorted(self.OPTION_COLS-set(df.columns))}"
        return df

    @locations.raw("zero_curve", fmt=SubDir.PANDAS_PARQUET).cache(
                    "0.2",
                    uid = lambda date : f"{fmt_date(date)}",
                    exclude_args=['self','verbose'])
    def zero_curve(
        self,
        date: dt.date,
        *,
        verbose: Context = Context.quiet,
    ) -> pd.DataFrame:
        """
        Fetch the zero-coupon yield curve for a given date.

        Returns columns: 'date', 'days', 'rate'
        Rate is continuously compounded using ACT/365, i.e. ``df = exp(-rate * calendar_days/365)``.

        Use :func:`quantdata.widb.client.interpolate_zero_curve`
        to interpolate discount factors for specific days to expiry.

        Parameters
        ----------
        date : :class:`dt.date`
            Observation date.
        verbose : :class:`cdxcore.verbose.Context`, default ``Context.quiet``
            Channel to print verbosity information, if any.

        Returns
        -------
        Zero curve: :class:`pandas.DataFrame`
            Zero-curve data. Use :func:`quantdata.widb.client.interpolate_zero_curve`
            to interpolate discount factors for specific days to expiry.
        """
        date  = to_date(date)
        conn = self.connect()
        assert not conn is None, "WIDB connection is not established."
        zerocd = _resolve_table(conn, logical="zerocd", candidates=["zerocd"])
        query = f"""
            SELECT date, days, rate
            FROM {zerocd}
            WHERE date = '{date}'
            ORDER BY days
        """
        verbose.write(lambda : f"Fetching USD zero curve for {date}.")
        return conn.raw_sql(query)

    @locations.raw("forwards", fmt=SubDir.PANDAS_PARQUET).cache(
                    "0.3",
                    uid = lambda secid, date : f"{int(secid)}/{fmt_date(date)}",
                    exclude_args=['self','verbose'])
    def forwards(
        self,
        secid: int,
        date: dt.date,
        *,
        verbose: Context = Context.quiet,
    ) -> pd.DataFrame:
        """Fetch forward prices for a given secid on a date.

        Returns columns: date, exdate, am_settlement, forward_price

        Parameters
        ----------
        secid : int
            OptionMetrics security ID. Use :meth:`quantdata.widb.client.WIDBClient.lookup_secid` to obtain the secid for a given ticker.
        date : :class:`dt.date`
            Observation date.
        verbose : :class:`cdxcore.verbose.Context`, default ``Context.quiet``
            Channel to print verbosity information, if any.

        Returns
        -------
        Forward table: :class:`pandas.DataFrame`
            Forward price data with columns ``date``, ``exdate``, ``am_settlement``, ``forward_price``.
        """
        verify_inp( isinstance(secid, int), f"'secid' must be an integer; found '{secid}'")
        date  = to_date(date)
        secid = int(secid)
        conn = self.connect()
        assert not conn is None, "WIDB connection is not established."
        fwdprd = _resolve_table(
            conn,
            logical="fwdprd",
            candidates=["fwdprd{year}", "fwdprd"],
            date=date,
        )
        query = f"""
            SELECT
                date,
                expiration AS exdate,
                amsettlement AS am_settlement,
                forwardprice AS forward_price
            FROM {fwdprd}
            WHERE secid = {secid}
            AND date = '{date}'
        """
        verbose.write(lambda : f"Fetching forward prices for secid {secid} on {date}.")
        return conn.raw_sql(query)

    @locations.raw("spot_price", fmt=SubDir.PICKLE).cache(
                    "0.2",
                    uid = lambda secid, date : f"{secid}/{fmt_date(date)}",
                    exclude_args=['self', 'verbose'])
    def spot_price(
        self,
        secid: int,
        date: dt.date,
        *,
        verbose: Context = Context.quiet,
    ) -> float:
        """Fetch the spot (close) price of the underlying for a given date.

        Parameters
        ----------
        secid : int
            OptionMetrics security ID. Use :meth:`quantdata.widb.client.WIDBClient.lookup_secid` to obtain the secid for a given ticker.
        date : :class:`dt.date`
            Observation date.
        verbose : :class:`cdxcore.verbose.Context`, default ``Context.quiet``   
            Channel to print verbosity information, if any.

        Returns
        -------
        Spot : float
            Spot price.

        Raises
        ------
        Spot not found: :class:`LookupError`
            If no spot price is available. This likely means either the ``secid`` is not covered by Option Metrics,
            or ``date`` is not a business day for this underlying.
        """
        verify_inp( isinstance(secid, int), f"'secid' must be an integer; found '{secid}'")
        date  = to_date(date)
        secid = int(secid)
        conn = self.connect()
        assert not conn is None, "WIDB connection is not established."
        secprd = _resolve_table(
            conn,
            logical="secprd",
            candidates=["secprd", "secprd{year}"],
            date=date,
        )
        query = f"""
            SELECT close
            FROM {secprd}
            WHERE secid = {secid}
            AND date = '{date}'
        """
        verbose.write(lambda : f"Fetching spot price for secid {secid} on {date}.")
        df = conn.raw_sql(query)
        if df.empty:
            raise LookupError(f"No spot price found for secid {secid} on {date}.")
        return float(df["close"].iloc[0])

@version("0.1")
def interpolate_zero_curve(
    zero_curve: pd.DataFrame,
    days_to_expiry: np.ndarray,
) -> np.ndarray:
    """
    Interpolate discount factors from the zero-coupon curve using ACT/365.

    Parameters
    ----------
    zero_curve : :class:`pandas.DataFrame`
        Must have columns ``days`` and ``rate``.
    days_to_expiry : :class:`numpy.ndarray`
        Calendar days to each expiry, as floats

    Returns
    -------
    Discount factors: :class:`numpy.ndarray`
        Discount factors ``exp(-rate * days / 365)``.
    """
    assert "days" in zero_curve.columns and "rate" in zero_curve.columns, (
        "zero_curve must have 'days' and 'rate' columns"
    )
    curve_days = zero_curve["days"].values.astype(float)
    curve_rates = zero_curve["rate"].values.astype(float) # in pcnt e.g. 3 = 3%
    dte_float = days_to_expiry.astype(float)
    rates = np.interp(dte_float, curve_days, curve_rates)
    return np.exp(-rates * dte_float / 36500.0)

