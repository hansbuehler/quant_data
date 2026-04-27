"""
WRDS connection and OptionMetrics query helpers.

Wraps the ``wrds`` Python library to execute SQL queries against
the OptionMetrics (IvyDB) dataset on WRDS into a :class:`quantdata.wrds.WRDSConnection` class
which provides access via the following functions:

* :meth:`quantdata.wrds.WRDSConnection.lookup_secid(ticker)`: look up the OptionMetrics secid for a given ticker symbol.
* :meth:`quantdata.wrds.WRDSConnection.zero_curve(date)`: fetch the zero-coupon yield curve for a given date.
* :meth:`quantdata.wrds.WRDSConnection.spot_price(secid, date)`: fetch the spot price of the underlying for a given secid and date.
* :meth:`quantdata.wrds.WRDSConnection.option_prices(secid, date)`: fetch raw option price records for a given secid and date.
* :meth:`quantdata.wrds.WRDSConnection.forward_prices(secid, date)`: fetch forward prices for a given secid and date

"""

from __future__ import annotations

from quantdata import env, PrettyObject, Locations
from cdxcore.subdir import SubDir, verify, version
from cdxcore.verbose import Context
from cdxcore.util import fmt_date
import datetime as dt
import pandas as pd
import wrds as wrds
from dataclasses import dataclass
import warnings as warnings
import os as os

# Cache available table names per (library, username)
_TABLES_CACHE: dict[tuple[str, str], set[str]] = {}

def _pick_optionm_library(conn: wrds.Connection) -> str:
    """Pick an OptionMetrics library that exists for this WRDS account."""
    libs = conn.list_libraries()
    for lib in ("optionm_all", "optionm"):
        if lib in libs:
            return lib
    raise RuntimeError(
        "WRDS libraries do not include 'optionm_all' or 'optionm'. "
        "Your WRDS account may not have OptionMetrics/IvyDB access."
    )

def _tables_in_library(conn: wrds.Connection, library: str) -> set[str]:
    username = WRDSClient.credentials().username
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

    OptionMetrics on WRDS is often year-partitioned (e.g. opprcd2010) and/or
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
        "Your WRDS account may not have the required dataset access, or the table names differ."
    )

@version("0.1")
@dataclass(frozen=True)
class WRDSCredentials:
    """ WRDS login credentials read from environment variables. """
    username: str #: User account name for `WRDS <https://wrds-www.wharton.upenn.edu/>`__. For live access, set the environment variable ``WRDS_USERNAME`` to your account name.
    wrds_off: bool = False #: If ``True``, WRDS access is disabled. To disable WRDS access, set the environment variable ``WRDS_OFF`` to ``1``.

    @staticmethod
    def create() -> WRDSCredentials:
        """ Returns :class:`dys.core.Environment.WRDSCredentials` with WRDS credentials read from ``WRDS_USERNAME`` """
        username = os.environ.get("WRDS_USERNAME", None)
        wrds_off = os.environ.get("WRDS_OFF", None) == "1"

        if username is None:
            raise RuntimeError("For live access to WRDS, set the environment variable WRDS_USERNAME: currently " + ("not set" if username is None else f"'{username}'")+".")
        if wrds_off:
            warnings.warn("WRDS is disabled")
        return WRDSCredentials(username=username,wrds_off=wrds_off)

@version("1.0", dependencies=[WRDSCredentials])
class WRDSClient(object):
    """
    WRDS connectivity.
    See `the IvyDB reference manual <https://wrds-www.wharton.upenn.edu/documents/1995/IvyDB_US_v6.0_Reference_Manual.pdf>`_ for details on the OptionMetrics dataset and table schemas.

    Paramemers
    ----------
    connect : bool, default ``False``
        If True, establish the connection immediately. Otherwise, delay until the first query is executed.
    """

    locations = env.locations("wrds", default_fmt=SubDir.PANDAS_PARQUET)

    _conn : wrds.Connection | None = None #: Active WRDS connection, or None if not connected.
    _credentials : WRDSCredentials | None = None # Credentials for WRDS connection, read from environment variables when first needed.

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

    def credentials(self) -> WRDSCredentials:
        """ Returns WRDS credentials read from environment variables. """
        if WRDSClient._credentials is None:
            WRDSClient._credentials = WRDSCredentials.create()
        return WRDSClient._credentials
    
    def connect(self) -> wrds.Connection:
        """ Establish a WRDS connection if not already connected. Returns a :class:`wrds.Connection` object. """
        if WRDSClient._conn is None:
            verify( not environment.test_mode, "Attempted to connect to WRDS in test mode. Set QUANTDATA_TEST_MODE=0 to enable connections." )
            credentials = self.credentials()
            verify( not credentials.wrds_off, "Attempted to connect to WRDS while WRDS access is disabled. Set QUANTDATA_WRDS_OFF=0 to enable connections." )
            WRDSClient._conn  = wrds.Connection(wrds_username=credentials.username) 
        return WRDSClient._conn
    
    @locations.raw(fmt=SubDir.PICKLE).cache(
                    "0.2",
                    uid = "ticker_to_secid_map",
                    exclude_args=['self'])
    def ticker_to_secid_map(self) -> dict[str, int]:
        """
        Fetch a mapping of all tickers to secids from the securd table.
        Use :meth:`quantdata.wrds.client.WRDSConnection.lookup_secid` for individual lookups, which is cached separately.
        """
        conn = self.connect()
        assert not conn is None, "WRDS connection is not established."
        securd = _resolve_table(conn, logical="securd", candidates=["securd", "securd1"])
        query = f"""
            SELECT ticker, secid
            FROM {securd}
        """
        df = conn.raw_sql(query)
        return int( dict(zip(df["ticker"], df["secid"])) )

    @version("0.3", dependencies=["ticker_to_secid_map"])
    def lookup_secid(self, ticker: str) -> int:
        """Look up the OptionMetrics (IvyDB) secid for a given ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol, e.g. 'SPX'.

        Returns
        -------
        int
            The secid.

        Raises
        ------
        ValueError
            If the ticker is not found.
        """
        ticker_map = self.ticker_to_secid_map()
        secid      = ticker_map.get(ticker, None) 
        if secid is None:
            raise ValueError(f"Ticker '{ticker}' not found in OptionMetrics securd table.")
        return int(secid)

    @locations.raw("option_prices", fmt=SubDir.PANDAS_PARQUET).cache(
                    "0.4", 
                    uid = lambda secid, date : f"{int(secid)}/{fmt_date(date)}", 
                    exclude_args=['self', 'verbose'])
    def option_prices(
        self,
        secid: int,
        date: dt.date,
        *,
        verbose: Context = Context.quiet
    ) -> pd.DataFrame:
        """Fetch all option price records for a given secid on a single date.

        Columns returned are contained in :attr:`quantdata.wrds.client.WRDSConnection.OPTION_COLS`, which includes:
        'symbol', 'optionid', 'am_settlement', 'cp_flag', 'exercise_style', 'strike_price', 'exdate', 'best_bid', 'best_offer', 'volume', 'open_interest', 'impl_volatility', 'delta', 'gamma', 'vega', 'theta', 'ss_flag', 'expiry_indicator', 'contract_size'.
        See `the IvyDB reference manual <https://wrds-www.wharton.upenn.edu/documents/1995/IvyDB_US_v6.0_Reference_Manual.pdf>`_ page 20.

        Parameters
        ----------
        secid : int
            OptionMetrics security ID. Use :meth:`quantdata.wrds.client.WRDSConnection.lookup_secid` to obtain the secid for a given ticker.
        date : :class:`dt.date`
            Observation date.
        verbose : :class:`cdxcore.verbose.Context`, default :attr:`cdxcore.verbose.Context.quiet`
            Channel to print verbosity information, if any.

        Returns
        -------
        pd.DataFrame
            Raw option prices for the date.
        """
        secid = int(secid)
        conn = self.connect()
        assert not conn is None, "WRDS connection is not established."

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

        Parameters
        ----------
        date : :class:`dt.date`
            Observation date.
        verbose : :class:`cdxcore.verbose.Context`, default :attr:`cdxcore.verbose.Context.quiet`
            Channel to print verbosity information, if any.

        Returns
        -------
        pd.DataFrame
            Zero-curve data.
        """
        conn = self.connect()
        assert not conn is None, "WRDS connection is not established."
        zerocd = _resolve_table(conn, logical="zerocd", candidates=["zerocd"])
        query = f"""
            SELECT date, days, rate
            FROM {zerocd}
            WHERE date = '{date}'
            ORDER BY days
        """
        verbose.write(lambda : f"Fetching USD zero curve for {date}.")
        return conn.raw_sql(query)

    @locations.raw("forward_prices", fmt=SubDir.PANDAS_PARQUET).cache(
                    "0.3",
                    uid = lambda secid, date : f"{int(secid)}/{fmt_date(date)}",
                    exclude_args=['self','verbose'])
    def forward_prices(
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
            OptionMetrics security ID. Use :meth:`quantdata.wrds.client.WRDSConnection.lookup_secid` to obtain the secid for a given ticker.
        date : :class:`dt.date`
            Observation date.
        verbose : :class:`cdxcore.verbose.Context`, default :attr:`cdxcore.verbose.Context.quiet`
            Channel to print verbosity information, if any.

        Returns
        -------
        pd.DataFrame
            Forward price data.
        """
        secid = int(secid)
        conn = self.connect()
        assert not conn is None, "WRDS connection is not established."
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
            OptionMetrics security ID. Use :meth:`quantdata.wrds.client.WRDSConnection.lookup_secid` to obtain the secid for a given ticker.
        date : :class:`dt.date`
            Observation date.
        verbose : :class:`cdxcore.verbose.Context`, default :attr:`cdxcore.verbose.Context.quiet`
            Channel to print verbosity information, if any.

        Returns
        -------
        float
            Spot price.

        Raises
        ------
        ValueError
            If no spot price is available.
        """
        secid = int(secid)
        conn = self.connect()
        assert not conn is None, "WRDS connection is not established."
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
        verbose.write(lambda : f"Fetching spot price for secid={secid} date={date}")
        df = conn.raw_sql(query)
        if df.empty:
            raise ValueError(f"No spot price found for secid={secid} on {date}")
        return float(df["close"].iloc[0])

