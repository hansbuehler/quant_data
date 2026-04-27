"""
Basic polygon i/o
"""

from quantdata.massive import MassiveClient, SubDir
from quantdata.biz_calendar import TZ_NY
from quantdata.massive.trade_conds import pack_conditions
from cdxcore.version import version
from cdxcore.util import fmt_date
from datetime import date
import polars as pl
import requests as requests

cache_root = MassiveClient.locations.raw("fetch",fmt=SubDir.POLARS_PARQUET)

# =============================================================================
# Polygon fetch: v3 trades / quotes (SIP). Keep ts_ns in ns.
# =============================================================================

@version("0.1", dependencies=[MassiveClient])
def has_any_quotes(ticker: str, d: date, *, client: MassiveClient | None = None) -> bool:
    client   = MassiveClient() if client is None else client
    url      = f"{MassiveClient.BASE_URL}/v3/quotes/{ticker}"
    response = client.get(url, params={
                "timestamp": d.strftime("%Y-%m-%d"),   # YYYY-MM-DD
                "limit": 1,
                "order": "asc",
                "sort": "timestamp"
    })
    return len(response['results']) > 0

@version("0.1", dependencies=[MassiveClient])
def has_any_trades(ticker: str, d: date, *, client: MassiveClient | None = None) -> bool:
    client    = MassiveClient() if client is None else client
    url       = f"{MassiveClient.BASE_URL}/v3/trades/{ticker}"
    response  = client.get(url, params={
                "timestamp": d.strftime("%Y-%m-%d"),   # YYYY-MM-DD
                "limit": 1,
                "order": "asc",
                "sort": "timestamp"
    })
    return len(response['results']) > 0

# =============================================================================
# Polygon fetch: v3 trades / quotes (SIP). Keep ts_ns in ns.
# =============================================================================

@cache_root.cache("0.1", dependencies=[MassiveClient], exclude_args=["client"],
                  label=lambda ticker,d: f"trades/{ticker}/{fmt_date(d)}")  
def fetch_trades_sip_ns(ticker: str, 
                        d: date, *, 
                        client: MassiveClient | None = None) -> pl.DataFrame:
    """
    Gets trades from /v3/trades/{ticker} sorted by sip_timestamp asc.
    Used both for options and underlyings.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "AAPL230616C00150000" for Apple 150 call expiring on 2023-06-16.
    d : date
        Date for which to fetch trades.
    client : MassiveClient, optional
        MassiveClient to use for the request. If None, a new MassiveClient is created.
        Passing a client is useful for caching.

    Returns
    -------
    Result : :class:`polars.DataFrame`
        Returns a data frame with the below columns. The dataframe can have lenght zero, but it will have the correct data schema.
        
        * ts_ns: Datetime[ns, tz]
        * price: Float64
        * size: Float64
        * trade_code: UInt32
    """
    client = MassiveClient() if client is None else client
    url = f"{MassiveClient.BASE_URL}/v3/trades/{ticker}"
    params = {
        "timestamp": d.strftime("%Y-%m-%d"),
        "order": "asc",
        "limit": 50000,
        "sort": "sip_timestamp",
    }
    rows = client.paginate(url, params)
    rows = [ dict(price=r["price"], 
                  size=r["size"], 
                  sip_timestamp=r["sip_timestamp"],
                  trade_code=pack_conditions(r.get("conditions", []))) for r in rows ]
    schema = {
        "ts_ns": pl.Datetime("ns", TZ_NY),
        "price": pl.Float64,
        "size": pl.Float64,
        "trade_code": pl.UInt32,
    }

    if not rows:
        return pl.DataFrame(schema=schema)

    df = pl.DataFrame(rows, infer_schema_length=None)
    out = df.select(
            pl.from_epoch(pl.col("sip_timestamp").cast(pl.Int64), time_unit="ns")
            .dt.convert_time_zone(TZ_NY.key)
            .alias("ts_ns"),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64),
            pl.col("trade_code").cast(pl.UInt32),
        ).sort("ts_ns")
    return out

@cache_root.cache("0.1", dependencies=[MassiveClient], exclude_args=["client"],
                  label=lambda ticker,d: f"quotes/{ticker}/{fmt_date(d)}")  
def fetch_quotes_sip_ns(ticker: str, d: date, *, client: MassiveClient | None = None) -> pl.DataFrame:
    """
    Fetches quotes from /v3/quotes/{ticker} sorted by sip_timestamp asc.
    Used both for options and underlyings.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "AAPL230616C00150000" for Apple 150 call expiring on 2023-06-16.
    d : date
        Date for which to fetch quotes.
    client : MassiveClient, optional
        MassiveClient to use for the request. If None, a new MassiveClient is created.
        Passing a client is useful for caching.

    Returns
    -------
    Result : :class:`polars.DataFrame`
        Returns a data frame with the below columns. The dataframe can have lenght zero, but it will have the correct data schema.
        
        * ts_ns: Datetime[ns, tz]
        * bid_price: Float64
        * bid_size: Float64
        * ask_price: Float64
        * ask_size: Float64
    """
    client = MassiveClient() if client is None else client
    url = f"{MassiveClient.BASE_URL}/v3/quotes/{ticker}"
    params = {
        "timestamp": d.strftime("%Y-%m-%d"),
        "order": "asc",
        "limit": 50000,
        "sort": "sip_timestamp",
    }
    rows = client.paginate(url, params)
    schema = {
        "ts_ns": pl.Datetime("ns", TZ_NY),
        "bid_price": pl.Float64,
        "bid_size": pl.Float64,
        "ask_price": pl.Float64,
        "ask_size": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)

    df = pl.DataFrame(rows, infer_schema_length=None)
    out = (
        df.select(
            pl.from_epoch(pl.col("sip_timestamp").cast(pl.Int64), time_unit="ns")
            .dt.convert_time_zone(TZ_NY.key)
            .alias("ts_ns"),
            pl.col("bid_price").cast(pl.Float64),
            pl.col("bid_size").cast(pl.Float64),
            pl.col("ask_price").cast(pl.Float64),
            pl.col("ask_size").cast(pl.Float64),
        )
        .sort("ts_ns")
    )
    return out

@cache_root(fmt=SubDir.PICKLE).cache("0.1", dependencies=[MassiveClient], exclude_args=["client"],
                  label=lambda ticker,d,adjusted,can_be_none: f"ocv/{ticker}/{fmt_date(d)} {'a1' if adjusted else 'a0'} {'c0' if can_be_none else 'c1'}")  
def fetch_day_ocv(ticker: str, d: date, *, adjusted : bool, can_be_none : bool = False, client: MassiveClient|None = None) -> dict | None:
    """
    Fetch open/close/volume for ``ticker`` on date ``d``.
    Returns dict with keys: open, close, volume, adjusted

    If no trade was recorded raise an exception except of ``can_be_none`` is True, in which case return ``None``.

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
    client : MassiveClient, optional
        MassiveClient to use for the request. If None, a new MassiveClient is created.
        Passing a client is useful for caching.

    Returns
    -------
    dict
        Dict with keys: ``open [float]``, ``close [float]``, ``volume [float]``, ``adjusted [bool]``
    """
    client = MassiveClient() if client is None else client
    url = f"{MassiveClient.BASE_URL}/v1/open-close/{ticker}/{d:%Y-%m-%d}"
    try:
        data = client.get(url, {"adjusted": "false" if not adjusted else "true"})
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404 and can_be_none:
            return None
        raise requests.exceptions.HTTPError(f"No open/close/volume for {ticker} on {d} url {url}: {e.response.text}", response=e.response, request=e.request) from e

    # Polygon open-close has "open","close","volume" (and more)
    open = data.get("open")
    close = data.get("close")
    volume = data.get("volume")
    has_data = open is not None and close is not None and volume is not None
    if not has_data:
        if can_be_none:
            return None
        raise RuntimeError(f"Cannot get {'adjusted' if adjusted else 'unadjusted'} open/close/volume for {ticker} on {d}. Massive URL used: '{url}'.")

    return {
        "open": float(open),
        "close": float(close),
        "volume": float(volume),
        "adjusted": adjusted,
    }
