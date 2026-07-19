"""
openqai.data.massive.expiries

Per-ticker option expiry flags via the Massive API.

For each trading day, checks whether ANY option on the underlying expires on that day
and classifies it (MONTHLY, QUAD_WITCHING, etc.).

The :func:`ticker_expiries` function uses an incremental cache: it only fetches the
dates that are not already stored, so repeated calls for overlapping date ranges are
cheap.
"""

from __future__ import annotations

import warnings
from datetime import date

import polars as pl
from cdxcore.err import verify
from cdxcore.jcpool import JCPool
from cdxcore.util import fmt_date
from cdxcore.verbose import Context
from cdxcore.version import version

from openqai.data.massive import MassiveClient
from openqai.data.tickers import _locations_tickers
from openqai.data.tickers.ticker_calendar import EXPIRY

_expiries_cache = _locations_tickers

# ---------------------------------------------------------------------------
# Calendar classification (pure Polars, no API)
# ---------------------------------------------------------------------------


@version("0.1")
def _third_friday_expr() -> pl.Expr:
    """Calendar third Friday for the month containing the ``date`` column."""
    d = pl.col("date")
    m_start = d.dt.month_start()
    wd0 = m_start.dt.weekday()
    off = (pl.lit(5) - wd0 + pl.lit(7)) % pl.lit(7)  # Polars ISO weekday: Fri=5
    first_friday = m_start + off * pl.duration(days=1)
    return first_friday + pl.duration(days=14)


@version("0.1", dependencies=[_third_friday_expr])
def classify_expiry(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add ``has_expiry`` (Int8, values from :class:`openqai.data.tickers.ticker_calendar.EXPIRY`)
    to a DataFrame that already contains a ``date`` column of trading days.

    The classification is purely calendar-based:

    * :attr:`EXPIRY.QUAD_WITCHING`: shifted third Friday of Mar/Jun/Sep/Dec
    * :attr:`EXPIRY.MONTHLY`: shifted third Friday of other months
    * :attr:`EXPIRY.NORMAL`: any other trading day
    """
    verify("date" in df.columns, "df must contain a 'date' column")

    df0 = df.with_columns(pl.col("date").cast(pl.Date))
    df1 = df0.with_columns(
        pl.col("date").dt.year().alias("_y"),
        pl.col("date").dt.month().alias("_m"),
        _third_friday_expr().alias("_third_friday"),
    )

    expiry_by_month = df1.group_by(["_y", "_m"]).agg(
        pl.col("_third_friday").first().alias("_third_friday"),
        pl.col("date")
        .filter(pl.col("date") <= pl.col("_third_friday").first())
        .max()
        .alias("_monthly_expiry_date"),
    )

    df2 = df1.join(expiry_by_month, on=["_y", "_m"], how="left")

    is_monthly = pl.col("date") == pl.col("_monthly_expiry_date")
    is_qtr = pl.col("_m").is_in([3, 6, 9, 12])

    out = df2.with_columns(
        pl.when(is_monthly & is_qtr)
        .then(pl.lit(int(EXPIRY.QUAD_WITCHING)))
        .when(is_monthly)
        .then(pl.lit(int(EXPIRY.MONTHLY)))
        .otherwise(pl.lit(int(EXPIRY.NONE)))
        .alias("has_expiry")
        .cast(pl.Int8)
    ).drop(["_y", "_m", "_third_friday", "_monthly_expiry_date"])

    return out


# ---------------------------------------------------------------------------
# Per-day API check
# ---------------------------------------------------------------------------


@version("0.1", dependencies=[classify_expiry, MassiveClient])
def ticker_expiries_day(ticker: str, dt: date) -> tuple[date, int | None]:
    """
    Check via the Massive API whether any option on ``ticker`` expires on ``dt``.

    Returns
    -------
    (dt, expiry_code)
        ``expiry_code`` is an :class:`EXPIRY` int value, or ``None`` if no options expire.
    """
    client = MassiveClient()
    url = f"{MassiveClient.BASE_URL}/v3/reference/options/contracts"
    params = {
        "underlying_ticker": ticker,
        "expiration_date.gte": dt.strftime("%Y-%m-%d"),
        "expiration_date.lte": dt.strftime("%Y-%m-%d"),
        "limit": 1000,
        "order": "asc",
        "expired": True,
        "sort": "expiration_date",
        "as_of": dt.strftime("%Y-%m-%d"),
    }
    res = client.paginate(url, params)
    if not res:
        return dt, None

    df = None
    try:
        df = pl.DataFrame(res, strict=False).select(
            pl.col("expiration_date").str.strptime(pl.Date, "%Y-%m-%d").alias("date")
        )
        df = (
            df.group_by("date")
            .agg(pl.len().alias("num_expiring"))
            .with_columns(pl.col("num_expiring") > 0)
            .rename({"num_expiring": "has_expiry"})
        )
        df = classify_expiry(df)
    except TypeError as e:
        warnings.warn(str(e))
        raise

    if df.is_empty():
        return dt, None

    assert df.height == 1, f"Expected 1 row but got {df.height}"
    return dt, int(df["has_expiry"].to_list()[0])


# ---------------------------------------------------------------------------
# Incremental-cache wrapper
# ---------------------------------------------------------------------------


@version("0.1", dependencies=[ticker_expiries_day, MassiveClient])
def ticker_expiries(
    ticker: str,
    business_days: list[date],
    *,
    num_workers: int = -2,
    threading: bool = False,
    verbose: Context = Context.quiet,
) -> pl.DataFrame:
    """
    For each date in ``business_days``, flag whether any option on ``ticker`` expires
    (0-DTE check) and return the expiry type.

    Results are stored in an incremental cache keyed by ``ticker`` so only the
    missing dates are fetched on subsequent calls.

    Parameters
    ----------
    ticker : str
    business_days : list[date]
        Sorted list of NYSE trading days to check.
    num_workers : int
        Parallel workers.
    threading : bool
        Use threads instead of processes.
    verbose : Context

    Returns
    -------
    pl.DataFrame
        Columns: date (Date), has_expiry (Int8) — only rows where has_expiry is not null.
    """
    head_version = ticker_expiries.version.unique_id48
    cache_key = f"expiries/{ticker}"

    expiries: pl.DataFrame | None = (
        _expiries_cache.read(cache_key, None, version=head_version)
        if _expiries_cache.cache_mode.read
        else None
    )

    known_dates: set[date] = set(expiries["date"].to_list()) if expiries is not None else set()
    missing = sorted(set(business_days) - known_dates)
    tmx = verbose.timer()

    if missing:
        with verbose.write_t(
            lambda: f"Fetching {len(missing)} expiry dates for {ticker}..."
        ) as tme:
            jcpool = JCPool(num_workers, threading=threading)
            found: dict[date, int | None] = {}

            for dt_, code in jcpool.parallel(
                jcpool.delayed(ticker_expiries_day)(ticker=ticker, dt=dt_) for dt_ in missing
            ):
                verbose.report(
                    1,
                    lambda: f"\r  {ticker}: {int((len(found) + 1) * 100 // len(missing))}%...",
                    end="",
                )
                found[dt_] = code

            verbose.write(lambda: f"\rFetched {len(missing)} dates for {ticker}. Took {tme}.")

        new_rows = pl.DataFrame(
            {
                "date": pl.Series(list(found.keys()), dtype=pl.Date),
                "has_expiry": pl.Series(list(found.values()), dtype=pl.Int8),
            }
        )
        expiries = pl.concat([expiries, new_rows]) if expiries is not None else new_rows
        expiries = expiries.sort("date")

        if _expiries_cache.cache_mode.write:
            _expiries_cache.write(cache_key, expiries, version=head_version)

    result = expiries.filter(
        pl.col("date").is_in(business_days) & pl.col("has_expiry").is_not_null()
    )

    dupes = result.group_by("date").len().filter(pl.col("len") > 1)
    if dupes.height > 0:
        raise RuntimeError(f"Duplicate expiry dates for {ticker}: {dupes.head()}")

    verbose.write(
        lambda: (
            f"Found {result.height} expiries for '{ticker}' between "
            f"{fmt_date(business_days[0])} and {fmt_date(business_days[-1])}. Took {tmx}."
        )
    )
    return result
