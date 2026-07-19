"""
openqai.data.tickers.ticker_calendar
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build a calendar for a given ticker:

* Option Expiries (standard mechanical ones, i.e. monthly, quarterly, EOM)
* Dividends
* Stock splits
* Ticker changes (e.g. name changes, ticker changes, etc.)
* Earnings (from Yahoo Finance, as these are not easily available from other sources)



"""

from openqai.data.tickers import _locations_tickers
from openqai.data.yahoo import YFClient
from openqai.core.biz_calendar import (
    DateRange,
    get_business_days,
    next_business_day,
)
from openqai.data.massive import MassiveClient

from cdxcore.pretty import PrettyObject
from cdxcore.subdir import version, Context, SubDir
from cdxcore.util import fmt_date
import polars as pl
from datetime import date, timedelta
import warnings as warnings
from enum import IntEnum

import numpy as np

class EXPIRY(IntEnum):
    """Expory codes"""

    NONE = 0  # not an expiry
    STANDARD = (
        1  # not one of the mechanical expiries below, but still an expiry (e.g. 0DTE or weeklies)
    )
    QUAD_WITCHING = 2  # third friday of March, June, September, December
    MONTHLY = 3  # third friday of other months
    QUARTERLY = 4  # quarterly expiry, i.e. third friday of March, June, September, December (same as quad witching, but we keep it separate for now in case we want to add more codes later)
    EOM = 5  # end of month expiry (last business day of the month, if not quad witching or monthly)


def _date_code(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


SCHEMA_TICKER_CALENDAR = {
    "date": pl.Date,
    "expiry_code": pl.Int8,
    "is_ex_div": pl.Boolean,
    "is_split": pl.Boolean,
    "has_ticker_change": pl.Boolean,
    "is_earnings": pl.Boolean,
    "is_half_day": pl.Boolean,
}
"""Schema of the :class:`polars.DataFrame` returned by :func:`build_ticker_calendar`."""

def _date_code( d : date ) -> int:
    return d.year*10000 + d.month*100 + d.day

@version("0.1")
def get_standard_expiries(date_range: DateRange) -> pl.DataFrame:
    """
    Computes standard expiry codes for a list of dates.
    Expiry codes computed by this function are mechanical:

    * ``QUAD_WITCHING``: third friday of March, June, September, December
    * ``MONTHLY``: third friday of other months
    * ``EOM``: last business day of the month
    * ``QUARTERLY``: quarterly expiry, i.e. last business day of the month of March, June, September, December

    Parameters
    ----------
    date_range : DateRange
        Date range for which to compute expiry codes.
        The function will only return expiry codes for dates within this range.

    Returns
    -------
    Expiries : :class:`polars.DataFrame`
        DataFrame with columns:
        * ``date``: date of the expiry
        * ``expiry_code``: expiry code, one of the values of the :class:`EXPIRY` enum.
    """

    start_year = date_range.start.year
    end_year = date_range.end.year
    business_days = set(date_range.get_business_days())
    dates = get_business_days(
        date_range.start,
        next_business_day(date_range.end, exclude_half_days=False, count_start=False),
        exclude_half_days=False,
    )  # (this is cached, in case its equal to businbess days)
    datecodes = np.array([_date_code(d) for d in dates], dtype=np.int32)
    assert np.min(np.diff(datecodes)) >= 1, (
        f"Internal error: business days should be at least 1 day apart not {np.min(np.diff(datecodes))}"
    )

    expiries = []
    codes = []

    for year in range(start_year, end_year + 2):
        for month in range(1, 13):
            first_day = date(year, month, 1)

            # get last business of previous month
            last_prev_month = first_day - timedelta(days=1)
            lpm_code = _date_code(last_prev_month)
            ix = np.searchsorted(datecodes, lpm_code, side="right")
            if ix > 0:
                assert lpm_code >= datecodes[ix-1], f"Internal error: lpm_code {lpm_code} should be >= datecodes[{ix-1}]={datecodes[ix-1]}"
                if dates[ix-1] in business_days:
                    assert dates[ix-1] not in expiries, f"Internal error: duplicate expiry date {dates[ix-1]}"
                    expiries.append( dates[ix-1] )
                    codes.append( int( EXPIRY.EOM if last_prev_month.month not in {3, 6, 9, 12} else EXPIRY.QUARTERLY ) )

            # get third friday of this month
            first_friday_offset = (4 - first_day.weekday()) % 7
            third_friday = first_day + timedelta(days=first_friday_offset + 14)
            tf_code = _date_code(third_friday)
            ix = np.searchsorted(datecodes, tf_code, side="right")
            if ix > 0:
                assert tf_code >= datecodes[ix-1], f"Internal error: tf_code {tf_code} should be >= datecodes[{ix-1}]={datecodes[ix-1]}"
                if dates[ix-1] in business_days:
                    assert dates[ix-1] not in expiries, f"Internal error: duplicate expiry date {dates[ix-1]}"
                    expiries.append( dates[ix-1] )
                    codes.append( int( EXPIRY.QUAD_WITCHING if month in {3, 6, 9, 12} else EXPIRY.MONTHLY ) )

            if year==end_year+1:
                # we only had to handle the first month of the next year
                break

    return pl.DataFrame(
        {"date": pl.Series(expiries, dtype=pl.Date), "expiry_code": pl.Series(codes, dtype=pl.Int8)}
    ).sort("date")


@version("0.1", dependencies=[MassiveClient.get_dividends])
def has_dividends(ticker: str, start: date, end: date) -> pl.DataFrame:
    """
    Retrieve dividends between two dates from [``/stocks/v1/dividends``](https://massive.com/docs/api/stocks/v1/dividends/).
    """
    client = MassiveClient()
    df = client.get_dividends(ticker, start, end)
    df = df.rename({"ex_dividend_date": "date"})
    df = df.unique(subset=["date"]).sort("date").with_columns(pl.lit(True).alias("is_ex_div"))
    df = df.select(["date", "is_ex_div"])
    return df


@version("0.1", dependencies=[MassiveClient.get_splits])
def has_splits(ticker: str, start: date, end: date) -> pl.DataFrame:
    """Retrieve stock splits between two dates from [``/stocks/v1/splits``](https://massive.com/docs/api/stocks/v1/splits/)."""
    client = MassiveClient()
    df = client.get_splits(ticker, start, end)
    df = df.rename({"execution_date": "date"})
    df = df.unique(subset=["date"]).sort("date").with_columns(pl.lit(True).alias("is_split"))
    df = df.select(["date", "is_split"])
    return df



@_locations_tickers(fmt=SubDir.PANDAS_PARQUET).cache(
    "0.1",
    dependencies=[
        MassiveClient.get_dividends,
        MassiveClient.get_splits,
        MassiveClient.has_ticker_changes,
        YFClient.has_earnings,
        DateRange,
    ],
    uid=lambda ticker, date_range: f"calendars/{ticker}/{date_range}",
    exclude_args=["verbose"],
)
def build_ticker_calendar(
    ticker: str,
    date_range: DateRange, *,
    verbose: Context = Context.all,
) -> pl.DataFrame:
    """
    Build a per-day ticker-specific calendar for a given equity ticker.

    Parameters
    ----------
    ticker : str
        Equity ticker for which to build the calendar.  
    date_range : :class:`DateRange`
        Date range for which to build the calendar.
    verbose : :class:`cdxcore.verbose.Context`, default ``Context.all``
        Verbose context for logging progress and timing. Default is ``Context.all``.

    Returns
    -------
    Calendar : :class:`polars.DataFrame`
        DataFrame with the schema :data:`SCHEMA_TICKER_CALENDAR` with columns:

        * ``date``: date of the calendar entry
        * ``expiry_code``: expiry code, one of the values of the :class:`EXPIRY` enum, indicating the type of expiry on that date (if any).
        * ``is_ex_div``: boolean indicating whether this date is an ex-dividend date for the ticker.
        * ``is_split``: boolean indicating whether this date is a stock split date for the ticker.
        * ``has_ticker_change``: boolean indicating whether this date has a ticker change event for the ticker.
        * ``is_earnings``: boolean indicating whether this date is an earnings date for the ticker.
        * ``is_half_day``: boolean indicating whether this date is a half trading day (e.g. day before a holiday).
    """
    half_days = date_range.get_half_days()

    dfs = PrettyObject()
    start = date_range.start
    end = date_range.end

    with verbose.write_t(
        lambda: f"Building calendar for {ticker} between {fmt_date(start)} and {fmt_date(end)}:"
    ) as tmx:
        verbose1 = verbose(1)
        if not date_range.exclude_half_days:
            with verbose1.write_t(
                lambda: (
                    f"Retrieving half days for {ticker} between {fmt_date(start)} and {fmt_date(end)}:"
                )
            ) as tme:
                half_days = date_range.get_half_days()
                if not len(half_days) == 0:
                    dfs.half_days = pl.DataFrame(
                        {
                            "date": pl.Series(half_days, dtype=pl.Date),
                            "is_half_day": pl.Series([True] * len(half_days), dtype=pl.Boolean),
                        }
                    )

        with verbose1.write_t(
            lambda: (
                f"Retrieving dividends for {ticker} between {fmt_date(start)} and {fmt_date(end)}... "
            ),
            end="",
        ) as tme:
            dfs.expiries = get_standard_expiries(date_range)
            assert set(dfs.expiries.columns) == {"date", "expiry_code"}, (
                f"Internal error retrieving expiries for {ticker} between {fmt_date(start)} and {fmt_date(end)}: expected columns {{'date', 'expiry_code'}} but got {set(dfs.expiries.columns)}."
            )
            verbose1.write(
                lambda: f"done; {ticker} has {dfs.expiries.height} expiries. This took {tme}.",
                head=False,
            )

        with verbose1.write_t(
            lambda: (
                f"Retrieving dividends for {ticker} between {fmt_date(start)} and {fmt_date(end)}... "
            ),
            end="",
        ) as tme:
            dfs.divs = has_dividends(ticker, start=start, end=end)
            assert set(dfs.divs.columns) == {"date", "is_ex_div"}, (
                f"Internal error retrieving dividends for {ticker} between {fmt_date(start)} and {fmt_date(end)}: expected columns {{'date', 'is_ex_div'}} but got {set(dfs.divs.columns)}."
            )
            verbose1.write(
                lambda: f"done; {ticker} has {dfs.divs.height} dividends. This took {tme}.",
                head=False,
            )

        with verbose1.write_t(
            lambda: (
                f"Retrieving splits for {ticker} between {fmt_date(start)} and {fmt_date(end)}... "
            ),
            end="",
        ) as tme:
            dfs.splits = has_splits(ticker, start=start, end=end)
            assert set(dfs.splits.columns) == {"date", "is_split"}, (
                f"Internal error retrieving splits for {ticker} between {fmt_date(start)} and {fmt_date(end)}: expected columns {{'date', 'is_split'}} but got {set(dfs.splits.columns)}."
            )
            verbose1.write(
                lambda: f"done; {ticker} has {dfs.splits.height} splits. This took {tme}.",
                head=False,
            )

        with verbose1.write_t(
            lambda: (
                f"Retrieving ticker changes for {ticker} between {fmt_date(start)} and {fmt_date(end)}... "
            ),
            end="",
        ) as tme:
            dfs.events = MassiveClient().has_ticker_changes(ticker, start=start, end=end)
            assert set(dfs.events.columns) == {"date", "has_ticker_change"}, (
                f"Internal error retrieving ticker changes for {ticker} between {fmt_date(start)} and {fmt_date(end)}: expected columns {{'date', 'has_ticker_change'}} but got {set(dfs.events.columns)}."
            )
            verbose1.write(
                lambda: f"done; {ticker} has {dfs.events.height} ticker changes. This took {tme}.",
                head=False,
            )

        with verbose1.write_t(
            lambda: (
                f"Retrieving earnings {ticker} between {fmt_date(start)} and {fmt_date(end)}... "
            ),
            end="",
        ) as tme:
            dfs.earnings = YFClient().has_earnings(ticker, start=start, end=end)
            assert set(dfs.earnings.columns) == {"date", "is_earnings"}, (
                f"Internal error retrieving earnings for {ticker} between {fmt_date(start)} and {fmt_date(end)}: expected columns {{'date', 'is_earnings'}} but got {set(dfs.earnings.columns)}."
            )
            verbose1.write(
                lambda: f"done; {ticker} has {dfs.earnings.height} earnings. This took {tme}.",
                head=False,
            )

        del verbose1

        df = None
        for fname, fdata in dfs.items():
            if fdata is None:
                continue

            # Detect duplicate dates in this fdata BEFORE joining
            if fdata.select(pl.col("date").is_duplicated().any()).item():
                dupes = (
                    fdata.filter(pl.col("date").is_duplicated())
                    .select("date")
                    .unique()
                    .sort("date")
                )
                raise RuntimeError(
                    f"{fname} contains duplicate 'date' entries (showing up to 10): "
                    f"{dupes.head(10).to_dict(as_series=False)['date']}"
                )

            df = fdata if df is None else df.join(fdata, on="date", how="full", coalesce=True)

        bool_cols = [col for col, dtype in SCHEMA_TICKER_CALENDAR.items() if dtype == pl.Boolean]
        assert "date" in df.columns, f"Internal error: expected a 'date' column while building the calendar for {ticker}."
        assert set(df.columns) <= set(SCHEMA_TICKER_CALENDAR), ("Internal error: unknonw columns", set(df.columns) - set(SCHEMA_TICKER_CALENDAR) )

        for col in bool_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(False).cast(pl.Boolean).alias(col))
            else:
                # fill bools with False to safe memory
                df = df.with_columns(pl.col(col).fill_null(False).cast(pl.Boolean))

        if "expiry_code" not in df.columns:
            df = df.with_columns(pl.lit(int(EXPIRY.NONE), dtype=pl.Int8).alias("expiry_code"))
        else:
            df = df.with_columns(pl.col("expiry_code").cast(pl.Int8))

        # faster duplicate check than group_by len
        if df.select(pl.col("date").is_duplicated().any()).item():
            dupes = df.filter(pl.col("date").is_duplicated()).select("date").unique()
            warnings.warn(str(dupes.head()))
            raise RuntimeError("Date duplicates found!")

        df = df.select([pl.col(col).cast(dtype) for col, dtype in SCHEMA_TICKER_CALENDAR.items()])

        verbose.write(lambda: f"Built calendar for {ticker} between {fmt_date(start)} and {fmt_date(end)}. This took {tmx}.")

    return df.sort("date")
