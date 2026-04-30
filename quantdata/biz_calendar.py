"""
Business day calendar utilities.

Uses pandas_market_calendars to get NYSE trading days and identify
early-close (half) days.
"""

from __future__ import annotations

import datetime as dt
import math as math
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas_market_calendars as mcal
import polars as pl
from cdxcore.err import verify_inp
from cdxcore.version import version

import quantdata as quantdata  # NOQA

TZ_NY = ZoneInfo("America/New_York")
""" New York time zone """

DAYS_PER_YEAR = 252
""" Standardized business days per year """

AM_EXIRY_SUBTRACT = 0.2
""" How much to subtract from 'days' to expiry to account for AM settlement """

NYSE_CALENDAR = mcal.get_calendar("NYSE")
""" NYSE calendar provided by `pandas market calendars <https://pandas-market-calendars.readthedocs.io/en/latest/usage.html>`__ """

_BUSINESS_DAYS_CACHE = {}
_EARLY_DAYS_CACHE = {}


class NMS:
    r"""
    NMS(z,t) := exp( scale * z * sqrt((eps+t)/(eps+1))

    Scale is standardized to z=1. being +10% at 1 year at log(1.1) ~ 10% volatility
    """

    EPS = 1.0 / 255.0 / 8.0  #: Small epsilon for NMS computation.
    SCALE = math.log(1.1)  # 10% vol

    @staticmethod
    def from_nms(z: float, t: float, backbone=np) -> float:
        return backbone.exp(NMS.SCALE * z * backbone.sqrt((NMS.EPS + t) / (NMS.EPS + 1.0)))

    @staticmethod
    def to_nms(k: float, t: float, backbone=np) -> float:
        return (backbone.log(k) / NMS.SCALE) / backbone.sqrt((NMS.EPS + t) / (NMS.EPS + 1.0))


def to_date(d: str | dt.datetime | dt.date | np.datetime64 | None) -> dt.date | None:
    """Convert ``d`` to a :class:`datetime.date`, or ``None`` if ``d`` is ``None``."""
    if d is None:
        return d
    if isinstance(d, np.datetime64):
        d = d.astype("datetime64[D]").astype(dt.date)
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    if isinstance(d, str):
        return dt.date.fromisoformat(d)
    raise ValueError(f"Unsupported type for date conversion: {type(d)}")


@version("0.1")
def get_business_days(
    start: str | dt.date, end: str | dt.date, exclude_half_days: bool = False
) -> list[dt.date]:
    """
    Return NYSE trading days within start and end.

    This function returns business days inclusive of ``start`` and ``end``. That means that if ``start`` equals ``end``, and if that is a trading day,
    then this function returns that day.

    Parameters
    ----------
    start : str | :class:`datetime.date`
        Start date (inclusive).
    end : str | :class:`datetime.date`
        End date (inclusive).
    exclude_half_days : bool, default ``False``
        If True, remove early-close trading sessions from the calendar.

    Returns
    -------
    list[:class:`datetime.date`]
        Sorted trading dates (timezone-naive, date-only precision).

    -------
    pd.DatetimeIndex
        Sorted trading dates (timezone-naive, date-only precision).
    """
    global _BUSINESS_DAYS_CACHE
    start = to_date(start)
    end = to_date(end)
    exclude_half_days = bool(exclude_half_days)

    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    key = (start, end, exclude_half_days)
    schedule = _BUSINESS_DAYS_CACHE.get(key, None)
    if schedule is not None:
        return schedule

    schedule = NYSE_CALENDAR.schedule(start_date=str(start), end_date=str(end))

    if exclude_half_days and len(schedule) > 0:
        early = NYSE_CALENDAR.early_closes(schedule)
        schedule = schedule.drop(early.index, errors="ignore")

    schedule = schedule.index.normalize().tz_localize(
        None
    )  # index is a DatetimeIndex of trading days in the range
    schedule = [dt.date(d.year, d.month, d.day) for d in schedule]

    _BUSINESS_DAYS_CACHE[key] = schedule
    return schedule


@version("0.1", dependencies=[get_business_days])
def count_business_days_between(
    start: dt.date | np.ndarray,
    end: dt.date | np.ndarray,
    exclude_half_days: bool = False,
) -> int | np.ndarray:
    """
    Count the number of trading days in ``(start, end]``, i.e. from ``start`` (exclusive) to ``end`` (inclusive).
    If ``end`` < ``start``, then count the number of trading days in ``[end, start)`` and return the negative of that count.

    The count *includes* ``end`` if it is a trading day, but *excludes* ``start``, assuming, essentially, an end-of-day convention.
    Half days are counted as full trading days unless ``exclude_half_days`` is ``True``.

    Parameters
    ----------
    start : ``datetime.date`` | :class:`numpy.ndarray` of dates
        Start date(s) (exclusive).
    end : ``datetime.date`` | :class:`numpy.ndarray` of dates
        End date(s) (inclusive).
    exclude_half_days : bool, default ``False``
        If True, remove early-close trading sessions from the calendar before counting.

    Returns
    -------
    Count: ``int`` or :class:`numpy.ndarray`
        Number of trading days.
    """
    start = to_date(start) if not isinstance(start, np.ndarray) else start
    end = to_date(end) if not isinstance(end, np.ndarray) else end
    is_vec = isinstance(start, np.ndarray) or isinstance(end, np.ndarray)
    start = np.asarray(start, dtype="datetime64[D]")
    end = np.asarray(end, dtype="datetime64[D]")

    mstart = min(np.min(start), np.min(end))
    mend = max(np.max(start), np.max(end))
    trading_days = get_business_days(mstart, mend, exclude_half_days=exclude_half_days)

    if not is_vec:
        # Scalar case: count how many trading days are in (start, end]
        if start <= end:
            count = np.sum((trading_days > start) & (trading_days <= end))
        else:
            count = -np.sum((trading_days >= end) & (trading_days < start))
        return int(count)

    # Vectorised case: for each pair of start[i], end[i], count trading days in (start[i], end[i]]
    td_array = np.asarray(trading_days, dtype="datetime64[D]")
    lo = np.searchsorted(td_array, start, side="right")
    hi = np.searchsorted(td_array, end, side="right")
    return (hi - lo).astype(np.int32)


@version("0.1", dependencies=[get_business_days])
def add_business_days(
    start: dt.date, num_days: int, exclude_half_days: bool = False, *, count_start: bool = False
) -> dt.date:
    """
    Add a number of business days to a date.

    Parameters
    ----------
    start : ``datetime.date``
        Start date. This function raises a :class:`ValueError` if ``start`` is not a business day, or if
        it is a half business day and ``exclude_half_days`` is ``False``.
    num_days : int
        Number of business days to add. Can be negative to subtract business days.
    exclude_half_days : bool, default ``False``
        If ``True``, remove early-close trading sessions from the calendar before counting.
    count_start : bool, default ``False``
        Whether to count ``start`` if it is a business day. If ``False``, then the counts starts from the first business day after start.
        This flag has no effect if ``start`` is a business day.

        If ``start`` is not a business day, and if ``count_start`` is ``True``, then:

        * If ``num_days=1`` then the function will the return the first business day following the non-business day ``start``.
        * If ``num_days=-1`` then the function will return the first business day preceding the non-business day ``start``.

    Returns
    -------
    ``datetime.date``
        Resulting date after adding the specified number of business days.
    """
    if isinstance(start, np.ndarray):
        start = np.asarray(start, dtype="datetime64[D]")
        if isinstance(num_days, np.ndarray):
            verify_inp(
                start.shape == num_days.shape,
                f"start and num_days must have the same shape, but got {start.shape} and {num_days.shape}",
            )
            return np.asarray(
                [
                    add_business_days(s, n, exclude_half_days=exclude_half_days)
                    for s, n in zip(start, num_days)
                ],
                dtype="datetime64[D]",
            )
        return np.asarray(
            [add_business_days(s, num_days, exclude_half_days=exclude_half_days) for s in start],
            dtype="datetime64[D]",
        )

    if isinstance(num_days, np.ndarray):
        return np.asarray(
            [add_business_days(start, n, exclude_half_days=exclude_half_days) for n in num_days],
            dtype="datetime64[D]",
        )

    # scalar version
    start = to_date(start)
    num_days = int(num_days)

    if num_days >= 0:
        schedule = get_business_days(
            start,
            start + dt.timedelta(days=(num_days // 5) * 7 + 7),
            exclude_half_days=exclude_half_days,
        )
        assert len(schedule) > 0, "Internal error - this should never happen"
        assert schedule[0] >= start, (
            f"Schedule should start on or after {start}, but starts at {schedule[0]}"
        )
        if start == schedule[0] and count_start:
            # start is a business day
            num_days = max(0, num_days - 1)  # count start, so we need to subtract 1 from num_days
        return schedule[num_days]

    else:
        schedule = get_business_days(
            start + dt.timedelta(days=-(abs(num_days) // 5 * 7 + 7)),
            start,
            exclude_half_days=exclude_half_days,
        )
        assert len(schedule) > 0, "Internal error - this should never happen"
        assert schedule[-1] <= start, (
            f"Schedule should end on or before {start}, but ends at {schedule[-1]}"
        )
        schedule.reverse()  # reverse to count backwards
        num_days = -num_days
        if start == schedule[0] and count_start:
            # start is a business day
            num_days = max(0, num_days - 1)  # count start, so we need to subtract 1 from num_days
        return schedule[num_days]


@version("0.1", dependencies=[add_business_days])
def prev_business_day(
    date: dt.date, exclude_half_days: bool = False, *, count_start: bool = False
) -> dt.date:
    """
    Get the strictly previous business day before the given date.

    Parameters
    ----------
    date : ``datetime.date``
        Reference date. This date does not have to be a business day.
    exclude_half_days : bool, default ``False``
        If True, remove early-close trading sessions from the calendar before counting.
    count_start : bool, default ``False``
        Whether to count ``date`` if it is a business day. If ``False``, then the counts starts from the first business day before date.
        That means even if ``date`` is a business day, the function will return the previous business day before it.
        If ``True``, then if ``date`` is a business day, the function will return that date.

        This flag has no effect if ``date`` is a business day.

    Returns
    -------
    ``datetime.date``
        Previous business day before the given date.
    """
    return add_business_days(date, -1, exclude_half_days=exclude_half_days, count_start=count_start)


@version("0.1", dependencies=[add_business_days])
def next_business_day(
    date: dt.date, exclude_half_days: bool = False, *, count_start: bool = False
) -> dt.date:
    """
    Get the strictly next business day after the given date.

    Parameters
    ----------
    date : ``datetime.date``
        Reference date. This date does not have to be a business day.
    exclude_half_days : bool, default ``False``
        If True, remove early-close trading sessions from the calendar before counting.
    count_start : bool, default ``False``
        Whether to count ``date`` if it is a business day. If ``False``, then the counts starts from the first business day after date.
        That means even if ``date`` is a business day, the function will return the next business day after it.
        If ``True``, then if ``date`` is a business day, the function will return that date.
        This flag has no effect if ``date`` is a business day.

    Returns
    -------
    ``datetime.date``
        Next business day after the given date.
    """
    return add_business_days(date, 1, exclude_half_days=exclude_half_days, count_start=count_start)


@version("0.1")
def get_half_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """
    Return a list of half days (early closes) between start and end, inclusive.
    """
    global _EARLY_DAYS_CACHE

    start = to_date(start)
    end = to_date(end)
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    key = (start, end)
    schedule = _EARLY_DAYS_CACHE.get(key, None)
    if schedule is not None:
        return schedule

    schedule = NYSE_CALENDAR.schedule(start_date=str(start), end_date=str(end))
    schedule = NYSE_CALENDAR.early_closes(schedule=schedule)
    schedule = schedule.index.normalize().tz_localize(
        None
    )  # index is a DatetimeIndex of trading days in the range
    schedule = [dt.date(d.year, d.month, d.day) for d in schedule]
    _EARLY_DAYS_CACHE[key] = schedule
    return schedule


@version("0.1", dependencies=[get_business_days, get_half_days])
@dataclass
class DateRange:
    """A simple date range class."""

    start: dt.date | str  #: Start date (inclusive).
    end: dt.date | str  #: End date (inclusive).
    exclude_half_days: bool = True  #: Whether to exclude half days (early closes) from the business day count. Default is True.

    def __post_init__(self):
        if isinstance(self.start, str):
            self.start = dt.date.fromisoformat(self.start)
        if isinstance(self.end, str):
            self.end = dt.date.fromisoformat(self.end)
        verify_inp(
            isinstance(self.start, dt.date),
            f"'start' must be a datetime.date, but found {type(self.start)}",
        )
        verify_inp(
            isinstance(self.end, dt.date),
            f"'end' must be a datetime.date, but found {type(self.end)}",
        )

        if self.start > self.end:
            raise ValueError(f"'start' ({self.start}) must be at or before 'end' ({self.end})")

        bdays = self.get_business_days()
        if len(bdays) == 0:
            raise ValueError(
                f"No business days in the range {self.start} to {self.end} with half days {'excluded' if self.exclude_half_days else 'included'}"
            )

    def __str__(self):
        return f"{self.start}_{self.end}{'x' if self.exclude_half_days else 'i'}"

    def __hash__(self):
        return hash((self.start, self.end, self.exclude_half_days))

    def get_business_days(self) -> list[dt.date]:
        """Get business days in this date range."""
        return get_business_days(self.start, self.end, exclude_half_days=self.exclude_half_days)

    def get_half_days(self) -> list[dt.date]:
        """Get half days (early closes) in this date range."""
        return get_half_days(self.start, self.end) if not self.exclude_half_days else []


@version("0.1", dependencies=[get_half_days])
def session_open_close(d: dt.date) -> tuple[dt.datetime, dt.datetime, int, bool]:
    """
    NYSE RTH session open and close for date ``d``.

    Returns
    -------
    (open_dt, close_dt, session_seconds, is_half_day)
        * open_dt: 09:30 NY
        * close_dt: 13:00 NY on half-days, 16:00 NY otherwise
        * session_seconds: total session length in seconds
        * is_half_day: whether this is an early-close day
    """
    is_hd = bool(get_half_days(d, d))
    open_dt = dt.datetime(d.year, d.month, d.day, 9, 30, tzinfo=TZ_NY)
    close_dt = (
        dt.datetime(d.year, d.month, d.day, 13, 0, tzinfo=TZ_NY)
        if is_hd
        else dt.datetime(d.year, d.month, d.day, 16, 0, tzinfo=TZ_NY)
    )
    session_seconds = int((close_dt - open_dt).total_seconds())
    return open_dt, close_dt, session_seconds, is_hd


@version("0.1")
def full_second_grid_ns(open_dt: dt.datetime, close_dt: dt.datetime) -> pl.Series:
    """
    Per-second Polars Series for RTH bins: ``[open_dt, open_dt+1s, ..., close_dt-1s]``.

    The grid is end-exclusive so the last bin key is ``close_dt - 1s`` and represents
    ``[close_dt-1s, close_dt]`` (inclusive close handled by timestamp mapping in the caller).
    """
    s = pl.datetime_range(
        open_dt,
        close_dt,
        interval="1s",
        time_zone=TZ_NY.key,
        eager=True,
        closed="left",
    )
    return s.dt.cast_time_unit("ns").rename("dt_ns")
