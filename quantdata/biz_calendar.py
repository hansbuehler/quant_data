"""
Business day calendar utilities.

Uses pandas_market_calendars to get NYSE trading days and identify
early-close (half) days.
"""

from __future__ import annotations

import quantdata as quantdata#NOQA

from cdxcore.version import version
from cdxcore.err import verify_inp
import datetime as dt
import numpy as np
import pandas_market_calendars as mcal
from zoneinfo import ZoneInfo
import math as math
from dataclasses import dataclass

TZ_NY = ZoneInfo("America/New_York")
""" New York time zone """

DAYS_PER_YEAR = 252
""" Standardized business days per year """

AM_EXIRY_SUBTRACT = 0.2
""" How much to subtract from 'days' to expiry to account for AM settlement """

NYSE_CALENDAR = mcal.get_calendar("NYSE")
""" NYSE calendar provided by `pandas market calendars <https://pandas-market-calendars.readthedocs.io/en/latest/usage.html>`__ """

class NMS:
    r"""
    NMS(z,t) := exp( scale * z * sqrt((eps+t)/(eps+1))

    Scale is standardized to z=1. being +10% at 1 year at log(1.1) ~ 10% volatility
    """

    EPS = 1./255./8. #: Small epsilon for NMS computation.
    SCALE = math.log(1.1) # 10% vol

    @staticmethod
    def from_nms(z: float, t: float, backbone = np ) -> float:
        return backbone.exp( NMS.SCALE * z * backbone.sqrt( (NMS.EPS + t) / (NMS.EPS + 1.) ) )
    
    @staticmethod
    def to_nms(k: float, t: float, backbone = np ) -> float:
        return (backbone.log(k) / NMS.SCALE) / backbone.sqrt( (NMS.EPS + t) / (NMS.EPS + 1.) )

def to_date( d: str | dt.datetime | dt.date | np.datetime64 | None ) -> dt.date | None:
    """ Convert ``d`` to a :class:`datetime.date`, or ``None`` if ``d`` is ``None``. """
    if d is None:
        return d
    if isinstance(d, np.datetime64):
        d = d.item()
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    if isinstance(d, str):
        return dt.date.fromisoformat(d)
    raise ValueError(f"Unsupported type for date conversion: {type(d)}")

@version("0.1")
def get_business_days( start : str|dt.date, end: str|dt.date, exclude_half_days: bool = True ) -> list[dt.date]:
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
    exclude_half_days : bool, default ``True``
        If True, remove early-close trading sessions from the calendar.

    Returns
    -------
    list[:class:`datetime.date`]
        Sorted trading dates (timezone-naive, date-only precision).

    -------
    pd.DatetimeIndex
        Sorted trading dates (timezone-naive, date-only precision).
    """
    start             = to_date(start)
    end               = to_date(end)
    exclude_half_days = bool(exclude_half_days)

    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    schedule = NYSE_CALENDAR.schedule(start_date=str(start), end_date=str(end))

    if exclude_half_days and len(schedule) > 0:
        early = NYSE_CALENDAR.early_closes(schedule)
        schedule = schedule.drop(early.index, errors="ignore")

    schedule = schedule.index.normalize().tz_localize(None)  # index is a DatetimeIndex of trading days in the range
    schedule = [dt.date(d.year,d.month,d.day) for d in schedule]
    return schedule

@version("0.1", dependencies=[get_business_days])
def count_business_days_between(
    start: dt.date | np.ndarray,
    end: dt.date | np.ndarray,
    exclude_half_days : bool = False,
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
    start    = to_date(start) if not isinstance(start, np.ndarray) else start
    end      = to_date(end) if not isinstance(end, np.ndarray) else end
    is_vec   = isinstance(start, np.ndarray) or isinstance(end, np.ndarray)
    start    = np.asarray(start, dtype="datetime64[D]") 
    end      = np.asarray(end, dtype="datetime64[D]") 

    mstart       = min(np.min(start), np.min(end))
    mend         = max(np.max(start), np.max(end))
    trading_days = get_business_days( mstart, mend, exclude_half_days=exclude_half_days )

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
def add_business_days( start: dt.date, num_days: int, exclude_half_days: bool = False ) -> dt.date:
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
        If True, remove early-close trading sessions from the calendar before counting.

    Returns
    -------
    ``datetime.date``
        Resulting date after adding the specified number of business days.
    """
    if isinstance(start, np.ndarray):
        start    = np.asarray(start, dtype="datetime64[D]")
        if isinstance(num_days, np.ndarray):
            verify_inp( start.shape == num_days.shape, f"start and num_days must have the same shape, but got {start.shape} and {num_days.shape}" )
            return np.asarray( [ add_business_days(s, n, exclude_half_days=exclude_half_days) for s, n in zip(start, num_days) ], dtype="datetime64[D]" )
        return np.asarray( [ add_business_days(s, num_days, exclude_half_days=exclude_half_days) for s in start ], dtype="datetime64[D]" )

    if isinstance(num_days, np.ndarray):
        return np.asarray( [ add_business_days(start, n, exclude_half_days=exclude_half_days) for n in num_days], dtype="datetime64[D]" )

    # scalar version
    start    = to_date(start)
    num_days = int(num_days)

    if num_days >= 0:
        schedule = get_business_days( start, start + dt.timedelta(days=num_days*2+5), exclude_half_days=exclude_half_days   )
        assert schedule[0] >= start, f"Schedule should start on or after {start}, but starts at {schedule[0]}"
        if not start == schedule[0]:
            raise ValueError(f"'start' {start} is not a{' full' if exclude_half_days else ''} business day")
        return schedule[num_days]
    
    else:
        schedule = get_business_days( start + dt.timedelta(days=num_days*2-5), start, exclude_half_days=exclude_half_days )
        assert schedule[-1] <= start, f"Schedule should end on or before {start}, but ends at {schedule[-1]}"
        if not start == schedule[-1]:
            raise ValueError(f"'start' {start} is not a{' full' if exclude_half_days else ''} business day")
        return schedule[len(schedule)-1+num_days]

@dataclass(frozen=True)
class DateRange:
    """ A simple date range class. """
    start: dt.date
    end: dt.date
    exclude_half_days: bool = True

    def __post_init__(self):
        verify_inp( isinstance(self.start, dt.date), f"'start' must be a datetime.date, but found {type(self.start)}" )
        verify_inp( isinstance(self.end, dt.date), f"'end' must be a datetime.date, but found {type(self.end)}" )

        if self.start > self.end:
            raise ValueError(f"'start' ({self.start}) must be at or before 'end' ({self.end})")
        
        bdays = self.get_business_days()
        if len(bdays) == 0:
            raise ValueError(f"No business days in the range {self.start} to {self.end} with half days {'excluded' if self.exclude_half_days else 'included'}")

    def __str__(self):
        return f"{self.start}_{self.end}{'x' if self.exclude_half_days else 'i'}"
    
    def get_business_days(self) -> list[dt.date]:
        """ Get business days in this date range. """
        return get_business_days(self.start, self.end, exclude_half_days=self.exclude_half_days)
    
