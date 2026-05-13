"""
NCDE VoluVol data set generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This file contains then benchmark generator for the Volume & Volality benchmark.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from cdxcore.subdir import SubDir, version, CacheMode
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import fmt_list
from cdxcore.verbose import Context
from cdxcore.jcpool import JCPool

from quantdata.biz_calendar import DateRange, session_open_close, DAYS_PER_YEAR
from quantdata.massive.processing import filter_valid_quotes, filter_valid_trades
from quantdata.ncde.voluvol import _locations_ncde_voluvol
from quantdata.pool import PoolConfig
from dataclasses import dataclass, field
from datetime import date

_cache = _locations_ncde_voluvol.training


@version("0.1")
@dataclass(frozen=True)
class ExperimentScope:
    """
    Definition of the experiment
    """

    tickers             : list[str] = field(default_factory=lambda: ["AAPL"])  # List of tickers to be used for this experiment.
    date_range          : DateRange = DateRange("2025-01-01", "2025-01-31", exclude_half_days=True) # :class:`DateRange` defining the date range for this experiment.
    forecast_horizon_min: float     = 15 # Forecast horizon in minutes for the targets.

    def __post_init__(self):
        assert self.tickers, "Expected at least one ticker"
        assert self.forecast_horizon_min > 0 and self.forecast_horizon_min <= 60*6, "Expected forecast horizon to be positive and less than or equal to 6 hours (a trading day has 6.5 hours)"

    def __str__(self) -> str:
        return f"tck={fmt_list(sorted(self.tickers), link=',')} {self.date_range}"

# -----------------------------------------------------
# utilities
# -----------------------------------------------------

def _add_ms_since_open(df: pl.DataFrame, open_dt) -> pl.DataFrame:
    open_ns = pd.Timestamp(open_dt).value
    df = df.sort("ts_ns").with_columns(
        ((pl.col("ts_ns").cast(pl.Int64) - pl.lit(open_ns, dtype=pl.Int64)) // 1_000_000)
        .cast(pl.Int64)
        .alias("time_of_day_ms")
    )
    df = df.drop("ts_ns")
    return df

def _build_target(trades: pl.DataFrame, forecast_horizon_min: float, close_delta_ms: int) -> pl.DataFrame:
    prices         = trades["price"].to_numpy()
    sizes          = trades["size"].to_numpy()
    time_of_day_ms = trades["time_of_day_ms"].cast(pl.Int64).to_numpy()

    # compute cumumlative volume and cash volume, and variance of log returns
    total_volume = np.cumsum(sizes, dtype=np.float64)
    total_cash_volume = np.cumsum(prices * sizes, dtype=np.float64)
    if len(total_cash_volume) > 0:
        total_cash_volume = total_cash_volume / float(prices[0])

    variance = np.zeros(len(prices), dtype=np.float64)
    if len(prices) > 1:
        log_returns = np.log(prices[1:] / prices[:-1])
        variance[1:] = np.cumsum(log_returns**2, dtype=np.float64)

    # cut off at the forecast horizon
    horizon_ms = int(round(float(forecast_horizon_min) * 60_000))
    start_idx = np.flatnonzero(time_of_day_ms + horizon_ms <= close_delta_ms)
    start_ts = time_of_day_ms[start_idx]
    horizon_end = np.searchsorted(time_of_day_ms, start_ts + horizon_ms, side="right") - 1
    assert len(horizon_end) == len(start_idx), (
        f"Expected horizon_end to have the same length as retained trades, but got {len(horizon_end)} and {len(start_idx)}"
    )
    assert np.all(horizon_end >= start_idx), (
        "Expected horizon_end to be greater than or equal to retained trade indices"
    )

    inc_volume = total_volume[horizon_end] - total_volume[start_idx]
    inc_cash_volume = total_cash_volume[horizon_end] - total_cash_volume[start_idx]
    inc_variance = variance[horizon_end] - variance[start_idx]
    assert np.all( inc_volume >= 0), "Expected incremental volume to be non-negative, but got some negative values"
    assert np.all( inc_cash_volume >= 0), "Expected incremental cash volume to be non-negative, but got some negative values"   
    assert np.all( inc_variance >= 0), "Expected incremental variance to be non-negative, but got some negative values"

    # annualize variance
    # note that we do not scale by actual trading hours (~time_of_day_ms) b
    inc_variance = inc_variance * 6.5 * DAYS_PER_YEAR * 60. / float(forecast_horizon_min)

    targets = pl.DataFrame(
        {
            "time_of_day_ms": start_ts,
            "log_volume": np.log1p(inc_volume),
            "log_cash_volume": np.log1p(inc_cash_volume),
            "volatility": np.sqrt(inc_variance),
        },
        schema={
            "time_of_day_ms": pl.Int64,
            "log_volume": pl.Float64,
            "log_cash_volume": pl.Float64,
            "volatility": pl.Float64,
        },
    )
    return targets.sort("time_of_day_ms")

# -----------------------------------------------------
# utilities
# -----------------------------------------------------

@_cache(fmt=SubDir.PYTREE_HDF5).cache(
    "0.4",
    dependencies=[filter_valid_trades, filter_valid_quotes],
    uid=lambda ticker, d, forecast_horizon_min: f"experiment {forecast_horizon_min}/{ticker}/{d}",
    exclude_args=["pool_config", "verbose"] )
def generate_ticker_day( ticker: str, d: date, forecast_horizon_min: int):
    """
    Generate an experiment entry for a given date ``d`` for a given ``ticker``.
    
    * Converts time stamps into milliseconds since market open.
    * Computes forward floating volume, cash volume and (annualized) volatilities as forecast "targets" for the horizon given by ``forecast_horizon_min``.
    * Drops time stamps within ``forecast_horizon_min`` before the close.

    This function is cached with HDF5 for fast i/o.
    All data are return in 32 bit precision. 

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "AAPL"
    date : :class:`datetime.date`
        Date for which to generate the data, e.g. date(2025, 1, 2)
    forecast_horizon_min : int
        Rolling forecast horizon in minutes, e.g. 15

    Returns
    -------
    Results: dict
        A dictionary containing "trades", "quotes", and forward "targets" for the given ticker and date.    
        The schema are:
        
        * quotes: "time_of_day_ms", "bid_price", "bid_size", "ask_price", "ask_size"
        * trades: "time_of_day_ms", "price", "size"
        * targets: "time_of_day_ms", "log_volume", "log_cash_volume", "volatility". The latter represent the future values from ``time_of_day_ms`` to ``time_of_day_ms + forecast_horizon_min[ms]``.

        The "time_of_day_ms" column represents the time in milliseconds since the market open, and is aligned across trades, quotes, and targets. The "targets" are forward looking values computed at each trade timestamp for the given forecast horizon.

        The "time_of_day_ms" entries are returned as ``int32`` while the other values are returned as ``float32`` for efficient storage and i/o.
    """

    trades = filter_valid_trades(ticker, d, within_session=True)
    quotes = filter_valid_quotes(ticker, d, within_session=True)
    open_dt, close_dt, _ = session_open_close(d)
    close_delta_ms = int((close_dt - open_dt).total_seconds() * 1_000)
    horizon_ms = int(round(float(forecast_horizon_min) * 60_000))

    assert set(trades.columns) == {"ts_ns", "price", "size", "trade_code"}, f"Unexpected columns in trades: {trades.columns}"
    assert set(quotes.columns) == {"ts_ns", "bid_price", "bid_size", "ask_price", "ask_size"}, f"Unexpected columns in quotes: {quotes.columns}"

    quotes  = _add_ms_since_open(quotes, open_dt)
    trades  = _add_ms_since_open(trades, open_dt)
    targets = _build_target(trades, forecast_horizon_min, close_delta_ms)

    quotes = quotes.filter(pl.col("time_of_day_ms") <= close_delta_ms - horizon_ms)
    trades = trades.filter(pl.col("time_of_day_ms") <= close_delta_ms - horizon_ms)
    targets = targets.filter(pl.col("time_of_day_ms") <= close_delta_ms - horizon_ms) # should be NOOP

    quotes = quotes.with_columns(   pl.col("time_of_day_ms").cast(pl.Int32),
                                    pl.col("bid_price").cast(pl.Float32).fill_nan(0.0),
                                    pl.col("bid_size").cast(pl.Float32).fill_nan(0.0),
                                    pl.col("ask_price").cast(pl.Float32).fill_nan(0.0),
                                    pl.col("ask_size").cast(pl.Float32).fill_nan(0.0) )
    trades = trades.with_columns(   pl.col("time_of_day_ms").cast(pl.Int32),
                                    pl.col("price").cast(pl.Float32).fill_nan(0.0)  , 
                                    pl.col("size").cast(pl.Float32).fill_nan(0.0) ).drop("trade_code")
    targets = targets.with_columns( pl.col("time_of_day_ms").cast(pl.Int32),
                                    pl.col("log_volume").cast(pl.Float32).fill_nan(0.0),
                                    pl.col("log_cash_volume").cast(pl.Float32).fill_nan(0.0),
                                    pl.col("volatility").cast(pl.Float32).fill_nan(0.0) )

    # turn into numpy arrays for fast i/o
    def assert_no_infinite(x):
        assert np.isfinite(x).all(), "Expected all values to be finite, but got some infinite values"
        return x
    quotes  = {col: assert_no_infinite(quotes[col].to_numpy()) for col in quotes.columns}
    trades  = {col: assert_no_infinite(trades[col].to_numpy()) for col in trades.columns}
    targets = {col: assert_no_infinite(targets[col].to_numpy()) for col in targets.columns}
    return {"trades": trades, "quotes": quotes, "targets": targets}

@_cache(fmt=SubDir.PICKLE).cache(
    "0.2",
    dependencies=[generate_ticker_day, ExperimentScope],
    uid = lambda experiment_scope : f"experiment {experiment_scope.forecast_horizon_min}/gen {named_unique_filename48_8(f'{experiment_scope}', experiment_scope)}",
    exclude_args=["pool_config", "verbose"],
)
def _generate_experiment_sample_set(
    experiment_scope: ExperimentScope,
    *,
    pool_config: PoolConfig = PoolConfig(),
    override_cache_mode : CacheMode|str|None = None,
    verbose: Context = Context.all,
) -> dict[str, Any]:
    """
    Generate all samples for a given experiment.

    This main purpose of this function is to parallelize the generation across tickers and dates, and to cache the entire generated data set for fast i/o in the experiments.
    The dummy return value by this function is cached; its version control ensures that the underlying data is re-computed if necessary.

    Parameters
    ----------
    experiment_scope : :class:`ExperimentScope`
        Defines the experiment settings.
    pool_config : :class:`PoolConfig
        Configuration for the parallel pool used to generate the data. Set ``num_workers=0`` to disable all parallelism (e.g. to debug).
    override_cache_mode : CacheMode | str | None, default ``None``
        Allows forcing a caching mode. Useful settings:

        * ``update``: forces re-generation if the immediate cache (i.e. :func:`generate_ticker_day`). Does not trigger an update of the cached raw data from Massive.
        * ``cacheonly``: forces loading from cache, and raises an error if the cache is not available. Useful to verify that the data can be loaded from cache without triggering a re-generation.

        You can trigger a global update of all data by setting the ``QUANTDATA_CACHE_MODE`` environment variable; c.f. :class:`quantdata.Environment`.
    verbose : :class:`Context`, default ``Context.all``
        Verbose context for logging and timing. The total time taken to generate the data is logged at the end of the generation process.
        Set to ``Context.quiet`` to turn off all messaging.        
    """

    dates = experiment_scope.date_range.get_business_days()
    forecast_horizon_min = experiment_scope.forecast_horizon_min
    tickers = sorted(experiment_scope.tickers)

    counter = { ticker : dict() for ticker in tickers }

    with verbose.timer() as tme:
        pool = JCPool( pool_config.num_workers, threading=False )

        def process( ticker, date, forecast_horizon_min, override_cache_mode):
            r   = generate_ticker_day( ticker, date, forecast_horizon_min, override_cache_mode=override_cache_mode )
            cnt = {c: len(d['time_of_day_ms']) for c, d in r.items()}
            return ticker, date, cnt

        total = len(tickers) * len(dates)
        cnt   = 0
        verbose.write(lambda: f"\rGenerating {total} samples for {len(tickers)} tickers and {len(dates)} days from {dates[0]} to {dates[-1]}{' excluding half days' if experiment_scope.date_range.exclude_half_days else ''}... ", end='')
        for ticker, date, td_cnt in pool.parallel( pool.delayed(process)(ticker, date, forecast_horizon_min, override_cache_mode) for ticker in tickers for date in dates ):
            counter[ticker][date] = td_cnt
            cnt +=1 
            verbose.write(lambda: f"\rGenerated data for {ticker} for {date}. Done {cnt}/{total} ({cnt/total*100:.2f}%). So far this took {tme}... ", end='')        
        verbose.write(lambda: f"\rGenerated data for all tickers for {len(dates)} days from {dates[0]} to {dates[-1]}{' excluding half days' if experiment_scope.date_range.exclude_half_days else ''}. This took {tme}.")

    return counter

@dataclass
class SampleID:
    """ Identifier for a sample, consisting of a ticker and a date. """
    ticker: str  
    date: date

    def __str__(self) -> str:
        return f"{self.ticker} {self.date}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SampleID):
            return False
        return self.ticker == other.ticker and self.date == other.date
    
    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, SampleID):
            return NotImplemented
        if self.ticker != other.ticker:
            return self.ticker < other.ticker
        return self.date < other.date
    
    def __hash__(self) -> int:
        return hash((self.ticker, self.date))

class ExperimentLoader(object):
    """
    Utility class to load experiment data.

    You can generate a specific sample by using :func:`get_sample` or ``[]``.

    Parameters
    ----------
    experiment_scope : :class:`ExperimentScope`
        Defines the experiment settings.
    pool_config : :class:`PoolConfig
        Configuration for the parallel pool used to generate the data. Set ``num_workers=0`` to disable all parallelism (e.g. to debug).
    override_cache_mode : CacheMode | str | None, default ``None``
        Allows forcing a caching mode. Useful settings:

        * ``update``: forces re-generation if the immediate cache (i.e. :func:`generate_ticker_day`). Does not trigger an update of the cached raw data from Massive.
        * ``cacheonly``: forces loading from cache, and raises an error if the cache is not available. Useful to verify that the data can be loaded from cache without triggering a re-generation.

        You can trigger a global update of all data by setting the ``QUANTDATA_CACHE_MODE`` environment variable; c.f. :class:`quantdata.Environment`.
    verbose : :class:`Context`, default ``Context.all``
        Verbose context for logging and timing. The total time taken to generate the data is logged at the end of the generation process.
        Set to ``Context.quiet`` to turn off all messaging.
    """

    def __init__(self,  experiment_scope: ExperimentScope, *,
                        pool_config: PoolConfig = PoolConfig(),
                        override_cache_mode : CacheMode|str|None = None,
                        verbose: Context = Context.all):
        """ init """
        self.experiment_scope = experiment_scope
        counters = _generate_experiment_sample_set(experiment_scope, pool_config=pool_config, override_cache_mode=override_cache_mode, verbose=verbose)

        self._samples : dict[SampleID, dict[str, int]]= {} # 
        for ticker in sorted(experiment_scope.tickers):
            dates = sorted(counters[ticker])
            for date in dates:
                cnt = counters[ticker][date]
                self._samples[SampleID(ticker=ticker, date=date)] = cnt

    @property
    def samples(self) -> dict[SampleID, dict[str, int]]:
        """ Returns a dictionary of samples, indexed by :class:`SampleID`, each containing 
        a dictionary with the counts of ``trades``, ``quotes``, and ``targets``. """
        return self._samples
    
    def __len__(self) -> int:
        """ Returns the number of samples in this experiment. """
        return len(self._samples)
    
    def __getitem__(self, sample_id: SampleID|int) -> dict[str, int]:
        """ Get the counts of trades, quotes, and targets for a given sample id. """
        return self.get_sample(sample_id, cache_only=True)

    def get_sample_id( self, sample_id: SampleID|int) -> SampleID:
        """ Return the :class:`SampleID` for a given sample id, typically for an integer index. """
        if isinstance(sample_id, int):
            sample_id = list(self._samples.keys())[sample_id]
        return sample_id

    def get_sample(self, sample_id: SampleID|int, *, cache_only : bool = False ) -> dict[str, Any]:
        """ Load the data for a given sample id. The data is returned as a dictionary containing "trades", "quotes", and "targets" for the given ticker and date. The schema are:

        * quotes: "time_of_day_ms", "bid_price", "bid_size", "ask_price", "ask_size"
        * trades: "time_of_day_ms", "price", "size"
        * targets: "time_of_day_ms", "log_volume", "log_cash_volume", "volatility". The latter represent the future values from ``time_of_day_ms`` to ``time_of_day_ms + forecast_horizon_min[ms]``.

        The "time_of_day_ms" column represents the time in milliseconds since the market open, and is aligned across trades, quotes, and targets. The "targets" are forward looking values computed at each trade timestamp for the given forecast horizon.

        The "time_of_day_ms" entries are returned as ``int32`` while the other values are returned as ``float32`` for efficient storage and i/o.

        Parameters
        ----------
        sample_id : :class:`SampleID` | int
            Either the sample id containing the ``ticker`` and ``date`` for which to load the data, or an integer index.
        cache_only : bool, optional
            If True, only load the data from the cache, if available. Default is False.

        Returns
        -------
        Result : dict
            See above.
        """
        if isinstance(sample_id, int):
            sample_id = list(self._samples.keys())[sample_id]
        return generate_ticker_day(sample_id.ticker, sample_id.date, self.experiment_scope.forecast_horizon_min, override_cache_mode="cacheonly" if cache_only else None)


