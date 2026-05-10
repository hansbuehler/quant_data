"""
Simple data for SANOS
"""

from __future__ import annotations

from quantdata import env#NOQA
from quantdata.dysanos.simple import locations_simple
from quantdata.widb.arbfree import SCHEMA_ARB_FREE_DAY_EXPIRY, SCHEMA_ARB_FREE_DAY_SURFACE
from quantdata.widb.timeseries import MkArbFree, build_time_series, TimeseriesConfig, RawDayFilter, ExpiryFilter, SCHEMA_ENRICHED_DAY#NOQA
from quantdata.biz_calendar import DateRange
from quantdata.pool import PoolConfig
from quantdata.torch import BaseSamplerConfig, BaseSampler

from cdxcore.version import version
from cdxcore.verbose import Context
from cdxcore.util import fmt_date, fmt_dict, fmt_list
from cdxcore.err import verify_inp
from cdxcore.pretty import PrettyObject
from cdxcore.subdir import SubDir, CacheMode
from cdxcore.uniquehash import named_unique_filename48_8, unique_hash16
from cdxcore.jcpool import JCPool

training_root_simple = locations_simple.training

import datetime as dt
import numpy as np
import math as math
from dataclasses import dataclass, field    
from collections.abc import Sequence

@version("0.1")
@dataclass(frozen=True)
class SamplingConfig:
    """
    Sampling configuration for :class:`SimpleSampler`.
    
    This configuration drives the number of (time) steps per sample, the slide applied (e.g. how man days ahead the next sample is), and the validation split.
    """

    num_steps: int = 25              #: Steps per sample, including terminal 'y'
    num_slide: int = 5               #: Number of steps to slide for the next sample. If num_slide < num_steps, samples will overlap.
    pct_validation: float = 0.2      #: Percentage of samples to use for validation.
    seed : int = 42                  #: Random seed for reproducibility.

    def __post_init__(self):
        verify_inp( 0.0 < self.pct_validation < 1.0, "'pct_validation' must be in (0,1)" )   
        verify_inp( self.num_steps >= 2, "'num_steps' must be at least 2, as the last step is the prediction target" )
        verify_inp( self.num_slide > 0, "'num_slide' must be positive" )

    def __str__(self):#by convention use a some what readable string with comma separated key=value pairs
        return f"stp={self.num_steps},sld={self.num_slide},vld={self.pct_validation:.2f},seed={self.seed}"

@version("0.", dependencies=[RawDayFilter,ExpiryFilter,TimeseriesConfig,SamplingConfig])
@dataclass(frozen=True)
class SimpleConfig(BaseSamplerConfig):
    """
    Master Configuration

    All-in-one configuration for the simple sampler, which includes all the necessary information to generate the samples,
    including the filters and configurations for the time series generation, as well as the sampling configuration.
    This is used both for generating the samples and for caching, ensuring that any change in the configuration will
    lead to a different cache key and thus a different set of samples.
    """
    tickers           : list[str] = field(default_factory=lambda: ["SPY"])  #: Tickers to include in the sampler. Can be a single ticker as a string, or a list of tickers.
    date_range        : DateRange = field(default_factory=lambda: DateRange(start=dt.date(2025, 1, 1), end=dt.date(2025, 2, 1), exclude_half_days=True))  # which dates
    raw_day_filter    : RawDayFilter = field(default_factory=lambda: RawDayFilter())  #: Filter for raw days, applied before sampling.
    expiry_filter     : ExpiryFilter = field(default_factory=lambda: ExpiryFilter())  #: Filter for option expiries, applied before sampling.
    mk_arb_free       : MkArbFree = MkArbFree(no_arb_mode="expiry") #: Whether to use the arbitrage-free version of the data, and if so which version (expiry or surface).
    timeseries_config : TimeseriesConfig = field(default_factory=lambda: TimeseriesConfig())  #: Configuration for sampling days, applied before sampling.
    sampling_config   : SamplingConfig = field(default_factory=lambda: SamplingConfig())  #: Configuration for sampling, applied after the above filters.   

    def __post_init__(self):
        verify_inp( isinstance(self.tickers, Sequence) and len(self.tickers) > 0, "'tickers' must be a list of strings" )
        verify_inp( len(set(self.tickers)) == len(self.tickers), "'tickers' must not contain duplicates" )

    @property
    def no_arb_mode(self) -> str|None:
        """ Returns ``None`` if no arbitrage pre-processing is applied, or the mode of arbitrage pre-processing (e.g. ``"expiry"`` or ``"surface"``) if it is applied. """
        return self.mk_arb_free.no_arb_mode if self.mk_arb_free is not None else None
    
    def timeseries_fields(self) -> set:
        """ Returs the fields to be requested from the time series (depends on the arbitrage pre-processing mode). """
        if self.no_arb_mode == "expiry":
            return SIMPLE_INCLUDE_FIELDS_NOARB_EXPIRY
        if self.no_arb_mode == "surface":
            return SIMPLE_INCLUDE_FIELDS_NOARB_SURFACE
        return SIMPLE_INCLUDE_FIELDS
    
    def fields_to_features(self) -> dict[str, str|None]:
        """
        Returns the map from the fields of the time series and the modified fileds in :func:`sample_simple_days`
        to the features included in the sample, depending on the arbitrage pre-processing mode.
        """
        if self.no_arb_mode == "expiry":
            return SIMPLE_FEATURES_NOARB_EXPIRY
        if self.no_arb_mode == "surface":
            return SIMPLE_FEATURES_NOARB_SURFACE
        return SIMPLE_FEATURES
    
    def features(self) -> set[str]:
        """ Returns the features to be included in the final sample, depending on the arbitrage pre-processing mode. """
        return set( (f if n is None else n) for f, n in self.fields_to_features().items() )

    def base_cache_dir_uid(self) -> str:
        """ Base directory UID up to ticker """
        x1   = named_unique_filename48_8(f'arbfree {str(self.expiry_filter)},{str(self.raw_day_filter)},{str(self.mk_arb_free)}', self.expiry_filter, self.raw_day_filter, self.mk_arb_free)
        x2   = named_unique_filename48_8( f"{self.timeseries_config}", self.timeseries_config )
        x3   = named_unique_filename48_8( f"{self.sampling_config}", self.sampling_config )
        return f"{x1}/{x2}/{x3}/{str(self.date_range)}"
    
    def cache_dir_uid(self) -> str:
        """ UID including directory informaton """
        return f"{self.base_cache_dir_uid()}/{named_unique_filename48_8( fmt_list(sorted(self.tickers)), sorted(self.tickers))}"
    
    def unique_id(self) -> str:
        """ Unique ID for the configuration, including tickers. """
        return unique_hash16(self.cache_dir_uid())

    def instantiate_from_cache( self, *, verbose : Context = Context.quiet ) -> BaseSampler:
        """ 
        Creates the Sampler corresponding to this config from cache only.
        This function will fail if not all data are available.
        """
        return SimpleSampler( simple_config=self, pool_config=PoolConfig.off(), override_cache_mode="cacheonly", verbose=verbose )

SIMPLE_INCLUDE_FIELDS = {
    'spot',
    'spot_prev',
    'is_valid',
    'is_american',
    'is_am',
    'strike',
    'df',
    'tte_y',
    'fwd',
    'nms',
    'call_ask',
    'call_bid',
    'ask_vol',
    'bid_vol',
    'log_volume',
    'log_open_interest',
    'pure_strike',
    'pure_ask_price',
    'pure_ask_delta',
    'pure_ask_gamma',
    'pure_ask_vega',
    'pure_ask_theta',
    'pure_ask_dk',
    'pure_bid_price',
    'pure_bid_delta',
    'pure_bid_gamma',
    'pure_bid_vega',
    'pure_bid_theta',
    'pure_bid_dk'}
""" Fields the function requests from the time series if no arbitrage pre-pprocessing is applied. """
SIMPLE_INCLUDE_FIELDS_NOARB_EXPIRY=SIMPLE_INCLUDE_FIELDS|{
    'mask_arb_free',
    'fit1_vol',
    'pure_fit1_price',
    'pure_fit1_delta',
    'pure_fit1_gamma',
    'pure_fit1_vega',
    'pure_fit1_theta',
    'pure_fit1_dk'
}
""" Fields the function requests from the time series if ``"expiry"`` arbitrage pre-pprocessing is applied. """
SIMPLE_INCLUDE_FIELDS_NOARB_SURFACE=SIMPLE_INCLUDE_FIELDS_NOARB_EXPIRY|{
    'mask_arb_free',
    'fitX_vol',
    'pure_fitX_price',
    'pure_fitX_delta',
    'pure_fitX_gamma',
    'pure_fitX_vega',
    'pure_fitX_theta',
    'pure_fitX_dk'
}
""" Fields the function requests from the time series if ``"surface"`` arbitrage pre-pprocessing is applied. """

assert SIMPLE_INCLUDE_FIELDS <= set(SCHEMA_ENRICHED_DAY)|{'is_valid'}, f"SIMPLE_INCLUDE_FIELDS must be a subset of the available fields in SCHEMA_ENRICHED_DAY. Unknown fields: {sorted(SIMPLE_INCLUDE_FIELDS - set(SCHEMA_ENRICHED_DAY))}"
assert SIMPLE_INCLUDE_FIELDS_NOARB_EXPIRY <= set(SCHEMA_ARB_FREE_DAY_EXPIRY)|{'is_valid'}, f"SIMPLE_INCLUDE_FIELDS_NOARB_EXPIRY must be a subset of the available fields in SCHEMA_ARB_FREE_DAY_EXPIRY. Unknown fields: {sorted(SIMPLE_INCLUDE_FIELDS_NOARB_EXPIRY - set(SCHEMA_ARB_FREE_DAY_EXPIRY))}"
assert SIMPLE_INCLUDE_FIELDS_NOARB_SURFACE <= set(SCHEMA_ARB_FREE_DAY_SURFACE)|{'is_valid'}, f"SIMPLE_INCLUDE_FIELDS_NOARB_SURFACE must be a subset of the available fields in SCHEMA_ARB_FREE_DAY_SURFACE. Unknown fields: {sorted(SIMPLE_INCLUDE_FIELDS_NOARB_SURFACE - set(SCHEMA_ARB_FREE_DAY_SURFACE))}"

_SIMPLE_FEATURES_ALWAYS = {
    'spot_ref':None,  # actual $
    'spot_ret':None,  # spot(t) / spot(t-1) - 1
    'is_valid':None,
    'is_american':None,
    'is_am':None,
    'df':None,
    'tte_y':None,
    'fwd':None,       # divided by spot_ref
    'nms':None,
    'ask_vol':None,
    'bid_vol':None,
    'log_volume':None,
    'log_open_interest':None,
    'pure_strike':None, # strike/fwd, in dollars
    'pure_ask_price':None,
    'pure_bid_price':None
}
""" Features we always extract """
SIMPLE_FEATURES = _SIMPLE_FEATURES_ALWAYS|{
    'pure_ask_delta':None,
    'pure_bid_delta':None,
    'pure_ask_gamma':None,
    'pure_bid_gamma':None,
    'pure_bid_vega':None,
    'pure_ask_vega':None,
    'pure_bid_theta':None,
    'pure_ask_theta':None,
    'pure_ask_dk':None,
    'pure_bid_dk':None
}
""" Features we extract if no arbitrage pre-processing is applied, and how they are translated """
SIMPLE_FEATURES_NOARB_EXPIRY = _SIMPLE_FEATURES_ALWAYS|{
    'mask_arb_free':None,
    'fit1_vol':'fit_vol',
    'pure_fit1_price':'pure_fit_price',
    'pure_fit1_delta':'pure_fit_delta',
    'pure_fit1_gamma':'pure_fit_gamma',
    'pure_fit1_vega':'pure_fit_vega',
    'pure_fit1_theta':'pure_fit_theta',
    'pure_fit1_dk':'pure_fit_dk'
}    
""" Features we extract if ``"expiry"`` arbitrage pre-processing is applied, and how they are translated """
SIMPLE_FEATURES_NOARB_SURFACE = _SIMPLE_FEATURES_ALWAYS|{
    'mask_arb_free':None,
    'fitX_vol':'fitX_vol',
    'pure_fitX_price':'pure_fit_price',
    'pure_fitX_delta':'pure_fit_delta',
    'pure_fitX_gamma':'pure_fit_gamma',
    'pure_fitX_vega':'pure_fit_vega',
    'pure_fitX_theta':'pure_fit_theta',
    'pure_fitX_dk':'pure_fit_dk'
}    

@training_root_simple(fmt=SubDir.PYTREE_HDF5).cache(
    "0.5",
    dependencies=[SimpleConfig, build_time_series],
    uid = lambda ticker, simple_config: f"{simple_config.base_cache_dir_uid()}/{ticker}",
    exclude_args=['pool_config', 'refresh_cache', 'verbose']
)
def sample_simple_days(  ticker : str,
                         simple_config : SimpleConfig, *,
                         pool_config : PoolConfig = PoolConfig.off(),
                         override_cache_mode : CacheMode|None = None,
                         verbose : Context = Context.all ) -> dict:
    """
    Cached version of build_time_series which also normalizes the data:

    * ``pure_strike`` is the pure strike.
    * ``strike`` is divided by ``spot_ref``.
    * ``fwd`` is divided by ``spot_ref``.
    * ``call_ask`` and ``call_bid`` are divided by ``spot_ref``.
    * ``spot_ref`` is the original spot price, kept for reference.
    * ``spot_ret`` is the spot return, defined as spot(t) / spot(t-1) - 1.

    The result is a dict cached with HDF5 which contains ``dates`` (the dater per time index) and
    the features ``SIMPLE_FEATURES``,
     ``SIMPLE_FEATURES_NOARB_EXPIRY`` or ``SIMPLE_FEATURES_NOARB_SURFACE``, depending on 
      whether any and if so which no-arbitrage pre-processing is applied
    """

    timeseries_fields  = simple_config.timeseries_fields()  # fields to be read from time series
    fields_to_features = simple_config.fields_to_features() # translation map from time series + fiddling to features
    if not 'is_valid' in fields_to_features:
        raise ValueError(f"'is_valid' must be included in the fields to features mapping, as it is used for masking invalid options. Please include 'is_valid' in the mapping with a value of None or a feature name.")
    if 'dates' in fields_to_features:
        raise ValueError(f"'dates' cannot be included in the fields to features mapping, as it is used as a separate field in the output. Please remove 'dates' from the mapping.")

    r = build_time_series( 
                  ticker=ticker, 
                  date_range = simple_config.date_range,
                  include_fields = set(simple_config.timeseries_fields()), # which time series fields to use
                  exclude_fields = set(),
                  raw_day_filter = simple_config.raw_day_filter,
                  expiry_filter = simple_config.expiry_filter,
                  mk_arb_free = simple_config.mk_arb_free,
                  timeseries_config = simple_config.timeseries_config,
                  pool_config = pool_config,
                  override_cache_mode = override_cache_mode,
                  verbose = verbose )
    
    expected = set(timeseries_fields) | {"dates", "ticker"}
    assert set(r) == expected, f"Expected build_time_series to return fields {sorted(expected)}, got {sorted(r)}."

    # scale all cash quantities by initial spot.
    # In order not to lose sight of the cash relation to the previous spot, keep the return as well.
    # We also have spot_ref but that should not be a feature, anywhere.

    S, O = r.spot.shape

    if np.min(r.spot) <= 0.:
        raise ValueError(f"Expected positive spot, got {np.min(r.spot):.4g} for ticker {ticker}.")
    if np.min(r.spot_prev) <= 0.:
        raise ValueError(f"Expected positive spot_prev, got {np.min(r.spot_prev):.4g} for ticker {ticker}.")

    r.spot_ref = r.spot
    r.spot_ret = np.where( r.is_valid, r.spot / np.where( r.is_valid, r.spot_prev, 1. ) - 1.0, 0. )
    valid_spot = np.where( r.is_valid, r.spot, 1. )
    r.pure_strike = np.where( r.is_valid, r.strike / np.where( r.is_valid, r.fwd, 1. ), 1. )
    r.call_ask /= valid_spot
    r.call_bid /= valid_spot
    r.fwd /= valid_spot
    r.strike /= valid_spot

    # generate sortable date_ids for 
    r.spot_ref = r.spot_ref[:,:1] # spot is the same for all options
    r.spot_ret = r.spot_ret[:,:1] # spot return is the same for all options
    del r.spot, r.spot_prev, r.ticker
    date_diffs = [ b-a for b,a in zip(r.dates[1:], r.dates[:-1]) ]
    date_diffs = [ d.total_seconds() for d in date_diffs ]
    assert min(date_diffs) > 0., "Internal error: date_ids should be strictly increasing with a step of at least 0.5 to ensure proper ordering and separation of samples. Please contact the authors with an example if you encounter this error."

    if set(r) < set(fields_to_features)|{"dates"}:
        missing = set(fields_to_features)|{"dates"} - set(r)
        raise RuntimeError(f"{ticker}: internal error - missing the following fields: {fmt_list(sorted(missing))}.")

    # convert to target output
    # ------------------------

    out = dict(dates=r.dates)
    for src, tar in fields_to_features.items():
        v = r.pop(src)
        tar = tar if tar is not None else src
        assert isinstance(v, np.ndarray), f"Expected numpy array for field '{src}', got {type(v)} for ticker {ticker}."
        if v.dtype == np.bool_:
            out[tar] = v
        else:
            out[tar] = v.astype(np.float32, copy=False)
        assert np.all( np.isfinite(v) ), f"Expected all finite values for field '{src}', got non-finite values for ticker {ticker}."

    return dict(out) # for HDF5

# --------------------------------------------------------------
# Sampler
# --------------------------------------------------------------

@version("0.1")
@dataclass(frozen=True)
class SampleID:
    """ Identifies a sample """
    date: dt.date   #: the end date
    ticker: str     #: ticker
    index: int      #: the index into the time axis of this ticker

    def __lt__(self, other: SampleID) -> bool:
        if self.index < other.index:
            return True
        if self.index > other.index:
            return False
        return self.ticker < other.ticker
    
    def __eq__(self, other: SampleID) -> bool:
        return self.ticker == other.ticker and self.date == other.date and self.index == other.index

    def __str__(self) -> str:
        """ Strict identification """
        return f"{self.ticker} {self.index}:{fmt_date(self.date)} "
    
@version("0.1", dependencies=[SimpleConfig, sample_simple_days, SampleID])
class SimpleSampler(BaseSampler):
    """
    Samper for Dynamic SANOS.
    
    This sampler generates torch samples according to ``SimpleConfig``.   

    Key functions are::

    * :meth:`__call__` generates a random barch of a given size.

    * :meth:`all` is a generator that yields all samples in order for valuation runs. Samples towards the ends are padded.
      This is done using existing samples, so padded sample data will adhere to the same consistency conditions as valid samples).

    * :meth:`generate_batch` which generates a batch of samples given a list of :class:`SampleID`s, which identify the ticker, date and index of the sample. 
    """

    Config = SamplingConfig

    def __init__( self, simple_config : SimpleConfig, *, 
                        pool_config   : PoolConfig = PoolConfig.off(), 
                        cache_mode    : CacheMode|None = None,
                        verbose: Context = Context.quiet ) -> None:
        """
        Create full sampler.
        """
        rng = np.random.default_rng(simple_config.sampling_config.seed)
        self.tickers : str[list]              = sorted( set(simple_config.tickers) ) #: sorted list of tickers
        self.features                         = set(simple_config.features()) #: set of features included in the sample, depending on the arbitrage pre-processing mode.
        self.data : dict[str, dict]           = None #: data per ticker
        self.max_options : int                = 0 #: max number of options across the tickers and dates, used for padding
        self.config : SimpleConfig            = simple_config #: :class:`SimpleConfig` configuration for the sampler.
        self.trn_samples_ids : list[SampleID] = None #: List of :class:`SampleID`s for the training samples, sorted by date and ticker.
        self.vld_samples_ids : list[SampleID] = None #: List of :class:`SampleID`s for the validation samples, sorted by date and ticker.   
        self._shapes : dict[str, tuple]       = None 
        self._stats : PrettyObject            = None #: training set based stats.

        tme_all       = verbose.timer()
        ft_with_dates = self.features|{'dates'}

        def collate():
            # Cannot thread on top of multipricessing
            results = {}

            if pool_config.threading:
                pool = JCPool( pool_config.num_workers, threading=True )
                def process(ticker):
                    return sample_simple_days( ticker=ticker, simple_config=simple_config, pool_config=pool_config, override_cache_mode=cache_mode, verbose=Context.quiet )
                with verbose.write_t(lambda : f"Generating samples for {len(self.tickers)} tickers {fmt_list(self.tickers)} with from {simple_config.date_range.start} to {simple_config.date_range.end}... ", end='') as tme:
                    for ticker, tdata in pool.parallel( {ticker: pool.delayed(process)(ticker) for ticker in self.tickers} ):
                        results[ticker] = PrettyObject(tdata)
                    verbose.write(lambda : f"done; this took {tme}.", head=False )
            else:
                with verbose.write_t(lambda : f"Generating samples for {len(self.tickers)} tickers {fmt_list(self.tickers)} with from {simple_config.date_range.start} to {simple_config.date_range.end}... ", end='') as tme:
                    for ticker in self.tickers:
                        tdata = sample_simple_days( ticker=ticker, simple_config=simple_config, pool_config=pool_config, override_cache_mode=cache_mode, verbose=verbose(1) )
                        results[ticker] = PrettyObject(tdata)
                    verbose.write(lambda : f"done; this took {tme}.", head=False )

            for ticker, tdata in results.items():
                assert set(tdata) == ft_with_dates, f"Internal error: for ticker {ticker} missing features: {fmt_list(sorted(ft_with_dates - set(tdata)))} / superfluous features: {fmt_list(sorted(set(tdata) - ft_with_dates))}."

            return results
        
        self.data = collate()

        verbose.write(lambda : f"Found the following number of samples per ticker: {fmt_dict({ticker: len(tdata.dates) for ticker,tdata in self.data.items()})}.")

        # Collect full date range
        # -----------------------

        def setup():
            all_dates  = set()
            max_options = 0
            for ticker, tdata in self.data.items():
                assert set(tdata) == ft_with_dates, f"Internal error: for ticker {ticker} missing features: {fmt_list(sorted(ft_with_dates - set(tdata)))} / superfluous features: {fmt_list(sorted(set(tdata) - ft_with_dates))}."
                all_dates.update(tdata['dates'])
                max_options = max(max_options, tdata['is_valid'].shape[1])
            all_dates = sorted(all_dates)
            num_dates = len(all_dates)

            if num_dates <= simple_config.sampling_config.num_steps*3:
                raise ValueError(f"Not enough dates to create at least 3 non-overlapping samples with num_steps {simple_config.sampling_config.num_steps}: found only {num_dates} unique dates across the tickers {fmt_list(self.tickers)}.")

            num_av_dates = len(all_dates) - simple_config.sampling_config.num_steps*2
            ix_vld       = simple_config.sampling_config.num_steps + min(num_av_dates, max(0, int(round(float(num_av_dates)*(1.-simple_config.sampling_config.pct_validation)))))

            return all_dates[ix_vld], all_dates, num_dates, max_options
        first_vld_date, all_dates, num_dates, self.max_options = setup()

        verbose.write(lambda : f"Found date range {all_dates[0]} to {all_dates[-1]} with {num_dates} total dates. "+\
                               f"First validation date for {simple_config.sampling_config.num_steps} steps per sample is {first_vld_date}.")
        verbose.write(lambda : f"Using up to {self.max_options} options per date.")

        def get_sample_ids():    
            trn_samples_ids = set()
            vld_samples_ids = set()

            for ticker, tdata in self.data.items():
                dates     = tdata.dates
                trn_dates = [d for d in dates if d < first_vld_date]
                vld_dates = [d for d in dates if d >= first_vld_date]

                tnum_trn_samples = 0 if len(trn_dates) < simple_config.sampling_config.num_steps else \
                                   1 + ( len(trn_dates) - simple_config.sampling_config.num_steps ) // simple_config.sampling_config.num_slide
                tnum_vld_samples = 0 if len(vld_dates) < simple_config.sampling_config.num_steps else \
                                   1 + ( len(vld_dates) - simple_config.sampling_config.num_steps ) // simple_config.sampling_config.num_slide

                verbose.report(1, lambda : f"Ticker {ticker}: using {tnum_trn_samples} training samples (accross {len(trn_dates)} training dates), and {tnum_vld_samples} validation samples (accross {len(vld_dates)} validation dates).")

                ttrn_indices = np.arange( len(trn_dates)-simple_config.sampling_config.num_steps, step=-simple_config.sampling_config.num_slide, stop=-1 , dtype=np.int32)
                tvld_indices = np.arange( len(vld_dates)-simple_config.sampling_config.num_steps, step=-simple_config.sampling_config.num_slide, stop=-1 , dtype=np.int32)

                if len(ttrn_indices) > 0:
                    assert np.max(ttrn_indices)+simple_config.sampling_config.num_steps <= len(trn_dates), f"Internal error: max training index {np.max(ttrn_indices)} is out of bounds for dates of length {len(trn_dates)} and slide {simple_config.sampling_config.num_slide}"
                if len(tvld_indices) > 0:
                    assert np.max(tvld_indices)+simple_config.sampling_config.num_steps <= len(vld_dates), f"Internal error: max validation index {np.max(tvld_indices)} is out of bounds for validation dates of length {len(vld_dates)} and slide {simple_config.sampling_config.num_slide}"
                    tvld_indices += len(trn_dates)  # shift validation indices to be in the same index space as training indices, which start at 0 for the first date in 'dates'
                    assert np.max(tvld_indices)+simple_config.sampling_config.num_steps <= num_dates, f"Internal error: max validation index {np.max(tvld_indices)} is out of bounds for all dates of length {num_dates} and slide {simple_config.sampling_config.num_slide}"
                    assert np.min(tvld_indices) >= len(trn_dates), f"Internal error: min validation index {np.min(tvld_indices)} must exceed {len(trn_dates)} for the validation indices to be properly separated from training indices."

                trn_samples_ids |= { SampleID(ticker=ticker, date=dates[i], index=i) for i in ttrn_indices }
                vld_samples_ids |= { SampleID(ticker=ticker, date=dates[i], index=i) for i in tvld_indices }
                del trn_dates, vld_dates, ttrn_indices, tvld_indices, tnum_trn_samples, tnum_vld_samples

            if len(trn_samples_ids) == 0:
                raise ValueError(f"Not enough samples to create even one training sample for {num_dates} dates using {simple_config.sampling_config.num_steps} steps and sliding by {simple_config.sampling_config.num_slide}.")
            if len(vld_samples_ids) == 0:
                raise ValueError(f"Not enough samples to create even one validation sample for {num_dates} dates using {simple_config.sampling_config.num_steps} steps and sliding by {simple_config.sampling_config.num_slide}.")

            trn_samples_ids = list(trn_samples_ids)
            vld_samples_ids = list(vld_samples_ids)            
            rng.shuffle( trn_samples_ids )
            rng.shuffle( vld_samples_ids )
            return trn_samples_ids, vld_samples_ids
        
        self.trn_samples_ids,\
        self.vld_samples_ids = get_sample_ids()

        # generate shape information
        # --------------------------

        def make_shapes():
            shapes = {}
            tdata = self.data[self.tickers[0]]

            for k,v in tdata.items():
                if k in ['dates', 'ticker']:
                    continue
                shapes[k] = ( simple_config.sampling_config.num_steps if v.shape[0] > 1 else 1,
                              self.max_options if v.shape[1] > 1 else 1 ) 
            return shapes
        self._shapes = make_shapes()

        # get some descriptive stats from training data
        # ---------------------------------------------

        def trn_stats():
            stats = PrettyObject()
            stats.mean_pure_spread = 0.
            stats.mean_vol_spread = 0.
            stats.mean_sqrtVar_spread = 0.
            cnt = 0

            for sid in self.trn_samples_ids:
                tdata    = self.data[sid.ticker]
                slc      = slice( sid.index, sid.index+self.config.sampling_config.num_steps )
                num_tdates = len(tdata.dates)
                assert slc.stop <= num_tdates, f"Internal error: {sid.ticker} sample index {slc.start} + num_steps {self.config.sampling_config.num_steps} exceeds number of dates {num_tdates} for ticker {sid.ticker} with sample length {self.config.sampling_config.num_steps}"
                is_valid = tdata.is_valid[slc,:]
                pure_call_ask  = tdata.pure_ask_price[slc,:][is_valid]
                pure_call_bid  = tdata.pure_bid_price[slc,:][is_valid]
                pure_spread    = pure_call_ask - pure_call_bid
                ask_vol        = tdata.ask_vol[slc,:][is_valid] 
                bid_vol        = tdata.bid_vol[slc,:][is_valid]
                vol_spread     = ask_vol - bid_vol
                sqrtT          = np.sqrt( tdata.tte_y[slc,:][is_valid] )
                sqrtVar_spread = vol_spread * sqrtT

                stats.mean_vol_spread += vol_spread.sum()
                stats.mean_pure_spread += pure_spread.sum()
                stats.mean_sqrtVar_spread += sqrtVar_spread.sum()
                cnt += len(pure_spread)
            assert cnt > 0, f"Internal error: no valid options found in the training samples to compute stats."
            return PrettyObject({ k : x / float(cnt) for k, x in stats.items() })

        self._stats = trn_stats()

        verbose.write(lambda : f"Created {len(self.trn_samples_ids)} training samples and {len(self.vld_samples_ids)} validation samples based on the date split at {first_vld_date}. "+\
                               f"That implies {len(self.vld_samples_ids)/(len(self.trn_samples_ids)+len(self.vld_samples_ids)):.0%} of all samples are validation samples.")
        verbose.write(lambda : f"Total time taken: {tme_all}.")

    @property
    def unique_data_id(self) -> str:
        """ Unique ID for the data, based on the configuration. """
        return self.config.unique_id()
    
    @property
    def unique_data_dir_uid(self) -> str:
        """ Unique directory UID for the data, based on the configuration. """
        return self.config.cache_dir_uid()

    @property
    def num_steps(self):
        return self.config.sampling_config.num_steps
    
    @property
    def num_trn_samples(self) -> int:
        """ Get the number of training samples """
        return len(self.trn_samples_ids)
    
    @property
    def num_vld_samples(self) -> int:
        """ Get the number of validation samples """
        return len(self.vld_samples_ids)

    def stats(self) -> PrettyObject:
        """
        Training stats for data scaling:

        * ``mean_pure_spread``: mean of the pure call spread ``pure_ask_price - pure_bid_price`` across all valid options in the training samples.

        * ``mean_vol_spread``: mean of the volatility spread ``ask_vol - bid_vol`` across all valid options in the training samples.

        * ``mean_sqrtVar_spread``: mean of the square root of the variance spread ``vol_spread * sqrtT`` across all valid options in the training samples.
        """
        return self._stats
    
    def shapes( self, batch_size : int|None = None ) -> dict:
        """ Get the shapes of the tensors returned by __call__, based on the batch size and the internal shape information. The shapes will be of the form (batch_size, num_steps | 1, num_options | 1) """
        if batch_size is None:
            return self._shapes
        return { k : (batch_size, v[0], v[1]) for k,v in self._shapes.items() }

    def generate_batch( self, batch_ids : list[SampleID], device = None, *, pad_to_size : int|None = None ) -> dict:
        """
        Generate batch for a list of ``SampleID``s on the provided ``device``.
        
        If ``pad_to_size`` is not ``None`` the function will add a boolean ``pad_mask`` tensor of shape ``(batch_size, 1, 1)`` to the return batch, 
        which indicates which sample in the batch are valid samples (True) and which are padding samples (False) that should be ignored.

        Parameters
        ----------
        batch_ids : list[SampleID]
            List of ``SampleID``s for which to generate the batch. The batch will be generated in the order of the provided list.
        device : torch.device, default ``None``
            Device on which to place the tensors. If None, defaults to the current device.
        pad_to_size : int|None, default ``None``
            If not None, forces the batch to have exactly this many samples.

            * If the provided list of ``SampleID``s is shorter than ``pad_to_size``, the last sample will be repeated to pad the batch
              (this ensures that padded samples retain consistency with the original data scheme).
            * If the provided list of ``SampleID``s is longer than ``pad_to_size``, it will be truncated. This is useful for ensuring a consistent batch size, for example when using batch normalization layers in a model.

            If ``pad_to_size`` is set, then the returned batch will contain a boolean tensor ``pad_mask`` of shape ``(batch_size, 1, 1)``,
            which indicates which samples in the batch are valid (True) and which are padding (False). This will happen even if 
            the provided list of ``SampleID``s is already of length ``pad_to_size`` (in which case the mask will be all-True).

            If ``pad_to_size`` is None, all samples are valid and the mask not be present.

        Returns
        -------
        batch : dict[str, torch.Tensor]
            Dictionary containing the batch data, with keys corresponding to the data fields and values being
            tensors of shape ``(batch_size, num_steps|1, num_options|1)``.

            If ``pad_to_size`` is not ``None``, the batch will also contain a boolean tensor ``pad_mask`` of shape ``(batch_size, 1, 1)``,
            which indicates which samples in the batch are valid (True) and which are padding (False).
            If ``pad_to_size`` is None, the mask will not be present.            
        """

        # assemble
        import torch as torch # local import to avoid loading torch unless used

        cnt_pad_samples = None

        if not pad_to_size is None:
            cnt_pad_samples = 0
            if len(batch_ids) > pad_to_size:
                batch_ids = batch_ids[:pad_to_size]
            elif len(batch_ids) < pad_to_size:
                cnt_pad_samples = pad_to_size - len(batch_ids)
                batch_ids = batch_ids + [ batch_ids[-1] ]*cnt_pad_samples

        def assemble( tar, src, *, isample, ixsrc, num_tdates ):
            for k,v in src.items():
                if k in ['dates', 'ticker']:
                    continue
                assert k != 'pad_mask', f"Internal error: 'pad_mask' is a reserved field name and should not be present in the data for ticker {btid.ticker} {btid.date}."

                assert isinstance(v, np.ndarray), f"{k}: expected np.ndarray, got {type(v)}"
                assert len(v.shape) == 2, f"{k}: expected 2D array, got shape {v.shape}"
                assert v.shape[0] in [1,num_tdates], f"{k}: expected first dimension to be 1 or {num_tdates}, got shape {v.shape}"
                assert v.shape[1] <= self.max_options, f"{k}: expected second to be less than {self.max_options}, got shape {v.shape}"
                assert ixsrc+self.num_steps <= num_tdates, f"{k}: expected source index {ixsrc} + {self.num_steps} = {ixsrc+self.num_steps} to be less than number of dates {num_tdates}"

                v = v[ixsrc:ixsrc+self.num_steps,:] if v.shape[0] > 1 else v
                tv_shape  = ( len(batch_ids),
                              self.num_steps if v.shape[0] > 1 else 1,
                              self.max_options if v.shape[1] > 1 else 1 )
                if v.dtype == np.float32:
                    tv_dtype = torch.float32
                elif v.dtype == np.float64:
                    tv_dtype = torch.float64
                elif v.dtype == np.bool_:
                    tv_dtype = torch.bool
                elif v.dtype == np.int32:
                    tv_dtype = torch.int32
                else:
                    raise ValueError(f"{k}: unsupported dtype {v.dtype}")
                
                tv = tar.get(k, None)
                if tv is None:
                    tv = torch.zeros( tv_shape, dtype=tv_dtype, device=device )
                    tar[k] = tv
                else:
                    assert tv.shape == tv_shape, f"{k}: expected existing tensor shape {tv_shape}, got {tv.shape}"
                tv[isample,:,:v.shape[1]] = torch.from_numpy(v)

        tbatch     = {}
        for isample, btid in enumerate( batch_ids ):
            irets      = self.data[btid.ticker]
            num_tdates = len(self.data[btid.ticker].dates)
            assert btid.index+self.num_steps <= num_tdates, f"Internal error: sample index {btid.index} vs length {num_tdates} for ticker {btid.ticker} with sample length {self.num_steps}"            
            assemble( tbatch, irets, isample=isample, ixsrc=btid.index, num_tdates=num_tdates )

        # padding information
        if not cnt_pad_samples is None:
            pad_mask = np.ones( (len(batch_ids), 1, 1), dtype=bool )
            pad_mask[-cnt_pad_samples:,0,0] = False
            tbatch['pad_mask'] = torch.tensor( pad_mask, dtype=torch.bool, device=device )
        return tbatch
    
    def __call__( self, g, *, batch_size : int, device = None, return_batch_indices : bool = False, trn : bool = True  ) -> dict:
        """
        Generate a pytorch training batch of a given size, for either the training or validation set.
        Generated batches are dictionaries with elements of shape ``(batch_size, num_steps|1, num_options|1)``.

        Samples generated by this function do not pad, and therefore do not contain a ``pad_mask`` field.
         
        Parameters
        ----------
        g : :class:`torch.Generator`
            Random generator for reproducibility.
        batch_size : int
            Number of samples in the batch.
            Must be less than or equal to the number of available samples in the respective set.
        device : :class:`torch.device`, default ``None``
            Device on which to place the tensors. If None, defaults to the current device.
        return_batch_indices : bool, default ``False``
            If True, also return the list of :class:`SampleID`s corresponding to the generated batch.
            A :class:`SampleID` contains the ``ticker``, the start ``date``, and the ``index`` of the sample, the latter of which refers to the index of the start
            date into the time axis of the respective ticker's data.
                        
            That means you can identify the ticker and the start date associated by::

                batch_ids, batch = sampler(g, trn=True, batch_size=32, device=device, return_batch_indices=True)
                for bid in batch_ids:
                    print(f"Ticker: {bid.ticker}, start Date: {bid.date}, index: {bid.index}")

            This is useful for identifying bad data.
        trn : bool, default ``True``
            If True, generate a training batch; otherwise, generate a validation batch.

        Returns
        -------
        Results: dict
            If ``return_batch_indices`` is ``False``, a dictionary containing the batch data, with keys corresponding to the data fields and values being tensors of shape ``(batch_size, num_steps|1, num_options|1)``.
            The batch will always contain exactly ``batch_size`` sample. No padding will be applied.
        BatchIDs, Results: list, dict
            If ``return_batch_indices`` is ``True``, a tuple containing the list of :class:`SampleID`s and a dictionary containing the batch data, with keys corresponding to the data fields and values being tensors of shape ``(batch_size, num_steps|1, num_options|1)``.
            The batch will always contain exactly ``batch_size`` sample. No padding will be applied.
        """
        import torch as torch # local import to avoid loading torch unless used
    
        assert isinstance(g, torch.Generator), f"Expected 'g' to be a torch.Generator, got {type(g)}"
    
        sample_ids = self.trn_samples_ids if trn else self.vld_samples_ids
        assert batch_size <= len(sample_ids), f"Batch size {batch_size} is larger than the number of available {'training' if trn else 'validation'} samples {len(sample_ids)}."
        if batch_size<len(sample_ids):
            batch_idx  = torch.randperm(len(sample_ids), generator=g)[:batch_size].tolist() 
            ### --> constant batch: batch_idx  = list(range(batch_size)) 
            batch_ids  = [ sample_ids[i] for i in batch_idx ]
            assert len(batch_ids) == batch_size, f"Internal error: expected {batch_size} sample ids, got {len(batch_ids)}"
        else:
            batch_ids  = sample_ids

        batch = self.generate_batch( batch_ids=batch_ids, device=device, pad_to_size=None )
        assert not 'pad_mask' in batch, f"Internal error: batch should not contain 'pad_mask' when pad_to_size is None."
        if return_batch_indices:
            return batch_ids, batch
        return batch

    def all( self, trn : bool, batch_size : int, device = None, *, yield_size_and_batch : bool = False, pad : bool = True ) -> dict:
        """
        Batch iterator iterating the full training or validation set.
        Generated batches are dictionaries with elements of shape ``(batch_size, num_steps|1, num_max_options|1)``.
        
        If ``pad`` is True (the default), this function will add a boolean ``pad_mask`` tensor of shape ``(batch_size, 1, 1)``
        to the return batch, 
        which indicates which samples in the batch are valid (True) and which are padding (False) and should be ignored.

        This array will not be present if ``pad_to_size`` is False.

        Usage::

            for batch_size, batch in sampler.all(trn=True, batch_size=32, device=device, yield_size_and_batch=True):
                # process batch...
                # batch_size < 32 if we reached the end of the sample set

        Parameters
        ----------
        trn : bool
            If True, iterate over the training set; otherwise, iterate over the validation set.
        batch_size : int
            Number of samples in each batch. Must be less than or equal to the number of available samples in the respective set.
        device : :class:`torch.device`, default ``None``
            Device on which to place the tensors. If None, defaults to the current device.
        yield_size_and_batch : bool, default ``False``
            If True, yield a tuple of (actual_batch_size, batch) for each batch.
        pad : bool, default ``True``
            If True, forces each batch to have exactly ``batch_size`` samples by padding with repeats of the last sample if necessary.
            In this case the batch will contain an additional boolean field ``pad_mask`` of shape ``(batch_size, 1, 1)``,
            indicating which samples in the batch are valid (True) and which can be ignored (False).

            This function will repeat the last sample for padding, ensuring that sample
            consistency conditions are maintained.
            
        Yields
        ------
        batch: dict[str, torch.Tensor]
            A dictionary containing the batch data, with keys corresponding to the data fields and values being tensors
            of shape ``(actual_batch_size, num_steps|1, num_options|1)``.
                    
        batch_size, batch: tuple[int, dict[str, torch.Tensor]]
            If `yield_size_and_batch` is True, yields a tuple of the actual batch size and the batch.
            If ``pad``` is True, then, this will always be equal to the provided `batch_size`.
        """

        sample_ids  = self.trn_samples_ids if trn else self.vld_samples_ids
        num_samples = len(sample_ids)

        pad_to_size = None if not pad else batch_size
        batch_size  = min(batch_size, num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size
        for ib in range(num_batches):
            start     = ib * batch_size
            end       = min(start + batch_size, num_samples)
            batch_ids = sample_ids[start:end]
            act_size  = (end - start) if pad_to_size is None else pad_to_size

            if yield_size_and_batch:
                yield act_size, self.generate_batch( batch_ids=batch_ids, device=device, pad_to_size=pad_to_size )
            else:
                yield self.generate_batch( batch_ids=batch_ids, device=device, pad_to_size=pad_to_size )

# ================================================================
# Command-line interface
# ================================================================

import click
from quantdata.widb import cli_defaults
from quantdata.biz_calendar import to_date

@click.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--period",nargs=2, type=click.DateTime(formats=["%Y-%m-%d"]), default=cli_defaults.period, metavar="START END", help="Period YYYY-MM-DD YYYY-MM-DD")
@click.option('--halfdays', is_flag=True, help="Include half days")
@click.option('--workers', default=cli_defaults.workers, help='Number of workers', type=int)
@click.option('--threading', is_flag=True, help="Use only threading, no multiprocessing")
@click.option('--refresh', is_flag=True, help="Refresh data (this will only refresh the top level files)")
@click.option('--debug', is_flag=True, help="Enable debug mode with more verbose output and checks")

def main(tickers : list[str],
        period : list[dt.datetime], 
        halfdays : bool, 
        workers : int, 
        threading : bool, 
        refresh : bool,
        debug : bool):

    simple_config = SimpleConfig(
            tickers = tickers,
            date_range = DateRange(start=to_date(period[0]), end=to_date(period[1]), exclude_half_days=not halfdays),
            raw_day_filter = RawDayFilter(),
            expiry_filter = ExpiryFilter(),
            timeseries_config = TimeseriesConfig(),
            sampling_config = SamplingConfig(num_samples=None)
    )
    pool_config = PoolConfig(num_workers=workers, threading=threading)
    verbose = Context.all
    refresh_cache = refresh

    SimpleSampler(simple_config=simple_config, pool_config=pool_config, cache_mode="update" if refresh_cache else "on", verbose=verbose)
    print("Done.")

if __name__ == "__main__":
    main()
