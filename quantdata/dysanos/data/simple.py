"""
Simple data for SANOS: basically just data at 't' plus spot return.
"""

from __future__ import annotations

from quantdata import env#NOQA
from quantdata.dysanos.data import _training_root

from quantdata.widb.sampler import sample_days, SampleDaysConfig, RawDayFilter, ExpiryFilter, SCHEMA_ENRICHED_DAY
from quantdata.biz_calendar import DateRange
from quantdata.pool import PoolConfig

from cdxcore.version import version
from cdxcore.verbose import Context
from cdxcore.util import fmt_date, fmt_dict, fmt_list
from cdxcore.err import verify_inp
from cdxcore.pretty import PrettyObject
from cdxcore.subdir import SubDir
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.jcpool import JCPool

import datetime as dt
import numpy as np
import math as math
from dataclasses import dataclass, field    
from collections.abc import Sequence

_training   = _training_root("simple", fmt=SubDir.PYTREE_HDF5)

@version("0.1")
@dataclass(frozen=True)
class SamplingConfig:
    num_samples : int|None = None    #: Number of samples used or ``None`` for all available samples.
    num_steps: int = 25              #: Steps per sample, including terminal 'y'
    num_slide: int = 5               #: Number of steps to slide for the next sample. If num_slide < num_steps, samples will overlap.
    pct_validation: float = 0.2      #: Percentage of samples to use for validation.
    seed : int = 42                  #: Random seed for reproducibility.

    def __post_init__(self):
        verify_inp( self.num_samples is None or self.num_samples >= 2, "'num_samples' must be at least 2, or None" )
        verify_inp( 0.0 < self.pct_validation < 1.0, "'pct_validation' must be in (0,1)" )   
        verify_inp( self.num_steps >= 2, "'num_steps' must be at least 2, as the last step is the prediction target" )
        verify_inp( self.num_slide > 0, "'num_slide' must be positive" )

    def __str__(self):#by convention use a some what readable string with comma separated key=value pairs
        return f"ns={self.num_samples if not self.num_samples is None else 'all'},stp={self.num_steps},sld={self.num_slide},vld={self.pct_validation:.2f},seed={self.seed}"

@version("0.1", dependencies=[RawDayFilter,ExpiryFilter,SampleDaysConfig,SamplingConfig])
@dataclass(frozen=True)
class SimpleConfig:
    """ BlockSampler configuration """
    tickers : list[str] = field(default_factory=lambda: ["SPY"])  #: Tickers to include in the sampler. Can be a single ticker as a string, or a list of tickers.
    date_range : DateRange = field(default_factory=lambda: DateRange(start=dt.date(2025, 1, 1), end=dt.date(2025, 2, 1), exclude_half_days=True))  # which dates
    raw_day_filter : RawDayFilter = field(default_factory=lambda: RawDayFilter())  #: Filter for raw days, applied before sampling.
    expiry_filter : ExpiryFilter = field(default_factory=lambda: ExpiryFilter())  #: Filter for option expiries, applied before sampling.
    sample_days_config : SampleDaysConfig = field(default_factory=lambda: SampleDaysConfig())  #: Configuration for sampling days, applied before sampling.
    sampling_config : SamplingConfig = field(default_factory=lambda: SamplingConfig())  #: Configuration for sampling, applied after the above filters.   

    def __post_init__(self):
        verify_inp( isinstance(self.tickers, Sequence) and len(self.tickers) > 0, "'tickers' must be a list of strings" )
        verify_inp( len(set(self.tickers)) == len(self.tickers), "'tickers' must not contain duplicates" )

    def base_cache_uid(self) -> str:
        """ Base UID until ticker """
        x1   = named_unique_filename48_8( f"{self.raw_day_filter},{self.expiry_filter}", self.raw_day_filter, self.expiry_filter )
        x2   = named_unique_filename48_8( f"{self.sample_days_config}", self.sample_days_config )
        x3   = named_unique_filename48_8( f"{self.sampling_config}", self.sampling_config )
        return f"{x1}/{x2}/{x3}/{str(self.date_range)}"

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
    'ask_pure_price',
    'bid_pure_price',
    'ask_pure_delta',
    'bid_pure_delta',
    'ask_pure_gamma',
    'bid_pure_gamma',
    'bid_pure_vega',
    'ask_pure_vega',
    'bid_pure_theta',
    'ask_pure_theta',
    'ask_pure_dk',
    'bid_pure_dk'
}
assert SIMPLE_INCLUDE_FIELDS <= set(SCHEMA_ENRICHED_DAY)|{'is_valid'}, f"SIMPLE_INCLUDE_FIELDS must be a subset of the available fields in SCHEMA_ENRICHED_DAY. Unknown fields: {sorted(SIMPLE_INCLUDE_FIELDS - set(SCHEMA_ENRICHED_DAY))}"

@_training(fmt=SubDir.PYTREE_HDF5).cache(
    "0.1",
    dependencies=[SimpleConfig, sample_days],
    uid = lambda ticker, simple_config: f"{simple_config.base_cache_uid()}/{ticker}",
    exclude_args=['pool_config', 'refresh_cache', 'verbose']
)
def sample_simple_days(  ticker : str,
                         simple_config : SimpleConfig, *,
                         pool_config : PoolConfig = PoolConfig.off(),
                         refresh_cache : bool = False, 
                         verbose : Context = Context.all ) -> dict:
    """
    Cached version of sample_days
    """
    r = sample_days( 
                  ticker=ticker, 
                  date_range = simple_config.date_range,
                  include_fields = SIMPLE_INCLUDE_FIELDS,
                  exclude_fields = set(),
                  raw_day_filter = simple_config.raw_day_filter,
                  expiry_filter = simple_config.expiry_filter,
                  sample_days_config = simple_config.sample_days_config,
                  pool_config = pool_config,
                  refresh_cache = refresh_cache,
                  verbose = verbose )
    
    expected = set(SIMPLE_INCLUDE_FIELDS) | {"dates", "ticker"}
    assert set(r) == expected, f"Expected sample_days to return fields {sorted(expected)}, got {sorted(r)}."

    # scale all cash quantities by initial spot.
    # In order not to lose sight of the cash relation to the previous spot, keep the return as well.
    # We also have spot_ref but that should not be a feature, anywhere.

    S, O = r.spot.shape

    r.spot_ref = r.spot
    r.spot_ret = r.spot / r.spot_prev - 1.0
    r.call_ask /= r.spot
    r.call_bid /= r.spot
    r.fwd /= r.spot
    # generate sortable date_ids for 
    r.spot_ref = r.spot_ref[:,:1] # spot is the same for all options
    r.spot_ret = r.spot_ret[:,:1] # spot return is the same for all options
    del r.spot, r.spot_prev, r.ticker
    date_diffs = [ b-a for b,a in zip(r.dates[1:], r.dates[:-1]) ]
    date_diffs = [ d.total_seconds() for d in date_diffs ]
    assert min(date_diffs) > 0., "Internal error: date_ids should be strictly increasing with a step of at least 0.5 to ensure proper ordering and separation of samples. Please contact the authors with an example if you encounter this error."
    return dict(r) # for HDF5

# --------------------------------------------------------------
# Sampler
# --------------------------------------------------------------

@dataclass(frozen=True)
class SampleID:
    date: dt.date   #: the end date
    ticker: str     #: ticker
    index: int      #: the index into the time axis of this ticker (e.g. it can be used to look up the date associated with sample)

    def __lt__(self, other: SampleID) -> bool:
        if self.index < other.index:
            return True
        return self.ticker < other.ticker
    
    def __eq__(self, other: SampleID) -> bool:
        return self.ticker == other.ticker and self.date == other.date and self.index == other.index

    def __str__(self) -> str:
        """ Strict identification """
        return f"{self.ticker} {self.index}:{fmt_date(self.date)} "
    
@version("0.1", dependencies=[SimpleConfig, sample_simple_days, SampleID])
class SimpleSampler(object):

    def __init__( self, simple_config: SimpleConfig, *, 
                        pool_config : PoolConfig = PoolConfig.off(), 
                        refresh_cache=False, 
                        verbose: Context = Context.quiet ) -> None:
        """
        Create full sampler.
        """
        np.random.seed( simple_config.sampling_config.seed )
        self.tickers = sorted( set(simple_config.tickers) ) #: sorted list of tickers
        self.data : dict[str, dict] = None #: data per ticker
        self.max_options : int = 0 #: max number of options across the tickers and dates, used for padding
        self.simple_config : SimpleConfig = simple_config #: :class:`SimpleConfig` configuration for the sampler.
        self.trn_samples_ids : list[SampleID] = None #: List of :class:`SampleID`s for the training samples, sorted by date and ticker.
        self.vld_samples_ids : list[SampleID] = None #: List of :class:`SampleID`s for the validation samples, sorted by date and ticker.   
        self._shapes : dict[str, tuple] = None

        def collate():
            pool = JCPool( pool_config.num_workers, threading=True )
            def process(ticker):
                return sample_simple_days( ticker=ticker, simple_config=simple_config, pool_config=pool_config, refresh_cache=refresh_cache, override_cache_mode="update" if refresh_cache else None, verbose=Context.quiet )
            with verbose.write_t(lambda : f"Generating samples for {len(self.tickers)} tickers {fmt_list(self.tickers)} with from {simple_config.date_range.start} to {simple_config.date_range.end}... ", end='') as tme:
                results = {}
                for ticker, tdata in pool.parallel( {ticker: pool.delayed(process)(ticker) for ticker in self.tickers} ):
                    results[ticker] = PrettyObject(tdata)
                verbose.write(lambda : f"done; this took {tme}.", head=False )
            return results
        
        self.data = collate()

        verbose.write(lambda : f"Found the following number of samples per ticker: {fmt_dict({ticker: len(tdata.dates) for ticker,tdata in self.data.items()})}.")

        # Collect full date range
        # -----------------------

        def setup():
            all_dates  = set()
            max_options = 0
            for tdata in self.data.values():
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
                raise ValueError(f"Not enough samples to create even one training sample for {self.all_dates[0]} to {self.all_dates[-1]} with {num_dates} using {simple_config.sampling_config.num_steps} steps and sliding by {simple_config.sampling_config.num_slide}.")
            if len(vld_samples_ids) == 0:
                raise ValueError(f"Not enough samples to create even one validation sample for {self.all_dates[0]} to {self.all_dates[-1]} with {num_dates} using {simple_config.sampling_config.num_steps} steps and sliding by {simple_config.sampling_config.num_slide}.")

            trn_samples_ids = list(trn_samples_ids)
            vld_samples_ids = list(vld_samples_ids)            
            np.random.shuffle( trn_samples_ids )
            np.random.shuffle( vld_samples_ids )
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

        verbose.write(lambda : f"Created {len(self.trn_samples_ids)} training samples and {len(self.vld_samples_ids)} validation samples based on the date split at {first_vld_date}. "+\
                               f"That implies {len(self.vld_samples_ids)/(len(self.trn_samples_ids)+len(self.vld_samples_ids)):.0%} of all samples are validation samples.")

    @property
    def num_steps(self):
        return self.simple_config.sampling_config.num_steps
    
    @property
    def num_trn_samples(self) -> int:
        """ Get the number of training samples """
        return len(self.trn_samples_ids)    
    
    @property
    def num_vld_samples(self) -> int:
        """ Get the number of validation samples """
        return len(self.vld_samples_ids)

    def shapes( self, batch_size : int|None = None ) -> dict:
        """ Get the shapes of the tensors returned by __call__, based on the batch size and the internal shape information. The shapes will be of the form (batch_size, num_steps | 1, num_options | 1) """
        if batch_size is None:
            return self._shapes
        return { k : (batch_size, v[0], v[1]) for k,v in self._shapes.items() }

    def generate_batch( self, batch_ids : list[SampleID], device = None ) -> dict:
        """ Generate batch for a list of ``SampleID``s on the provided ``device``. """
        # assemble
        import torch as torch # local import to avoid loading torch unless used

        def assemble( tar, src, *, isample, ixsrc, num_tdates ):
            for k,v in src.items():
                if k in ['dates', 'ticker']:
                    continue

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
        return tbatch
    
    def __call__( self, g, *, trn : bool, batch_size : int, device = None, return_batch_indices : bool = False ) -> dict:
        """
        Generate a pytorch batch of a given size, for either the training or validation set.
        Generated batches are dictionaries with elements of shape ``(batch_size, num_steps|1, num_options|1)``.

        Parameters
        ----------
        g : :class:`torch.Generator`
            Random generator for reproducibility.
        trn : bool
            If True, generate a training batch; otherwise, generate a validation batch.
        batch_size : int
            Number of samples in the batch. Must be less than or equal to the number of available samples in the respective set.
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

        Returns
        -------
        Results: dict
            If ``return_batch_indices`` is ``False``, a dictionary containing the batch data, with keys corresponding to the data fields and values being tensors of shape ``(batch_size, num_steps|1, num_options|1)``.
        BatchIDs, Results: list, dict
            If ``return_batch_indices`` is ``True``, a tuple containing the list of :class:`SampleID`s and a dictionary containing the batch data, with keys corresponding to the data fields and values being tensors of shape ``(batch_size, num_steps|1, num_options|1)``.
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

        if return_batch_indices:
            return batch_ids, self.generate_batch( batch_ids=batch_ids, device=device )
        return self.generate_batch( batch_ids=batch_ids, device=device )

    def all( self, trn : bool, batch_size : int, device = None, yield_size_and_batch : bool = False ):
        """
        Batch iterator iterating the full training or validation set.
        Generated batches are dictionaries with elements of shape ``(batch_size, num_steps|1, num_max_options|1)``.
        
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
            
        Yields
        ------
        batch: dict
            A dictionary containing the batch data, with keys corresponding to the data fields and values being tensors of shape (actual_batch_size, num_steps|1, num_options|1).                    
        batch_size, batch: tuple[int, dict]
            If `yield_size_and_batch` is True, yields a tuple of (actual_batch_size, batch) for each batch, where `actual_batch_size` is the number of samples in the batch
            (which may be less than `batch_size` for the last batch if the total number of samples is not divisible by `batch_size`).   
        """

        sample_ids  = self.trn_samples_ids if trn else self.vld_samples_ids
        num_samples = len(sample_ids)

        batch_size  = min(batch_size, num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size
        for ib in range(num_batches):
            start     = ib * batch_size
            end       = min(start + batch_size, num_samples)
            batch_ids = sample_ids[start:end]
            if yield_size_and_batch:
                yield end-start, self.generate_batch( batch_ids=batch_ids, device=device )
            else:
                yield self.generate_batch( batch_ids=batch_ids, device=device )



