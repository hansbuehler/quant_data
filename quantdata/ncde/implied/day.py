"""
Implied Day
^^^^^^^^^^^

This file generates sample data sets of selected options quote time series for a given day, indexed by expiry.
"""

from datetime import date
from cdxcore.jcpool import JCPool
from cdxcore.pretty import PrettyObject
from cdxcore.subdir import CacheMode, SubDir
from cdxcore.util import fmt_list
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.bs import bs
import polars as pl

from quantdata.biz_calendar import NMS5, DateRange, prev_business_day, session_open_close, DAYS_PER_YEAR, count_business_days_between, to_date
from quantdata.massive.client import MassiveClient
from quantdata.pool import PoolConfig
from quantdata.massive.options.contracts import ContractsFilter, filter_contracts, REJECTED
from quantdata.massive.processing import align_trades_to_quotes, align_quotes_to_trades, filter_valid_quotes, filter_valid_trades, TradeQuoteAlignmentConfig
from quantdata.utils.pl import verify_schema

from cdxcore.version import version
from cdxcore.verbose import Context
from dataclasses import asdict, dataclass, field
from cdxcore.uniquehash import unique_hash8, named_unique_filename48_8
import numpy as np
import math
import warnings

from quantdata.ncde.implied import _locations_ncde_implied
from quantdata.widb.bad import BAD
_cache = _locations_ncde_implied.processing(fmt=SubDir.POLARS_PARQUET)

OPT_TICK_SIZE = 0.01 # not quite true - this is the tick size for small priced options

@version("0.2", dependencies=[ContractsFilter])
@dataclass(frozen=True)
class OptionQuotesDayConfig:
    """
    Configuration for managing option quotes day data.
    
    """
    max_num_options_per_expiry : int|None = 1000
    contract_filter : ContractsFilter = ContractsFilter()
    alignment_config : TradeQuoteAlignmentConfig = TradeQuoteAlignmentConfig()
    overnight_fraction : float = 0.2
    dte_exclude_half_days : bool = False

    def __str__(self):
        return f"mxn={self.max_num_options_per_expiry}{',exchl' if self.dte_exclude_half_days else ''},{self.contract_filter},{self.alignment_config}"

@_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.1",dependencies=[align_quotes_to_trades],
    uid=lambda ticker, d, alignment_config: f"trades/{named_unique_filename48_8(str(alignment_config),alignment_config)}/{ticker}/{d}/{ticker}"
)
def _ticker_trade_series( ticker : str, d : date, alignment_config : TradeQuoteAlignmentConfig = TradeQuoteAlignmentConfig(), verbose : Context|None = Context.quiet) -> pl.DataFrame:
    """ Cached ticker series """
    verbose     = verbose if verbose is not None else Context.quiet
    with verbose.write_t(lambda : f"{ticker} {d}: Fetching time series for {ticker} {d}... ", end='') as tme:
        df = align_quotes_to_trades( ticker, d, config = alignment_config, within_session=True, drop_unmatched=False )
        verbose.write(lambda : f"done; found {len(df)} quotes; this took {tme}.", head=False)
    return df.sort("ts_ns")
    
OPT_QUOTE_BY_EXPIRY_SCHEMA = {
            # option quotes
            # option quotes may not have trades
            "t_d": pl.Float64,           # time since open in days
            "mid_price": pl.Float64,     # mid price (divded by prev_spot_close)
            "half_spread": pl.Float64,   # half spread (divided by prev_spot_close)
            "opt_tick_size": pl.Float64, # option tick size (divided by prev_spot_close)
            "size": pl.Float64,          # total size
            "bid_size_pcnt": pl.Float64, # bid size as percentage of total size
            "mid_ret": pl.Float64,       # pcnt return in mid
            "dt_d": pl.Float64,            # time diff in days since last quote
            "dte": pl.Float64,           # dte in days computed for each option quote, adjusted for time of day: ref_dte - (t_ms / (6.5 * 3600 * 1000)) * (1 - overnight_fraction)
            "nms5": pl.Float64,          # normalized strike
            "intrinsic": pl.Float64,      # intrinsic value (divided by prev_spot_close)
            "is_intrinsic": bool,          # whether the option is near the money, defined as having intrinsic value at least 10% of the mid price
            "bid_vol": pl.Float64,       # bid vol
            "ask_vol": pl.Float64,       # ask vol
            "mid_vol": pl.Float64,       # mid vol
            "bid_delta": pl.Float64,      # bid delta
            "ask_delta": pl.Float64,      # ask delta
            "mid_delta": pl.Float64,     # mid delta
            "bid_gamma": pl.Float64,      # bid gamma
            "ask_gamma": pl.Float64,      # ask gamma
            "mid_gamma": pl.Float64,     # mid gamma
            "bid_vega": pl.Float64,       # bid vega
            "ask_vega": pl.Float64,       # ask vega
            "mid_vega": pl.Float64,       # mid vega
            "bid_theta": pl.Float64,       # bid theta
            "ask_theta": pl.Float64,       # ask theta
            "mid_theta": pl.Float64,       # mid theta
            # option trades
            "trade_t_d": pl.Float64,  # time diff in days since last trade
            "trade_price": pl.Float64,  # trade price (divided by prev_spot_close)
            "trade_size": pl.Float64,   # trade size 
            "trade_direction": pl.Float64, # trade direction: (trade_price - mid_price) / half_spread
            "trade_ret": pl.Float64,      # trade return: (trade_price - prev_trade_price) / prev_trade_price
            "trade_dt_d": pl.Float64,    # time diff in days since last trade
            # spot quotes
            # all spot quotes have trades
            "spot_quote_t_d": pl.Float64,   # time diff in days since last spot quote
            "spot_mid_price": pl.Float64,   # spot mid price (divided by prev_spot_close)
            "spot_half_spread": pl.Float64, # spot half spread (divided by prev_spot_close)
            "spot_size": pl.Float64,        # spot total size
            "spot_bid_size_pcnt": pl.Float64, # spot bid size as percentage of
            "spot_mid_ret": pl.Float64,     # spot mid price return: (spot_mid_price - prev_spot_mid_price) / prev_spot_mid_price
            "spot_quote_dt_d": pl.Float64,  # time diff in days since last spot quote
            # spot trades
            "spot_trade_t_d": pl.Float64,  # time diff in days since last spot trade
            "spot_trade_price": pl.Float64,  # spot trade price (divided by prev_spot_close)
            "spot_trade_size": pl.Float64,   # spot trade size  
            "spot_trade_direction": pl.Float64, # spot trade direction: (spot_trade_price - spot_mid_price) / spot_half_spread
            "spot_trade_ret": pl.Float64,    # spot trade return: (spot_trade_price - prev_spot_trade_price) / prev_spot_trade_price
            "spot_trade_dt_d": pl.Float64,    # time diff in days since last spot trade
            # ref data
            "ticker": str,
            "opt_ticker": str,
            "date": pl.Date,
            "exp_date": pl.Date,
            "is_call": bool,
            "is_american": bool,
            "strike": pl.Float64,         # divided by prev_spot_close
            "prev_spot_close": pl.Float64,
}
""" Schema returned by _opt_expiry_quote_time_series """

@_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.32",
    dependencies=[align_trades_to_quotes,_ticker_trade_series],
    uid=lambda ticker, opt_ticker, d, overnight_fraction, dte_exclude_half_days, alignment_config:\
        "opt_quotes/"+\
        f"{named_unique_filename48_8(f'onf={overnight_fraction}{',xclhlf' if dte_exclude_half_days else ''},{alignment_config}',dte_exclude_half_days,overnight_fraction,alignment_config)}/"+\
        f"{ticker}/{d}/{named_unique_filename48_8(opt_ticker)}"
)
def _opt_quote_time_series( ticker : str, opt_ticker : str, d : date, 
                            overnight_fraction : float = 0.2, 
                            dte_exclude_half_days : bool = True,
                            alignment_config : TradeQuoteAlignmentConfig = TradeQuoteAlignmentConfig(), *, verbose : Context|None = Context.quiet) -> pl.DataFrame:
    """
    Option quote time series aligned with spot trades.
    """
    verbose     = verbose if verbose is not None else Context.quiet
    spot_trades = _ticker_trade_series( ticker, d, alignment_config=alignment_config, verbose=verbose )

    """
    spot_trades schema:
        SCHEMA_ALIGN_QUOTES_TO_TRADES = {
            "ts_ns": np.dtype("datetime64[ns]"),
            "price": np.float64,
            "size": np.float64,
            "trade_code": np.uint32,
            "quote_ts_ns": np.dtype("datetime64[ns]"),
            "bid_price": np.float64,
            "bid_size": np.float64,
            "ask_price": np.float64,
            "ask_size": np.float64,
        }
    """

    with verbose.write_t(lambda : f"{ticker} {d}: Fetching time series for option {opt_ticker} {d}... ", end='') as tme:
        opt_quotes  = align_trades_to_quotes( opt_ticker, d, config=alignment_config, within_session=True, drop_unmatched=False ).sort("ts_ns")
        if len(opt_quotes) == 0:
            client     = MassiveClient()
            num_quotes = len( client.fetch_quotes(opt_ticker, d) )
            num_trades = len( client.fetch_trades(opt_ticker, d) )
            num_valid_quotes = len( filter_valid_quotes( opt_ticker, d))
            num_valid_trades = len( filter_valid_trades( opt_ticker, d))

            warnings.warn(f"{ticker} {d} option {opt_ticker}: No valid quotes were aligned with valid trades. Number of quotes: {num_quotes}, valid quotes: {num_valid_quotes}, number of trades: {num_trades}, valid trades: {num_valid_trades} -- skipping this option.")
            return opt_quotes
        verbose.write(lambda : f"done; found {len(opt_quotes)} quotes; this took {tme}.", head=False)

    """
    opt_quotes schema:
        SCHEMA_ALIGN_TRADES_TO_QUOTES = {
            "ts_ns": np.dtype("datetime64[ns]"),
            "bid_price": np.float64,
            "bid_size": np.float64,
            "ask_price": np.float64,
            "ask_size": np.float64,
            "trade_ts_ns": np.dtype("datetime64[ns]"),
            "price": np.float64,
            "size": np.float64,
            "trade_code": np.uint32,
        }    
    """

    spot_cols   = set(spot_trades.columns)
    spot_trades = spot_trades.rename({k: "spot_" + k for k in spot_cols})

    result = opt_quotes.join_asof(
            spot_trades,
            left_on="ts_ns",
            right_on="spot_ts_ns",
            strategy="backward" )
    result       = result.with_columns( *[ pl.col("spot_" + c).forward_fill().backward_fill() for c in spot_cols ] )
    result       = result.filter(pl.col("bid_price").is_not_null())
    assert len(result) > 0, f"{ticker} {d} option {opt_ticker}: No aligned quotes found for option {opt_ticker} on {d} after joining {len(opt_quotes)} quites with {len(spot_trades)} spot trades...?"

    # convert
    # -------

    client       = MassiveClient()
    ref_data     = client.option_contract_overview( opt_ticker, d )
    prev_close   = client.consistent_prev_close( ticker, d, exclude_half_days=False )
    strike       = float( ref_data["strike_price"] )
    expiry       = to_date( ref_data["expiration_date"] )
    is_american  = ref_data["exercise_style"].lower().startswith("a")
    is_call      = ref_data["contract_type"].lower().startswith("c") 
    open, close,\
    is_half_day  = session_open_close(d)
    diff_oc_us   = (close - open).total_seconds() * 1_000_000
    dte          = float( count_business_days_between( d, expiry, exclude_half_days=dte_exclude_half_days ) )

    result   = ( result.
            rename(dict(price="trade_price", size="trade_size", spot_price="spot_trade_price", spot_size="spot_trade_size", spot_ts_ns="spot_trade_ts_ns")).
            with_columns(
                ( pl.col("ask_price") / prev_close ).alias("ask_price"),
                ( pl.col("bid_price") / prev_close ).alias("bid_price"),
                ( pl.col("trade_price") / prev_close ).alias("trade_price"),
                ( pl.col("spot_ask_price") / prev_close ).alias("spot_ask_price"),
                ( pl.col("spot_bid_price") / prev_close ).alias("spot_bid_price"),
                ( pl.col("spot_trade_price") / prev_close ).alias("spot_trade_price"),
            ).
            with_columns(
                ( pl.col("ts_ns").sub(open).dt.total_microseconds() / diff_oc_us ).alias("t_d"),
                ( pl.col("trade_ts_ns").sub(open).dt.total_microseconds() / diff_oc_us ).alias("trade_t_d"),
                ( pl.col("spot_trade_ts_ns").sub(open).dt.total_microseconds() / diff_oc_us ).alias("spot_trade_t_d"),
                ( pl.col("spot_quote_ts_ns").sub(open).dt.total_microseconds() / diff_oc_us ).alias("spot_quote_t_d"),  
                ( (pl.col("ask_price") + pl.col("bid_price")) / 2.0 ).alias("mid_price"),
                ( (pl.col("ask_price") - pl.col("bid_price")) / 2.0 ).alias("half_spread"),
                ( (pl.col("spot_ask_price") + pl.col("spot_bid_price")) / 2.0 ).alias("spot_mid_price"),
                ( (pl.col("spot_ask_price") - pl.col("spot_bid_price")) / 2.0 ).alias("spot_half_spread"),
                ( pl.col("bid_size") + pl.col("ask_size") ).alias("size"),
                ( pl.col("bid_size") / (pl.col("bid_size") + pl.col("ask_size")).clip(lower_bound=1) ).alias("bid_size_pcnt"),
                ( pl.col("spot_bid_size") + pl.col("spot_ask_size") ).alias("spot_size"),
                ( pl.col("spot_bid_size") / (pl.col("spot_bid_size") + pl.col("spot_ask_size")).clip(lower_bound=1) ).alias("spot_bid_size_pcnt"),     
           ).
            with_columns(
                ( (pl.col("trade_price") - pl.col("mid_price")) / pl.col("half_spread") ).alias("trade_direction"),
                ( (pl.col("spot_trade_price") - pl.col("spot_mid_price")) / pl.col("spot_half_spread") ).alias("spot_trade_direction"),
            ).
            with_columns(
                pl.col("mid_price").pct_change().alias("mid_ret"),
                pl.col("trade_price").pct_change().alias("trade_ret"),
                pl.col("spot_trade_price").pct_change().alias("spot_trade_ret"),
                pl.col("spot_mid_price").pct_change().alias("spot_mid_ret"),
                ( dte + ( 1. - pl.col("t_d") ) * (1. - overnight_fraction)).alias("dte"),
                pl.col("t_d").diff().alias("dt_d"),
                pl.col("trade_t_d").diff().alias("trade_dt_d"),
                pl.col("spot_trade_t_d").diff().alias("spot_trade_dt_d"),
                pl.col("spot_quote_t_d").diff().alias("spot_quote_dt_d"),
            ).
            with_columns(
                ( ( strike / pl.col("spot_mid_price") ).log() * (( 1.+NMS5.EPS )/( pl.col("dte")/DAYS_PER_YEAR + NMS5.EPS )).sqrt() / NMS5.SCALE ).alias("nms5"),
                pl.lit(expiry).alias("exp_date"),
                pl.lit(d).alias("date"),
                pl.lit(strike/prev_close).alias("strike"),
                pl.lit(ticker).alias("ticker"),
                pl.lit(opt_ticker).alias("opt_ticker"),
                pl.lit(is_american).alias("is_american"),
                pl.lit(is_call).alias("is_call"),
                pl.lit(prev_close).alias("prev_spot_close"),
                pl.lit(OPT_TICK_SIZE/prev_close).alias("opt_tick_size"),
            ).
            drop("bid_price", "ask_price", "spot_ask_price", "spot_bid_price",
                "ts_ns", "trade_ts_ns", "spot_trade_ts_ns", "spot_quote_ts_ns", 
                "bid_size", "ask_size", "spot_bid_size", "spot_ask_size",
                "trade_code", "spot_trade_code")
    )

    opt_mid         = result["mid_price"].to_numpy()
    opt_half_spread = result["half_spread"].to_numpy()
    opt_tick_size   = result["opt_tick_size"].to_numpy()
    tte_y           = result["dte"].to_numpy() / DAYS_PER_YEAR
    spot            = result["spot_mid_price"].to_numpy()
    strike          = result["strike"].to_numpy() 
    is_call         = result["is_call"].to_numpy()

    assert np.all( np.isfinite(opt_mid) ), f"{ticker} {d} option {opt_ticker}: Found non-finite values in mid_price after conversion...?"
    assert np.all( np.isfinite(opt_half_spread) ), f"{ticker} {d} option {opt_ticker}: Found non-finite values in half_spread after conversion...?"
    assert np.all( np.isfinite(tte_y) ), f"{ticker} {d} option {opt_ticker}: Found non-finite values in tte_y after conversion...?"
    assert np.all( np.isfinite(spot) ), f"{ticker} {d} option {opt_ticker}: Found non-finite values in spot after conversion...?"
    assert np.all( np.isfinite(strike) ), f"{ticker} {d} option {opt_ticker}: Found non-finite values in strike after conversion...?"
    assert np.all( opt_half_spread >= opt_tick_size/2.-1E-8 ), f"{ticker} {d} option {opt_ticker}: half spread should not be below half tick size, but found half_spread {np.min(np.abs(opt_half_spread-opt_tick_size/2.)):.4g}"
    opt_ask = opt_mid + opt_half_spread
    opt_bid = opt_mid - opt_half_spread
    assert np.all( opt_bid >= opt_tick_size/2.-1E-8 ), f"{ticker} {d} option {opt_ticker}: half spread should not be below half tick size, but found half_spread {np.min(opt_bid-opt_tick_size/2.):.4g}"

    intrinsic    = np.where( is_call, np.maximum(0., spot - strike), np.maximum(0., strike - spot) )
    is_intrinsic = intrinsic + opt_tick_size*1.1 >= opt_bid 
    result = result.with_columns( 
        pl.Series("intrinsic", intrinsic),
        pl.Series("is_intrinsic", is_intrinsic) )

    sqrtT    = np.sqrt(tte_y)
    what     = dict(bid=opt_bid, ask=opt_ask, mid=opt_mid)
    rel_k    = strike / spot

    for w, price in what.items():
        vols   = bs.implied( k=rel_k, prices=price, is_call=is_call, sqrtT=sqrtT, on_exceed_bounds="warn" )
        r      = bs(k=rel_k, vol=vols, sqrtT=sqrtT, is_call=is_call, what=bs.VEGA|bs.DELTA|bs.GAMMA|bs.THETA)
        r.vol  = vols
        result = result.with_columns(  [ pl.Series(f"{w}_{k}", v) for k, v in r.items() ] )
            
    verify_schema( result, OPT_QUOTE_BY_EXPIRY_SCHEMA, header=lambda : f"{ticker} {d} expiry {expiry} {opt_ticker}: Schema validation failed for enriched option quote time series" )
    return result

@_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.16",
    dependencies=[_opt_quote_time_series, filter_contracts],
    uid=lambda ticker, d, expiry, opt_ticker_data, option_quotes_day_config: \
    f"opt_expiry_quotes/{named_unique_filename48_8(str(option_quotes_day_config),option_quotes_day_config)}/{ticker}/{d}/{expiry}/tickers {unique_hash8(set(opt_ticker_data['opt_ticker'].to_list()))}"
)
def _opt_expiry_quote_time_series( ticker : str, 
                                   d : date, expiry : date, 
                                   opt_ticker_data : pl.DataFrame,
                                   option_quotes_day_config : OptionQuotesDayConfig, *,
                                   exclude_half_days_from_prev_business_day : bool = True,
                                   pool_config : PoolConfig = PoolConfig.off(),
                                   verbose : Context|None = Context.quiet) -> pl.DataFrame:
    """
    Option quote time series aligned with spot trades, for all options of a given expiry.
    """
    verbose      = verbose if verbose is not None else Context.quiet
    opt_tickers  = set(opt_ticker_data['opt_ticker'].to_list())
    open, close,\
    is_half_day  = session_open_close(d)
    pool         = pool_config.pool()

    with verbose.write_t(lambda : f"{ticker} {d} expiry {expiry}: Computing enriched time series for {len(opt_tickers)} options:", end='') as tme:
        is_single     = pool.is_no_pool or pool.is_threading
        cnt           = 0
        tme           = verbose.timer()
        results       = []
        excluded      = []
        
        for opt_ticker, _ in pool.parallel( { opt_ticker: pool.delayed( _opt_quote_time_series )( ticker, 
                                                                                                  opt_ticker, 
                                                                                                  d, 
                                                                                                  dte_exclude_half_days=option_quotes_day_config.dte_exclude_half_days,
                                                                                                  overnight_fraction=option_quotes_day_config.overnight_fraction,
                                                                                                  alignment_config=option_quotes_day_config.alignment_config, 
                                                                                                  verbose=verbose(1) if is_single else None, cache_generate_only=True ) for opt_ticker in opt_tickers } ):
            cnt += 1
            r = _opt_quote_time_series( ticker, opt_ticker, d,
                                        dte_exclude_half_days=option_quotes_day_config.dte_exclude_half_days,
                                        overnight_fraction=option_quotes_day_config.overnight_fraction, 
                                        alignment_config=option_quotes_day_config.alignment_config, 
                                        verbose=Context.quiet, override_cache_mode="cacheonly" )

            if len(r) == 0:
                verbose.write(lambda : f"\r{ticker} {d} expiry {expiry}: Computing enriched time series for {len(opt_tickers)} options: processed {opt_ticker} {cnt}/{len(opt_tickers)}: this option had no valid quotes -- {math.ceil(float(cnt)/float(len(opt_tickers))*100)}% done; so far this took {tme}...", end='')
                excluded.append(opt_ticker)
            else:
                if cnt==1 or tme.interval_test(1.):
                    verbose.write(lambda : f"\r{ticker} {d} expiry {expiry}: Computing enriched time series for {len(opt_tickers)} options: processed {opt_ticker} {cnt}/{len(opt_tickers)} ... {math.ceil(float(cnt)/float(len(opt_tickers))*100)}% done; so far this took {tme}...", end='')

                assert r['opt_ticker'].n_unique() == 1 and r['opt_ticker'].unique()[0] == opt_ticker, f"{ticker} {d} expiry {expiry} {opt_ticker}: Expected all rows to have the same opt_ticker {opt_ticker}, but found multiple opt_tickers in the result...?"
                verify_schema( r, OPT_QUOTE_BY_EXPIRY_SCHEMA, header=lambda : f"{ticker} {d} expiry {expiry} {opt_ticker}: Schema validation failed for enriched option quote time series" )
                results.append(r)
            del r

    results      = pl.concat(results).sort("t_d", "opt_ticker")
    verbose.write(lambda : f"\r{ticker} {d}: Computed enriched time series at expiry {expiry} for {len(opt_tickers)-len(excluded)} out of {len(opt_tickers)} options; this took {tme}.")
    return results

@version("0.1", dependencies=[_opt_quote_time_series, filter_contracts])
def generate_day( ticker : str, d : date, 
                  option_quotes_day_config : OptionQuotesDayConfig, *,
                  pool_config : PoolConfig = PoolConfig.off(),
                  cache_generate_only : bool = False,
                  override_cache_mode : CacheMode|str|None = None,
                  verbose     : Context = Context.quiet
                  ):
    """
    Generate polars data frame per expiry for all eligible options.
    """

    verbose.write(lambda : f"{ticker} {d}: Starting data generation with config {option_quotes_day_config}:")

    # contract selection
    # ------------------

    df = filter_contracts( ticker, d=d, exclude_half_days=True, drop_no_trades_today=True,
                           contracts_filter=option_quotes_day_config.contract_filter, pool_config=pool_config, verbose=verbose )
    df = df.filter(pl.col("reject_code") == REJECTED.USED)

    # reference data
    # --------------

    expiries  = df["exp_date"].unique().sort().to_list()
    prev_spot = float( df["spot_ref_prev_close"].unique()[0] )

    ref_data    = ( df.with_columns( pl.format("{} {} DTE {} {} {} @ {} ({}%)", pl.col("ticker"), pl.col("exp_date"), pl.col("days_to_exp"), pl.col("exercise_style"), pl.col("contract_type"), pl.col("strike_price").round(2), (100.0 * pl.col("strike_price")/prev_spot).round(2)).alias("opt_str") ).
                    select( pl.col("ticker").alias("opt_ticker"), 
                            pl.col("exp_date"), 
                            pl.col("nms5").alias("ref_nms5"),
                            pl.col("exercise_style")
                                .str.to_lowercase()
                                .str.starts_with("a")
                                .alias("is_american"),
                            pl.col("contract_type")
                                .str.to_lowercase()
                                .str.starts_with("c")
                                .alias("is_call"),                                   
                            ( pl.col("strike_price")/prev_spot ).alias("strike"),
                                pl.col("days_to_exp").alias("ref_dte"), 
                                pl.lit(prev_spot).alias("prev_spot") ).with_columns( pl.lit(ticker).alias("ticker") )
            )
    ref_data      = { expiry: ref_data.filter(pl.col("exp_date") == expiry).drop("exp_date") for expiry in expiries }

    if option_quotes_day_config.max_num_options_per_expiry is not None:
        verbose.write(lambda : f"{ticker} {d}: Limiting to at most {option_quotes_day_config.max_num_options_per_expiry} options per expiry, selected by smallest absolute nms5:")
        for expiry, ref_df in ref_data.items():
            n_cnt  = len(ref_df)
            ref_df = (
                ref_df
                .with_columns(abs_nms5=pl.col("ref_nms5").abs())
                .sort("abs_nms5")
                .head(option_quotes_day_config.max_num_options_per_expiry)
                .drop("abs_nms5")
            )
            ref_data[expiry] = ref_df
            min_nms5 = ref_df["ref_nms5"].min()
            max_nms5 = ref_df["ref_nms5"].max()
            verbose.report(1,lambda : f"{ticker} {d} expiry {expiry}: Selected {len(ref_df)}/{n_cnt} options, sorted by smallest absolute nms5. Resulting options range from nms5 {min_nms5:.3f} to {max_nms5:.3f}.")

    # time series
    # -----------

    with verbose.write_t(lambda : f"{ticker} {d}: Starting time series generation for expiries {fmt_list(expiries)} with config {option_quotes_day_config} and pool config {pool_config}:") as tme:
        results = {}
        for expiry in expiries:
            opt_ticker_data = ref_data[expiry]
            r = _opt_expiry_quote_time_series( ticker, d, expiry, opt_ticker_data, option_quotes_day_config, pool_config=pool_config,
                                            verbose=verbose, cache_generate_only=cache_generate_only,
                                            override_cache_mode=override_cache_mode )
            results[expiry] = r
        verbose.write(lambda : f"{ticker} {d}: Completed time series generation for expiries {fmt_list(expiries)}; this took {tme}.")
    return results



# ================================================================
# Command-line interface
# ================================================================

import click
from quantdata.widb import _cli_defaults
import datetime as dt
from cdxcore.util import verify_inp, fmt_date

@click.command()
@click.argument("ticker", required=True, type=str)
@click.option("--period",nargs=2, type=click.DateTime(formats=["%Y-%m-%d"]), default=_cli_defaults.period, metavar="START END", help="Period YYYY-MM-DD YYYY-MM-DD")
@click.option('--workers', default=_cli_defaults.workers, help='Number of workers', type=int)
@click.option('--threading', is_flag=True, help="Use only threading, no multiprocessing")
@click.option('--refresh', is_flag=True, help="Refresh data (this will only refresh the top level files)")

def main(ticker : str,
        period : list[dt.datetime], 
        workers : int, 
        threading : bool, 
        refresh : bool):

    verify_inp( period is not None and len(period) == 2, "A start and end date must be specified")

    start, end         = period
    date_range         = DateRange(start=to_date(start), end=to_date(end), exclude_half_days=False )
    dates              = sorted( set(date_range.get_business_days())-set(BAD.DAYS_ALL) )
    pool_config        = PoolConfig(num_workers=workers, threading=threading)

    for d in dates:
        _ = generate_day(ticker, d, OptionQuotesDayConfig(), pool_config=pool_config, verbose=Context.all, cache_generate_only=True )


if __name__ == "__main__":
    main()
