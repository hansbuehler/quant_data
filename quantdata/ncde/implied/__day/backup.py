


from datetime import date
from cdxcore.jcpool import JCPool
from cdxcore.pretty import PrettyObject
from cdxcore.subdir import SubDir
from cdxcore.util import fmt_list
from cdxcore.uniquehash import named_unique_filename48_8
import polars as pl

from quantdata.biz_calendar import prev_business_day
from quantdata.massive.client import MassiveClient
from quantdata.pool import PoolConfig
from quantdata.massive.options.contracts import ContractsFilter, filter_contracts, REJECTED
from quantdata.massive.processing import TradeQuoteAlignmentConfig, align_trades_to_quotes, align_quotes_to_trades

from cdxcore.version import version
from cdxcore.verbose import Context
from dataclasses import asdict, dataclass, field
from cdxcore.uniquehash import unique_hash8 
import numpy as np
import math

from quantdata.ncde.implied.day import _ncde_implied_day_root as _cache
from IPython.display import display, Markdown, clear_output


@version("0.1", dependencies=[ContractsFilter])
@dataclass(frozen=True)
class KalmanConfig:
    contract_filter : ContractsFilter = ContractsFilter()
    max_num_options_per_expiry : int|None = 1000
    trade_match_lookback_ms: int = 250
    trade_match_lookfwd_ms: int = 250

    def __str__(self):
        return f"mxn={self.max_num_options_per_expiry},{self.contract_filter}"

@_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.1",dependencies=[align_quotes_to_trades],
    uid=lambda ticker, d: f"{ticker}/{d}/{ticker}"
)
def ticker_trade_series( ticker : str, d : date, *, verbose : Context|None = Context.quiet) -> pl.DataFrame:
    """ Cached ticker series """
    verbose     = verbose if verbose is not None else Context.quiet
    with verbose.write_t(lambda : f"{ticker} {d}: Fetching time series for {ticker} {d}... ", end='') as tme:
        df = align_quotes_to_trades( ticker, d, within_session=True, drop_unmatched=False )
        verbose.write(lambda : f"done; found {len(df)} quotes; this took {tme}.", head=False)
    return df.sort("ts_ns")
    
@_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.14",
    dependencies=[align_trades_to_quotes,ticker_trade_series],
    uid=lambda ticker, opt_ticker, d: f"{ticker}/{d}/{named_unique_filename48_8(opt_ticker)}"
)
def opt_quote_time_series( ticker : str, opt_ticker : str, d : date, *, verbose : Context|None = Context.quiet):
    """
    Option quote time series aligned with spot trades.
    """
    verbose     = verbose if verbose is not None else Context.quiet
    spot_trades = ticker_trade_series( ticker, d, verbose=verbose )

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
        opt_quotes  = align_trades_to_quotes( opt_ticker, d, within_session=True, drop_unmatched=False ).sort("ts_ns")
        if len(opt_quotes) == 0:
            # this should not happen: we previous checked that there are quotes.
            client     = MassiveClient()
            has_quotes = client.has_any_quotes(opt_ticker, d)
            has_trades = client.has_any_trades(opt_ticker, d)
            raise RuntimeError(f"{ticker} {d} option {opt_ticker}: No quotes found for option {opt_ticker} on {d}...? has_quotes={has_quotes}, has_trades={has_trades}")
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
    result      = result.with_columns( *[ pl.col("spot_" + c).forward_fill().backward_fill() for c in spot_cols ] )
    result      = result.filter(pl.col("bid_price").is_not_null())
    assert len(result) > 0, f"{ticker} {d} option {opt_ticker}: No aligned quotes found for option {opt_ticker} on {d} after joining {len(opt_quotes)} quites with {len(spot_trades)} spot trades...?"
    return result

def generate_day( ticker : str, d : date, 
                  kalman_config : KalmanConfig, *,
                  pool_config : PoolConfig = PoolConfig.off(),
                  verbose     : Context = Context.quiet
                  ):
   


    verbose.write(lambda : f"{ticker} {d}: Starting data generation with config {kalman_config} and pool config {pool_config}")

    # contract selection
    # ------------------

    df = filter_contracts( ticker, d=d, exclude_half_days=True, drop_no_trades_today=True,
                           contracts_filter=kalman_config.contract_filter, pool_config=pool_config, verbose=verbose )
    df = df.filter(pl.col("reject_code") == REJECTED.USED)

    dfx = df.filter(pl.col("ticker") == "O:SPY250123C00614000")
    if len(dfx) == 0:
        print("O:SPY250123C00614000 not found")
    else:
        print(dfx)

    # reference data
    # --------------

    expiries  = df["exp_date"].unique().sort().to_list()
    prev_spot = float( df["spot_ref_prev_close"].unique()[0] )

    ref_data    = ( df.with_columns( pl.format("{} {} DTE {} {} {} @ {} ({}%)", pl.col("ticker"), pl.col("exp_date"), pl.col("days_to_exp"), pl.col("exercise_style"), pl.col("contract_type"), pl.col("strike_price").round(2), (100.0 * pl.col("strike_price")/prev_spot).round(2)).alias("opt_str") ).
                    select( pl.col("ticker").alias("opt_ticker"), 
                            pl.col("exp_date"), 
                            pl.col("nms5"),
                            pl.col("exercise_style")
                                .str.to_lowercase()
                                .str.starts_with("a")
                                .alias("is_american"),
                            pl.col("contract_type")
                                .str.to_lowercase()
                                .str.starts_with("c")
                                .alias("is_call"),                                   
                            ( pl.col("strike_price")/prev_spot ).alias("strike"),
                                pl.col("days_to_exp").alias("dte"), 
                                pl.lit(prev_spot).alias("prev_spot") ).with_columns( pl.lit(ticker).alias("ticker") )
            )
    ref_data      = { expiry: ref_data.filter(pl.col("exp_date") == expiry).drop("exp_date") for expiry in expiries }

    if kalman_config.max_num_options_per_expiry is not None:
        verbose.write(lambda : f"{ticker} {d}: Limiting to at most {kalman_config.max_num_options_per_expiry} options per expiry, selected by smallest absolute nms5:")
        for expiry, ref_df in ref_data.items():
            n_cnt = len(ref_df)
            ref_df   = ref_df.with_columns(abs_nms5=pl.col("nms5").abs()).sort("abs_nms5").head(kalman_config.max_num_options_per_expiry).drop("abs_nms5")
            min_nms5 = ref_df["nms5"].min()
            max_nms5 = ref_df["nms5"].max()
            verbose.report(1,lambda : f"{ticker} {d} expiry {expiry}: Selected {len(ref_df)}/{n_cnt} options, sorted by smallest absolute nms5. Resulting options range from nms5 {min_nms5:.3f} to {max_nms5:.3f}.")

    # time series
    # -----------

    opt_tickers    = set()
    opt_to_expiry  = {}
    for expiry, exp_ref_data in ref_data.items():
        exp_opt_tickers = set(exp_ref_data["opt_ticker"].unique().to_list())
        assert len(opt_tickers & exp_opt_tickers) == 0, f"{ticker} {d}: Mismatch in option tickers for expiry {expiry}: expected {opt_tickers^exp_opt_tickers} already seen...?"
        opt_tickers     |= exp_opt_tickers
        for opt_ticker in exp_opt_tickers:
            opt_to_expiry[opt_ticker] = expiry
        del exp_opt_tickers
    del expiry, exp_ref_data

    with verbose.write_t(lambda : f"{ticker} {d}: Computing enriched time series for {len(opt_tickers)} options for expiries {fmt_list(sorted(expiries))}:") as tme:
        def pull():
            pool          = JCPool( **asdict(pool_config) )
            is_single     = pool_config.num_workers in [0,1] and pool_config.threading
            results       = { e: {} for e in expiries }
            cnt           = 0
            tme           = verbose.timer()

            for opt_ticker, _ in pool.parallel( { opt_ticker: pool.delayed( opt_quote_time_series )( ticker, opt_ticker, d, verbose=verbose(1) if is_single else None, cache_generate_only=True ) for opt_ticker in opt_tickers } ):
                cnt += 1
                r = opt_quote_time_series( ticker, opt_ticker, d, verbose=Context.quiet, override_cache_mode="cacheonly" )
                assert len(r) > 0, f"{ticker} {d}: No data found for option {opt_ticker} - expected at least one quote aligned with a spot trade. This may indicate an issue with the data or the alignment process... ref_data {ref_data}"   

                if cnt==1 or tme.interval_test(1.):
                    verbose.report(1,lambda : f"\r{ticker} {d}: Processing option {opt_ticker} {cnt}/{len(opt_tickers)} ... {math.ceil(float(cnt)/float(len(opt_tickers))*100)}% done; so far this took {tme}...", end='')

                r = r.with_columns( pl.lit(opt_ticker).alias("opt_ticker") )
                expiry = opt_to_expiry[opt_ticker]
                results[expiry][opt_ticker] = r

            assert set(results) == set(expiries), f"{ticker} {d}: Mismatch in expiries: expected {sorted(expiries)}, got {sorted(results)}"
            return results

        results = pull()

        assert set(results) == set(ref_data), f"{ticker} {d}: Mismatch in expiries between results and reference data: expected {sorted(expiries)}, got {sorted(results)}, expected {sorted(ref_data)}"
        assert set(ref_data) == set(expiries), f"{ticker} {d}: Mismatch in expiries between results and reference data: expected {sorted(expiries)}, got {sorted(results)}, expected {sorted(ref_data)}"

        # merge per-option results into per-expiry results
        for expiry, opt_dict in results.items():
            expiry_ref_data = ref_data.pop(expiry)

            opt_tickers     = set( expiry_ref_data["opt_ticker"].to_list() )
            opt_tick_d      = set( opt_dict )

            assert opt_tick_d == opt_tickers, f"{ticker} {d}: Mismatch in options for expiry {expiry}: expected {sorted(opt_tick_d)}, got {sorted(opt_tickers)}"

            opt_dfs = []
            for opt_ticker, df in opt_dict.items():
                opt_dfs.append( df )
            opt_dfs         = pl.concat(opt_dfs).sort("ts_ns")
            opt_tickers     = set( expiry_ref_data["opt_ticker"].to_list() )
            df_tickers      = set( opt_dfs["opt_ticker"].unique().to_list() )
            if opt_tickers != df_tickers:
                missing = opt_tickers - df_tickers
                extra   = df_tickers - opt_tickers
                assert opt_tickers == df_tickers, f"{ticker} {d}: Mismatch in tickers for expiry {expiry}: df_ticker is missing: {fmt_list(missing)}, df_ticker has extra: {fmt_list(extra)}"
            results[expiry] = dict(opt_data=opt_dfs, ref_data=expiry_ref_data)

        verbose.write(lambda : f"\r{ticker} {d}: Computed enriched time series for {len(df)} options; this took {tme}.")
    return results

clear_output()
d = prev_business_day(date(2025,1,23), exclude_half_days=True, count_start=True )
pool_config= PoolConfig(num_workers=0, threading=False) 
#df = opt_quote_time_series("SPY","O:SPY250623P00601000",d, verbose=Context.all)

results = generate_day("SPY", d, KalmanConfig(), pool_config=pool_config, verbose=Context.all)

for d, df_dict in results.items():
    display(Markdown(f"## Expiry {d}"))
    display(Markdown(f"#### Expiry {d} Opt Data"))
    display(df_dict["opt_data"])
    display(Markdown(f"#### Expiry {d} Ref Data"))
    display(df_dict["ref_data"])
#add_business_days: 2025-01-23 0 True True [datetime.date(2025, 1, 23), datetime.date(2025, 1, 22), datetime.date(2025, 1, 21), datetime.date(2025, 1, 17), datetime.date(2025, 1, 16)] -> 2025-01-23
