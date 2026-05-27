"""
Generatr

"""


from datetime import date
from cdxcore.util import fmt_list
import polars as pl

from quantdata.massive.client import MassiveClient
from quantdata.pool import PoolConfig
from quantdata.massive.options.contracts import ContractsFilter, filter_contracts, REJECTED

from cdxcore.version import version
from cdxcore.verbose import Context
from dataclasses import dataclass, field
from cdxcore.uniquehash import unique_hash8 


@version("0.1", dependencies=[ContractsFilter])
@dataclass(frozen=True)
class KalmanConfig:
    contract_filter : ContractsFilter = ContractsFilter()
    max_num_options : int|None = 200




def generate_day( ticker : str, d : date, 
                  kalman_config : KalmanConfig, *,
                  pool_config : PoolConfig = PoolConfig.off(),
                  verbose     : Context = Context.quiet
                  ):
   


   # get all options on this day we care about
    df = filter_contracts( ticker=ticker, d=d, contracts_filter=kalman_config.contract_filter, pool_config=pool_config, verbose=verbose )
    df = df.filter(pl.col("reject_code") == REJECTED.NONE)

    # get 200 options with smallest absolute nms5
    if kalman_config.max_num_options is not None:
        n_cnt = len(df)
        df = df.with_columns(abs_nms5=pl.col("nms5").abs()).sort("abs_nms5").head(kalman_config.max_num_options)
        verbose.write(lambda : f"{ticker} {d}: Selected {len(df)}/{n_cnt} options with smallest absolute nms5")

    expiries = df["expiry"].unique().sort()
    verbose.write(lambda : f"{ticker} {d}: Found expiries {fmt_list(sorted(expiries))}")



