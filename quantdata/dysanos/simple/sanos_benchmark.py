"""
Fit SANOS to the market using linear programming.
"""





from attr import dataclass

from quantdata.dysanos.simple.sampler import SimpleSampler, SimpleConfig, SamplingConfig, ExpiryFilter, RawDayFilter, SampleDaysConfig, DateRange, PoolConfig, sample_simple_days
from datetime import date


from cdxcore.verbose import Context
from cdxcore.subdir import SubDir, CacheMode
from cdxcore.pretty import PrettyObject
from dataclasses import dataclass

@dataclass(frozen=True)
class SanosBenchmark:
    label  : str

    


def create_sanos_benchmark( label  : str,
                            ticker : str,
                            simple_config : SimpleConfig, *,
                            pool_config : PoolConfig = PoolConfig.off(),
                            override_cache_mode : CacheMode|None = None,
                            verbose : Context = Context.all ):
    """
    Create SANOS Fitting Benchmark
    
    
    """
    data = PrettyObject(
            sample_simple_days( 
                ticker = ticker,
                simple_config = simple_config,
                pool_config = pool_config,
                override_cache_mode = override_cache_mode,
                verbose = verbose
            )
        )
        
    
    """

    days = sample_simple_days(  ticker : str,
                         simple_config : SimpleConfig, *,
                         pool_config : PoolConfig = PoolConfig.off(),
                         override_cache_mode : CacheMode|None = None,
                         verbose : Context = Context.all ) -> dict:



def fit_sanos( sampler : SimpleSampler ):

