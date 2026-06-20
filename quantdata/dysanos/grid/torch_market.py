"""
Grid Torch Market
^^^^^^^^^^^^^^^^^

Defines the market data used for Grid models.
"""

from cdxcore.pretty import PrettyObject
from cdxcore.version import version
from quantdata.dysanos.grid.interpolated import GridConfig, interpolate_time_series, Context, OptionsFilter, RawDayFilter, ExpiryFilter, MkArbFree
from quantdata.biz_calendar import DateRange
from quantdata.pool import PoolConfig
from datetime import date
from quantdata.torch import BaseEnvironment, torch
import numpy as np
from dataclasses import dataclass, field

@version("0.1", dependencies=[GridConfig, OptionsFilter, RawDayFilter, ExpiryFilter, MkArbFree, interpolate_time_series])
@dataclass(frozen=True)
class GridMarketDefinition:
    ticker         : str = "SPX"                                                                    # ticker  
    date_range     : DateRange = DateRange(start=date(2020, 1, 1), end=date(2025, 8, 30))           # date range
    grid_config    : GridConfig = GridConfig(smoothness=0.25, mode="nms", grid_num_strikes=20)      # option grid to model
    raw_day_filter : RawDayFilter = RawDayFilter()                                                  # raw day filter
    expiry_filter  : ExpiryFilter = ExpiryFilter()                                                  # expiry filter
    mk_arb_free    : MkArbFree = MkArbFree()                                                        # market arbitrage free
    options_filter : OptionsFilter = OptionsFilter()                                                # options filter

    @version("0.1", dependencies=[interpolate_time_series])
    def load( self, *, pool_config : PoolConfig = PoolConfig.off(), verbose : Context = Context.all ) -> PrettyObject:
        """ Load the market data; this is a wrapper around :func:`interpolate_time_series` that returns a PrettyObject with the market data. """
        data = PrettyObject( interpolate_time_series( self.ticker,
                                                      date_range = self.date_range, 
                                                      options_filter=self.options_filter,
                                                      raw_day_filter=self.raw_day_filter,
                                                      expiry_filter=self.expiry_filter,
                                                      mk_arb_free=self.mk_arb_free,
                                                      grid_config=self.grid_config, 
                                                      pool_config=pool_config, 
                                                      verbose=verbose ) )
        data.grid_market_def = self

        # as extreme as it sounds the round trip vols -> variance -> vols is pretty inexact numerically.
        data.variances = (data.atm_vols**2) * data.expiries
        del data.atm_vols
        return data

