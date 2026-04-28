"""
Ticker related analytics
"""

from quantdata import env, version, SubDir, Context#NOQA
from quantdata.massive import MassiveClient

tickers_cache = env.locations.processing("tickers")

