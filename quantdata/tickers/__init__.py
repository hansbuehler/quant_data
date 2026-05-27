"""
Ticker Data Module
^^^^^^^^^^^^^^^^^^

This module contains ticker specific analytics which collect data from a number of data classes.
You may need access to different data source to be able to use some of the functions contained
in this sub-module.
"""

from quantdata import env, version, SubDir, Context#NOQA
from quantdata.massive import MassiveClient

_locations_tickers = env.locations.processing("tickers")

