"""
WRDS Option Metric Ivy DB Data Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Key entry points:

* :class:`WIDBClient`: main client for accessing cached raw WIDB data.
* :func:`collect_raw_day`: basic filtering and cleaning of raw daily WIDB data.
* :func:`enrich_day`: computation of synethetic forwards, implied vols, and greeks
* :func:`mk_arb_free`: making sure that it is possible to fit an arb-free fit through bid/ask. Remove strikes and even expiries
  whenever necessary.
"""

from quantdata import env, SubDir, Context, version#NOQA
from quantdata.widb.client import WIDBClient, interpolate_zero_curve

_locations_widb = WIDBClient.locations

class _cli_defaults:
    period = ("2005-01-01", "2025-09-01") #: Default start and end date for CLI commands.
    workers = 4 #: Default number of workers for CLI commands.