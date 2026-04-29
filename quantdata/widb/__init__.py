"""
WRDS IvyDB handling
"""

from quantdata import env, SubDir, Context, version#NOQA
from quantdata.widb.client import WIDBClient, interpolate_zero_curve

class cli_defaults:
    period = ("2005-09-01", "2025-09-01") #: Default start and end date for CLI commands.
    workers = 4 #: Default number of workers for CLI commands.