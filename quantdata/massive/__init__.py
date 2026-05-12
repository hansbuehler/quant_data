"""
Data pipeline for `massive.com <https://massive.com/>`_ market data.
"""

from quantdata import env, SubDir, Context, version#NOQA
from quantdata.massive.client import MassiveClient

class _cli_defaults:
    period = (str(MassiveClient.START_DATE), "2025-12-01") #: Default start and end date for CLI commands.
    workers = 4 #: Default number of workers for CLI commands.
