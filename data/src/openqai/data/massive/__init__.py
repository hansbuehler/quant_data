"""
Massive.com Data Module
^^^^^^^^^^^^^^^^^^^^^^^

Data pipeline for `massive.com <https://massive.com/>`_ market data.
"""

from openqai.core import env, SubDir, Context, version#NOQA
from openqai.data.massive.client import MassiveClient

_locations_massive = MassiveClient.locations

class _cli_defaults:
    period = (str(MassiveClient.START_DATE), "2025-12-01") #: Default start and end date for CLI commands.
    workers = 4 #: Default number of workers for CLI commands.
