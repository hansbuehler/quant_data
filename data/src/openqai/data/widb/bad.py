"""
Bad data directory
==================

This file contains all known bad data points:

* General bad days
* Bad days per ticker
* Bad options
* Likely last valid date

Please update this file as you experience errors in the data.
"""

from __future__ import annotations
import quantdata as quantdata#NOQA
import datetime as dt

class BAD:
    """
    Container class for BAD data points in various categories.
    """

    LAST_LIKELY_VALID_DATE = dt.date(2025, 9, 1)
    """ Likely last valid date in the data. This is an estimate based on the most recent data issues. """

    DAYS_ALL = { 
        dt.date(2004, 10, 11), # no zero curve  
        dt.date(2004, 11, 11), # no zero curve
        dt.date(2007, 11, 12), # no zero curve
        dt.date(2018, 12, 4), # expiries with zero business days?
        dt.date(2025, 1,8) # expiries with zero business days?
        } 
    """ Days where some core information is missing, e.g. zero curves. """

    DAYS_TICKER = {'SPX': {
                        dt.date(2006, 11, 17),  # no forward for 1
                        dt.date(2012, 11, 17),  # no forward for 1
                        dt.date(2015, 8, 21)    # cannot find arb-free surface
                        },
                   'RUT': {
                        dt.date(2009, 2, 20),   # no forward for 1
                        },
                   'NDX': {
                        #dt.date(2012, 4, 4),    # suspicious forward prices in fwd_tbl, min=0.002745 spot=2745.0
                        } 
                  }
    """ Days where data per ticker are missing or are faulty. """

    OPTIONS = {\
        'SPX' : {'SPX 291221C400000','SPX 261218C200000','SPX 301220C400000',
              'SPZ.OT', 'SPZ.CT', # invalid implied forward
              'JXB.OT', 'JXB.CT', # invalid implied forward
              'JXB.BT', 'JXB.NT', # invalid implied forward
        }
    }
    """ Specific options to exclude due to data issues. """

    OPTIONS_BY_DATE = { 
        # (ticker, date): set of option symbols to exclude
        'SPX': {
            dt.date(2024, 5, 6): {'SPX 291221C400000'},
        }
    }
    """ Options to exclude by date and ticker. """

