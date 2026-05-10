# [Massive](http://www.massive.com) Data Access

This directory contains Ivy DB data processing in the following files:

* [client.py](client.py) provides the ``MassiveClient`` class which wraps various minutae of downloading data from massive (e.g. paging). This class has self-explanatory functions such 
  as ``fetch_trades``, ``fetch_quotes``, ``fetch_bars`` and a few others; see documentation.
  This functionality is basically raw access to data. It provides very little processing, except that it filters by trade conditions for you, and encodes
  them into a ``trade_code`` per row. 

  It is recommended that you use [processing.py](processing.py) ``extract_trades`` instead of the raw ``fetch_trades`` to obtain trades.
  The former applies a few standard trade filters.

* [options/contracts.py](options/contracts.py) gives access to all option contracts listed for a given ticker on a given date.

## Data Access

To access this data:

  1. [Create an account](https://wrds-www.wharton.upenn.edu/pages/support/getting-started/) and [enable MFA](https://wrds-www.wharton.upenn.edu/pages/about/log-in-to-wrds-using-two-factor-authentication/).
  2. [Follow the WRDS instructions](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-from-your-computer/) to ensure you have 
    the ``pgpass`` pass file. *This is important.*
  3. Set the ``WRDS_USERNAME`` variable to your username.   
  4. Run the ``viz_client.ipynb`` notebook.



