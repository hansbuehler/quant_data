# WIDB - [WRDS](WRDS.md) Ivy DB from Option Metrics

This directory contains Ivy DB data processing in the following files:

* [client.py](client.py) provides raw, cached access to IvyDB with functions like `options`, `lookup_ticker` etc. Please see the file documentation.
* [day.py](day.py) provides ``collect_raw_day`` to collect all relevant data for a day of options for a given ticker; ``enrich_day`` in addition:
    * Computes synthetic minimal spread call options.
    * Computes an implied forward from inverse-spread weighted bid/ask call prices.
    * Computes implied volatilities and (pure) greeks.

* [sampler.py](sampler.py) provides ``sample_days`` which creates a time series of sample data.

## Data Access

To access this data:

  1. [Create an account](https://wrds-www.wharton.upenn.edu/pages/support/getting-started/) and [enable MFA](https://wrds-www.wharton.upenn.edu/pages/about/log-in-to-wrds-using-two-factor-authentication/).
  2. [Follow the WRDS instructions](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-from-your-computer/) to ensure you have 
    the ``pgpass`` pass file. *This is important.*
  3. Set the ``WRDS_USERNAME`` variable to your username.   
  4. Run the ``viz_client.ipynb`` notebook.



