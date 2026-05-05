# WIDB - [WRDS](WRDS.md) Ivy DB from Option Metrics

This directory contains Ivy DB data processing in the following files:

* ``client.py`` provides raw, cached access to IvyDB with functions like :meth:`WRDSClient.options`.
* ``day.py`` is used to download data, remove bad days or stale data, and enrich the data set with business days to expiry, implied vols and (pure) greeeks. 
* ``sampler.py`` creates a time series for a period of days for a given ticker, filtering by expiry, moneyness and other variables.

## Data Access

To access this data:

  1. [Create an account](https://wrds-www.wharton.upenn.edu/pages/support/getting-started/) and [enable MFA](https://wrds-www.wharton.upenn.edu/pages/about/log-in-to-wrds-using-two-factor-authentication/).
  2. [Follow the WRDS instructions](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-from-your-computer/) to ensure you have 
    the ``pgpass`` pass file. *This is important.*
  3. Set the ``WRDS_USERNAME`` variable to your username.   
  4. Run the ``viz_client.ipynb`` notebook.



