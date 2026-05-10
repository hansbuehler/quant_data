# WIDB - [WRDS](WRDS.md) Ivy DB from Option Metrics

This directory contains the WIDB ingestion and preprocessing pipeline used to turn raw OptionMetrics data into daily surfaces and sampled time-series tensors.

The main modules are:

* ``client.py`` for cached access to the underlying WIDB tables.
* ``day.py`` for building one cleaned and enriched market day.
* ``arbfree.py`` for fitting an arbitrage-free surface for one day.
* ``timeseries.py`` for collecting many business days into a model-ready time series.

## Client Functionality

``client.py`` is the low-level access layer around the ``wrds`` package. It provides the following functionality:

* ``WRDSCredentials.create()`` reads WIDB credentials from the environment and allows access to be disabled explicitly.
* ``WIDBClient.connect()`` lazily opens and reuses a WRDS connection, while resolving the correct OptionMetrics library and table names for the current account.
* ``ticker_to_secid_map()``, ``lookup_secid()`` and ``lookup_ticker()`` map between user-facing tickers such as ``SPX`` and OptionMetrics security ids.
* ``options(secid, date)`` fetches the raw option quote table for one security and date.
* ``spot_price(secid, date)`` fetches the underlying close price for one security and date.
* ``forwards(secid, date)`` fetches the OptionMetrics forward table for one security and date.
* ``zero_curve(date)`` fetches the USD zero-coupon curve for one date.
* ``interpolate_zero_curve(zero_curve, days_to_expiry)`` converts the downloaded zero-curve rows into discount factors for specific expiries.

All of these fetchers are cached through the project cache locations, so most downstream code works with stable local artifacts rather than hitting WRDS repeatedly.

## Time Series Pipeline

The time-series pipeline is split into three stages: daily raw collection and enrichment in ``day.py``, optional arbitrage-free fitting in ``arbfree.py``, and multi-day assembly in ``timeseries.py``.

### day.py

``day.py`` builds the per-day market snapshot in two steps.

* ``RawDayFilter`` configures the initial quote cleaning, currently centered on the minimum allowed bid.
* ``collect_raw_day(ticker, d, raw_day_filter, ...)`` downloads spot, previous spot, option quotes, forward data and the zero curve for one date, removes known bad days and bad option rows, drops same-day expiry, crossed, NA or tiny-bid quotes, computes discount factors and reference forwards, and returns a cleaned daily table with one row per strike/expiry pair.
* ``ExpiryFilter`` configures the expiry-level checks used when turning raw quotes into normalized option surfaces.
* ``enrich_day(ticker, d, raw_data=None, raw_day_filter=..., expiry_filter=..., ...)`` converts the raw day into a normalized options surface: it estimates an implied forward per expiry, combines call and put information into synthetic "pure" call prices, filters unusable strikes and expiries, and computes implied vols and Greeks for the pure bid and ask prices.

In practice, ``collect_raw_day()`` is the ingestion step and ``enrich_day()`` is the main preprocessing step used by later stages.

### arbfree.py

``arbfree.py`` adds an optional surface-fitting stage on top of ``enrich_day()``.

* ``mk_arb_free_day(ticker, d, enriched_data=None, raw_day_filter=..., expiry_filter=..., mk_arb_free=..., ...)`` runs ``quantdata.utils.arbfree.MkArbFree`` on the enriched per-expiry pure prices.
* The function keeps the retained expiries and original strike rows, adds ``mask_arb_free`` to indicate which strikes survived the fit, and writes fitted arbitrage-free prices and Greeks into the ``pure_fit1_*`` and ``pure_fitX_*`` columns.

Use this stage when you want a daily surface that is aligned to the original market rows but cleaned up so downstream models can work from a surface that is internally arbitrage-free.

### timeseries.py

``timeseries.py`` turns a sequence of daily frames into a single sampled data object.

* ``TimeseriesConfig`` defines the time-series sampling rules, including limits on moneyness, days to expiry, liquidity, AM settlement handling, minimum options per expiry and the maximum number of options per day.
* ``build_time_series(ticker, date_range, include_fields, exclude_fields, raw_day_filter, expiry_filter, mk_arb_free, timeseries_config, ...)`` loops over the business days in the requested range, loads or generates each day through ``enrich_day()`` or ``mk_arb_free_day()``, applies the time-series filter, and packs the selected fields into fixed-size numpy arrays together with an ``is_valid`` mask.
* ``table_of_time_series(...)`` produces a compact tabular summary of the sampled arrays, including shape, dtype, size and simple descriptive statistics.

Set ``mk_arb_free=None`` to build the time series directly from ``enrich_day()``, or pass a configured ``MkArbFree`` instance to use the arbitrage-free branch of the pipeline.

## Data Access

To access this data:

  1. [Create an account](https://wrds-www.wharton.upenn.edu/pages/support/getting-started/) and [enable MFA](https://wrds-www.wharton.upenn.edu/pages/about/log-in-to-wrds-using-two-factor-authentication/).
  2. [Follow the WRDS instructions](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-from-your-computer/) to ensure you have 
    the ``pgpass`` pass file. *This is important.*
  3. Set the ``WRDS_USERNAME`` variable to your username.   
  4. Run the ``viz_client.ipynb`` notebook.



