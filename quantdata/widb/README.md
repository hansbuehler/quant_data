# WIDB - [WRDS](WRDS.md) Ivy DB from Option Metrics

This directory contains the WIDB ingestion and preprocessing pipeline used to turn raw OptionMetrics data into daily surfaces and sampled time-series tensors.

The main modules are:

* ``client.py`` for cached access to the underlying WIDB tables.
* ``day.py`` for building one cleaned and enriched market day.
* ``arbfree.py`` for fitting an arbitrage-free surface for one day.
* ``timeseries.py`` for collecting many business days into a model-ready time series.


## Data Access

To access this data:

  1. [Create an account](https://wrds-www.wharton.upenn.edu/pages/support/getting-started/) and [enable MFA](https://wrds-www.wharton.upenn.edu/pages/about/log-in-to-wrds-using-two-factor-authentication/).
  2. [Follow the WRDS instructions](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-from-your-computer/) to ensure you have 
    the ``pgpass`` pass file. *This is important.*
  3. Set the ``WRDS_USERNAME`` variable to your username.   
  4. Run the ``viz_client.ipynb`` notebook.

This is the process to generate the ``pgpass`` file:

![WIDB Login](images/widb-login.png)

## Generate data

There a number of notebooks which will show how to call the functionality contained in WIDB.

To mass download raw data and generate "daily" snapshots, call:

```bash
cd quant_data/quantdata/widb
python -m day SPX --workers 4
```

To embedd this data with no-arbitrage fits (see below), call:

```bash
cd quant_data/quantdata/widb
python -m arbfree SPX --workers 4 --proceed
```

The ``--proceed`` keyword lets you continue processing even if you encounter numerical errors

![WIDB Login](images/mkarbfree-process.png)


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

Run the ``viz_day.ipynb`` notebook to test this functionality.

* ``collect_raw_day(ticker, d, raw_day_filter, ...)`` downloads spot, previous spot, option quotes, forward data and the zero curve for one date, removes known bad days and bad option rows, drops same-day expiry, crossed, NA or tiny-bid quotes, computes discount factors and reference forwards, and returns a cleaned daily table with one row per strike/expiry pair.

  Use ``RawDayFilter`` to specify a minimum bid price for a strike to be 
included.

* ``enrich_day(ticker, d, raw_data=None, raw_day_filter=..., expiry_filter=..., ...)`` converts the raw day into a normalized options surface: it estimates an implied forward per expiry, combines call and put information into synthetic "pure" call prices, filters unusable strikes and expiries, and computes implied vols and Greeks for the pure bid and ask prices. It drops options whose implied vol is outside the specified range.

  Use ``ExpiryFilter`` to configure this process.

In practice, ``collect_raw_day()`` is the ingestion step and ``enrich_day()`` is the main preprocessing step used by later stages.
The former will transparently call the  latter if requested.

The file ``day.py`` has a commandline interface which allows generating days on disk with default settings.
```
  python -m day SPX --period 2025-06-01 2025-09-01
```

### arbfree.py

``arbfree.py`` adds an optional surface-fitting stage on top of ``enrich_day()``.

Run the ``viz_arb_free.ipynb`` notebook to test this functionality.

* ``mk_arb_free_day(ticker, d, enriched_data=None, raw_day_filter=..., expiry_filter=..., mk_arb_free=..., ...)`` runs ``quantdata.utils.arbfree.MkArbFree`` on the enriched per-expiry pure prices.
* The function keeps the retained expiries and original strike rows, adds ``mask_arb_free`` to indicate which strikes survived the fit, and writes fitted arbitrage-free prices and Greeks into the ``pure_fit1_*`` and ``pure_fitX_*`` columns. The latter is the result of a "surface"
fit, if requested.

Use this stage when you want a daily surface that is aligned to the original market rows but cleaned up so downstream models can work from a surface that is internally arbitrage-free.

There are two main effects:

1) The function will attempt to construct per expiry a fit within bid/ask for the provided strikes. If this fails, arbitrage is actionable and the expiry
   is removed. The function may also remove some strikes if they are causing arbitrage (typically at the boundaries) or if the likelihood of reaching
   them is zero (e.g. the arbitrage-free options at that point form a straight line).
   
   Finding an arbitrage-free fit is a linear program which is usally very fast, but it can become slow if the problem is closed to violating
   arbitrage. The ``MkArbFree`` class has a timeout parameter for this reason. If the process times out for a given expiry, that expiry is removed.

   The resulting ``fit1`` returned by the function represents a fit within bid/ask spread per expiry. That does not imply that the surface is globally
   free of arbitrage.

2) If enabled, the second step is to fit a closest globally arbitrage-free surface. This can involve a lot strikes and is often very slow.
   Once finished the process returns with ``fitX`` the global fit. Within machine precision existence of a ``fitX`` within bid/ask is equivalent
   to absence of arbitrage. However, practically, ``fitX`` may not be within bid/ask but only a close fit.

The file ``arbfree.py`` has a commandline interface which allows generating days on disk with default settings.
```
  python -m arbfree SPX --period 2025-06-01 2025-09-01
```

### timeseries.py

``timeseries.py`` turns a sequence of daily frames into one dictionary. Such a dictionary can be efficiently streamed to and from disk using [HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_m__u_g.html) which has superiour throughput for blocks of numpy data.

* ``build_time_series(ticker, date_range, include_fields, exclude_fields, raw_day_filter, expiry_filter, mk_arb_free, timeseries_config, ...)`` loops over the business days in the requested range, loads or generates each day through ``enrich_day()`` or ``mk_arb_free_day()``, applies the time-series filter, and packs the selected fields into fixed-size numpy arrays together with an ``is_valid`` mask.

  The  ``TimeseriesConfig`` defines the time-series sampling rules, including limits on moneyness, days to expiry, liquidity, AM settlement handling, minimum options per expiry and the maximum number of options per day.

* ``table_of_time_series(...)`` produces a compact tabular summary of the sampled arrays, including shape, dtype, size and simple descriptive statistics.
Here is an example output for 62 days "SPX" using arb-free mode ``expiry`` (that is where there are ``fit1`` but no ``fitX`` prices and greeks):

  | Field | Shape | Dtype | Size | Mean | Std | Min | Max |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | is_valid | (62, 1000) | bool | 60.55KB | 1 | 0 | 1 | 1 |
  | dates | (62,) | date |  |  |  |  |  |
  | ask_vol | (62, 1000) | float32 | 242.19KB | 0.1393 | 0.02385 | 0.07024 | 0.2944 |
  | bdte | (62, 1000) | float32 | 242.19KB | 57.16 | 79.79 | 1 | 639.2 |
  | bid_vol | (62, 1000) | float32 | 242.19KB | 0.1385 | 0.02378 | 0.06905 | 0.2934 |
  | call_ask | (62, 1000) | float32 | 242.19KB | 160.3 | 157.9 | 2.054 | 2644 |
  | call_bid | (62, 1000) | float32 | 242.19KB | 159.6 | 157.5 | 1.953 | 2641 |
  | call_opt_id | (62, 1000) | int32 | 242.19KB | 1.686e+08 | 3.505e+06 | 1.434e+08 | 1.715e+08 |
  | call_symbol | (62, 1000) | object | 484.38KB |  |  |  |  |
  | date | (62, 1) | datetime64[ms] | 496 bytes | 1.753e+12 | 2.292e+09 | 1.749e+12 | 1.756e+12 |
  | df | (62, 1000) | float32 | 242.19KB | 0.9902 | 0.01286 | 0.9001 | 0.9999 |
  | expiry | (62, 1000) | datetime64[ms] | 484.38KB | 1.76e+12 | 1.024e+10 | 1.749e+12 | 1.829e+12 |
  | expiry_id | (62, 1000) | int32 | 242.19KB | 2.025e+08 | 3.611e+04 | 2.025e+08 | 2.027e+08 |
  | fwd | (62, 1000) | float32 | 242.19KB | 6298 | 185.1 | 5935 | 6943 |
  | fwd_ref | (62, 1000) | float32 | 242.19KB | 6300 | 185.1 | 5937 | 6934 |
  | is_am | (62, 1000) | bool | 60.55KB | 0.298 | 0.4574 | 0 | 1 |
  | is_american | (62, 1000) | bool | 60.55KB | 0 | 0 | 0 | 0 |
  | log_open_interest | (62, 1000) | float32 | 242.19KB | 2.586 | 0.8028 | 1 | 5.716 |
  | log_volume | (62, 1000) | float32 | 242.19KB | 1.819 | 0.623 | 1 | 4.88 |
  | mask_arb_free | (62, 1000) | bool | 60.55KB | 0.8556 | 0.3515 | 0 | 1 |
  | mkt_call_ask | (62, 1000) | float32 | 242.19KB | 160.6 | 158.3 | 2.05 | 2664 |
  | mkt_call_bid | (62, 1000) | float32 | 242.19KB | 159.3 | 157.2 | 1.95 | 2624 |
  | mkt_call_log_open_interest | (62, 1000) | float32 | 242.19KB | 2.322 | 0.9009 | 0 | 5.418 |
  | mkt_call_log_volume | (62, 1000) | float32 | 242.19KB | 1.325 | 0.8318 | 0 | 4.88 |
  | mkt_put_ask | (62, 1000) | float32 | 242.19KB | 149.4 | 134.4 | 4 | 4164 |
  | mkt_put_bid | (62, 1000) | float32 | 242.19KB | 148 | 133.4 | 3.8 | 4123 |
  | mkt_put_log_open_interest | (62, 1000) | float32 | 242.19KB | 1.89 | 0.9847 | 0 | 5.415 |
  | mkt_put_log_volume | (62, 1000) | float32 | 242.19KB | 1.19 | 0.8665 | 0 | 4.294 |
  | nms | (62, 1000) | float32 | 242.19KB | -0.04815 | 0.5418 | -3.944 | 3.536 |
  | pure_ask_delta | (62, 1000) | float32 | 242.19KB | 0.5116 | 0.1414 | 0.0161 | 0.9462 |
  | pure_ask_dk | (62, 1000) | float32 | 242.19KB | -0.4895 | 0.139 | -0.9379 | -0.009242 |
  | pure_ask_gamma | (62, 1000) | float32 | 242.19KB | 10.16 | 6.969 | 0.1809 | 66.8 |
  | pure_ask_price | (62, 1000) | float32 | 242.19KB | 0.0258 | 0.02563 | 0.0003439 | 0.4281 |
  | pure_ask_theta | (62, 1000) | float32 | 242.19KB | 0.0877 | 0.04983 | 0.001792 | 0.6658 |
  | pure_ask_vega | (62, 1000) | float32 | 242.19KB | 0.1498 | 0.09717 | 0.01153 | 0.6312 |
  | pure_bid_delta | (62, 1000) | float32 | 242.19KB | 0.5115 | 0.1422 | 0.01259 | 0.9474 |
  | pure_bid_dk | (62, 1000) | float32 | 242.19KB | -0.4895 | 0.1397 | -0.9389 | -0.007261 |
  | pure_bid_gamma | (62, 1000) | float32 | 242.19KB | 10.22 | 7.023 | 0.1585 | 67.68 |
  | pure_bid_price | (62, 1000) | float32 | 242.19KB | 0.02569 | 0.02556 | 0.000327 | 0.4275 |
  | pure_bid_theta | (62, 1000) | float32 | 242.19KB | 0.08707 | 0.04934 | 0.001392 | 0.657 |
  | pure_bid_vega | (62, 1000) | float32 | 242.19KB | 0.1497 | 0.09714 | 0.01128 | 0.6312 |
  | pure_fit1_delta | (62, 1000) | float32 | 242.19KB | 0.5116 | 0.1418 | 0.01438 | 0.9468 |
  | pure_fit1_dk | (62, 1000) | float32 | 242.19KB | -0.4895 | 0.1393 | -0.9384 | -0.008274 |
  | pure_fit1_gamma | (62, 1000) | float32 | 242.19KB | 10.19 | 6.996 | 0.174 | 67.25 |
  | pure_fit1_price | (62, 1000) | float32 | 242.19KB | 0.02574 | 0.02559 | 0.0003361 | 0.4278 |
  | pure_fit1_theta | (62, 1000) | float32 | 242.19KB | 0.08738 | 0.04958 | 0.001595 | 0.6612 |
  | pure_fit1_vega | (62, 1000) | float32 | 242.19KB | 0.1497 | 0.09715 | 0.01141 | 0.6312 |
  | pure_strike | (62, 1000) | float32 | 242.19KB | 0.9982 | 0.02756 | 0.5878 | 1.71 |
  | put_opt_id | (62, 1000) | int32 | 242.19KB | 1.686e+08 | 3.505e+06 | 1.434e+08 | 1.715e+08 |
  | put_symbol | (62, 1000) | object | 484.38KB |  |  |  |  |
  | spot | (62, 1) | float32 | 248 bytes | 6249 | 171.3 | 5936 | 6502 |
  | spot_prev | (62, 1) | float32 | 248 bytes | 6241 | 174.3 | 5912 | 6502 |
  | strike | (62, 1000) | float32 | 242.19KB | 6287 | 251.9 | 4000 | 1.1e+04 |
  | tte_y | (62, 1000) | float32 | 242.19KB | 0.2268 | 0.3166 | 0.003968 | 2.537 |
  | TOTAL |  |  | 11.83MB |  |  |  |  |


