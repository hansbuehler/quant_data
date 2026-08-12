# Dysanos smooth market files

`smooth_interpolated_market()` in `market.py` fits one trading day's normalized
option market with a smooth, statically arbitrage-free SANOS surface. Its result
is cached in `PYTREE_HDF5` format under the Dysanos `training/market` cache.

The function returns an empty dictionary for a missing, excluded, or empty day.
For a successful day, the file contains the complete schema described below.

## Conventions

Let:

- `M` be the number of filtered input-market expiries;
- `N[i]` be the number of filtered strikes at input expiry `i`;
- `T` be the number of expiries retained by the final `ArbFreePricer`, with
  `T <= M`;
- `X[t]` and `Q[t]` be the numbers of extended and surface-model strikes at
  retained expiry `t`.

Expiries are year fractions. `pure_strike` is forward moneyness, `K / F`.
Pure call prices are divided by the discounted forward, so the strikes, prices,
and model densities stored below are dimensionless. Arrays are NumPy arrays
after loading and reconstruction unless stated otherwise.

## Top-level fields

| Field | Stored type | Meaning |
| --- | --- | --- |
| `schema_version` | `int` | Self-describing market contract version. The current version is `4`. |
| `ticker` | `str` | Requested ticker. |
| `d` | `datetime.date` | Trading date represented by the file. |
| `spot_ref` | `float` | Unnormalized spot price on `d`, in the ticker's price currency. |
| `spot_ret` | `float` | One-day simple spot return, `spot_ref / previous_spot - 1`. |
| `market` | mapping | Filtered input quotes grouped by expiry. See [Market fields](#market-fields). |
| `af_pricer` | mapping | Serialized `ArbFreePricer`. See [ArbFreePricer fields](#arbfreepricer-fields). |
| `fit_result` | mapping | Diagnostics obtained by repricing the filtered market. See [Fit-result fields](#fit-result-fields). |

## Market fields

The `market` mapping describes the filtered quotes passed to the final SANOS
fit. Expiries are increasing, and strikes are increasing within each expiry.

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `uid` | `str` | Eight-character identity hash of `raw_day_filter`, `expiry_filter`, `options_filter`, and the input `mk_arb_free` configuration. |
| `expiries` | `(M,)` float array | Sorted unique times to expiry in years. |
| `strikes` | list of `M` arrays `(N[i],)` | Forward-normalized strikes `K / F`. |
| `bids` | list of `M` arrays `(N[i],)` | Pure call bid prices corresponding to `strikes`. |
| `asks` | list of `M` arrays `(N[i],)` | Pure call ask prices corresponding to `strikes`; every ask is strictly above its bid. |
| `expiry_dates` | list of `M` ISO date strings | Exact contract expiry dates aligned with `expiries`. These, rather than rounded year fractions, identify weekly rolls. |
| `expiry_ids` | `(M,)` integer array | WIDB expiry identifiers, including the settlement suffix. |
| `is_am` | `(M,)` boolean array | Whether each expiry uses AM settlement. |
| `is_american` | list of `M` boolean arrays `(N[i],)` | Exercise-style flag for each call/put pair. |
| `pre_af_masks` | list of `M` boolean arrays `(N[i],)` | Upstream option-selection mask before the final smooth `af_pricer` fit. Together with side-specific quote validity, it defines the real quote universe. |
| `model_masks` | list of `M` boolean arrays `(N[i],)` | Intersection of `pre_af_masks` and the final smooth `af_pricer` retained-expiry/strike mask. It is all false for an expiry dropped by the final fit and is retained for consumers of the AF fit. |
| `call_bids`, `call_asks` | lists of `M` arrays `(N[i],)` | Real call bid and ask prices divided by `DF * F`, before any call/put synthesis. |
| `put_bids`, `put_asks` | lists of `M` arrays `(N[i],)` | Real put bid and ask prices divided by `DF * F`, before any call/put synthesis. |
| `call_log_volumes`, `put_log_volumes` | lists of `M` arrays `(N[i],)` | Side-specific `log10(1 + volume)` observations. |
| `call_log_open_interests`, `put_log_open_interests` | lists of `M` arrays `(N[i],)` | Side-specific `log10(1 + open_interest)` observations. |
| `call_active`, `put_active` | lists of `M` boolean arrays `(N[i],)` | Side-specific executable-quote masks defined before the final AF fit. A side is active when `pre_af_masks` is true, its prices are finite, its bid is nonnegative, and its ask is strictly above its bid. These masks do not inherit strikes or expiries dropped by `af_pricer`. |
| `af_call_mids` | list of `M` arrays `(N[i],)` | Prices from the final smooth `af_pricer` at every market strike. |
| `af_put_mids` | list of `M` arrays `(N[i],)` | Put reference mids obtained from the same pricer by normalized parity, `af_call_mid - 1 + strike`. |
| `af_mid_available` | list of `M` boolean arrays `(N[i],)` | Whether both AF call and parity-derived put reference prices are finite. This is reference availability only and never selects the real quote universe. |

The input `OptionsFilter` must have `normalize=True`. This requirement produces
the top-level `spot_ref` and `spot_ret` fields; the pure strike and price fields
are already forward-normalized by the WIDB enrichment pipeline.

The real call/put fields are normalized once inside `market.py`. At that point
WIDB prices and forwards have both been divided by spot, so
`market_quote / (df * fwd)` is exactly `cash_quote / (DF * F)`. Neither `df`
nor `fwd` is written into the market mapping. Calls and puts remain separate
quote observations even though their AF reference prices share one distribution.

## ArbFreePricer fields

The `af_pricer` mapping is an HDF5-safe representation returned by
`ArbFreePricer.to_dict()`. Reconstruct it through the supported API:

```python
from quantdata.sanos.arbfree import ArbFreePricer

pricer = ArbFreePricer.from_dict(market_file["af_pricer"])
```

`from_dict()` validates the complete field set, restores NumPy arrays and index
dtypes, and decodes optional values. Do not pass the mapping directly to the
class constructor.

### Configuration and retained-expiry fields

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `smoothness` | `float` | SANOS smoothing parameter, strictly greater than `0.01`. |
| `expiry_linear` | `bool` | Whether per-expiry fits use linear payoffs while the full surface uses the configured smoothing. |
| `min_half_spread` | `float` | Floor applied to the half-spread when normalizing fit errors. |
| `expiries` | `(T,)` float array | Increasing expiries retained by the final fit. The fitter may drop input expiries. |
| `variances` | `(T,)` float array | ATM total variances, `volatility**2 * expiry`, used by the smooth interpolator. |
| `input_index` | `(T,)` integer array | Maps retained expiry `t` to `market.expiries[input_index[t]]`. |

### Input-strike fits

Each list below has one entry per retained expiry. For retained index `t`, set
`i = input_index[t]`; each entry has shape `(N[i],)`.

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `masks` | list of `T` boolean arrays | Marks input strikes used by the no-arbitrage fit. `False` identifies strikes removed because of boundary slopes, negligible density, or failure to fit inside the spread. |
| `strikes` | list of `T` float arrays | Copies of the retained expiries' input strikes. |
| `bids` | list of `T` float arrays | Copies of the corresponding pure call bids. |
| `asks` | list of `T` float arrays | Copies of the corresponding pure call asks. |
| `fits1` | list of `T` float arrays | Per-expiry arbitrage-free prices. On active strikes, `bid <= fits1 <= ask` up to numerical tolerance. These fits need not be free of calendar arbitrage across expiries. |
| `fitsX` | list of `T` float arrays or `None` | Globally statically arbitrage-free surface prices at the input strikes. They are the closest surface fit and need not all lie inside bid/ask. `None` for an expiry-only pricer. |

### Per-expiry probability grids

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `xstrikes` | list of `T` arrays `(X[t],)` | Extended per-expiry strike grids. They contain active input strikes plus lower and upper boundary strikes. |
| `xdensities` | list of `T` arrays `(X[t],)` | Risk-neutral probability weights on `xstrikes`. Within numerical tolerance they are nonnegative, sum to one, satisfy `xstrikes[t] @ xdensities[t] == 1`, and reprice the per-expiry fit. |

### Full-surface probability grids

These fields are populated for the surface mode used by
`smooth_interpolated_market()` and are `None` for an expiry-only pricer.

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `mstrikes` | list of `T` arrays `(Q[t],)` or `None` | Model strike grids for the full surface. Each is a superset of its `xstrikes` grid and can include strikes propagated from later expiries. |
| `mdensities` | list of `T` arrays `(Q[t],)` or `None` | Martingale probability weights on `mstrikes`. Each density is nonnegative, sums to one, has normalized mean one, and the family is ordered consistently with convex order. |
| `xix_to_mix` | list of `T` integer arrays or `None` entries | For a non-`None` entry, maps `xstrikes[t]` into `mstrikes[t]`: `mstrikes[t][xix_to_mix[t]] == xstrikes[t]`. An entry is `None` when the two grids coincide. The whole field is `None` for an expiry-only pricer. |

### Timing fields

| Field | Type | Meaning |
| --- | --- | --- |
| `run_time_expiry` | `float` or `None` | Wall-clock seconds spent constructing the per-expiry fits. |
| `run_time_surface` | `float` or `None` | Wall-clock seconds spent constructing the full surface. `None` when no surface fit was run. |

### Representation of `None` in HDF5

The `PYTREE_HDF5` writer does not natively support `None`. In the raw
`af_pricer` mapping, `ArbFreePricer.to_dict()` therefore encodes every `None` as
the reserved string `"__ARB_FREE_PRICER_NONE__"`, including `None` entries in
`xix_to_mix`. `ArbFreePricer.from_dict()` decodes this representation
transparently. Consumers should not interpret or replace the sentinel directly.

## Fit-result fields

`fit_result` reprices every quote in `market`, not only the expiries and strikes
retained internally by the pricer. The pricer interpolates where necessary.
Because `smooth_interpolated_market()` calls `compute_market_fit(and_vols=False)`,
there is no `fit_vols` field in these files.

For option `j` at expiry `i`, define

```text
half_spread = max(0.5 * (ask - bid), af_pricer.min_half_spread)
fit_error   = (max(0, fit - ask) + max(0, bid - fit)) / half_spread
```

The error is zero inside the bid/ask spread and is measured in half-spread
units outside it.

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `fits` | list of `M` arrays `(N[i],)` | Surface prices at every filtered market strike. |
| `fit_errs` | list of `M` arrays `(N[i],)` | Half-spread-normalized error for each price in `fits`. |
| `pcnt_fit` | `float` | Fraction of all quotes with positive `fit_errs`, despite the historical name: this is the fraction outside bid/ask, between zero and one. |
| `median_err` | `float` | Median positive fit error, or zero if every price is inside bid/ask. |
| `mean_err` | `float` | Mean positive fit error, or zero if every price is inside bid/ask. |
| `median_total_err` | `float` | Median fit error over all quotes, including zeros. |
| `mean_total_err` | `float` | Mean fit error over all quotes, including zeros. |

## Cache identity

The cache path is determined by the filter configurations, smoothing parameter,
ticker, and date. Operational arguments such as `verbose` and
`message_context` do not alter the numerical result or cache identity.
