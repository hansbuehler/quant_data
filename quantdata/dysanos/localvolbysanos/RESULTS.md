# Two-day DLV pilot results

The direct discrete Dupire/SANOS inversion was run for SPX on 2024-03-05 and
2024-03-06. The second SANOS price-fitting LP was deliberately excluded from
the reported result: with loose DLV bounds it repriced the AF calls accurately
but selected a non-identified density with most DLVs at the upper bound.

The direct inversion does not exhibit that explosion. Across the two days,
the median DLV is about 13%, the 99th percentile is below 50% on 2024-03-05 and
below 37% on 2024-03-06, and no finite DLV exceeds 2. The maximum is 0.668 on
2024-03-05 and 1.344 on 2024-03-06.

There are sparse negative calendar call increments: 8 of 1,236 grid nodes on
2024-03-05 and 18 of 1,234 on 2024-03-06. Those nodes are retained in the JSON
diagnostic as non-finite DLVs rather than silently clipped. The most negative
increments are -8.37e-5 and -3.27e-4, respectively. The discrete densities are
nonnegative and satisfy mass and martingale constraints to floating-point
accuracy, so the residual issue is in calendar differencing on the chosen
irregular grids, not strike convexity.

Artifacts:

- `results/two_day_summary.json`: complete grid and per-expiry diagnostics.
- `results/two_day_dlvs.png`: DLV curves by expiry for both sessions.

## Full reference grid ablation

Using the entire merged reference grid at every expiry removes all material
negative call increments. The remaining negative values have maximum magnitude
`1.00e-12` on 2024-03-05 and `3.27e-11` on 2024-03-06, consistent with floating-
point noise.

The unmasked full grid is not directly usable, however. Deep short-expiry wings
have density at or below numerical precision, making the DLV ratio undefined or
very large. There are 217 non-finite values and a maximum finite DLV of 6.20 on
2024-03-05; on 2024-03-06 there are 191 non-finite values and the maximum is
45.93.

The natural hybrid works cleanly: compute densities and time increments on the
full common grid, then retain DLVs only inside each expiry's original NMS range.
All 1,236 and 1,234 requested nodes are finite, no DLV exceeds 2, and the daily
aggregate results are:

| Date | Median | 95th percentile | 99th percentile | Maximum |
| --- | ---: | ---: | ---: | ---: |
| 2024-03-05 | 0.12925 | 0.25445 | 0.49795 | 0.66831 |
| 2024-03-06 | 0.12896 | 0.24267 | 0.36245 | 0.88864 |

The raw full-grid artifacts are under `results/full_reference_grid/`.

## Full-grid input-price recovery

The inferred DLVs were also run forward through the SANOS implicit transition,
starting from the time-zero point mass. Non-finite DLVs were set to zero only
for this reconstruction; these are the deep-wing nodes where the density flux
and call increment are numerically zero. All finite DLVs, including the large
deep-wing values, were retained.

The reconstructed densities recover every full-grid AF input call well within
the configured `1e-8` normalized-price tolerance:

| Date | Input prices | DLVs set to zero | Median abs. error | RMSE | Max abs. error | Max error (forward bp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-03-05 | 1,820 | 217 | 4.86e-17 | 9.97e-14 | 1.05e-12 | 1.05e-8 |
| 2024-03-06 | 1,810 | 191 | 8.08e-15 | 3.22e-12 | 3.37e-11 | 3.37e-7 |

Thus the unmasked wing DLVs are poorly identified as local-volatility numbers,
but they do not prevent recovery of the input option surface. The detailed
per-expiry recovery errors and density diagnostics are recorded under
`input_price_recovery` in the full-grid JSON artifact.

## 2025 panel

The full-reference-grid inversion was run over the locally eligible 2025 SPX
sample. The market-data layer currently stops before 2025-09-01, giving 165
eligible NYSE sessions through 2025-08-29. Of those, 163 completed. The two
missing sessions are the known-bad 2025-01-08 market and an uncached 2025-07-03
market which could not establish a non-interactive WRDS connection.

The statistical panel has exact shape `[163 dates, 10 expiries, 100 strikes]`.
Every strike is the forward-relative ratio `K/F`. The model coordinate uses the
same 100 NMS positions for every date and expiry, with each expiry's fixed
`sqrt(T)` and that date's front-week ATM volatility. Requiring one contiguous
finite support across the complete sample reduces the effective NMS interval
from the requested `[-3, 2]` to `[-1.40902, 0.93935]`. There are no null or
non-finite log-DLV-squared or log-forward-volatility observations in the
balanced panel.

The annual run reveals behavior that the two-day pilot did not: deep-wing DLVs
can explode when the discrete density is tiny. Even on the balanced interval,
the maximum DLV is 21.85 on 2025-04-07 at 120 days and effective NMS -1.386.
This makes the proposed log-DLV-squared transform essential and suggests using
robust diagnostics when fitting the first AR(1) benchmark.

Input-price recovery passes the `1e-8` tolerance on 158 of 163 sessions. Three
failures are only marginally above tolerance (`1.42e-8` to `1.94e-8`), while
2025-05-09 and 2025-07-15 have material maximum normalized price errors of
about `2.62e-3` and need separate investigation before treating the complete
panel as production-quality.

Annual artifacts under `results/2025_full_reference_grid/`:

- `raw_full_grid.parquet`: the unprojected common-grid calculation.
- `balanced_panel.parquet`: fixed 100-point effective-NMS panel for modeling.
- `daily_summary.parquet`: daily DLV and price-recovery diagnostics.
- `report.json`: scope, support, failures, and aggregate diagnostics.
- `year_state.png`: ATM DLV and ATM forward-volatility heatmaps.

## 2025 arbitrage-free market fit

The surface fit was evaluated on all 162,699 retained market quotes. A quote's
spread error is zero inside bid/ask; outside the spread it is the distance to
the nearest boundary divided by `max(half-spread, 1e-4)`. Quote weighted,
96.91% of prices are inside bid/ask. The error over all quotes has mean 0.0450
half-spreads and median zero. Conditional on being outside, the mean is 1.457
and the median is 0.675 half-spreads. Across dates, the daily percentage inside
has mean 96.91% and median 97.50%.

The prominent conditional-mean spike is 2025-02-14. The independent per-expiry
fits put all 994 retained quotes inside bid/ask, while the calendar-arbitrage-
free surface puts 51 outside, leaving 94.87% inside. Every expiry through 104
days remains inside. The misses are concentrated from 114 to 462 days,
especially the 220--273 day expiries. Their conditional mean is 11.20 and
median 4.54 half-spreads. The maximum, 55.90 half-spreads, is a 253-day quote
where the normalized call fit is 0.01813 versus a 0.01234/0.01254 market. Thus
the spike is a localized long-end distortion introduced by the cross-expiry
surface adjustment, amplified by very narrow spreads, not a broad failure of
the per-expiry fits.

Market-fit artifacts:

- `market_fit_daily.parquet`: one row per date.
- `market_fit_by_expiry.parquet`: daily expiry-level summaries.
- `market_fit_quote_errors.parquet`: retained quote-level errors.
- `market_fit_monthly.parquet`: quote-weighted monthly summaries.
- `market_fit_report.json`: aggregate and monthly results.
- `market_fit_time.png`: time-series plot.

## Surface PCA and matrix-exponential AR(1)

Spatial PCA was applied separately to the centered `10 x 100` log-DLV-squared
and log-forward-volatility surfaces. Five factors explain 91.90% of log-DLV-
squared variation and 99.68% of log-forward-volatility variation. The first
factors alone explain 77.89% and 96.42%, respectively.

The first five PCA scores were scaled to unit sample standard deviation and fit
with the irregular-time model

```text
z(t + dt) = exp(G dt) z(t) + error,
```

where `dt` is the actual calendar-year gap and zero score is the sample mean
surface. Both point estimates are mean reverting: every eigenvalue of `G` has
negative real part. The one-calendar-day transition spectral radii are 0.9646
for log DLV squared and 0.9689 for log forward volatility. Their slowest fitted
mode half-lives are 19.23 and 21.92 calendar days. The conclusion is unchanged
for every PCA rank from one through ten, although the very short half-lives of
higher factors indicate that these factors mostly capture fast noise. In-sample
R-squared across the five standardized factors is 0.479 and 0.501.

Surface-dynamics artifacts:

- `surface_history_3d.mp4`: both state surfaces over time, with effective NMS
  strike and `sqrt(T)` expiry axes.
- `pca_factors_3d.png`: the first three loading surfaces for both states.
- `pca_summary.parquet`, `pca_loadings.parquet`, and `pca_scores.parquet`:
  reusable PCA outputs.
- `mean_reversion_rank_check.parquet`: AR stability from one to ten factors.
- `surface_dynamics_report.json`: full generators, transitions, eigenvalues,
  half-lives, fit statistics, and artifact inventory.
