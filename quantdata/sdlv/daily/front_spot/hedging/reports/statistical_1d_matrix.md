# Corrected statistical one-day digital comparison

`statistical_1d_matrix.py` produces `STATISTICAL_DETAILS.pdf`: one landscape
page for each statistical model and objective, with all risk-aversion settings
on the same three graphs. Each page shows mean, median, and 10--90% absolute
daily P&L for all ten one-day digital targets, normalized to a $1 million cash
payout, alongside same-date unhedged, BS delta-and-vega, and SANOS-LV
references. Missing corrected settings are omitted and marked pending in the
coverage summary, never plotted as zero-valued results.

Pass each completed corrected run using `--case MODEL OBJECTIVE RISK RUN`, where
`MODEL` is `Covariance` or `Historical-SANOS`. The script accepts only runs
with `mean_reduction=demean` and matching data-manifest identity. The source
runs and page coverage are recorded in `summary.json`.

`STATISTICAL_SUMMARY.pdf` is written only after all 26 corrected cases are
available on at least 100 identical completed dates by default. The April 2024
diagnostic uses `--min-common-dates 22` for its 22-session month. For each statistical
model, it ranks the 13 cases by their equally weighted target-level mean
absolute P&L, selects the best seven, and averages their target-level means.
It independently repeats the ranking and selection using medians. The 10--90%
summary range pools date-level losses from those seven median-selected cases;
it is descriptive, not an independent-sample confidence interval. This is an
**ex-post descriptive comparison**, not a trading rule or unbiased
out-of-sample model-selection result. Incomplete and older undemeaned runs
are never used to fill missing cells.
