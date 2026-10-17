# Corrected statistical one-day digital comparison

`statistical_1d_matrix.py` produces `STATISTICAL_DETAILS.pdf`: one landscape
page for each statistical model, objective, and risk-aversion setting. Each
completed page shows mean and median absolute daily P&L for all ten one-day
digital targets, normalized to a $1 million cash payout, against the same-date
unhedged, BS delta-and-vega, and SANOS-LV references. Missing corrected runs
appear as clearly marked pending pages, not as zero-valued results.

Pass each completed corrected run using `--case MODEL OBJECTIVE RISK RUN`, where
`MODEL` is `Covariance` or `Historical-SANOS`. The script accepts only runs
with `mean_reduction=demean` and matching data-manifest identity. The source
runs and page coverage are recorded in `summary.json`.

`STATISTICAL_SUMMARY.pdf` is written only after all 26 corrected cases are
available on at least 100 identical completed dates. For each statistical
model, it ranks the 13 cases by their equally weighted target-level mean
absolute P&L, selects the best seven, and averages their target-level means.
It independently repeats the ranking and selection using medians. This is an
**ex-post descriptive comparison**, not a trading rule or unbiased
out-of-sample model-selection result. Incomplete and older undemeaned runs
are never used to fill missing cells.
