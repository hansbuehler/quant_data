# One-day digital hedging frontiers, 27 October 2026

These retained plots use the 44 April–May 2024 dates and ten one-day digital targets per date. The risk parameters are \(\lambda=1,10,100,1000\) where a complete 44-date result is available. Entry costs are charged once, with no exit charge in P&L. Values are USD thousands per USD 1 million digital payout. The base AD-Seq-Vol diffusion has complete mean–variance, entropy, and power-exponential results; fine-tuned AD-Seq-Vol has no complete 44-date cell yet and is not drawn.

## SANOS Greek

The full SANOS Greek hedge costs a mean of 3.308 and a maximum of 9.013 across the 440 target-date observations. The **global cap** is 100%, 90%, ..., 0% of that 9.013 maximum. The second curve caps each hedge at the stated fraction of its own full cost. At every constrained point, option holdings minimize the eight-factor linearized variance subject to the entry-cost budget, while spot neutralizes delta at zero cost. No existing hedge positions are proportionally scaled.

| Global cap | Mean entry cost | P&L SD | Mean downside loss |
|---:|---:|---:|---:|
| 100% | 3.308 | 215.720 | 83.683 |
| 50% | 3.190 | 215.973 | 83.596 |
| 30% | 2.536 | 215.471 | 81.596 |
| 10% | 0.901 | 255.673 | 94.167 |
| 0% | 0.000 | 299.258 | 131.085 |

The 100% point is the original SANOS Greek hedge; the 0% point is a spot-only hedge, not unhedged. The optimizer minimizes predicted factor risk, so out-of-sample P&L SD need not be monotone. `RESULTS.csv` has every 10% step; `RESULTS.json` records the source and cap rules. All 5,341 constrained solves were certified optimal; no budget or position-cap failures occurred.

## Retained figures

- `SANOS_GREEK_COST_CAP_FRONTIER.pdf`: global and per-hedge SANOS Greek cost caps against SD and mean downside loss.
- `COST_VERSUS_RISK.pdf`: model frontiers with the cost-capped SANOS Greek curve and a coverage table distinguishing complete, partial, failed, and unrun cells.
- `COST_VERSUS_MEAN_DOWNSIDE_LOSS.pdf`: frontiers with downside loss \(E[\max(-\Pi,0)]\).
- `COST_VERSUS_SD_AND_MEAN_LOSS.pdf`: both comparisons in one PDF, with cost versus out-of-sample P&L standard deviation first, a cell-coverage table, and cost versus mean downside loss last. A partial cell shows its saved-date count out of 44 and is not plotted until complete.
- `COST_VERSUS_MEAN_ABSOLUTE_ERROR.pdf`: frontiers with \(E[|\Pi|]\).
- `HEDGE_VERSUS_PAYOFF.pdf`: realized reversed-long-hedge values versus digital payoffs. These are not optimized short-side superhedges.

SANOS parametric is fitted separately for mean–variance, entropy, power-exponential, and CVaR at \(\lambda=1,10,100,1000\) on all 44 dates; no cell failed certification or hit a position cap. Under the zero-mean Gaussian factor law, entropy and mean–variance have the same optimizer at equal risk aversion, although their cells are fitted and recorded separately. `plot_frontiers.py` and `plot_payoff_loss.py` reproduce the comparison figures from the external experiment root selected by `QUANTDATA_ROOT`; `quantdata/hedging/plot_reoptimized_greek_cost_cap.py` reproduces the dedicated cap plot. The original per-date hedge results and generated-model results remain outside Git under `experiments/hedging/`.

SANOS historical PCA uses the same 252 rolling SANOS return observations, with spot separated from eight residual option factors. Objective-specific cells enter the figures only after all 44 dates and ten targets pass the numerical gate; partial or failed cells remain visible in the coverage page.

The comparison PDF includes every certified cell in the matched April–May objective grids. The coverage page lists incomplete and failed cells without substituting partial-period statistics. Its coverage is a snapshot of saved records at report generation and should be refreshed after additional cells complete. Superseded pilot figures, old-dataset rolling results, and their experiment-only launcher/report code have been removed; current scenario books, checkpoints, and independent production cohorts are retained.

## Raw distributions and drift-aware frontiers

`SANOS_RAW_DISTRIBUTIONS_AND_FRONTIERS.pdf` has 70 distribution pages and four frontier pages for the same 44 dates and ten one-day digitals. `plot_sanos_raw_distributions.py` regenerates it from the retained external scenario books and per-date hedge records; `SANOS_RAW_DISTRIBUTIONS_SUMMARY.json` records its aggregated values. Give the script a new `--output` directory beneath `QUANTDATA_ROOT/experiments/hedging` for each run.

The distributions compare **raw realized costed P&L** against the law available before trading. Unhedged, BS delta + vega, and SANOS Greek use raw returns from the preceding 252-day SANOS replay. SANOS historical uses its fitted, training-demeaned 252-scenario law; SANOS parametric uses its fitted centered factor-Gaussian law. No post-hoc mean correction is applied to either plotted distribution. The fixed references are shown once per product rather than repeated under objectives they do not optimize.

The four frontier pages put entry cost or entry cost plus corrected pre-cost mean loss on the vertical axis, against realized P&L SD or centered realized 5% expected shortfall on the horizontal axis. The mean correction subtracts only the target and hedge-instrument means estimated from each preceding 252-day training book, frozen before the next-day return. The risk axes use the raw realized P&L. With 44 dates, empirical 5% expected shortfall averages each target's worst two centered returns plus 20% of its third-worst, divided by 2.2. These 44-date tail estimates are descriptive rather than precise out-of-sample rankings.
