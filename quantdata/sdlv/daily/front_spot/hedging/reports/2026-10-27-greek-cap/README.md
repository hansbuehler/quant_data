# One-day digital hedging frontiers, 27 October 2026

These retained plots use the 44 April–May 2024 dates and ten one-day digital targets per date. The plotted risk parameters are \(\lambda=1,10,100\); the infinite-risk cases are excluded. Entry costs are charged once, with no exit charge in P&L. Values are USD thousands per USD 1 million digital payout.

## SANOS Greek cost-cap trial

The full SANOS Greek hedge costs a mean of 3.308 and a maximum of 9.013 across the 440 target-date observations. The **global cap** is 100%, 90%, ..., 0% of that 9.013 maximum. Whenever an existing hedge exceeds the cap, all its spot and option positions are scaled by the same fraction to meet it; otherwise it is unchanged. The second curve scales every existing hedge by the stated fraction of its own cost. Both are feasible cost-capped actions, **not** solutions of a new minimum-risk optimization problem.

| Global cap | Mean entry cost | P&L SD | Mean downside loss | Binding observations |
|---:|---:|---:|---:|---:|
| 100% | 3.308 | 215.720 | 83.683 | 0.0% |
| 50% | 3.190 | 214.391 | 82.586 | 13.4% |
| 30% | 2.536 | 227.729 | 91.522 | 67.0% |
| 10% | 0.901 | 344.137 | 154.598 | 100.0% |
| 0% | 0.000 | 435.885 | 197.143 | 100.0% |

The small improvement at the 50% global cap is an observed out-of-sample result, not evidence that less hedging always lowers risk. The 100% and 0% endpoints reproduce the original SANOS Greek and unhedged results. `RESULTS.csv` has every 10% step; `RESULTS.json` records the source and cap rules.

## Retained figures

- `SANOS_GREEK_COST_CAP_FRONTIER.pdf`: global and per-hedge caps against SD and mean downside loss.
- `COST_VERSUS_RISK.pdf`: model frontiers with the SANOS Greek global-cap curve.
- `COST_VERSUS_MEAN_DOWNSIDE_LOSS.pdf`: frontiers with downside loss \(E[\max(-\Pi,0)]\).
- `COST_VERSUS_MEAN_ABSOLUTE_ERROR.pdf`: frontiers with \(E[|\Pi|]\).
- `HEDGE_VERSUS_PAYOFF.pdf`: realized reversed-long-hedge values versus digital payoffs. These are not optimized short-side superhedges.

`plot_cost_caps.py`, `plot_frontiers.py`, and `plot_payoff_loss.py` reproduce the figures from the external experiment root selected by `QUANTDATA_ROOT`. Run them in that order using the project interpreter. The original per-date hedge results and generated-model results remain outside Git under `experiments/hedging/`; this directory retains only the reported figures, summary data, and plotting code.
