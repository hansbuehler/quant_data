# Historical SANOS weekly quantitative review

**Date:** 2026-10-12
**Verdict:** **PASS** — all three blockers from the first review are resolved.
The public frozen and restarted schedules complete end to end, and adversarial
regressions now cover the former forward-normalization, carried-book, and
cost-functional failures.

## Resolved blockers

### 1. Restarted target suffixes preserve inception-forward normalization

`weekly_scenarios.py:289-304` now maps a historical forward-normalized shock
`Z(n)` into the fixed episode claim basis as

```text
S(t)/S(0) * [F_t(t+n)/S(t)] / [F_0(t+n)/S(0)] * Z(n).
```

This is the required current-forward-to-fixed-inception-forward carry ratio.
The realized prefix remains immutable. The new non-flat-forward regression at
`test_historical_sanos_weekly.py:285` independently reconstructs all four
future restarted levels and matches them to machine precision.

### 2. Pricing and execution books are now separated

`weekly_hedger.py:91-116` fits fresh-book long, short, and zero policies solely
for pricing. It separately fits the executed pure hedge from the actual carried
spot and option positions, with instrument gains centered to remove historical
drift. `weekly_adapter.py:122-127` supplies the live carried book, and
`weekly_adapter.py:150-173` returns the pure-policy row rather than a difference
of two incompatible absolute books.

The adversarial regression at `test_historical_sanos_weekly.py:392` starts with
five units of a cash-settling option and a large spread. The pricing books start
at zero, while the executed policy retains all five units and incurs zero
turnover cost. The former implementation returned zero and paid a 0.5-unit
liquidation cost.

### 3. Reported prices use the fitted cost functional

`weekly_hedger.py:287-323` now evaluates each fitted policy with the same cost
convention used by its optimizer: expected-spread generalized lasso for
mean-variance/mean-volatility, and smoothed scenario costs for entropy and
power-exponential. Buyer and seller prices are the corresponding fresh-book
long/zero/short certainty-equivalent differences at
`weekly_hedger.py:114-116`. Exact realized costs remain confined to common
out-of-sample accounting.

The regression at `test_historical_sanos_weekly.py:453` uses strongly
scenario-varying execution weights. It verifies the returned price is the
fitted expected-cost value and is deliberately different from the old
exact-net-cost risk evaluation.

## Confirmed mechanics

- Historical blocks remain strictly linked and finish no later than the fit
  date. Current live contracts are replayed at current remaining DTE, NMS, and
  option side, then held at fixed cash strike inside each historical block.
- Spot, option, discount, and fit-spot units reconcile. Settlement is discounted
  cash payoff without an exit spread; expired contracts stay dead; surviving
  terminal positions are liquidated.
- Frozen weekly reuses one five-row fit. Restarted weekly creates five dated
  fits, uses only each date's realized claim prefix, and anchors each execution
  fit to the actually carried position.
- Buyer and seller values are fitted at the requested total claim quantity and
  divided by that quantity before entering the per-fit-spot price record.

## Non-blocking qualifications

- Adjacent chronological train/validation blocks overlap by up to four
  transitions. In addition, centered execution gains use the eventual
  full-sample mean during the preliminary split. Validation is therefore a
  descriptive in-window diagnostic, not an independent holdout. It is not used
  for selection or early stopping; the final policy legitimately refits all
  pre-as-of observations.
- The convex optimizer includes the declared quadratic ridge, while the
  reported economic certainty equivalent excludes that numerical regularizer.
  Thus the reported value is the cost-functional CE of the regularized fitted
  policy, not the exact unregularized value-function supremum.
- `HistoricalSanosConfig.rolling_observations` is included in model identity,
  but the adapter does not verify that the runner's supplied training window
  has that exact length. Defaults agree at 252.
- `WeeklyScenarioBatch` describes all arrays as episode-origin PV units, while
  restarted historical batches are intentionally in current-fit-spot units.
  The calculations are consistent, but the generic docstring is imprecise.
- `HistoricalWeeklyHedgeFit.cap_hits` counts pure-policy cap hits twice at
  `weekly_hedger.py:120-149`. This can overstate the diagnostic count but does
  not affect actions or the convergence gate.

## Evidence

- Focused CPU suite: **34 passed** in 5.23 seconds.
- Ruff and `git diff --check`: passed for the amended regression file.
- Public frozen-weekly mean-variance replay, 2023-01-17: one fit, 207 contracts,
  buyer `0.39961325`, seller `0.61886516`, transaction cost `0.02862447`, and
  finite actions; elapsed 43.9 seconds.
- Public restarted-weekly mean-variance replay: five dated fits, finite prices
  and actions, transaction cost `0.05556384`; elapsed 102.9 seconds.
