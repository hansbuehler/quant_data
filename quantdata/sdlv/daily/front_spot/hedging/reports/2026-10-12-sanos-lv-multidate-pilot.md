# SANOS-LV multi-date CPU pilot

## Scope

- Run date: 2026-10-12.
- Dataset: `spx-2022-2025-hedging-public-candidate-v1`.
- Origins: eight evenly spaced gap-free dates from 2023-01-17 through
  2025-08-28.
- Work per origin: one rolling 252-transition factor fit followed by all 20
  agreed one-day and three-day digital targets.
- Total: 160 target hedges on CPU.

The date selector rejected 2024-07-17 and 2024-11-27 because each was followed
by a retained-data gap, then advanced to the next valid origin. The benchmark
therefore does not interpret a multi-session gap as a one-day hedge.

## Results

| Diagnostic | Result |
| --- | ---: |
| Converged targets | 160 / 160 |
| Position-cap hits | 0 |
| Maximum nonzero hedge options | 8 |
| Maximum absolute action | 159.694 |
| Mean entry gross exposure | 161.985 |
| Maximum entry gross exposure | 500.493 |
| Mean realized transaction cost | 0.014003 |
| Maximum realized transaction cost | 0.059253 |
| Maximum factor residual norm | 2.895e-17 |
| Maximum absolute delta residual | 0 |
| Mean smoothed-residual coverage | 73.592% |
| Mean fixed-strike clipping share | 2.528% |
| Maximum fixed-strike clipping share | 2.660% |
| Mean flat-forward-tail share | 6.758% |
| Maximum flat-forward-tail share | 8.490% |
| Wall time | 26.28 seconds |

Entry gross exposure is the sum of absolute spot and option actions. Realized
transaction cost includes entry and liquidation under the common accounting
rule. The one-day hedge is opened from zero, so entry gross exposure is also
the initial turnover measure. These diagnostics establish CPU feasibility and
numerical consistency; they are not hedging-performance estimates.

## Dates

The origins were 2023-01-17, 2023-05-26, 2023-10-12, 2024-03-05,
2024-07-19, 2024-12-02, 2025-04-15, and 2025-08-28. Every date produced 20
converged targets with no cap hit and exact delta neutralization to reported
precision.
