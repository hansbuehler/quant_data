# Daily target-notional calibration

## Scope

- Run date: 2026-10-12.
- Dataset: `spx-2022-2025-hedging-public-candidate-v1` only.
- Calibration sample: 242 valid one-session transitions ending no later than
  2022-12-30.
- Targets: all 20 agreed one-day and three-day digital claims.
- Diagnostic hedge: independent covariance, mean-volatility with `kappa=1`.

For target `i` with maturity `H_i`, the fixed benchmark quantity is

```text
q_i = sigma_atm(2022-12-30, H_i) * sqrt(H_i / 252)
      / SD_2022(target_i return).
```

Thus every scaled target has the same unhedged standard deviation as one
maturity-matched ATM spot-risk unit. The quantities are fixed before the 2023
out-of-sample period.

## Results

| Diagnostic | Result |
| --- | ---: |
| Valid initialization transitions | 242 |
| Targets | 20 |
| Minimum target quantity | 0.0260113 |
| Median target quantity | 0.0612384 |
| Maximum target quantity | 0.1455880 |
| Median transaction cost / reference risk | 1.6873% |
| 90th percentile transaction cost / reference risk | 3.7759% |
| Maximum transaction cost / reference risk | 7.5449% |
| Converged target hedges | 20 / 20 |
| Position-cap hits | 0 |

The transaction-cost fraction is invariant to a common target rescaling for
the homogeneous mean-volatility objective with proportional costs, unless a
zero-action region or position cap changes the solution. Consequently the
pilot identifies a natural risk unit, not an empirically optimal absolute
trade size. The benchmark therefore uses one ATM spot-risk unit and a loose
numerical action cap of 1,000; any cap hit fails the numerical gate.
