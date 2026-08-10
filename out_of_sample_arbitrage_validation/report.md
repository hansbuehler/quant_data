# Fast frozen-position validation

The exact saved frictionless positions from `empirical_arbitrage_results/delta0_prop0/positions.npy` were used without refitting. Their independently reconstructed 40k frictionless payoff matches the saved payoff path by path to a maximum absolute difference of `6.11e-16`.

Costs are exactly:

- delta=1: `sum(abs(a[t,l]) * mc_error_train[t,l])`, where `mc_error_train = std(dX_train, ddof=1)/sqrt(40000)`;
- 1 bp: `sum(abs(a[t,l]) * 0.0001 * abs(X[path,t,l]))`.

Both datasets have 9 time steps and 141 instruments and contain no non-finite `dX` or `X` values.

| paths | cost applied to frozen frictionless position | minimum | mean | negative paths | negative fraction | conclusion |
|---:|---|---:|---:|---:|---:|---|
| 40,000 | none | -1.00001e-10 | 0.0754443 | 46 below -1e-10* | 0.1150%* | empirical pathwise certificate within solver tolerance |
| 40,000 | 1 training MC error | -0.0548756 | 0.0205686 | 17,904 | 44.7600% | no arbitrage |
| 40,000 | 1 bp of X | -0.0162402 | 0.0648113 | 527 | 1.3175% | this frozen portfolio is not pathwise non-negative |
| 400,000 | none | -0.0375553 | 0.0755699 | 874 | 0.2185% | positive mean, not arbitrage |
| 400,000 | 1 training MC error | -0.0924309 | 0.0206943 | 177,774 | 44.4435% | no arbitrage |
| 400,000 | 1 bp of X | -0.0490960 | 0.0649380 | 5,243 | 1.31075% | positive mean, not arbitrage |

\*The apparent 40k negatives are only solver-boundary values around `-1e-10`; the minimum is `-1.000005e-10`.

The 400k delta=1 mean has standard error `9.2914e-5` and 95% confidence interval `[0.0205122, 0.0208764]`. It therefore remains a statistically positive expected-payoff signal, but emphatically not an arbitrage: almost 44.5% of independent paths lose money.

The prior claim that an independently optimized 1 bp portfolio separates the 40k sample is not retested here because its exact position array is absent. Applying 1 bp to the saved frictionless-selected portfolio is a different cross-test, shown above. No portfolio was optimized on the 400k validation data.

## Conclusion

The “no real arbitrage under one MC-error cost” result remains in place on 400k independent paths. More broadly, the frozen frictionless position also fails pathwise non-negativity without costs. The evidence supports an expected-return signal with risk / finite-sample separation, not structural pathwise arbitrage.
