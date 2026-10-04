# Fixed-contract static versus deep hedge summary

Date: 18 March 2025  
Risk aversion: lambda = 50  
Market: the same initially listed options are available to the static and deep hedges; normalized spot is also included. No rolling or floating-strike options are used.

## 5D ATM call spread

Prices are shown as buyer bid / seller ask.

| Objective | Static hedge | Static width | Daily deep hedge | Deep width | Width reduction |
|---|---:|---:|---:|---:|---:|
| Mean-variance under power-exp-derived Q | -0.1539 / 1.2408 | 1.3948 | -0.0457 / 1.1316 | 1.1773 | 15.6% |
| Entropy OCE under entropy-derived Q | 0.4473 / 0.6369 | 0.1896 | 0.4478 / 0.6369 | 0.1890 | 0.3% |
| Power-exp OCE under power-exp-derived Q | 0.4460 / 0.6355 | 0.1895 | 0.4553 / 0.6341 | 0.1788 | 5.6% |

The entropy density has only about 78 effective validation paths, so that row is statistically fragile. The power-exp density retains about 3,694 effective paths. Mean-variance uses the fixed power-exp-derived measure, not a mean-variance change of measure.

## Day-3 KO equity payoff

The payoff is

`Z = S5 * 1{S3 <= 1}`.

It pays the normalized day-5 equity level if it survives the day-3 ATM knockout and zero otherwise. Power-exp OCE is used.

| Hedge | Buyer bid | Seller ask | Width |
|---|---:|---:|---:|
| Static | 0.1061 | 0.8810 | 0.7748 |
| Daily deep hedge | 0.2668 | 0.6658 | 0.3989 |

Deep hedging reduces the KO interval width by approximately 48.5%. The buyer price rises and seller price falls, so both sides improve in the expected direction. The power-exp density has about 2,755 effective validation paths.

At the knockout observation, the deep policy sees the contemporaneous spot and knockout state before selecting its subsequent hedge.
