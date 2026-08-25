# FLEX batch-descent capacity benchmark

Status: **complete**.

Geometry: canonical NMS `(-3,2)`, 200 inner strikes, 14 weekly model expiries from 4.8 to 69.8 business days, 2,828 native parameters per market date, and one forward-mu coordinate per expiry interval.

Every stage uses every eligible real call and put book, with no validation holdout. The native state is learned directly by batched gradient descent; no LP solution is used as an input or benchmark.

Compression optimizer configuration:

- supervised decoder learning rate: `0.002`;
- quote-objective learning rate: `0.0002`;
- compiled SANOS optimizer steps: `True`;
- final candidate and reported metrics: eager SANOS pricing;
- convergence and no-regression gates: mandatory, with no fallback model selection.

## 1. Direct 2,828-state batch descent

| Bucket | Options | Zero excess | Conditional positive mean | Conditional positive median | Surface mean | Surface median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3492782 | 71.4044% | 4.36899 | 1.32237 | 1.24934 | 0 |
| 0DTE | 2430 | 26.5844% | 13.7216 | 11.7925 | 10.0738 | 6.38454 |
| 1DTE | 77236 | 20.3001% | 17.939 | 10.6678 | 14.2973 | 5.87089 |
| 2DTE | 102320 | 29.7977% | 12.9598 | 8.50368 | 9.09807 | 3.29836 |

Whole-surface error against `market.py` `fit`:

| Measure | Half-spreads | Normalized price |
| --- | ---: | ---: |
| Mean absolute error | 1.7964 | 9.12326e-05 |
| Median absolute error | 0.439314 | 3.66646e-05 |
| Root mean square error | — | 0.000222382 |
| Signed bias | — | -1.93926e-05 |

Mean training objective: `1.3391582`.

Market-training convergence: **True**; trailing-window relative improvement `0.00015979178969470023`; final gap to best `8.552105090143964e-07`.

## 2a. Initial Global 140

| Bucket | Options | Zero excess | Conditional positive mean | Conditional positive median | Surface mean | Surface median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3492782 | 46.6518% | 4.70915 | 2.46183 | 2.51225 | 0.194753 |
| 0DTE | 2430 | 26.9959% | 13.7837 | 11.8273 | 10.0627 | 6.34695 |
| 1DTE | 77236 | 19.9298% | 18.2168 | 11.0047 | 14.5862 | 6.14152 |
| 2DTE | 102320 | 28.5096% | 13.6613 | 8.96879 | 9.76653 | 3.79225 |

Whole-surface error against `market.py` `fit`:

| Measure | Half-spreads | Normalized price |
| --- | ---: | ---: |
| Mean absolute error | 3.20471 | 0.0001874 |
| Median absolute error | 1.19338 | 0.0001049 |
| Root mean square error | — | 0.000317335 |
| Signed bias | — | -1.8177e-05 |

Mean training objective: `2.6724829`.

## 2b. Fine-tuned Global 140

| Bucket | Options | Zero excess | Conditional positive mean | Conditional positive median | Surface mean | Surface median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3492782 | 70.5474% | 3.95646 | 1.27946 | 1.16528 | 0 |
| 0DTE | 2430 | 25.9671% | 13.3238 | 11.7761 | 9.864 | 6.58426 |
| 1DTE | 77236 | 22.4274% | 16.0762 | 9.32349 | 12.4707 | 4.90189 |
| 2DTE | 102320 | 31.2412% | 11.4875 | 7.32954 | 7.89865 | 2.77828 |

Whole-surface error against `market.py` `fit`:

| Measure | Half-spreads | Normalized price |
| --- | ---: | ---: |
| Mean absolute error | 1.71877 | 8.95966e-05 |
| Median absolute error | 0.437192 | 3.80687e-05 |
| Root mean square error | — | 0.000208677 |
| Signed bias | — | -1.78565e-05 |

Mean training objective: `1.251221`.

Checkpoint SHA-256: `7cb25766857c1f599e8208dc515572ff564a282c1f128b2d6bd08203a5ef5a77`.
Latent-state SHA-256: `518b58a7955f3c66e9f9146320b90ecf6db2a326bcaaa09e120192622247055d` with shape `[874, 140]`.
Model identity: `LatentStateSanosModel` version `1.0 { FlexGeometryConfig: 1.1 }`.

Market-training convergence: **True**; trailing-window relative improvement `0.0005014442968001449`; final gap to best `4.3476683969644e-06`.

## 2a. Initial Triangular Uniform 140

| Bucket | Options | Zero excess | Conditional positive mean | Conditional positive median | Surface mean | Surface median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3492782 | 59.5939% | 4.65432 | 1.7693 | 1.88063 | 0 |
| 0DTE | 2430 | 26.3786% | 13.598 | 11.7672 | 10.011 | 6.35004 |
| 1DTE | 77236 | 20.1577% | 18.2135 | 10.8585 | 14.5421 | 6.1895 |
| 2DTE | 102320 | 28.2985% | 14.0903 | 9.0322 | 10.103 | 3.94324 |

Whole-surface error against `market.py` `fit`:

| Measure | Half-spreads | Normalized price |
| --- | ---: | ---: |
| Mean absolute error | 2.49842 | 0.000136657 |
| Median absolute error | 0.654207 | 5.99009e-05 |
| Root mean square error | — | 0.000284328 |
| Signed bias | — | -1.8658e-05 |

Mean training objective: `2.0055504`.

## 2b. Fine-tuned Triangular Uniform 140

| Bucket | Options | Zero excess | Conditional positive mean | Conditional positive median | Surface mean | Surface median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3492782 | 63.4394% | 4.57363 | 1.68095 | 1.67214 | 0 |
| 0DTE | 2430 | 26.1317% | 13.5032 | 11.7452 | 9.97461 | 6.4211 |
| 1DTE | 77236 | 20.5526% | 17.7671 | 10.5694 | 14.1155 | 5.95457 |
| 2DTE | 102320 | 28.7940% | 13.6534 | 8.78102 | 9.72207 | 3.7419 |

Whole-surface error against `market.py` `fit`:

| Measure | Half-spreads | Normalized price |
| --- | ---: | ---: |
| Mean absolute error | 2.26841 | 0.000121645 |
| Median absolute error | 0.567056 | 5.12187e-05 |
| Root mean square error | — | 0.000262384 |
| Signed bias | — | -2.00302e-05 |

Mean training objective: `1.7855644`.

Checkpoint SHA-256: `baaa51e579e8265540dc656aaadc9a21af2117fab7a1c7a9c58f0b7be88170b2`.
Latent-state SHA-256: `18c41b9a69d27aa8ad920399f5d2b483b26cb277050d5570dd6937f3c9cad184` with shape `[874, 140]`.
Model identity: `LatentStateSanosModel` version `1.0 { ForwardMuFlexGeometryConfig: 1.0  bf7508bf`.

Market-training convergence: **True**; trailing-window relative improvement `0.0004204923631391082`; final gap to best `2.1967963973782445e-05`.
