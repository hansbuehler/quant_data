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

## 2a. Initial Triangular Uniform 140

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

## 2b. Fine-tuned Triangular Uniform 140

| Bucket | Options | Zero excess | Conditional positive mean | Conditional positive median | Surface mean | Surface median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3492782 | 68.4331% | 4.28082 | 1.47553 | 1.35132 | 0 |
| 0DTE | 2430 | 25.7202% | 13.2823 | 11.6333 | 9.86605 | 6.57406 |
| 1DTE | 77236 | 21.6182% | 16.6554 | 9.83001 | 13.0548 | 5.41433 |
| 2DTE | 102320 | 29.9257% | 12.5025 | 8.03133 | 8.76103 | 3.20999 |

Whole-surface error against `market.py` `fit`:

| Measure | Half-spreads | Normalized price |
| --- | ---: | ---: |
| Mean absolute error | 1.91515 | 0.000100084 |
| Median absolute error | 0.46657 | 4.1346e-05 |
| Root mean square error | — | 0.000226456 |
| Signed bias | — | -1.87602e-05 |

Mean training objective: `1.4470793`.

Checkpoint SHA-256: `8206a528aee26420c49572efa3d1d62742785e993361e1ddd9e0b03d27ac20c0`.
Latent-state SHA-256: `55c92dcdf59b66cc65a85d87f92bac5067488f46f31b6b3e21828baa811d534b` with shape `[874, 140]`.
Model identity: `LatentStateSanosModel` version `1.0 { ForwardMuFlexGeometryConfig: 1.0  bf7508bf`.

Market-training convergence: **True**; trailing-window relative improvement `0.0002494304243402717`; final gap to best `3.5613670031705227e-06`.
