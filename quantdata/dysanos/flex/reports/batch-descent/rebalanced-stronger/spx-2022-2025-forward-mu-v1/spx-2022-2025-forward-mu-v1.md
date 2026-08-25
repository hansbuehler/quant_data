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

## 2a. Initial Triangular Stronger 140

| Bucket | Options | Zero excess | Conditional positive mean | Conditional positive median | Surface mean | Surface median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3492782 | 43.3095% | 5.70827 | 2.96203 | 3.23605 | 0.435268 |
| 0DTE | 2430 | 26.4198% | 13.7518 | 11.662 | 10.1186 | 6.3731 |
| 1DTE | 77236 | 19.8237% | 18.7841 | 11.0226 | 15.0604 | 6.10912 |
| 2DTE | 102320 | 27.1306% | 14.6777 | 9.36465 | 10.6956 | 4.16322 |

Whole-surface error against `market.py` `fit`:

| Measure | Half-spreads | Normalized price |
| --- | ---: | ---: |
| Mean absolute error | 3.9505 | 0.000253648 |
| Median absolute error | 1.43262 | 0.0001221 |
| Root mean square error | — | 0.000457225 |
| Signed bias | — | -2.0616e-05 |

Mean training objective: `3.4335702`.

## 2b. Fine-tuned Triangular Stronger 140

| Bucket | Options | Zero excess | Conditional positive mean | Conditional positive median | Surface mean | Surface median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3492782 | 70.9779% | 4.04483 | 1.35446 | 1.1739 | 0 |
| 0DTE | 2430 | 25.8848% | 13.2011 | 11.7118 | 9.78402 | 6.66053 |
| 1DTE | 77236 | 22.7264% | 15.7325 | 8.97679 | 12.1571 | 4.76445 |
| 2DTE | 102320 | 31.0106% | 11.4926 | 7.3429 | 7.92867 | 2.81306 |

Whole-surface error against `market.py` `fit`:

| Measure | Half-spreads | Normalized price |
| --- | ---: | ---: |
| Mean absolute error | 1.72699 | 9.04588e-05 |
| Median absolute error | 0.44471 | 3.77529e-05 |
| Root mean square error | — | 0.000210353 |
| Signed bias | — | -1.6929e-05 |

Mean training objective: `1.2602461`.

Checkpoint SHA-256: `367fe85e67b0a09719c74d72dca5ab5fbfb16564e3c5828b13e482a3cfbb638a`.
Latent-state SHA-256: `99d0054a3bf3e03113571d1a571b1af2722d5e60e509d2006d5fb3210ddfe0f3` with shape `[874, 140]`.
Model identity: `LatentStateSanosModel` version `1.0 { ForwardMuFlexGeometryConfig: 1.0  bf7508bf`.

Market-training convergence: **True**; trailing-window relative improvement `2.7261475718204797e-05`; final gap to best `7.042789489802835e-06`.
