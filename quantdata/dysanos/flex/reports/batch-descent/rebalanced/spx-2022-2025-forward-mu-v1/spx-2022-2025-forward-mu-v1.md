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

## 2a. Initial Triangular Conservative 140

| Bucket | Options | Zero excess | Conditional positive mean | Conditional positive median | Surface mean | Surface median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3492782 | 69.9030% | 4.11641 | 1.39807 | 1.23892 | 0 |
| 0DTE | 2430 | 25.7202% | 13.2224 | 11.6874 | 9.82158 | 6.66511 |
| 1DTE | 77236 | 22.3795% | 16.0721 | 9.27182 | 12.4753 | 4.97466 |
| 2DTE | 102320 | 30.4603% | 11.8109 | 7.57911 | 8.21327 | 2.95864 |

Whole-surface error against `market.py` `fit`:

| Measure | Half-spreads | Normalized price |
| --- | ---: | ---: |
| Mean absolute error | 1.79677 | 9.48205e-05 |
| Median absolute error | 0.455731 | 3.92617e-05 |
| Root mean square error | — | 0.000218615 |
| Signed bias | — | -1.75985e-05 |

Mean training objective: `1.3287561`.

## 2b. Fine-tuned Triangular Conservative 140

| Bucket | Options | Zero excess | Conditional positive mean | Conditional positive median | Surface mean | Surface median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3492782 | 70.1150% | 4.11256 | 1.4024 | 1.22904 | 0 |
| 0DTE | 2430 | 25.6790% | 13.2 | 11.656 | 9.81036 | 6.65569 |
| 1DTE | 77236 | 22.4533% | 15.9902 | 9.21448 | 12.3998 | 4.93238 |
| 2DTE | 102320 | 30.5151% | 11.7346 | 7.52346 | 8.1538 | 2.93419 |

Whole-surface error against `market.py` `fit`:

| Measure | Half-spreads | Normalized price |
| --- | ---: | ---: |
| Mean absolute error | 1.78542 | 9.41062e-05 |
| Median absolute error | 0.452319 | 3.89e-05 |
| Root mean square error | — | 0.000217026 |
| Signed bias | — | -1.71902e-05 |

Mean training objective: `1.3183099`.

Checkpoint SHA-256: `9899c8bc5c71b2f39783a2e63833295f9bf5f7ad658ad666f26e5d2bc0c48b05`.
Latent-state SHA-256: `c7cc77cf1d379f9246ab85b86aada07fc8100f8de62a57d71104012a8645c4f7` with shape `[874, 140]`.
Model identity: `LatentStateSanosModel` version `1.0 { ForwardMuFlexGeometryConfig: 1.0  bf7508bf`.

Market-training convergence: **True**; trailing-window relative improvement `4.441118371336421e-05`; final gap to best `6.406828854291461e-06`.
