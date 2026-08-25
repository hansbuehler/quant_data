# Phase 3 flat native encoder, state 2828

Status: **complete**.

This is same-set amortized calibration: each date's full eligible quote set enters the invariant RBF summary and is reconstructed through the capped-unbounded native SANOS state. The frozen 2025 split measures temporal generalization of that calibration map, not masked-quote valuation.

| split | date objective | all inside | weekly inside | positive mean | positive median |
| --- | ---: | ---: | ---: | ---: | ---: |
| train through 2024 | 1.06931 | 72.8000% | 86.0515% | 3.66671 | 1.26853 |
| frozen 2025 | 7.11224 | 28.2805% | 26.1971% | 9.30596 | 6.16143 |
| 2025 direct native ceiling | 1.59667 | 69.1676% | 83.1603% | 4.89178 | 1.48363 |

The explicit surface-state size is `2828`; the native SANOS output size is `2828`.

DLV² uses `min_dlvsq + exp(z)`, is mechanically capped for pricing at `max_dlvsq`, and carries the AAE smooth log-coordinate cap penalty.

| split | DLV² nodes at/above cap | mean surface cap penalty |
| --- | ---: | ---: |
| train through 2024 | 0.000000% | 4.6238e-06 |
| frozen 2025 | 0.081349% | 6.27536 |
