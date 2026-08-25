# Phase 3 flat native encoder, state 140

Status: **complete**.

This is same-set amortized calibration: each date's full eligible quote set enters the invariant RBF summary and is reconstructed through the capped-unbounded native SANOS state. The frozen 2025 split measures temporal generalization of that calibration map, not masked-quote valuation.

| split | date objective | all inside | weekly inside | positive mean | positive median |
| --- | ---: | ---: | ---: | ---: | ---: |
| train through 2024 | 1.08729 | 71.3386% | 83.7714% | 3.53579 | 1.19577 |
| frozen 2025 | 8.15621 | 26.2271% | 24.4106% | 10.4545 | 7.09186 |
| 2025 direct native ceiling | 1.59667 | 69.1676% | 83.1603% | 4.89178 | 1.48363 |

The explicit surface-state size is `140`; the native SANOS output size is `2828`.

DLV² uses `min_dlvsq + exp(z)`, is mechanically capped for pricing at `max_dlvsq`, and carries the AAE smooth log-coordinate cap penalty.

| split | DLV² nodes at/above cap | mean surface cap penalty |
| --- | ---: | ---: |
| train through 2024 | 0.000000% | 1.45736e-05 |
| frozen 2025 | 0.236772% | 71.2551 |
