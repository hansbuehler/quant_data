# Phase 3 differentiable half-spread network

A two-hidden-layer width-64 SiLU network predicts log normalized half-spread and remains native Torch code inside the simulator. Training uses only 2022--2024 and gives each date equal total weight; 2025 is frozen temporal validation. No 2026 data is read.

## Tick handling

- Input books below `$0.25` premium are excluded from training.
- Inference is floored at half of a $0.05 tick below $3 premium and half of a $0.10 tick otherwise.
- The filter retains 99.960% of training books and 99.984% of validation books.
- Dollar premium is approximated as normalized AF price times spot. This intentionally avoids rates/forward inputs but is an approximation.

## 2025 results

| universe | date-balanced MAE | date-balanced RMSE | date-balanced R2 |
| --- | ---: | ---: | ---: |
| eligible | 0.00013638627 | 0.00025765338 | 0.499750 |
| all, tick-floor extrapolation | 0.00013636579 | 0.00025763555 | 0.499764 |

The low-premium threshold, exchange-tick approximation, architecture, log-Huber loss, and optimizer schedule are ad-hoc Phase 3 choices. The all-book row explicitly extrapolates the network beyond its training filter; the eligible row is the primary test.
