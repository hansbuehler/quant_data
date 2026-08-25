# FLEX expiry-state allocation diagnostic

Days: `874`. Global state: `140`. 
Contiguous out-of-fold partitions: `5`.

## Grid 0 rank spectrum

| Target | 90% | 95% | 99% | 99.5% | 99.9% | 99.99% |
|---|---:|---:|---:|---:|---:|---:|
| Grid 0 itself | 5 | 7 | 13 | 15 | 23 | 34 |
| Full Global surface observable through grid 0 | 16 | 29 | 71 | 90 | 123 | 138 |

## Sequential Direct-grid to Global-state diagnostic

Selected relative ridge: `0.1`.

| Retained predictable gain per step | Total states | OOF surface variance explained | Allocation, grid 0 to 13 |
|---:|---:|---:|---|
| 50% | 17 | 56.5192% | `[1, 2, 1, 0, 1, 0, 0, 0, 0, 9, 0, 1, 1, 1]` |
| 70% | 17 | 59.6026% | `[1, 3, 0, 0, 1, 9, 0, 0, 0, 0, 0, 1, 1, 1]` |
| 80% | 19 | 62.4195% | `[2, 4, 0, 0, 1, 0, 0, 0, 0, 10, 0, 0, 1, 1]` |
| 90% | 22 | 59.4665% | `[4, 4, 0, 0, 0, 0, 11, 0, 0, 0, 0, 1, 1, 1]` |
| 95% | 23 | 63.2128% | `[5, 4, 0, 0, 0, 0, 11, 0, 0, 0, 0, 1, 1, 1]` |
| 99% | 54 | 63.4267% | `[11, 0, 3, 0, 0, 0, 0, 0, 38, 0, 0, 0, 1, 1]` |

These out-of-fold allocations are a rejection diagnostic, not a recommended budget: even all Direct expiry blocks recover only about 63% of the independently fine-tuned Global surface through this linear map. Raw Direct and Global coordinates therefore do not share a sufficiently stable gauge.

## Conditional grid 1 rank after grid 0

| Grid 0 states | Residual Global variation | Grid 1 rank for 90% | 95% | 99% |
|---:|---:|---:|---:|---:|
| 20 | 7.8382% | 57 | 76 | 104 |
| 30 | 4.7615% | 61 | 77 | 99 |

## Ideal triangular approximation of the fitted Global surface

Grid 0 is fixed at each candidate size. One-state exchanges then optimize the remaining 140-state budget against decoded Global raw-surface variation.

| Grid 0 states | Explained Global variation | Optimized allocation |
|---:|---:|---|
| 20 | 99.2546% | `[20, 108, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` |
| 30 | 99.5919% | `[30, 98, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` |
