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

## Ideal triangular approximation of the fitted Global surface

Grid 0 is fixed at each candidate size. One-state exchanges then optimize the remaining 140-state budget against decoded Global raw-surface variation.

| Grid 0 states | Explained Global variation | Optimized allocation |
|---:|---:|---|
| 10 | 97.0013% | `[10, 15, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5]` |
| 20 | 98.4127% | `[20, 15, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 4]` |
| 30 | 98.9921% | `[30, 14, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 3]` |
| 40 | 99.3011% | `[40, 13, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 2]` |
