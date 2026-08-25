# Phase 3 market-to-Strong encoder

Status: **complete**.

The decoder and encoder were trained only through 2024-12-31. All 2025 dates were frozen temporal validation, and every date carried total loss weight one regardless of its quote count.

| model/split | date objective | all inside | weekly inside | state MSE |
| --- | ---: | ---: | ---: | ---: |
| train-only Strong direct | n/a | 71.5282% | n/a | 0 |
| encoder train | 1.12707 | 69.6294% | 81.2309% | 2.12712 |
| encoder 2025 OOS | 8.75539 | 25.1493% | 23.4330% | 12.1091 |

The 2025 state error uses a least-squares projection of independent 2025 native rows into the frozen train-only Strong gauge. It is an oracle diagnostic, not a fitted validation state.

## State variation

- Target train mean coordinate standard deviation: `0.913048`
- Encoded train mean coordinate standard deviation: `0.843405`
- Encoded 2025 mean coordinate standard deviation: `0.709021`

## Ad-hoc choices

- 14 x 11 fixed expiry/NMS RBF anchors
- 5-business-day expiry bandwidth and 0.5-NMS bandwidth
- two SiLU hidden layers of width 512
- AdamW learning rates, epoch counts, clipping, and weight decay
- ordinary Euclidean Strong-state supervision normalized per coordinate
- absolute supervised-reconstruction convergence tolerance 1e-5

See `phase3-market-encoder.json` for complete histories, convergence audits, surface errors, checkpoint digests, and source provenance.
