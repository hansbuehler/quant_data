# Phase 3 Hierarchical market-to-Strong encoder

Status: **complete**.

The decoder and encoder were trained only through 2024-12-31. All 2025 dates were frozen temporal validation, and every date carried total loss weight one regardless of its quote count.

| model/split | date objective | all inside | weekly inside | state MSE |
| --- | ---: | ---: | ---: | ---: |
| train-only Strong direct | n/a | 71.5282% | n/a | 0 |
| encoder train | 1.10474 | 72.1516% | 85.6786% | 1.72245 |
| encoder 2025 OOS | 11.5404 | 23.7739% | 20.1537% | 12.9628 |

The 2025 state error uses a least-squares projection of independent 2025 native rows into the frozen train-only Strong gauge. It is an oracle diagnostic, not a fitted validation state.

## State variation

- Target train mean coordinate standard deviation: `0.913048`
- Encoded train mean coordinate standard deviation: `0.928335`
- Encoded 2025 mean coordinate standard deviation: `0.852341`

## Ad-hoc choices

- 14 x 11 fixed expiry/NMS RBF anchors
- 5-business-day expiry bandwidth and 0.5-NMS bandwidth
- fourteen expiry-causal two-layer width-128 SiLU networks; block j uses RBF expiry anchors <=j and predicted states <j
- AdamW learning rates, epoch counts, clipping, and weight decay
- ordinary Euclidean strong-state supervision normalized per coordinate
- absolute supervised-reconstruction convergence tolerance 1e-5
- 0.2% trailing supervised convergence tolerance at the expiry-causal non-zero reconstruction floor

See `phase3-market-encoder-hierarchical.json` for complete histories, convergence audits, surface errors, checkpoint digests, and source provenance.
