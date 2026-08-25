# Phase 3 half-spread attribution

This CPU-only benchmark predicts normalized price half-spread. Every date has 
equal total training weight, so quote-rich dates do not dominate. The temporal 
split is frozen at training through 2024-12-31 and validation from 2025-01-02 
through 2025-08-29; no 2026 data is read.

## Data

- Training: 712 dates and 2,784,378 books.
- Validation: 162 dates and 708,404 books.
- Calls and puts are separate real books. Prices and spreads are normalized by 
  discounted forward in the upstream market pipeline.
- Greeks use the smooth AF fitted call, with puts converted through normalized 
  parity. Delta is `Phi(d1)` for calls and `Phi(d1)-1` for puts.

## Temporal out-of-sample results

| model | date-balanced MAE | date-balanced RMSE | date-balanced R2 | raw MAE |
| --- | ---: | ---: | ---: | ---: |
| linear | 0.00023552034 | 0.00033782761 | 0.139891 | 0.00024454448 |
| hist_gradient_boosting | 0.00019720279 | 0.00028810117 | 0.374463 | 0.00020342884 |

## Model choices and limitations

The linear design is intercept plus `vega/sqrt(TTE)`, signed delta, and NMS. 
The nonlinear control is a deterministic histogram gradient booster using 
vega, `sqrt(TTE)`, NMS, log strike, and signed delta. Its depth/leaf count, 
learning rate, iteration count, minimum leaf size, and L2 penalty are ad-hoc 
Phase 3 choices, not validation-selected hyperparameters. Squared-error loss 
targets expected half-spread rather than a conditional median.

The model predicts half-spread first and the simulator divides by current 
option price. Reported proportional ratios therefore become unstable for 
near-zero option values; the JSON records the explicit price floor and robust 
quantiles. Zero-strike calls are spot positions and bypass this model in favor 
of a separately configured spot cost. The fitted linear formula has a native, 
gradient-preserving Torch inference method; the tree model remains a CPU 
attribution benchmark because hard tree splits have no useful state gradient. 
The model is an attribution benchmark, not an execution guarantee, and 2025 
is used only for temporal evaluation.
