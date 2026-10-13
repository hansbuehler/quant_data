# Robust-covariance mean-variance production subset

The three robust-covariance mean-variance policies are complete and
authenticated on the same public dataset and current target-notional cohort as
the clean SANOS-LV reference run. The remaining covariance objectives are still
running, so these values are an authenticated subset rather than the final
model comparison.

## Provenance

- Dataset manifest SHA-256:
  `58741552e0f6eea82378a5f26a61e1df04b9d14f3ab166883089812cf903007a`.
- Target-notional identity:
  `1.1:fd6839a0915e3d7c6ee793adc3b025568f627c187f2c7eb6bc6fce540818d769`.
- Every policy contains all 611 eligible origins and 12,220 target records.
- Every policy has zero position-cap hits and zero nonconverged results.

| Risk parameter | Run identity | Results SHA-256 |
|---:|:---|:---|
| 0.1 | `8c8ae181202b90a0f28dc8a3f689729826843712178093aae33c055025f082ec` | `5ec474dd6c5f2e9efbb839360b4e4ce39df0730f9356a844b16ee04854e1b57d` |
| 1 | `d86904f55a728d229567283655b13c21e3cdaa70bb18ba079c59b2af305de310` | `17cfc415edac6c4e961a6b623018a1b217b29bdb8db8220e500a65a9e9664369` |
| 10 | `d4215ac9308899d9a1618223bc242f4fd43e9d39af4f7a10706ce6e57feceb49` | `47ff392ba1b453a01da72f2d5b9905af085e7fd0cf00c026472b6cb623406142` |

Each file hash equals the hash pinned by its successful production-queue
record.

## Interim headline metrics

The table uses the established report formulas. RMSE, expected shortfall, and
transaction cost are basis points of inception normalized spot. Gross exposure
is in normalized entry notional. Maximum drawdown is cumulative and therefore
is not directly comparable to one-period RMSE.

| Policy | Pure RMSE | RMSE reduction | Pure ES 95% | Mean cost | Mean gross exposure | Economic book MDD |
|:---|---:|---:|---:|---:|---:|---:|
| SANOS-LV reference | 70.80 | 56.61% | 134.71 | 10.10 | 2.6640 | 6,779.67 |
| Covariance MV, 0.1 | 150.73 | 7.62% | 215.13 | 0.66 | 0.0048 | 1,723.98 |
| Covariance MV, 1 | 135.85 | 16.74% | 209.59 | 0.74 | 0.0062 | 1,720.29 |
| Covariance MV, 10 | 129.25 | 20.79% | 257.48 | 1.48 | 0.0518 | 1,754.14 |

The common unhedged RMSE is 163.16 basis points. On this completed subset,
SANOS-LV achieves the largest RMSE reduction, while the covariance policies
trade much smaller books and incur much lower entry cost and cumulative book
drawdown. Final interpretation waits for the remaining fitted objectives and
models.
