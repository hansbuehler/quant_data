# PCA-IV-AR(1) versus SANOS

Status: **incomplete**.

GridMarketDefinition.load() first fits a smooth statically arbitrage-free SANOS surface to each date and evaluates it on the regular grid. This comparison tests the SANOS generative decoder against direct PCA when both use the same cleaned, SANOS-interpolated mid_vols observations; it does not test SANOS preprocessing against raw OptionMetrics interpolation.

Static surface arbitrage and sampled fixed-contract return diagnostics are distinct. Failure to detect a one-sided opportunity in the sampled contracts does not prove absence of arbitrary dynamic arbitrage.

| Run | Matched | PCA static fraction | SANOS static fraction | PCA/SANOS validation local candidates |
|---|---:|---:|---:|---:|
| paths=1000,alpha=5,mode=one-step,return=expiry | no | 0.987635 | N/A | 0 / N/A |
| paths=1000,alpha=5,mode=one-step-weekly,return=expiry | no | 0.964996 | N/A | 0 / N/A |
| paths=1000,alpha=5,mode=path,return=expiry | no | 0.990398 | N/A | 0 / N/A |
| paths=1000,alpha=5,mode=weekly,return=expiry | no | 0.985786 | N/A | 0 / N/A |
| paths=1000,alpha=None,mode=one-step,return=expiry | no | 0.987635 | N/A | 0 / N/A |
| paths=1000,alpha=None,mode=one-step-weekly,return=expiry | no | 0.964996 | N/A | 0 / N/A |
| paths=1000,alpha=None,mode=path,return=expiry | no | 0.986049 | N/A | 0 / N/A |
| paths=1000,alpha=None,mode=weekly,return=expiry | no | 0.977543 | N/A | 0 / N/A |

## Incomplete matches

- paths=1000,alpha=5,mode=one-step,return=expiry: expected one nominal SANOS report, found 0
- paths=1000,alpha=5,mode=one-step-weekly,return=expiry: expected one nominal SANOS report, found 0
- paths=1000,alpha=5,mode=path,return=expiry: expected one nominal SANOS report, found 0
- paths=1000,alpha=5,mode=weekly,return=expiry: expected one nominal SANOS report, found 0
- paths=1000,alpha=None,mode=one-step,return=expiry: expected one nominal SANOS report, found 0
- paths=1000,alpha=None,mode=one-step-weekly,return=expiry: expected one nominal SANOS report, found 0
- paths=1000,alpha=None,mode=path,return=expiry: expected one nominal SANOS report, found 0
- paths=1000,alpha=None,mode=weekly,return=expiry: expected one nominal SANOS report, found 0

## Interpretation

No material-improvement conclusion is issued until every PCA run has one complete SANOS report with all material configuration fields verified and the required metrics present.
