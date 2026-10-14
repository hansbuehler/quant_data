# Public-Candidate Dataset Validation

## Verdict

**PASS for schema integrity, source-data exclusion, checksums, and complete
SANOS/DLV reconstruction.** The artifact is technically self-contained and the
hedging code does not need the original WRDS data root. Its manifest remains
labelled `public_candidate`; this validation is not a grant of redistribution
rights or a substitute for the release metadata required to relabel it
`public`.

## Artifact

- Directory:
  `/home/hans/quantdata_data/datasets/hedging/spx-2022-2025-hedging-public-candidate-v1`
- Schema: `hedging-public-candidate-v1`
- Manifest SHA-256:
  `58741552e0f6eea82378a5f26a61e1df04b9d14f3ab166883089812cf903007a`
- Retained dates: 891, from 2022-01-03 through 2025-08-29.
- Quote-side model-price coordinates: 2,199,572.
- Paper-retained coordinates: 2,192,458.
- Recommended `(-2,2)` NMS hedge coordinates: 2,014,740.

## Validation

On 2026-10-14 the repository validator ran with
`QUANTDATA_ROOT=/tmp/quant_data/hedging/public-dataset-audit`, an otherwise empty
project data root. `validate_public_dataset(..., reconstruct_models=True)`
completed in 60.80 seconds and passed:

- all manifest file checksums and declared row counts;
- exact schemas and rejection of prohibited unexpected columns;
- normalized forward-return, expiry, and population-mask invariants;
- every stored SANOS model price from the released SANOS state and geometry;
- every stored DLV model price from the released DLV state and geometry;
- every stored SANOS and DLV Black implied volatility and validity flag; and
- every stored SANOS and DLV ATM implied volatility.

The dataset contains no absolute spot, market bid, market ask, market midpoint,
option identifier, symbol, volume, open interest, vendor Greek, or vendor
implied volatility. Stored SANOS and DLV prices are model outputs. Stored
half-spreads are normalized released cost inputs and are not prices or
midpoints.

## Release Qualification

The manifest deliberately records `distribution="public_candidate"`, null
release-rights fields, and a dirty generator worktree from the historical
export. The loader permits this status for research execution but requires a
clean generator worktree plus license, rights basis, approver, approval date,
and approval reference before accepting `distribution="public"`. No such legal
or administrative approval is inferred by this technical review.
