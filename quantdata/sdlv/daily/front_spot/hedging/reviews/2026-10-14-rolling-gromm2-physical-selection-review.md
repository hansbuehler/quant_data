# Rolling GROMM2 Physical-Selection Review

## Verdict

**PASS for the implemented candidate-versus-front-only selection and its
authenticated loading path.** No quantitative or provenance blocker was found
in the current implementation. This is an implementation gate, not completed
rolling-production evidence: the current 611-origin preparation stage is
complete, while migration, current-code rolling fitting, and daily or weekly
production remain incomplete.

## Scope

The review covered:

- `models/gromm/rolling_dynamics.py`, including the physical candidate,
  front-only null, and unavailable branches;
- `models/gromm/rolling_gromm2.py`, including dated-artifact authentication,
  redundant selection reconciliation, and authenticated legacy affine
  migration;
- `data/rolling_calibration.py`, including blocked holdout selection and the
  linked Newey--West gain statistic;
- `runners/fit_rolling_gromm2_range.py`, including complete-range
  authentication and candidate/fallback/unavailable counts;
- the focused rolling GROMM2, rolling calibration, and range tests; and
- available range artifacts under
  `/home/hans/quantdata_data/training/hedging/gromm2-rolling-v3` and the empty
  current migration target
  `/home/hans/quantdata_data/training/hedging/gromm2-rolling-v4`.

## Quantitative Findings

### Candidate gate

The full 16-factor candidate is accepted exactly when all three precommitted
conditions hold:

1. linked holdout loss improvement is at least `0.005`;
2. the linked Newey--West gain statistic is at least `1.96`; and
3. the daily Euclidean norm of the market price of risk is at most `3.0`.

`fit_constant_physical_drift_from_inputs` computes the signed validation gain,
its linked Newey--West standard error, and the resulting statistic before the
full-window refit. `_select_physical_drift` independently reconstructs the same
decision and rejects an inconsistent `candidate.accepted` flag.

The density-weighted achievable/projection fraction is persisted for both the
candidate and front-only null, but it does not enter acceptance. This is the
correct interpretation: it measures how much raw empirical drift lies in the
model's attainable diffusion span. It is a descriptive specification statistic,
not a condition for an equivalent measure or risk-neutral martingality.

### Front-only null and unavailable state

When the candidate fails, the fallback retains the independently identified
full-window front-factor market price of risk and sets the other 15 factor risk
prices exactly to zero. The projected drift is recomputed through the same
state-localized loading field; no empirical drift is injected outside the
model's diffusion span. The fallback is accepted only when its selected vector
is finite and its daily norm is at most `3.0`.

A finite constant Brownian risk-price vector gives the intended equivalent
finite-horizon Gaussian measure change. The front-only specification is a
nested null, not a claim that the remaining empirical DLV premia are zero in
the population. If the front-only vector breaches its norm gate, selection is
`unavailable`; `physical_accepted` is then false and P simulation is not
eligible. Q fitting and Q simulation remain logically separate.

## Authentication and Migration

`load_rolling_gromm2` hashes the affine history, dynamics NPZ, and dynamics
report, checks the exact public-dataset and source identities, reconstructs
candidate and baseline norms, reapplies the three candidate gates and fallback
gate, and reconciles the selected vectors and flags across report, NPZ, and
dated manifest. Projection fractions are checked for consistency only. A
tampered selection label, flag, vector, manifest identity, dated geometry, or
source hash is rejected.

The range loader requires the exact eligible public-origin cohort, verifies
each dated manifest hash, reopens every dated artifact through
`load_rolling_gromm2`, and reconciles candidate, fallback, and unavailable
counts. A range cannot become P-ready merely by editing its summary.

Legacy migration authenticates the legacy range and each source manifest, then
hard-links only source history, prepared books, and the accepted affine history
into a distinct target root. It writes a new `prepared` manifest under the
current source/config identity. Legacy dynamics, risk prices, acceptance flags,
and physical-selection decisions are not linked or copied; current dynamics
must be recomputed before the target can load as complete.

## Artifact Evidence

- `range-prepare-2023-2025-v4.json` reports `status=complete`, 611 eligible and
  selected origins, and 611 completed preparation records for public manifest
  SHA-256
  `58741552e0f6eea82378a5f26a61e1df04b9d14f3ab166883089812cf903007a`.
- The existing `range-fit-2023-2025.json` is a superseded legacy-schema partial
  fit: 143 of 611 origins, zero accepted physical origins, and the obsolete
  50% projection configuration. It is not evidence for the current selection.
- The preparation and partial legacy fit share a mutable staged root, so dated
  manifest hashes may change after the preparation snapshot as later fitting
  writes its stage. Only a completed authenticated current range is admissible
  as fit evidence.
- `/home/hans/quantdata_data/training/hedging/gromm2-rolling-v4` currently has
  no dated migrated fits. Therefore no claim is made here about current-code
  candidate/fallback frequencies, completed P availability, rolling production,
  or benchmark results.

## Verification

Focused CPU-only tests were run with CUDA hidden and JAX x64 enabled:

```text
python -m pytest -q -p no:cacheprovider \
  tests/quantdata/sdlv/daily/hedging/models/test_gromm2_rolling.py \
  tests/quantdata/sdlv/daily/hedging/test_rolling_calibration.py \
  tests/quantdata/sdlv/daily/hedging/test_fit_rolling_gromm2_range.py
```

Result: **29 passed**. The tests cover blocked holdout selection, the linked HAC
gain gate, candidate and front-only selection, manifest/NPZ/report tampering,
future-history rejection, exact transition windows, authenticated legacy affine
migration, range restart, and complete-range reauthentication.

## Limitations

- There is no completed current-code 611-origin fit or migration range to audit.
- There is no production P/Q daily or weekly result based on the new selection.
- The unavailable branch is implemented and authenticated, but the focused
  suite does not contain a dedicated end-to-end unavailable-artifact fixture;
  range validation nevertheless permits only the consistent false/false
  candidate/fallback flags with `physical_accepted=false`.
- The daily norm bound is a precommitted numerical/economic admissibility rule,
  not a proof that the selected P specification is the true physical measure.
