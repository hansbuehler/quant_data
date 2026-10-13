# SANOS-LV current-notional production authentication

The clean SANOS-LV daily production run is complete and authenticated against
the distributable public-candidate dataset and the fixed 2022 target-notional
calibration.

## Scope

- Dataset manifest SHA-256:
  `58741552e0f6eea82378a5f26a61e1df04b9d14f3ab166883089812cf903007a`.
- Evaluation window: every eligible daily origin from 2023-01-01 through the
  final eligible public-panel origin.
- Completed origins: 611 of 611.
- Completed target records: 12,220, comprising 20 daily digital targets at
  every origin.
- Run identity:
  `c67b549ca9eeb404be9e70c6fe2ba22c827d8d4ec75115e403a0bb2e54666a68`.
- Target-notional identity:
  `1.1:fd6839a0915e3d7c6ee793adc3b025568f627c187f2c7eb6bc6fce540818d769`.
- Result SHA-256:
  `9e4252f60540d6830eacda10b451677320760404e4599a5caa4e346367262b2d`.
- Numerical diagnostics: zero position-cap hits and zero nonconverged results.

## Authentication procedure

The production queue pinned matrix SHA-256
`2b02cf2f88fc81ebdfb3cc1e61705963ccf7ff991767c3854577a9806ff5a3d7`
and recorded a successful exit together with the result hash above. The dataset
manifest on disk reproduces the run's manifest hash. The fixed target-notional
artifact contains calibration hash
`fd6839a0915e3d7c6ee793adc3b025568f627c187f2c7eb6bc6fce540818d769`,
which reproduces the target-notional identity in `run.json`.

The current `summarize_completed_daily_run` validator then reloaded the public
panel, recomputed eligible origins, parsed and validated every dated record,
reconciled all 12,220 records with the immutable 20-target book, and rebuilt
`results.json`. The rebuilt file had the same byte-level SHA-256 as the
production artifact.

## Reporting boundary

The historical-SANOS result currently referenced by the summary PDF uses a
superseded target-notional identity. It must not be mixed with this authenticated
SANOS-LV run. Comparative tables are regenerated only after the corrected
historical-SANOS production sweep completes under the same current target
notionals.
