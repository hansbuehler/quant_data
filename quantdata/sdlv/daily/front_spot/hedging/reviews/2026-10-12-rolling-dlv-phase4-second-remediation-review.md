# Independent Phase-4 second-remediation review

## Gate recommendation

**PASS.** The two blockers in the prior review are fixed. Ridge selection now
uses a factor/front-likelihood representation fitted only on the first 168
transitions, and P acceptance now requires a signed economic improvement and a
positive paired linked-chain Newey--West statistic. The full 252-transition
representation is used only for the post-selection refit and current model.

## Verified remediations

1. **The validation representation is training-only.**
   `fit_rolling_dlv_calibration` copies the first 168 transitions into a
   separate history and fits `selection_factors` with an as-of date equal to
   that block's final endpoint. `fit_rolling_constant_physical_drift` rejects
   any selection fit whose origin dates, endpoint dates, or rank do not match
   the training prefix. Candidate Q drifts, density weights, front risk price,
   loading basis, and fitted risk prices all use this selection representation.
   Only after ridge selection is the chosen ridge refitted with the full
   252-transition representation.

2. **The P gate is now signed and statistically screened.**
   Validation improvement is no longer clamped. The defaults require at least
   `0.5%` improvement, paired gain statistic at least `1.96`, and daily
   Brownian-shift norm at most `3.0`. The gain standard error uses a
   Bartlett/Newey--West(5) sum and excludes autocovariance products that cross
   gaps in the exact-session chain. GROMM P generation rejects an unaccepted
   calibration before simulation.

3. **The one-standard-error rule is HAC-based.**
   Candidate losses are evaluated on the final 84 transitions using the
   training-only basis. The standard error of the best validation loss uses the
   same linked-chain Newey--West calculation, and the largest ridge within that
   error is selected. On the real date, ridge `0.1` has the minimum loss
   `30,760.55`; the HAC one-standard-error cutoff is `32,978.68`, making ridge
   `10.0` the largest eligible candidate.

## Real public-candidate check

The deterministic 2023-01-17 CPU calibration independently reproduces:

- training-only selection window: 2022-01-03 through 2022-09-09, 168 pairs;
- selected ridge: `10.0`;
- validation loss / fixed-front baseline: `32,906.24 / 33,533.28`;
- signed validation improvement: `1.869904%`;
- paired validation gain / HAC standard error: `627.0402 / 91.6738`;
- paired HAC statistic: `6.83991`;
- final daily risk-price norm: `0.370618`;
- acceptance: `True`.

A 32-path, eight-substep CPU P smoke was finite over all 163 instruments. A
copy of the same calibration with `accepted=False` was rejected before path
generation.

## Remaining non-blocking qualifications

- Ridge selection and the `1.96` gain gate use the same 84-date block. The gate
  is therefore a useful precommitted screening rule, but its statistic is not
  an unadjusted post-selection 95% hypothesis test. A separately untouched
  gate block, nested rolling validation, or resampling of the complete
  selection procedure is required for that interpretation.
- The earlier factor-zero posterior-variance qualification remains: on this
  date its filtered score variance is `0.995098`, producing an approximately
  `0.169%` covariance-trace overstatement when the loading is simulated against
  a fresh unit Brownian and PCA is fitted to the residual. This is numerically
  small here but should be corrected or explicitly documented.
- Factor zero still uses observed likelihood curvature rather than a
  serially-robust error, and likelihood fitting should reject an unsuccessful
  optimizer result rather than checking finiteness alone.
- The Phase-4 report still contains the pre-remediation ridge `1.0`, `6.3239%`
  gain, and daily norm `1.72369`; it should be updated to the values above.
- Tests cover the positive default thresholds and linked-chain Newey--West
  helper, but an end-to-end synthetic regression should also prove that a
  negative-gain fit is unaccepted and that holdout perturbations cannot change
  the training-only selection representation.

## Verification performed

- Focused CPU suite: `33 passed` in `12.39s`.
- Independent reconstruction of all seven ridge candidate losses and their
  HAC errors on the public 2023-01-17 window.
- Deterministic public-candidate calibration and accepted/rejected P execution
  checks described above.
- No implementation files were modified by this review.
