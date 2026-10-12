# Independent Phase-4 remediation re-review

## Gate recommendation

**FAIL.** The exact front-mixture likelihood, future-reference guard, realized-
transition projection, and executable P path are materially improved. However,
the advertised blocked holdout is fitted with a representation learned from the
holdout itself, and the default P acceptance predicate can accept a fit that is
worse than its fixed-front baseline. Both must be corrected before Phase 4 is
used as a production P generator.

## Blockers

1. **The trailing validation block is not untouched.**
   `fit_rolling_dlv_calibration` first fits the front likelihood, AR coefficients,
   and rank-16 loading basis on all 252 transitions. Only afterward does
   `fit_rolling_constant_physical_drift` split the same dates 168/84 for ridge
   selection. Consequently the validation dates influence factor zero, every
   PCA loading, and every validation-date Q drift before candidate losses are
   evaluated. Fit the representation on the first block for selection, then
   refit it on all 252 dates only after selecting the ridge. Alternatively,
   describe the current result as conditional tuning, not held-out validation.

   This does not overturn the reported date, but it is a real protocol defect.
   On 2023-01-17 the published full-window representation selects ridge `1.0`
   with `6.3239%` improvement. A representation independently fitted on only
   the first 168 transitions also selects `1.0`, with `5.9326%` improvement;
   its front risk price is `-1.55578` rather than `-1.20750`.

2. **The default P acceptance test does not reject a worse validation fit.**
   Validation improvement is clamped as
   `max(0, (baseline - selected) / baseline)`, while the default required
   improvement is `0.0`. A negative raw improvement therefore becomes zero and
   passes. Preserve the signed improvement and require a strictly positive,
   precommitted margin (preferably on the paired daily loss differences) before
   setting `accepted=True`.

   The actual 2023-01-17 fit would survive a meaningful check: the paired
   baseline-minus-selected validation gain is `2,123.17`, with Newey--West(5)
   standard error `326.83` and statistic `6.50`. The defect is in the gate's
   general semantics, not this date's observed result.

## Passed checks

- The front observation likelihood is the correct finite mixture of the DLV
  front-node probabilities and an independent 5% lognormal background. The
  filtered score is the posterior mean of the latent physical innovation.
  Synthetic injected annual risk prices `2` and `5` are recovered within `0.2`.
- The real 2023-01-17 likelihood converges successfully: risk price `-1.20750`,
  curvature standard error `1.03203`, optimizer status `0`, and filtered-score
  mean `-1.48e-7`.
- A factor-alignment reference must strictly predate the current as-of date;
  the future-reference regression test passes.
- The coordinatewise AR diagnostic uses exact linked-session chains and a
  Bartlett/Newey--West score variance. It is diagnostic only; the pooled P fit
  correctly uses realized annualized transitions `252 * Delta log(DLVsq)`.
- Free surface-factor risk-price errors use a date-clustered sandwich. Units are
  consistent: daily loadings are multiplied by `sqrt(252)`, Q log drift is
  relative drift minus half annual factor variance, and the P premium is
  annual loading times annual Brownian risk price.
- P generation refuses a missing or unaccepted rolling physical fit. The real
  accepted calibration produced 32 finite CPU P paths over all 163 instruments.
- Runtime model entry points require `PublicHedgingPanel`. Static inspection
  found no private-source imports from runtime benchmark or model modules; the
  private exporter remains separated behind a lazy data-package import.

## Non-blocking corrections

- The filtered factor-zero score has variance `0.995098`, but its OLS loading is
  simulated against a fresh unit-variance Brownian and PCA is applied to a
  residual that still contains the unobserved conditional innovation. This
  overstates the real date's retained covariance trace by `1.3820`, about
  `0.169%` of total innovation trace `816.388`. Use a standardized observable
  factor or remove the posterior conditional-variance contribution before PCA.
- The one-standard-error ridge rule uses an iid standard error of daily losses.
  A HAC error is more appropriate for a contiguous time-series validation
  block. The acceptance comparison should use paired loss differences.
- Factor zero receives an observed-curvature likelihood error, not a
  serially-robust clustered error, and likelihood fitting checks finiteness but
  not `result.success`. Both should be hardened.
- Add direct regression tests for the HAC calculation, signed acceptance
  rejection, P-simulation rejection, and the genuinely training-only factor
  representation used during ridge selection.

## Verification performed

- Focused CPU suite: `31 passed` in `10.85s`.
- Deterministic public-candidate calibration: 2023-01-17, rank 16, 252 exact
  one-session transitions; reproduced `86.4983%` retained variance, ridge
  `1.0`, daily risk-price norm `1.72369`, and reported `6.3239%` validation
  improvement.
- Independent train-only representation comparison and paired Newey--West
  validation calculation, as reported above.
- CPU P smoke: 32 paths, eight substeps, 163 instruments, all finite.
