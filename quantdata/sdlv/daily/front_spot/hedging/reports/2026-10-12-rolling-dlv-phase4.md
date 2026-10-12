# Rolling SDLV Phase-4 calibration

## Scope

The Phase-4 implementation uses only the distributable public-candidate panel.
It fits 16 joint front-spot/DLV innovation factors on 252 exact one-session
fixed-expiry, fixed-cash-strike transitions, keeps the first eight factors for
comparable DLV-vega bumps, aligns consecutive bases without future references,
evaluates the analytical risk-neutral DLV drift, and estimates an equivalent
physical measure change.

## Corrections after the first independent review

The first review failed the gate for three reasons. All three implementation
paths were replaced or guarded before re-review:

1. The front factor is no longer inferred from the separate SANOS surface or
   from a PIT of the convolved return. The observed normalized spot is modeled
   exactly as a discrete DLV-front node times the independent 5% lognormal
   background. The annual front market price of risk maximizes this mixture
   likelihood. Factor-zero innovations are posterior means of the latent
   physical Brownian innovation under the same likelihood.
2. Factor alignment now rejects any reference whose as-of date is not strictly
   earlier than the current fit.
3. The rolling surface risk price is no longer an unregularized projection of
   thousands of annualized coordinate drifts. It is fitted to realized daily
   transitions. Ridge strength is selected on the final third of the trailing
   window using the one-standard-error rule, then refitted on all 252 dates.
   Date-clustered sandwich standard errors, holdout improvement, and the daily
   Brownian-shift norm are stored. P simulation refuses a fit that fails its
   precommitted holdout-improvement or daily-shift checks.

The coordinatewise AR fit is now a diagnostic. It uses a five-lag Newey--West
standard error with products included only across linked one-session pairs.

## Controlled identification check

Synthetic observations generated from the exact DLV-front/background mixture
recover injected annual front risk prices of 2 and 5 within 0.2. The filtered
physical innovation has mean zero at the fitted likelihood optimum. This is a
regression test, not a claim that the latent front node is observed.

## Deterministic public-data check

For the 2023-01-17 origin:

- rank-16 innovation variance coverage: `86.4983%`;
- HAC-screened physical-AR coordinates: `1,669 / 4,800`;
- front market price of risk: `-1.20750`;
- front likelihood standard error: `1.03203`;
- selected rolling surface ridge: `10.0`;
- rolling daily risk-price norm: `0.370618`;
- validation loss: `32,906.24`;
- fixed-front-only validation loss: `33,533.28`;
- signed validation improvement: `1.869904%`;
- paired validation gain: `627.0402`;
- linked-chain Newey--West standard error: `91.6738`;
- paired validation-gain statistic: `6.83991`;
- precommitted P-fit acceptance: passed.

A 32-path, eight-substep CPU P smoke was finite over all 163 hedge instruments.
This only checks the executable path; it is not a statistical hedge result.

## Verification

- Exact-mixture likelihood recovery is tested at two nonzero risk prices.
- Future-reference factor alignment is rejected by regression test.
- Projection tests cover free and partially fixed market prices of risk and
  finite clustered standard errors.
- The current-state calibration separates conditional and unconditional P
  drift diagnostics and fixes factor zero to the likelihood estimate.
- Permutation/sign and near-degenerate Procrustes tests preserve covariance and
  state reconstruction.
- Independent second-remediation review passed the Phase-4 gate. The review
  report records the remaining non-blocking statistical qualifications.
