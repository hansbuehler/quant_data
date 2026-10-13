# GROMM2 rolling physical-projection diagnosis

## Verdict

**FAIL for the physical-measure artifact.** The 81.251% market-fit result is a
separate Q-market calibration result. The dated artifact must remain incomplete
and must not supply P scenarios. The reported 22.2983% projection is not caused
by float32 roundoff: under the current empirical AR target, the target is
genuinely outside the fitted rank-16 diffusion span. However, that target and
its estimator are not the accepted compatible-P construction used elsewhere in
the hedging benchmark, so the failure should not be interpreted as evidence
that a properly estimated compatible P measure is impossible.

The precommitted gates must not be relaxed. The appropriate remediation is to
fix the information window and precision, replace the old physical estimator
with the validated compatible-P procedure on the GROMM2 geometry, and rerun on
the first genuinely eligible origin. If the corrected artifact still misses a
gate, GROMM2 remains Q-only for that origin.

## Evidence

### 1. Float64 does not repair the projection

The failed detached run evaluated the Q drift in JAX float32 despite explicit
float64 requests (`quantdata/gromm2/estimation.py:515` and
`quantdata/gromm2/estimation.py:559`). An independent CPU recomputation with
`JAX_ENABLE_X64=1` reproduced:

| calculation | projection fraction | maximum daily oracle risk-price norm |
| --- | ---: | ---: |
| failed run | 0.22298347 | 10.99326 |
| independent float64 | 0.22299111 | 10.99215 |

The projection changes by only `7.64e-6`. Float64 is required for the numerical
contract, but its absence is not the cause of the failure. The artifact also
fails the independent daily-risk-price gate of 3.0 by a wide margin.

The current worktree enables x64 in the CLI runner before fitting
(`runners/fit_rolling_gromm2.py:21`). A correctness guard is still required at
the estimation API boundary, because `estimate_models` and
`fit_rolling_gromm2_market` can be called without that runner.

### 2. The attempted window was not a 252-transition window

`load_estimation_panel` drops every transition whose released
`business_sessions_from_previous_observation` is not one
(`quantdata/gromm2/estimation.py:232`--`quantdata/gromm2/estimation.py:235`).
The prepared history contains 253 states but only 248 accepted pairs. The four
excluded endpoints are:

- 2022-03-23;
- 2022-08-12;
- 2022-10-31;
- 2022-11-28.

Each reports a two-session gap. The serialized report therefore says
`pair_count=248`, although `RollingGromm2Config.window_transitions` is 252 and
the loader requires 252. `PublicHedgingPanel.prior_one_day_origins` correctly
rejects 2023-01-10 as having only 248 prior exact pairs. 2023-01-17 is the first
eligible date: its exact window contains 252 pairs supported by 257 unique
states from 2022-01-03 through 2023-01-17.

The current worktree's exact-pair selection is directionally correct, but the
complete-artifact loader still requires exactly 253 affine-history dates
(`models/gromm/rolling_gromm2.py:344`--`models/gromm/rolling_gromm2.py:350`).
That rejects the real 257-state exact-pair history. The loader must reconcile
the affine dates to the manifest's exact origin/endpoint union, not to
`window_transitions + 1`. The focused synthetic test passes because it does not
exercise a complete artifact with gaps.

### 3. The 22.3% is a real upper-bound failure for the current target

The current estimator first fits 8,800 independent coordinatewise AR(1)
conditional drifts (`quantdata/gromm2/estimation.py:334`--
`quantdata/gromm2/estimation.py:348`). For each date it then projects that
cross-section minus the analytical Q drift onto the 16 density-localized
loading columns (`quantdata/gromm2/estimation.py:576`--
`quantdata/gromm2/estimation.py:607`).

Independent weighted least-squares bounds on the same float64 Q drift give:

| rank-16 projection | explained target energy |
| --- | ---: |
| implemented ridge, fixed PIT factor zero | 22.2991% |
| zero ridge, factor zero fixed | 22.2992% |
| zero ridge, all 16 factors free date by date | 23.1416% |

The last row is an optimistic in-sample oracle: it gives every date its own
unconstrained 16-vector. It still cannot approach 50%. Therefore no precision
change, ridge adjustment, or factor-zero adjustment can make the present
rank-16 span pass the current AR-target gate. This is genuine unattainability
of the *estimated unscreened coordinatewise-AR drift*, not a solver failure.

The saved statewise oracle market prices also are not the executable predictor.
They are subsequently regressed on level and lagged-score features
(`quantdata/gromm2/estimation.py:393`--`quantdata/gromm2/estimation.py:446`).
On the failed artifact, the fitted predictor's in-window maximum daily norm is
7.597 and its as-of predicted daily norm is about 3.01. Thus neither the oracle
nor the actual state predictor provides a comfortably accepted P change.

### 4. The physical estimator is not the validated compatible-P construction

There are three material semantic differences.

1. **Front-factor identification.** GROMM2 maps the observed return through the
   marginal convolved CDF and uses the mean PIT score
   (`quantdata/gromm2/estimation.py:79`--`quantdata/gromm2/estimation.py:97` and
   `quantdata/gromm2/estimation.py:517`--`quantdata/gromm2/estimation.py:519`).
   The accepted construction maximizes the exact discrete DLV-front times
   independent 5% lognormal likelihood and uses posterior latent front
   innovations. On these same 248 pairs, the PIT estimate is `-0.58463`; the
   exact-mixture estimate is `-1.63919` with standard error `1.06723`.

2. **Drift target and selection.** GROMM2 projects an unscreened coordinatewise
   AR drift separately at every historical state. The accepted Phase-4 method
   fits one compatible risk-price vector to realized fixed-contract
   transitions, chooses ridge on a blocked trailing holdout with the
   one-standard-error rule, and requires signed holdout gain, linked-chain HAC
   significance, and the daily-norm gate before refitting the full window
   (`data/rolling_calibration.py:767`--`data/rolling_calibration.py:945`).

3. **Gate meaning.** GROMM2's projection fraction measures the fit of oracle
   date-specific risk prices before the feature regression. It neither measures
   the final predictor's physical drift nor supplies an out-of-sample test. It
   is therefore not interchangeable with the validated compatible-P
   acceptance statistics.

Replacing only the front identification confirms that this mismatch is not a
small numerical correction. Re-estimating the loading basis with the exact
mixture factor zero and fixing its exact risk price raises the no-ridge
statewise projection only to 22.7893%, while increasing the maximum daily norm
to 13.251.

## Required remediation

1. Start production rolling fits at 2023-01-17, not 2023-01-10, and construct
   exactly 252 released one-session pairs. Store and validate their explicit
   origin and endpoint dates; allow the supporting state union to exceed 253.
2. Require JAX x64 inside the estimation API as well as in the detached CLI.
   Regenerate the dynamics artifact from the corrected window; do not reuse the
   float32 artifact.
3. Port the accepted compatible-P estimator to the 45-by-200 GROMM2 geometry:
   exact DLV-front/background likelihood, filtered latent factor-zero
   innovations, analytical x64 Q drift, blocked train/holdout ridge selection,
   one-standard-error choice, linked-chain HAC gain statistic, and full-window
   refit only after selection.
4. Record both the precommitted projection fraction and the accepted Phase-4
   validation diagnostics. Apply the 50% projection and daily-norm limits
   unchanged. Gate the norm on the executable predicted risk price over the
   training states and current as-of state, not only on intermediate oracle
   targets.
5. If the corrected 2023-01-17 rank-16 fit still explains less than 50% or
   exceeds the norm limit, expose Q only. Changing factor rank or constructing
   a drift-aware loading basis is a new precommitted model specification, not a
   permissible repair to this run.

## Verification performed

- Independent float64 public-data recomputation of the current constant fit.
- Weighted no-ridge/free-factor span bounds and risk-price norms.
- Exact front-mixture likelihood and filtered-factor re-estimation on the same
  pairs.
- Public-panel exact-pair count audit for 2023-01-10 through 2023-01-17.
- Focused CPU tests: 5 passed (`test_estimation.py` and
  `test_gromm2_rolling.py`).
- No production implementation file was modified by this review.

## Final Q-only fallback re-review

**PASS for Q-only use; P remains correctly rejected.** The loader now derives
the admissible density-mass residual from the serialized affine-HDF5 precision,
requires the residual to remain within eight source epsilons, and then
renormalizes in float64. This is a bounded representation-roundoff correction,
not a relaxation of the probability constraint: the regression accepts the
float32 decode residual and rejects a 0.1% mass error.

An independent authenticated replay of the 2023-01-17 artifact found a maximum
pre-normalization mass error of `1.96401095e-8` against a float32-derived bound
of `9.53674316e-7`; normalized row mass was accurate to `2.22044605e-16`. The
artifact rejected P generation because `physical_accepted` is false, while four
Q paths over 163 instruments were finite. The focused daily suite passed all
five tests.

The Q-only outcome is the correct gate result. The market calibration and
rank-16 Q diffusion are usable independently of the failed compatible-P fit,
whereas silently substituting Q dynamics for P would be incorrect. There are no
remaining blockers to Q-only daily replay for this artifact.
