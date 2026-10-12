# VolGAN and diffusion daily-integration remediation review

## Verdicts

- **VolGAN: PASS.** The remaining-maturity digital and settlement-cost blockers
  from the earlier review are fixed, and the public 2023-01-17 checkpoint replay
  passes its committed gate.
- **Diffusion: PASS.** The shared blockers are fixed, the return-channel
  dispersion is now observable, and the public checkpoint replay passes its
  committed gate.

No blocking issue remains in the reviewed daily-baseline integration.

## Remediation evidence

1. **Smile-consistent digital.** A three-day digital is now valued after one
   generated day by centered finite-difference differentiation of the same
   interpolated generated call smile used for the vanilla book
   (`models/generated_scenarios.py:197`,
   `models/generated_scenarios.py:385`). The analytic sloped-smile regression
   includes the missing smile-slope term (`test_generated_scenarios.py:104`).
   The clipping count is retained in both model results rather than silently
   hiding invalid generated probabilities (`models/volgan/adapter.py:165`,
   `models/diffusion/adapter.py:172`).

2. **Cash-settlement cost.** The predictable option-cost proxy now decides
   whether an instrument settles from the fixed contract's `expiry_date`, not
   from a floating-point DTE equality (`models/generated_scenarios.py:255`).
   The regression uses the observed public-data value
   `1.0000000577419996` (`test_generated_scenarios.py:142`).

3. **Real 3D endpoint replay.** Replaying the public 2023-01-17 to 2023-01-18
   state with the realized normalized spot and reconstructed endpoint grid gave
   generated 3D ATM call-digital PV `0.1224005926`, consistent with the earlier
   independent call-curve derivative `0.1223983545`; no probability was
   clipped. The actual SANOS digital mark is `0.1049093770`, leaving
   `0.0174912156` of fixed-grid representation error. This is no longer an
   internal pricing inconsistency: it is the disclosed approximation from
   reducing SANOS to a 9-by-11 IV grid.

4. **Real 1D cost replay.** The public origin book contains 29 contracts that
   settle on 2023-01-18, all carrying noisy DTE
   `1.0000000577419996`. Every generated cost equals one opening half-spread;
   maximum error and aggregate excess exit charge are both exactly zero.

5. **Carry and fixed contracts.** The existing one-day settlement and
   longer-maturity fixed-strike mappings remain unchanged
   (`models/generated_scenarios.py:144`). In the same real replay, generated
   versus realized option-return mean absolute error was `1.016e-4` and maximum
   absolute error was `4.484e-4`, attributable to the fixed-grid surface
   representation.

6. **Timing and semantics.** Both adapters fit and condition through the
   origin date and explicitly require their forecast date to equal the realized
   hedge endpoint (`models/volgan/adapter.py:103`,
   `models/diffusion/adapter.py:106`). Both continue to label scenarios as
   historical `P`, use the common released SANOS market price, and route
   realized gains and costs through common benchmark accounting.

## Model-specific checks

### VolGAN

The just-load public checkpoint replay passed the committed fixture: 146
training examples, 40 completed epochs, 128 scenarios, 163 instruments,
validation loss `1.1131898007`, zero digital clips, and a converged common
cost-lasso result.

### Diffusion

The just-load public checkpoint replay passed the committed fixture: 146
training sequences, three completed epochs, 64 scenarios, 163 instruments,
validation loss `1.1463043690`, zero digital clips, and a converged common
cost-lasso result. The result now reports return-channel maximum within-path
grid dispersion (`models/diffusion/hedger.py:149`); the replay value is
`0.0311154090`.

## Non-blocking findings

1. The real 3D endpoint residual (`0.01749` PV, about 16.7% of the SANOS mark)
   is economically material. It is a coarse-grid baseline limitation, not a
   carry or digital-derivative bug, and should be measured across dates before
   interpreting 3D hedge comparisons.
2. Probability clipping is transparent but does not repair static arbitrage in
   a generated smile. Production runs should summarize the clip rate and treat
   persistent or material clipping as a model-quality failure.
3. The diffusion replay's return-channel dispersion (`0.0311`) is large
   relative to its mean generated daily log return (`-0.00202`). Averaging the
   broadcast field is defined and now auditable, but future production gates
   should set a scale-aware tolerance or generate the scalar return through a
   dedicated head.
4. Both committed one-day smoke targets remain degenerate on these small
   scenario sets: no generated digital settles in the money, so the fitted
   action and transaction cost are zero. The real 3D replay validates the
   corrected valuation path, but a committed non-degenerate 3D end-to-end
   fixture would strengthen regression coverage.

## Verification

- Focused CPU suite: `24 passed` in `7.37s`.
- VolGAN public checkpoint just-load smoke: passed.
- Diffusion public checkpoint just-load smoke: passed.
- Real public 3D endpoint and 1D settlement-cost replays: passed.
- `gpuctl resources --json` could not reach its daemon and used the local
  fallback; all RAM and disk checks reported `OK`. No GPU was used.
- No implementation file was modified by this review.
