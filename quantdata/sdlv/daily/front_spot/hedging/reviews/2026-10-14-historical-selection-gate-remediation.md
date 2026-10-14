# Historical SANOS Selection-Gate Remediation

## Verdict

**PASS.** Daily Historical-SANOS policy acceptance now depends on the final
252-observation refit. The 80% selection fit remains a warm start and held-out
validation diagnostic, but its numerical status no longer rejects a certified
full-window policy.

`HistoricalSanosConfig` is version `1.11`. The version change alters both the
model configuration identity and `DailyEvaluationSpec.run_identity`, so records
created under the superseded `1.10` gate cannot be silently reused or mixed.
Fresh production must use a new commit-pinned output directory.

## Production evidence

The corrected exact-cost production sweep exposed three distinct cases:

- Entropy risk parameter `1`, origin `2023-06-13`: the held-out selection fit
  for `digital-call-3d-nms+1.00` stalled, while the final 252-row KKT residual
  was `9.951046032855127e-9`, below the declared `1e-8` tolerance, with no final
  cap hit. This is a valid final policy and motivated the gate correction.
- Entropy risk parameter `0.1`, origin `2023-07-20`: every target retains two
  final internal cap hits, one from the shared zero policy and one from its
  claim policy. The executable claim-minus-zero hedges do not clip, but the
  virtual policies are genuinely boundary-limited and remain failed.
- Entropy risk parameter `10`, origin `2023-06-13`: selection-only failures no
  longer reject 19 certified final policies. `digital-put-3d-nms-0.75` remains
  failed: its final claim KKT residual is `1.1931077479825048e-4` and two
  adjacent one-day put positions clip at the executable `1,000`-unit cap.

The numerical tolerance, objective, transaction costs, internal cap, and
executable cap are unchanged. Selection convergence, selection KKT residuals,
and selection cap hits are now persisted separately from the final fit
diagnostics.

## Verification

- The focused Historical-SANOS module passes 18 tests, including a regression
  in which both selection attempts fail but the final full-window KKT
  certificate passes.
- Python compilation, Ruff import checks, and `git diff --check` pass.
- Independent quantitative review confirms the behavioral gate and the `1.11`
  provenance change, while requiring the genuine risk-`0.1` and risk-`10` cap
  events to remain failures.
