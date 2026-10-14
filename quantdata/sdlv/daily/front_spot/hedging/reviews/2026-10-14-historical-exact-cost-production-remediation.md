# Historical SANOS Exact-Cost Production Remediation

## Verdict

**PASS.** The entropy and power-exponential daily solvers retain their original
objectives, costs, bounds, and `1e-8` KKT tolerance. A line-search status is no
longer allowed to override an independently satisfied original-action KKT
certificate, and a genuinely stalled positive/negative split solve receives
one canonical restart from its recovered signed action.

## Production defect

The immutable `edf28ccdb52ccac1788e20a51ce226220a5f74f3` production cohort
reported numerical-gate failures in all six exact-cost cases: entropy and
power-exponential at risk parameters `0.1`, `1`, and `10`. The first entropy
failure occurred at `2023-01-23` for `digital-call-1d-nms+1.00`. Its final
original-action KKT residual was `7.89814556882064e-9`, already below the
declared `1e-8` tolerance, but SciPy returned a non-success line-search status.
The implementation incorrectly required both certificates.

One full-book preliminary selection fit exposed a second manifestation. Its
positive/negative split stalled at KKT residual `1.561325913141663e-6`; a
single restart from the canonical signed action reached
`8.780999949325755e-9` without changing the statistical problem or tolerance.

## Correction

- For entropy and power-exponential, convergence is certified by the
  independently recomputed KKT residual of the original signed-action problem.
- SciPy's line-search status remains diagnostic metadata rather than an
  additional convergence condition.
- A solve that still fails the KKT gate is restarted once from its canonical
  signed action. The same data, objective, costs, box bounds, and tolerance are
  reused, and iterations from both attempts are reported.
- Executable and internal solver cap hits remain separate hard production
  gates.

## Evidence

- The focused Historical SANOS module passes 17 tests, including forced
  line-search failure and forced canonical restart for both exact-cost
  objectives.
- A real 20-target entropy replay at risk parameter `0.1` completed with zero
  failures and zero cap hits. Its maximum claim-policy KKT residual was
  `9.877515358911119e-9`; the zero-policy residual was
  `7.449043132254295e-9`.
- A real 20-target power-exponential replay at risk parameter `0.1` completed
  with zero failures and zero cap hits. Its maximum claim-policy KKT residual
  was `9.381458099814935e-9`; the zero-policy residual was
  `8.664376411663907e-9`.
- The diagnostic replay JSON SHA-256 values were respectively
  `543891fd249f42d8745550dacdcfaa0d633fff2e2141a4e8f9528fd6d597149b`
  and
  `229b26f8800ebc430a9860544431a01fc94c3fd7220cc168641b4382c2dbaf5e`.
- Independent quant review found the KKT certificate necessary and sufficient
  for the convex exact-cost problems, confirmed correct bound handling and
  cap gating, and found no correctness blocker.

The failed `edf28cc` records remain failed diagnostics. They must not be
relabelled; the six cases require fresh commit-pinned production runs under the
corrected source.
