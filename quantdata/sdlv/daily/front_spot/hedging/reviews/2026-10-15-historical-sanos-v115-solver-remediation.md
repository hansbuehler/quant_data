# Historical-SANOS v1.15 Solver Remediation Review

Date: 2026-10-15

## Scope

This review covers the strict Historical-SANOS daily and weekly solver failures
encountered by the version-1.13 and version-1.14 production queues. The review
separates genuine economic or execution-bound outcomes from numerical solver or
certificate failures and verifies the version-1.15 remediation without changing
the declared hedge objectives, transaction costs, action bounds, scenario
population, or acceptance tolerance.

## Failure Classification

The prior failures are not one homogeneous class.

- Daily mean-variance at risk parameter 10 was a numerical certificate failure.
- Daily mean-volatility at risk parameter 0.1 contains a genuine recession
  direction. The cases at risk parameters 1, 2, and 3 were numerical failures.
- Daily entropy and power-exponential at risk parameter 0.1 contain legitimate
  internal action-cap signals rather than solver-certificate failures.
- Frozen-weekly mean-volatility at risk parameters 0.1, 1, and 2 contains
  genuine recession directions. The weekly entropy and power-exponential
  failures were numerical.
- Restarted-weekly mean-variance at risk parameter 1 contains a legitimate
  internal action-cap signal.

These classifications remain explicit production outcomes. Version 1.15 does
not relabel a recession direction or a genuine cap hit as convergence.

## Remediation

**PASS.** Independent quantitative review found no remaining implementation or
objective-semantics blocker.

- Daily mean-variance is solved as its exact convex quadratic program, using
  OSQP first and CLARABEL as an independent fallback. The raw action is checked
  against the original signed-action KKT conditions before any numerical
  canonicalization.
- Daily mean-volatility and weekly conic objectives certify the raw solver point
  against the unchanged objective and primal constraints. A failed or
  inaccurate CLARABEL point receives one independent SCS solve rather than a
  relaxed acceptance threshold.
- A solver status of `optimal_inaccurate` is accepted only when the independently
  reconstructed exact objective is finite and agrees with the reported value,
  all original primal constraints pass, and the applicable objective-specific
  certificate passes. The complete primary and fallback trace is retained.
- Canonicalization occurs only after a raw certificate failure and is accepted
  only when its exact objective delta and independent certificate pass.
- Exact mean-volatility recession tests and both internal and executable cap
  diagnostics remain fail-closed.

## Verification

The focused Historical-SANOS suite passes 68 tests with two expected numerical
warnings. Syntax compilation, Ruff import checks, and `git diff --check` also
pass. The independent reviewer reran the four targeted raw-certificate and
fallback regressions; all four pass.

The final code-frozen real-data replays use only the distributable dataset and
the fixed 2022 target-notional artifacts:

- Daily mean-variance, risk parameter 10, origin 2023-01-18: 20 of 20 targets,
  zero cap hits, and zero nonconverged results. Evidence:
  `/tmp/quant_data/hedging/historical-remediation-v115/final-daily-mv10/output/summary.json`.
- Daily mean-volatility, risk parameter 1, origin 2023-07-13: 20 of 20 targets,
  zero cap hits, and zero nonconverged results. Evidence:
  `/tmp/quant_data/hedging/historical-remediation-v115/final-daily-meanvol1/output/summary.json`.
- Frozen-weekly entropy, risk parameter 10, origin 2023-01-30: 13 of 13 targets,
  zero fit or executed cap hits, zero nonconverged fits, and zero recession
  directions. The formerly problematic barrier fit records its independently
  certified `optimal_inaccurate` result. Evidence:
  `/tmp/quant_data/hedging/historical-remediation-v115/final-weekly-entropy10/output/summary.json`.

The three summary SHA-256 values, in the order listed above, are
`7e1b0be7a75f4b009a18c9b6daa1be3c601415226981a7b8efca82404c6077ea`,
`5b53c3683e96b70b5e51984719785e43ae285a6e9eedf93dc26f7fd562049377`,
and `03152392a0a60204dcbe893d2cf1940bd21be8b91f62007e0e06afb6bf599654`.

The reviewed source SHA-256 values are:

- `config.py`: `c0b520507987b32a6db2aa625a306d8963b784a6b9521d9a42a4c5b5356c858d`
- `hedger.py`: `71ffbe585569d19f63d0f2fa01ae1714e2ec1149b83d6183aa1bbe6a27cd665c`
- `weekly_hedger.py`: `4bec502652fd147ba5de3a523de041fddcd62ab19987567889c058a6713e218f`
- daily tests: `e545880f9f4e4afacd762775c4324d733d60374d9dbe68f6d4e52952de355b27`
- weekly tests: `4e3e42d06f93269cef008dab9ec9d482f546548cd46b6ce0eb5a4ccc1063cf77`
