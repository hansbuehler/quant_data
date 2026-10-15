# Historical-SANOS v1.20 diagnostic-semantics review

Date: 2026-10-16

Reviewed remediation: `HistoricalSanosConfig` v1.20 in the shared worktree

## Confirmed behavior

The model-layer change correctly separates a certified boundary optimum from a
numerical failure:

- `_minimize_entropy_cost` and `_minimize_power_cost` now report success from
  the unchanged independent numerical certificate rather than from cap usage;
- `HistoricalWeeklyOpenLoopFit.converged` no longer becomes false merely because
  a certified policy reaches the internal solver cap;
- `HistoricalWeeklyHedgeFit.converged` likewise no longer folds cap counts into
  numerical convergence;
- cap counts remain explicit and the strict weekly gate still rejects them when
  `require_no_cap_hits` is true;
- with `--allow-cap-hits`, cap-only runs may be recorded, while independent
  nonconvergence and recession-direction flags remain fatal.

No objective, cap, transaction cost, solver tolerance, or certificate condition
is weakened.

## Failure-artifact semantics

The remediation passes `require_no_cap_hits` into the numerical-failure
payload and persists it as canonical gate policy.  The payload continues to
record raw `fit_cap_hits`, `executed_cap_hits`, and reconciled `cap_hits`, but
cap classifications are derived only when the cap gate is enabled.  Therefore:

- allowed cap hits combined with nonconvergence classify solely as
  `nonconverged-fit`;
- allowed cap hits combined with a recession direction classify first as
  `mean-volatility-recession-direction`, with `nonconverged-fit` also retained
  when independently present;
- strict mode continues to emit `position-cap-hit` and/or
  `policy-fit-cap-hit` from the same raw counts;
- artifacts without the new policy field retain backwards-compatible strict
  cap handling when loaded.

The runner applies the same policy to runtime enforcement and artifact
construction.  The regression tests load both allowed-cap failure artifacts,
confirm their classifications and triggered gates, and confirm that the raw cap
counts remain present.  No diagnostic evidence is suppressed.

## Verification

The focused model, weekly-runner, and production-failure suite

`python -m pytest -q tests/quantdata/sdlv/daily/hedging/models/test_historical_sanos_weekly.py tests/quantdata/sdlv/daily/hedging/test_weekly_runner.py tests/quantdata/sdlv/daily/hedging/test_production_failures.py`

completed with **66 passed**.  An independent daily/weekly runner and
production-failure compatibility suite completed with **22 passed**:

`python -m pytest -q tests/quantdata/sdlv/daily/hedging/test_daily_runner.py tests/quantdata/sdlv/daily/hedging/test_weekly_runner.py tests/quantdata/sdlv/daily/hedging/test_production_failures.py`

`git diff --check` also passed for the reviewed source and tests.

## Verdict

**PASS.**  Allowed cap hits remain observable without masquerading as rejected
gates, while genuine nonconvergence and recession directions remain fatal and
strict cap enforcement remains unchanged.

Independent reviewer: QuantAgent (`historical_failure_quant_review`)
