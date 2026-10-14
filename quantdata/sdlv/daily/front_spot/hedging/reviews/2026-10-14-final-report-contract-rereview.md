# Final report contract re-review

## Scope

This review covers the remediation of the final hedging-report trust boundary,
daily and weekly statistical summaries, and rendered benchmark appendices. It
does not certify production results that have not yet completed.

## Findings

Independent quantitative re-review passed the Python remediation:

- daily completed-run results bind the exact aggregation configuration,
  exposure definition, thresholds, and immutable dated-result slice hashes;
- the final report accepts only the canonical version-2 matrix commitment and
  authenticates its dataset, daily matrix, weekly matrix, optional failure
  ledgers, fixed reporting commit, and commitment SHA-256 both before
  aggregation and after rendering;
- daily summaries retain the complete overall, year, volatility-regime, and
  year-by-regime distributions together with prices, costs, turnover,
  exposure, sparsity, cap hits, and nonconvergence diagnostics;
- daily model comparisons use equal-weighted 20-target books with five-lag
  Bartlett HAC inference and a 2,000-sample moving-block bootstrap;
- weekly model comparisons retain equal-weighted 13-target books with
  Hansen--Hodrick and moving-block-bootstrap inference; and
- the generated LaTeX contains complete multipage daily and weekly objective,
  target, pricing, cost, and paired-comparison tables rather than headline-only
  excerpts.

The focused re-review ran 31 tests together with syntax compilation, Ruff
import checks, and `git diff --check`; all passed. A real daily-plus-weekly
fragment compile also passed without overfull-box warnings.

## Operational gate

The source remediation is approved, but the installed coordinator at the time
of review is not. Its launcher still writes the superseded version-1 matrix
commitment and recomputes matrix hashes after outcomes exist. Before any final
production case may start, it must be replaced by a launcher pinned to the
remediation commit which:

1. writes the version-2 commitment before generated or GROMM production;
2. records the public dataset path and manifest hash, both canonical matrix
   paths and hashes, and any failure-ledger paths and hashes;
3. stores the commitment SHA-256 once at creation and never recomputes or
   replaces it from outcome-dependent state;
4. validates the commitment immediately before report generation; and
5. passes the commitment path, fixed commitment SHA-256, and fixed reporting
   commit to the final report builder.

The final benchmark remains incomplete until the 57 daily and 94 weekly cells
have authenticated terminal outcomes, the production PDF is regenerated from
their precommitted matrices, the complete suite passes on the reporting commit,
and a separate quantitative results review reconciles the finished numbers.
