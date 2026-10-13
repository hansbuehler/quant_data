# Weekly production milestone quant review

Date: 2026-10-13
Re-review: 2026-10-13, after remediation of the five initial blockers

## Verdict

**PASS.** The five production blockers from the initial review are resolved.
The weekly runner now authenticates complete five-decision episodes, numerical
environment, model artifacts, public-data provenance, the CPU sizing handoff,
and exclusive writer ownership. No source code was changed in this review.

## Scope

- `benchmark/weekly_sizing.py` and `benchmark/weekly_runner.py`
- `runners/weekly.py`, `runners/prepare_weekly_inputs.py`, and
  `runners/run_weekly.py`
- weekly adapters and checkpoint logic for GROMM, VolGAN, and diffusion
- SANOS replay pricing in `data/market_panel.py`
- associated contract, panel, sizing, runner, pricing, adapter, and generated
  model tests

## Resolved blockers

### 1. Exact five-decision and terminal reconciliation

`WeeklyModelRun` now requires exactly five aligned decisions, five hedge-gain
steps, six transaction-cost entries including unwind, and reconciliation of
step totals with the spot/option accounting decomposition
(`benchmark/weekly_runner.py:319`). Record validation recomputes the five
decision dates and terminal date from the public panel; frozen runs must repeat
the origin fit five times, while restarted runs must contain the five ordered
decision-date fits and five distinct fit identities (`runners/weekly.py:430`,
`runners/weekly.py:489`, `runners/weekly.py:508`). The prior one-row truncation
reproduction is now rejected with `weekly model run must contain exactly five
decisions`.

### 2. Numerical environment is part of run identity

GROMM, VolGAN, and diffusion configuration snapshots now include device,
precision, and debug state (`runners/run_weekly.py:385`,
`runners/run_weekly.py:424`, `runners/run_weekly.py:460`,
`runners/run_weekly.py:635`). The canonical configuration snapshot is embedded
in `WeeklyEvaluationSpec.to_dict`, whose hash is the run identity
(`runners/weekly.py:139`). Independent unit coverage confirms diffusion
float32 and float64 builds have different snapshots and therefore different
run identities.

### 3. Model and policy artifact fingerprints are persisted

Every decision now carries separate market-model and policy-artifact
fingerprint tuples (`benchmark/weekly_runner.py:304`); they are accumulated by
the benchmark, serialized into every origin record, and required again on read
(`benchmark/weekly_runner.py:408`, `runners/weekly.py:557`,
`runners/weekly.py:622`). GROMM records the affine-history and dynamics hashes
plus all four signed-policy checkpoint hashes
(`models/gromm/weekly_adapter.py:59`, `models/gromm/weekly_adapter.py:269`).
VolGAN and diffusion record the exact dated model-checkpoint hash
(`models/volgan/weekly_adapter.py:55`,
`models/diffusion/weekly_adapter.py:55`). Their weekly open-loop policy is not
a separate persisted checkpoint; its realized action matrix is stored directly
in the weekly record. Generated-model records cannot pass validation with a
missing market fingerprint, and GROMM cannot pass with missing policy
fingerprints (`runners/weekly.py:470`).

### 4. Public checksums and generated-checkpoint provenance are mandatory

Both CPU preflight and production construct `PublicHedgingPanel` with
`validate=True`, invoking full public-dataset validation before loading the
panel (`runners/prepare_weekly_inputs.py:77`, `runners/run_weekly.py:170`,
`data/market_panel.py:601`). The runner and generated-model adapters also
require the public panel interface rather than a private/raw-data loader
(`runners/weekly.py:301`, `models/volgan/adapter.py:75`,
`models/diffusion/adapter.py:77`).

VolGAN and diffusion pass the public manifest hash into training, include it
in checkpoint filenames and payloads, require it when resuming the current
checkpoint, and require equality when loading a rolling warm start or diffusion
base model (`models/volgan/training.py:93`,
`models/volgan/training.py:215`, `models/volgan/training.py:677`,
`models/volgan/training.py:745`, `models/diffusion/training.py:112`,
`models/diffusion/training.py:240`, `models/diffusion/training.py:675`,
`models/diffusion/training.py:746`). The reviewed weekly runtime contains no
WRDS, OptionMetrics, or private raw-data access path.

### 5. Authenticated preflight and single-writer execution are mandatory

Production now rejects a missing target-notional artifact SHA-256 before
opening the dataset, verifies that exact file hash, and includes it in the run
identity (`runners/run_weekly.py:157`, `runners/run_weekly.py:164`,
`runners/run_weekly.py:172`, `runners/run_weekly.py:203`). An independent CLI
check confirmed the missing-hash invocation exits with the intended parser
error.

The whole resumable evaluation is protected by one non-waiting writer lock on
the output directory (`runners/weekly.py:278`). An independent adversarial
check held that lock and confirmed a second writer is rejected before it can
read or modify run state.

## Quantitative gates retained

- **No look-ahead:** fixed-2022 sizing uses only episodes settling in 2022;
  every market and policy fit receives an as-of panel restricted to strictly
  prior training origins.
- **Units and normalization:** target payoff, common price, hedge gains, and
  costs remain discounted episode-origin spot-one quantities; restarted
  fit-date prices are mapped back to origin units before aggregation.
- **Complete target book:** the immutable spec requires the exact ordered
  13-target catalogue, and every computed or resumed origin is checked against
  it (`runners/weekly.py:117`, `runners/weekly.py:437`).
- **Cap gate:** the default rejects any spot or option coordinate reaching the
  precommitted position cap (`runners/weekly.py:502`).
- **SANOS replay boundary:** the shared `OptionQuote` positive-value invariant
  remains strict. Numerically negligible non-positive SANOS replay prices are
  floored to a positive value, while materially negative prices raise
  (`data/market_panel.py:404`).

## Independent verification

- Focused CPU suite: **74 passed** in 42.85 seconds.
- Adversarial single-writer check: **passed**.
- Production CLI missing-preflight-hash check: **passed**.
- Python compilation: **passed** for reviewed changed modules and tests.
- Ruff import-order check: **passed**.
- `git diff --check`: **passed**.
- Resource check: RAM and disk were `OK`; `gpuctl` used its local fallback
  because the daemon was unreachable. No GPU work was started.

## Production note

This is a code-path and invariant review, not an endorsement of numerical
results that have not yet been produced. Production result acceptance should
still require zero cap hits, complete 13-target output at every eligible
origin, and the precommitted model-specific quantitative diagnostics.
