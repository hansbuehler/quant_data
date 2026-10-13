# Hedging benchmark

This package implements the benchmark specified in
`quantdata/hedging/PLAN.md` on the distributable SPX public candidate.

## Data

Set `QUANTDATA_ROOT` and place the candidate at:

```text
$QUANTDATA_ROOT/datasets/hedging/spx-2022-2025-hedging-public-candidate-v1
```

The candidate contains fitted SANOS and DLV states, interpreting geometries,
model prices, normalized half-spreads, forward/discount curves, ATM curves, and
forward-adjusted spot returns. It contains no raw WRDS price, midpoint, spot,
identifier, symbol, volume, open-interest, vendor-IV, or vendor-Greek field.

This candidate is the exclusive runtime and training data source. Stored SANOS
states provide the reference interpolation for every off-grid market mark or
target price. The private exporter and legacy raw-data loaders are not model or
training dependencies.

## Current status

- The public dataset reconstructs and validates without the source data root.
- Common normalized contracts, execution accounting, one-day digital pricing,
  and pricing-versus-hedging P&L decomposition are implemented.
- The SANOS-LV candidate implements the complete 1D/3D digital family, a
  trailing 252-transition fixed-contract DLV fit, eight spot-oriented factors,
  local-vol delta/factor vegas, a sparse minimum-predictable-option-cost factor
  hedge, and a residual spot-delta hedge. Its independent Phase-2 CPU review
  passed. The resumable production runner has passed a real one-date,
  20-target public-data smoke and restart check. The first full run exposed a
  small-coefficient HiGHS equality-scaling failure on 2023-06-05; factor-row
  scaling fixes the numerical cause without relaxing the hedge tolerance, and
  the clean versioned 2023--2025 rerun completed all 611 origins and 12,220
  target records without a cap hit or nonconvergence.
- The full-universe covariance Phase-3 implementation, CPU smoke, and
  independent quant re-review passed; its 611-origin production run is in
  progress.
- The resumable daily runner also supports VolGAN and AD-Seq-Vol. One-origin
  integration runs passed for all 20 calibrated targets on CPU and a
  `gpuctl`-managed GPU respectively, with no cap hit. These abbreviated fits
  validate orchestration only; they are not benchmark results. Generated-model
  rolling fits now start from one immutable pre-evaluation checkpoint rather
  than cold-training at every origin. The source checkpoint must predate the
  rolling fit, its SHA-256 is part of every checkpoint and run identity, and
  analytical normalization rebasing preserves the source model's economic map
  before current-window updates. The initial update budget is selected using
  only 2022 origins and the one-standard-error rule.
- Rolling rank-16 SDLV calibration now uses 252 exact-session fixed-contract
  transitions, aligns consecutive factor bases without future references,
  reserves the first eight factors for Greeks, and evaluates the analytical Q
  drift. The front risk price uses the exact DLV-front/lognormal-background
  likelihood. Surface P-drift regularization is selected on a trailing holdout
  block and carries explicit uncertainty and acceptance diagnostics. The
  Phase-4 remediation review passed.
- The five-session weekly engine freezes the inception cash-strike option book,
  rebalances daily, settles expiring options without a closing spread, closes
  the first later expiry when needed, and supports frozen and restarted refit
  schedules. All 13 weekly targets settle on the inception-forward-normalized
  path represented by the origin SANOS and DLV states. Actual spot-over-origin
  is retained separately for hedge-contract replay and costs; the compounded
  rolling-forward path is retained only as a carry-basis diagnostic. Its
  common SANOS/DLV pricing, exact fit-window facade, explicit fit-price units,
  Hansen--Hodrick overlap inference, and 50-episode circular block bootstrap
  passed independent re-review. Model-specific weekly adapters and one unified
  resumable frozen/restarted runner are implemented. Production requires full
  public-member checksum validation, a CPU-preflighted 2022 target-notional
  artifact, exact five-decision records, fitted-artifact fingerprints, and a
  single writer. A real one-origin frozen SANOS-LV run and no-recompute restart
  check pass. The corresponding restarted one-origin run also passes, with its
  exact five dated refits validated on restart. Full weekly benchmark
  production remains outstanding.
- The completed-run results layer validates every record against its immutable
  run and dataset identities, recomputes eligible origins, and rejects forged
  target books or numerical gates. It reports unhedged, economic,
  pure-hedging, and pricing losses overall, by year, by 2022-fixed
  ATM-volatility regime, and by year/regime intersection, including all four
  agreed risk functionals, normalized entry turnover/exposure, cap rates, and
  self-describing HAC inference. Its independent quant review passed.
- The first eligible rolling GROMM2 artifact uses exactly 252 released
  one-session pairs ending on 2023-01-17. Its 200-strike fit places 84.90% of
  observations inside spread and its 16 factors explain 95.15% of innovation
  variance. The compatible-P fit fails the unchanged projection and holdout
  gates, so the artifact is Q-only. Bounded float32 density normalization now
  permits authenticated Q replay while preserving strict rejection of malformed
  density mass; a real four-path, 163-instrument replay passed and P generation
  remains disabled. The rolling range runner now authenticates every reused
  artifact, writes atomic per-origin range evidence, prevents concurrent range
  writers, and emits four-minute heartbeats. A fit below the fixed 80% spread
  gate is continued deterministically from its authenticated checkpoint to
  4,000 and, if required, 8,000 epochs; the attained epoch count and retry
  count are authenticated rather than weakening the gate. Its first-origin
  resume smoke passed; preparation of the full 611-origin range is in progress. At an origin
  following a missing intermediate release, the rolling fit keeps 252 exact
  one-session pairs and appends the current origin state for its geometry and
  DLV level rather than manufacturing a multi-session transition.
- See `quantdata/hedging/PLAN.md` and the stored independent reviews before
  interpreting development smokes as benchmark results.

Run the current CPU smokes with:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.covariance.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.gromm.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.volgan.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.diffusion.smoke
```

Run the weekly engine gate with:

```bash
pytest -q tests/quantdata/sdlv/daily/hedging/test_weekly_benchmark.py
```

Prepare the fixed weekly target quantities on CPU, then run either weekly
schedule with the emitted artifact hash:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.runners.prepare_weekly_inputs \
  --output "$QUANTDATA_ROOT/processing/hedging/target-notionals/weekly-preflight.json"

python -m quantdata.sdlv.daily.front_spot.hedging.runners.run_weekly \
  --model sanos-lv \
  --schedule frozen-weekly \
  --target-notional-artifact <artifact-from-preflight> \
  --target-notional-artifact-sha256 <sha256-from-preflight>

python -m quantdata.sdlv.daily.front_spot.hedging.runners.run_weekly \
  --model sanos-lv \
  --schedule restarted-weekly \
  --target-notional-artifact <artifact-from-preflight> \
  --target-notional-artifact-sha256 <sha256-from-preflight>
```

The same entry point accepts `historical-sanos`, `gromm`,
`gromm-q-dlv-vega`, `volgan`, `diffusion-ad-seq-vol`, and
`diffusion-ad-seq-vol-ft`. Generated-model
checkpoints carry the public manifest identity; rolling warm starts or
fine-tuning bases trained against another dataset are rejected.

Run or resume one CPU daily evaluation with:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.runners.run_daily \
  --model sanos-lv
```

Prepare and then fit the complete rolling GROMM2 artifact range with separate
CPU and managed-GPU stages:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.runners.fit_rolling_gromm2_range \
  --dataset "$QUANTDATA_ROOT/datasets/hedging/spx-2022-2025-hedging-public-candidate-v1" \
  --artifact-root "$QUANTDATA_ROOT/training/hedging/gromm2-rolling-v3" \
  --start 2023-01-01 --end 2025-12-31 --stage prepare

gpuctl run --gpus 1 -- \
  python -m quantdata.sdlv.daily.front_spot.hedging.runners.fit_rolling_gromm2_range \
  --dataset "$QUANTDATA_ROOT/datasets/hedging/spx-2022-2025-hedging-public-candidate-v1" \
  --artifact-root "$QUANTDATA_ROOT/training/hedging/gromm2-rolling-v3" \
  --start 2023-01-01 --end 2025-12-31 --stage fit
```

Both stages revalidate completed origins before reusing them. The fit summary
is accepted as production evidence only when all eligible origins are present
and every recorded dated-manifest hash still matches. This completion enables
Q cases; P cases additionally require every dated physical-drift gate to pass.

Train a generated-model initializer using the complete 247-date 2022 sample:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.runners.train_generated \
  --model volgan \
  --label <fixed-label> \
  --checkpoint-directory "$QUANTDATA_ROOT/training/hedging/volgan/initializer"
```

Select the rolling update budget on explicitly chosen 2022 origins before
running 2023--2025. The selection initializer must itself predate the first
validation origin:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.runners.select_generated_budget \
  --model volgan \
  --label <fixed-label> \
  --warm-start-checkpoint <selection-initializer.pt> \
  --validation-start 2022-07-01 \
  --validation-end 2022-12-30 \
  --observation-count 126 \
  --candidate-epochs <comma-separated-precommitted-grid> \
  --checkpoint-directory <selection-checkpoints> \
  --output <selection.json>
```

The production `run_daily` invocation then uses the separately trained
end-2022 initializer via `--warm-start-checkpoint` and the selected value via
`--epochs`. A fixed initializer makes origin fits independent and restartable;
no fitted 2023 state is allowed to initialize a different 2023 origin. Each
completed fit's SHA-256 is retained in all 20 origin records, while the large
optimizer checkpoint is removed under its training lock by default. Use
`--retain-fit-checkpoints` only when the full per-origin states are required;
an interrupted fit is never removed and therefore resumes exactly.

The daily runner writes an immutable run specification, one atomic record per
origin, and a completion summary. It refuses mismatched datasets, model
configurations, incomplete target books, nonconvergence, and position-cap hits
unless nonconvergence is explicitly allowed.

After a run completes, validate and summarize it with:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.runners.summarize_daily \
  --run "$QUANTDATA_ROOT/experiments/hedging/daily/<model>/<run>"
```

Aggregation requires every eligible origin to be complete. Volatility-regime
thresholds are fixed from the 2022 initialization sample separately at each
target maturity; evaluation data never select those thresholds.

Before evaluation, the runner deterministically recalibrates the fixed target
quantities from the strictly-2022 public sample. Each scaled target has one
maturity-matched ATM spot-risk unit, the exact quantities are included in the
run identity, and every result must report its pinned quantity. The production
numerical action cap is 1,000 and any cap hit fails the default gate. The
calibration is stored atomically under
`$QUANTDATA_ROOT/processing/hedging/target-notionals/`, keyed by the public
dataset manifest and complete sizing configuration, so all model runs reuse
the same audited quantities.

All monetary outputs use discounted origin-PV units per unit origin spot.

## Production matrix

Inspect the complete daily experiment matrix without launching work:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.runners.production_matrix \
  --output-root "$QUANTDATA_ROOT/experiments/hedging/daily" \
  --checkpoint-root "$QUANTDATA_ROOT/training/hedging/daily"
```

The matrix distinguishes runnable CPU cases, staged GPU commands that still
await their review gate, and blocked cases whose selected initializer, update
budget, fine-tuning base, or dated GROMM artifact is absent. It never treats a
missing dependency as an implicit default.

Supply a completed GROMM range and the fixed 2022 policy architecture with:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.runners.production_matrix \
  --output-root "$QUANTDATA_ROOT/experiments/hedging/daily" \
  --checkpoint-root "$QUANTDATA_ROOT/training/hedging/daily" \
  --gromm-market-artifact-root "$QUANTDATA_ROOT/training/hedging/gromm2-rolling-v3" \
  --gromm-architecture-selection \
    "$QUANTDATA_ROOT/training/hedging/gromm/architecture-2022-v1/result-v2.json" \
  --gromm-artifact-range-summary \
    "$QUANTDATA_ROOT/training/hedging/gromm2-rolling-v3/range-fit-2023-2025.json"
```

Build the corresponding frozen- and restarted-weekly matrix from the same
evidence and the authenticated 2022 sizing preflight with:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.runners.weekly_production_matrix \
  --output-root "$QUANTDATA_ROOT/experiments/hedging/weekly" \
  --checkpoint-root "$QUANTDATA_ROOT/training/hedging/weekly" \
  --weekly-input-preflight \
    "$QUANTDATA_ROOT/processing/hedging/target-notionals/weekly-preflight.json" \
  --gromm-market-artifact-root "$QUANTDATA_ROOT/training/hedging/gromm2-rolling-v3" \
  --gromm-architecture-selection \
    "$QUANTDATA_ROOT/training/hedging/gromm/architecture-2022-v1/result-v2.json" \
  --gromm-artifact-range-summary \
    "$QUANTDATA_ROOT/training/hedging/gromm2-rolling-v3/range-fit-2023-2025.json"
```

The matrix command also accepts authenticated generated-model inputs through
paired initializer manifests and 2022 epoch-selection artifacts:
`--volgan-initializer-manifest`, `--volgan-epoch-selection`,
`--diffusion-initializer-manifest`, `--diffusion-epoch-selection`,
`--diffusion-ft-initializer-manifest`, and
`--diffusion-ft-epoch-selection`. The matrix verifies the public-dataset
identity, pre-2023 timing, selected epoch, initializer bytes, and fine-tuning
base bytes before staging a command; the runner rechecks the pinned hashes at
launch. Missing dependencies remain explicit blocked cases rather than falling
back to implicit defaults.

Generated cases remain `waiting-gate` until the corresponding model supplies
`--volgan-production-gate`, `--diffusion-production-gate`, or
`--diffusion-ft-production-gate`. A gate is a
`hedging-generated-production-gate-v2` JSON object containing the model and
dataset-manifest identities, `status: "pass"`, the exact clean reviewed Git
commit, and workflow-specific evidence bindings. The allowed scopes distinguish
daily, frozen-weekly, and restarted-weekly runs and, within each, published
settings from rolling trained settings. Each scope binds the exact initializer
manifest, initializer checkpoint, epoch-selection artifact, selected epoch
budget, and fine-tuning base where applicable. The referenced quant-review and
GPU-readiness artifacts are structured JSON records that must independently
repeat the same model, dataset, commit, scopes, bindings, and passing status;
their bytes are pinned by SHA-256. An absent gate leaves commands staged. An
invalid, stale, out-of-scope, or tampered gate blocks only the affected cases.
Ready generated cases carry the reviewed commit in the matrix, and the queue
rechecks the exact gate bytes, clean tracked source, and exact `HEAD`
immediately before launch.

Execute an explicit ready subset sequentially with atomic queue state and
per-case logs:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.runners.execute_production_matrix \
  --matrix "$QUANTDATA_ROOT/experiments/hedging/daily/production-matrix-v1/matrix.json" \
  --state "$QUANTDATA_ROOT/experiments/hedging/daily/production-matrix-v1/cpu-queue.json" \
  --interpreter /home/hans/vqdata/bin/python \
  --resource cpu \
  --case historical-sanos-mean-variance-1
```

The executor replaces only the matrix's Python placeholder, forces CPU-only
child environments for CPU queues, checks resource headroom before every new
case, pins the exact matrix SHA-256 in queue state, resumes completed queue
records, emits four-minute heartbeats, and
accepts completion only after validating the full 12,220-record results
artifact. A matrix whose bytes change after the queue is armed is rejected.
Legacy queue state without a matrix digest is rejected unless the operator
explicitly supplies `--adopt-unpinned-state` after checking the matrix.
For a parameter sweep, `--continue-after-failure` retains an already failed
case as terminal evidence, runs the remaining selected cases, and finishes the
queue as `complete-with-failures`; it never relabels a failed numerical gate as
a result.
Production-matrix commands also pin the public dataset manifest SHA-256, which
the daily and weekly runners recheck before fitting or evaluation. Daily
production commands additionally run the full released-dataset checksum
validation so unchanged manifest bytes cannot mask changed payload files.
