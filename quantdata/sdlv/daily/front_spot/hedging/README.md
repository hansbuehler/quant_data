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
  the clean versioned 2023--2025 rerun is running as a resumable CPU job.
- The full-universe covariance Phase-3 implementation, CPU smoke, and
  independent quant re-review passed.
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
  passed independent re-review. Model-specific weekly policies remain.
- The completed-run results layer validates every record against its immutable
  run and dataset identities, recomputes eligible origins, and rejects forged
  target books or numerical gates. It reports unhedged, economic,
  pure-hedging, and pricing losses overall, by year, by 2022-fixed
  ATM-volatility regime, and by year/regime intersection, including all four
  agreed risk functionals, normalized entry turnover/exposure, cap rates, and
  self-describing HAC inference. Its independent quant review passed.
- The GROMM2 reconciliation is read-only. Its expanding-support architecture is
  the intended scenario direction, but the current full-sample, Q-only
  artifacts are not eligible for the rolling benchmark.
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

Run or resume one CPU daily evaluation with:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.runners.run_daily \
  --model sanos-lv
```

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
