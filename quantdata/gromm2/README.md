# Gromm v2 4,3

The current calibration uses the consolidated source market books and saved
DYSANOS smooth surface parameters. Each observed contract retains its own
strike, expiry, call/put identity, and observed half-spread. Its fitting band is
the DYSANOS fair value plus/minus that half-spread; market midpoints are not the
calibration targets. Original bid–ask coverage is reported separately.

Use the **existing trained SANOS parameter sets**. Do not regenerate, refit,
or replace them for GROMM2. Pin their exact paths with `--source-inventory`;
do not choose newly generated alternatives by scanning for the longest
surface. Read the saved `af_pricer` and evaluate it at the required market
strikes and expiries. Beyond its final saved node, use the SANOS pricer's
native rule: retain the terminal latent density and its terminal kernel
volatility. This is model-based terminal pricing, not additional observed
calibration evidence. Record the source-node horizon and use of that rule
in each GROMM fit report. A 125-day tower does not require refitting SANOS
to create new 125-day source nodes.

Only the GROMM DLV representation is fitted. The saved SANOS source
parameters remain read-only throughout calibration and simulation.
If an individual historical cache file is absent, a small JSON reference
may point to that date's already fitted row in the existing parameter
dataset. The reference records dataset-file hashes; it does not copy or
re-estimate SANOS parameters.

The tower has 200 interior strikes in NMS `(-4,3)`, two exterior boundaries,
daily expiries through 25 business days, and five-day intervals through 125
business days. The Black background volatility is fixed at 5%. Observations
outside NMS `(-3.5,2.5)` receive 10% fitting weight. Both AM and PM contracts
are retained, except the existing 0.2-business-day exclusion.

## Two stages

1. Fit each observation date independently on the grid constructed from that
   date's information. Save quote-level errors, original-market coverage,
   density invariants and coverage across dates. Check all dates before
   starting the rolling estimation.
2. For each eligible origin `t`, hold its spot-relative tower grid fixed and
   refit every available observation in the preceding 252-business-day window
   to its own market contracts and DYSANOS fair values. An interpolated own-date
   state may initialize the optimizer, but every historical state is refitted
   on the `t` grid. Align each one-session pair to fixed cash strikes and fixed
   expiries before estimating the coordinatewise AR(1) of log DLV-squared.

The date `t` is excluded from AR(1) training. Missing observations do not extend
the historical window; pairs crossing more than one business session are
excluded. Report interpolation boundary clipping and unidentified AR slopes.
Each origin has its own fitted history and AR(1); an AR(1) estimated from a
single sequence of changing own-date grids is not substituted for it.

The current run estimates only **29 August 2025**, using its preceding
252-business-day source window. Pass `--origin-date 2025-08-29` to avoid
calibrating unrelated origins. Historical states are fitted chronologically
on this origin's fixed grid, initializing from the previous fitted date's
weights. Only the first required history state starts without prior weights.
Completed estimates are retained rather than rerun solely for warm-starting.

## Retained results

The [26 October 2026 progress snapshot](results/gromm_v2_4_3_2025_08_29_progress_2026_10_26.json)
records the completed origin-date fit and historical calibration reports,
with source hashes and the external run directory. It is an incomplete-run
snapshot, not a completed AR(1) estimate or a passed no-arbitrage audit.

## Execution

Use the project interpreter, `QUANTDATA_ROOT` and `PYTHONPATH`. The one-date
float64 fit runs substantially faster on CPU than on the shared GPU. The
rolling runner uses up to four CPU workers and writes checkpoints and flushed
progress to the uniquely identified external run directory.

```bash
JAX_PLATFORMS=cpu /home/hans/vqdata/bin/python \
  -m quantdata.gromm2.rolling_calibration \
  --source-inventory /path/to/pinned/saved-sanos-inventory.json \
  --run-directory /path/to/external/unique/run \
  --origin-date 2025-08-29 --workers 4
```

`--inventory-only` checks saved parameter availability and records which
dates need native terminal pricing. It never rebuilds the source fit.
The legacy `--source-root` discovery mode retains its strict source-node
coverage check; the current task uses the pinned-parameter mode instead.

The source inventory and both calibration stages are restartable. A completed
date must have at least 90% coverage and valid density mass/forward moments to
unlock rolling AR(1). Weak fits receive continuation where the optimizer has
reached its epoch ceiling; unresolved dates remain visible in the summary and
require review. Core and outer-wing metrics remain separate in each date's
report.

## Rolling unconditional audit

For every completed rolling origin, simulate 100,000 one-business-day `Q`
transitions with eight Brownian substeps using the production front bridge
and risk-neutral DLV drift. The fitted historical innovations supply the
shared `P`/`Q` diffusion; the estimated physical AR(1) drift is not used in
the `Q` transition. The first factor is fitted jointly to front-distribution
Gaussian spot scores, with 15 residual PCA factors.

Test each fixed contract's next-day gain divided by inception spot, without
return centering or empirical measure preconditioning. The reduced book has
11 NMS levels `(-4,-3.5,-2.5,-1.5,-0.5,0,0.5,1,1.5,2.5,3)` at each five-day
expiry through 125 business days, plus the one-day front expiry. Use puts
below ATM and calls otherwise. Including spot, this gives 287 coordinates.

Streaming moments and sample standard errors treat each complete antithetic
pair as one independent observation. Report coordinate means, ordinary `z`,
pointwise 99% confidence intervals and a familywise 99% Bonferroni screen.
Separately report the economic-floor score `mean / (1e-8 + SE)`; this is not
a Gaussian `z` statistic. Density invariants and drift-taming frequency are
saved alongside the statistics.

```bash
JAX_PLATFORMS=cpu /home/hans/vqdata/bin/python \
  -m quantdata.gromm2.rolling_audit \
  --calibration-root /path/to/external/unique/run/calibration \
  --watch --origin 2025-08-29 --backend cpu --paths 100000 --substeps 8
```

This task is **CPU-only**, including path generation, pricing and audit;
do not claim a GPU. Simulation and pricing are batched, JIT-compiled JAX
operations, with compiled scans rather than Python loops over paths.
The watcher waits for the requested origin and never requests a GPU in CPU
mode. It saves chunk-level restart state and per-date statistics under
`rolling/YYYY-MM-DD/unconditional_q/`, with an aggregate JSON summary in the
calibration root. No conditional policy is trained for these daily audits.
