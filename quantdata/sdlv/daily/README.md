# Daily stochastic DLV

This directory contains the retained daily 25-expiry by 200-strike stochastic
DLV calibration workspace.

## Contents

- `train.py`: end-to-end JAX dataset preparation, fitting, and audit runner.
- `data/`: immutable 891-date market calibration tensor.
- `artifacts/`: initial 300-epoch and continued 1000-epoch fits for the full,
  affine-100, affine-200, and affine-400 state representations.
- `artifacts.py`: stable local paths and reconstruction helpers.

The historical reports retain the absolute paths used when the runs were
created. Use `quantdata.sdlv.daily.DATASET_PATH` and
`quantdata.sdlv.daily.FITS` for the relocated paths.

Run the relocated CLI with `python -m quantdata.sdlv.daily.train`. The former
`python -m quantdata.dysanos.flex.run_daily_25_jax_end_to_end` command remains
available as a compatibility entry point.

```python
from quantdata.sdlv.daily import load_raw_grid

raw_grid = load_raw_grid("affine-200", date_index=0)
assert raw_grid.shape == (25, 200)
```

The continued fits contain 891 date codes and use their corresponding final
1000-epoch checkpoints. The `full` fit stores all raw grids directly; affine
fits reconstruct them as `mean + codes @ loadings`.

## Historical VIX comparison

`vix_comparison.py` values the retained paper checkpoint without retraining.
It compares official Cboe VIX closes (see `quantdata/vixdata/README.md`) with:

- The exact continuous-strike log-contract integral of the SDLV Black
  mixture, including its independent 5% background volatility. The model's
  business-day grid is mapped to actual NYSE session dates, total variance
  is interpolated to 30 calendar days, and annualization uses ACT/365.
- The Cboe discrete-strike formula on actual, **unfiltered** OptionMetrics
  Friday-expiry books, using the parity-implied forward, K0, the two-zero-bid
  cutoff, strike-spacing weights, and AM/PM settlement minutes. Model prices
  and market midpoints are evaluated on the same selected strip. No NMS
  restriction or volume/open-interest screen is applied to that strip.

The latter implements the formula but cannot reproduce the exact published
index tick: cached OptionMetrics EOD quotes and zero curves are not Cboe's
timestamped constituent feed and Treasury spline. The assumed index timestamp
is 16:15 New York; AM settlement is 09:30 and PM settlement is 16:00. SDLV's
existing AM business-clock adjustment remains 0.8 business days. Dates whose
bracketing Friday exceeds the trained 25-business-day horizon have no discrete
model result: there is no silent extrapolation. The continuous series remains
available. These are contemporaneous historical surface diagnostics, not
out-of-sample forecasts or a new dynamic no-arbitrage test.

Example (use a new external output directory each time):

```bash
JAX_PLATFORMS=cpu python -m quantdata.sdlv.daily.vix_comparison \
  --output-directory /path/to/external/sdlv/vix-comparison/unique-run
```

Outputs include aligned CSV/Parquet series, JSON metrics/provenance, selected
contracts and per-expiry audits, and PNG/PDF figures. An independent numerical
OTM option integral checks the analytic log-contract result. NAS inputs are
read through `quantdata.env.locations`; missing option books are reported,
not downloaded. Formula tests are in `tests/quantdata/vixdata/test_formula.py`.

The same evaluator accepts a completed `Gromm-v1-4,3` training run using
`--training-run-directory RUN_DIRECTORY`. It verifies the completion report,
checkpoint architecture, and dataset SHA before loading the new geometry.
For a detached follow-on job, `--wait-for-training-unit UNIT` waits for this
project's full-fit service; it does not interrupt training or evaluate an
intermediate checkpoint. Use `JAX_PLATFORMS=cpu` for these evaluations.

`quantdata.dysanos.filtered_vix` applies the shared discrete formula to
the cached market spreads supplied to SANOS, not its fitted prices:

```bash
python -m quantdata.dysanos.filtered_vix \
  --baseline-directory RAW_MARKET_VIX_RUN --output-directory NEW_FILTERED_RUN
python -m quantdata.sdlv.daily.plot_vix \
  --model-directory MODEL_VIX_RUN --filtered-directory NEW_FILTERED_RUN \
  --output-directory NEW_FOUR_SERIES_PLOT
```

The SANOS input has matched call/put bid-ask books, minimum bid `0.011`,
expiry quality, spread/implied-volatility, volume and open-interest filters;
it is not the full raw VIX strip. The synthetic call-equivalent spread uses
both call and put books. Original raw-market two-zero-bid stopping strikes
are preserved, then the filtered strip's `F`, `K0`, and strike spacings are
recomputed. Results also include original real-book prices at those input
strikes and the additional cached SANOS model mask. No missing tails are
invented. Conditional bid/ask sensitivities hold midpoint selection fixed;
they are not replicas of the official bid/ask VIX indices.

The four-series figure uses official VIX, filtered-input data VIX, discrete
GROMM VIX, and the analytic full-strike GROMM variance swap evaluated at the
**same two expiry dates** and interpolated with the same calendar weights.
Fit statistics use only common complete dates. Pass
`--filtered-data-directory NEW_FILTERED_RUN` to `vix_comparison` to produce
that figure automatically after its model evaluation.

For a provisional comparison while training continues, copy the atomically
saved `fit/training-state.npz` to an immutable snapshot in a new external run
directory, then use `python -m quantdata.sdlv.daily.interim_vix` with
`--checkpoint-state SNAPSHOT --dataset MARKET_NPZ --baseline-directory RAW_VIX_RUN
--filtered-directory FILTERED_VIX_RUN --output-directory NEW_OUTPUT` and
`JAX_PLATFORMS=cpu`. This independently validates the snapshot in float64 and
reprices the saved, verified raw-market constituent strips. Its reports and
plots explicitly identify the interim epoch; they neither replace a final
training report nor interrupt training.

## Gromm-v1-4,3 wider-grid fit

`wide_fit.py` retains the paper's GROMM v1/SDLV architecture: 25 daily
expiries, 100 one-day plus 100 25-day NMS nodes, direct per-date DLV parameters,
and the independent 5% Black smoothing. Both node templates use `(-4, 3)`.
All quotes passing the original upstream eligibility/quality filters are
retained without the final NMS cutoff; this is not the unfiltered raw VIX
option strip. Weights are 1 inside expiry-specific NMS `[-3.5, 2.5]` and 0.1
outside. Each date's weighted loss is divided by its sum of quote weights;
dates remain equally weighted. The original excess-plus-0.05-midpoint
half-spread loss and DLV cap/barrier are unchanged.

The exact all-quote gradient and Adam schedule match the paper's retained
reports: 300 epochs with cosine learning rate `0.02 -> 0.001`, followed by
700 epochs `0.005 -> 0.00005`, resetting Adam at continuation as the original
runner did. Continuation retains the original `1e-5` gradient stopping
tolerance. Initialization is cold, using the original ATM scales; weights
from the narrower grid are not silently transferred.

Create a unique external run directory, set the project interpreter/data
environment explicitly, and run:

```bash
python -m quantdata.sdlv.daily.wide_fit prepare --run-directory RUN_DIRECTORY
gpuctl run --gpus 1 -- python -m quantdata.sdlv.daily.wide_fit pilot --run-directory RUN_DIRECTORY
gpuctl run --gpus 1 -- python -m quantdata.sdlv.daily.wide_fit fit --run-directory RUN_DIRECTORY
```

Preparation reads the same 891 cached dates without downloads or fresh LP
fits and checks that ATM initializers match the retained paper tensor. The
four-date 100-epoch pilot must complete before the full fit. Outputs include
`fit/Gromm-v1-4,3.npz`, restartable optimizer state, input hashes, histories,
core/tail fit statistics, and independent float64 density checks. The runner
holds a file lock, logs a heartbeat every minute, checkpoints on graceful
stop, and monitors host resource health. Rerunning `fit` resumes the identical
dataset/code/schedule. Calibration stability does not establish dynamic Q
no-arbitrage; a new dynamics audit is separate.

For a reviewed runner-only compatibility repair, `--resume-code-sha256 OLD_SHA`
accepts that exact prior `wide_fit.py` hash while still requiring identical
dataset, training kernel, model and schedule. The report records the original
checkpoint hash, identity and resumed epoch. A completed pilot can therefore
rerun final validation without repeating its optimizer steps. Float64 checks
use `jax.enable_x64()` and verify the actual reconstruction dtype.
The original float32 finite-difference point-mass initializer may have a few
ulps of signed boundary roundoff. A checkpoint exceeding the ordinary
negativity tolerance is accepted only if its total negative mass remains
bounded by that initializer error and an independent float64 reconstruction
passes the strict positivity, mass and forward checks. Such events are logged;
the training kernel, optimizer and objective remain unchanged.

### Fresh wider-grid Q diagnostics

`wide_audit.py` requires the completed wider-fit report and matching dataset
hash. It freezes that checkpoint in a new external audit directory, estimates
the historical diffusion afresh from the wider fitted surfaces (2022--2024
one-session pairs, rank 8, the paper's expiry/strike smoothing), and runs the
unchanged eight-step front-spot Q simulator. No narrow-grid weights, diffusion
factors, simulated populations or fitted policies are imported.

The default is 100,000 fresh paths / 50,000 independent antithetic pairs,
20 business days, spot plus 44 fixed-strike options. It reports the 570
coordinate-wise unconditional tests and the squared and entropy conditional
tests, using the original two-fold affine-plus-deep optimizers. Whole paths
and complete pairs stay in one fold; standard errors use pairs. This is the
restricted-universe test, not a test of every new wing strike. The new random
seed is 20261024. Population chunks and policy checkpoints support recovery
within this new run; they never refer to a previous model's results.

```bash
gpuctl run --gpus 1 -- /home/hans/vqdata/bin/python -u -m quantdata.sdlv.daily.wide_audit \
  --run-directory RUN_DIRECTORY --output UNIQUE_EXTERNAL_AUDIT_DIRECTORY --device cuda
```

Use `queue_wide_audit.py --run-directory ... --output ... --fit-unit EXACT_UNIT`
to wait for the fresh full fit and a free managed GPU without interrupting
training. Both jobs require the explicit project environment. The audit emits
`identity.json`, `calibration.json`, `diffusion.npz`, `unconditional.json`,
both conditional reports and pair scores, and finally `q-audit.json`. Its
heartbeat and host-resource checks run every minute. An explicitly supplied
`--snapshot` is labelled interim and is not a completed-fit audit.

For the **rolling 252-date** model, add
`--rolling-calibration COMPLETED_ROLLING_DATE_DIRECTORY`. This validates the
dataset, fitted weights, reference date, strictly past window, numerical source
hashes and reconstructed diffusion against the saved rolling audit, then uses
its exact saved initial state and diffusion. It does not silently use the
default 2022--2024 estimator. A fresh population retains the features required
by both conditional tests; previous unconditional moment files cannot provide
those features. Use a new external output directory and explicit seed, e.g.:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/hans/vqdata/bin/python -u \
  -m quantdata.sdlv.daily.wide_audit \
  --run-directory TRAINING_RUN --rolling-calibration COMPLETED_ROLLING_DATE_DIRECTORY \
  --output UNIQUE_EXTERNAL_AUDIT_DIRECTORY --device cpu --paths 100000 --seed 910261026
```

This runs unconditional plus squared and entropy conditional diagnostics on
the same fresh 100,000 paths, with complete-pair two-fold separation. "Full"
here means all three diagnostics, still in the paper's restricted spot-plus-44
universe; it does not mean an all-strike-grid audit.

### Standalone main21 addendum

`quantdata/sdlv/tex/main21.tex` is a standalone wider-grid addendum, not a
copy of the full paper. It contains only the new comparison and its
implementation appendix, in `sec_wide_grid.tex` and
`sec_wide_grid_details.tex`. `main2.tex` and its existing results are
retained unchanged.

Regenerate the deliberately retained tables, historical VIX figure, source
series and provenance in `tex/generated/wide_grid/` using the final reports:

```bash
JAX_PLATFORMS=cpu /home/hans/vqdata/bin/python -m quantdata.sdlv.daily.paper_wide_grid \
  --training-run TRAINING_RUN --audit-directory Q_AUDIT_DIRECTORY \
  --baseline-vix ORIGINAL_V1_VIX_DIRECTORY --filtered-vix FILTERED_VIX_DIRECTORY \
  --output-directory UNIQUE_EXTERNAL_ARTIFACT_DIRECTORY \
  --publish-directory quantdata/sdlv/tex/generated/wide_grid
```

The generator requires matching final-checkpoint hashes across VIX and Q
results and verifies identical market constituents, expiry clocks and official
VIX observations for both model variants. It recomputes comparison statistics
on the common 696 dates. Exact input paths and hashes are retained in
`generated/wide_grid/metrics.json`; no training or simulation is rerun.

From `quantdata/sdlv/tex`, build into a unique external directory with
`TEXMFOUTPUT=EXTERNAL_BUILD latexmk -pdf -outdir=EXTERNAL_BUILD main21.tex`.
The addendum does not require the main paper's bibliography or auxiliary
files. Only the intended final PDF and
figure/table artifacts belong in the Git tree, not auxiliary build files.

The wider-grid mean-IV and PCA illustrations are a separate CPU-only pass:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hans/vqdata/bin/python -m quantdata.sdlv.daily.paper_wide_factors \
  --training-run TRAINING_RUN --audit-directory Q_AUDIT_DIRECTORY \
  --output-directory UNIQUE_EXTERNAL_FACTOR_DIRECTORY \
  --publish-directory quantdata/sdlv/tex/generated/wide_grid
```

This loads the completed audit's `diffusion.npz` (no refit), checks its
reference state and explained variance against the final fit/audit, and
computes the equal-date arithmetic mean of model Black IV across all 891
dates on each day's market-ATM-scaled NMS grid. It also plots the reference
state, saved variance spectrum, and central one-day shocks for all eight
retained factors, holding spot and the front-expiry state fixed. The new
appendix is `sec_wide_grid_factors.tex`; `factor_surfaces.npz` and
`factor_metrics.json` retain the plotted arrays and source hashes.

### Rolling 252-date unconditional Q audit

`rolling_unconditional.py` uses the last **252 observed fitted trading dates
strictly before** each initial date. Only exact one-session pairs whose two
endpoints lie inside that window enter the fresh AR/PCA diffusion estimate;
missing sessions are not bridged. The reference state remains that date's
contemporaneous final wide-grid fit. For 29 August 2025 the window is
20 August 2024--28 August 2025. There are 639 eligible starting dates,
10 January 2023--29 August 2025.

The coordinator first runs a 512-path numerical preflight on 29 August, then
the 100,000-path unconditional audit on that date and every other eligible
date. Proceeding after preflight is based on numerical validity, not a
statistical pass. Each full date uses 50,000 independent antithetic pairs,
20 simulated business days, spot plus 44 fixed options, eight factors and
eight substeps. There is **no conditional-policy training** in this run.

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hans/vqdata/bin/python -u -m quantdata.sdlv.daily.rolling_unconditional \
  --training-run TRAINING_RUN --output UNIQUE_EXTERNAL_ROLLING_DIRECTORY
```

Use the explicit project working directory, `QUANTDATA_ROOT`, `PYTHONPATH`,
project `TMPDIR` and persistent log for a detached service. The coordinator
claims one managed GPU per date and releases it between dates. It waits if
no GPU is available; its child checks resources and logs progress every
minute. Previous complete pair-moment chunks are restart points. Completed
date results are never silently rerun or overwritten with a different plan.

Each date retains `calibration.json`, `diffusion.npz`, `identity.json`,
`pair-moments/*.npz`, `coordinate-statistics.npz` (means, pair-based standard
errors and z scores for every live simulated-day/instrument), and
`unconditional.json`. Root `plan.json` and `summary.json` record intended
and completed dates. Moment aggregation is tested against the original
paper's full-population calculation. Significance thresholds are
coordinatewise diagnostics, not family-wise tests across all 639 dates.

To run only one date on CPU, use `--one-date --date 2025-08-29 --device cpu`
with `JAX_PLATFORMS=cpu` and `JAX_ENABLE_X64=1`. This uses the same vectorized,
JIT-compiled JAX simulation and tridiagonal solves; there is no Python loop
over individual paths. CPU mode never starts the all-date coordinator or
claims a GPU.

To transfer a stopped GPU audit to CPU without discarding complete chunks,
also pass `--resume-from ORIGINAL_DATE_DIRECTORY`, retain its seed, path
count, chunk size and batch size, and use a **new sibling output directory**.
The driver checks data, geometry, seeds and numerical-source hashes, verifies
the reconstructed model, then uses the exact saved initial state and diffusion.
It copies only contiguous complete antithetic-pair moment chunks, recording
their hashes and CPU/GPU provenance in `continuation.json`. The original run
is preserved. Mixed-backend results retain the original precision conventions;
the backend change does not promise bitwise-identical CPU/GPU trajectories.
