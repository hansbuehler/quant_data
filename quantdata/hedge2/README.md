# Hedge2

Rolling European and exotic hedging benchmark, developed on the
`hedging-benchmark` branch, not `main`. This is research software with
unfinished model-law and rolling-result validation, not a production release.

## Documentation

[PLAN.md](PLAN.md) is the single specification, status summary, evidence
index and code/quant review checklist. This overview contains entry points
only. Earlier duplicated progress diaries, including `REVIEW_NOTES.md`, were
consolidated on 2026-11-07; the unmodified originals and hashes are retained
outside Git in the publication-validation run identified in the plan.

## Current position — 2026-11-07

- Daily zero-backbone DLV conversion and both rolling eight-factor PCA/AR(1)
  variants are implemented. The input interpolator is saved SANOS with
  smoothness 0.1; these are not rolling DYSANOS ML fits.
- Current-front-density spot sampling and two explicit spot/surface coupling
  candidates pass unit tests and a three-date cash-repricing diagnostic.
  Neither coupling is selected for production; next-surface guard activity
  and terminal integration remain unresolved.
- No-forward Greek populations and independent historical SANOS replay are
  running on CPU. Earlier raw-historical-spot generated hedges remain
  labelled baselines, not results for the corrected front-density law.
- Full generated-model and terminal-horizon populations are incomplete.
  Test success does not establish conditional calibration or no arbitrage.
- Isolated-branch validation: **756 passed, two allocated-GPU tests skipped**;
  import sorting and syntax checks pass. The two original warnings are
  preserved. No GPU is allocated to this project at this snapshot.

## Code map

| Area | Main modules |
|---|---|
| Saved source and licensed mids | `source`, `cash`, `historical`, `observations` |
| Direct daily DLV state | `dlv`, `pricing`, `state`, `batch_pricing` |
| Rolling eight-factor dynamics | `dynamics`, `rolling`, `pca_audit` |
| Front law and candidate dependence | `front_spot`, `front_scores`, `front_joint`, `front_scenarios` |
| Claims and Black reference | `claims`, `black_hedge`, `reference_run` |
| Greek and parametric hedges | `greeks`, `greek_hedge`, `greek_budget`, `premium`, `parametric` |
| Sample-based objectives | `objectives`, `optimization`, `certification`, `dual`, `polish` |
| One-day and episode controllers | `objective_run`, `episode_run`, `episodes`, `greek_frontier` |
| Terminal simulation and policy | `terminal_market`, `terminal`, `terminal_policy`, `terminal_train`, `terminal_roll` |
| Independent replay and reporting | `audit_daily`, `audit_episode`, `audit_greek_frontier`, `report_objectives`, `report_frontiers` |
| Immutable restart | `snapshot_run`, `snapshot_validation`, `resume_objectives` |
| Optional IV-grid foundation | `iv_grid`, `iv_grid_pilot`; not a completed competitor hedge |

Each adapter has standalone small tests in `tests/quantdata/hedge2`.
Executable modules accept explicit external-data paths; use `--help` before
launching a run. The plan records compatibility and accounting caveats.

## Storage and validation

Source lives under `quantdata/hedge2`; tests under `tests/quantdata/hedge2`.
Licensed data, arrays, checkpoints, logs and disposable PDFs stay outside
Git, under `/home/hans/quantdata_data/quant_data/hedge2/runs/<unique-run>`.
They are not implicitly approved for public redistribution.

From an environment with the repository dependencies installed, run CPU
checks without claiming a GPU:

```bash
QUANTDATA_ROOT=/home/hans/quantdata_data \
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
HEDGE2_GPU_TEST=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest -q tests/quantdata/hedge2 \
  --basetemp /tmp/quant_data/hedge2-review-unique/pytest
ruff check --select I quantdata/hedge2 tests/quantdata/hedge2
```

Use a fresh project-owned temporary directory for each execution. The exact
interpreter, CPU runtime overlay, source hashes, warning text and command
environment of the latest verified run are retained in its evidence folder.
