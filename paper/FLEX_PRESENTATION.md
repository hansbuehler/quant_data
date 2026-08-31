# SANOS ICCF presentation handover

Last updated: 2026-08-31 (Sommer WSL)

## Current status

The weekly and daily+weekly affine-200 model runs are complete. Their fit, dynamics, PCA, and invariant-distribution results have been added to the presentation.

- Revised presentation: `paper/2026 09 02 SANOS ICCF.pptx`
- Untouched pre-review backup: `paper/2026 09 02 SANOS ICCF.pre-codex-review.pptx`
- Revised deck: 50 slides, 13,489,671 bytes
- Backup deck: 34 slides, 9,560,080 bytes
- Revised deck SHA-256: `d20bdef65ba346f438f723486068ab622d4bba19b1170f540349fef613b1146e`
- Backup SHA-256: `78593adec45dac5e8d1db0a713a92d88c46ddd1d7759bd8c9e826660337ac52a`

The conditional-Q existence experiment is paused. Its checkpoints and progress data were preserved.

The task-related source, tests, configuration, documentation, and final deck
are committed together. Nothing was pushed.

## Agreed presentation brief

The presentation changes were made to show:

1. The weekly geometry extrapolates poorly before its first 5D expiry node.
2. Adding explicit 1D, 2D, 3D, and 4D nodes fixes the short end, at the cost of no longer having a purely weekly expiry grid.
3. Affine compression to a 200-dimensional state causes little deterioration in market fit.
4. Complete per-market-expiry fit results for four models:
   - weekly native;
   - weekly affine-200;
   - daily+weekly native;
   - daily+weekly affine-200.
5. The invariant-mean/base implied-volatility surface and implied-volatility PCA factors for both affine-200 models.
6. AR(1) statistics for market returns, Gaussian innovations, and historical-innovation resampling at 1W, 2W, and 3W.
7. Simulated invariant-distribution implied-volatility surfaces, including a final visual “gimmick.”
8. Clearly marked new/replacement slides, with superseded major slides hidden rather than deleted.
9. Short review remarks where they clarify terminology or interpretation.

## Model definitions and data population

Both models use 891 dates from 2022 through August 2025 and exclude the 0.2D/AM-expiry population.

- Weekly geometry expiry nodes: `5, 10, 15, ..., 70` business days.
- Daily+weekly geometry expiry nodes: `1, 2, 3, 4, 5, 10, 15, ..., 70` business days.
- Shared normalized-moneyness range: `(-3, 2)`.
- Affine state dimension: 200.
- Decoder: affine SANOS decoder (`u_t = b + W h_t`), without an additional nonlinear network.
- Total weekly population: 3,566,742 call/put quote observations.
- Total daily+weekly population: 3,566,258 call/put quote observations.

Here a quote/book observation is one call or put bid/ask observation. Calls and puts at the same expiry and strike are separate observations.

The 200 affine state coordinates should not be described as literal PCA factors. The IV-space PCA factors in the presentation are computed afterwards from decoded invariant state variation.

## Completed model artifacts

### Weekly affine-200 joint model

- Root: `/home/hans/quantdata_data/training/flex-shared-strikes/joint-direct/integer-weekly-no-am-all-history-v1`
- Checkpoint: `checkpoints/joint-direct-200.pt`
- Completion marker: `reports/joint-direct-200-complete.json`
- Checkpoint SHA-256: `f10f226bdbb514e855104ece7ab3fb3d4ac87e5b1e8b38b5b4e2ef96938419e7`
- Status: complete/converged
- Best epoch: 295
- Weekly transition spectral radius: 0.959735
- Slowest-mode half-life: 16.87 weeks

### Daily+weekly affine-200 joint model

- Root: `/home/hans/quantdata_data/training/flex-shared-strikes/joint-direct/dte1-5-weekly-no-am-all-history-v1`
- Checkpoint: `checkpoints/joint-direct-200.pt`
- Completion marker: `reports/joint-direct-200-complete.json`
- Checkpoint SHA-256: `395549f12737b8ee301c2c4b35999379f7ef55e7e2cfe0fcda6cf414068ae2ac`
- Status: complete/converged
- Best epoch: 299
- Weekly transition spectral radius: 0.953502
- Slowest-mode half-life: 14.56 weeks

## Market-fit results

Excess is measured in half-spreads:

`((fit - ask)^+ + (bid - fit)^+) / ((ask - bid) / 2)`.

Conditional mean and median are computed only over observations with positive excess.

| Model | Quotes | Inside bid–ask | Mean excess | Conditional mean | Conditional median |
|---|---:|---:|---:|---:|---:|
| Weekly native | 3,566,742 | 73.2911% | 1.1091 | 4.1525 | 1.4238 |
| Weekly affine-200 | 3,566,742 | 72.9744% | 1.0996 | 4.0689 | 1.3980 |
| Daily+weekly native | 3,566,258 | 81.8431% | 0.3157 | 1.7387 | 0.6891 |
| Daily+weekly affine-200 | 3,566,258 | 80.0141% | 0.3320 | 1.6610 | 0.6806 |

Short-end inside-spread rates show the geometry effect directly:

| Model | 1D | 2D | 3D | 4D | 5D |
|---|---:|---:|---:|---:|---:|
| Weekly native | 24.10% | 32.46% | 42.05% | 55.71% | 45.40% |
| Weekly affine-200 | 24.42% | 32.65% | 42.20% | 55.02% | 45.27% |
| Daily+weekly native | 93.55% | 94.45% | 94.94% | 95.29% | 68.41% |
| Daily+weekly affine-200 | 84.64% | 89.59% | 91.93% | 92.61% | 67.21% |

The compression cost in the aggregate inside-spread rate is 0.32 percentage points for the weekly model and 1.83 percentage points for the daily+weekly model. Expiry geometry is the dominant effect.

Source reports:

- Weekly native: `/home/hans/quantdata_data/training/flex-shared-strikes/capacity/spx-2022-2025-shared-strikes-integer-weekly-no-am-v1/reports/market-expiry-fit.json`
- Weekly affine-200: `/home/hans/quantdata_data/training/flex-shared-strikes/capacity/spx-2022-2025-shared-strikes-integer-weekly-no-am-v1/reports/market-expiry-fit-affine-200.json`
- Daily+weekly native: `/home/hans/quantdata_data/training/flex-shared-strikes/capacity/spx-2022-2025-shared-strikes-dte1-5-weekly-v1/reports/market-expiry-fit.json`
- Daily+weekly affine-200: `/home/hans/quantdata_data/training/flex-shared-strikes/capacity/spx-2022-2025-shared-strikes-dte1-5-weekly-v1/reports/market-expiry-fit-affine-200.json`

## AR(1) dynamics results

The reported radius is

`d²/d = r Q⁻¹ rᵀ / 200`,

where `r` is an exact-horizon state return and `Q` is the fitted horizon covariance.

| Statistic | Weekly | Daily+weekly |
|---|---:|---:|
| State dimension | 200 | 200 |
| Dates | 891 | 891 |
| Exact 1W innovations | 861 | 861 |
| `rho(A)` | 0.9597 | 0.9535 |
| Slowest half-life | 16.9 weeks | 14.6 weeks |
| Minimum real eigenvalue of `K` | 0.0411 | 0.0476 |
| Invariant covariance condition number | 3.41 | 3.80 |
| Invariant covariance effective rank | 193.7 | 193.7 |

The normal mode is analytic Gaussian/chi-squared. Sampling mode independently draws historical one-week innovations and propagates them through the fitted transition. Sampling used 1,000,000 draws per reported horizon.

Probability of exceeding the market 95% radius:

| Model | Horizon | Market | Gaussian | Historical resampling |
|---|---|---:|---:|---:|
| Weekly | 1W | 4.99% | 0.131% | 4.99% |
| Weekly | 2W | 5.02% | 0.000944% | 2.37% |
| Weekly | 3W | 5.05% | 0.000573% | 2.08% |
| Daily+weekly | 1W | 4.99% | 0.0193% | 5.13% |
| Daily+weekly | 2W | 5.02% | 0.0000475% | 2.61% |
| Daily+weekly | 3W | 5.05% | 0.0000117% | 2.24% |

Interpretation:

- One-week empirical replay is exact by construction, apart from Monte Carlo sampling error.
- The independently assembled 2W and 3W returns are genuine convolution checks.
- Gaussian innovations are far too thin-tailed relative to the observed state returns.
- Historical resampling remains materially closer at 2W and 3W, although it does not reproduce the market tail fully.

## Implied-volatility PCA and invariant surfaces

The invariant state covariance was solved from the stable AR(1). The affine SANOS decoder was differentiated at the invariant mean, and the decoded IV Jacobian was combined with the invariant covariance. The displayed PCA moves are one invariant standard deviation. PCA signs are arbitrary.

| Cumulative decoded-IV variance | Weekly | Daily+weekly |
|---|---:|---:|
| PC1 | 95.11% | 94.31% |
| PC1–2 | 97.79% | 97.08% |
| PC1–4 | 99.33% | 98.88% |
| PC1–8 | 99.76% | 99.54% |

The deck includes:

- both invariant-mean/base implied-volatility surfaces;
- PC1–8 surfaces on one slide for each model, with an independent visible scale
  for each factor;
- six invariant-distribution ATM term structures for each model;
- one paired complete invariant-distribution surface draw.

The weekly model has no independent 1D–4D state directions, so its short-end shapes inherit extrapolation from the 5D node. The daily+weekly model can express front-end dislocations explicitly.

Invariant-distribution figures are static draws from `N(b, Sigma_infinity)`, not simulated time paths. The final paired surface uses the same 200-dimensional standard-normal seed, mapped through each model’s own invariant covariance and decoder.

## Final slide map

The deck now contains 50 slides. Added slides use the original Oxford blue
table treatment and carry a blue `REVIEW — NEW` badge. Replacement slides
carry a blue `REPLACES N` badge.

### Fit section

- Slide 20: hidden original weekly-model slide.
- Slide 21: replacement market-fit summary and all-expiry line chart.
- Slide 22: conditional-excess curves and compression interpretation.
- Slide 23: example fit between model expiries.

### AR dynamics section

- Slide 25: hidden original AR-statistics slide.
- Slide 26: replacement stability and tail comparison.
- Slide 27: weekly market/normal/resampling table at 1W, 2W, and 3W.
- Slide 28: daily+weekly market/normal/resampling table at 1W, 2W, and 3W.

### PCA section

- Slide 29: hidden original PCA slide.
- Slide 30: replacement decoded-IV PCA spectrum comparison.
- Slide 31: invariant-mean/base IV surfaces.
- Slide 32: weekly PC1–8 surfaces on one slide.
- Slide 33: daily+weekly PC1–8 surfaces on one slide. This slide is visible
  and immediately follows the weekly PCA slide.

### Simulation and conclusion

- Slide 35: hidden original simulation slide.
- Slide 36: replacement invariant ATM term-structure draws.
- Slide 37: paired complete invariant-volatility surface draw (“gimmick”).
- Slide 44: hidden original summary slide.
- Slide 45: replacement summary.

### Appendix and contact

- Slide 46: weekly-native 1D–69D table.
- Slide 47: weekly-affine-200 1D–69D table.
- Slide 48: daily+weekly-native 1D–69D table.
- Slide 49: daily+weekly-affine-200 1D–69D table.
- Slide 50: contact.

Slides 18 and 19 also contain small visible review remarks clarifying affine coordinates versus IV PCs and weekly dynamics steps versus expiry-node grids.

## Presentation validation

The following checks were performed after the final 50-slide regeneration:

- opened successfully with `python-pptx`;
- PowerPoint opened a Windows temporary copy successfully;
- every new/replacement slide was exported to PNG by PowerPoint;
- the four dense per-expiry tables, base surface, both combined PC1–8 slides,
  AR tables, and final summary were visually inspected;
- the PPTX ZIP package passed `ZipFile.testzip()` with no corrupt member;
- the five superseded originals are hidden, not deleted.

## Source changes related to the completed model runs

The relevant source work includes:

- `quantdata/dysanos/flex/run_all_history_joint_direct.py`
  - restored market configuration handling;
  - added `--day-batch-size` for the joint stage.
- `quantdata/dysanos/flex/run_shared_strikes_full_encoder.py`
  - preserves exact explicit expiry-grid geometry in the market cache.
- `quantdata/dysanos/flex/joint_direct.py`
  - permits checkpoint warm starts when only operational day batch size changes.
- `tests/test_flex_joint_direct.py`
  - covers explicit market-expiry geometry and restart batch-size compatibility.

Verification performed for those focused changes:

- `python -m pytest -q -p no:cacheprovider tests/test_flex_direct_state.py tests/test_flex_joint_direct.py tests/quantdata/dysanos/flex/test_smooth_q_objective.py`
- Result: 60 passed.
- Ruff import checks passed.
- Python compilation checks passed.

## Repository state and handover cautions

- Branch: `main`
- Upstream: `origin/main`
- HEAD: `b6928c3 Refine FLEX joint transport training`
- Remote: `https://github.com/hansbuehler/quant_data.git`
- The final deck and handover are tracked. The pre-review backup remains a local
  recovery artifact and is intentionally not committed.
- Unrelated notebook changes remain outside the task commit.
- The user approved the commit; no push was made.
- Generated W&B logs, obsolete FLEX failure snapshots, the accidental `NUL`
  file, and task-specific temporary analysis/render files were removed. Current
  completed model checkpoints and paused conditional-Q checkpoints were kept.
- Do not delete or overwrite the pre-review backup.
- The one-off chart-generation scripts and PowerPoint validation renders were kept under `/tmp` and Windows temporary storage; the durable results are embedded in the PPTX and the durable source reports/checkpoints are listed above.
