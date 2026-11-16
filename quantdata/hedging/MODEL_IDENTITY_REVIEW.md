# DYSANOS paper-model identity correction

Date: 2026-11-12. The model labelled `DYSANOS v1` in the current audit is
**not the paper model**. Its figures must not be used to assess the paper's
calibrated affine-200 state-space model. The replacement full-state adapter
is now implemented; results are stored separately and do not retroactively
relabel the old surrogate forecasts.

## Correction execution

The authenticated joint paper checkpoint loads with its original model-version
check. Its exact daily embedding is positive definite (minimum covariance
eigenvalue 0.0009287881); no covariance clipping or PCA truncation is required.
The focused embedding tests pass and the 64-path, 2024-04-01 decoder pilot
completed. The 512-path-per-origin run completed all 44 April--May origins using
the full calibrated 200-dimensional dynamics. Its predicted variance trace is
0.494684, forecast MSE 1.684396 and variance/MSE ratio 0.293686. These values
replace neither an out-of-sample result nor the old control rows by relabelling.
Because the checkpoint used all
891 dates in fitting, its results are marked **in-sample** throughout. The
adapter is `model_audit/paper_v1.py`; output is
`/home/hans/quantdata_data/quant_data/hedge2/runs/model-corrections-20261112-v1/paper-v1-full/`.

Separately, the standalone GROMM1 probability-weighted innovation fit completed
for its saved 2025-08-29 origin. Its independent-Y latent likelihood, market fit,
geometry, return convention, historical weights and 16-factor budget are
preserved; P/Q drift fields are recomputed. Probability-weighted innovation
retention is 91.8520%, versus 90.9079% for the compatible baseline. This is a
single-origin compression comparison, not the older GROMM2 cohort's variance /
forecast-MSE statistic. Its matched decoded-IV experiment is stored under the
same correction root, in `gromm1-pweight-forecast-v2/`; the inference-validated
weighted archive is `gromm1-pweight-v2/dynamics.npz`.

The matched decoded comparison also completed: eight Brownian substeps, 512
paths per cell, same random numbers and zero invalid paths in all four cells.
Q log-IV variance trace is 1.481412 before weighting and 1.521679 after
(+2.72%); P is 2.230006 before and 2.758307 after (+23.69%). This pilot does
not show a reduction in decoded variance from weighting. There is no matched
next-session outcome in this single-origin control, so it supplies no new
variance/forecast-MSE calibration ratio. Do not replace or compare these raw
variance traces with the older April--May GROMM2 ratios in the table below.

## Verified differences

| Component | Paper | Current audit surrogate |
| --- | --- | --- |
| Affine representation | Learned `u=b+B h`, 200 state coordinates | Unweighted PCA/SVD projection of saved native states |
| Surface calibration | Refine affine bias, loadings and states against market quote loss | No market-objective refinement of that projection |
| Final dynamics | Jointly refined stable 200-dimensional VAR and full innovation covariance | Separate eight-dimensional VAR on native change PCs |
| Estimation horizon | Exactly five NYSE sessions | Exactly one session |
| Role of PCA | Initialize affine fit; decoded-IV PCA describes resulting invariant variation | PCA truncation determines the fitted dynamics |
| Eight displayed PCs | Decoded-IV diagnostic directions, not the underlying state dimension | Eight native-coordinate factors replace the dynamics |

The paper's equations and calibration appendix were checked directly in
`/home/hans/python/quant_data/quantdata/dysanos/tex/dy21.tex`:
affine state dimension at line 989; joint state process at 1043; five-session
likelihood at 1071; joint objective at 1103; decoded-IV diagnostics at 1222;
market refinement after PCA initialization at 1971; joint update steps at 2010.
The paper explicitly says all 891 dates enter the reported joint fit at 2045.

The corresponding retained joint-calibration implementation is
`quantdata/dysanos/flex/joint_direct.py`: `fit_joint_direct_model` alternates
market quote/cap updates with transition/gauge updates. The current audit
instead uses `model_audit/projection_metrics.py:affine_projection`,
`fit_projected`, and the one-session `quantdata/hedge2/dynamics.py:fit_dynamics`.
`model_audit/versioned.py` selects rank 8 for the row called v1.

The retained daily+weekly joint checkpoint was read after SHA-256 verification:

`/home/hans/quantdata_data/training/flex-shared-strikes/joint-direct/dte1-5-weekly-no-am-all-history-v1/checkpoints/joint-direct-200.pt`

SHA-256: `395549f12737b8ee301c2c4b35999379f7ef55e7e2cfe0fcda6cf414068ae2ac`.
It has schema `flex-joint-direct-dynamics-v1`, completed phase, 300 epochs,
state size 200, and 200-by-200 damping, rotation and innovation matrices.
The current audit's 2024-04-01 v1 artifact has an 8-by-8 AR matrix,
200-by-8 PCA directions and 252-by-9 paired spot/factor residuals.
This establishes the architecture mismatch; it does not establish that this
older no-AM checkpoint reproduces every number in the current paper draft.

## Consequences for the variance analysis

- The low variance/forecast-MSE ratio of the audit's v1 is a result for the
  eight-PC daily surrogate, not evidence against the paper model.
- The weighted/native-PCA comparisons remain ablations of the tested pipeline.
  They do not show that market-calibrated affine-200 compression is unnecessary.
- Decoded invariant-IV PCA coverage, native-coordinate PCA retention,
  reconstruction of realized returns, and predictive variance/forecast-MSE
  ratios have different denominators and cannot be substituted for one another.
- The paper's daily simulation must use its specified generator and a validated
  corresponding daily innovation covariance. Neither refitting an independent
  daily VAR nor simply dividing weekly innovation covariance by five is an
  identity-preserving replacement.
- A checkpoint fitted on all 891 dates includes April--May 2024. Scores on
  those dates are not genuine rolling out-of-sample tests. A causal comparison
  requires appropriately dated fits, or an explicitly in-sample reproduction.
- The newly specified SANOS v2 (ATM-determined geometry, smoothness input 0.1)
  remains a new design. Existing STRT-16 audit artifacts are not that new fit.

Affected paper-v1 comparisons and model-specific production reruns are held
pending use of a verified paper-model adapter. Existing numeric arrays are
preserved as surrogate controls, not silently relabelled as corrected results.

## Existing variance/forecast-MSE figures

All rows below use the unchanged 44-origin April--May reference and the same
49 floating log-IV coordinates. The statistic is mean conditional covariance
trace divided by mean squared forecast-error norm. The reference is one;
mean misspecification also affects its denominator. These are descriptive
44-origin ratios, not independent-sample calibration certificates.

| Model actually evaluated | Variance / forecast MSE |
| --- | ---: |
| Eight-PC daily surrogate, previously labelled DYSANOS v1 | 0.045722 |
| Retained STRT-16 daily experiment, labelled DYSANOS v2 | 0.513043 |
| FuNVol | 0.436959 |
| VolGAN PCA8 | 0.000201945 |
| AD-Seq-Vol | 0.247518 |
| AD-Seq-Vol-FT | 0.158428 |
| GROMM-P | 2.063586 |
| GROMM-Q | 1.815547 |
| Correct paper affine-200 model | Not yet measured in this audit |

Source: unchanged `summary.json` in
`/home/hans/quantdata_data/quant_data/hedge2/runs/model-dynamics-audit-20261112-two-versions-v2/`.
The existing 0.1-source/0.25-observable-reference difference is retained and
disclosed; no dataset or spot-return convention was changed by this review.
