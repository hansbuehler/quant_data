# GROMM2 architecture reconciliation

## Scope

This is a read-only comparison of the hedging GROMM implementation with the
current `quantdata/gromm2` work. No `quantdata/gromm2` file was modified. Its
focused mathematical suite passes 12 tests.

## Headline conclusion

GROMM2 should inform the benchmark's market-scenario architecture, but its
current full-sample artifacts cannot be used directly in the rolling
out-of-sample hedge test. The hedge-policy architecture and the option-market
scenario architecture are separate decisions:

- retain the four precommitted affine/deep/LSTM hedge-policy candidates;
- replace the older fixed-grid scenario source with GROMM2 only after a dated,
  rolling-as-of artifact interface supplies both `P` and `Q` one-day scenarios;
- do not import a live research project into the hedging package. Stabilize an
  immutable artifact contract or extract genuinely shared numerical kernels.

## Material differences

| Component | Current hedging GROMM | Current GROMM2 |
| --- | --- | --- |
| Expiry geometry | Released fixed 25-expiry DLV state | 45 expiries: daily to 25D, then 5D to 125D |
| Strike geometry | Common fixed strike support | Expiry-dependent expanding supports and barycentric transports |
| Historical fit | Rolling 252 one-session transitions | Terminal-date geometry plus a joint full-panel affine fit and 863-pair diffusion fit |
| Diffusion | Rolling fixed loadings | 16 factors with optional selective state-space amplitudes |
| Front treatment | Exact one-day information bridge | Exact bridge integrated with the expanding tower |
| Tail control | Exceptional drift taming on the fixed grid | The same exceptional taming plus density-localized loadings |
| Measures | `P` and `Q` daily kernels | Current path runner is `Q` only |
| Benchmark output | Exact current hedge book and target digital | Audit option grid; no benchmark target-book adapter yet |

The current GROMM2 estimate reports 92.458% explained innovation variance for
16 factors. Its selective state-space model is accepted on validation
(negative log likelihood 0.4664 versus 0.5539 for the constant baseline). It
also reports that 10.803% of fixed-contract interpolation queries are clipped and a
35.910% physical-drift projection fraction; both require explicit benchmark
diagnostics rather than silent acceptance.

## Required integration gates

1. Produce one artifact per benchmark as-of date using only its rolling
   information set. Both current construction stages leak future information:
   the geometry is defined from the terminal panel date and the affine DLV
   history is jointly calibrated across the full panel. Truncating only the
   later 863-pair diffusion sample is therefore insufficient.
2. Implement and test the GROMM2 physical-measure step. Physical coefficients
   are estimated, but the current path simulator calls only the risk-neutral
   step.
3. Add a one-day fixed-contract adapter that prices the exact benchmark option
   universe and all 20 daily digital targets in common discounted origin-spot
   units.
4. Preserve the exact bridge, expanding supports, transport tower,
   density-localized loadings, and 5% background process. Use the benchmark's
   precommitted eight DLV substeps; eight is a benchmark convention rather than
   an intrinsic GROMM2 architecture requirement.
5. Resolve simulator memory before production. The attempted 100,000-path Q
   runs failed during XLA allocation/autotuning at batches 4,096 and 1,024; no
   successful production path artifact is present.
6. Re-run conditional and unconditional Q martingale tests, plus the accepted
   physical-drift diagnostics, on the dated adapter before architecture or
   hedge-performance selection.

Until these gates pass, the existing hedging GROMM remains a development smoke
only and must not be presented as the final GROMM benchmark result.
