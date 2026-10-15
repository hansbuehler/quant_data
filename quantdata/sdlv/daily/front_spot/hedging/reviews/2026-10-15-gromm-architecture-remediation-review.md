# GROMM Architecture Remediation Review

Date: 2026-10-15

## Scope

This independent review covers the replacement of the 2022 GROMM hedge-policy
architecture study after `RollingDlvFactorConfig` changed from version 1.1 to
1.2. The change adds partial-observation covariance extraction and active-DLV
coordinate masking, so the prior architecture evidence and checkpoints cannot
be relabelled for current production.

## Result

**PASS.** No quantitative or production-binding blocker remains.

- The replacement study ran from clean commit
  `fe7c3a7321345e8ced636c5f75518766b830e452` against only the distributable
  public-candidate dataset. Its launcher explicitly forces JAX and Torch to
  CPU and completed successfully.
- The authenticated result is
  `/home/hans/quantdata_data/training/hedging/gromm/architecture-2022-v2-fe7c3a7/result.json`,
  SHA-256
  `ed68dd71a87465507191f476cf85eea691596751fa122d95ac21c7381abb166b`.
- The study uses origin `2022-12-30`, a 200-session calibration window, Q and
  P populations, 40 cases, 128 paths per case, 5,120 paths in total, 16 DLV
  factors, 20 policy features, 206 instruments, and mean-variance risk
  parameter 10. The compatible P drift is accepted.
- All four prescribed architectures have finite train and validation losses,
  report convergence, and have zero position-cap hits. Their checkpoints use
  float64, seed `20261012`, maximum/minimum/patience epochs `2000/100/100`, and
  `batch_pairs=512`; every checkpoint has no CUDA RNG state.
- Reconstructing the one-standard-error rule selects
  `gromm-architecture-2022-h64-d3-w128`. The best validation loss is
  `0.08181613770898842` and the one-standard-error threshold is
  `0.09032001233718524`.

## Production Binding

The reviewed coordinator launcher SHA-256 is
`d729a52ee477f1db48b1e8eb68cf608eeb185a7fb3cb680f0efaade89891973d`.
It pins the architecture result and CPU-study launcher hashes, invokes the
commit-pinned architecture loader, independently validates the study
population, four geometries, finite losses, convergence, zero cap hits, and
completion marker, and passed its preflight-only run.

Fresh matrix construction from the pinned production worktree reports every
GROMM case ready: 27 of 27 daily cases and 54 of 54 weekly cases. The primary
scratch matrix SHA-256 values are
`9e8027373baa3451f2f1b6e712e8d0583bfa0ee9de587dcbc6bf729669f502cc`
and `258840d416ee4963bc9ae79786d6f06b7978e73b5502ee17c53ddb68749d0789`.
An independent reconstruction also passed all cases.

## Verification

The reviewer independently ran the current loader, reconstructed parameter
counts and the one-standard-error decision, inspected all four checkpoint
metadata records, checked CPU-only provenance and completion logs, audited the
hardened coordinator preflight, and rebuilt both production matrices from the
pinned worktree.
