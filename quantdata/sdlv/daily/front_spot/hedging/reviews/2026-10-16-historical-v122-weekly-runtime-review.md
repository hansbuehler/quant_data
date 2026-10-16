# Historical-SANOS v1.22 weekly runtime review

## Verdict

**Leave the active four-worker queue unchanged.** No restart or live
intervention has a demonstrated, provenance-safe speed advantage.

The reviewed service runs from immutable, read-only source commit
`8b7ca46c73ddf278ebcb23b7ddcb4eb9b12b8edf`. Its launcher is
`/tmp/quant_data/hedging/launchers/historical-v122-weekly-8b7ca46.sh`, has mode
`0444`, SHA-256
`f2322723db58b633ae25faad0f6b80d2ae773644e3303d5a4a4a489240d84521`,
and fixes `MAX_PARALLEL=4`. All four live worker environments pin OMP,
OpenBLAS, MKL, BLIS, NumExpr, and vecLib to one thread.

## Runtime evidence

At review time the active frozen-weekly cases reported:

| Objective | Completed origins | Seconds/origin | Estimated remaining |
|---|---:|---:|---:|
| mean--variance, 0.1 | 254 / 528 | 53.7 | 4.09 h |
| mean--variance, 1 | 248 / 528 | 54.9 | 4.27 h |
| mean--variance, 10 | 242 / 528 | 56.2 | 4.47 h |
| entropy, 0.1 | 8 / 528 | 323.7 | 46.75 h |

The run already avoids material repeated work. Weekly scenarios and pricing
contexts are cached by fit and decision state, the zero-claim policy is reused,
and compiled CVXPY geometries are parameter-cached. Long, short, and pure fits
are intentionally concurrent. Resume is origin-atomic and authenticating:
existing records are re-read and revalidated, so a restart would lose at most
the current origin but has no known benefit.

## Threading finding

Clarabel 0.11.1 retains its default `max_threads=0`, which may use all physical
cores internally through Rayon/faer even though the standard BLAS libraries are
single-threaded. A short process sample showed scheduler churn, but not evidence
that changing this setting would make the run faster. Changing floating
reduction order during an authenticated production run could also change
diagnostics near a numerical certificate threshold.

A future, isolated exact-origin benchmark may compare explicit Clarabel thread
limits, such as one and six threads. Any successor must reproduce the certified
objective and diagnostics before it may replace the current implementation.
This is not a reason to alter or restart the active v1.22 queue. The historical
matrix is CPU-only and neither uses nor requires a GPU.
