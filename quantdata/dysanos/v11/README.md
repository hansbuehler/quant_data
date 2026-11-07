# DYSANOS 1.1

Post-paper implementation. The paper sources and historical results are not
modified in place.

- [Current specification and model disposition](../MODELS.md)
- [Code and quantitative review guide](../REVIEW.md)

## Entry points

| Module | Role |
| --- | --- |
| `rolling_daily.py` | Current daily full recalibration, GPU-resident rolling window, checkpoints and progress. |
| `daily_fit_report.py` | Per-origin own-day fit, weighted previous-day mean/median fit and runtime. |
| `licensed_market.py` | Observed bid/ask reader and certified historical SANOS source loader. |
| `prepare_dlv_initialization.py`, `dlv_initialization.py` | Algebraic DLV initialization on the requested grid. |
| `jax_affine200.py`, `jax_affine_lbfgs.py`, `jax_geometry.py` | Shared JAX pricer, initial affine/direct fits and geometry. |
| `audit_model_grid.py`, `audit_daily_fit.py`, `audit_jax_affine.py` | Grid-price, daily and expiry-level fit diagnostics. |
| `atm_spot_correlation.py` | Historical SANOS, affine reconstruction and generated-path leverage comparison. |
| `jax_joint_ar.py` | Historical weekly pilot; **not** the daily rolling runner. |

The earlier 252-session dynamics studies and their disposition are summarized
in [MODELS.md](../MODELS.md#historical-comparators). Preserve their reports;
do not relabel them as results of the new 250-date sweep.

Generated outputs remain outside Git in unique run directories. Use an
explicit interpreter, working directory and project environment when launching
detached jobs, and allocate GPUs through `gpuctl` as required by `AGENTS.md`.
