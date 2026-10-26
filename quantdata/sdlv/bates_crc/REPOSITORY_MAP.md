# Repository map: discovery for the DLV x extended-Bates CRC comparison

Scope: `quant_data-main` (this repository) plus the separate
`heston_bates_oracle` sandbox referenced by the task ("the repository
containing our previous Heston/Bates, SANOS reconstruction, dynamic-lift and
compensator experiments" — a sibling directory under `Agentic/`, **not** part
of this git repository). Everything below was verified by reading source and,
where marked, by running code in a fresh `.venv312` (Python 3.12, built from
this repo's own `pyproject.toml`) with `QUANTDATA_TEST_MODE=1`.

Builds on, and does not repeat in full, the earlier
`quantdata/dysanos/flex/crc_extension/PHASE1_AUDIT.md` (same repository, a
prior task). New findings specific to this task are marked **(new)**.

## 0. Which subsystem is "the" dy2 statistical simulator

`quantdata/dysanos/` contains several independent subsystems: `flex/`,
`grid/gendysanos`, `grid/rndysanos`, `implied/`, `localvolbysanos/`,
`tangent_levy/`. As established in the prior audit, **`flex/`** is the one
matching the supplied `dy2.pdf` manuscript's description (two term
structures `sigma^ell`/`eta`, weekly AR(1), 200-dimensional affine state,
Sections 4.1/4.5/4.6 of `dy2.pdf`) and is treated as "the dy2 statistical
simulator" throughout this document.

## 1. The actual SANOS/DLV decoder and its price evaluation

**(new)** There are two distinct decoders, not one:

| Decoder | File | Role |
|---|---|---|
| `quantdata.sanos.arbfree.{MkArbFree, ArbFreePricer}` | `quantdata/sanos/arbfree.py` (1427 lines), `quantdata/sanos/sanos.py` (`SanosFitter`, cites arXiv:2601.11209 directly in its docstring), `sanos_tools.py`, `tridiag.py` | The **production, day-by-day, convex-optimisation (cvxpy) static SANOS fit** used to turn one day's raw WIDB bid/ask quotes into an arbitrage-free mid-price surface. This is the "actual SANOS/DLV decoder" in the sense of the base paper. Units: `min_dlv=0.01`, `max_dlv=50` — an **unsquared** local-volatility-scale bound, not `dlvsq`. |
| `quantdata.dysanos.flex.sanos.FlexSanosDecoder` | `quantdata/dysanos/flex/sanos.py` | A **torch-differentiable, parametrised** re-implementation of the same Wissel/Magnus finite-grid DLV recursion (confirmed directly from the `w_plus`/`w_minus` weight construction — the same `A_m(ell^m) q^m = q^{m-1}` structure), consuming `dlvsqs` (**squared**, `[0.01^2, 3.0^2]`), `forward_vols`, `eta` and producing a `SanosSurface` via `forward_batch`/`price_calls_batch`. Used inside the joint latent-state model and the weekly simulator. |

**Consequence for Section B (conventions)**: the two decoders use different
local-volatility scalings (`dlv` vs `dlvsq`) and different smoothing
conventions (`arbfree`'s `smoothness` vs `flex`'s `eta` multiplier on
`forward_vol`, confirmed by tracing into `_black_call`'s literal
`sqrt_variance` argument in the prior audit). A conversion table is required
before any cross-model comparison; see `MODEL_CONVENTIONS.md`.

## 2. Static calibration and preprocessing

`quantdata/dysanos/market.py::smooth_interpolated_market` (cached function):
loads one day's filtered WIDB options
(`quantdata.widb.options_filter.filter_options_day`), fits the day with
`quantdata.sanos.arbfree.MkArbFree(smoothness=..., surface_expiry_linear=True)`,
and returns `af_call_mids`/`af_put_mids`/`model_masks`/`fit_result`. This is
the **cross-sectional (single-day) calibration boundary**: everything
upstream is raw-data handling; everything downstream (FLEX) treats this
output as its target.

`quantdata/dysanos/flex/market.py::FlexMarketDay` adapts that output to
FLEX's fixed-weekly geometry (`FlexGeometryConfig`: `num_inner_strikes=200`,
`nms_range=(-3.0,2.0)`, `num_model_expiries=14`,
`first_model_business_dte=4.8`, `model_expiry_step_business_days=5.0`).

`quantdata/dysanos/flex/training.py`: the joint native/compressed-state
fitting routines (`materialize_day`, `materialize_batch`, `quote_objective`,
`evaluate_model`, `market_native_raw_initial_states`, `training_convergence_audit`).
**(new)** This file has **no CLI/`__main__`** — it is a library consumed by
the `run_*.py` scripts (`run_shared_strikes.py`,
`run_shared_strikes_joint_direct.py`,
`run_shared_strikes_joint_hierarchical.py`, `run_phase3_encoder.py`, ...),
each a standalone experiment entry point, not a single canonical CLI.

## 3. Historical state estimation, compression and transition estimation

`quantdata/dysanos/flex/weekly_dynamics.py`: PCA-reduced (`_fit_level_pca`),
per-expiry-block, **hierarchical VAR(1)** (`HierarchicalBlockFit(intercept,
own_lag, previous_lag)`, ridge-regularised `_ridge_coefficients`), fixed
5-business-day stride, joint innovation covariance, a separate joint
log-spot drift/beta/orthogonal-std block, and a **gated** continuous-time
generator check (`_verified_generator`: condition-number ceiling,
matrix-exponential round-trip tolerance) — a P-side stability diagnostic
only, confirmed in the prior audit to contain no `Q`-restriction anywhere.
`STRONG_STATE_SIZES = (16,19,17,15,13,10,10,10,5,5,5,5,5,5)` (14 expiry
blocks, state dimension 135 before further PCA reduction — **(new)**, not
identical to the round "200" figure quoted in `dy2.pdf`'s abstract, which
refers to the **native**, not reduced, affine map `u_t = b + B h_t`,
`h_t in R^200`, per `dy2.pdf` Definition 3.1).

`fixed_expiry_ar1.py`/`last_expiry_ar1.py`: separate coordinatewise AR(1)
fits for the DLV **tower's own calendar roll** (new front expiry entering,
old one settling) — a different object from the weekly cross-expiry
dynamics above.

## 4. Spot simulation and its dependence on surface shocks

`WeeklySanosSimulator._simulate_internal` (`weekly_simulation.py`): two
`spot_mode`s. `"normal"`: spot uses the dynamics model's own joint Gaussian
log-spot innovation (loads on the surface innovation via
`weekly_spot_front_beta` plus an orthogonal residual — confirmed genuine
joint spot-surface dependence, not a bolted-on independent spot process).
`"implied"`: spot's one-step gross return is drawn from the **current
decoded front-week SANOS density's own inverse CDF**
(`front_week_sanos_inverse_cdf`), i.e. spot realisations are statically
consistent with the surface at every step — required when using a trained
`measure_kernel` (target-measure simulation asserts this mode).

## 5. Fixed-contract repricing, expiry handling and grid management

`_option_values` (spot-normalised pricing, reused by `hjm_roll_residuals`
and the CRC extension's Section 2 code), `hjm_roll_residuals`/
`hjm_roll_moment_loss` (the **already-correct, pathwise discrete
martingale/HJM roll residual**, formalised and tested in the prior CRC
extension task: raw P fails with t-stat 16.9 at 100,000 paths; a
gradient-trained target-measure kernel reduces it to t-stat 8.9, not zero).
`make_tradable_option_grid`, `WeeklyOptionGridConfig`, `TradableOptionGrid`
manage the fixed cash-strike/expiry book across weekly rolls.

## 6. Existing ordinary/generalised Bates oracle implementations

**None exist anywhere in `quant_data-main`** — verified directly:
`grep -rl "Bates\|bates" --include="*.py" quantdata/` and
`grep -rl "Heston\|heston" --include="*.py" quantdata/` match **only** the
files added by the prior CRC-extension task
(`flex/crc_extension/section2_pricing_identity.py`, which fits synthetic
Heston-*free* dynamics, not a Bates oracle). The generalised-Bates oracle
(`generalized_bates.py`, `oracle.py`, jump extension, Fourier pricer,
moment-explosion certificate, exact continuous mixture — everything
Sections 6/12 of `dy2.pdf`'s companion "DYSANOS II" manuscript build on)
lives **entirely** in the separate `heston_bates_oracle` sandbox
(`Agentic/heston_bates_oracle/`, outside this git repository). Implementing
`dy2.pdf` Section 6's `Theorem~thm:dlv-bates` / Section 12's
`Theorem~thm:bcrc-reserve` construction inside `quant_data-main` will need
either porting that oracle in, or calling it as an external dependency —
a decision for the next task, flagged here rather than assumed.

## 7. CRC parameter updates and extension representations

**None pre-existed** beyond `weekly_measure_training.py`'s SMC/gradient-based
"target-measure kernel" fit (already audited: it narrows but does not
eliminate the martingale residual for the tested family — see
`flex/crc_extension/SECTION2_3_FINDINGS.md`). **This is a fitted
approximation, not the exact admissible reset the current `dy2.pdf`
manuscript's Section 12 `Theorem~thm:bcrc-reserve` specifies** (a two-sided
scheduled L\'evy-reserve region with an *analytically certified*
nonnegative-triplet correction, no gradient descent involved). The prior
CRC-extension task's `section5_crc_sequence.py` implements the generic
"evolve -> propose -> constrained inverse -> feasibility decision -> continue"
**sequence machinery** (steps A-G) on top of the FLEX decoder, exploiting
that decoder's automatic static admissibility — this is a genuine
price-preserving reset mechanism, but it is **not** the Bates-specific
exact-reserve construction of `dy2.pdf` Section 12, which has not been
implemented anywhere in either repository. **Do not conflate these three
objects** (SMC target-measure kernel; the generic CRC sequence machinery;
the Bates-reserve exact theorem) when scoping the next task.

## 8. Tests, checkpoints, reports and data loaders

**Tests** (all pass, verified): `tests/test_dysanos_torch_simulator.py`,
`tests/test_dysanos_market.py` (18/18, pre-existing baseline);
`tests/test_flex_weekly_phase3.py` (uses a **monkeypatched**
`filter_options_day` to exercise the real `smooth_interpolated_market` ->
`MkArbFree`/`ArbFreePricer` fitting logic on synthetic-but-realistic raw
quotes, **without** needing cached WIDB data — this is the reusable pattern
for building further smoke tests without real data, see Section D below);
`tests/test_flex_crc_extension.py` (9/9, prior task). Also present:
`tests/test_gendysanos_*.py`, `tests/test_rndysanos_*.py` (the other `grid/`
subsystems, not audited here — out of scope for this comparison).

**Checkpoints**: **(new)** only one trained artefact exists in this checkout,
`flex/reports/phase3-spread/feed_forward.pt` — a spread-cost regression
network, unrelated to the state/dynamics model. No `WeeklyDynamicsFit`, no
trained `JointHierarchicalDynamicsModel`/`LatentStateSanosModel` state dict,
and no trained `weekly_measure` kernel are present anywhere in this
checkout. **`dy2.pdf`'s reported 891-date, 3.57M-quote SPX experiment was
not run in, and its checkpoint is not present in, this environment.**

**Reports** (JSON/markdown write-ups, not runnable checkpoints):
`flex/reports/{phase3-weekly,phase3-encoder,phase3-native-encoder,
phase3-spread,batch-descent}/`. The `batch-descent/spx-2022-2025-forward-mu-v1*`
files are the closest match to `dy2.pdf`'s reported experiment by name and
date range, but contain summary metrics/markdown, not raw quotes or a
loadable model — see `DATA_STATUS.md`.

**Data loaders**: `quantdata.widb.options_filter.filter_options_day`,
`quantdata.widb.day` (`RawDayFilter`, `ExpiryFilter`, `enrich_day`),
`quantdata.widb.arbfree.mk_arb_free_day` (CLI: `python -m arbfree SPX RUT NDX
--period 2020-05-01 2022-01-31`, requires live WRDS/Massive credentials or a
pre-populated cache — see `DATA_STATUS.md`).

## 9. Call graph (fixed-contract pricing path)

```
raw WIDB quotes
  -> quantdata.widb.options_filter.filter_options_day
  -> quantdata.dysanos.market.smooth_interpolated_market
       -> quantdata.sanos.arbfree.MkArbFree / ArbFreePricer   (static, per-day, cvxpy)
  -> quantdata.dysanos.flex.market.FlexMarketDay              (fixed-weekly adapter)
  -> quantdata.dysanos.flex.training.materialize_day/batch    (native/compressed state fit)
  -> quantdata.dysanos.flex.model.{LatentStateSanosModel, JointHierarchicalDynamicsModel}
       -> quantdata.dysanos.flex.model.bound_native_raw       (-> dlvsqs, forward_vols, eta)
       -> quantdata.dysanos.flex.sanos.FlexSanosDecoder.forward_batch -> SanosSurface
  -> quantdata.dysanos.flex.weekly_dynamics.TorchWeeklyDynamics.step   (P transition)
  -> quantdata.dysanos.flex.weekly_simulation.WeeklySanosSimulator
       .decode_surface / ._option_values / .hjm_roll_residuals
       (+ .weekly_measure.NormalWeeklyMeasureKernel, optional Q-candidate)
```

## 10. Executable smoke test (this task)

Reused, not invented, entry points; verified to pass in this environment:

```bash
cd quant_data-main
python3.12 -m venv .venv312        # if not already present
.venv312/bin/python -m pip install -e . pytest pytest-cov
.venv312/bin/python -m pytest tests/test_dysanos_torch_simulator.py \
    tests/test_dysanos_market.py tests/test_flex_weekly_phase3.py \
    tests/test_flex_crc_extension.py -q
```
Result (this environment): `test_dysanos_torch_simulator.py` +
`test_dysanos_market.py`: **18/18** (previously verified);
`test_flex_crc_extension.py`: **9/9** (previously verified);
`test_flex_weekly_phase3.py`: **24/25** when run as a full file — one test,
`test_antithetic_front_audit_and_q1_only_training`, fails by a hair
(`0.00617 < 0.005` assertion) when run as part of the whole file, but
**passes in isolation** (twice, reproducibly). This is order-dependent RNG
state leakage between tests in that file, a **pre-existing baseline
characteristic** (no file in `flex/` was modified by this or the prior
task) — reported honestly here rather than rounded up to "25/25", and
flagged as worth a global-seed-isolation fix before this file is relied on
for the next task's numerical comparisons. `conftest.py`
sets `QUANTDATA_TEST_MODE=1` automatically, so no manual environment
variable is required for `pytest`.

**(new) Important test-mode limitation, found directly, not assumed**:
`QUANTDATA_TEST_MODE=1` uses `cache_mode="cacheonly"` — it can only **read**
already-cached fixture files under `tests/stored_data/`; it cannot compute a
*new* derived quantity even from already-cached raw inputs whose specific
processed-cache key was never populated. Confirmed directly: calling
`quantdata.widb.arbfree.mk_arb_free_day("TEST", date(2025,6,3))` in test
mode raises `CacheMustExistError` because that specific derived cache entry
was never generated, even though the underlying raw WIDB fixture for that
date exists. **The only way to exercise new code paths against realistic
data without live credentials is the monkeypatch pattern already used by
`test_flex_weekly_phase3.py`** (bypass the raw-data cache boundary directly
with synthetic-but-schema-correct data), not by pointing at
`tests/stored_data/` directly for anything not already pre-generated.

## 11. Missing dependencies and mathematical mismatches (concise summary)

**Missing dependencies**:
- WRDS Option Metrics credentials (`WRDS_USERNAME` + `~/.pgpass`) or a
  `MASSIVE_TOKEN`, plus a populated `QUANTDATA_ROOT` — see `DATA_STATUS.md`.
- A Bates oracle inside `quant_data-main` (none exists; the only one is in
  the separate `heston_bates_oracle` sandbox) — needed before
  `coherent_dlv_x_ordinary_bates` or `coherent_dlv_x_extended_bates_crc`
  (Section 6, 7 above) can be implemented, not merely stubbed.
- A trained FLEX checkpoint (none present) to reproduce `dy2.pdf`'s own
  reported numbers rather than a freshly-fit synthetic instance.

**Mathematical/convention mismatches, found and recorded (not silently
reconciled)**:
1. `dlv` (`quantdata.sanos`, unsquared, `[0.01,50]`) vs `dlvsq`
   (`quantdata.dysanos.flex`, squared, `[0.01^2,3.0^2]`) — same underlying
   discrete-local-variance quantity, different exponent convention; a
   conversion has not yet been independently re-derived and checked equal
   under both modules' own butterfly-mass normalisation.
2. `smoothness` (`arbfree`) vs `eta` (`flex`, a *multiplier on
   `forward_vol`*, not a standard deviation) are different parameters with
   no existing mapping between them.
3. The "200" in `dy2.pdf`'s abstract (the **native** affine state
   $h_t\in\R^{200}$) is not the same number as the **PCA-reduced** weekly
   dynamics state actually simulated forward in time (`STRONG_STATE_SIZES`
   sums to 135) — both are real, both are used, but at different stages of
   the pipeline; conflating them would misstate either the decoder's
   expressiveness or the dynamics model's dimensionality.
4. `flex/weekly_dynamics.py` has **no jump component anywhere** — purely
   Gaussian innovations. `dy2.pdf`'s own no-replay tail extension
   (Section 4.2.1) adds a scalar stochastic-volatility-style factor, not
   jumps either. Any Bates-oracle comparison therefore compares a
   jump-diffusion oracle against a pure-diffusion (plus one extra
   variance-of-innovation factor) fitted simulator — a genuine, not
   cosmetic, model-class difference to keep visible in any future reported
   comparison, not averaged away.
5. `hjm_roll_residuals`/`hjm_roll_moment_loss` (the codebase's own
   pathwise HJM/roll-residual machinery) is confirmed to be exactly the
   discrete form of `dy2.pdf`'s companion manuscript's
   `eq:bates-offgrid-drift` (prior task, direct derivation) — this is a
   **match**, not a mismatch, and should be the shared residual metric for
   any future comparison rather than a newly invented one (`MODEL_CONVENTIONS.md`
   Section 5).
6. `tests/test_flex_weekly_phase3.py::test_antithetic_front_audit_and_q1_only_training`
   is order-dependent flaky in the full-file run (Section 10) — a
   pre-existing baseline issue, not introduced here, but worth fixing
   (global seed isolation) before relying on that file's numbers precisely.

## 12. Deliverables index (this task)

- `REPOSITORY_MAP.md` — this file.
- `MODEL_CONVENTIONS.md` — Section B.
- `DATA_STATUS.md` — Section C.
- `quote_import.py` + `tests/test_bates_crc_quote_import.py` — the import
  adapter, Section C.
- `comparison_interface.py` + `tests/test_bates_crc_comparison_interface.py`
  — the common interface, the working `Dy2FlexAdapter` and
  `CoherentChainBlackAdapter`, the two explicit `NotImplementedAdapter`
  stubs, the comparison harness, and `DEFAULT_SPLIT`/`ChronologicalSplitConfig`
  — Section D.

## 13. Update (later task): Theorem~thm:dlv-bates / Section 12 are now implemented

Section 6's `NotImplementedAdapter("coherent_dlv_x_extended_bates_crc", ...)`
stub above is **stale** as of the "Implement the explicit DLV x
extended-Bates construction" task: `heston_riccati.py`, `reserve_crc.py`,
`bates_transform.py`, `dlv_bates_product.py` (the last exposes
`DlvBatesCrcAdapter`, satisfying `PriceModelAdapter` exactly like
`Dy2FlexAdapter`/`CoherentChainBlackAdapter`), `verification.py`, and
`AUDIT.md` implement and verify the full construction (no port of the
external `heston_bates_oracle` sandbox was needed — the Riccati/jump/reserve
machinery was re-derived and independently cross-checked directly in this
repository). See `report/DLV_Bates_CRC_Report.pdf` for the full writeup.
`default_adapters()` no longer lists a stub for this model name.

