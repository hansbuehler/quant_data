# Phase 1 audit: the existing DYSANOS FLEX statistical simulator

Scope: this audits `quantdata.dysanos.flex` (the "FLEX" subsystem) as run against
the local `QUANTDATA_TEST_MODE=1` fixture data (`tests/stored_data/`), which was
verified to reproduce the existing test suite exactly (18/18 tests pass,
`tests/test_dysanos_torch_simulator.py` + `tests/test_dysanos_market.py`, Python
3.12, fresh `.venv312` built from `pyproject.toml`). No production/WRDS credentials
or `QUANTDATA_ROOT` were available in this environment; all statements below are
either (a) read directly from source, or (b) verified by executing the existing
test-mode pipeline — never inferred from the DYSANOS II manuscript alone.

## Exact implementation identified

**Module**: `quantdata/dysanos/flex/` (not `grid/gendysanos`, `grid/rndysanos`,
`implied/`, `localvolbysanos/`, or `tangent_levy/` — those are separate,
independently-developed DYSANOS subsystems in the same package; see "Scope note"
below).

**Core files**:
- `config.py` — `FlexGeometryConfig` (grid geometry) and its subclasses
  (`CappedUnboundedDlvFlexGeometryConfig`, `FixedEtaFlexGeometryConfig`,
  `ForwardMuFlexGeometryConfig`, `SharedStrikeCappedUnboundedDlvFlexGeometryConfig`,
  `TriangularPcaDecoderConfig`), `WeeklyDynamicsConfig` (statistical transition),
  `JointHierarchicalDynamicsConfig`/`JointDirectDynamicsConfig`.
- `model.py` — `bound_native_raw` (the decoder: unconstrained latent → bounded
  native state), `NativeStateSanosModel`, `LatentStateSanosModel`,
  `JointDirectDynamicsModel`, `JointHierarchicalDynamicsModel`.
- `sanos.py` — `FlexSanosDecoder`, `SanosSurface`: the DLV→density→price map.
- `weekly_dynamics.py` — `TorchWeeklyDynamics`, `WeeklyDynamicsFit`: the fitted
  empirical (P) weekly transition law. **This is "the existing AR(1) law."**
- `weekly_measure.py` / `weekly_measure_training.py` — `NormalWeeklyMeasureKernel`,
  `GenericWeeklyMeasureKernel`, SMC-based **target-measure** kernel fitting. See
  "Important prior work" below — this is the closest existing artifact to a Q
  construction, and must be understood before anything is built "from scratch."
- `weekly_simulation.py` — `WeeklySanosSimulator`: path simulation, option
  repricing, and **`hjm_roll_residuals`** (see below — this already computes the
  literal pathwise object Section 2 of the assignment asks for).
- `fixed_expiry_ar1.py`, `last_expiry_ar1.py` — coordinatewise AR(1)/rollover
  innovation fits for the DLV tower's calendar roll (distinct from
  `weekly_dynamics.py`'s cross-expiry hierarchical AR(1); these concern the
  tower's roll-forward, not the weekly state transition).

**Checkpoint**: no trained model checkpoint is present in this git checkout except
`flex/reports/phase3-spread/feed_forward.pt` (a spread-cost regression, not the
state/dynamics model). `flex/reports/` contains several JSON/markdown experiment
write-ups (`phase3-weekly`, `phase3-encoder`, `phase3-native-encoder`,
`batch-descent/spx-2022-2025-forward-mu-v1*`) but no serialized `WeeklyDynamicsFit`
or trained `JointHierarchicalDynamicsModel` state dict. **Any numbers reported
below as "fitted" come from running the fitting code fresh against the local
fixture data in this session, not from a pre-existing checkpoint** — this is
stated explicitly per the assignment's own requirement to identify the exact
checkpoint used (there is none to load; a fresh fit on fixture data is used
instead, and is labelled as such throughout).

## Latent state and decoder

`bound_native_raw(raw, geometry)` maps an unconstrained torch tensor to
`NativeParameters(dlvsqs, forward_vols, eta)`:
- `dlvsqs`: shape `(T, N)`, the discrete-local-variance-squared coordinate at
  each of `T` model expiries and `N` inner strikes, bounded to
  `[min_dlvsq, max_dlvsq] = [0.01², 3.0²]` via a sigmoid (or a soft cap +
  barrier penalty for `CappedUnboundedDlvFlexGeometryConfig`).
- `forward_vols`: shape `(T,)`, one per-expiry scalar, bounded
  `[min_forward_vol, max_forward_vol] = [0.01, 1.0]`.
- `eta`: either a single fixed constant (`FixedEtaFlexGeometryConfig`) or a
  learned scalar bounded `[min_eta, max_eta] = [0.01, 0.50]`.

## Mapping to DLV coordinates and density/probabilities

`FlexSanosDecoder` (`sanos.py`) runs the **Wissel/Magnus finite-grid recursion**
directly: given `dlvsqs[m]` at expiry `m`, it builds the local weighted
second-difference operator (`w_plus`, `w_minus`, exactly
`K_n² Δt / (K_{n+1}-K_{n-1})` type weights) and forward-steps the discrete density
via `_sanos_step`, i.e. the same `A_m(ℓ^m) q^m = q^{m-1}` structure as
Magnus Core.tex / this session's Task 7/8 work — confirmed directly from the
weight construction, not assumed from naming alone.

## The two term structures: grid geometry vs. smoothing

These are **structurally separate** in the code:
- **Grid geometry**: `num_inner_strikes` (200 by default), `nms_range`
  (normalized-moneyness range, `(-3.0, 2.0)`), `num_model_expiries` (14),
  `first_model_business_dte`/`model_expiry_step_business_days` (the calendar
  roll schedule) — all in `FlexGeometryConfig`, fixed per run, not part of the
  fitted state.
- **Smoothing**: `eta` (see below) and `forward_vols` — **`forward_vols` is
  itself part of the fitted/decoded state**, so the "smoothing term structure"
  is not a fixed side-channel; it is jointly estimated with the DLV grid.

## Variance vs. standard deviation convention (verified, not assumed)

Traced directly through `price_first_expiry_calls_from_parameters` →
`_black_call`: the kernel's total dispersion is
`total_vol = forward_vol * eta * sqrt(dt)`, and this is passed to `_black_call`
as the parameter **literally named `sqrt_variance`**, used directly (not
squared) in `d1 = (log(S/K) + 0.5*sqrt_variance²)/sqrt_variance` — i.e. it is a
**standard deviation**, matching the standard Black convention and this
session's own `heston_bates_oracle` SANOS convention (`w = η²` there; here
`sqrt_variance = forward_vol·η·√Δt` directly). **`eta` itself is a dimensionless
multiplier on `forward_vol`**, not a standard deviation on its own — this is a
materially different role from `heston_bates_oracle`'s η (which *was* the
smoothing standard deviation directly). This distinction must be preserved
exactly when porting any Q-restriction formula between the two codebases.

## Statistical transition law and innovation covariance

`weekly_dynamics.py`'s `TorchWeeklyDynamics`/`WeeklyDynamicsFit`: a
**PCA-reduced, per-expiry-block, hierarchical VAR(1)** —
`HierarchicalBlockFit(intercept, own_lag, previous_lag)` fits each expiry
block's reduced PCA scores as an affine function of its own lag **and** the
previous (shorter-maturity) expiry block's lag, at a fixed `weekly_session_stride`
(5 business days), via ridge-regularized least squares
(`_ridge_coefficients`). A joint innovation covariance
(`weekly_joint_innovation_covariance`) and a **separate** joint log-spot AR(1)
piece (`weekly_log_spot_drift`, `weekly_spot_front_beta`,
`weekly_spot_orthogonal_std` — spot's own drift plus its regression beta onto
the front-expiry surface innovation plus an orthogonal residual std) complete
the weekly step. A continuous-time generator `K = -log(A)/Δt` is computed
**and gated** (`_verified_generator`: condition-number ceiling
`maximum_transition_condition_number=1e8`, and an explicit round-trip check
`exp(K·Δt) ≈ A` to relative tolerance `1e-8`) — i.e. the code already checks
whether its own fitted discrete-time AR(1) is consistent with *some*
continuous-time linear SDE, but this is a **pure P-side stability/consistency
diagnostic**, not a no-arbitrage check: nowhere in this file does `Q`, a
risk-neutral drift restriction, or option-price martingality appear.

## Front-expiry spot law and spot–surface dependence

The weekly log-spot innovation is `drift + front_beta · (surface innovation) +
orthogonal_std · (independent noise)` — i.e. spot is **not** independent of the
surface's own innovation; it loads on it via `weekly_spot_front_beta`, plus an
idiosyncratic residual. This is a genuine joint spot–surface P-law, fitted from
the same weekly regression, not a separately bolted-on spot process.

## Moving-grid, maturity-roll and settlement conventions

`fixed_expiry_ar1.py`/`last_expiry_ar1.py` handle the DLV **tower's** roll
(new front expiry entering, old one settling) via `exact_weekly_pairs` — a
declared calendar-exact pairing, not an approximate/interpolated roll. The
front-expiry settlement payoff used in the codebase's own audit is visible
directly in `hjm_roll_residuals`'s docstring: `relu(R-k) - c_current[1](k)`
(`R` = gross spot return over the step) — the ordinary intrinsic payoff, tested
against the *current* front-expiry price, not a barycentric/grid-pinned
substitute.

## Existing pricing-drift / price-preserving-reset constraints (the central finding)

**`WeeklySanosSimulator.hjm_roll_residuals`** (`weekly_simulation.py`) already
computes, **pathwise** (not yet as a conditional expectation), exactly the
discrete-time martingale/roll condition Section 2 of the assignment asks to
impose and test:
```
non-front, m>=2:  R * c_next[m-1](k/R) - c_current[m](k)
front (m=1):       relu(R-k) - c_current[1](k)
```
Its own docstring is explicit and correct about scope: *"Conditional
expectations of these residuals are the discrete weekly HJM/roll restrictions.
The method deliberately returns pathwise values: a trainer can impose
conditional moments without incorrectly forcing every stochastic payoff
residual to zero."* This is used by exactly one caller,
`weekly_measure_training.py`'s **`WeeklyMeasureTrainConfig`** — a **sequential
Monte Carlo (SMC) fit of a "target-measure" kernel**
(`NormalWeeklyMeasureKernel`/`GenericWeeklyMeasureKernel`, a learned
mean-shift/log-scale reweighting of the AR(1) innovations, regularized toward
`P` via `p_regularization_weight`) — i.e. **there is already a prior,
substantial attempt in this codebase at exactly the P→Q construction this
assignment asks for**, via importance-reweighting the fitted AR(1) rather than
constructing an admissible Q-family from first principles.

**What this existing mechanism does *not* establish** (audited, not assumed):
1. It is a **trained, SMC-fitted, regularized-toward-P kernel** — its output is
   a Monte Carlo estimate of a measure that approximately minimizes the roll
   residual's conditional expectation, not a closed-form admissible Q-family
   with checkable drift-restriction equations (Section 3 of the assignment).
   No rank/conditioning/residual report for a *feasibility equation* was found;
   the training loss is the thing being minimized, not a residual being
   certified zero.
2. No **consistent recalibration** step (Section 5: evolve → compute target →
   propose new parameters → constrained inverse reproducing exact prices →
   feasibility accept/reject) was found anywhere in `flex/`. `grep -l
   "recalib"` across `flex/*.py` matches only comments/filenames referencing
   "reset" and "recalib" in a handful of `run_*` scripts and
   `weekly_measure_training.py` (the SMC fit itself), none of which implement
   an exact price-preserving inverse under positivity/mass/calendar
   constraints. **This is the clearest gap between what exists and what
   Section 5 of the assignment requires.**
3. The measure-kernel's own regularization (`p_regularization_weight`) keeps it
   *close to P*, which is a sensible training stabilizer but is not itself
   evidence of Q-admissibility (equivalence, correct static-arb-free marginals,
   or exact martingality of the *decoded* option prices under the trained
   kernel) — this has not yet been verified either way in this audit and is
   the natural first question for Section 3.

**Joint estimation is not currently reported as consistent recalibration
anywhere found** — `training.py` jointly fits native/compressed state and
structural parameters to one day's book (ordinary calibration), and
`weekly_dynamics.py`/`weekly_measure_training.py` jointly fit statistical
transition and (separately) a measure-change kernel across many days — neither
is labelled, nor should be treated, as CRC.

## Important prior work already in this codebase (must be read before designing anything new)

A family of `run_*_q_existence.py` scripts
(`run_atm_q_existence.py`, `run_state_generic_q_existence.py`,
`run_vector_q_existence.py`, `run_all_state_vector_q_existence.py`,
`run_all_state_joint_generic_q_existence.py`,
`run_six_state_joint_generic_q_existence.py`,
`run_all_state_vector_smooth_q_existence.py`) already investigate **exactly**
the Section 3 question ("does the existing decoder admit the required
Q-dynamics") using W2 (Wasserstein-2) optimal-transport-based roll conditions,
against real historical joint-hierarchical fits
(`_DEFAULT_JOINT_COMPLETION` points at
`training/flex-shared-strikes/joint-hierarchical/v1/reports/joint-hierarchical-200-complete.json`,
not present in this checkout/environment — no `QUANTDATA_ROOT` was
configured). **Their result artifacts were not accessible in this
environment** (no cached `q-existence` reports found under
`flex/reports/` or the local `tests/stored_data/` fixture tree). Before
building any new Q-family construction, these scripts' methodology and (once
accessible) their conclusions should be read in full — duplicating or
contradicting unread prior work here would violate the assignment's own
explicit instruction not to "silently change the model" or reinvent existing
results.

## Reproducible baseline (preserved, not modified)

`QUANTDATA_TEST_MODE=1`, `.venv312` (Python 3.12, `pip install -e .` from
`quant_data-main/pyproject.toml`), confirmed via
`pytest tests/test_dysanos_torch_simulator.py tests/test_dysanos_market.py`:
**18/18 pass**. This exact command is the baseline reproduction point for all
comparisons below; no file in `quantdata/dysanos/flex/` outside
`flex/crc_extension/` (this new subdirectory) has been modified.

## Scope note: which DYSANOS is "the" DYSANOS

`quantdata/dysanos/` contains several largely independent subsystems beyond
`flex/`: `grid/gendysanos` and `grid/rndysanos` (a different, JAX/torch grid
construction with its own arbitrage-tracing and OCE tooling), `implied/` (a
PCA-based implied-surface simulator), `localvolbysanos/` (a real-trading P&L
attribution layer), and `tangent_levy/` (a Carmona-Nadtochiy tangent-Lévy
implementation with its own explicit `physical_measure.py`). This audit
targets `flex/` specifically, because it is the subsystem whose language
("AR(1)", "weekly", joint DLV+grid+smoothing+dynamics estimation) matches the
assignment's description exactly, and because it is the only subsystem with an
explicit, already-present pathwise HJM/roll-residual diagnostic
(`hjm_roll_residuals`) directly on point for Section 2. If a different
subsystem was intended, the mapping above would need to be redone against it;
this should be confirmed before Section 5's implementation work proceeds.
