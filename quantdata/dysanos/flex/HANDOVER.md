# FLEX handover

## Purpose and snapshot

This document transfers the FLEX fixed-weekly SANOS research project to a new
machine or agent. Read [`FLEX.md`](FLEX.md) first: it is the authoritative
model specification. This file records the implementation state, evidence,
artifact dependencies, and next actions as of 2026-08-25.

The last pushed research implementation before this handover was
`572f3e0be93cf8e0367f479318e64ea87afcae83` on `main`. Source and tests live in
`quantdata/dysanos/flex` and `tests`. Large checkpoints and numerical reports
do not belong in the Git worktree.

There was no active FLEX worker at handover. Old transient `systemd` services,
monitor heartbeats, PIDs, and mount identities are machine-local and must not
be treated as live after migration.

## One-paragraph state of the research

The current surface uses 14 fixed weekly SANOS expiries and 200 shared
spot-relative inner strikes per expiry. Full native calibration is rich enough,
but market-to-state inference generalizes poorly from 2022--2024 into 2025.
Both Direct-200 and expiry-triangular Hierarchical-200 retain most of the static
native capacity when one state is optimized directly per held-out date. The
Hierarchical-200 jointly learned matrix-exponential AR(1) is the selected
historical `P` model because it is stable and forecasts better than Direct-200.
An exact, non-collapsed, P-free native SANOS martingale transition now solves
all roll lines W2 through W14. The open problem is not existence of a native
`Q`; it is constructing a useful `Q` on a state shared with the historical
process, preferably supported by or equivalent to `P`, without relaxing the
option martingale equalities.

## Machine and repository setup

The intended WSL checkout is:

```text
~/python/quant_data
```

Use the project Python environment, historically `~/vqdata/bin/python`, rather
than the system interpreter. `quant_data` and `packages/cdxcore` are expected to
be installed editable. Follow the repository `AGENTS.md` and `STYLE_GUIDE.md`.

If the checkout is exposed through CIFS/OneDrive and Git reports dubious
ownership, verify the mount first and then add only this repository as a safe
directory. A native WSL checkout owned by the current user should not require
that exception.

All persistent paths must come from `quantdata.env.locations`, initialized in
`quantdata/__init__.py`. Do not add host literals to FLEX source. Configure
either `QUANTDATA_ROOT` for a complete standard tree or the individual values:

```text
QUANTDATA_RAW_DATA
QUANTDATA_PROCESSING_DATA
QUANTDATA_TRAINING_DATA
QUANTDATA_EXPERIMENTS
QUANTDATA_TEMP_DATA
```

The canonical persistent mount name for the migrated machine is
`/mnt/storage`; do not restore the legacy mount spelling. Prefer, for example,
`/mnt/storage/quantdata/training` through `QUANTDATA_TRAINING_DATA`; code should
still ask `quantdata.env.locations.training` for the resolved path. Raw market
data may be on a separate mount and must be configured rather than guessed.

Verify the resolved locations before importing or training models:

```bash
~/vqdata/bin/python -c "from quantdata import env; print(env.locations)"
```

Completion records intentionally retain original absolute paths for
provenance. Active FLEX loaders rebase recognized logical artifact suffixes
beneath the current `quantdata.env.locations.training` root and accept the
rebased file only when its SHA-256 is unchanged.

## Data contract

- Underlying: SPX option books from `quantdata/dysanos/market.py`, adapted by
  `quantdata/dysanos/flex/market.py`.
- Market value target: the smooth arbitrage-free `fit` already returned by the
  market pipeline.
- Observation uncertainty: real bid and ask; the main loss minimizes excess
  outside that interval plus a weak linear pull toward `fit`.
- Instruments: calls and puts, including ITM options. Zero-strike calls are a
  simulator hedge instrument, not part of the Q-existence option audit.
- Rates/forwards: removed after upstream spot normalization.
- Eligibility: market expiry no later than the last model expiry and strike in
  the per-expiry canonical NMS `[-3,2]` range.
- Date weighting: mean over quotes within each date, then mean over dates.
  Quote-rich dates do not receive more weight.
- Training: 729 usable dates through 2024 for the current shared-strike stack.
- Validation: 162 dates from 2025-01-02 through 2025-08-29.
- 2026: never read or used.

The shared-strike native universe has 891 dates and 3,569,180 eligible option
books: 2,860,776 training books and 708,404 validation books.

## Model contract

For expiry `j` and date `t`:

```text
x[t,j] : 202 constrained SANOS parameters
u[t,j] : 202 unconstrained real-support coordinates
h[t,j] : compressed real-support state used by dynamics
```

Each `x[t,j]` contains 200 DLV-squared values, one forward volatility, and one
forward `mu`. Across 14 expiries the native state has 2,828 coordinates.
Current raw DLV-squared coordinates use

```text
DLVsq_raw = 0.01^2 + exp(z)
DLVsq_pricing = min(DLVsq_raw, 3^2)
```

with a separate smooth cap penalty. The mechanical cap is not implemented with
a saturating sigmoid.

The active compressed allocation is:

```text
(23, 27, 25, 22, 19, 14, 14, 14, 7, 7, 7, 7, 7, 7)
```

which sums to 200. Direct-200 uses one global affine decoder.
Hierarchical-200 has raw block relation

```text
u[t,j] = Decoder_j(h[t,1], ..., h[t,j]).
```

The current common strike support nevertheless depends on terminal cumulative
strike volatility. Therefore raw block hierarchy is not yet exact option-price
hierarchy. This distinction is the main structural issue for sequential `Q`.

## Verified static and historical-measure results

### Static representation

| result | all quotes inside bid/ask | interpretation |
| --- | ---: | --- |
| Shared-strike native calibration, all 891 dates | 73.4281% | Full representation ceiling across the complete sample. |
| Native independent 2025 calibration | 70.8692% | Held-out-date native ceiling; no market encoder. |
| Full 2,828-state market encoder, 2025 | 26.28% | Actual train-through-2024 market-to-state validation; weak. |
| Direct-200 quote-optimal 2025 states | 65.8973% | Frozen decoder with one optimized state per date; oracle capacity. |
| Hierarchical-200 quote-optimal 2025 states | 63.8836% | Frozen decoder with one optimized state per date; oracle capacity. |

The short end is poor even under independent native calibration. Do not blame
0DTE--2DTE containment primarily on compression without a new controlled
geometry experiment.

### Historical weekly dynamics under `P`

Both 200-state models were learned jointly with their decoder and quote states.
They use stationary, non-diagonal matrix-exponential weekly laws. Spot remains
a unit-root output and is not included in the stationary surface state.

| model | weekly spectral radius | slowest half-life | 2025 1W/2W/4W RMSE | conclusion |
| --- | ---: | ---: | --- | --- |
| Direct-200 | 0.96377 | 18.78 weeks | 1.20384 / 1.20618 / 1.18662 | Stable and beats persistence, but worse than unconditional at 1W and 2W. |
| Hierarchical-200 | 0.90282 | 6.78 weeks | 1.06585 / 1.05770 / 1.06156 | Selected `P` model; beats persistence at all horizons and is close to unconditional. |

The principal Hierarchical-200 defect is under-dispersion in later-expiry
one-week innovations. This is a measured limitation, not an instability.

### Execution cost

The selected cost model is a two-hidden-layer width-64 SiLU network predicting
log normalized price half-spread from vega, square-root TTE, NMS, log strike,
signed call/put delta, and log approximate dollar option price. A post-estimate
SPX tick floor handles low-premium options. Frozen-2025 date-balanced MAE is
`0.00013639`, RMSE `0.00025765`, and R-squared `0.49975`.

## Target-measure research status

### 1. P-relative Gaussian line: valid control, not a solution

The first Gaussian W2-to-W1 kernel preserved full support and was trained with
fresh antithetic Torch paths. Validation RMSE changed from 4.064 to 4.024 bps
spot, an improvement of only 0.98%. The law was finite and non-degenerate but
did not establish the required conditional martingale accuracy. Generic,
cumulative-OCE, clipped-density, and iterated transport attempts did not yield
a robust independent-path solution and must not be presented as accepted `Q`.

### 2. Arbitrary fixed-state ATM existence: successful but narrow

`run_atm_q_existence.py` ignores the calibrated P transition. At the retained
2025-04-04 state it draws spot from the front SANOS distribution and learns a
full-support conditional law for next front state. The latest pooled fixed-
stream experiment used 24,000 fit and 24,000 disjoint validation paths.

| metric | fit | independent validation |
| --- | ---: | ---: |
| adjusted ATM roll mean | +0.108 bps spot | -1.153 bps spot |
| adjusted replication MC SE | 1.405 bps | 1.132 bps |
| raw ATM roll mean | +2.184 bps spot | +0.188 bps spot |
| raw replication MC SE | 1.421 bps | 1.599 bps |
| return standard deviation | 286.65 bps | 283.70 bps |

This passes its semantic gate and is not a zero-return construction. It proves
only that one state and one ATM option admit a simple conditional law. It does
not prove one law for multiple states, strikes, or expiries, and it does not
establish equivalence to `P`.

### 3. Exact native SANOS Q: successful feasibility anchor

For adjacent native SANOS densities,

```text
A[j] q[j] = q[j-1]
P[j] = A[j]^(-T)
```

and the tridiagonal M-matrix structure gives a row-stochastic discrete
martingale transition. Together with the independent mean-one lognormal
backbone increment, this reconstructs the full next-expiry call curve.

The audit used all 162 quote-optimal 2025 states. W2-to-W1 through W14-to-W13
all passed on 162/162 dates. Worst deterministic tower residual across the
lines was below `1.3e-12`. Every one of the 32,724 intended fixed-state ITM
option returns had positive variance; minimum relative-return standard
deviation was 1.609%.

This is the strongest result in the project. It proves a non-collapsed native
martingale roll, but it is P-free and uses a revealed discrete node plus its
natural shifted grid. It does not by itself define an equivalent change of
measure from the continuous historical `h` process.

## Artifact transfer

The logical artifact root is:

```text
quantdata.env.locations.training / "flex-shared-strikes"
```

It is not part of the source repository. At handover the complete tree was
about 485 MiB. Copying it preserves all experiments; retraining is possible but
expensive. The minimum provenance-rich current chain is:

| subtree below `flex-shared-strikes` | approximate size | purpose |
| --- | ---: | --- |
| `capacity/` | 21 MiB | Native shared-strike calibration. |
| `compressed-encoders/` | 26 MiB | Direct/Hierarchical-200 decoder sources. |
| `joint-direct/v1/` | 25 MiB | Direct-200 control and target history. |
| `joint-hierarchical/v1/` | 19 MiB | Selected P model, quote-optimal validation states, and target history. |
| `native-sanos-q/v1/` | 17 MiB | Exact W2--W14 audits and option-return variance audit. |
| `q-existence/` | less than 1 MiB | Latest fixed-state ATM Q-existence result. |

Selected immutable hashes are:

| artifact | SHA-256 |
| --- | --- |
| Hierarchical-200 checkpoint | `000c328dfac95a1e19ac06fc746d13b8d73097e4b9aac294364ee535a445f1ec` |
| Hierarchical target history | `b5a53a3a8f3001d56c9ca73975bbcb27a0f7260a75c4fa85139e5cff0fc7530f` |
| Converged 2025 quote-optimal hierarchical states | `3ee5a11da966b5fcc54cc9ae3f6da2899b57e3594104ae4a79c25713369719b9` |
| Exact native W2--W14 report | `c43897bb85dc438c00d926ce2adc283ed8fb0a92b4e028e884f9b02d3df3fdc2` |
| Pooled ATM Q-existence checkpoint | `959d6936b5db1a6d8f9524997d053e912dd22c138c68b8f403fc861a011150cb` |
| Pooled ATM Q-existence report | `dfba02690b14587bbddbcf9a81b265096cf85d96bd5da204bb0791930a347141` |

Do not infer completion from a checkpoint or report filename alone. Validate
the corresponding `*-complete.json`, its schema/status/generation, artifact
hashes, finite tensors, and semantic completion fields. Historical failure
records may coexist with a later successful completion proof; compare their
timestamps and generations rather than treating a `latest.failure.json` name
as authoritative forever.

## Code map

- `market.py`: FLEX market adaptation and eligibility.
- `config.py`, `model.py`, `sanos.py`, `training.py`: core geometry, state
  transforms, SANOS pricing, and static objectives.
- `run_shared_strikes.py`, `run_shared_strikes_capacity.py`: shared-strike
  construction and native calibration.
- `market_encoder.py`, `run_shared_strikes_full_encoder.py`: full market-to-
  native validation control.
- `joint_direct.py`, `run_shared_strikes_joint_direct.py`: jointly learned
  Direct-200 decoder and `P` law.
- `joint_hierarchical.py`, `run_shared_strikes_joint_hierarchical.py`: selected
  Hierarchical-200 decoder and `P` law.
- `weekly_simulation.py`, `weekly_measure.py`,
  `weekly_measure_training.py`: Torch-native weekly simulation and P-relative
  target-kernel experiments.
- `finite_martingale_transport.py`, `native_sanos_q.py`,
  `run_native_sanos_q_audit.py`: exact native SANOS martingale law and audits.
- `run_atm_q_existence.py`: narrow P-free fixed-state ATM existence control.
- `spread_cost.py`, `run_spread_ffw.py`: differentiable execution cost.
- `failure_reporting.py`, `pipeline_monitor.py`,
  `pipeline_monitor_watchdog.py`, `monitor_services.py`: durable failure,
  provenance, resource, and process supervision.

## Verification after migration

First verify source and environment without launching training:

```bash
cd ~/python/quant_data
git status --short --branch
~/vqdata/bin/python -c "from quantdata import env; print(env.locations)"
~/vqdata/bin/python -m py_compile quantdata/dysanos/flex/*.py
```

Then run the focused structural suite:

```bash
~/vqdata/bin/python -m pytest -q \
  tests/test_flex_joint_direct.py \
  tests/test_flex_joint_hierarchical.py \
  tests/test_flex_weekly_phase3.py \
  tests/quantdata/dysanos/flex/test_artifact_locations.py \
  tests/quantdata/dysanos/flex/test_atm_q_existence.py \
  tests/quantdata/dysanos/flex/test_finite_martingale_transport.py \
  tests/quantdata/dysanos/flex/test_native_sanos_q.py \
  tests/quantdata/dysanos/flex/test_native_sanos_q_audit.py \
  tests/test_flex_pipeline_monitor.py \
  tests/test_flex_pipeline_monitor_watchdog.py
```

GPU policy at handover was physical GPU 0 only. Before a real run, verify that
`cuda:0` means physical GPU 0 under the selected `CUDA_VISIBLE_DEVICES` and run
the small Torch/native-Q GPU tests. Do not launch training merely because CUDA
is visible.

## Ordered next steps

1. **Complete migration verification.** Resolve locations, copy the selected
   artifact chain, validate completion proofs and hashes, run the focused CPU
   tests, then run a GPU0 smoke test. Record any changed dependency versions.
2. **Make shared strikes price-triangular.** Introduce root geometry state `g`
   and add automatic Jacobian tests showing that later expiry blocks cannot
   move earlier option prices while `g` is fixed. Preserve the current decoder
   and its metrics as the control.
3. **Define the common causal state.** Include `g`, revealed discrete node `I`,
   backbone noise, and the hierarchical continuous states. Give `P` support in
   the same coordinates before discussing entropy or equivalence.
4. **Solve W2 with hard constraints.** Begin at exact native `Q`, whiten and
   SVD-reduce redundant option returns, and move toward `P` by a constrained
   homotopy or conditional minimum-entropy projection. Do not start from an
   unconstrained P-relative objective that can trade away martingality.
5. **Validate per state.** Use large fixed antithetic random numbers during the
   inner solve and equally large fresh independent paths for validation.
   Report each option's conditional mean in MC-standard-error units and verify
   strictly positive return variance, full innovation rank, support/Jacobian
   bounds, and absence of long-run explosion.
6. **Extend only after W2 passes.** Freeze the accepted first line, solve W3
   conditional on it, and continue. Stop on material degradation and retain
   the last valid line. The far-end newly created expiry requires its own tail
   rule.
7. **Optional diagnostic.** Generalize the successful one-state ATM arbitrary
   law to many states with one conditional network. This can diagnose
   learnability, but it is secondary to the common-state exact-Q program and
   cannot replace multi-strike martingale tests.

## Claims that must remain separate

- Quote-optimal held-out `h[t]` is a decoder-capacity oracle, not a market
  encoder.
- The full market encoder's weak 2025 fit is real and accepted only for
  structural research.
- Front implied-spot sampling makes spot and the current front option
  consistent under that sampled law; it does not repair later expiries.
- Exact native SANOS `Q` proves martingale feasibility in its own filtration;
  it is not yet a P-equivalent law.
- The fixed-state ATM control proves one local conditional moment, not a common
  multi-option or multi-state measure.
- Unconditional zero drift is weaker than conditional zero drift at each
  current state.
- A live PID, checkpoint filename, or absence of a process is not completion;
  use the durable completion proof and semantic validator.
