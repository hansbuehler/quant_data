# FLEX research handover

Last updated: 2026-08-31 on Sommer WSL.

## Executive status

FLEX has two distinct workstreams that must not be conflated:

1. **Historical-law (`P`) surface dynamics.** The all-history weekly and
   daily-front/weekly-tail affine-200 models are complete. Both have stable
   fitted weekly AR(1) dynamics and presentation-ready market-fit, PCA, and
   invariant-distribution diagnostics.
2. **Existence and learning of a risk-neutral law (`Q`).** This work deliberately
   does not assume that `Q` is a reweighted or locally adjusted version of the
   fitted historical law. It asks whether a sufficiently flexible conditional
   transport can directly satisfy the finite collection of one-week option-roll
   moment restrictions. Fixed-state and six-state experiments succeeded. A
   single joint law over all 891 historical states has not yet passed independent
   validation and confirmation simultaneously.

The latest all-state `Q` run is paused. Its checkpoint reports `status=training`
because it was stopped between audits, but no worker is currently expected to
be running. Do not interpret that field as evidence of a live process. Preserve
the retained checkpoints until the next objective is agreed.

All persistent artifacts are under `/home/hans/quantdata_data`; do not restore
old `/mnt/storage/quantdata` paths. Generated W&B logs and obsolete failure
snapshots were cleaned from the source repository. The completed `P` models and
paused `Q` checkpoints were not deleted.

## Completed all-history `P` models

Both models use 891 dates from 2022 through 2025-08-29, exclude the 0.2-day/AM
population, and use an affine SANOS decoder

`u_t = b + W h_t`, with `h_t in R^200`.

There is no additional nonlinear network on top of the affine decoder.

| Model | Expiry nodes (business days) | Option books | Native inside bid/ask | Affine-200 inside bid/ask |
| --- | --- | ---: | ---: | ---: |
| Weekly | 5, 10, ..., 70 | 3,566,742 | 73.2911% | 72.9744% |
| Daily-front + weekly | 1, 2, 3, 4, 5, 10, ..., 70 | 3,566,258 | 81.8431% | 80.0141% |

The weekly model extrapolates poorly before 5D. Adding explicit 1--4D nodes is
the main improvement; the 200-state affine compression is a much smaller fit
cost. The fitted dynamics are:

| Model | `rho(A)` | Slowest half-life | Invariant covariance effective rank |
| --- | ---: | ---: | ---: |
| Weekly | 0.959735 | 16.87 weeks | 193.7/200 |
| Daily-front + weekly | 0.953502 | 14.56 weeks | 193.7/200 |

Gaussian innovations are too thin-tailed relative to market state returns.
Replaying independently sampled historical one-week innovations is exact at 1W
by construction and is materially closer than the Gaussian law at 2W and 3W,
but remains lighter-tailed than the market.

Artifacts:

- Weekly:
  `/home/hans/quantdata_data/training/flex-shared-strikes/joint-direct/integer-weekly-no-am-all-history-v1`
- Daily-front + weekly:
  `/home/hans/quantdata_data/training/flex-shared-strikes/joint-direct/dte1-5-weekly-no-am-all-history-v1`
- Presentation handover: [`paper/FLEX_PRESENTATION.md`](../../../paper/FLEX_PRESENTATION.md)

## Independent `Q` question

At an observed state `(s_0, h_0)`, the front-week spot distribution is sampled
first. For each sampled future spot `s_t`, a law generates the next front
surface state `h_t^1`. For each of 11 in-the-money call/put instruments, the
one-week roll return is

`R_j = next option value_j - current option value_j`.

The target is a conditional law satisfying

`E_Q[R_j | s_0, h_0] = 0`, for all 11 instruments,

up to independently measured Monte Carlo error. This is an existence problem
for *a* risk-neutral conditional law over the modeled finite set of instruments.
It is not a claim of uniqueness, equivalence to the fitted `P` law, or a complete
continuous-strike martingale measure.

The most flexible successful state-local law is a spot-conditional triangular
transport. Let `z_s` be the standard-normal transform of the sampled spot's
probability rank under the front-week mixture. Given the current front state,
`z_s`, and independent Gaussian noise `epsilon`, the network generates

`h_t^1 = T_theta(h_0^1, z_s, epsilon)`.

The triangular monotone transform has full continuous support and no artificial
discrete branch. Its context network has depth 3 and width 256. A single-state
law has its own parameters; the joint model shares one state-conditional
transport across historical states.

## Control variate and reported errors

The spot return is a denoiser, not part of the risk-neutral constraint. For each
historical state and each instrument, each dataset separately estimates

`beta_j = Cov(R_j, S_t / S_0 - 1) / Var(S_t / S_0 - 1)`.

The reported adjusted sample mean is

`mu_hat_j = mean(R_j) - beta_j mean(S_t / S_0 - 1)`.

`beta` and the spot mean are recomputed independently in training, validation,
and confirmation. Their gradients are detached. The raw return mean and raw
variance remain mandatory diagnostics. For the earlier pass/fail criterion,
the reporting scale was

`max(SE(R_j), abs(beta_j) SE(S_t / S_0 - 1))`.

Therefore the noise-scaled statistic is

`z_j = mu_hat_j / max(SE(R_j), abs(beta_j) SE(spot return))`.

This adjustment must not be described as changing the payoff, changing the
sampled spot, or constructing a density. It only removes correlated Monte Carlo
noise from the estimator.

## `Q` experiment history and measured results

### 1. One fixed state and 11 instruments

The fixed-state vector law converged and passed independent validation using
24,000 paths. The independent raw worst absolute mean/MC-error statistic was
`1.436`, below the target of 2. This established that the finite 11-instrument
moment problem is feasible at at least one historical state.

Artifact:

- Root:
  `/home/hans/quantdata_data/training/flex-shared-strikes/q-existence/vector-fixed-state-exact-pooled-v1`
- Checkpoint SHA-256:
  `77df36bc60ba5bd5e67b706f43466e52ac4de3144e6d0ae53ffd068fd58838f7`

### 2. Five additional independent states

Five difficult states were fitted independently with the generic triangular
transport. All five completed and passed their independent validation and
confirmation gates, each using 24,000 fit, validation, and confirmation paths.
Together with the original state, this supplied six state-local teachers.

Artifact:

- `/home/hans/quantdata_data/training/flex-shared-strikes/q-existence/generic-independent-five-v1`

### 3. One joint transport over six states

A single shared conditional transport was initialized from the six state-local
teachers and trained jointly. It passed:

| Dataset | Worst absolute noise-scaled adjusted mean | Worst-10% mean |
| --- | ---: | ---: |
| Independent validation | 1.568 | 1.130 |
| Independent confirmation | 1.261 | 0.680 |

This is the clearest positive evidence that the state-local solutions can be
represented by one network rather than requiring unrelated parameters for each
state.

Artifact:

- Root:
  `/home/hans/quantdata_data/training/flex-shared-strikes/q-existence/generic-joint-six-v1`
- Checkpoint SHA-256:
  `e1a0c898f2595a136095e47fa937e293d6add3ed6e784fd54acce6a132571aa9`

### 4. One joint generic transport over all 891 states

The six-state network was distilled into an affine state initialization and
extended to all historical states. After 571 sweeps it was close but failed the
agreed simultaneous gate:

| Dataset | Worst absolute noise-scaled adjusted mean | Worst-10% mean | Gate |
| --- | ---: | ---: | --- |
| Independent validation | 1.977 | 0.810 | pass |
| Independent confirmation | 2.245 | 1.057 | fail |

The independent confirmation failure is small in aggregate but decisive for
the stated worst-constraint target. This result must be retained as a near miss,
not reported as an all-state risk-neutral solution.

Artifact:

- Root:
  `/home/hans/quantdata_data/training/flex-shared-strikes/q-existence/generic-joint-all-history-affine-init-v1`
- Checkpoint SHA-256:
  `bf7d0a91d071b360da884b100e1456732e6d301be9f86c256c541e9924039122`

An earlier all-state common-random vector audit also failed: its independent raw
worst statistic was `4.104`, while the raw worst-10% mean was `1.962`. The broad
tail was acceptable, but one or a few states remained problematic.

### 5. IID path redesign and smooth price-normalized objective

The next experiment corrected the sampling design so that every path contains
one genuinely paired spot draw and one surface-state draw. Each sweep uses
10,000 fresh paths split into ten 1,000-path cuts. Fixed validation and
confirmation datasets each use 100,000 paths split into ten non-overlapping
10,000-path blocks.

For block `b`, state `s`, and option `j`, let

`q_bsj = mu_hat_bsj / current_option_price_sj`.

The smooth objective is

`L = LSE_tau((mean_b q_bsj)^2) + 0.25 sum_r w_r LSE_tau(q_(r),sj^2)`,

where block losses are ordered from worst to best and
`w_r` is proportional to `0.5^r`. The first completed run used `tau=0.05`.
Training paths were fully refreshed each sweep. Validation and confirmation
also report exact ten-block sign-flip tests and their mutual consistency.

The loss converged numerically, but the martingale diagnostics did not:

| Dataset | Worst noise-scaled adjusted mean | Worst-10% mean |
| --- | ---: | ---: |
| Validation | 30.742 | 3.011 |
| Confirmation | 37.036 | 3.528 |

The run completed 1,465 sweeps and achieved a best validation loss of
`3.168e-6`. This is evidence that the smooth price-normalized loss can become
small while still missing the desired simultaneous statistical statement. It
is an objective-alignment/generalization problem, not evidence of arbitrage.

Artifact:

- Root:
  `/home/hans/quantdata_data/training/flex-shared-strikes/q-existence/vector-all-history-iid10k-smooth-price-v2`
- Best-validation checkpoint SHA-256:
  `c0b273fbc9dca108399097e191db0758217c5788866a195fcdacea7829663caa`

A sharper tail-focused continuation used `tau=0.0015`. It was paused after 273
sweeps. Its latest training worst/worst-10% statistics were `9.072 / 1.336`, but
fixed validation remained `29.717 / 3.032`. The discrepancy confirms that more
epochs alone are not the current answer.

Artifact:

- Root:
  `/home/hans/quantdata_data/training/flex-shared-strikes/q-existence/vector-all-history-iid10k-smooth-price-tail-v3`
- Best-validation checkpoint SHA-256:
  `eeabb22c4aee28a9b67f0abec0b65820af89e94e52364da9cf508aae455f8bbe`

## What has and has not been established

Established:

- The finite 11-instrument conditional moment problem is solvable for multiple
  individual historical states.
- One shared generic network solves the six tested states.
- A joint all-history network can get close to the two-MC-error target.
- The generic transport does not need to be derived from `P`.
- One-to-one spot/surface path sampling and independent large audits are now
  implemented.

Not established:

- No all-891-state network has passed both validation and confirmation.
- No continuous-strike or all-expiry risk-neutral measure has been proved.
- The learned law has not yet passed equivalence, density-floor, innovation-rank,
  long-horizon, or sequential-expiry gates.
- A low price-normalized smooth loss is not itself a martingale validation.

A finite collection of option moments usually leaves many stochastic degrees of
freedom, so non-uniqueness is expected. Non-existence can still arise if the
requested moments lie outside the convex hull generated by the allowed support,
if the transport is too restricted, if positivity/integrability constraints
exclude the needed law, or if the option quotes themselves imply incompatible
conditional restrictions. The current near-miss does not diagnose any of these
as a genuine arbitrage. The successful state-local fits argue against a simple
pointwise feasibility obstruction; the present failure is at the shared-network
and validation level.

## Recommended next `Q` step

Do not resume the paused tail run unchanged. First choose a training objective
that is aligned with the final simultaneous statistical gate.

Recommended sequence:

1. Start from the retained all-history generic-transport checkpoint rather than
   from the statewise affine-vector law.
2. Keep the genuine one-spot/one-surface draw, 10,000 fresh training paths, and
   fixed independent 100,000-path validation and confirmation sets.
3. Keep raw return means, raw variances, adjusted means, and worst/worst-10%
   statistics in every report.
4. Optimize a smooth maximum of squared means using a detached but statistically
   meaningful scale. Compare price normalization against a stabilized estimate
   of `max(SE(R_j), abs(beta_j) SE(spot))`; do not silently change the reporting
   criterion when changing the loss.
5. Use gradient accumulation or larger effective training samples to reduce
   sweep noise before increasing model width.
6. Require validation and confirmation to agree blockwise. Stop only when both
   independently pass the same predeclared gate.
7. If a small set of states remains, inspect its option-payoff Jacobian and the
   singular values of the moment map. That distinguishes insufficient network
   sensitivity from a potentially incompatible moment system.

## Source map and verification

Key source files:

- `run_state_generic_q_existence.py`: state-local generic triangular transport.
- `run_all_state_joint_generic_q_existence.py`: one generic transport over many
  historical states.
- `run_all_state_vector_q_existence.py`: statewise affine-vector law and control
  variate/statistical reporting.
- `run_all_state_vector_smooth_q_existence.py`: IID all-state smooth-objective
  trainer with fixed validation and confirmation blocks.
- `smooth_q_objective.py`: price-normalized LSE loss, robust block aggregation,
  sign-flip test, and validation/confirmation consistency statistic.
- `run_all_history_joint_direct.py`: all-history affine-200 `P` models.

Focused checks at the latest source commit:

```text
python -m pytest -q -p no:cacheprovider \
  tests/test_flex_direct_state.py \
  tests/test_flex_joint_direct.py \
  tests/quantdata/dysanos/flex/test_smooth_q_objective.py
```

Result: `60 passed`.

Ruff import checks and Python compilation checks passed for the changed FLEX
modules. The presentation package contains 50 slides, opens with `python-pptx`,
and passes `ZipFile.testzip()`.

## Repository and operational cautions

- Repository: `/home/hans/python/quant_data`
- Branch: `main`
- Completed implementation/presentation commit: `c1e123b`
- No push was performed.
- Two unrelated modified notebooks remain outside the FLEX commits; preserve
  them as user work.
- `paper/2026 09 02 SANOS ICCF.pre-codex-review.pptx` is an intentionally
  retained local backup and is not tracked.
- Never delete the completed `P` roots or the retained `Q` checkpoints without a
  new explicit decision about which research branches are abandoned.
