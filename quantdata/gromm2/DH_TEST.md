# GROMM2 deep-hedging experiment: code-review guide

Snapshot: **28 October 2026**. Implementation baseline: `a5d5743` on `main`,
following `581a54b` and `e543868`. This document records the latest agreed
experiment, what the committed code actually implements, and how to review
and test it. It is not a declaration that the latest experiment has completed.

**Reading this evolving review:** Sections 1-8 preserve the pre-implementation
snapshot above. Sections 9-10 record the subsequent factor-book decision and
implementation. The preceding million-path pipeline was stopped before hedge
training because it uses the superseded trading universe. Do not restart it
and interpret its results as the new experiment. The new sampler preserves
the model dynamics, rather than correcting gains by changing the measure.

## 1. Objective and latest agreed experiment

Compare the original-Q Monte Carlo expectation of the forward-start straddle

\[
H=\left|\frac{S_{20}}{S_{10}}-1\right|
\]

with learned seller and buyer indifference prices. Time is in business days,
the initial normalized spot is one, and prices, gains and the cash payoff are
in inception-spot units. There is no additional discounting in this normalized
experiment. The claim fixes on day 10 and pays on day 20.

- Generate **1,000,000 training paths and 1,000,000 independent validation
  paths**, keeping each antithetic pair within one set.
- Use the saved GROMM2 **AR(1)** Q model specified in Section 3, with eight
  Brownian substeps per day. Hedge actions are daily, not at each substep.
- Use original equal-Q weights: **no return recentering, fitted path weights,
  or empirical change of measure**.
- Train independent policies for empty, short-one-claim and long-one-claim
  portfolios. They may run in the same training loop, but must not share
  trainable parameters or optimizer state.
- Use power-exponential OCE with **risk aversion 100**, an affine-plus-deep
  action and recurrent state. Do not trade spot.
- Retain the agreed tiny option turnover cost, **0.1 bp of contemporaneous
  spot notional**, and numerical position cap/penalty. Settlement is free.
- First run with the additional day-10 ATM call rebalance permission described
  below, then without it, on the same paths and information set.
- Retain the simulator-only **zero-Black-background control** alongside the
  saved 5% background, without refitting historical DLVs or the AR(1) law.
  This changes option prices and the simulated model; it is not a fresh
  market calibration or a historical DLV-smoothness change.
- Validate throughout training and stop on sustained validation stagnation.
  There is no third final test sample by explicit user choice.

### Trading universe and timing

Let

\[
\mathcal K=\{0.95,0.98,0.99,1,1.01,1.02,1.05\}.
\]

1. **Static contracts:** strikes \(kS_0\), \(k\in\mathcal K\), with absolute
   expiries day 10 and day 20. These contracts remain the same each day and
   can be rebalanced daily strictly before their own expiry.
2. **New floating cohorts:** at each entry day \(\tau=0,\ldots,10\), quote
   new contracts with strikes \(kS_\tau\) and expiry \(\tau+10\). Their strikes
   and expiries are fixed once issued; they do not reset afterwards.
   Positions entered in these daily cohorts are **held to cash settlement**,
   not sold next day at 9 DTE or forcibly rolled into a different option.
   The latest clarification stops new floating entries **after day 10**;
   all positions therefore settle by day 20. There is no day-29 runoff.
3. **OTM side at issuance:** puts for \(k<1\), calls for \(k\geq1\), including
   an ATM **call**, not an ATM straddle. A contract does not switch side as
   spot subsequently moves.
4. **Optional day-10 call:** the call with strike \(S_{10}\) and expiry day 20
   can be rebalanced on days 10 through 19 in the "with" case. In the
   "without" case, the ordinary day-10 ATM floating cohort can still be
   entered and held, but not subsequently rebalanced.
5. **Identical contracts must be netted:** the day-0 floating cohort overlaps
   the static day-10 book. The optional call is also the day-10 floating ATM
   call. These must not become independent economic instruments with duplicate
   costs or artificial cap relief. Static-contract trading permissions remain
   available where a contract is identical to a static contract.

For a locked cohort position \(n_{\tau,k}\), its terminal trading gain is

\[
n_{\tau,k}\left[
 \phi_{\tau,k}(S_{\tau+10})-V_\tau(K_{\tau,k},\tau+10)
\right],
\qquad K_{\tau,k}=kS_\tau.
\]

The purchase premium must be financed even though the position is locked.
Settlement must be counted once, without an exit cost. Ongoing inventory
must remain visible to the policy. Neither the future settlement value nor
future strikes may enter the state used to choose the action.

## 2. What exists now, versus what remains to be built

| Component | Committed implementation / actual status |
|---|---|
| Original-Q, 20-day AR(1) scenario generator | Implemented and exercised on CPU and allocated GPU. |
| Fixed option book in existing sampler | **22 options:** 5/10/15/20-day expiries at 0.98/0.99/1/1.01/1.02, plus 0.95/1.05 at day 20. Superseded for the next experiment. |
| Optional contracts in existing million-path sampler | **Five** options born on day 10, expiring on day 20, with strikes \(S_{10}\times\{0.98,0.99,1,1.01,1.02\}\). Superseded by the single-call permission above. |
| Three independent policy networks | Implemented in `portfolio_hedging.py`; small allocated-GPU smoke test completed, not a converged million-path result. |
| Latest static 14-option plus daily locked-cohort book | **Pending:** sampler, storage, inventory/action masks, gain/cost accounting and regression tests must be extended. Existing `BornOptionPolicy` allows rebalancing throughout a contract's life; a birth mask alone does not implement locked cohorts. |
| Earlier 100k training + 100k validation result | Complete for the older 22-option book, using the earlier joint trainer with a **shared LSTM**, not the independent three-policy experiment. |
| Superseded million-path pipeline | Stopped on specification change; GPU allocation released. No million-path hedge policies trained. |
| Paper | `quantdata/gromm2/tex/gr20.tex` and `gr20.pdf`; background reference, not evidence that the latest hedge experiment ran. |

The existing `million_forward_experiment.py` is useful to review for process
supervision, but **its hard-coded book is not the latest specification**.
The README also retains descriptions of successive older experiments;
distinguish those versions rather than combining their assumptions.

## 3. Model being used

The selected saved model is **GROMM2 legacy (-3,2), affine-200, 16-factor
AR(1)**, origin **2025-08-29**. It is not GROMM v1, a Heston model, a
Mamba/LSTM diffusion backbone, or the newer rolling `(-4,3)` calibration.
The LSTM in this experiment is the **hedging policy**, not the dynamics law.

- Saved history: 891 dates from 2022-01-03 through 2025-08-29; 863 fitted
  one-session factor transitions, last origin 2025-08-28.
- Earlier joint whole-history affine calibration, not the new 252-day
  rolling-origin protocol. Saved aggregate inside-spread coverage is
  **83.5881%**; this is historical fit provenance, not a hedge-performance score.
- 200 interior strike nodes plus two boundaries. The fixed-origin tower has
  daily expiries 1 through 25 business days, then every five days through 125.
  Every maturity needed by the revised 20-day experiment is on the daily grid.
- The independent Black background has annualized volatility **5%**. It
  multiplies the latent spot; the front bridge settles the latent node, not
  necessarily the total spot on a grid strike when Black smoothing is nonzero.
- Saved DLV states, loading surfaces and AR(1) parameters are read-only.
  Barycentric transports are reconstructed in float64 from unchanged saved
  supports, with source-precision checks; this is not a refit or projection of
  probabilities. Source hashes and decoded mass/mean checks are recorded.

### Dynamics and numerical caveats

Read [`bridge.py`](bridge.py),
[`simulation.py`](simulation.py) and
[`dlv.py`](dlv.py) together.

- `advance_front_density` implements the Gaussian information bridge,
  driven by the first joint factor, including final node sampling. Its
  finite-step front transition is analytically exact, up to numerical
  evaluation; that does **not** make the stochastic DLV integrator exact.
- The AR(1) predicts bounded, calibrated amplitudes of spatial log-DLV-squared
  loading columns from past factor scores. Gaussian shocks remain standard
  normal. The AR state updates only after the day's shock has been observed.
  Loading fields are interpolated in remaining maturity as fixed expiries age.
- `risk_neutral_log_drift` recursively computes density loadings and the
  corresponding Q drift. With absolute density loading \(E_i\),
  \(\eta_i=E_i/q_i\), and log-DLV loading vector \(\xi_i\), the relative
  DLV-squared level drift is \(-\xi_i^\top\eta_i\); the log drift additionally
  subtracts \(\tfrac12\|\xi_i\|^2\). Estimated physical drift is not used.
- At each of eight substeps, the tower and Q drift are recomputed. Loadings
  are attenuated in low-density nodes using relative density floor `1e-4`.
  Large relative-level-drift increments are logarithmically tamed beyond
  absolute increment `0.5`. These safeguards are part of the implemented
  numerical construction and must be assessed, not omitted from a model review.
- The density solves preserve the intended positivity/mass/mean structure,
  but the exponential DLV step, splitting and drift taming do **not** establish
  exact finite-step martingality. Inspect saved diagnostics and unconditional
  gains before attributing a profitable hedge to a true conditional drift.
- GPU path generation uses the shared parallel-cyclic-reduction tridiagonal
  solver (`pcr`), rather than the slower Thomas scan. Solver equivalence has
  dedicated numerical tests and a retained validation artifact.

### External, read-only model inputs

These paths are on the current host and are deliberately not copied into Git.
The adapter also requires the two adjacent JSON reports. Hashes below are
the source hashes recorded in the completed scenario identity.

| File | SHA-256 |
|---|---|
| `/home/hans/quantdata_data/training/gromm2/inputs/2025-08-29-affine200/affine-200-history.h5` | `487f781cd786ceaf3461a4cfd1d8b4e8883a95b452960a912e46058bb037f492` |
| `/home/hans/quantdata_data/training/gromm2/inputs/2025-08-29-affine200/affine-200-models-rank16-final.npz` | `4e5e594187dae7cefe77eb12ae4f3eb04612c420783a1426b2efdf6fe1b73937` |
| `/home/hans/quantdata_data/training/gromm2/inputs/2025-08-29-affine200/affine-200-models-rank16-final.json` | `c1c9a8966d44ae3b041bee672ab1a64957c2d544cd33047a5898c932dd255582` |
| `/home/hans/quantdata_data/training/gromm2/factor-laws/2025-08-29-affine200-v1/factor-laws.npz` | `bbbb27d180ff621f261ca3cde23a5b0649dd14851002cdd51f25351c4b45654b` |
| `/home/hans/quantdata_data/training/gromm2/factor-laws/2025-08-29-affine200-v1/factor-laws.json` | `f86d5a8019e90e91293944fa1dd37b2d0fa2280e1af9c60483f30f144b94394b` |

## 4. Policy, optimization and price definitions

### Predictable action and recurrent state

The existing policy uses current observed/latent features, previous holdings
and the previous LSTM hidden state:

\[
\begin{split}
c_t&=(\widetilde x_t,h_t,n_{t-1}/C),\\
r_t&=a_{\rm affine}(c_t)+a_{\rm deep}(c_t),\\
n_t&=C\tanh(r_t/C)\quad\text{on permitted instruments},\\
(h_{t+1},\ell_{t+1})
 &=\operatorname{LSTM}(\widetilde x_t,n_{t-1}/C,n_t/C;h_t,\ell_t).
\end{split}
\]

Defaults: LSTM width 32; two-hidden-layer SiLU deep head of width 64; affine
and deep output layers initialized to zero. Cap \(C=10\); soft penalty on
raw actions beyond eight units, weight `1e-4`. Near-cap frequency is reported.
The economic OCE excludes this numerical penalty; checkpoint selection uses
OCE loss plus the penalty. This is a capped numerical search, not a proof
about the unrestricted optimal hedge.

State features include time, spot, known fixing after day 10, live option
prices/masks, front density, 16 spatial log-DLV projections, maturity-level
log-DLV means, past cumulative Brownian factors, Black background level,
AR(1) state and predictable factor scales. The old saved base feature width
is 360, augmented by optional-quote information in the mapped loader. This
is a compact latent representation, **not the complete DLV Markov state**.
Both with/without-option cases observe the same information. Fit feature
normalization on training paths only, reducing across path and time axes.

Locked cohort holdings, contract identities and their remaining maturities
still need to be integrated into this state/action design for the new book.

### Utility and independent objectives

For \(\lambda=100\), the implemented utility is

\[
u_\lambda(x)=
\begin{cases}
(1-e^{-\lambda x})/\lambda,&x\geq0,\\
x-\lambda x^2/2,&x<0.
\end{cases}
\]

Write \(G_\theta\) for self-financing trading gains **net of actual costs**,
and let \(s\in\{0,1,-1\}\) denote zero, short H and long H. Each policy has
its own parameters, recurrent memory, OCE translation, optimizer, scheduler
and selected checkpoint. It minimizes

\[
-\left[m_s+\mathbb E^Q u_\lambda(G_{\theta_s}-sH-m_s)\right]
+\gamma\,\mathbb E^Q[\text{numerical action penalty}].
\]

For fixed contracts the existing gain calculation is
\(\sum_{t,i}n_t^i(V_{t+1}^i-V_t^i)\), with holdings zeroed after expiry.
An intrinsic settlement mark is included once, not treated as a sale with
transaction cost. Trading costs are
\(10^{-5}\sum_{t,i} S_t|n_t^i-n_{t-1}^i|\) on valid trade dates.
Future cohort contracts require the separate entry/settlement treatment in
Section 1; do not take returns between different daily floating instruments.

### Optimization and validation

The independent trainer uses `JointTrainConfig` operational defaults:

| Setting | Value / interpretation |
|---|---|
| Optimizer | Separate Adam per portfolio, initial LR `2.5e-4`, minimum `1e-6`. |
| Training/evaluation batch | 256 / 4,096 paths; "full-sample validation" means aggregate every validation path in batches. |
| Precision | Neural weights/features float32; scenario prices, payoff and economic gain calculations retain float64. |
| Gradient clipping | Norm 5; non-finite losses/gradients are errors, not discarded observations. |
| Improvement threshold | `max(1e-7, 1e-4 * abs(best validation objective))`. |
| LR review | Halve after 25 validation-stagnation epochs; review no later than the user's 50-epoch guidance. |
| Stopping | Per-policy 50 epochs without qualifying validation improvement; restore each policy's own best state. |
| Safety ceiling | 2,000 epochs; hitting it is explicitly **not certified convergence**. |
| Baseline | Epoch-zero no-action candidate, with translation initialized using training data only. |
| Plots | Training and validation curves per portfolio, first epoch/every five epochs/final stopping state. |

Validation is explicitly used for checkpoint selection, LR scheduling and
early stopping. It is independent of training path generation, but is **not
an untouched final test set**. Published validation MC error bars must retain
that qualification. The earlier 100k result used a shared-LSTM joint trainer;
do not describe its weights as independent because the new trainer is.

### Prices and three-standard-error brackets

Let \(\rho(X)=-\sup_m\{m+\mathbb E[u_\lambda(X-m)]\}\), and after freezing
the learned policy define

\[
DH(sH)=\rho(G_{\widehat\theta_s}-sH),\qquad
p_{\rm ask}=DH(H)-DH(0),\qquad
p_{\rm bid}=DH(0)-DH(-H).
\]

Evaluation reoptimizes only the scalar OCE translation for each frozen P&L
sample. It does not refit the hedge. The Monte Carlo reference is
\(p_{\rm MC}=\mathbb E^Q H\).

For \(N=2M\) paths, form adjacent antithetic-pair score averages. The price
SE is their **sample** SD (`ddof=1`) divided by \(\sqrt M\). For risk,
the per-path OCE envelope score is
\(-\widehat m-u_\lambda(X-\widehat m)\). For bid, ask and gaps to expectation,
**difference the path scores first**, so their covariance is retained.
Report each point estimate plus/minus three SEs, and the expanded bracket

\[
[p_{\rm bid}-3\,SE_{\rm bid},\quad p_{\rm ask}+3\,SE_{\rm ask}].
\]

These are first-order, frozen-policy MC uncertainties, not training-error
bounds or a simultaneous coverage guarantee. Validation checkpoint selection
adds uncertainty not included here. Crossed estimated quotes are flagged,
not silently sorted or clipped. Finite-step Q bias, costs, cap constraints
and imperfect optimization can all separate learned prices from expectation.

## 5. Code map and recommended reading order

Paths below are relative to the repository so links work in a Git checkout.
All referenced source and tests are tracked; large inputs/checkpoints are
external. Review the implementation, not only the launcher defaults.

| File | Main review targets |
|---|---|
| [`legacy_forward_model.py`](legacy_forward_model.py) | `load_legacy_ar1_model`: source hashes, dates, saved geometry, transport precision, decoded density checks, AR-state initialization. |
| [`forward_hedging.py`](forward_hedging.py) | `ForwardScenarioConfig`, `instrument_book`, `independent_normal_block`, both pricing functions, `scenario_function`, `generate_scenarios`: timing, causality, birth/settlement, RNG namespaces and HDF5 checkpoints. |
| [`bridge.py`](bridge.py) | `front_density_volatility`, `advance_front_density`: front information process and exact settlement-node sampling. |
| [`simulation.py`](simulation.py) | `one_day_risk_neutral_step`, density localization, drift taming, independent Black process. |
| [`dlv.py`](dlv.py) | Sparse transport, tower recursion, `risk_neutral_log_drift`, absolute versus relative density loadings. |
| [`dynamics.py`](dynamics.py), [`geometry.py`](geometry.py) | Remaining-maturity interpolation and mass/mean-preserving transport. |
| [`tridiagonal.py`](tridiagonal.py), [`quantdata/jax/tridiagonal.py`](../jax/tridiagonal.py) | Thomas/PCR dispatch and shared differentiable batched solver. |
| [`scenario_statistics.py`](scenario_statistics.py) | Streaming centered pair moments, payoff price, raw/scaled daily and settlement screens. |
| [`hedging_data.py`](hedging_data.py) | Read-only contiguous HDF5 mapping, completion checks, digest cache, matched information across books. |
| [`hedging_policy.py`](hedging_policy.py) | `HedgingPolicy`, training-only normalization, utility/OCE, paired SEs, precision, checkpoint helpers and progress/resource handling. Its original spot-trading runner is **not** the latest options-only trainer. |
| [`joint_hedging.py`](joint_hedging.py) | Reused `OptionsOnlyPolicy`, `OptionsPolicyConfig`, `JointTrainConfig`; earlier shared-LSTM experiment remains here. |
| [`portfolio_hedging.py`](portfolio_hedging.py) | `BornOptionPolicy`, `IndependentHedges`, `train_independent_portfolios`, `portfolio_quote_statistics`: independent weights, loss signs, costs, stopping/resume, both quote sides. |
| [`hedging_uncertainty.py`](hedging_uncertainty.py) | OCE envelope scores and paired covariance-aware MC errors. |
| [`plot_hedging_training.py`](plot_hedging_training.py) | `plot_independent_losses`; train/validation axes and selected epochs. |
| [`million_forward_experiment.py`](million_forward_experiment.py) | Superseded six-stage book specification; still relevant for isolation of sampling/training and signal propagation. |
| [`gpu_forward_queue.py`](gpu_forward_queue.py), [`runtime.py`](runtime.py) | Allocation ownership, fail-closed GPU selection, resource checks, progress, graceful stop and release. |
| [`rolling_audit.py`](rolling_audit.py), [`rolling_calibration.py`](rolling_calibration.py) | Shared pair/resource/report utilities; separate newer rolling-model route and its fit-quality gate. |
| [`quantdata/torch/`](../torch/), [`pyproject.toml`](../../pyproject.toml) | `BaseModel`, `BaseEnvironment`, shared feed-forward layers and dependencies. |
| [`hedging_measure.py`](hedging_measure.py) | Earlier empirical-measure experiment only. It must **not** supply weights to the current equal-Q experiment. |

## 6. Existing data, results and stopped-run state

The stopped pipeline's root is

```text
/home/hans/quantdata_data/experiments/gromm2/million-forward-independent-20261028-v1
```

Inspection on 2026-10-28 found:

| Case | Training paths complete | Validation paths complete | Hedge training |
|---|---:|---:|---|
| Saved 5% Black background | 1,000,000 | 1,000,000 | Not started. |
| Zero background, no historical refit | 343,040 / 1,000,000 | Not started | Not started. |

The service `quant-data-gromm2-million-independent-20261028-v1.service`
exited following the requested checkpoint stop. Its log records a complete
path-batch interruption, not forced killing. The supervisor labels checkpoint
trust conservatively as the last periodic checkpoint, with final save not
independently verified. The HDF5 completion counter is 343,040, but this is
**not** a complete zero-background sample. Allocation `d5f61660e4a7` was released.
Preserve the data; do not overwrite its identity with a revised book.

Old complete million-path HDF5 shapes, per split:

| Dataset | Shape | Type |
|---|---|---|
| `features` | `(1000000, 20, 360)` | float32 |
| `prices`, spot then 22 fixed options | `(1000000, 21, 23)` | float64 |
| `fixing_option_prices`, old five-option book | `(1000000, 21, 5)` | float64 |
| `payoff` | `(1000000,)` | float64 |
| `diagnostics`, batch aggregates repeated for each pair | `(500000, 20, 4)` | float64 |

The diagnostics are reduced over a generated batch, then broadcast to each
antithetic-pair row in that batch. They are **not** independent pair-level
diagnostic observations; do not estimate a diagnostic SE from those repeated
values. For a partial sample, only its completed prefix is meaningful. Inspect
`generate_scenarios` and the completion counter before reducing it.
Mapped training requires complete, contiguous, uncompressed datasets; merely
having an allocated million-row shape does not make a partial sample complete.

Completed 5% control expectation, in inception-spot cash units:

| Split | Mean H | One MC SE | Mean plus/minus three MC SEs |
|---|---:|---:|---|
| Training | 0.01959304580 | 0.00002849624 | [0.01950755709, 0.01967853451] |
| Validation | 0.01961674652 | 0.00002853067 | [0.01953115451, 0.01970233854] |

Each set has 500,000 independent antithetic pairs. These expectation estimates
are for unchanged dynamics and target, but the saved quote tensors are the
**old** trading book, not the revised daily floating cohorts.

Existing unconditional screens also need review rather than assuming Q means
an exact numerical martingale. The old book has 310 live daily coordinates
including spot, and 22 inception-to-expiry option coordinates:

| Screen | Training max absolute statistic | Validation max absolute statistic |
|---|---:|---:|
| Raw daily gains | 2.9932 | 5.9459 |
| Daily gains divided by current spot | 3.0149 | 5.9459 |
| Inception-to-expiry option gains | 3.3781 | 2.6799 |

These are ordinary mean/paired-SE pointwise Gaussian screens, not familywise
claims or conditional tests. In particular, the 5.9459 maximum is not a clean
pass. No million-path conditional hedge result exists. The current summary
screens cover the fixed book, **not** the old five born options or the latest
floating cohorts. All newly tradable contracts need appropriate gain checks.

The deliberately retained earlier 100k result is
[`forward_straddle_lambda100_100k_2026_10_28.json`](results/forward_straddle_lambda100_100k_2026_10_28.json).
Its validation expectation is approximately **1.97076%** of initial spot and
its learned seller price **2.15832%**, with respective one-SE errors about
**0.00941%** and **0.00893%** of initial spot. It uses the old shared-LSTM
joint hedge and must not be reported as the pending independent/cohort result.
The shared solver validation artifact is
[`pcr_solver_validation_2026_10_27.json`](results/pcr_solver_validation_2026_10_27.json).

## 7. Testing environment and reproducible checks

### Observed host environment

Checked on 2026-10-28, using `/home/hans/vqdata/bin/python`:

| Component | Observed version |
|---|---|
| Platform | WSL2 Linux, x86-64; kernel `6.18.33.2-microsoft-standard-WSL2` |
| Python | 3.12.3 |
| NumPy / SciPy / h5py | 2.5.4 / 1.18.1 / 3.16.0 |
| Torch / compiled CUDA | 2.14.1 / 13.0 |
| JAX / jaxlib | 0.11.2 / 0.11.2 |
| Flax / Optax | 0.12.10 / 0.2.8 |
| cdxcore / filelock | 0.1.96 / 4.1.0 |
| Matplotlib / pytest / Ruff | 3.11.2 / 9.1.1 / 0.17.0 |

These are an environment snapshot, **not an exported dependency lock**.
`pyproject.toml` declares the project and `gpu` extra (`jax[cuda13]`). The
code additionally imports h5py and Matplotlib; do not assume their current
transitive installation constitutes a complete direct dependency declaration.
No package installation or GPU training was needed for this documentation task.

The following focused CPU suite ran for this review: **174 passed, one
GPU-only test skipped**, in 86.68 seconds. There was one expected repository
test-mode initialization warning. Production sampler/trainer imports did not
load any currently dirty source modules from other projects.

```bash
cd /home/hans/python/quant_data
mkdir -p /tmp/quant_data/dh-code-review-20261028-v1
export QUANTDATA_ROOT=/home/hans/quantdata_data
export PYTHONPATH=/home/hans/python/quant_data
export QUANTDATA_TEMP_DATA=/tmp/quant_data/dh-code-review-20261028-v1/quantdata
export TMPDIR=/tmp/quant_data/dh-code-review-20261028-v1
export MPLCONFIGDIR=/tmp/quant_data/dh-code-review-20261028-v1/matplotlib
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4

/home/hans/vqdata/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp="$TMPDIR/pytest" \
  tests/quantdata/gromm2/test_forward_hedging.py \
  tests/quantdata/gromm2/test_legacy_forward_model.py \
  tests/quantdata/gromm2/test_hedging_policy.py \
  tests/quantdata/gromm2/test_joint_hedging.py \
  tests/quantdata/gromm2/test_hedging_uncertainty.py \
  tests/quantdata/gromm2/test_hedging_measure.py \
  tests/quantdata/gromm2/test_hedging_data.py \
  tests/quantdata/gromm2/test_portfolio_hedging.py \
  tests/quantdata/gromm2/test_scenario_statistics.py \
  tests/quantdata/gromm2/test_million_forward_experiment.py \
  tests/quantdata/gromm2/test_simulation.py \
  tests/quantdata/gromm2/test_dlv.py \
  tests/quantdata/gromm2/test_tridiagonal.py \
  tests/quantdata/gromm2/test_sparse_transports.py \
  tests/quantdata/gromm2/test_gpu_forward_queue.py \
  tests/quantdata/sdlv/daily/test_tridiagonal_solver.py
```

These tests use small synthetic fixtures and do not need the external fitted
history or million-path arrays. The repository's `tests/conftest.py` selects
cache-only test mode; its stored-data roots are not production market inputs.
Choose a fresh project-specific temporary directory for a concurrent run;
pytest may remove a reused `--basetemp` directory.

Ruff checks for import ordering, fatal syntax errors and undefined/unused
names (`--select I,F,E9 --no-cache`) also passed on the principal sampler,
loader, hedge, uncertainty, plotting and pipeline modules and their focused
regression tests. All 30 local Markdown links were checked for existence;
every referenced source file was already tracked before this document.

The review's persistent test log is external:
`/home/hans/quantdata_data/experiments/gromm2/dh-code-review-20261028-v1/pytest.log`.
Previous allocated-GPU sampling/compiled-training smoke tests are under
`/home/hans/quantdata_data/experiments/gromm2/million-forward-pilot-20261028-v1`;
they are not convergence evidence. GPU execution was not rerun for this review.

### Test map

All files below are in [`tests/quantdata/gromm2/`](../../tests/quantdata/gromm2/),
except the shared SDLV solver test in the command above.

- `test_forward_hedging.py`: independent RNG namespaces, causal observations,
  exact intrinsic settlement, zero-background kernels and day-10 option birth.
- `test_legacy_forward_model.py`: source hashes, origin/history checks,
  unchanged states, transport precision recovery and AR(1) compatibility.
- `test_hedging_policy.py`, `test_joint_hedging.py`: utility/OCE, action timing,
  costs, gradients, recurrent parameter sharing semantics and restart behavior.
- `test_portfolio_hedging.py`: disjoint policy parameters, birth-price exclusion,
  costs, both quote sides, independent selected states and exact interrupted
  three-policy-minibatch resume on the deterministic CPU fixture.
- `test_hedging_data.py`: mmap values, matched information sets, incomplete
  sample and incompatible-layout rejection.
- `test_hedging_uncertainty.py`, `test_scenario_statistics.py`: paired score
  covariance, analytic constant-payoff checks and streaming/full-array agreement.
- `test_simulation.py`, `test_dlv.py`, `test_sparse_transports.py`,
  `test_tridiagonal.py`: density/operator/drift identities and solver checks.
- `test_million_forward_experiment.py`, `test_gpu_forward_queue.py`: old stage
  ordering, control configuration, supervisor signals and allocation behavior.
- `test_hedging_measure.py`: earlier measure-change implementation only; its
  passing tests do not authorize using that measure in the current experiment.

### Execution and checkpoint safety

Use the explicit project interpreter, working directory, `QUANTDATA_ROOT`
and `PYTHONPATH` in every detached job; validate imports/root discovery before
material work. Acquire CUDA only through `gpuctl`, never by choosing a device
independently. Use `gpuctl resources --json` before starting and at most
five-minute intervals during background inspections; honor its configured
WARN/STOP actions. The existing supervisors emit flushed progress every minute.

Sampling and training run as separate processes so JAX and Torch do not retain
both allocators' GPU caches throughout a stage. Scenario HDF5 checkpoints
advance only by complete batches and carry model/config/software identities.
Training checkpoints retain all three networks, independent best states,
optimizers, schedulers, normalization, epoch/permutation/batch offset and
NumPy/Torch/CUDA RNG state. Resume identity includes source digests and model
versions. A new book needs a new version and unique external run directory;
do not relabel old arrays or load incompatible checkpoints.

## 8. Priority review questions and acceptance gates

1. **Financial timing:** implement the latest cohort book, with free settlement,
   paid entry premiums, static daily rebalancing, no accidental daily liquidation,
   and no option birth-price gain. Test day 0, 10, 19 and 20 explicitly.
2. **Identity/netting:** represent overlaps as one economic contract, particularly
   day-0 static/floating and day-10 optional-call/floating duplicates. Costs,
   caps and trading permissions must apply to net economic positions.
3. **Causality:** expose only current quotes, known strikes/fixings and past
   latent information; locked inventory must survive to expiry. With/without
   comparisons must differ in trading permissions, not available information.
4. **Martingality versus numerics:** check gains for every newly traded contract,
   not just the old fixed book; investigate material unconditional defects.
   The exact front bridge and continuous-time Q drift alone do not certify
   an exact discrete-time risk-neutral simulator. Do not hide defects by
   enabling the rejected empirical measure adjustment.
5. **Objective and signs:** distinguish utility from risk; minimize negative
   OCE for gains minus liability. Check seller/buyer/zero subtraction, financing,
   real costs versus numerical penalties, and frozen-translation evaluation.
6. **Statistical scope:** retain antithetic pairs as independent units, score
   covariance in prices, and validation-selection caveats. Report cap hits,
   convergence reasons and crossed brackets rather than suppressing them.
7. **Reproducibility:** review configuration/version dependencies, file identity,
   partial-batch resume, mmap assumptions and RNG restoration. CPU exact-resume
   tests do not establish bitwise CUDA reproducibility after compiler changes.
8. **Launch gate:** add deterministic regression tests for the new book before
   restarting production. First run a small CPU accounting test, then an
   allocated-GPU smoke test, and only then the full million-path comparison.

Everything needed for source review and the synthetic tests is in Git. Large
market/model inputs, complete/partial path samples, checkpoints and runtime
logs intentionally remain outside Git at the documented host paths. This
snapshot does not claim those data are publicly redistributable or portable
to another machine without arranging access.

## 9. Additional factor-portfolio benchmark: daily ATM call

This specifies a **separate 1,000,000-training plus 1,000,000-validation-path
experiment**, implemented in the new pipeline described in Section 10. The
trading book comprises spot, portfolios of real vanilla options chosen to
span the model factors, and a new **floating one-business-day ATM call** each
day. No directly tradable latent X, background Y, or artificial Brownian
factor security is introduced. Option portfolios must be financed and marked
as actual underlying contract holdings, not as changes in PCA coordinates.

At each trading date t = 0,...,19:

- Quote a new call with cash strike K_t = S_t and expiry t+1.
- Choose its position using information available at t only.
- Pay its inception premium and the experiment's applicable option entry cost.
- Hold this contract to next-day cash settlement, then remove the position.
- Offer a different ATM call at the next date; do not reset the strike of the
  old contract or take a price difference between the two contracts.

For position n_t in that day's call, the trading gain before entry costs is

\[
n_t\left[(S_{t+1}-S_t)^+-C_t(S_t,t+1)\right].
\]

Expiry settlement carries no exit cost. All such calls settle by day 20,
including the call entered on day 19. The earlier restriction stopping new
**10-DTE** floating entries after day 10 does not stop these daily 1-DTE
entries. If a factor portfolio contains the same call, net its holdings with
the directly chosen call position before computing turnover, costs and caps.

This call replaces the proposed synthetic-Y trading control, not the need
to verify spanning. Compute its joint diffusion exposures alongside spot and
the option portfolios, and report loading rank, singular values and residual
factor risk. Adding it does not by itself establish completeness or exact
daily replication. At zero Black smoothing Y is constant and S=X; with a
nonzero background, separation must come from the actual option exposures.

Retain the target H, original equal-Q weights, independent zero/short/long
policies, risk aversion 100, and paired Monte Carlo uncertainty conventions
defined above. Accounting, causality and numerical-equivalence checks are
recorded in Section 10. Initial loading rank is not a certificate of rank
along every future path or of exact replication under daily trading.

## 10. Paired real-contract implementation

The new entry point is
[`paired_experiment.py`](paired_experiment.py), with
[`paired_forward.py`](paired_forward.py) for sampling,
[`market_books.py`](market_books.py) for contract identities
and factor baskets, and
[`market_hedging.py`](market_hedging.py) for execution and
the independent recurrent policies. The old million-path runner is not used.
Paper text is unchanged.

### Sampling optimizations

- The 5% and zero-background controls use one identical latent trajectory per
  path. Only their independent Black levels, cash spots, option marks and
  payoff differ. This is possible because the requested zero-background
  control does not refit the saved DLVs or AR(1) law.
- `simulation.fused_risk_neutral_increments` streams the same density/loading
  recursion by expiry, retaining only scalar drift, variance and diffusion
  increments over the full tower. It does not broadcast all factor loadings
  across every path and maturity at once. The original implementation remains
  the numerical reference when the new `factor_scales` argument is absent.
- Each day's JIT kernel receives only still-active model expiry rows; there
  are no dummy expired maturity solves. Kernels are reused across batches
  and across the independent training/validation splits.
- Latent features are stored once per split. Both books and both smoothing
  controls reuse them through read-only memory-mapped batch views. There is
  no second complete latent simulation for another hedge book.

The reference/equivalence harness is
[`benchmark_paired_forward.py`](benchmark_paired_forward.py).
At 1,024 paths, the initial two-split GPU comparison gave payoff discrepancies
below `9e-16`, common option-mark discrepancies below `7e-14`, and differences
below `1.4e-7` after normalizing float32 latent features using population SD
with floor `1e-4`. Raw float32 feature differences reached `3.8e-6`; these
are not being described as bitwise-identical features. The larger reference
implementation exhausted GPU memory at batch 4,096, so that reference is
evaluated in 1,024-path chunks when testing larger optimized batches.

### Two actual-option hedge books

**Cohort book:** 14 static 10D/20D contracts and 70 additional contracts born
on days 1-10. The day-0 10D cohort is the existing static book, not seven extra
instruments. Cohorts remain locked to expiry except for the agreed day-10 ATM
call, which can be rebalanced through day 19. If a day-10 contract exactly
duplicates a static contract pathwise, its duplicate action is disabled;
the static contract retains its trading permissions. No spot is traded in
this book. All contracts settle by day 20.

**Factor book:** spot, 15 fixed-inception PCA-mimicking option baskets and
the new 1D ATM call each day. The basket candidate pool comprises fixed
positive-strike calls/puts at inception-relative strikes 0.90 through 1.10
in increments of 0.02, with model expiries 25,30,...,125 business days.
An inception-only, column-normalized pivoted-QR selection retains a smaller
common contract subset for both background controls before fitting the
minimum-norm basket exposures. Exact contracts and weights are saved in
`factor_baskets.json`; this file is required to resume an existing sample.
The baskets use one unit of gross contract notional each. Weights are fixed
after inception, so their subsequent factor exposures can rotate; these are
**financed option portfolios**, not securities paying Gaussian factor shocks.

These are model-priced vanilla contract types usable as a real-instrument
benchmark, not a completed replay of historical listed quotes. A historical
replay must select the actual available listed strikes and calendar expiries
causally; the simulated relative strike labels alone do not prove listing or
liquidity on a market date.

The complete 231-candidate inception pilot achieved loading rank **16** with
zero background and **17** with 5% background after including spot and the
daily ATM call. The selected subset's rank and conditioning are recorded
separately in its report. Do not turn an inception rank result into a claim
of global completeness. Daily hedging, costs and finite-step Q defects remain.

The selected production pool contains **44 contracts** and retains combined
inception rank **17/17** with 5% smoothing and **16/16 active drivers** with
zero smoothing. The retained
[equivalence summary](results/paired_forward_equivalence_2026_10_28.json)
records an additional 4,096-path comparison for each split. Common option
marks differ by less than `9.3e-14`, payoff differences are at most `1.2e-15`,
and floor-normalized float32 feature differences are below `4.2e-7`.
The zero-background payoff comparison is exact in this check. Warm generation
and pricing for **both backgrounds and both books together** takes about
**24.5 seconds per 4,096 paths**; the original one-background reference takes
about 29.3 seconds for that many paths in smaller chunks. The books differ,
so these are labelled workload timings, not an identical-work speedup claim.

Both books observe the same causal feature view, including current contract
marks, known birth strikes, live masks and compact latent information.
The factor book charges the 0.1 bp option cost on **net underlying contract
turnover**, including liquidation of its still-live baskets at day 20. It
does not charge a separate fictitious cost for each PCA label. Daily call
settlement is free. Spot cost is explicitly configurable and currently zero
for this factor-spanning diagnostic. Raw head caps and net underlying option
caps are ten units, with soft-cap penalties and cap-frequency reporting.

### Estimation, outputs and validation

The three policies have separate LSTMs, affine/deep heads, translations,
optimizers, schedulers and selected states. They are trained on the same
minibatch sequence, which is equivalent to optimizing the sum of their
independent objectives; it does not share parameters. Training batch size
remains **256** regardless of the generation batch size. Validation plateau
stopping and learning-rate review remain as described in Section 4; production
training is not stopped at the two-epoch smoke-test ceiling.

The reports include `DH_sum = DH(H)+DH(-H)+DH(0)`, each separate risk, the
seller ask and buyer bid, gaps to Q expectation, and paired three-SE bounds.
Loss plots show training/validation curves for the sum and all three subtasks.
Both background controls are explicitly labelled, including the zero case.
Generation also records unconditional spot-scaled gain screens for the real
contract book and for basket/spot/daily-call gains. Birth and post-expiry
coordinates are excluded from the fixed-contract screens.

New regression coverage is in
[`test_fused_simulation.py`](../../tests/quantdata/gromm2/test_fused_simulation.py),
[`test_market_hedging.py`](../../tests/quantdata/gromm2/test_market_hedging.py) and
[`test_paired_forward.py`](../../tests/quantdata/gromm2/test_paired_forward.py).
Tests cover the streamed versus reference drift, cash settlement, locked
holdings, birth-price exclusion, exact-contract netting, matched information,
free daily-call settlement, real-leg costs, shared sampling, mapped storage
and source-identity rejection. The release focused CPU run passed **176 tests**
with one allocation-dependent GPU test skipped. Both book implementations
also completed two-epoch compiled GPU smoke tests on 2,048 training and 2,048
validation paths; these are execution checks, **not converged hedge results**.

The new external run root is
`/home/hans/quantdata_data/experiments/gromm2/paired-real-books-20261028-v1`.
It contains equivalence reports, test logs, execution plan/status, completed
batch counters, generation summaries, and the eventual independent-policy
checkpoints and quotes. The full population price/hedge results must be read
from that run's completion reports, not inferred from its pilot or from the
older stopped pipeline.

Production is launched under
`quant-data-gromm2-paired-real-books-20261028-v1.service`, with generation batch
4,096 and training batch 256. It first completes the million-path training
and independent million-path validation samples, then runs the zero-background
factor/cohort hedges and the 5%-background factor/cohort hedges. Every case
fits all three independent policies. The status and price files are
`execution_status.json`, `train_progress.json`, `validation_progress.json`,
`train_report.json`, `validation_report.json` and the per-model `quotes.json`.
The GPU supervisor releases its allocation at completion. No full-population
DH prices were available when this implementation note was written.

### 10.7 Frozen-policy transaction-cost comparison

[`hedging_cost_comparison.py`](hedging_cost_comparison.py)
evaluates the same saved best policy for each of the zero, short-claim and
long-claim portfolios both with and without transaction costs. For each path,
it adds the actual accumulated costs back to net trading gains before
computing the power-exponential OCE. It does not subtract an average cost
from an already computed risk measure. Holdings, neural weights, normalization,
checkpoint selection and equally weighted Q paths remain unchanged. Only the
scalar evaluation translation is optimized, as required by the OCE definition.

Reports contain both sets of bid/ask quotes, individual portfolio risks,
the combined DH sum, and covariance-aware antithetic-pair Monte Carlo errors.
This is a cost-removed evaluation of cost-trained strategies, not a newly
optimized frictionless hedge. An independent CPU worker snapshots the current
checkpoint without interrupting GPU training, then `--follow-finals` evaluates
each of the four finalized book/background cases when its quotes are ready.
The snapshot records its checkpoint hash and each portfolio's selected epoch.
Outputs live below the external run's `cost-comparisons/` directory.

The retained [frozen PCA cost-comparison snapshot](results/pca_hedge_cost_removal_snapshot_2026_10_28.json)
records the million-path training/validation prices and three-SE bounds quoted
in this thread. These are provisional paused PCA checkpoints under the original
spot-notional costs, not converged results from the subsequent unit-L1 book.

Focused tests are in
[`test_hedging_cost_comparison.py`](../../tests/quantdata/gromm2/test_hedging_cost_comparison.py).

### 10.8 Unit-turnover normalizer for subsequent runs

The alternative-book run and future PCA runs use a pure L1 turnover normalizer
with coefficient `0.1e-4`, without multiplying by current spot. For the
alternative book this is charged per traded option contract; cash settlement
at expiry remains free. For PCA runs this is charged per basket, not on the
net underlying option legs. New daily calls incur one entry charge and no
expiry charge. The still-live PCA baskets incur terminal liquidation turnover.
Spot's normalizer remains zero by default.

`UnitTurnoverBookConfig` and `UnitTurnoverBookPolicy` preserve these conventions
under a new checkpoint identity and a `-unit-l1` label suffix. Archived PCA
checkpoints and their reported prices remain unchanged. The original
`MarketBookConfig` and `MarketBookPolicy` retain the old spot-notional costs;
use `--legacy-spot-costs` explicitly to load or resume them. Frozen cost-removal
evaluation supports both identities. The early alternative-book attempt with
the old costs is retained separately, not mixed with the new loss history.

The lightweight `hedging_price_monitor.py` worker reads saved validation
histories once per minute, combining each of the three independently selected
checkpoints to track bid, ask and the unchanged Q expectation. It flags a
crossed book or a Q expectation outside the point quotes without treating that
alone as a significance test. It does not read path arrays or run policies.
Interim epoch histories have no bid/ask MC errors; those are added from the
final frozen-policy quote report when available. Q's three-SE interval is
available throughout. Changes are retained in `price-monitor/history.jsonl`
and the current snapshot in `price-monitor/latest.json` under the active
run's control directory. Monitoring stops when that GPU job ends.
