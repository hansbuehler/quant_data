# FLEX Phase 4: front-implied spot and sequential weekly Q lines

## Scope

The first target-measure experiment deliberately started with two maturities.
It uses the accepted jointly trained Hierarchical-200 decoder and weekly P law,
not the earlier sequential decoder/PCA pipeline.  The decoder, SANOS geometry,
P transition, and historic states are frozen.  The experiment:

1. verifies that sampling spot from the current front SANOS inverse CDF
   reproduces current one-week calls; and
2. learns only the first conditional state kernel under Q so that current
   two-week fixed-cash-strike calls equal expected rolled one-week calls.

After Q1 passes its semantic and measured-fit gates, the Gaussian experiment
continues one weekly line at a time. Generic/non-Gaussian measure change remains
deferred. Training states end in 2024. The independently quote-fitted 2025
`h_t` states are used only for validation. No 2026 data is read.

## Timing and martingale equations

Let `h_t=(h_t^1,...,h_t^14)` be the unconstrained, full-real-support Strong-200
state and let `C_t^j(k)` be a normalized SANOS call with `j=1` denoting the
front weekly expiry.  Rates and forwards remain absent.  The current front
surface supplies the Q spot rank:

```text
z_s ~ Normal(0,1)
U   = Phi(z_s)
R   = S[t+1]/S[t] = F^{-1}_{front,t}(U).
```

The inverse CDF is the complete SANOS front distribution: the discrete DLV
martingale mixture convolved with its mean-one lognormal backbone.  Therefore

```text
C_t^1(k) = E_Q[(R-k)^+].
```

The next restriction uses a fixed cash strike.  In spot-normalized units the
same option has next-week normalized strike `k/R`, hence

```text
C_t^2(k) = E_Q[R C_{t+1}^1(k/R) | h_t].
```

In float32, a decoded density can miss unit mass and unit mean by roughly
`1e-5`. The sampling implementation clamps only tiny negative numerical mass,
renormalizes the discrete mixture to unit mass, and rescales its support to
unit mean before inversion. Bisection supplies the forward root; an implicit
straight-through correction supplies its derivative without allowing a Newton
tail correction to change the forward draw. This prevents a probability above
the rounded mixture mass from producing an infinite spot return.

The final new state entering the rolled front price is `h_{t+1}^1`, not
`h_{t+1}^2`.  Because the decoder is triangular, the current 2W target depends
on current `(h_t^1,h_t^2)`, while the rolled 1W price depends only on next
`h_{t+1}^1` and realized spot.

## Normal Q1 kernel

The joint Hierarchical-200 P checkpoint contains a 200-dimensional stationary
surface law but no separate spot equation.  Before constructing Q, regress the
realized weekly log-spot return under P on the first block of whitened weekly
surface innovations:

```text
Delta log S = drift_P + beta_P' z_P^1 + sigma_perp z_perp.
```

This fit uses only dated states and spot returns through 2024.  It determines
the complete Gaussian spot/surface covariance.  A Cholesky factor with spot
ordered first then supplies a standard-normal **correlated spot rank** and the
conditional surface innovation.  Implied sampling applies `Phi` and the front
SANOS inverse CDF to that P-fitted spot rank.  It therefore preserves the
learned P Gaussian copula while replacing the spot marginal.  Q1 never alters
the spot rank itself.

Q1 changes only the first 23-dimensional ordered conditional surface-normal
block:

```text
context = (h_t^1, h_t^2, log R)
z_Q^1   = mean_theta(context)
          + diag(exp(log_scale_theta(context))) L_theta z_P^1.
```

`L_theta` is lower triangular with a strictly positive exponential diagonal.
Consequently the conditional Gaussian has full support and remains equivalent
to the P proposal.  All remaining standard-normal blocks are unchanged.  The
complete frozen P innovation Cholesky then maps the ordered normals into the
next 200-state vector.  This also propagates the Q1 shock into later state
blocks without allowing later shocks to affect `h_{t+1}^1`.

The regularizer is the analytic conditional Gaussian
`KL(Q1 || P1)` in these ordered standard-normal coordinates, per first-block
coordinate.  It is zero at initialization.

For line `r>1`, the same construction changes only `z_Q^r`. Its context contains
the current state through expiry `r+1`, realized spot, and the already
transformed lower-expiry innovations `(z_Q^1,...,z_Q^{r-1})`. Earlier fitted
lines are frozen. The active-line regularizer is the actual conditional
Gaussian `KL(Q_r || P_r)` given that random triangular context. The restriction
is

```text
C_t^{r+1}(k) = E_Q[R C_{t+1}^{r}(k/R) | h_t].
```

Each accepted checkpoint seeds exactly one new line in an independently
restartable checkpoint. This preserves the hierarchical factorization and
prevents a later line from silently changing an already certified restriction.

## Live Monte Carlo objective

Each optimizer update chooses origin dates with equal probability.  For every
origin it generates fresh Torch normals in exact `z,-z` antithetic pairs,
constructs spot and the Q state inside the Torch graph, decodes the next front
surface through SANOS, and differentiates the squared conditional moment:

```text
mean_paths[R C_{t+1}^1(k/R) - C_t^2(k)].
```

The loss is the mean squared conditional moment across origin dates and 11
fixed normalized strikes with log-moneyness from `-0.30` to `0.30`, plus a
small conditional KL penalty.  Paths are never cached or pregenerated.  A
separate fixed-seed antithetic audit estimates Monte Carlo standard errors;
the training batches always receive new draws.

The network width/depth, strike set, learning rate, KL weight, path counts,
and epoch count are ad-hoc feasibility choices and must not be treated as a
final Q specification.

## Results

Q1 completed on GPU0 after 400 epochs. The initial run exposed the float32
inverse-CDF tail issue described above; exact replay of the failing saved RNG
state passed after the repair, and the run resumed from its epoch-75 checkpoint.

| 2W to 1W audit | P baseline RMSE | Gaussian Q1 RMSE | relative change |
|---|---:|---:|---:|
| pre-2025 training dates | 3.281 bps spot | 3.138 bps spot | -4.37% |
| 2025 validation dates | 4.064 bps spot | 4.024 bps spot | -0.98% |

The 2025 fraction of conditional moments within three Monte Carlo standard
errors changed from `74.52%` to `74.64%`. Thus Q1 is finite, non-degenerate,
and improving, but the gain is modest. The largest residuals remain around the
central strikes rather than the near-zero-price far upper wing.

The front-week audit used 4,096 antithetic paths per origin. On 2025 dates its
call RMSE was `0.77` bps spot, `99.61%` of call moments were within three Monte
Carlo standard errors, and the largest decoded analytic mass/mean errors were
about `2.0e-5` before the sampling repair.

The sequential Gaussian continuation starts with the 3W-to-2W line. It stops
when the active line's 2025 `Q RMSE / P RMSE` ratio rises by both at least 25%
relatively and 0.10 absolutely versus the preceding line. These thresholds are
ad-hoc research controls and are recorded in every report; reaching the
requested final line or hitting this degradation gate is a successful pipeline
termination, while numerical or convergence failure is not.

## Power-exp OCE target and sampleable transport

The generic W2 experiment asks a slightly different question from the direct
Gaussian moment fit: can an OCE density tilt first define a useful equivalent
law, and can that law then be converted into a live sampler?  For each current
state `h_t`, one eleven-call action `a(h_t)` is shared across all future spot
ranks.  A separate cash translation `y(h_t,s)` is solved exactly for each spot
draw:

```text
E_P[u'(y(h_t,s) + a(h_t)' R) | h_t, S[t+1]=s] = 1,
D = u'(y(h_t,s) + a(h_t)' R).
```

Here `R` is the vector of W2-to-rolled-W1 fixed-strike price residuals and `u`
is the power-exponential utility.  The conditional normalization implies
`E_P[D | h_t,S[t+1]]=1`; consequently the OCE tilt changes the conditional
surface law but preserves the front-week implied spot marginal exactly.  At an
interior population optimum, the action first-order condition is
`E_P[D R | h_t]=0`.

The action is a three-hidden-layer, width-256 SiLU network of the current first
two expiry blocks.  Repeated corrections additionally receive current
`log(S_t)`, standardized with the pre-2025 training population mean and standard
deviation; validation data never enter this normalization.  The accepted first
correction predates this extra input and is retained exactly, while every
incremental correction is spot-conditioned.  Thus the resulting cumulative
density is conditional on `(h_t^1,h_t^2,S_t)` without changing the accepted
baseline law.  Its eleven outputs are expressed in a global inverse-square-root
second-moment basis for the call returns.  Fresh training batches use equally
weighted dates, randomized stratified spot ranks, and antithetic surface
normals.  No path set is stored.

The unconstrained population OCE was not a robust sampling target.  Two
production attempts with different learning rates developed the same signature:
roughly 40% of recent gradients hit their clip, whitened actions kept growing,
and moment error did not improve.  A small smoke audit also found a worst
effective sample size of only `2.71 / 128`.  This is objective-driven density
concentration rather than merely a bad learning rate.

The retained experiment therefore applies a fixed smooth radial budget `c` to
the action in whitened coordinates:

```text
b_c = c tanh(||b||/c) b/||b||.
```

This makes the approximation explicit: `c` indexes a W2-error versus density-
concentration frontier.  It is an ad-hoc research control, not part of the
definition of an exact equivalent martingale measure.

| whitened norm budget | audit grid | train P -> OCE RMSE | 2025 P -> OCE RMSE | worst / median ESS | outcome |
|---:|---:|---:|---:|---:|---|
| 0.5 | 16 spot x 256 surface | 17.754 -> 17.705 bps | 16.453 -> 16.344 bps | 62.32 / 254.42 | healthy but only 0.28% train improvement; rejected by the 1% gate |
| 1.0 | 64 spot x 128 surface | 6.874 -> 6.800 bps | 6.418 -> 6.077 bps | 16.10 / 125.25 | gate passed; zero contexts below 5% ESS |

The different audit grids explain the different Monte Carlo floors in the two
rows; the within-row changes are the relevant comparison.  An independent
fixed-seed audit of the accepted `c=1` target gave `6.789 -> 6.730` bps on the
729 training dates and `7.057 -> 6.679` bps on all 162 validation dates.

To sample from the OCE law, the first hierarchical surface-normal block uses
the generic triangular map

```text
F(z) = shift + scale*z + amplitude*tanh(z) + L*z,
q(z) = (1-g)z + g F(z),
```

where `L` is strictly lower triangular and `scale>|amplitude|`.  Its triangular
Jacobian is strictly positive, the map is onto the complete Euclidean support,
and it starts at the identity.  The transport minimizes reverse KL to the OCE
target using fresh paired P paths; the spot rank is passed through unchanged.

After 1,500 transport updates, every checkpoint and training record was finite.
The median reverse-KL proxy fell by about 27% from the first 250 to the last 250
updates.  On the all-date audit, the sampler-to-target moment mismatch was
`0.494` bps on training and `0.426` bps on 2025.  The minimum coordinate
derivative was `0.898` on training and `0.935` on 2025, so the learned map
retained full-support monotonicity.

A same-seed, 4,096-path 2025 audit measured the original P W2 residual at
`4.108` bps and the transported law at `3.727` bps, a `9.27%` reduction.  The
fraction inside three Monte Carlo standard errors changed from `74.35%` to
`75.03%`.  Thus this is a non-degenerate, directly sampleable improvement, not
an exact martingale law: the remaining residual is material and concentrated
around central strikes.  The front-week check remained separate and strong at
`0.743` bps with `100%` of moments inside three Monte Carlo standard errors.

The accepted artifacts are under
`/media/storage/quantdata/training/flex-shared-strikes/oce-transport-q1/`
`bounded-cap100-distill-v10`.  The retained source implementation is
`oce_transport.py` with orchestration in `run_oce_transport_q1.py`.

## Iterated W2 OCE semantics

An iterated correction must operate under the latest **sampled** law, rather
than multiply density ratios evaluated on new paths from the original P law.
Starting from the accepted transport `T1`, round `r` therefore:

1. samples equal-weight paths from the current composition `Q_r`;
2. fits one spot-conditioned OCE action on those paths;
3. fits an incremental full-support map `U_(r+1)` to that OCE target; and
4. samples the next law as `T1(U2(...U_(r+1)(Z)))`.

Every incremental action and transport receives `(h_t^1,h_t^2,log(S_t))`, with
`log(S_t)` standardized only on the pre-2025 training set.  Each training date
belongs to one fixed origin batch.  Batch `b` resets its device generator to a
documented seed `base_seed+b`, so its stratified spot ranks and antithetic
surface normals are identical every time that batch is revisited.  Parameters
change the transported paths, but the underlying random numbers do not.

Starting with the iterated corrections, the whitened action budget is a hard
radial cap rather than a tanh or sigmoid saturation.  If `b` is the raw action
and `c` the budget, training uses

```text
b_used = b min(1,c/||b||),
L_cap = lambda_cap mean[(max(||b||-c,0))^2].
```

Thus every admissible action is unchanged, an inadmissible action is
mechanically safe in the OCE calculation, and the explicit penalty supplies an
inward gradient from its uncapped value.  `lambda_cap=1` is currently ad hoc
and is recorded with the training configuration.  The older accepted first
transport remains a frozen historical source; its smooth action target is not
silently reinterpreted.

The round gate compares direct, equal-weight samples before and after the new
transport on common audit random numbers.  It checks train improvement, bounded
validation degradation, and a positive transport Jacobian.  It deliberately
has no importance-sampling ESS criterion: accepted paths are generated by the
updated transport itself.  Earlier experiments that only multiplied OCE
weights on original-law paths are diagnostic failures and are not sequential-Q
model candidates.

## Direct conditional-moment law

The next W2 experiment removes the strategy/OCE optimizer from training.  Let
`R_i` be the one-week residual between the current W2 call and the rolled W1
call at the same cash strike.  If every traded option satisfies

```text
E_Q[R_i | h_t,S_t] = 0,
```

then every predictable linear option strategy has zero conditional gain by
linearity.  Random unit-sphere portfolios would only give a noisy estimator of
the squared norm of this finite moment vector.  The implementation therefore
minimizes that norm directly.

The first feasibility stage uses only the ATM two-week call, with normalized
strike `K/S_t=1`.  Starting from the accepted Q1 sampler, a new conditional
full-support triangular map `U_theta` minimizes

```text
mean_dates[(E_Q[R_ATM | h_t,S_t] / return_scale)^2]
    + beta * KL(U_theta#N || N).
```

Dates receive equal weight.  Spot ranks remain exact draws from the current
front-week SANOS distribution, and the incremental map changes only the
relevant future surface innovation conditional on current state, current spot,
and realized next spot.  The KL is evaluated from the map's exact Jacobian, so
the model remains directly sampleable inside Torch and uses no importance
weights.  Fixed antithetic training normals and independent fixed audit normals
separate optimization from validation.  The learning rate, `beta=1e-3`, path
counts, and ATM-first restriction are ad-hoc feasibility choices.  The full
eleven-strike vector is attempted only after the scalar experiment succeeds.

The scalar fit completed 2,500 updates. Its original 4,096-path audit reduced
training conditional-moment RMSE from `11.692` to `8.246` bps spot, but the 2025
estimate was statistically inconclusive at `7.992 -> 8.030` bps. A separate
high-path audit therefore froze the fitted transport and evaluated all 162
2025 origins with 16 independent replications of the original randomized
32-spot by 128-surface design, or 65,536 direct paths per origin. Source and
candidate use common random numbers within each replication. Monte Carlo
standard errors are estimated from the sample variation of the 16 independent
replication moments; treating the individual antithetic paths as IID would be
incorrect.

The high-path audit found a genuine but incomplete out-of-sample improvement:

| 2025 ATM W2 statistic | accepted Q1 source | scalar W2 transport |
|---|---:|---:|
| conditional-moment RMSE | 5.990 bps spot | 5.364 bps spot |
| conditional-moment MAE | 4.542 bps spot | 3.631 bps spot |
| maximum absolute moment | 17.549 bps spot | 20.768 bps spot |
| origins within 3 MC standard errors | 48.77% | 64.20% |
| RMSE beyond the 3-MC-SE band | 3.549 bps spot | 2.916 bps spot |

Thus the candidate improves validation RMSE by `10.45%`, and the paired
source/candidate calculation makes clear that the change is not Monte Carlo
noise. It is not an exact scalar martingale law: 58 of 162 validation origins
remain outside three estimated Monte Carlo standard errors, and the worst raw
origin becomes worse. More training or a better conditional transport is
needed before the eleven-strike restriction is justified. The retained audit
is generated by `run_direct_moment_transport_w2_high_path_audit.py` and stored
under `direct-moment-w2-atm/v1/high-path-validation/v1` in training storage.

### Fixed-state scalar OCE control

To separate conditional-law feasibility from cross-state learning, a control
experiment freezes one validation state and one ATM W2-to-W1 call residual. A
4,096-path scan selects the state with the largest absolute candidate-law
moment. The scalar power-exp OCE action is then solved on 131,072 fresh paths
and evaluated without refitting on 131,072 independent paths. The action is one
number shared across every spot rank. Only the spotwise cash translations are
re-solved on test paths because they normalize the conditional density; they
are not additional option positions.

The selected state was `2025-04-04`. The independent test raw option moment was
`-22.964 +/- 3.659` bps spot, so shorting one normalized option has positive
expected gain at `6.28` Monte Carlo standard errors. The fitted power-exp action
was `-715.806` in raw normalized notional, or `-21.902` in return-RMS units. Its
independent-test OCE objective was `1.013 > 0`. The fitted first-order residual
is zero to numerical precision; on test it is `-7.758` bps with a combined
fit-plus-test Monte Carlo error of `5.208` bps, or `1.49` standard errors.

This is not a density-collapse artifact. The test density ranged from `0.000409`
to `8.888`, its worst conditional surface ESS fraction was `37.42%`, its median
was `53.25%`, and spotwise mass error was `1.1e-14`. Both positive and negative
option returns were present. Thus the one-state, one-option inner problem works;
the remaining scalar W2 difficulty comes from learning one conditional
transport that generalizes across the entire state space. The retained control
is generated by `run_fixed_state_scalar_oce.py` and stored under
`direct-moment-w2-atm/v1/fixed-state-scalar-oce/v1` in training storage.

### Fixed-state eleven-option OCE control

The next control keeps the same `2025-04-04` state and replaces the scalar
position by eleven strikes with log-moneyness from `-0.30` to `0.30`. An
initial all-call solve is a retained negative diagnostic: far-OTM two-week
calls have return RMS as low as `0.05` bps spot, the return second-moment
condition number was `6.18e9`, and the unconstrained optimizer built raw
notionals as large as `10.7m`. Its fitted density collapsed to one of 256
surface paths in some spot buckets and the independent OCE objective reversed
sign. That action is rejected.

The corrected experiment uses the project's tradable-option convention: ITM
calls at and below ATM and ITM puts above ATM. Put-call parity subtracts the
one-week spot return from each put residual. Because the front-week spot
marginal is fixed, the centered finite-sample spot return is used as a control
variate; this changes Monte Carlo conditioning, not the population martingale
restriction. The portfolio is charged one basis point of spot per unit raw
option notional and solved over increasing raw-notional L1 balls. The L1 cap
is an exact mechanical projection. Transaction costs are applied by proximal
soft-thresholding, with no saturating sigmoid or tanh.

Fit and independent test each use 32 replications of a 32-spot by 256-surface
design, or 262,144 paths, with disjoint fixed random streams. The reported
frontier combines the convergence-certified cap 1--300 results from generation
g4 with the separately fenced cap-1000 continuation g5:

| L1 cap | iterations | fit net OCE | test net OCE | test net mean PnL (bps spot) | test minimum ESS |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.00256 | 0.00355 | 35.51 | 100.00% |
| 3 | 15 | 0.00768 | 0.01064 | 106.54 | 99.99% |
| 10 | 49 | 0.02551 | 0.03539 | 355.13 | 99.92% |
| 30 | 140 | 0.07581 | 0.10545 | 1,065.39 | 99.32% |
| 100 | 538 | 0.24454 | 0.34341 | 3,551.31 | 93.81% |
| 300 | 2,066 | 0.67134 | 0.96848 | 10,653.94 | 69.42% |
| 1,000 | 18,404 continuation steps | 1.71843 | 2.71269 | 35,513.14 | 29.13% |

Every optimum is on its L1 boundary and allocates the entire position to one
instrument: short the ITM call with `K/S=0.941765`. The test mean after the
one-basis-point charge is positive at every cap. At cap 1,000 the independent
density remains normalized to `1.4e-14`, ranges from `1.16e-5` to `12.31`, and
has median surface ESS `52.32%`; it is concentrated but is not the one-path
collapse of the rejected all-call solve. The monotone boundary solution means
that this experiment identifies one dominant conditional pricing error rather
than an interior, diversified eleven-option portfolio.

Generation g4 deliberately failed when cap 1,000 had not converged after
10,000 steps. The generic supervisor mistakenly made one identical restart
before the changed PID was detected and stopped. The runner now raises the
existing deterministic `TrainingConvergenceError`, which the supervisor treats
as terminal. Generation g5 used a new fenced output root, validated the exact
g4 report as its warm-start provenance, converged after 18,404 further steps,
and completed with zero restarts. The retained reports are under
`fixed-state-vector-oce/itm-l1-converged-v4` and
`fixed-state-vector-oce/itm-l1-cap1000-v5` in training storage.

### Zero-cost mean-variance and entropy control

A later diagnostic removes both transaction cost and notional caps. It keeps
the same `2025-04-04` current state and generates one fixed matrix of 24,000
joint next-week spot/surface returns, followed by an independent 24,000-path
matrix with identical geometry. Each matrix contains 24 independent
randomized replications of a 20-spot by 50-surface antithetic design. This
control uses one globally normalized density over joint paths; unlike the
candidate Q construction above, it deliberately does not preserve the entire
front-week spot marginal conditionally.

The mean-variance action

```text
a_MV = Cov(R)^(-1) E[R],
D_MV = exp(-a_MV' R) / E[exp(-a_MV' R)]
```

is only the first Newton approximation to the entropy projection. It fails
decisively here: the return covariance condition number is `2.397e9`, its
fitted density has median effective sample size `3.49 / 1,000`, and the
largest fitted weighted moment is `2,096.74` bps spot. On independent paths
the largest weighted moment is `6,044.75` bps spot. The linear Gaussian
identity `E[R]-Cov(R)a_MV=0` therefore has no useful implication after applying
the actual nonlinear exponential density.

The exact entropy dual is

```text
a_ENT = argmin_a log E[exp(-a' R)].
```

Its first fit stream has no finite optimum. A separating LP with
`||portfolio||_1 <= 1` finds a portfolio whose return is at least `0.03421`
bps spot on every one of the 24,000 paths and averages `1.00168` bps. Thus
zero is outside that sampled return convex hull, and the entropy objective is
correctly unbounded below. The frozen separator is not a strict arbitrage on
the independent paths: its minimum is `-56.47` bps, although `99.342%` of the
paths are positive and its mean remains `0.8709 +/- 0.1370` bps.

As a solver control, the independent stream has no strict separator. Starting
at zero, Newton with backtracking reaches a maximum absolute fitted moment of
`8.61e-11` bps in six steps; global ESS is `13,288 / 24,000`. This proves that
the deterministic entropy optimizer itself works when the sampled convex
hull admits the restriction. Freezing that action and applying it to the
other stream nevertheless leaves a maximum absolute moment of `111.10` bps.
The action is dominated by a nearly cancelling far-put contrast
`(-15,168,+16,123)`, consistent with the severe covariance conditioning.

Neither zero-cost method therefore passes independent-path validation, so the
planned 512-state interpolation experiment is not started from this result.
The retained detailed report is under
`direct-moment-w2-atm/v1/fixed-state-entropy/paths24000-v6` in training
storage.

### Arbitrary conditional-Q existence control

The next control deliberately ignores the calibrated P transition. It asks the
narrower existence question: for one fixed current state, can *any* law for the
next front-expiry state conditional on the front-week spot draw make the ATM
two-week-to-one-week option return a non-degenerate martingale? The current
state is the retained `2025-04-04` validation state. Spot is still sampled from
the current front-week SANOS implied distribution, preserving that marginal.

Let `z_s` be the standardized sampled log spot return. The unconstrained
23-dimensional next front state is sampled as

```text
h1[t+1] = location + spot_loading * z_s + branch_direction * b + sigma * epsilon,
b in {-1,+1} with equal probability,
epsilon ~ N(0,I),
sigma = 1e-4 + softplus(raw_sigma).
```

Thus the conditional state law has full support, but it is not represented as
a density or transport relative to P. Later-expiry state blocks are irrelevant
to this one rolled ATM restriction and remain at their current values. The
training objective gives the martingale mean a one-basis-point scale and asks
the two branch-average returns to approach `-25` and `+25` bps spot. The latter
is only a splitting aid: the acceptance test is based on the unconditional
sampled returns, not on the artificial branch labels.

Fit and validation each use 24,000 paths: 24 replications of a 20-spot by
50-surface antithetic design, with disjoint fixed random streams. The empirical
gate requires absolute mean at most the larger of one basis point and two
replication-level Monte Carlo standard errors, at least 25% of returns below
`-5` bps and at least 25% above `+5` bps, and return standard deviation between
10 and 1,000 bps spot. The deliberately broad upper bound detects explosion;
it is not a volatility calibration target.

| statistic | fit paths | independent validation paths |
|---|---:|---:|
| mean ATM return (bps spot) | -0.013 | -0.995 |
| replication-level MC SE (bps spot) | 3.772 | 4.262 |
| standard deviation (bps spot) | 294.121 | 293.329 |
| fraction below -5 bps | 56.98% | 57.26% |
| fraction above +5 bps | 42.14% | 42.08% |
| negative/positive branch means (bps spot) | -24.982 / +24.956 | -25.940 / +23.949 |

Both fit and independent validation pass. Training stopped after 101 of 1,500
allowed epochs; the learned Gaussian state noise scale was `0.01244`, safely
above its `0.0001` floor. This establishes the intended local existence result:
for this state and this ATM option, a simple spot-conditional, full-support law
can satisfy the sampled martingale restriction without collapsing the return
distribution.

It does **not** establish a common Q for multiple strikes, dates, or expiries;
equivalence to the calibrated P law; dynamic consistency; or absence of
arbitrage across the full surface. Its role is to separate failure of earlier
P-relative transports from impossibility of the one-option conditional moment
itself. The retained report is under
`arbitrary-q-atm/fixed-state-mixture-v3/reports/arbitrary-q-atm.json` in
training storage.

## Exact native SANOS Q benchmark

The decisive feasibility experiment drops the empirical P transition and asks
what transition is already implied by the calibrated shared-strike SANOS
surface. This is a construction of Q itself, not yet a change of measure from
P and not an entropy projection.

For one SANOS maturity step, write its density solve as

```text
A[j] q[j] = q[j-1].
```

On the shared support, `A[j]` is a nonsingular tridiagonal M-matrix. Therefore

```text
P[j] = A[j]^(-T)
```

is non-negative and satisfies, to numerical precision,

```text
P[j] 1 = 1,
P[j] K = K,
q[j-1] P[j] = q[j].
```

Thus `P[j]` is a row-stochastic martingale transition for the discrete SANOS
node. Retaining the previous mean-one lognormal backbone and multiplying by an
independent mean-one lognormal with total volatility

```text
sqrt(backbone_total_vol[j]^2 - backbone_total_vol[j-1]^2)
```

reconstructs the complete next-expiry call curve. This proves the all-strike
tower identity for the decoded surface; it does not merely fit eleven selected
option moments. The implementation uses a banded multi-right-hand-side solve,
so constructing the complete transition is `O(N^2)` rather than `O(N^3)`.

### Filtration and exact rolled surface

Let the front discrete node be `I=i` and let the front backbone realization be
`L`. The observed spot is `S[1]=K[i] L`. If `I` is revealed in the next market
filtration, the normalized remaining law is

```text
S[2] / S[1] = (K[J] / K[i]) G,
P(J=j | I=i) = P[2][i,j].
```

The next front surface is exactly another SANOS surface on its natural support
`K/K[i]`. Its time-zero density is the atom at node `i`, its first density is
row `i` of `P[2]`, and later densities are obtained by composing the later
native transition matrices. The remaining backbone at old expiry `j` is
`sqrt(v[j]^2-v[1]^2)`. This gives a non-degenerate, P-free, dynamically
arbitrage-free roll through the current last expiry. It does not determine a
new far-end expiry, which requires a separate Q tail rule.

The revealed node must genuinely be part of the modeled market information;
pricing with it while hiding it from a modeled trader would use the wrong
filtration. A filtration-minimal control integrates `I` out using its Bayes
posterior conditional on observed spot. That law is also exact, but generally
has `O(N^2)` lognormal components and is not expected to fit one standard
`N`-node state exactly.

The first rolled expiry is 4.8 business days, whereas the old W2-to-W1 SANOS
step spans five business days. The invariant coordinate is integrated DLV,
`DLVsq * dt`, so the exact annualized first rolled DLV is the old W2 DLV times
`5/4.8`. On the 2025 validation set, one date has two coordinates that move
from the ordinary cap 9 to 9.375. The exact oracle records integrated DLV and
flags these coordinates. A separate strict-cap-9 control affects only that
date and has maximum dense-strike tower error `0.00415` bps spot; it is small
but remains an approximate control and is not silently substituted for the
exact law.

### 2025 validation

The durable audit uses all 162 quote-optimal validation states from 2025-01-02
through 2025-08-29 with one vote per date. Every maturity line receives two
independent fresh 24,000-path Monte Carlo replications per date in addition to
the deterministic identities. No P dynamics, entropy objective, or 2026 data
enter the calculation.

| roll line | dates passing | worst exact residual | minimum weighted conditional std | minimum density on two-sided rows | worst absolute MC z-score |
|---|---:|---:|---:|---:|---:|
| W2 -> W1 | 162/162 | `1.06e-12` | 1.556% | 99.911% | 2.59 |
| W3 -> W2 | 162/162 | `1.20e-12` | 1.669% | 99.607% | 2.90 |
| W4 -> W3 | 162/162 | `3.14e-13` | 1.691% | 99.482% | 3.34 |
| W5 -> W4 | 162/162 | `2.49e-13` | 1.715% | 99.334% | 2.83 |
| W6 -> W5 | 162/162 | `1.38e-13` | 1.682% | 99.189% | 2.77 |
| W7 -> W6 | 162/162 | `2.04e-13` | 1.765% | 98.947% | 3.68 |
| W8 -> W7 | 162/162 | `1.48e-13` | 1.756% | 98.759% | 2.75 |
| W9 -> W8 | 162/162 | `3.82e-13` | 1.785% | 98.688% | 3.45 |
| W10 -> W9 | 162/162 | `1.31e-13` | 1.700% | 98.562% | 2.92 |
| W11 -> W10 | 162/162 | `1.29e-13` | 1.782% | 98.376% | 2.94 |
| W12 -> W11 | 162/162 | `1.47e-13` | 1.747% | 98.190% | 2.73 |
| W13 -> W12 | 162/162 | `2.12e-13` | 1.116% | 98.139% | 2.90 |
| W14 -> W13 | 162/162 | `8.53e-13` | 0.949% | 98.048% | 3.96 |

For W2, the fresh-path rolled ATM mark is positive on 38.47--46.15% and
negative on 53.85--61.53% of paths across dates. The equal-date mean is `0.106`
bps spot, and the largest individual-date mean is 2.59 Monte Carlo standard
errors from zero. The maximum revealed-node rolled-chain mass/mean residual is
`1.16e-13`. The result is therefore both exact analytically and demonstrably
non-collapsed.

An additional option-level audit fixes each initial 2025 calibrated surface and
`S[t]=1`, then measures the one-week mark-to-market return of every old W2
option under the native Q draw. The universe contains the ITM call for
`K<=S[t]` and ITM put for `K>S[t]` at each of the 202 shared model strikes. It
does not contain the zero-strike call. Exact summation over the discrete SANOS
nodes and 64-point Gauss--Hermite integration over the front backbone cover
`162*202=32,724` fixed-state option returns. All have strictly positive
variance: relative-return standard deviation has minimum `1.609%`, median
`33.639%`, and maximum `176.013%`; no value is below `1e-6`. Absolute
mark-change standard deviation has minimum `52.61` bps spot. Thus the exact W2
law is not a trivial zero-return measure for any intended ITM grid option. The
dedicated report is `w2-option-return-variance/` under the native-Q result root.

The retained implementation is `finite_martingale_transport.py`, with the
provenance-bound runner `run_native_sanos_q_audit.py`. The W2 and W2--W14
completion proofs and detailed per-date reports are stored under
`/media/storage/quantdata/training/flex-shared-strikes/native-sanos-q/v1/`.
The reproducible strict-cap control is the fenced
`w2-validation-strict-cap-v2` report in that directory.

### Torch-native simulation interface

`native_sanos_q.py` implements the same revealed-node construction entirely in
Torch for use inside a trading or Q-training loop. It accepts explicit
categorical uniforms and backbone normals, so callers can use fresh paths,
fixed common random numbers, or antithetic samples without a pre-generated path
store. The live state retains normalized spot, the revealed integer SANOS node,
and the current weekly expiry. Each step solves only the required tridiagonal
one-hot density row, samples the successor node, and applies the independent
mean-one lognormal backbone increment. Conditional call pricing propagates the
same row through every remaining modeled expiry on the exact shifted support.

The path generator has no empirical-P dependency, dense transition inverse,
entropy weight, or far-tail fallback. It deliberately stops at the last
calibrated expiry. Conditional pricing remains differentiable through the
surface parameters after the node has been fixed. The categorical node draw is
not pathwise differentiable in its probabilities; future optimization of those
probabilities must use an explicit score-function or relaxation estimator, not
a hidden straight-through approximation. Tests cover exact one- and two-step
cash-strike towers, static arbitrage, 80,000 fresh paths, gradients, batching,
`torch.compile`, the terminal-expiry boundary, and physical GPU0 execution.

### Consequence for the hierarchical decoder

The result also exposed a separate decoder issue. The current shared-strikes
mask is triangular in raw parameter blocks, but the common grid uses the final
cumulative strike volatility. Front prices therefore have small nonzero
derivatives with respect to later `h` blocks. Exact sequential Q calibration
requires an explicit root geometry state `g`, with

```text
C[j] = Decoder_j(g, h[1], ..., h[j]),
```

where `g` owns both shared-grid scales and revealed-node recentering and is held
fixed while later lines are changed. Until this Jacobian-zero property is
implemented and tested, the native rolled surface is the exact benchmark and
projection into the existing shared-NMS `h` state must be labelled approximate.

The next research stage is to keep this exact law as a hard feasible anchor,
then move toward P by minimizing an explicitly defined conditional distance
without relaxing the tower equalities. A sensible continuation is a homotopy
from the exact native law toward a P-containing reference proposal, using hard
martingale constraints at every stage. This reverses the earlier order: risk
neutrality is never traded away merely to reduce entropy distance.

Raw KL between the current P state law and this oracle is not yet meaningful.
P is continuous in the current hierarchical `h`, whereas native Q contains a
revealed discrete node and a custom shifted grid. They are singular and use
different coordinates. First put both laws on a common causal state

```text
z = (root geometry g, revealed node I, backbone noise, h[1], ..., h[M]),
```

retain `I` under P as well as Q, and ensure P has non-degenerate innovations in
every continuous coordinate. Then solve the state-conditional minimum-entropy
martingale projection of P,

```text
dQ_lambda/dP = exp(-lambda(c)' R) / E_P[exp(-lambda(c)' R)],
E_Q[R | c] = 0,
```

after whitening and removing redundant option returns by SVD. The finite
log-sum-exp dual is convex and should be solved per state with Newton or a trust
region on a large fixed antithetic random-number set, followed by equally large
fresh-path validation. Zero outside the interior of the empirical return
convex hull is a failure, not a reason to clip weights or soften the martingale
constraint.

A proposal `(1-epsilon) P + epsilon Q_native` may be used only as a continuation
diagnostic. Because the native component is singular to P, acceptance requires
its posterior mass to vanish as `epsilon -> 0` while dual norms remain finite.
Otherwise no P-supported equivalent martingale law has been established. If
importance weights remain too concentrated, distil the feasible solution into
an invertible conditional flow and reimpose the martingale equations with a
hard implicit projection. The previous cumulative OCE weighting and density
clipping are not resumed.
