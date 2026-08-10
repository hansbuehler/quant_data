# OCE Risk-Neutral Measure Change

## Operating perimeter

1. Only change files under `quantdata/dysanos/grid/rndysanos/` and `quantdata/tests/`. Ask explicitly before changing any file outside this directory.
2. Use `prep_experiment` pf `quantdata/dysanos/grid/rndysanos/torch_prep.py` to load or generate market-data universes. `prep_experiment` is cached, so it is fast.
3. Follow the repository-wide `STYLE_GUIDE.md`.
4. Use `quantdata/torch/ffw.py` for Torch feed-forward networks.
5. Code in `quantdata/dysanos/grid/` may be consulted for preprocessing and established conventions. In particular, `quantdata/dysanos/grid/torch_drift_changeV2.py` is the previous implementation and may be used as a reference, but do not modify it.
6. The target machine has 1 TB RAM and four 96 GB NVIDIA RTX PRO 6000 Blackwell GPUs. Acquire a GPU through `DeviceLock` from `quantdata/torch/__init__.py`. Prefer a clear single-GPU implementation unless profiling demonstrates that multi-GPU complexity is worthwhile.

## Data

`prep_experiment()` returns a list containing `RNDysanosExperiment.cnt_universe` independently simulated universes. The target experiment uses $S=100{,}000$ paths. Actual dimensions are controlled by `RNDysanosExperiment`; do not hard-code them in model logic.

For the standard target experiment:

- $TAU=9$ return periods;
- $T=7$ floating expiries;
- $N=20$ relative strikes per expiry;
- $XS=21$ extended-state variables ordered as $xstate=(s,h)$, where $s=\log S$ is the first coordinate and $h=(h^1,\ldots,h^{n_h})$ contains the 20 implied-volatility parameters;
- $TNX=1+T N=141$ instruments per time step.

All instruments and returns are determined by the current state. Option returns
hold the cash strike and contract maturity date fixed. For the standard
one-day return this means

$$
dC_{r,t,\omega}=C(K_{r,t,\omega},T_{t,\omega}-dt;z_{r,t+1})
-C(K_{r,t,\omega},T_{t,\omega};z_{r,t}),
$$

where the next-state normalized strike is
$K_{r,t,\omega}/S_{r,t+1}$. The dedicated generated-path experiment also
retains an `expiry` horizon for explicit hold-to-expiry controls; do not use
that payoff as a substitute for one-day `dC`.

Nominal universe fields are:

| Field | Shape | Typical dtype |
|---|---:|---:|
| `C` | `(S, TAU, T, N)` | `float32` |
| `dC` | `(S, TAU, T, N)` | `float32` |
| `dS` | `(S, TAU)` | `float32` |
| `option_dS` or `expiry_dS` | `(S, TAU, T)` | `float32` |
| `delta` | `(S, TAU, T, N)` | `float32` |
| `expiries` | `(1, 1, T)` | `float32` |
| `log_strikes` | `(S, TAU, T, N)` | `float32` |
| `spot` | `(S, TAU)` | `float32` |

Validate every actual shape and dtype rather than relying on this table.

### Returns and values

Delta-hedged option returns and gross traded notionals are defined elementwise by

$$
d\widetilde C=dC-\delta\,dS^{option},
\qquad
G_C=|C|+|\delta S|.
$$

Here $dS^{option}$ must match the option-return horizon: $S_{t+1}-S_t$ for
one-day repricing and $S_{T_\omega}-S_t$ for a hold-to-expiry control.

The option and hedge legs must not cancel when computing proportional costs.
With a common proportional rate $\delta_q$, one unit of the delta-hedged
option therefore costs $\delta_q(|C|+|\delta S|)$ to trade.

`DysanosConfig` must support `delta_hedged` and `cash_returns`. Define the instrument return vector $dH$ and value vector $H$ as follows:

| `delta_hedged` | `cash_returns` | $dH$ | $H$ |
|---|---|---|---|
| false | false | $(dS,dC)$ | $(S,C)$ |
| true | false | $(dS,d\widetilde C)$ | $(S,G_C)$ |
| false | true | $(dS/S,dC/C)$ | $(1,1)$ |
| true | true | $(dS/S,d\widetilde C/G_C)$ | $(1,1)$ |

Both vectors $dH$ and $H$ have final dimension $TNX$.
Raw spot and option values must be positive. Validate with an exception against that.

## Utility

Use the existing `torch_utility()` implementation in `quantdata/torch/utility.py`; do not duplicate it. The requested power-exponential utility is

$$
u(x)=
\begin{cases}
(1-\exp(-\lambda x))/\lambda, & x\ge0,\\
x-\lambda(-x)^2/2, & x<0.
\end{cases}
$$

`lmbda` is a strictly positive `DysanosConfig` parameter with default 1.
The existing Torch utility callable has the interface
`utility(pnl, y) = u(pnl+y)-y`.  This method has no cash offset: always pass a
scalar zero as `y`, so the callable returns `u(pnl)`.

For CVaR, do not learn a cash offset and do not directly optimize the
nonpositive function $(1+\lambda)\min(x,0)$, whose no-offset optimum would be
the zero strategy.  Instead set $\alpha=1/(1+\lambda)$ and mechanically
maximize the weighted empirical mean of the lowest $\alpha$ fraction of path
P&Ls.  Thus $\lambda=1/9$ retains the worst 90% of previous-measure mass.
During stochastic training select the tail within each minibatch.  At full
training valuation, select the tail across the entire training split, use a
common fractional weight for ties at the empirical cutoff, and freeze that
cutoff and boundary weight for validation and the returned density.  This
prevents validation leakage and makes the realized density a function of each
path plus training calibration, rather than of its validation batch peers.

For an entropy-utility warm start with risk aversion $\lambda$, use the
diagonal conditional-Gaussian approximation. If
$\mu_\omega(z)=E[dH_\omega\mid z]$ and
$v_\omega(z)=Var(dH_\omega\mid z)$, the single-instrument certainty equivalent
with transaction cost is
$$
a_\omega\mu_\omega-|a_\omega|c_\omega
-\frac{\lambda}{2}a_\omega^2v_\omega.
$$
Thus initialize the action network toward
$$
a^{MV}_\omega(z)=
\frac{sign(\mu_\omega(z))\max(|\mu_\omega(z)|-c_\omega(z),0)}
{\lambda\max(v_\omega(z),v_{floor})}.
$$
Clip this supervised target just inside the current per-instrument action cap,
fit only the action network (not the cap), then start ordinary entropy-OCE
optimization. Label this explicitly as a diagonal approximation: the exact
Gaussian solution is $\Sigma(z)^{-1}\mu(z)/\lambda$ (with the corresponding
multivariate cost subgradient), while the current moment estimator does not
learn cross-instrument conditional covariances.

## Task: OCE measure change

Let $P$ be the simulation measure and index a Monte Carlo realization by $r$
and time by $t$. The current "full" state is
$$
z^\mathrm{full}_{r,t}=(xstate_{r,t},\ log\_strikes_{r,t},\ expiries).
$$
Here the implementation array $xstate$ is ordered as $(s,h)$; the symbol $h$
denotes only the implied-volatility parameters and does not include log-spot.
The "raw" state for all options jointly is in theory just ``xstate`` since log-strikes are computed by the universe
engine off the extended state, and the expiries are constant:
$$
z^\mathrm{raw}_{r,t}:=xstate_{r,t}
$$
For a single instrument $\omega$ the correct state need to include either log-strike/expiry or
the instrument ID
$$
z^\mathrm{full}_{r,t,\omega}:=(xstate_{r,t},\ log\_strikes_{r,t,\omega},\ expiries_{\omega})
\qquad\mathrm{and}\qquad
z^\mathrm{raw}_{r,t,\omega}:=(xstate_{r,t},\ \omega)
$$

The action and conditional-moment networks are functions of $z$.  There is no
cash-offset parameter.  The realized density increment depends on the full
sequence of transition returns $dH_{r,0:TAU}$.

Networks can explode in this setting. For networks depending on $z$, use a full `FeedForwardConfig` network followed by `tanh`, then learn a single positive scale through `softplus`. The action network in `torch_drift_changeV2.py` illustrates this pattern. Check all model outputs for finiteness.

### Density semantics

Distinguish these objects explicitly in names, documentation, and outputs:

1. the previous terminal path/reference density $D^{prev}_r$ (as a
   non-trainable torch module);
2. the new terminal-path marginal-utility density increment;

Apply every density exactly once. Normalize densities where mathematically required and verify that they are finite, nonnegative, and normalized. Use log space when multiplying density increments over many time steps.

At the first measure-change stage, $D^{prev}_r=1$. At later stages,
$D^{prev}_r$ is the terminal path density returned by the previous stage and
is evaluated from the complete path $(z_{r,0:TAU},dH_{r,0:TAU},H_{r,0:TAU})$.

### Conditional moments

The paths are simulated under $P$. For every instrument $\omega$ define the random variable whose Monte Carlo mean is estimated by

$$
Y_{r,t,\omega}=D^{prev}_r\, dH_{r,t,\omega}.
$$

Compute its unconditional sampling variance directly from all full-training observations:

$$
\bar v_\omega=E_P\left[(Y_\omega-E_P[Y_\omega])^2\right].
$$

Do not learn the unconditional variance  $\bar v_\omega$ with a network.

If ``cost_cond`` $\delta_c$ is not zero, then for the conditional sampling variance, learn the two conditional moments
$$
DRIFT_\omega(z)=E_P[Y_\omega\mid z]
=E_P[D^{prev}_r dH_\omega\mid z],
$$

$$
SQR_\omega(z)=E_P[Y_\omega^2\mid z]
=E_P[(D^{prev}_r dH_\omega)^2\mid z].
$$

Calculate

$$
v_\omega(z)=\max\left(SQR_\omega(z)-DRIFT_\omega(z)^2,0\right).
$$

This is the sampling variance under $P$ of the density-weighted estimator, not $\operatorname{Var}_{P^{prev}}(dH_\omega\mid z)$. The square in $(D^{prev}_r dH_{r,t,\omega})^2$ is therefore required.

The conditional-moment targets must not retain an accidental autograd connection. Report separate training and validation losses for $DRIFT_\omega$ and $SQR_\omega$, together with conditional-variance calibration diagnostics. Pass $DRIFT_\omega$, $SQR_\omega$, and $v_\omega$ through valuation batching, trackers, checkpoint serialization, and visualization for both the full training and validation sets. Do not replace $v_\omega(z)$ with the unconditional variance or with the variance of the current minibatch.

### Two-phase training per universe

For each independently simulated universe, training has two sequential phases:

1. If `cost_cond` is nonzero, fit the conditional-moment model for
   $DRIFT_\omega(z)$ and $SQR_\omega(z)$ on that universe's training split.
   Use the validation split only to report moment-model validation losses and
   calibration diagnostics. Complete this phase before training the main utility
   model.
2. Freeze the fitted conditional-moment model, calculate
   $v_\omega(z)=\max(SQR_\omega(z)-DRIFT_\omega(z)^2,0)$ from its frozen
   outputs, and train the main action expected-utility model. The conditional-moment
   parameters must receive no gradients or optimizer updates during this
   phase.

Do not train the conditional-moment model and the main utility model concurrently.
Applying `detach`/`stop_gradient` between them prevents gradients from the OCE
objective entering the moment model, but it does not make concurrent training
equivalent: the estimated variance and therefore the OCE cost would continue
to change while the main objective is being optimized. A moment-model warm-up
followed by freezing and main-model training is acceptable, but it is the same
two-phase procedure and the main-model optimizer must not step before the
moment model is frozen.

When `cost_cond` is zero, skip the first phase and do not construct or train a
conditional-moment model. Compute the unconditional variance $\bar v_\omega$
directly from the full training observations as specified above; it is not a
learned first phase.

### Cost

Distinguish observation-level cost from global cost. For each observation $(r,t)$ and instrument $\omega$, define

$$
c^{cond}_{r,t,\omega}=\delta_c\sqrt{\frac{v_\omega(z_{r,t})}{n_{cost}}},
\qquad
c^{prop}_{r,t,\omega}=\delta_q|H_\omega(z_{r,t})|.
$$

These quantities can vary with $(r,t)$. Define the global unconditional cost per instrument by

$$
\bar c_\omega=\delta_u\sqrt{\frac{\bar v_\omega}{n_{cost}}}.
$$
The quantity $\bar c_\omega$ per instrument is calculated once from the full training experiment. It is constant across $(r,t)$ and must be reused unchanged for minibatches, full-training valuation, and validation valuation. Broadcast it across the path and time dimensions when combining costs.

The effective trading cost at observation $(r,t)$ is the elementwise maximum

$$
c_{r,t,\omega}=\max\left\{
c^{cond}_{r,t,\omega},
\bar c_\omega,
c^{prop}_{r,t,\omega}
\right\}.
$$

`DysanosConfig` contains:

- `cost_cond` ($\delta_c$), default `1` of ``learn_cost`` is ``True``, otherwise `0`.
- `cost_uncond` ($\delta_u$), default `1`;
- `cost_prop` ($\delta_q$), default `0.0001` (one basis point).

Because paths, rather than path-time observations, are the independent Monte
Carlo units, the cost is calibrated to the full training experiment with
$$
n_{cost}=S_{train}.
$$
It is *not* the size of a batch during training.

A minibatch estimates the variance, but it must not replace $S_{train}$ in the
standard-error denominator. Use the same $n_{cost}$ for training,
full-training valuation, and validation valuation. Validation MC-error
diagnostics use $S_{validation}$ complete paths; that is a different quantity
and must be labelled accordingly.

Use $c_{r,t,\omega}$, not $\bar c_\omega$ alone, in the utility objective and all observation-level diagnostics. Report `c_cond`, `c_prop`, global `bar_c`, and the resulting effective `c` separately, together with the fraction of observations/instruments for which each component is the binding maximum.

### OCE objective

Let the complete-path net trading return be

$$
\Pi_r=\sum_{t=0}^{TAU-1}\left\{
a_\theta(z_{r,t})^TdH_{r,t}
-|a_\theta(z_{r,t})|^Tc_{r,t}\right\}.
$$

There is no scalar cash offset. For entropy and power-exponential utility,
maximize over complete paths

$$
\sup_\theta E_P\left[D^{prev}_r u(\Pi_r)\right].
$$

For the mechanical CVaR branch, replace this expectation by the
previous-density-weighted mean of the lowest $\alpha$ fraction of $\Pi_r$.

Provide a benchmark action network with:

```python
pf_config = FeedForwardConfig(
    num_depth=2,
    num_width=100,
    activation="LeakyReLU",
    use_bias=True,
)
```

Also provide an unconditional deterministic-time benchmark
$a=(a_0,\ldots,a_{TAU-1})$ with
$a_t\in\mathbb R^{TNX}$ shared by all paths.  This has only $TAU\,TNX$
unrestricted parameters and tests unconditional time-$t$ moment conditions.
For increasing concave utility its negative objective is convex (including
the convex absolute transaction-cost term after sign reversal).  Support a
SciPy L-BFGS-B optimizer for the smooth cost-free case. Keep the complete
Torch objective and exact autograd gradient on the GPU; transfer only the
small deterministic action and gradient vectors between SciPy and Torch.
Require first-order stationarity rather than a flat objective for convergence.

Optionally preprocess the deterministic strategy instrument by instrument by
solving
$$
\sup_b\sum_{t,\omega}E_P\left[D^{prev}
u\left(b_{t,\omega}dH_{t,\omega}\right)\right].
$$
Store this fixed result as $b$, initialize the joint correction $a$ at zero,
and then solve
$$
\sup_a E_P\left[D^{prev}u\left(
\sum_{t,\omega}(a_{t,\omega}+b_{t,\omega})dH_{t,\omega}
\right)\right].
$$
Report the preprocessing and joint gradients separately. At a fully converged
joint optimum the decomposition into $a+b$ is only an initialization and
diagnostic device; the total action solves the same deterministic problem.

For generated one-day experiments, any Gaussian stock importance proposal
must target next-day path coordinates, not option-expiry coordinates. The
likelihood ratio remains the exact full-path $p/q$ mixture ratio. Report its
normalization, ESS, direct-$P$ log-spot mean/variance controls, and realized
log-variance control before interpreting the fitted density.

For entropy and power-exponential utility, if $\theta^*$ is optimal, define
the realized density by

$$
D^{new}_r=
\frac{D^{prev}_r\,u'\left(\Pi_r\right)}
{E_P[D^{prev}\,u'(\Pi)]}.
$$
Recompute the denominator on every valuation split. In particular, validation
uses its own $E_P[D^{prev}u'(\Pi)]$ normalization; there is no reason for the
training normalization to transfer unchanged. Report the split rescaling
factor and pre-normalization mean.

For mechanical CVaR, replace $u'(\Pi_r)$ by the frozen training-tail selector:
one below the cutoff, zero above it, and the calibrated fractional boundary
weight at a tie. Normalize it by the same complete-training-split rule.

At an interior functional optimum, normalization does not change the
first-order condition.  With transaction costs the near-martingale band is
$$
\left|E_P[D^{new}_r dH_{r,t,\omega}\mid z_{r,t}]\right|
\le E_P[D^{new}_r c_{r,t,\omega}\mid z_{r,t}].
$$

Once training finishes, `density()` returns a self-contained torch callable
accepting a complete path. State-only networks parameterize $a$ and the
conditional moments. The callable keeps a frozen copy of the previous density
and exposes the composed terminal path density
needed by the next measure-change stage.

At the final fitted epoch, compute the derivative of the full-split negative
utility objective with respect to every action-network parameter. Report
element count, non-finite count, $L^2$, RMS, mean-absolute, and
$L^\infty$ gradient statistics separately for training and validation. These
are full-split gradients, obtained by weighting complete-path batches by their
path fraction. A validation gradient need not be zero; its comparison with the
training gradient is an out-of-sample stationarity diagnostic.

An optional saved `arb(z; omega)` mask may disable individual long or short
option directions before P&L is formed. Spot remains active. Record the mask's
source report and candidate counts in the experiment output, and use a
different checkpoint identity for masked and unmasked runs.

# Task: Finding Arbitrage Directly

A static arbitrage is a trade $a(z_{t,r})\in \R^{TNX}$ conditional on $z_{t,r}$ with $\|a(z_{t,r})\|_1\leq 1$ such that
there exist any $\epsilon$, $\delta$ with
$$
  a(z_{t,r})^T dH_{t,r}(z_{t,r}) - |a(z_{t,r})|^T c_{t,r}(z_{t,r}) \geq 0
  \qquad\mathrm{and}\qquad
  P\left[ a(z_{t,r})^T dH_{t,r}(z_{t,r}) - |a(z_{t,r})|^T c_{t,r}(z_{t,r}) > \epsilon \|a(z_{t,r})\|_1  \right] > \delta
$$
where
* we assume cost are strictly positive,
* $dH$ are the scaled returns for this experiment such that each $|a(z_{t,r})^T 1|$ represents the net cash invested,
* therefore $\epsilon$ is the minimal required return on that cash.

An alternative with $\pi(z_{t,r}) := a(z_{t,r})^T dH_{t,r}(z_{t,r}) - |a(z_{t,r})|^T c_{t,r}(z_{t,r})$ is
$$
  E[ \min\{ \pi(z_{t,r}), 0 \} ] = 0
  \qquad\mathrm{and\ solve}\qquad
  \max: E[ \pi(z_{t,r}) - \epsilon \|a(z_{t,r})\|_1 ]
$$
and test that $E[ \pi(z_{t,r}) - \epsilon |a(z_{t,r})^T 1| ]$ is statistically positive.


Task
* Run experiments with different levels of $\epsilon$ and $\delta$ (for the arbitrage method)
* Suggest a network architecture and loss, or any optimization method, to find such $a$ with $\epsilon$ and $\delta$ set sufficiently to avoid numerical noise. 
* Search for such opportunities in the data set.


# Configuration and training

Use separate frozen dataclasses:

- `DysanosConfig` for model, utility, cost, density, label, group, seed, and `user_version`;
- `TrainConfig` for batch sizes, valuation batch size, optimizer, learning-rate schedule, gradient clipping, checkpoint intervals, and heavy-output intervals.

`epochs` is an explicit parameter of `train()`, not a `TrainConfig` field.
The main expected-utility learning rate is constant; do not use a predetermined
decay schedule.  Adam/AdamW is the stochastic network default.  Use full-split
SciPy L-BFGS-B for the smooth cost-free deterministic convex benchmark and
retain exact full-training gradient diagnostics for both optimizer families.
Allow warm start.

Train using  `trn_batch_size` batches on the training set. Valuate on training and valuation set using  `val_batch_size` batches for a full valuation run. The training run should only compute ``loss`` per epoch and is optimized for speed and memory use. Compute valuation for epochs when:
- After the first new computation
- Every `value_seconds` configured in `TrainConfig`, default 10.
- Every `value_update` epochs configured in `TrainConfig`, default 100.
- Upon `KeyboardInterrupt`. In this case catch the exception, inform the user, run valuation & serialize the model, then re-raise the exception.
- At the end.

When an update is run also store the model checkpoint. Ideally perform writing in a separate thread/process. If useful, feel free to use WandB or another tool.

Required training behavior:

- support `trn_batch_size` and `val_batch_size` in `TrainConfig`.
- derive any torch models from ``BaseModel`` from ``quantdata/torch/__init__.py``. Use ``BaseEnvironment`` to initialize the model. It provides ``verify``
  a function which is a NOOP unless ``debug`` is True. Use ``verify`` to validate numeric correctness of torch calculation such as ``isinfinite`` and
  rare logic checks (density being positive). The ``conditions`` argument of ``verify`` is a lambda function which will only be called if ``debug``
  is not ``False``. Similarly, the error message is alsoe a lambda function which will only be formatted if ``debug`` is ``True`` and the conditon
  was not met.
- report training-batch, full-training valuation, and validation losses separately;
- use suitable learning-rate scheduling and report the current learning rate;
- preserve NumPy and Torch RNG states across checkpoints and resumed runs;
- catch `KeyboardInterrupt`, write a safe checkpoint, inform the user, and re-raise;
- avoid computing or retaining full densities during a gradient step unless required;
- avoid unnecessary GPU-resident diagnostics and heavy outputs. Return only efficient quantites. Pass $D^\mathrm{prev}$ on as non-trainable network.

### Checkpointing and caching

Use `version` from `cdxcore.version` for code versions. Store at least:

- code/model version and `user_version`;
- complete configuration;
- epoch and trackers;
- model and optimizer state;
- NumPy and Torch RNG state.

Write checkpoints atomically through a temporary file followed by rename. If code version, `user_version`, objective semantics, density semantics, architecture, or checkpoint schema is incompatible, restart rather than loading silently. Save large density arrays at a separately configured low frequency.

## Required analytical outputs


All analytical outputs are per instrument unless stated otherwise and are computed separately for the full training and validation sets.

### Loss

The loss is -utility, a float.
Show per epoch. In the same graph show effective learning rate. Indicate epochs where training was restarted from disk to enable visual detection of breaks in serialization.
This is the only stat which is shown for training runs.

### Utility

Per epoch: mean utility, 

### Drift and MC error

For

$$
Y_{r,t,\omega}=D^{new}_r dH_{r,t,\omega},
$$

return the time-resolved statistics

$$
mean\_dH_{t,\omega}=E_r[Y_{r,t,\omega}],
\qquad
std\_dH_{t,\omega}=Std_r(Y_{r,t,\omega}),
\qquad
err\_dH_{t,\omega}=\frac{std\_dH_{t,\omega}}{\sqrt{S_{split}}}.
$$

For a compact statistic pooled across time, first form the complete-path
time-average
$$
\bar Y_{r,\omega}=\frac1{TAU}\sum_tY_{r,t,\omega},
$$
then report $E_r[\bar Y]$, $Std_r(\bar Y)$, and
$Std_r(\bar Y)/\sqrt{S_{split}}$. Do not estimate this error from the
dispersion of the $S\,TAU$ individual observations, because dates on the same
path are dependent.

State explicitly whether $\operatorname{Std}$ uses the population or sample convention. Visualize `mean_dH` with $\pm 2\,err\_dH$ bands and also visualize $mean\_dH/err\_dH$ with horizontal $\pm2$ reference lines.

Use the time/instrument matrix for the strict martingale and cost-band tests,
including the familywise multiplicity correction. Label pooled values as
summaries rather than separate independent conditions.

### Return on cost

The near-martingale statistic is based on a mean, not on sample quantiles. Return

$$
return\_on\_cost_\omega=\frac{E[D^{new}dH_\omega]}{E[D^{new}c_\omega]},
$$

with a protected denominator and a clearly documented alternative if costs are state dependent and a different normalization is required. The target is

$$
|return\_on\_cost_\omega|\le1.
$$

Additionally report the standard deviation and 5%, 10%, 50%, 90%, and 95% quantiles of the sample-level ratio $(D^{new}dH)/(D^{new}c)$, equivalently $dH/c$ where the density is positive. These quantiles are distribution diagnostics; they do not by themselves establish the near-martingale condition.

### Signed excess

Define the sample-level signed violation

$$
excess_{r,t,\omega}=D^{new}_r\,sign(dH_{r,t,\omega})\max(|dH_{r,t,\omega}|-c_{r,t,\omega},0).
$$

Return its mean, standard deviation, and 5%, 10%, 50%, 90%, and 95% quantiles. Also report the mean absolute violation and the fraction of observations outside the cost band.

### Additional diagnostics

Return and visualize, where appropriate:

- density mean, minimum, maximum, and quantiles;
- density effective sample size, $ESS=(\sum w)^2/\sum w^2$;
- density normalization error and non-finite count;
- gradient norm and non-finite-gradient count;
- action mean, standard deviation, maximum, and action-cap saturation rate;
- conditional first- and second-moment losses and variance calibration;
- each cost component and its binding frequency;
- loss, full-training loss, validation loss, learning rate, epoch, and elapsed time.

Use established visualization colors: training in red/purple/orange and validation in blue/green/light blue. Include label and group/session in figure titles. Model outputs must be wired through trackers, checkpoint serialization, valuation batching, and visualization.

## Verification requirements

Add focused tests under `tests/` for:

1. all four return/value modes;
2. delta-hedged returns and gross option-plus-hedge cost notionals;
3. cash-normalization floors and zero denominators;
4. conditional variance nonnegativity;
5. the elementwise maximum of all three cost components;
6. fixed full-training-size cost scaling under minibatching;
7. density finiteness, nonnegativity, normalization, and path composition;
8. train/validation isolation;
9. checkpoint round-trip, RNG restoration, and incompatible-version restart;
10. zero variance, a single observation, and non-finite inputs;
11. forward values and gradients of the OCE objective.

Use small deterministic tensors for mathematical tests. Run the narrowest relevant tests first, then the complete new test module. Do not claim notebook-level verification unless the notebook or an equivalent scripted workflow was actually executed.

## Usage expectations

The implementation must support multiple comparable runs, including delta hedging and cash normalization on/off, without cache collisions. Runs must be resumable and easy to monitor for density concentration, exploding networks, vanishing effective sample size, and failure of the near-martingale diagnostics.

Build a stand-alone tool usiong ``marimo`` https://github.com/marimo-team/marimo to
1) Display per
