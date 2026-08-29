# Risk Neutral Hierarchical SANOS Dynamics

Let $C( h^i_t; T_i, K )$ be the call prices for time-to-expirt $T_i$ at time $t$ with spot-relative strike $K$. The vector $h^i_t$ is the random state vector.
We postulate the following dynamics in $0$:
1. Use $C^1_0$ to sample spot $S_t$. In SANOS this is analytics. This way $C^1$ is always consistent with spot.
2. Same iteratively $h^i_{t+1} = \mathbf{G}( \epsilon^i_t | S_{t+1},\,h^{1:i-1}_{t+1}\, h^i_t )$ where $\epsilon_t^i \in \mathbb{R}^{n_i}$ is standard normal.

## Hard Matching the Second Week ($C^2$)

### Theoretical Setup

The main objective for fixed $S_t,h_0$ is and some loss $\|\cdot\|$ is
```math

    0 \equiv  \sup_K \mathbb{E}\left[
        \left\|
                 \frac{S_0}{S_t} C( h^2_0; T_2, K ) -  C\!\left( h^1_t = \mathbf{G}(  \epsilon^1_t | S_t,h_0  ); T_1, \frac{S_0 K}{S_t} \right)
    \right\| \Big|\ S_t, h_0\
    \right]
```
We may translate this for a random set $\mathcal{K}$ of strikes into a loss function which focuses on the worst offenders
```math
   0 \equiv \sup_{K\in\mathcal{K}}\frac1{\Omega}
    \sum_{\omega=1}^\Omega
        \left\|
         S_0 C( h^2_0; T_2, K ) - S_t\, C\!\left( h^1_t = \mathbf{G}(  \epsilon^1_t | S_t,h_0  );\ T_1, \frac{S_0 K}{S_t} \right)
    \right\|
```
Consider the functional
```math
    h^1_t \longmapsto f(h^1_t;K) := S_t\, C\!\left( h^1_t ;\ T_1, \frac{S_0 K}{S_t} \right) -  S_0 C( h^2_0; T_2, K ) \ .
```
The question is whether there exists a conditional law of $h^1_t$ such that all
of the selected option expectations are zero simultaneously.

### Existence with Finitely Many Strikes

Fix a historical state $a$. Let $\nu_a$ be the prescribed front-week law of
$S_t$, and let $\mathcal H_a(s)$ be the admissible set of future $h_t^1$ states
when $S_t=s$. For the $m=|\mathcal K|$ selected options, define the pathwise
value vector

```math
\Phi_a(s,h)
:=
\left(
s\,C\!\left(h;T_1,\frac{S_0K_j}{s}\right)
\right)_{j=1}^m
```

and the current target vector

```math
q_a
:=
\left(
S_0C(h_{0,a}^2;T_2,K_j)
\right)_{j=1}^m.
```

Here $C$ means the selected in-the-money option value, so it is replaced by
the corresponding put value where appropriate. An admissible conditional
kernel $Q_a(dh\mid s)$ solves the finite-strike inverse exactly when

```math
q_a
=
\int\!\!\int
\Phi_a(s,h)\,Q_a(dh\mid s)\,\nu_a(ds).
```

Allowing arbitrary randomized conditional kernels gives the feasible moment
set

```math
\mathcal F_a
:=
\left\{
\int\!\!\int
\Phi_a(s,h)\,Q_a(dh\mid s)\,\nu_a(ds)
:
Q_a(\,\cdot\mid s)\text{ is admissible}
\right\}.
```

Under standard measurability, compactness, continuity, and integrability
conditions, $\mathcal F_a$ is a closed convex subset of $\mathbb R^m$. Exact
feasibility is therefore the geometric condition $q_a\in\mathcal F_a$. The
attainable set of one finite neural transport is generally smaller and need
not itself be convex; approximation requires $q_a$ to lie in its closure.

#### Randomness versus the Number of Constraints

The number of stochastic factors is not a direct upper bound on the number of
expectations that can be matched. Even a distribution driven by one atomless
scalar random variable has infinitely many functional degrees of freedom. At
the level of probability laws, one uniform or Gaussian seed can generate an
arbitrary probability kernel on standard Borel spaces if arbitrary measurable
maps are allowed.

This does **not** mean that $\mathbf G$ creates infinitely many independent
pathwise risk factors. Every generated quantity remains a function of the
original spot and innovation seeds. It means that the choice of their
conditional distribution is infinite-dimensional. Consequently, eleven
strike constraints do not require eleven independent shocks.

Carathéodory's theorem gives another finite-dimensional view. Once the problem
has been reduced to a convex hull in $\mathbb R^m$, any feasible target can be
represented as a mixture of at most $m+1$ attainable kernels or payoff
vectors. Under the corresponding measurable-selection assumptions, an
atomless innovation can encode that finite mixture. A target in the relative
interior of this convex hull is robust to small price and Monte Carlo errors;
a boundary solution can be feasible but numerically fragile.

These measure-theoretic statements are more permissive than the implemented
transport. A finite neural network has finitely many parameters. A locally
Lipschitz map driven by $n_\epsilon$ conditional shocks has support of intrinsic
dimension at most $n_\epsilon$, and bounded or smooth parameterizations may
exclude atomic, singular, or sharply discontinuous kernels. The computational
question is therefore whether the target is attainable by the chosen model
class, not only whether an unrestricted conditional law exists.

#### When There Is No Solution

A solution can fail for several different reasons:

1. **Inherited arbitrage restrictions.** Every attainable expected option
   vector obeys strike monotonicity, convexity, payoff bounds, put-call parity,
   and the applicable martingale and calendar restrictions. Expectations
   preserve these inequalities. Static no-arbitrage of each observed surface
   is necessary but need not ensure compatibility with the fixed spot law and
   the admissible future SANOS surfaces.

2. **Support or law restrictions.** Matching may require future states, tail
   mass, atoms, or boundary mass excluded by $\mathcal H_a(s)$ or by imposed
   absolute-continuity, invertibility, boundedness, or smoothness conditions.

3. **Degenerate payoff geometry.** If all attainable $\Phi_a(s,h)$ satisfy an
   affine identity, every attainable expectation satisfies it. A target
   outside that affine subspace is impossible. A low-dimensional pathwise
   payoff manifold is not sufficient evidence by itself: its convex hull may
   still be full-dimensional.

4. **Shared-transport incompatibility.** Statewise kernels can exist while no
   single $\mathbf G_\theta$ represents all of them. Identical conditioning
   inputs requiring different laws are a genuine contradiction. More
   commonly, nearby inputs require transports that vary too sharply for the
   network, or the shared parameterization has insufficient capacity.

5. **Finite parameterization or optimization.** An unrestricted kernel may be
   feasible while the chosen latent dimension, neural architecture, bounds,
   or regularization excludes it. Alternatively, a feasible network may not
   be found by a non-convex optimizer. Neither case establishes an arbitrage in
   the market data.

There is a convex separation characterization of genuine non-existence. If
$q_a\notin\mathcal F_a$, then under the closedness assumptions there is an
option portfolio $\lambda\in\mathbb R^m$ such that

```math
\lambda^\mathsf T q_a
>
\int
\sup_{h\in\mathcal H_a(s)}
\lambda^\mathsf T\Phi_a(s,h)\,
\nu_a(ds).
```

The left side is the target value of the portfolio; the right side is the
largest value attainable under any admissible conditional choice of future
state. Such a $\lambda$ is a non-existence certificate. Numerically it becomes
rigorous only when the inner supremum is solved globally and the separation
margin exceeds sampling, discretization, tail, and optimization errors.

#### What Gradients over Many States Can Detect

Concatenate the population residuals and their parameter Jacobian,

```math
\mu(\theta)
:=
\left(\mathbb E[R_{aj}(\theta)]\right)_{a,j},
\qquad
J(\theta)
:=
\frac{\partial\mu(\theta)}{\partial\theta}.
```

The beta control variate reduces estimator variance but does not change this
population feasibility condition because $\mathbb E[X_a]=0$. A structural
diagnostic should estimate the per-constraint Jacobian on large fixed random
samples, preferably on several independent samples, rather than differentiate
only the scalar smooth-maximum loss.

The singular spectrum of the studentized Jacobian $D^{-1}J$ measures local
controllability. Full row rank means that every sufficiently small collection
of moment corrections is locally available. Small singular values identify
linear combinations of state-option constraints that the network changes only
weakly. Projecting $-\mu$ onto the range of $J$ gives the best infinitesimal
correction; a stable residual component orthogonal to that range is evidence
of a local structural obstruction. Jacobian-vector products and randomized
singular-value methods can estimate this without materializing the full
state-by-strike-by-parameter matrix.

This is a diagnostic, not a proof of global non-existence. Rank deficiency can
come from a saturated network, a symmetric or poor parameter point, or Monte
Carlo noise, even when a solution exists elsewhere. Conversely, full rank does
not prove that zero lies in the globally attainable set. Sampling many
historical states diagnoses the shared network only on those inputs unless one
also assumes regularity and adequate state-space coverage.

A stronger numerical test separates model capacity from economic feasibility.
For each sampled spot context, generate a rich set of admissible future states
and solve a convex feasibility problem for non-negative conditional weights.
If this enlarged problem is feasible but the network fit fails, the obstruction
is the architecture or optimizer. If it is infeasible with a stable positive
margin as the state grid, tails, and Monte Carlo sample are refined, its dual
produces an approximate separating portfolio $\lambda$ and evidence of genuine
non-existence.

Empirically, separate flexible transports have passed the Monte Carlo target
for six deliberately difficult historical states. This is useful evidence for
statewise feasibility, but it is not a proof for all historical states. The
all-history statewise affine parameterization did not pass its single worst
state, illustrating why failure of a restricted model class must not be
confused with failure of the unrestricted moment problem.

## Soft Matching the Second Week ($C^2$)

The inverse is learned from Monte Carlo estimates, so exact equality of a sample mean to zero is neither expected nor a useful stopping rule. We instead minimize a smooth maximum of beta-adjusted, studentized martingale errors. The final tolerance is $\delta=2$ Monte Carlo standard errors.

For historical state $a$ and strike $K_j\in\mathcal K$, define the one-week option roll

```math
R_{aj}
:=
S_t C\!\left(
    \mathbf G_\theta(\epsilon_t^1\mid S_t,h_{0,a});
    T_1,\frac{S_0K_j}{S_t}
\right)
-S_0C(h_{0,a}^2;T_2,K_j).
```

We use the in-the-money option at each strike: a call for $K_j\leq 1$ and a put for $K_j>1$. Let

```math
X_a:=\frac{S_t}{S_0}-1,
\qquad \mathbb E[X_a]=0.
```

The known spot expectation gives a control variate. For every state and option, beta is estimated from all $N$ paths in the current data set,

```math
\widehat\beta_{aj}
:=
\frac{\widehat{\operatorname{Cov}}(R_{aj},X_a)}
     {\widehat{\operatorname{Var}}(X_a)},
\qquad
\widehat\mu_{aj}
:=
\overline R_{aj}-\widehat\beta_{aj}\overline X_a.
```

Beta is a denoiser, not a change to the payoff or to the simulated law. It is computed separately for every state and separately on training, validation, and confirmation data.

### Studentized Smooth-Max Objective

Partition the paths into $B$ independent replication blocks. If $\overline R_{baj}$ and $\overline X_{ba}$ are the means in block $b$, define

```math
\operatorname{SE}_{R,aj}
:=
\frac{\operatorname{sd}_{b}(\overline R_{baj})}{\sqrt B},
\qquad
\operatorname{SE}_{X,a}
:=
\frac{\operatorname{sd}_{b}(\overline X_{ba})}{\sqrt B},
```

and

```math
d_{aj}
:=
\max\!\left\{
    \operatorname{SE}_{R,aj},
    |\widehat\beta_{aj}|\operatorname{SE}_{X,a}
\right\},
\qquad
z_{aj}:=\frac{\widehat\mu_{aj}}{d_{aj}}.
```

Both $\widehat\beta_{aj}$ and $d_{aj}$ are detached when differentiating. For
the set $\mathcal A$ of historical states, the training loss is

```math
\mathcal L_{\mathcal A}(\theta)
:=
\tau\log\!\left[
    \frac{1}{|\mathcal A||\mathcal K|}
    \sum_{a\in\mathcal A}\sum_{j\in\mathcal K}
    \exp\!\left(\frac{z_{aj}^2}{\tau}\right)
\right],
\qquad \tau=1.
```

This is a smooth approximation to the largest squared studentized error over
the complete historical panel. States are evaluated in memory-sized batches
while the network is held fixed. Their gradients are accumulated against this
one global objective, clipped once, and followed by exactly one Adam update per
sweep. Thus the state partition does not define separate objectives or permit
the shared network to change between state batches. The loss continues to move
all adjusted means toward zero; $|z_{aj}|\leq\delta$ is a validation rule, not a
hinge in the loss. Raw means, raw mean/MC-error ratios, and their worst-$10\%$
average are retained as diagnostics even though optimization uses the
beta-adjusted objective.

### Monte Carlo Protocol

Training uses $N=10{,}000$ paths per historical state, represented by ten
independent blocks of $1{,}000$ paths. The same underlying random numbers are
transformed through every historical state. At the beginning of every sweep,
all training random numbers are replaced by an independent pool. Beta is
recomputed for every state on every sweep. The fully resampled loss is noisy by
construction; Adam averages its stochastic gradients, while a moving average
of the sampled loss controls learning-rate reduction and plateau detection.
The fixed validation statistic controls model selection, and the best
validation checkpoint is preserved separately from the latest training state.

Validation and confirmation use the same historical initial states but independent path samples. Thus these audits test generalization across random paths, not across unseen states. They are fixed and disjoint:

```math
N_{\mathrm{validation}}
=N_{\mathrm{confirmation}}
=100{,}000=100\times1{,}000.
```

Validation may be inspected during training. Confirmation is not evaluated until the model is frozen after three consecutive validation sweeps satisfy

```math
\max_{a,j}|z_{aj}|\leq 2.
```

Since each audit uses ten times as many paths as one training sweep, its standard errors are approximately $1/\sqrt{10}$ as large. It therefore detects materially smaller systematic pricing errors.

### Block Bootstrap and Audit Consistency

We use $500$ non-parametric bootstrap samples of the $100$ independent replication blocks. The same bootstrap block multiplicities are used across all historical states, preserving the joint common-random-number dependence. Beta is re-estimated within every bootstrap sample from block-level path, spot, spot-square, and spot-payoff-product sufficient statistics.

For each state and option we report the bootstrap mean, Monte Carlo error, and the $2.5\%$, $50\%$, and $97.5\%$ quantiles for both raw and beta-adjusted means. We also bootstrap the joint maximum and the worst-$10\%$ average across all state-option constraints; this records the sampling distribution of the extreme statistics used for stopping.

Finally, validation and confirmation are compared using

```math
D_{aj}
:=
\frac{
    \widehat\mu^{\mathrm{validation}}_{aj}
    -\widehat\mu^{\mathrm{confirmation}}_{aj}
}{
    \sqrt{
        (\operatorname{SE}^{\mathrm{validation}}_{aj})^2
        +(\operatorname{SE}^{\mathrm{confirmation}}_{aj})^2
    }
}.
```

Agreement between validation and confirmation and agreement with zero are separate requirements: two audits can be mutually consistent while identifying the same non-zero pricing error.
