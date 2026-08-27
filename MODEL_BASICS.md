# Model Basics

This file records modelling primitives that may be reused across AAE research
projects. Project specifications remain authoritative for target construction,
costs, eligibility masks, model architecture, and reporting.

## OCE Monetary Utilities

We assume $\mathbb{P}$ is some measure and that $h$ is the observed "state" of the world. We do not assume that it contains all available information.

Let $u$ be a smooth utility function which is concave and increasing. 
We also assume that $1 = u'(0)$. 
We also impose
* $u'(\infty):=\lim_{x\uparrow \infty} u'(x) = 0$ f(the utility cannot grow infinitely), and
* $u'(-\infty):=\lim_{x\downarrow-\infty} u'(x) = \infty$. (This is a numerical conditon; technically $\lim_{x\downarrow-\infty} u'(x) > 1$ is sufficient.)


For some $X\in L^1$ define the OCE Monetary utility
```math
  U(X|h)^\mathbb{P} := \sup_{y\equiv y(h)\in\mathbb{R}}:\ \mathbb{E}^\mathbb{P}\left[ 
      u(y + X ) + y \big|\ h\
    \right]
```
The notation $y\equiv y(h)$ here means that the optimal real $y$ depends on the current state. If the state is a constant, then it is just a real number. 
Otherwise it will be a network as function of $h$ and the $\sup$ is taken over network weights. We will drop the dependency on the measure or the state if 
evident from the context for notational clarity.

The function $U$ has the following properties:
* It is **normalized** to $U(0)=0$ (this is a consequence of $u'(0)=1$).
* It is **monotone** i.e if $X\geq Y$ then $U(X) \geq U(Y)$.
* It is **concave** i.e. if $\alpha\in[0,1]$ then $U(\alpha X + (1-\alpha)Y) \geq \alpha U(X) + (1-\alpha) U(Y)$. This means it is *risk averse* with respect to $\mathbb{P}$ in the sense
  that $U^\mathbb{P}(X) \leq U^\mathbb{P}( \mathbb{E}^\mathbb{P}[X] )$.
* It is **cash invariant** ie for $x\equiv x(h)$ we have $U(x+Y|h)=x+U(Y|h)$. In particular this property implies that any optimal trade is independent of our current cash position (a feature not in line with econmic reality but with practicalities of operating in large organizations).


### Optimal Trading under OCE

Let $dH \in \mathbb{R}^d$ be the returns of $d$ tradable instruments
with convex positive cost $c(d)\geq 0$ where $c(0)=0$. Most commonly, $c(d) = \gamma' d$ for some proportional cost $\gamma \in \mathbb{R}_{\geq 0}^d$. 
Assume $Z$ is our current position.
The OCE-optimal trade $a^*\equiv a^*(h) \in \mathbb{R}^d$ in $dH$ given $Z$ is a solutuon to

```math
  U^*(Z|h):=
  \sup_{a\in \mathbb{R}^d}:\ U^\mathbb{P}\left( Z +
      a' dH - c(a) \Big|\ h\
    \right) =
  \sup_{a\in \mathbb{R}^d, y\in\mathbb{R}}:\ \mathbb{E}^\mathbb{P}\left[
      u(y + Z + a' dH - c(a) ) + y \Big| \ h \ 
    \right] \ .
```
The quantity $U^*(Z)$ is a cash quantity. It denotes the cash value we give to owning $Z$. However because $U^*(0)$ is not necessarily zero (so-called "statistical arbitrage") we define the minimal price we would pay for $Z$ by
$$
  \pi(Z) := U^*(Z) - U^*(0) \ .
$$

### OCE Near-Martingale Measdures

Let $c(a)=\gamma' |a|$ (see https://arxiv.org/abs/2111.07844 for the convex case).
We note that for $Z=0$ in particular an optimal solution $a^*$ and $y^*$ to above equation will essentially satisfy
* $\mathbb{E}^\mathbb{P}\left[ u'\left( y^* +  a^* {}' dH - \gamma' |a^*| \right) \right] = 1$ (derivative in $y^*$) and
* $\mathbb{E}^\mathbb{P}\left[ u'\left( y^* +  a^* {}' dH - \gamma' |a^*| \right) \left( \mathrm{sign}(a) dH^i - \gamma \right) \right] = 0$ (derivative in $a^{*,i}$).

This means that $\mathbb{Q}$ defined by 
$$
  \frac{ d\mathbb{Q} }{ d\mathbb{P} } := u'\left( y^* +  a^* {}' dH - \gamma' |a^*| \right) \in [ u'(-\infty), u'(\infty) ]
$$
is a **near-martingale measure** in the sense that
$$
  -\gamma \preceq \mathbb{E}^\mathbb{Q}\left[ dH \right] \preceq  \gamma \ .
$$



Because of our boundary conditions $u'(\infty)=0$ and $u'(-\infty)<1$

## Power-exponential utility

The entropy with
$$
  u(x) := \frac{ 1 - \exp(-\lambda x)}\lambda
$$
is by far the most intuitive utility. Its OCE measure is the entropy
$$
  U(X)=-\frac1\lambda \log \mathbb{E}\left[ \exp(-\lambda X) \right]
$$
which reduces to mean-variance for normal $X$. *However* it is numerically very punative and assigns negative infinite utility for short positions
in Black & Scholes assets. For that reason we prefer the **power-exponential** "power exp" utility defined as
```math
u(z)=
\begin{cases}
\frac{1-e^{-\lambda\,z}}\lambda , & z>0,\\
z-\frac\lambda 2 z^2, & z\leq 0.
\end{cases}
```

Its marginal utility is
```math
u'(z)=
\begin{cases}
e^{-\lambda z}, & z>0,\\
1-\lambda z, & z\leq 0.
\end{cases}
```
with second order derivative $u''(0)=-\lambda$.

The two branches and their first derivatives agree at zero. Thus `u` is
increasing, concave, continuously differentiable, and has unit marginal utility
at zero. Gains have exponentially decreasing marginal value, while the loss
branch has a quadratic utility penalty and only a linearly increasing density
tilt. The latter is less dominated by one extreme loss observation than a pure
exponential tilt, whose marginal weight grows exponentially throughout the
loss tail.

## Bounded commercial actions with an L1 excess penalty

The previously used OCE construction lets a model emit an unconstrained raw
action vector $a$. However, if too-good-to-be-true trades, or real arbitrage, is present then the
solution will be to take an infinite position. For real arbitrage this is true regardless of risk aversion due
to monotonicity. It is therefore prudent to reparamterize the optional action as a bounded 
value.  The bound can then interatively be raised if this is of interest. 

Using `tanh` times the bound will lead to saturated derivatives. To avoid this,
for a fixed $\operatorname{cap}>0$ define the radial rescaling onto the L1 ball
by

```math
\bar a
=a\,\frac{\operatorname{cap}}
{\max\left\{\operatorname{cap},\lVert a\rVert_1\right\}} \ .
```

The map is the identity in a neighbourhood of $a=0$, so its derivative there is
the identity. Outside the ball, it preserves the raw action's direction and its
piecewise tangential gradients; the L1 excess penalty below supplies a radial
gradient back towards the cap. This radial map is not the Euclidean projection
onto an L1 ball, which would soft-threshold coordinates and generally change
the action's direction.

For $a\in\mathbb{R}$ the simpler
$$
  \bar a := \max\{\min\{a, \operatorname{cap}\}, -\operatorname{cap}\}
$$
can be used.

The optimized objective is

```math
\sup_{a,y}:
\mathbb E\left[u\left(y+\bar a^\top dS - c(\bar a)\right)-y
-\eta\left(\lVert a\rVert_1-\operatorname{cap}\right)_+^2 \right] \ ,
\qquad \eta\geq0.
```
Here $\operatorname{cap}$ is the fixed L1 gross-action cap of the action that enters commercial PnL. For a
single instrument use $\operatorname{cap}=1$. For multiple instruments it may increase by a
redeclared rule as the eligible instrument set grows, but it is not learned.
