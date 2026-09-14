# Power-exponential utility adversary

This note specifies the first learned-drift experiment for the four-week
fixed-cash-grid shrinking tower. It deliberately separates action notional
from the cash-risk-tolerance scaling of an optimized certainty equivalent
(OCE).

## Trading return

At trade week \(t=0,\ldots,3\), there are

\[
M_t=200(4-t)
\]

live ITM option returns. The policy observes the current density/DLV state
\(H_t\) and produces an action \(a_t(H_t)\). The fixed deterministic aggregation
requested for the experiment is

\[
X(a)=\sum_{t=0}^3 \frac{a_t(H_t)^\top R_t}{4M_t}.
\]

The factors \(1/(4M_t)\) only choose the units of aggregate P&L. They do not
define either action leverage or OCE risk tolerance.

## Stable power-exponential OCE

Use the existing unit power-exponential utility

\[
u(z)=
\begin{cases}
1-e^{-z}, & z>0,\\
z-z^2/2, & z\leq0.
\end{cases}
\]

Its cash-scaled OCE at a fixed risk tolerance \(\tau>0\) is

\[
\mathcal U_\tau(X)
=\tau\sup_y\left\{
\mathbb E\left[u\left(y+\frac{X}{\tau}\right)\right]-y
\right\}.
\]

This expression is in the same cash units as \(X\). Initially use \(\tau=1\)
in normalized-return units. Treat \(\tau\) as an experiment setting, not as an
optimized action notional.

Evaluate the gain branch as `-expm1(-z)`. Evaluate log marginal utility as

\[
\log u'(z)=
\begin{cases}
-z, & z>0,\\
\log(1-z), & z\leq0,
\end{cases}
\]

using `log1p(-z)` on the loss branch. For each P&L sample, solve the scalar OCE
translation by bisection from

\[
\mathbb E\left[u'\left(y+X/\tau\right)\right]=1.
\]

Detach the solved translation in the outer update; this is the
envelope-theorem gradient. The resulting marginal weights are positive and
already have sample mean one.

## Action direction and notional

Separate a unit-ball direction from the scalar notional. Map each raw network
action to

\[
b_t(H_t)
=\frac{\widetilde a_t(H_t)}
       {\max\{1,\|\widetilde a_t(H_t)\|_1\}},
\qquad \|b_t(H_t)\|_1\leq1.
\]

For a fixed notional \(N\), define the normalized direction return and the
total cash OCE by

\[
Z_\theta=\sum_{t=0}^3\frac{b_t(H_t)^\top R_t}{4M_t},
\qquad
C(N)=\sup_{\theta_0,\ldots,\theta_3}\mathcal U_\tau(NZ_\theta).
\]

The per-unit diagnostic is

\[
V(N)=\frac{C(N)}{N}.
\]

Thus \(N\) multiplies the return inside the utility. Dividing the cash OCE by
\(N\) only reports value per unit of notional; it is not the training objective
for choosing \(N\).

The direction projection is exact. A small one-sided stabilization

\[
-\eta\sum_t
\bigl(\|\widetilde a_t(H_t)\|_1-1\bigr)_+^2
\]

may be used to pull raw actions back from the projection's flat radial region.
It must not be a positive penalty in a supremum, and it must not penalize
interior actions or force a nonzero action under the null. Because the
projection has already bounded the effective direction, this term is only an
optimization aid and not an economic risk penalty.

Solve a geometric sequence of fixed notionals rather than maximizing the
per-unit quantity over \(N\). Warm-start the next notional, but use fresh
simulation paths. Record the realized direction norms and the total-cash OCE
envelope derivative. For a fixed direction with normalized P&L \(Z\), the
derivative is

\[
g(N)=\mathbb E\left[
u'\left(y+NZ/\tau\right)Z
\right].
\]

Use a bracketed scalar root search for \(g(N)=0\) on the total cash OCE. A
finite near-arbitrage has an interior root. An arbitrage has \(g(N)>0\) over
successive larger notionals. Positive derivatives on training paths alone are
not evidence of arbitrage; the frozen policy and notional must repeat on
independent validation and confirmation paths.

For unit risk tolerance, the exact per-unit expression proposed for the
diagnostic curve is

\[
V(N)=\sup_{\widetilde a,y}
\frac{1}{N}\left(
\mathbb E\left[
u\left(y+
N\frac{\widetilde a^\top R}
       {\max\{1,\|\widetilde a\|_1\}}
\right)
\right]-y
\right)
-\eta(\|\widetilde a\|_1-1)_+^2.
\]

For small \(N\), the power-exponential OCE has the local expansion

\[
V(N)=\mathbb E[Z]-\frac{N}{2}\operatorname{Var}(Z)+o(N).
\]

Under the usual bounded-payoff conditions, \(V(N)\) approaches the worst
payoff \(\operatorname*{ess\,inf}Z\) as \(N\) grows. This curve therefore
shows how much per-unit value survives increasing leverage. It should be
evaluated over \(N\), not maximized over \(N\): concavity normally makes its
maximum occur at the smallest allowed notional.

## Why the inverse scaling answers a different question

The alternative

\[
\sup_{\widetilde a,N,y}
N\left(
\mathbb E\left[
u\left(y+
\frac{\widetilde a^\top R}
     {N\max\{1,\|\widetilde a\|_1\}}
\right)
\right]-y
\right)
-\eta(\|\widetilde a\|_1-1)_+^2
\]

is a valid cash-scaled OCE perspective. It is useful only with \(N\) fixed.
For normalized \(b\),

\[
N\,\mathcal U_1(b^\top R/N)
=\mathbb E[b^\top R]
-\frac{\operatorname{Var}(b^\top R)}{2N}
+o(N^{-1}).
\]

Optimizing \(N\) therefore removes the risk term and sends \(N\) to infinity
for any direction with positive estimated mean, including ordinary physical
risk premia and Monte Carlo estimation noise. It does not distinguish a
near-arbitrage from an arbitrage.

## Audit protocol

Train the policy, OCE translation, and any scalar notional only on training
paths. Freeze all of them for validation and confirmation. Use distinct random
seeds and at least 100,000 paths in each final audit. Report:

- OCE and its Monte Carlo standard error;
- mean, standard deviation, and loss probability of strategy P&L;
- realized maximum, median, and upper-tail action \(L^1\) norms;
- cap envelope derivative and its standard error;
- effective sample size of the normalized marginal-utility weights;
- validation and confirmation agreement.

The normalized marginal utility \(u'(y+X/\tau)\), not normalized utility
\(u/\mathbb E[u]\), is the positive Radon--Nikodym weight candidate.
