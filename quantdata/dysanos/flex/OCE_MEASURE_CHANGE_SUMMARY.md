# OCE measure-change experiments: status, evidence, and open issues

Status date: 30 September 2026.

This note records the fixed-path OCE experiments for the 25-day, four-expiry,
11-strike option book. It distinguishes numerical observations from working
hypotheses. The experiments are in-sample root-finding exercises on one fixed
Monte Carlo population; they are not out-of-sample no-arbitrage claims.

## Experimental population

- One million antithetic Monte Carlo paths from one initial state.
- Twenty-five option entry dates.
- Four option tenors at every entry date: 5D, 10D, 15D, and 20D.
- Eleven normalized-moneyness strikes per tenor, from -1.5 to 0.5 in steps of
  0.2.
- Forty-five traded returns per entry date: one one-day spot return and 44
  option returns.
- Every option return is a hold-to-expiry return:

  \[
  R_t^{j,i}=\frac{\operatorname{payoff}_{t+j}^{j,i}}{P_t^{j,i}}-1.
  \]

  Here `initial price` means the option price on its own entry date, not the
  global day-zero price.
- The action observes the current Direct-200 hidden state and eleven current
  option prices for each expiry.
- All reported training updates use the complete fixed population. The
  65,536-path chunks are only a GPU-memory device; parameters do not change
  between chunks.

## OCE objective and the cash intercept

For net trading P&L \(\Pi(a)\), the profiled exponential OCE is

\[
\operatorname{CE}_Q(a)=-\log E_Q[e^{-\Pi(a)}].
\]

For \(u(x)=1-e^{-x}\), the explicit cash-intercept form is

\[
\sup_y\{E_Q[u(y+\Pi(a))]-y\}.
\]

The first-order condition gives

\[
y^*=\log E_Q[e^{-\Pi(a)}],
\]

and substituting \(y^*\) recovers the profiled log-OCE exactly. Thus the current
objective has not omitted \(y\); it eliminates it analytically. Numerical scalar
optimizations agreed with the profiled objective to approximately \(10^{-9}\).

## Measure definitions

### Q1: convex unconditional empirical measure

Q1 is the minimum-relative-entropy density over the original one-million-path
empirical population subject to

\[
\left|E_{Q_1}[R_{t,k}]\right|\le c_k
\]

for all 25 dates and 45 instruments. There are 1,125 unconditional constraints.
The cost bands are 1 bp for spot and 10 bp for options.

Verified Q1 diagnostics:

- ESS: 765,493 paths, or 76.55%.
- Maximum constraint violation: \(3.15\times10^{-8}\).
- Maximum KKT residual: \(3.53\times10^{-8}\).
- Primal-dual gap: \(3.37\times10^{-9}\).
- All weights are strictly positive.

Q1 controls unconditional coordinate means. It does not impose conditional
martingale restrictions given hidden state and prices.

### Q2: first conditional OCE tilt

The first learned policy is fitted under Q1 with one-tenth transaction costs:
0.1 bp for spot and 1 bp for options. Its marginal-utility density defines Q2:

\[
\frac{dQ_2}{dQ_1}\propto e^{-\Pi(a_2)}.
\]

The policy consists of a shared two-layer width-128 network plus four
expiry-specific linear skips from 211 normalized features to 12 actions. The
four spot heads are summed, while the 44 option heads remain separate. A single
global L1 cap of 16 is applied to the resulting 45-dimensional action.

### Q3: second conditional OCE tilt

The completed Q3 experiment was implemented as a frozen Q2 policy plus a
trainable residual. The Q3 density used the incremental net P&L:

\[
\frac{dQ_3}{dQ_2}
\propto
\exp\!\left[-\{\Pi(a_2+d_3)-\Pi(a_2)\}\right].
\]

This faithfully implemented the earlier instruction
\(a_i=a_{i-1}+\text{new}_i\), but it is not the preferred numerical
parameterization. The corrected design is one cumulative policy initialized at
Q2 and then fully trainable, without freezing Q2 or forcing a separate residual
to cancel it.

The Q3 results below still refer to the frozen-plus-residual experiment. Q3 was
constructed from the original 500-update Q2, not from the later 600-update Q2.

## Verified numerical results

| Quantity | Result |
|---|---:|
| Q1 ESS | 765,493 |
| Original Q2 low-cost OCE | 0.198479 |
| Original Q2 ESS | 498,853 |
| Q2 full-cost adversary OCE | 0.016342 |
| Q2 full-cost paired MC SE | 0.000299 |
| Q2 full-cost z-score | 54.70 |
| Extended 600-update Q2 OCE | 0.208202 |
| Extended Q2 ESS | 493,933 |
| Q3 incremental low-cost OCE | 0.028334 |
| Q3 ESS | 483,687 |
| First Q3 full-cost audit OCE | -0.000695 |
| First Q3 full-cost paired MC SE | 0.000414 |
| First Q3 full-cost z-score | -1.68 |
| Continued Q3 full-cost OCE | 0.002528 |
| Continued Q3 full-cost paired MC SE | 0.000385 |
| Continued Q3 full-cost z-score | 6.57 |

The first Q3 full-cost audit was not an optimum: its action performed worse
than the available zero action and its gradient was material. A further 100
full-population updates found a statistically significant positive OCE. This is
an important example of why stopping when a nonstationary action happens to be
within Monte Carlo error of zero is unsafe.

Explicit-\(y\) checks:

| Policy | Profiled OCE | Explicit-\(y\) OCE | Analytic \(y^*\) |
|---|---:|---:|---:|
| Original Q2 | 0.198479085705 | 0.198479085705 | -0.198479085705 |
| Extended Q2 | 0.208202155636 | 0.208202155636 | -0.208202155636 |
| Continued full-cost Q3 adversary | 0.002527818902 | 0.002527818902 | -0.002527818902 |

## Observed numerical issues

These statements follow directly from saved objectives, gradients, actions, or
densities.

1. **The action cap is not responsible.** Cap activity is zero in all reported
   Q2 and Q3 runs.

2. **Minibatch noise is not responsible.** Every optimizer update uses the
   exact fixed-population objective and gradient.

3. **The cash intercept is not responsible.** Explicit and profiled OCE values
   agree to numerical precision.

4. **Neither learned policy reached its requested gradient tolerance.** The
   600-update Q2 audit had gradient RMS 0.00623. The continued full-cost Q3
   audit had gradient RMS 0.00266 and maximum absolute gradient 0.0999.

5. **Large radial first-order conditions remain.** For Q2, the tilted expected
   net P&L along the current action was approximately +0.046. For the continued
   full-cost Q3 action it was approximately -0.018 before the terminal audit.
   Thus simple scaling remained an improving direction.

6. **An exact scalar line search materially improves both saved actions.** With
   proportional costs and an inactive cap, P&L is homogeneous along a positive
   scale \(\alpha\). Exact full-population line searches gave:

   | Policy | \(\alpha^*\) | OCE at \(\alpha=1\) | OCE at \(\alpha^*\) | residual scale derivative |
   |---|---:|---:|---:|---:|
   | Extended Q2 | 1.131588 | 0.208202 | 0.211298 | \(2.59\times10^{-8}\) |
   | Continued full-cost Q3 adversary | 0.561148 | 0.002528 | 0.006479 | \(2.98\times10^{-9}\) |

7. **The frozen-plus-residual Q3 parameterization impedes rescaling.** To reduce
   the cumulative action, the residual must approximate the negative of the
   frozen Q2 function instead of directly adjusting Q2's weights.

8. **The deterministic Adam trace oscillates.** Gradient RMS and maximum
   coordinate change sharply between adjacent exact full-population updates,
   even while the longer-run objective trend improves.

## Plausible but not yet proved explanations

The following are hypotheses consistent with the observations; they are not
established results.

1. **Severe curvature anisotropy.** Nearby strikes, overlapping expiries, and
   overlapping hold-to-expiry periods produce highly correlated returns. The
   OCE Hessian may therefore contain both stiff and nearly flat directions.

2. **Redundant parameterization.** The nonlinear network and linear skip can
   represent overlapping state-action functions. This can create flat parameter
   valleys and make parameter-gradient norms less informative than action-space
   directional derivatives.

3. **Adam is poorly matched to an exact deterministic objective.** Adam's
   adaptive stochastic-gradient dynamics may oscillate in stiff directions and
   move too slowly in flat directions. A deterministic line-search or
   limited-memory quasi-Newton method may be more appropriate.

4. **Transaction-cost kinks contribute to gradient jumps.** The current net P&L
   uses \(c|a|\). Whenever an action crosses zero, its transaction-cost
   derivative changes discontinuously.

5. **The current gradient tolerance may be poorly scaled across parameters.** A
   raw parameter-gradient RMS mixes deep-network, skip, and bias directions.
   Economically meaningful projected derivatives, including a cumulative-action
   scale derivative, may be better stopping diagnostics.

## Long/short representation

Writing

\[
a=a^+-a^-,\qquad a^+,a^-\ge0,
\]

makes transaction cost linear in gross legs,

\[
c(a^++a^-).
\]

This is a useful reformulation, but clipping unconstrained logits with ReLU does
not remove every kink: it moves the nondifferentiability to the zero boundary of
each leg. Smooth positive maps such as softplus remove that kink but cannot
represent an exact zero with finite logits. A bounded or projected optimizer on
nonnegative legs is the cleanest formulation; the cap should apply to gross
exposure \(\sum_k(a_k^++a_k^-)\), not only to the net action.

At an optimum with strictly positive transaction costs, simultaneous positive
long and short legs for the same instrument are dominated. Numerically, however,
this complementarity should still be audited.

## Next controlled experiments

1. Re-run Q3 as one cumulative policy initialized from the extended Q2 policy;
   keep all Q3 parameters trainable and use the incremental P&L relative to the
   fixed Q2 baseline only for the density update.
2. Apply an exact scalar line search to the cumulative action during training,
   or add a directly trainable global/per-expiry action scale.
3. Compare deterministic L-BFGS with strong-Wolfe line search against Adam on
   the same fixed population and starting policy.
4. Test a separate long/short policy with gross-exposure cap and nonnegative
   legs, while retaining the signed-action experiment as the control.
5. Report action-space first-order conditions, paired Monte Carlo error, cap
   activity, and ESS alongside raw parameter-gradient statistics.

## Artifact locations

The numerical artifacts are outside the Git tree under:

`/home/hans/quantdata_data/experiments/dysanos-small-book-oce/single-state-25d-1m-nms-minus1p5-plus0p5-11-expiry-v1/`

The principal completed run is:

`linear-skip-full-batch-q1-q2-v2/`

The corrected cumulative-Q3 and quasi-Newton experiments are pending; their
results must not be conflated with the frozen-plus-residual Q3 values above.
