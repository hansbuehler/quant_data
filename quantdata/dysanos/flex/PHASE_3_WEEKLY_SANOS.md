# Phase 3: weekly hierarchical FLEX-SANOS

## Status and objective

This document specifies Phase 3. It is an implementation contract, not a
record of completed code or empirical results. Phase 3 learns weekly dynamics
under the historical measure $P$, simulates the dynamics and spot entirely in
Torch, and decodes every simulated state through the selected statically
arbitrage-free FLEX-SANOS surface.

The production Phase 2 representation is **Strong**, the 140-coordinate
expiry-triangular affine model. The word `strong` in Phase 3 must refer to this
specific allocation:

```text
d = (16, 19, 17, 15, 13, 10, 10, 10, 5, 5, 5, 5, 5, 5),  sum(d) = 140.
```

The core Phase 3 outputs are:

1. an all-components and a variance-reduced hierarchical weekly state model;
2. differentiable Torch path generation in normal-spot and implied-spot modes;
3. fixed-cash-strike option returns over one period and to expiry; and
4. temporally held-out likelihood and calibration diagnostics at 1, 2, and 4
   weeks.

Only physical GPU 0 may be used for Phase 3 training and production
evaluation. CPU preprocessing is allowed. GPU 1 must remain unused unless the
user changes this instruction.

### Canonical state notation

For every expiry `j`, `N_x=N+1+1=202`:

- \(x_t^j\) is the constrained SANOS block consisting of the DLV-squared,
  forward/backbone-volatility, and forward-\(\mu\) parameters;
- \(u_t^j\in\mathbb R^{N_x}\) is its unconstrained full-support preimage under
  the exponential/sigmoid transforms; and
- \(h_t^j\in\mathbb R^{N_h^j}\) is the compressed state used by the weekly
  AR(1) and any later target-measure kernel.

A hierarchical decoder satisfies
\(u_t^j=\operatorname{Decoder}_j(h_t^1,\ldots,h_t^j)\). An unrestricted
decoder is called **direct**. In **full** mode, \(N_h^j=N_x\) and the decoder
is the identity, so \(h_t^j=u_t^j\). Any centered, standardized, or PCA
coordinates used internally by an estimator are merely coordinates of \(h\)
and receive other symbols; they are not additional model states. This notation
supersedes the provisional use of \(D\) for raw state and \(x\) for PCA state.

Phase 3 first learns a train-through-2024 market-to-Strong encoder and does not
replay historical innovations. Historical calibrated Strong states supply its
supervision targets; untouched 2025 quote sets are the temporal encoder gate.
The normal-spot model is the primary calibrated $P$-model.
Implied-spot mode is a direct target-law sampler whose gross spot-return
marginal is the current front-week SANOS law. Phase 3 does not parameterize or
fit a measure change; changing the weekly mean and covariance is deferred.

## Frozen Phase 2 input

### Market geometry

There are $J=14$ fixed-tenor expiry buckets and $N=200$ inner strikes. In
business-year units,

\[
T_1=\frac{4.8}{\mathtt{DAYS\_PER\_YEAR}},\qquad
T_j=T_{j-1}+\frac{5}{\mathtt{DAYS\_PER\_YEAR}},\quad j=2,\ldots,14.
\]

The inner NMS coordinates span $[-3,2]$, contain exactly one zero, and use
canonical `NMS`, not `NMS5`. Given bounded strike forward volatilities
\(\sigma_{t,j}\in[0.01,1]\), define

\[
A_{t,j}=\sum_{\ell\le j}\sigma_{t,\ell}\sqrt{T_\ell-T_{\ell-1}},
\qquad T_0=0.
\]

If \(z_i\) is inner NMS node $i$, the inner normalized strike is

\[
k_{t,j,i}=\exp\!\left(z_i\sqrt{
 \frac{\sigma_{t,\epsilon}^2\epsilon+A_{t,j}^2}{1+\epsilon}}
 \right),\qquad
\sigma_{t,\epsilon}=\frac{A_{t,1}}{\sqrt{T_1}},
\]

where \(\epsilon=\mathtt{NMS.EPS}\). The two boundary strikes are 50% of the
lowest and 150% of the highest inner strike.

Phase 3 simulates and trades these 14 weekly model expiries only. It does not
create 0DTE instruments, intermediate market expiries, or left-extrapolated
tradables. The poor historical 0DTE-2DTE fit remains relevant to how the
Strong state was inferred, but it is not part of the simulated option
universe.

### Strong triangular decoder

Let \(h_t=(h_t^1,\ldots,h_t^{14})\), where
\(h_t^j\in\mathbb R^{d_j}\), denote the daily Strong state stored by the
selected checkpoint. Each decoder output block has 202 unconstrained coordinates,

\[
u_t^j=(u^{\mathrm{DLV}}_{t,j,1:200},
       u^{\sigma}_{t,j},u^{\mu}_{t,j}).
\]

The decoder is one affine layer with a bias per output coordinate and an exact
expiry-triangular mask:

\[
u_t^j=b^j+\sum_{\ell\le j}W^{j\ell}h_t^\ell.
\]

There are no hidden layers and the activation is `Identity`. It is important
not to describe this as a nonlinear network. The affine weights, biases, and
daily states were jointly fine-tuned through the quote objective after
sequential conditional-PCA initialization.

The parameter transforms map \(u_t^j\) to the constrained SANOS block
\(x_t^j\):

\[
\begin{aligned}
\mathrm{DLVsq}_{t,j,i}
 &=0.01^2+(3^2-0.01^2)\operatorname{sigmoid}(u^{\mathrm{DLV}}_{t,j,i}),\\
\sigma_{t,j}
 &=0.01+(1-0.01)\operatorname{sigmoid}(u^\sigma_{t,j}),\\
\mu_{t,j}
 &=0.01+(0.5-0.01)\operatorname{sigmoid}(u^\mu_{t,j}).
\end{aligned}
\]

Those equations describe the retained Strong checkpoint and therefore remain
part of its provenance.  The flat native-encoder experiment adds a new DLV²
gauge without silently reinterpreting that checkpoint.  In the new branch,

\[
\widetilde{\mathrm{DLVsq}}_{t,j,i}=0.01^2+
 \exp(z^{\mathrm{DLV}}_{t,j,i}),\qquad
\mathrm{DLVsq}_{t,j,i}=
 \min\!\left(\widetilde{\mathrm{DLVsq}}_{t,j,i},3^2\right).
\]

Thus the pricing cap is reached at the finite raw coordinate
\(z_{\max}=\log(3^2-0.01^2)\); it is not approached through a saturating
sigmoid.  Forward volatilities and forward mus retain their sigmoid bounds.
Training adds the AAE-equivalent smooth one-sided cap penalty

\[
\frac12\sum_{j,i}
 \left[
  \frac{w}{s}\operatorname{softplus}
   \left(\frac{z^{\mathrm{DLV}}_{t,j,i}-z_{\max}}{w}\right)
 \right]^2,
\qquad w=0.10,\quad s=0.35.
\]

Pricing is flat above the cap, while this separate penalty supplies the
restoring gradient.  The width and scale are inherited research
hyperparameters and must be reported rather than treated as calibrated facts.

The backbone total volatility is not \(\mu_j A_j\). It is the cumulative
forward-mu curve

\[
B_{t,j}=\sum_{\ell\le j}
 \mu_{t,\ell}\sigma_{t,\ell}\sqrt{T_\ell-T_{\ell-1}}.
\]

The Torch SANOS recursion uses the 200 DLV squares on each 202-point support
to solve a tridiagonal density step. Each decoded density has unit mass and
unit normalized mean. A normalized call at a model expiry is a mixture of
zero-carry Black calls with support points as the conditional spots, mixture
weights equal to the SANOS density, and backbone total volatility \(B_{t,j}\).
Between model expiries, FLEX linearly interpolates the two conditional call
prices and linearly interpolates \(B^2\). Before \(T_1\), it interpolates from
the deterministic time-zero density and zero backbone variance. Puts follow
from normalized put-call parity. Rates, dividends, discount factors, and
forward levels do not enter after market normalization.

The completed Strong fit used 874 dates from 2022-01-03 through 2025-08-29 and
3,492,782 eligible call and put books. Its reported objective was `1.260246`,
and `70.9779%` of books were inside bid/ask. The fit is not uniformly good at
the short end:

| bucket | inside bid/ask | conditional mean excess | conditional median excess |
| --- | ---: | ---: | ---: |
| all | 70.9779% | 4.0448 | 1.3545 |
| 0DTE | 25.8848% | 13.2011 | 11.7118 |
| 1DTE | 22.7264% | 15.7325 | 8.9768 |
| 2DTE | 31.0106% | 11.4926 | 7.3429 |

These figures are inherited input diagnostics, not Phase 3 performance.
The source report is
[the completed rebalanced Strong report](reports/batch-descent/rebalanced-stronger/spx-2022-2025-forward-mu-v1/spx-2022-2025-forward-mu-v1.json).
It records checkpoint SHA-256
367fe85e67b0a09719c74d72dca5ab5fbfb16564e3c5828b13e482a3cfbb638a
and latent-state SHA-256
99d0054a3bf3e03113571d1a571b1af2722d5e60e509d2006d5fb3210ddfe0f3.
Phase 3 must verify, not merely copy, those identities when loading the
selected artifact.

## Source data and temporal split

### Market pipeline

For each date, use `quantdata.dysanos.market.smooth_interpolated_market()` via
the FLEX adapter in `quantdata/dysanos/flex/market.py`:

1. `filter_options_day()` loads and filters the option market and supplies
   normalized prices, `spot_ref`, and expiry metadata.
2. `smooth_interpolated_market()` fits the large smooth arbitrage-free SANOS
   surface and stores its fitted prices as `af_call_mids` and `af_put_mids`.
3. The FLEX adapter reconstructs the serialized `ArbFreePricer` only to obtain
   the ATM curve used by the exact NMS eligibility filter. The option fitting
   target itself is already the payload's `fit` price.
4. FLEX retains active finite European call and put books separately, through
   the last exact model expiry and within the per-expiry canonical-NMS range.

Phase 3 needs the date, Strong state, and `spot_ref`. `FlexMarketDay` currently
does not expose `spot_ref`, so the implementation must carry it from the
underlying payload or add it to a Phase 3 data record. Join states and spots by
the exact ordered market dates and validate the checkpoint market fingerprint;
never join by row number alone.

### 2022-2025 observations and strict split

Use every valid Strong date from 2022 through the end of the available August
2025 sample. There are 874 usable dates:

| role | dates | observations |
| --- | --- | ---: |
| training | 2022-01-03 through 2024-12-31 | 712 |
| temporal validation/OOS | 2025-01-02 through 2025-08-29 | 162 |

There are 241, 240, 231, and 162 dates in 2022, 2023, 2024, and 2025,
respectively. No 2026 data may enter fitting, hyperparameter selection,
normalization, PCA, or reporting.

This is the required primary split. Do not use the 2025 observations to select
PCA rank, regularization, optimizer stopping, covariance floors, or any other
hyperparameter. Pair dates with the business calendar, not row offsets. The
current source has 671 valid five-business-day training pairs and 155 valid
five-business-day OOS pairs. The 10- and 20-business-day OOS evaluations have
150 and 140 valid endpoint pairs, respectively. Report the actual counts from
the loaded, fingerprinted dataset rather than assuming these audit counts.

Every origin date has total statistical weight one in PCA diagnostics,
dynamics fitting, and validation, irrespective of how many option quotes were
available on that date. A transition likewise has total weight one. The
millions of Phase 2 quote observations must not be treated as independent
dynamics observations.

A clean end-to-end experiment has an explicit prerequisite ordering:

1. keep the selected Strong allocation and triangular affine architecture, but
   refit and freeze its decoder/loadings using dates no later than 2024-12-31;
2. train a set-invariant market-to-Strong-state encoder on 2022-2024 quotes,
   first against the refitted direct daily states and then through the real
   bid/ask-plus-fit SANOS quote objective; and
3. evaluate that frozen encoder on 2025 quotes by the same quote objective
   before using its 2025 states for dynamics validation.

The encoder should use the previously selected transformer or Gaussian radial
basis style of set aggregation, not a generic slow Deep Set. Every input date
has total training weight one regardless of quote count. Its precise
architecture is still outstanding and must be versioned and selected without
2025 data.

Its unordered quote records must include at least expiry, NMS/log-strike,
call/put side, bid, ask, smooth fit, half-spread, and the available log-volume
and log-open-interest fields, with normalization fitted on 2022-2024 only.
The output is the 140-coordinate Strong block layout. Tests must permute quote
order without changing the output, and the report must show direct-state MSE
before fine-tuning plus the standard quote-containment/excess statistics after
fine-tuning, overall and at the short end.

Only after this encoder gate passes should the weekly dynamics be fitted to
the 2022-2024 encoded Strong states and validated on the directly encoded 2025
states. Optimizing an independent held-out state against each 2025 quote
surface is a useful oracle diagnostic, but it is not the live encoder result.

The currently selected Strong decoder and loadings were trained through
2025-08-29. Using that artifact unchanged is acceptable only for a preliminary
**dynamics-on-a-fixed-representation** diagnostic. It is representation-leaky
and must not be called a clean end-to-end OOS test. The report must record
which provenance and encoder gate were used.

## Per-expiry state PCA

PCA is fitted separately to each expiry block and only on training dates. For
the clean run, these are the frozen encoder outputs; independently optimized
daily embeddings are an oracle-only diagnostic. For block \(j\), center the
Strong state levels and compute

\[
h_t^j=\bar h^j+E_jp_t^j,
\]

where the columns of \(E_j\) are deterministic-sign PCA directions. Provide
two modes:

- `pca_all`: \(r_j=d_j\), so \(E_j\) is square and no state variation is
  discarded;
- `reduced`: \(r_j\) is the smallest rank whose cumulative training level
  variance is at least 99.5%.

The 99.5% comparison uses the population covariance over training dates. Do
not inflate retained components to restore discarded variance. Decode a
simulated \(p^j\) back to \(h^j\) before applying the
frozen triangular decoder.

The completed 2022-2024 direct-Strong diagnostic gave these ranks:

```text
pca_all = (16, 19, 17, 15, 13, 10, 10, 10, 5, 5, 5, 5, 5, 5) = 140
reduced = (10, 14, 15, 14, 12, 10, 10, 10, 5, 5, 5, 5, 5, 5) = 125
```

These values are diagnostics for independently calibrated Strong embeddings,
not hard-coded ranks. The production run must recompute PCA on the frozen
train-through-2024 market-encoder outputs and report achieved variance by
expiry. The 140-to-125 direct-state result warns that this PCA may not create a
substantially smaller dynamic model.

PCA variance in latent coordinates is not invariant to an invertible rescaling
of \(h\), and the affine decoder admits such gauge transformations. The primary
experiment freezes the checkpoint gauge exactly. As a diagnostic, also report
decoder-output reconstruction error by expiry after PCA. Any column
standardization, whitening, or decoder-metric PCA is a separate experiment and
must not silently replace the frozen-gauge result.

Before fitting dynamics, audit the representation used by that candidate. For
the retained Strong checkpoint, report raw-coordinate quantiles and the
fraction of sigmoid probabilities below and above configured tail thresholds.
For the capped-unbounded DLV² branch, report the fraction of DLV nodes at or
above \(z_{\max}\), their maximum log excess, and the per-surface cap penalty;
forward volatilities and forward mus still use sigmoid-tail diagnostics. Repeat
the audit on simulated paths before accepting them. Do not clip a simulated
state back into its historical range. Frequent sigmoid saturation or cap
contact is model-failure evidence. Diagnostic thresholds and reductions belong
in external trainer/evaluation audits or debug-only verification, not inside
the compiled simulation hot path.

## Hierarchical weekly dynamics under P

### Conditional state construction

Use a five-business-day base step, denoted Δ=1 week. This is an explicit
Musiela/fixed-time-to-expiry convention. The fixed-tenor buckets are curve
coordinates: after one step, an old bucket-\(j\) contract has rolled
to bucket \(j-1\), while \(h_{n+1}^j\) describes a new option with \(j\) weeks
remaining. This distinction is essential for fixed-contract returns.

Standardize each retained training PCA coordinate by its population standard
deviation and write the resulting row vector as \(z_t\). For expiry block
\(j\), fit the adjacent-business-session affine equation

\[
z_{t+1}^j=c_j+z_t^jB_{jj}
 +\mathbf 1_{j>1}z_t^{j-1}B_{j-1,j}+e_{t+1}^j.
\]

The previous expiry enters only at time \(t\): there is no contemporaneous
\(z_{t+1}^{j-1}\) regressor. Assemble the block-upper-bidiagonal daily
transition \(B\), solve its affine stationary equilibrium, and require the
principal logarithm of the complete non-symmetric matrix to be real, stable,
and numerically reliable. Define

\[
K_d=-\log B,
\qquad
\mathcal A=\exp(-5K_d)=B^5.
\]

The matrix exponential creates transitive effects from earlier expiries while
preserving the expiry-causal ordering. No symmetrization, eigenvalue clipping,
coefficient scaling, or silent fallback is permitted. The five-session affine
shift is computed from the same estimated equilibrium. Stationarity applies
to the complete SANOS surface state; spot level is deliberately excluded.

### Stability and mean reversion

The complete daily transition must pass the explicit generator round trip

\[
\frac{\|\exp(-K_d)-B\|_F}{\|B\|_F}
\le \varepsilon_K.
\]

Stable eigenvalues are not sufficient. Also fail the fit if the maximum
standardized transition-power norm over the configured long horizon exceeds
its cap, or if fitted invariant variance divided by empirical training variance
falls outside the configured interval. Estimate an unrestricted exact-week
surface innovation covariance on exact five-session residuals and apply only a
reported numerical eigenvalue floor. This retains measured cross-expiry shock
dependence without a contemporaneous state regressor.

On the production market-encoder history, 2022--2024 provides 675 exact daily
pairs and 671 exact weekly pairs. Reduced PCA has ranks
`(8,13,13,12,12,10,10,10,5,5,5,5,5,5)`, total 118. Its accepted weekly
transition has spectral radius `0.82869`, maximum standardized power norm
`1.25708`, invariant/training variance ratio `0.94824`, and longest front
half-life `3.69` weeks. PCA-all-140 mode has spectral radius `0.86185`, maximum
power norm `1.34275`, variance ratio `0.95280`, and longest front half-life
`4.66` weeks.

Report, by expiry and for the assembled surface process:

- one-week conditional log likelihood, RMSE, and \(R^2\);
- every eigenvalue of the daily blocks and of the full \(K_d\);
- mean-reversion half-lives
  \(\log(2)/\operatorname{Re}(\lambda(K_j))\) in weeks;
- innovation eigenvalues and cumulative variance;
- condition numbers and covariance floors;
- empirical versus fitted stationary means and covariances; and
- the spectral radius of the assembled one-week transition.

For integer \(a\ge1\), use \(\mathcal A(a)=\mathcal A^a\). When aggregating a fitted
weekly Gaussian innovation over several weeks, include all propagated
innovation covariance terms; do not multiply one-week standard deviation by
\(\sqrt a\) unless the transition is the identity. The implementation may
advance one exact weekly step repeatedly, which is the reference path.

The step is always one week and its innovation law is time-homogeneous.
Precompute the complete one-week transition, affine shift, innovation
covariance, and covariance factor outside the inner Torch simulation loop and
register them as model tensors. Also precompute the 2W and 4W exact propagated
transition, covariance, and factor tensors used by validation. The inner normal-mode
step is matrix multiplication plus a covariance-factor multiplication; it
must not call torch.matrix_exp or factor a covariance per path or per step.

### Identifiability and regularization

There are 671 five-business-day training pairs in the current audit. Dense
own-lag and preceding-expiry coefficient matrices can be weakly identified even
though each individual block is at most 19-dimensional. The implementation
must include and report:

- a diagonal-own-lag baseline;
- the requested stable projected full-own-lag model;
- explicit ridge penalties on the structural regressions;
- train-only sensitivity to those penalties; and
- parameter count versus effective transition count.

The ridge coefficients are ad-hoc hyperparameters. They must live in the
versioned configuration and appear in every report. Select them with a
train-only temporal exercise, for example fitting 2022-2023, scoring 2024, and
then refitting the selected fixed configuration on all 2022-2024 dates. This
inner split is itself ad hoc and must be recorded. If the full model does not
improve held-out likelihood over diagonal \(K\), retain the diagonal model as
the statistical baseline but report the full model as the requested result;
do not silently substitute it.

## Joint spot model

Let \(s_n=\log S_n\). Surface dynamics remain autonomous: spot does not enter
any \(h^j\) transition. Fit the weekly log return jointly with the front-expiry
innovation after fitting the surface, following the DYSANOS surface-first
construction. If \(z_{n+1}^1\) is the whitened front-block innovation, fit

\[
\Delta s_{n+1}=a_s+\beta_s^\top z_{n+1}^1
                  +\sigma_{s,\perp}z_{n+1}^{s},
\qquad z_{n+1}^{s}\sim N(0,1),
\]

where \(z^s\) is independent of all surface innovations. Always retain the
orthogonal spot factor, including in all-components PCA mode. Report the front-state
explained fraction and \(\sigma_{s,\perp}\). The fitted spot level has a unit
root; only the surface state is stationary.

The direct-Strong 2022-2024 diagnostic attributed `0.49563` of weekly spot
innovation variance to the whitened front block. This is not a production
encoder-state validation result.

### Normal spot mode

`normal` is the primary physical-measure mode. Draw fresh Torch Gaussian noise
at every weekly step, apply the equation above, and update

\[
S_{n+1}=S_n\exp(\Delta s_{n+1}).
\]

There is no historical innovation sampling or path replay.

### Implied spot mode

`implied` keeps the fitted Gaussian copula between the front surface
innovation and spot but replaces the one-week Gaussian spot marginal. Let

\[
z^{\mathrm{cop}}=
\frac{\beta_s^\top z^1+\sigma_{s,\perp}z^s}
     {\sqrt{\beta_s^\top\beta_s+\sigma_{s,\perp}^2}},
\qquad u=\Phi(z^{\mathrm{cop}}).
\]

Decode the **current**, pre-transition front-week surface and let
\(F_{\mathrm{SANOS},n}^{(1)}\) be its complete normalized terminal-spot CDF.
Then

\[
\frac{S_{n+1}}{S_n}
=\left(F_{\mathrm{SANOS},n}^{(1)}\right)^{-1}(u).
\]

This must use the full distribution priced by FLEX, including the Black
backbone, not just the discrete DLV density. If front support points are
\(k_a\), masses are \(p_a\), and front backbone total volatility is \(B_1\),
the gross-return CDF for \(g>0\) is

\[
F_{\mathrm{SANOS}}(g)=\sum_a p_a\,
 \Phi\!\left(\frac{\log(g/k_a)+\tfrac12B_1^2}{B_1}\right).
\]

Implement its inverse with a bracketed monotone Torch solve and explicit tail
tolerances. The production Strong bounds imply \(B_1>0\); an invalid zero or
negative front backbone is rejected. The current implementation uses 64
bisection steps, a ten-standard-deviation bracket, and the dtype-aware clamp
\(32\epsilon_{\rm machine}\). These are ad-hoc numerical constants and must be
promoted to configuration before model selection depends on them.

Use only information available at the start of the transition. In particular,
do not use \(x_{n+1}\), the realized next spot, or a future fitted surface to
construct the inverse CDF.

The implied marginal is the risk-neutral front SANOS law inserted into the
selected weekly copula. It is not a \(P\)-spot model and must not be scored or
labelled as a \(P\) likelihood. Under a named target measure \(Q\), it makes the
front call automatically consistent:

\[
c_t^1(k)=E_Q[(S_{t+1}/S_t-k)^+\mid\mathcal F_t].
\]

It is sampled directly, not by drawing \(P\)-paths and attaching likelihood
weights. Its mean-one gross-return property replaces the fitted normal-mode
drift.

## Exact-week sampling-measure interface

Implementation update: the shared-strikes branch now implements this interface
in `weekly_measure.py` and trains it in `weekly_measure_training.py`. The
paragraphs below retain the design rationale; `PHASE_3_WEEKLY_CODE.md` records
the precise conditional-normal and triangular-generic parameterizations now in
code. The older Strong-140 production branch remains a P-only milestone.

Any future change of measure is to be defined on the assembled exact one-week transition,
after the per-expiry PCA coordinates, hierarchical conditioning, matrix
exponentiation, and covariance propagation have been composed. It is not
defined on an Euler increment or on an individual PCA shock.

Let \(Y_{n+1}=(\Delta s_{n+1},h_{n+1})\), matching the implementation's joint
innovation ordering, where \(h\) is the complete
stationary SANOS surface state, weekly spot return is an output, and log-spot
level is updated separately as a unit root. Under the fitted reference
measure,

\[
Y_{n+1}\mid h_n\sim
N\!\left(m_P(h_n),Q_P(h_n)\right).
\]

The shared-strikes branch exposes a first-class Torch measure kernel which maps the reference conditional
law and current state to a target conditional law:

\[
\mathcal M_\vartheta:
\left(h_n,m_P(h_n),Q_P(h_n)\right)
\longmapsto
\left(m_{\mathbb M}(h_n),Q_{\mathbb M}(h_n)\right).
\]

Such an identity kernel would return \(m_{\mathbb M}=m_P\) and
\(Q_{\mathbb M}=Q_P\) exactly. A non-identity kernel may change both the
conditional mean and conditional covariance. Parameterize the target
covariance by a lower-triangular factor with strictly positive softplus
diagonal plus a dtype-aware floor, or by another construction that guarantees
positive definiteness by design. Sampling is direct:

\[
Y_{n+1}=m_{\mathbb M}(h_n)+L_{\mathbb M}(h_n)z_{n+1},
\qquad
L_{\mathbb M}L_{\mathbb M}^\top=Q_{\mathbb M},
\qquad z_{n+1}\sim N(0,I).
\]

The implementation generates no reference paths, likelihood ratios,
or path weights. A mean-only change with fixed covariance would be the
Gaussian/Girsanov-style special case. Changing covariance is a more general
discrete one-week transition-kernel change and must not be mislabelled as
continuous-time Girsanov. These kernels are research candidates layered on the
frozen calibrated P model; they are not yet production risk-neutral measures.

Implied spot applies the front-SANOS inverse
to a standardized target-law scalar. Any target covariance transformation
must be restandardized before \(\Phi\), and its Gaussian mean shift excluded,
so the rank remains uniform and the exact front SANOS marginal is preserved.

### Weekly HJM/roll restriction hook

For later expiries, the target weekly measure must also satisfy the normalized
zero-rate roll identity

\[
c_t^j(k)=E_Q\left[
 R_{t+1}\,c_{t+1}^{j-1}(k/R_{t+1})\mid\mathcal F_t
\right],\qquad j=2,\ldots,J,
\]

where \(R_{t+1}=S_{t+1}/S_t\). The front case rolls to payoff as above. The
Torch simulator exposes the pathwise residuals

\[
R_{t+1}c_{t+1}^{j-1}(k/R_{t+1})-c_t^j(k)
\]

(and front payoff residuals), plus a reusable instrumented moment-loss helper.
Conditioning instruments must use only the current state. Future
measure-change research can use this hook to learn the HJM restriction. It is
not optimized or used as a Phase 3 fit/validation criterion, and Phase 3 does
not claim that its calibrated \(P\) model satisfies these conditional moments.

For the hierarchical target-measure construction, apply the roll identity
with a one-expiry index shift. Denote by
\(\widehat h_{t+1}^{j\to j-1}\) the compressed state generated by transition
line \(j\): the left index identifies its originating current maturity and the
right index its remaining tenor after the weekly roll. Once drawn, it is
stored as \(h_{t+1}^{j-1}\). It must not be confused with the newly created
cross-sectional state \(h_{t+1}^{j}\). In full mode this notation remains
unchanged because \(h=u\) and the decoder is the identity.

Use the fitted hierarchical \(P\)-AR(1) draw as the reference proposal. Sample
\(R_{t+1}\) through the current front SANOS inverse CDF, then learn a
candidate-\(Q\) conditional kernel for
\(\widehat h_{t+1}^{2\to1}\), conditional on the realized lower/common factor,
\(R_{t+1}\), and the relevant current state including \(h_t^2\). Decode the
rolled front-week option through \(h\to u\to x\to\mathrm{SANOS}\) and choose
the kernel so its conditional expectation matches the current two-week
option. Thus the state modified by the measure change is precisely the state
appearing in the payoff integrand.

Continue front-to-back. Current \(j\)-week options constrain the target kernel
for \(\widehat h_{t+1}^{j\to j-1}\), conditional on the already realized lower
roll lines, spot, and the relevant current prefix. This is the intended
hierarchical-\(h\) construction. The new longest-tenor state remains governed
by a regularized boundary/reference law unless the calibration surface is
extended by another current maturity.

This iterative construction requires the \(h\to u\) decoder to be
expiry-triangular: once a rolled option moment has been matched, later
roll-line draws must not alter its decoded value. For a direct decoder, learn
one joint target kernel for \((R_{t+1},h_{t+1})\) and impose all maturity-roll
conditions together. The chain-rule decomposition of such a joint kernel into conditional densities is
only a sampling factorization; because its components must be fitted jointly,
it is not the iterative maturity-by-maturity measure change proposed here.

The raw count of option strikes versus stochastic coordinates is only a first
feasibility diagnostic. A mean-only normal change with \(d\) effective controls
can locally match \(n\) independent moments only if the moment Jacobian has
rank \(n\le d\). A full covariance change has additional controls, while a
generic conditional density or invertible transport has functional degrees of
freedom and may match more moments than the noise dimension. Strike-price
conditions are also not independent because one terminal distribution imposes
monotonicity and convexity across the curve. Report the moment-Jacobian
singular values, effective rank, optimized residual, and kernel conditioning;
do not infer existence or impossibility from quote count alone.

The expiry-triangular decoder, per-expiry PCA, conditional adjacent-expiry
dynamics, and fixed-tenor roll mimic an HJM/Musiela maturity hierarchy. SANOS
enforces static no-arbitrage on every decoded surface. They do **not** impose
the complete risk-neutral HJM drift restrictions on dynamic option prices.
Calibrating a target-measure kernel to those restrictions is future work and
must not be claimed by Phase 3.

## Torch-native simulation contract

The path model, random draws, matrix exponentials, covariance factors,
front-to-back transition, implied-CDF inversion, SANOS decoding, option
pricing, returns, and costs must remain Torch tensors on one device. Do not
pre-generate NumPy paths and copy them to Torch. The simulator must:

- accept either a `torch.Generator` or caller-supplied standard-normal tensors;
- support a reparameterized differentiable path from supplied noise to prices
  and returns;
- avoid `.detach()`, `.cpu()`, NumPy, and SciPy in the simulation path;
- preserve the `BaseEnvironment` dtype and device.

The current reference implementation is eager and unblocked. A compiled path
and path-blocking wrapper are performance follow-ups; they must not be claimed
until identical-noise equivalence is separately tested.

Configuration and fitted parameters must be versioned and checkpointed with
the exact Strong checkpoint digest, market fingerprints, PCA basis, date
split, optimizer state, scheduler state, RNG state, and convergence audit.

Use these axis names consistently:

```text
P : simulated paths
H : option-entry/trading weeks requested by the caller
J : 14 model expiry buckets
I : maximum nonzero tradable strikes per expiry
R : total retained PCA state size (140 full, data-selected in reduced mode)
```

At minimum, expose the following logical operations:

```text
fit(dates, h_states, spot_ref) -> fitted hierarchical model
step(h, log_spot, noise, spot_mode) -> next_h, next_log_spot
simulate(initial_h, initial_spot, H, noise, spot_mode) -> state/spot path
decode(h) -> u -> x -> SANOS surface
price(surface, spot, cash_strikes, remaining_expiries, option_side) -> prices
option_returns(..., return_mode, cost_rates) -> prices, returns, costs, metadata
```

Names may follow repository conventions, but their timing and tensor shapes
must be equivalent and documented in public docstrings.

## Tradable options and return definitions

### Instrument construction

The caller supplies either one shared log-moneyness vector or one strictly
increasing vector per expiry,
\(\kappa_{j,i}=\log(K_{j,i}/S_{\mathrm{entry}})\). The simulator flattens the
possibly unequal expiry grids and returns expiry and side metadata for every
instrument. Do not force the 200-point SANOS density support to be the traded
grid.
At entry week \(n\), path \(p\), define the fixed cash strike

\[
K_{p,n,j,i}=S_{p,n}\exp(\kappa_{j,i}).
\]

This cash strike and its calendar expiry remain fixed after entry. Add one
zero-strike call to every expiry independently of the supplied grid. It has
price \(S_{p,n}\) at every positive remaining expiry and payoff \(S\), and is
the spot/delta-hedging instrument. The flattened metadata uses `-inf` only as a
masked log-moneyness sentinel; every numerical strike and pricing operation
branches on the zero-strike flag before exponentiation or division.

For nonzero strikes, default `itm` selection is:

- call when \(K\le S_{\mathrm{entry}}\);
- put when \(K>S_{\mathrm{entry}}\).

`otm` reverses those choices. The option side is fixed at entry and is not
reclassified when spot moves. The ATM tie-to-call convention is ad hoc and
must be configurable if both ATM sides are needed. Zero-strike calls are
always retained and always ITM.

With zero rates and forwards, if \(o_n^\pm(T,k)\) is the normalized FLEX price,
the cash value of the fixed contract is

\[
V_{p,n}^{i,j}=S_{p,n}\,
 o_n^\pm\!\left(T_j,\frac{K_{p,n,j,i}}{S_{p,n}}\right).
\]

### One-period return

For an option entered at week \(n\), the one-week liquidation value is

\[
X_{p,n}^{i,j}=\begin{cases}
S_{p,n+1}\,o_{n+1}^\pm
 \left(T_{j-1},K_{p,n,j,i}/S_{p,n+1}\right),&j>1,\\
(S_{p,n+1}-K_{p,n,1,i})^+,&j=1\text{ call},\\
(K_{p,n,1,i}-S_{p,n+1})^+,&j=1\text{ put}.
\end{cases}
\]

The gross simple period return is

\[
R_{p,n,\mathrm{period}}^{i,j}=\frac{X_{p,n}^{i,j}}{V_{p,n}^{i,j}}-1.
\]

This is a return on the same cash strike and calendar-expiry contract, not a
comparison of the same moving moneyness coordinate.

### Return to expiry

Hold an option entered at \(n\) in bucket \(j\) to week \(n+j\). With
\(\Pi^\pm(S,K)\) denoting call or put payoff,

\[
R_{p,n,\mathrm{expiry}}^{i,j}
=\frac{\Pi^\pm(S_{p,n+j},K_{p,n,j,i})}{V_{p,n}^{i,j}}-1.
\]

To produce \(H\) entry cohorts for all \(J\) expiries, period mode needs states
through week \(H\), while expiry mode needs spot and current state through week
\(H-1+J\). Tail states beyond the \(H\) entry weeks need not be returned. The
simulator may stream them while keeping only outstanding strikes and the
eventual return in an output of logical shape

```text
(P, H, Q)
```

where \(Q\) is the flattened number of valid options, including one zero call
per expiry. The returned expiry/side/zero-strike metadata reconstructs the
logical expiry grids without padded fake instruments.

The \(T_1=4.8\)-business-day option is settled at the next five-business-day
simulation step. This 0.2-day overnight convention is inherited and ad hoc;
record it in metadata.

### Proportional costs

Let \(c\ge0\) be a proportional half-spread rate relative to contemporaneous
option value. For a one-period long round trip, report

\[
R_{\mathrm{net}}
=\frac{X-V-c(V+X)}{V}.
\]

At cash settlement charge entry cost \(cV\) but no exit option spread. Return
\(V\), \(X\), entry and exit cost amounts, their sum, and the gross return
separately so a Torch trading loop can
apply costs to position turnover instead of assuming a long round trip. A
price floor or eligibility mask is required before dividing by \(V\), and its
value is an outstanding ad-hoc numerical/economic choice; the current code has
both a dtype-tiny return guard and, for the feed-forward cost model, the
explicit input filter and tick floor documented below.

The real half-spread model is separate from the completed Strong artifact.
Cost rates may be supplied explicitly to simulation, inferred by the saved
Torch model, or supplied by the grid configuration. Zero-call/spot cost remains
separate in every case.

### Half-spread models

Fit two execution-cost models on CPU using real eligible call and put books.
The current benchmark retains every positive-time book supplied by the FLEX
market adapter, including intermediate market expiries, while excluding exact
zero-time books. The simulated universe remains weekly-only. For quote \(i\)
on date \(d\), the regression target is normalized price half-spread

\[
y_{d,i}=\frac{\mathrm{ask}_{d,i}-\mathrm{bid}_{d,i}}{2}.
\]

The simulation-facing proportional rate is computed afterwards as
\(\widehat y/\max(\mathrm{option\ price},p_{\min})\). This price-based target
was selected because it corresponds directly to quoted execution cost. Report
both price-unit errors and proportional-ratio diagnostics so the explicit
floor cannot hide instability on cheap options.

The required CPU models are:

1. a linear model with intercept and features
   \(\mathrm{vega}/\sqrt{\mathrm{TTE}}\), signed delta, NMS, and any explicitly
   documented side indicator needed to distinguish calls and puts; and
2. a light nonlinear model using vega, \(\sqrt{\mathrm{TTE}}\), NMS,
   log-strike, signed delta, and the same side convention.

Both estimators fit half-spread in levels and clamp negative predictions to
zero. The light model is a histogram gradient booster with 100 iterations,
learning rate 0.05, 15 leaves, minimum leaf size 100, and L2 penalty
\(10^{-6}\). These choices, squared-error loss, and the clamp are ad-hoc fixed
hyperparameters, not choices selected on 2025.

A third model is the simulation-facing differentiable alternative. It is a
two-hidden-layer width-64 SiLU network with inputs vega, square-root TTE, NMS,
log strike, signed delta, and log approximate dollar option price. It predicts
log normalized half-spread with a date-weighted Huber loss. Approximate dollar
price is normalized AF price times spot, so neither rate nor forward is an
input. Books below $0.25 approximate premium are removed before fitting. At
inference, the prediction is floored at half the configured SPX tick: $0.025
below $3 premium and $0.05 otherwise. The `maximum` floor is differentiable
almost everywhere and remains native Torch code inside trading-loop paths.

The $0.25 filter, $3 threshold, tick values, network width/depth, log target,
Huber beta, and optimizer schedule are explicit ad-hoc choices. Primary metrics
use the filtered universe; the report also evaluates every book under the
tick-floor extrapolation. On the fixed 2025 split the filter retains 99.984% of
books and the network achieves date-balanced MAE `0.00013639`, RMSE
`0.00025765`, and R2 `0.49975`.

Each date has total loss weight one, so each quote on date \(d\) receives
weight \(1/n_d\). This prevents a date with more option quotes from dominating
the cost fit. Use the same strict 2022-2024 training and 2025 temporal OOS
split as the dynamics. The current report provides raw and date-balanced MAE,
RMSE, and R2, proportional-cost quantiles, and date-balanced calibration by
business DTE, NMS, and signed delta. Additional vega, price, and call/put
calibration cuts remain desirable diagnostics rather than completed outputs.

Vega and signed delta are zero-carry Black quantities inferred from the smooth
AF fitted price: puts are first converted to calls by parity, implied
volatility is inverted, call delta is \(\Phi(d_1)\), put delta is
\(\Phi(d_1)-1\), and vega is \(\phi(d_1)\sqrt T\). They are not finite
differences of the Strong decoder. The zero-strike call is absent from this
regression and uses the separate spot-cost configuration.

## Training, convergence, and validation

Fit both `pca_all` and `reduced` variants on the same 2022-2024 data. Fit the
surface process first, then the conditional spot equation. The projected ridge
fit is a closed-form NumPy/SciPy estimation and may run on CPU; it has no
iterative optimizer whose last epoch could be called convergence. Its gate is
successful finite linear algebra, full diagnostics, stable projected blocks,
a real positive-stable full generator, positive covariance, and non-collapsed
innovations. Iterative encoder training must separately pass its declared
convergence audit. Any Torch training or production evaluation may use only
physical GPU 0.

Phase 3 is structural research around the later construction of risk-neutral
measures. Held-out likelihood, market fit, and forecast scores are diagnostics,
not production-quality acceptance thresholds. A weak fit remains an acceptable
research result if estimation genuinely converged, the state process is finite,
stable and non-collapsed, simulation preserves the structural invariants, and
comparisons with honest train-only baselines are shown. Poor performance must
be reported without relabelling it as a numerical or data-pipeline failure.

The primary validation is temporal and uses January-August 2025 only. For
every valid origin, construct realized endpoints at exactly 5, 10, and 20 business days;
do not approximate horizons by row offsets when a market date is missing.
Report the exact origin and endpoint dates and exclusions.

### Predictive diagnostics

Only normal mode enters the primary
\(P\)-likelihood validation. For that mode, the hierarchical model is affine
Gaussian after recursive substitution. Compute the exact 1W, 2W, and 4W predictive mean and covariance,
including the spot coordinate, and report:

- joint and per-coordinate negative log likelihood;
- state-block conditional log likelihood by expiry;
- realized spot log-return log density and PIT;
- 50%, 80%, 90%, 95%, and 99% spot interval coverage;
- state RMSE/MAE and covariance calibration;
- energy score for the joint state and spot return; and
- comparison with random-walk, diagonal-AR, and unconditional-Gaussian
  baselines fitted only on training data.

Implied spot mode is not scored as a \(P\) likelihood. Report its realized
spot-return CDF rank, interval coverage, and direct-simulation distribution as
a separate target-law diagnostic. If its own hybrid one-step or multiweek
density is estimated, label that density by the implied target law and never
place it in a \(P\)-likelihood comparison. Multiweek distributions use direct
Torch Monte Carlo paths; no likelihood weighting is permitted.

Use 2025-01-02 as the default single illustrated historic origin because it is
the first OOS date and supports all three horizons. This date is an ad-hoc
reporting choice; aggregate rolling-origin results are authoritative.

Rolling five-business-day origins and multiweek targets overlap. Confidence
intervals and model comparisons must therefore use a horizon-aware moving
block bootstrap or HAC adjustment. Report unadjusted observation counts and
effective/non-overlapping counts. Even with 162 OOS dates, dependence and the
relatively short market regime limit power, especially at 4W.

### Simulation validation

From the illustrated origin and from a sample of other OOS origins:

1. simulate with fixed supplied noise in eager mode; when compiled execution
   is added, require agreement within dtype tolerance;
2. when path blocking is added, compare blockwise and unblocked generation
   using identical supplied noise;
3. verify full and reduced paths are finite and decoder densities retain unit
   mass, unit mean, and non-negativity tolerance at every checked step;
4. compare long-run simulated surface moments with the fitted stationary law;
5. verify no lookahead by perturbing future observed states and requiring the
   same simulated paths from the same initial state and noise; and
6. validate period and expiry returns against hand-computed flat/constant
   paths, including \(j=1\), zero-strike calls, puts, costs, and the extended
   expiry horizon.

## Ad-hoc choices and unresolved conventions

Every item below must remain visible in configuration or reporting:

| item | current specification | status |
| --- | --- | --- |
| state allocation | Strong `[16,19,17,15,13,10,10,10,5,5,5,5,5,5]` | empirically selected in Phase 2 |
| PCA cutoff | 99.5% per expiry | user-selected; gauge-dependent |
| live state initialization | train-through-2024 Gaussian-RBF set summary and width-512 SiLU market encoder | implementation/training prerequisite |
| saturation thresholds | configured sigmoid-tail probabilities | diagnostic/ad hoc |
| training/OOS split | 2022-2024 / Jan-Aug 2025 | strict chronological split |
| weekly step | 5 business days | user-selected |
| first expiry | 4.8 business days | inherited overnight convention |
| drift form | standardized lag-only adjacent daily hierarchy, full \(K_d=-\log B\) | selected against full-K controls |
| stationary center | affine equilibrium estimated on 2022--2024 daily pairs | finite-sample/ad hoc |
| ridge strength | zero default | OLS follows DYSANOS/grid; sensitivity required |
| full-transition numerical gate | real/stable log, exp/log round trip, max power norm, and invariant/training variance bounds | thresholds are ad hoc |
| covariance floor | dtype-scaled positive floor | numerical, must be reported |
| ATM side | call in ITM mode | ad hoc |
| traded log-strike grid | caller supplied | unresolved production grid |
| option price floor | $0.25 feed-forward input filter; dtype-tiny return guard; `1e-6` spread-ratio floor | ad hoc |
| zero-call transaction cost | `spot_proportional_cost`, default zero | ad hoc |
| spread target | normalized price half-spread in levels | ad hoc; report proportional diagnostics |
| light spread model | fixed small histogram gradient booster | ad hoc, not 2025-selected |
| Torch spread model | width-64 x 2 SiLU, log-Huber, explicit SPX tick floor | ad hoc, not 2025-selected |
| target measure | not parameterized in Phase 3; normal mode is calibrated under \(P\) | deferred research |
| implied inverse-CDF tolerances | 64 bisections, 10-sigma bracket, `32*eps` clamp | hard-coded/ad hoc |
| implied multiweek density estimator and path count | one fixed validated choice | unresolved/ad hoc |
| illustrated OOS origin | 2025-01-02 | ad hoc; rolling results dominate |

## Principal risks and acceptance criteria

The principal statistical risk is over-parameterization: 671 overlapping
training transitions in the current audit are being used to estimate a
hierarchical system whose retained state is 140 dimensional in `pca_all` mode and
is expected to remain near-full after the production PCA. Per-expiry
factorization makes estimation possible but does not create information.
Regularization sensitivity, baseline comparisons, and uncertainty intervals
are mandatory.

The principal representation risk is that Strong was optimized for static
quote fit, not temporal smoothness, and its latent gauge is not intrinsically
identified. The preliminary Jan-June PCA ranks suggest that the production
state will remain near-full, but the
authoritative 2022-2024 ranks remain to be computed. Poor 0DTE-2DTE fit may
also contaminate front-state inference even though weekly tradables avoid
intermediate short-end repricing.

The principal measure risk is implied spot mode. FLEX-SANOS supplies a
risk-neutral terminal marginal, while the dynamics are fitted under \(P\).
Implied mode is useful for testing and for the later sampling-measure
discussion, but it is not evidence that the physical spot law has been
learned.

Phase 3 is accepted only when:

- the train-through-2024 Strong refit and set-invariant market encoder converge,
  remain finite and non-degenerate, and report their frozen 2025 quote objective
  before dynamics validation; no minimum 2025 fit score is imposed at this
  structural-research stage;
- full and reduced closed-form fits pass finite-completion, non-collapse,
  transient-amplification, invariant-scale, decoded-saturation, and stability
  checks; both current production encoder-state fits pass this condition;
- all production simulation and pricing operations are Torch-native on GPU 0;
- both spot modes and both return modes pass timing, shape, formula, gradient,
  and no-lookahead tests;
- the weekly universe contains no 0DTE or intermediate-expiry instruments;
- all spread models use equal total date weights and report strict 2025 temporal
  OOS performance; the Torch model additionally proves gradient propagation and
  its explicit tick floor;
- reports identify representation provenance and every ad-hoc choice;
- 1W/2W/4W rolling OOS results include dependent-sample uncertainty; and
- `PHASE_3_WEEKLY_CODE.md` records the implemented formulas, checkpoint
  identities, test commands, convergence audits, and measured validation
  results without overstating the small sample.
