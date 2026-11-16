# Training-objective audit — 14 November 2026

This is an implementation check, not a new performance result. The three
implementations must not all be described as exact reproductions of the papers.
The common 44-origin evaluation does not make their training objectives identical.

## DYSANOS v1: paper structure and corrected midpoint target

Reference: [DYSANOS v4, equations (20), (26), (27) and Appendix A](https://arxiv.org/html/2608.12587v4).

The implementation uses the actual market-refined affine-200 decoder and full
jointly refined five-session VAR. It does not replace the final dynamics by an
eight-factor PCA. Native quote fitting, affine initialization and market
fine-tuning precede alternating quote and transition updates. The final loss is

\[
\mathcal L_{\rm quote}+\mathcal L_{\rm cap}
+0.10\mathcal L_{\rm Gaussian,5D}+0.10\mathcal L_{\rm gauge}.
\]

The dissipative generator, full innovation factor, per-dimension Gaussian
likelihood, equal-date quote weighting and gauge penalty follow the paper.
The rolling 250-close population is an intentional change from the paper's
891-date all-history calibration. Continuing the original schedule after its
300-epoch convergence check fails is also an explicit run policy, not the
paper's fixed epoch count.

**Retired trial:** the original market loader populated `fit_prices` from the
certified SANOS fitted price. The quote objective uses
`excess + 0.05 * abs(price - fit_prices) / half_spread`. Equation (20) uses
the observed bid/ask midpoint for that second term. The actual bid and ask
still determine the dominant excess term. Existing checkpoints are therefore
not certified as exact paper-objective fits. The replacement run
`paper-dysanos1-rolling-20261114` uses `--paper-midpoint`: the centering target
is the observed bid/ask midpoint. It reuses raw cached market observations,
not fitted native, affine or joint weights from the retired run. Each origin
is initialized independently. Fit completion and one-day forecast validity
are recorded separately.

Sources: `model_audit/legacy_rolling.py:market_from_cache`,
`dysanos/flex/training.py:per_date_quote_objective`, and
`dysanos/flex/joint_direct.py:joint_dynamics_loss`, `fit_joint_direct_model`.

A separate diagnostic on the intermediate, unaccepted April 1 checkpoint at
epoch 369 found that its learned weekly covariance has no positive-definite
equal daily embedding (minimum daily eigenvalue approximately -0.03076).
This is not an objective change or a final-fit result. Do not repair it by
eigenvalue clipping or by dividing the weekly covariance by five.

## DYSANOS v2: agreed STRT-16 experiment, not the paper's joint estimator

V2 uses independently reconstructed daily SANOS/DLV coordinates from the saved
smoothness-0.1 market surfaces. It removes the linear spot-related native changes,
constructs a density-weighted full change span, maps that span through the
current straddle-price Jacobian, and retains sixteen principal directions of
straddle changes scaled by the square root of maturity. Thus its projection
optimizes local observable straddle-change reconstruction, not market bid/ask
loss plus a jointly learned VAR likelihood.

The projected states then receive a separately estimated, stability-constrained
daily VAR, with the complete native spot loading restored. The scalar tail
diffusion is fitted from whitened residual radii using the original log-variance
likelihood. The bounded decoder and its clipping diagnostics are retained.
There is no jointly refined affine-200 decoder or five-session VAR in this
variant. Its name denotes the requested comparison variant, not a published
DYSANOS v2 paper model.

Sources: `model_audit/legacy_v2.py:fit_origin`, `independent_tail`,
`model_audit/projection_metrics.py:metric_projection`, `fit_projected`, and
`hedge2/dynamics.py:fit_dynamics`.

## GROMM2: paper quote objective with explicit repository extensions

Reference: [GROMM v2, equations (20)–(26)](https://arxiv.org/html/2610.16732v2).
The user requested the implementation in `~/python/openqdata_models_gromm`.
The exact checked source is frozen and checksummed in the external run's
`SOURCE.json`. It is not modified to make this comparison look paper-identical.

The retired repository pilot targeted certified smoothness-0.1 SANOS prices plus/minus
observed half-spreads, rather than the original observed bid/ask intervals.
With `e = (model_price - SANOS_price) / max(half_spread, 1e-6)` and
`huber(x) = sqrt(1 + x*x) - 1`, its loss is the weighted per-view mean of

\[
\operatorname{huber}((|e|-1)^+)+0.01\operatorname{huber}(e),
\]

plus strike-curvature and expiry-smoothness penalties of weight 0.001 each.
Outside the reported NMS band the quote weight is 0.01. The paper instead uses
linear bid/ask excess plus a 0.05 absolute-midpoint term and a cap penalty.
Both use the bounded-positive DLV decoder, but those training losses are not equal.

The current `repository-gromm2-paper-20261114` run replaces that quote loss
explicitly with the paper's linear bid/ask excess, 0.05 absolute-midpoint
term and restoring cap penalty. Observed bid/ask intervals are used without
the pilot's half-spread floor or smoothing penalties. The change is versioned
in `gromm_paper_objective.PAPER_OBJECTIVE` and every Adam checkpoint identity.
No old-objective weights or optimizer moments are loaded. Checksummed CPU
views, geometry and numerical initializers may be reused after validating
their source and data identities and replacing their quote targets.

Dynamics use aligned fixed-cash-strike/fixed-expiry coordinate AR(1) regressions,
smoothed innovations, density-weighted factors, and a latent-X spot likelihood
consistent with the independent Black smoothing factor. This differs from the
paper's PIT-score regression and residual PCA. GROMM2 also intentionally changes
the tenor/grid geometry and backbone, as documented by its repository. The
compatible P/Q drift construction is retained; this does not make calibration
identical to the paper.

Sources in the frozen repository: `calibrate.py:_view_loss`, `fit`,
`estimate.py:estimate`, `latent_fit`, and `pipeline.py:market_price_of_risk`.
Our `model_audit/repository_gromm2.py` adds exact Adam-state restart checkpoints
and resource-managed execution. `model_audit/gromm_paper_objective.py` supplies
the explicit paper quote-loss override; the frozen source is not edited.

## Execution boundary

One allocation is registered to task `01a12f09-4975-7431-a499-7c321ab35fa5`,
named “Hedging — DYSANOS and GROMM”. DYSANOS v1 GPU fitting runs while one CPU
worker prepares GROMM2's quote views and initializers. Only fully prepared
GROMM origins may enter GPU training. Both 44-origin fit queues and the
GPU-dependent GROMM P/Q forecasts share the allocation; the controller releases
it when GPU work ends, including ordinary error exits. Resource-stop handling
retains the allocation pending the host-policy release approval. Errors and
checkpoints are preserved, not treated as completed model results.
