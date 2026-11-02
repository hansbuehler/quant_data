# Strict conditional-audit research snapshot: 2 November 2026

This is an implementation checkpoint, **not a conditional-martingale pass**.
The active goal remains unfinished. Raw populations, native checkpoints,
training logs and immutable run declarations are retained outside Git under
`/home/hans/quantdata_data/experiments/gromm2/`.

## Scope and original gates

The current population uses the 29 August 2025 `(-4,3)` model, 16 joint
factors, eight substeps per day, 100,000 antithetic paths and 125 business
days. The agreed audit book is spot plus eleven strikes at every fifth-day
expiry, not every interior model strike. Returns retain their original Q
weights; there is no return centering or fitted measure change.

Entropy and squared actions require separate full affine-plus-deep fits on
both outer training folds. Antithetic pairs remain together. The opposite
fold is untouched during fitting; original normalization, training masks,
objectives and stationarity thresholds must remain fixed.

The latest accepted full-network endpoint is fold 0, squared, native iteration
1628. Its raw gradient norm is `0.3434736036148236` versus the original gate
`3.6155365702241535e-8`; its affine-metric norm is `0.00019809816340876917`
versus `5.24887111114172e-7`. **Neither gate passes.** The other three actions
have not completed strict full-network convergence. There is no four-policy
freeze or completed fresh-population confirmation.

## Numerical work

The strict continuation, verification and bundle modules preserve original
fold/data identities, raw optimizer history, model parameters, RNG and work
counts. Numerical proposals do not alter the simulated Q law. Their inner
linear-system tolerances are not substitutes for the original nonlinear
gradient gates.

- Exact-H trust-boundary and direct-history shadows were not adopted.
- The periodic-restart shifted-H trial exhausted 120 Hessian products.
- Removing unconditional ten-iteration restarts solved the shifted system in
  76 iterations, but its accepted shadow worsened the affine residual; it was
  not adopted. Smaller initial damping also exhausted its 120-product budget.
- The current read-only trial uses unshifted exact-H, full-reorthogonalized
  P-weighted Arnoldi with frozen positive history preconditioning. It is not
  SPD-CG or Euclidean MINRES on a nonsymmetric left-preconditioned operator.
  CPU regressions and a tiny managed-GPU parity check pass; the complete
  population result remains pending at this snapshot.

The new proposal requires independently confirmed original-metric residuals,
strict canonical descent and non-worsening predicted gradient norms before
canonical Armijo/exact-H remainder checks. Rank-unreliable, nonconverged,
non-descent and zero proposals cannot become successful policy artifacts.
Work caps, zero actions and objective decreases alone do not prove stationarity
or absence of arbitrage. Existing unconditional rejections remain reported.

## Entry points

- `weighted_arnoldi_audit.py` and `minimum_residual_audit_proposal.py`: the
  current experimental unshifted numerical proposal and deterministic tests.
- `shifted_exact_audit_proposal.py`, `shifted_exact_gate_audit_proposal.py`
  and `trust_region_audit_optimization.py`: provenance-preserved control
  methods, not default certified replacements.
- `strict_population_verification.py`, `strict_action_verification.py` and
  `strict_action_bundle.py`: original-phase verification/publication machinery.
  They do not yet support a future native Arnoldi training phase; a shadow
  must not be relabelled as a supported trained checkpoint.

The active external trial is `unshifted-arnoldi-probe-20261102-v1/`.
Its source-pinned declaration, independent reviews, progress ledger and final
report are authoritative; no outcome is implied by committing these sources.
