# Sections 2-3: the pricing identity, tested; Q-feasibility, ranked

Code: `section2_pricing_identity.py`. Data honesty: synthetic-but-structurally-valid
simulator (same construction as the existing test suite's `_simulator`/`_history`
fixtures) — the local test-mode fixture data cannot support a genuinely fitted
multi-year P law (see `PHASE1_AUDIT.md`). Every number below is a
machinery/feasibility validation, not a claim about real SPX dynamics.

## Section 2: the discrete-time pricing identity

Derived directly (not merely asserted): for the normalised decoder
`H(t,S_t,h_t;T,K) = S_t · c(h_t; T-t, K/S_t)`, the discrete-time martingale
condition `E_Q[H(t+Δ,Z_next;T,K)|Z_t] = H(t,Z_t;T,K)` reduces, after dividing
by `S_t` and writing `R = S_{t+Δ}/S_t`, to exactly the object
`WeeklySanosSimulator.hjm_roll_residuals` already computes pathwise:
`R·c_next[m-1](k/R) − c_current[m](k)` for non-front expiries, with the correct
`relu(R−k)` terminal payoff at the front. This is confirmed by direct
derivation from the identity, not by trusting the existing docstring.

**Tested, well-powered (100,000 paths, one weekly step, 3 model expiries × 5
strikes = 15 (T,K) pairs — a declared finite protected book, no claim about
the rest of the surface):**

| Law | max|t-stat| across the 15 tested contracts | mean |residual| by expiry |
|---|---|---|
| Raw fitted P (AR(1), untouched) | **16.9** | [0.00056, 0.0052, 0.0067] |
| Gradient-trained `NormalWeeklyMeasureKernel` (600 epochs, instrumented on `[1, current_state]`, converged: training loss 2.8e-5 → 1.2e-5) | **8.9** | [0.00045, 0.0043, 0.0053] |

**Finding, not assumed**: the raw P law decisively fails martingality (exactly
as the codebase's own `hjm_roll_moment_loss` docstring already states it does
not claim otherwise). The existing target-measure-kernel machinery, trained to
convergence on this synthetic data, **reduces but does not eliminate** the
residual — still highly statistically significant. This is the honest,
measured answer the assignment asks for, not an assumption in either
direction.

## Section 3: drift feasibility — rank, conditioning, residual

Linearised (discrete-time analogue of the continuous-time tangent-space test
used throughout this research programme, e.g. `heston_bates_oracle`'s Task
5/8 compensator-range analysis): does an additive mean-shift
`μ ∈ ℝ^{state_dim}` to the P-law's next-period state exist that cancels the
**mean** roll residual across all 15 tested contracts simultaneously, to
first order? Jacobian computed via autograd at `μ=0` (60,000-path Monte Carlo
mean, 15×8 Jacobian).

```
residual_dim = 15, mu_dim = 8
jacobian_rank = 6          (RANK-DEFICIENT: 2 of 8 directions inert)
jacobian_condition_number = inf
r0_norm = 0.02639
residual_norm_after_linearised_correction = 0.00814
reduction_fraction = 0.691  (NOT 1.0)
```

**Finding**: a general additive-mean-shift drift family is rank-deficient for
this decoder/instrument set (the longest-dated expiry block's 2 coordinates
have zero first-order effect on the tested residual within one week — a
plausible, checkable structural fact, not a numerical artefact: confirmed
`mu_star`'s last two components are exactly 0), and even restricted to its
own 6-dimensional achievable subspace, cannot fully cancel a 15-dimensional
residual target (69% reduction, not 100%). **This is not evidence that no
richer family can work** (a full covariance tilt, a larger instrument set, or
Route-B-style structural constructions are untested) — it is evidence that
*this specific, simple linear-in-mean family* is insufficient, reported
precisely as such, mirroring exactly the honesty standard
`heston_bates_oracle`'s Task 8b used for its own family-by-family verdicts.

**Answering "does the existing decoder admit the required Q-dynamics":** not
established by either of the two families tested here (the existing trained
target-measure kernel, or a first-order additive mean-shift). Section 5 below
proceeds with the CRC recalibration *sequence* regardless, since that
sequence's own step (E) — the constrained inverse under exact price
preservation — is a different, more tractable object than a global dynamic
no-arbitrage guarantee, and is the assignment's most clearly identified gap
(Phase 1 audit: no such step exists anywhere in `flex/` today).
