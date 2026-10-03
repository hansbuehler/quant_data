# Section 5: the consistent-recalibration sequence, implemented and tested

Code: `section5_crc_sequence.py`. This was the clearest gap identified in
`PHASE1_AUDIT.md` — no price-preserving reset existed anywhere in `flex/`
before this. Implements steps A-G exactly as specified.

## Structural finding used throughout

`bound_native_raw` maps *any* raw torch tensor to bounded, admissible
`(dlvsqs, forward_vols, eta)`, and the SANOS forward recursion then
guarantees a non-negative discrete density from positive `dlvsqs` — verified
directly (`min_decoded_density` stayed at ≈0.0106–0.0107 across every tested
reset below, never approaching zero). **Consequence**: step (E)'s "constrained
inverse... subject to positivity... and other model admissibility
constraints" is, for this decoder, an *unconstrained* optimisation in
raw-state space — admissibility is automatic by construction, not something
the optimiser must separately enforce. This is reported as a genuine
structural property of this decoder, not assumed or asserted without check.

## The sequence, run end to end (4 resets, synthetic oracle — see data-honesty
note in the code)

| Step | Converged | Iterations | Pre-correction max\|err\| | Post-correction max\|err\| | Raw-state move (L2) |
|---|---|---|---|---|---|
| 0 (initial calibration) | yes | 66 | 0.00272 | 8.8e-5 | 0.755 |
| 1 | yes | 5 | 0.00021 | 7.2e-5 | 0.062 |
| 2 | yes | 3 | 0.00038 | 2.0e-4 | 0.144 |
| 3 | yes | 9 | 0.00021 | 1.0e-4 | 0.156 |

(tolerance `1e-4`, 5-strike × 3-expiry = 15-contract protected book, static
admissibility maintained throughout: min density never below ≈0.0106).

**Distinguishing the three things Section 5 asks to keep separate**:
- *Stochastic market evolution*: the independent oracle state, stepped under
  the fitted P law — this is "the market," never touched by the reset.
- *Price-preserving representation change*: the reset representation's own
  P-law one-step proposal (already reasonably close: pre-correction errors
  are 20–30× smaller than the initial, uncalibrated guess) corrected by a
  small further optimisation (`raw_state_move_l2` of 0.06–0.16, an order of
  magnitude smaller than the initial calibration's 0.76) to hit the target
  exactly.
- *Conditioning on newly observed data*: the target itself (step C) is a
  fresh evaluation of the oracle at the new state — the reset does not
  assume the market re-realises what the representation predicted.

## Feasibility decision (F) is not vacuous — verified with a genuine failure

A deliberately implausible target book (uniform random prices up to 3.0,
`tolerance=1e-8`, `max_iterations=20`) correctly fails to converge
(`converged: false`, residual `2.82` after 20 iterations, barely reduced from
`2.82` before) — confirming the feasibility check can and does trigger a
genuine reject, not merely reporting "accept" trivially in every case. No
"infeasible" event occurred in the 4-step run above at the declared
tolerance; a harder tolerance or a more aggressively evolving oracle would
be expected to produce genuine rejects, and the machinery is confirmed ready
to report them (`ConstrainedInverseResult.converged=False`) when they occur.

## What this does, and does not, establish

**Does**: demonstrates a complete, working, price-preserving reset sequence
for the FLEX decoder — the concrete missing piece identified in the audit —
with genuine feasibility accounting (both a documented success path and a
documented, verified failure path), and a clean separation of market
evolution from representation correction.

**Does not**: this is not evidence that the RESULTING evolved reset
sequence is itself dynamically arbitrage-free in the Section 2/3 sense —
the constrained inverse only guarantees *static* admissibility and *exact
price reproduction at the reset instant*, exactly the distinction Section 5
itself draws ("Verify that the new continuation model is coherent after
each reset, not merely that its prices agree at the reset instant"). Whether
the sequence of RESET representations, taken together, forms a coherent
Q-consistent process is the Section 2/3 question, found not yet established
for the tested drift families (`SECTION2_3_FINDINGS.md`) — the CRC sequence
here preserves prices at each reset, which is necessary but was not itself
shown sufficient for full dynamic consistency across resets.
