# Task 0 audit: DLV × extended-Bates + Section 12 CRC reserve construction

Scope: audits `heston_riccati.py`, `reserve_crc.py`, `bates_transform.py`
against the DYSANOS II manuscript's `thm:dlv-bates`, `cor:bates-offgrid-drift`,
and Section 12 (`lem:bcrc-branching`, `thm:bcrc-reserve`, the synthetic reset
sequence). Written before claiming exactness, per the task instruction.
Status legend: **PROVEN** (closed-form/algebraic, checked numerically as a
regression, not as the proof itself), **VERIFIED** (numerically confirmed to
near machine precision against an independent computation), **ASSUMED /
NOT YET BUILT** (required by the theorem but not implemented in this
repository as of this task), **OPEN** (a genuine gap or unresolved question).

## 1. True martingality

**PROVEN + VERIFIED.** `D(u,τ)` solves the Riccati ODE with `D(u,0)=0`; at
`u=-i`, `u²+iu=0` identically, so the RHS of the Riccati equation vanishes
whenever `D=0`, and by ODE uniqueness `D(-i,τ)≡0` for all `τ≥0` — not just at
`τ=0`. Hence `A(-i,τ)=κθ∫₀^τD dr ≡0` too. The Gaussian jump exponent
`j_p(u)` is written in the manuscript's own compensated Lévy-Khintchine
normalisation, so `j_p(-i)=λ_J[e^{m_J+δ_J²/2}-1-κ_J]=0` identically for
every `(λ_J,m_J,δ_J)`. Both facts are checked in
`test_bates_crc_heston_riccati.py` to `<1e-13`.

For the **full combined transform** (background + reserve correction),
`test_martingale_normalisation_full_transform_pre_and_post_reset` checks
`Φ_t(0,T)=Φ_t(-i,T)=1` to `<1e-8` (quadrature-limited, not floating-point
limited) at every one of the manuscript's four states (initial + after each
of the three resets) and every one of the four model expiries. This is a
genuine end-to-end check of the assembled object, not an inference from the
individually-verified pieces.

**Caveat (OPEN, deferred to Task 4/#108):** this establishes martingality of
the Y-factor's own Q-dynamics only. `thm:dlv-bates` additionally requires
`S=XY` itself to be a martingale under the product measure, which needs the
X-chain's own martingale property (checked separately, see §2) *and* a
genuine independence argument (also §2) — neither has been assembled with
the Bates Y-factor in this repository yet, so the full product martingality
of `S` is **NOT YET VERIFIED** end-to-end. Do not read §1's checks as
covering `S`.

## 2. Conditional independence of X and Y

**ASSUMED / NOT YET BUILT.** The current task's deliverables so far
(`heston_riccati.py`, `reserve_crc.py`, `bates_transform.py`) implement only
the Y-factor (the extended-Bates background) and its reserve CRC. The
X-factor (coherent latent DLV) and the product construction `S=XY` are
Task 4/#108, not started as of this audit. Consequently the independence
hypothesis in `thm:dlv-bates` is **not exercised by any code in this
repository yet** — it cannot be verified or refuted until §4's finite-state
X-chain exists and is combined with Y. Flagged here explicitly rather than
silently assumed away.

## 3. Terminal settlement

**ASSUMED / NOT YET BUILT**, for the same reason as §2: terminal settlement
of the product `S_T=X_TY_T` against the finite-mixture pricing formula
`C_t(T,K)=Σ_iq_i(t,T)E[(x_iY_T-K)^+|·]` requires the X-chain, which is
Task 4/#108. What IS checked here: `fourier_call_price`'s own terminal
condition (`T≤t` returns the intrinsic payoff `max(y_forward-strike,0)`
directly, not through the transform) and — for `T>t` — the Fourier price is
checked to lie in `[0,1]` (a "pure"-unit forward-normalised call must be
between zero and the forward) across every state and strike tested
(`test_fourier_call_price_is_nonnegative_and_below_forward`).

## 4. Finite horizon

**PROVEN by construction.** `T_*=20/252` is a fixed finite constant in the
manuscript's own synthetic sequence; all quadratures (`integrated_reserve`,
`fourier_call_price`) are over compact intervals `[t,T]⊂[0,T_*]`; no
component requires `τ→∞` limits. `moment_strip_report` additionally checks
that the moments needed for the chosen Fourier damping (`α=1.5`, i.e.
`p_max>α+1=2.5`) are finite at the horizons actually used — this is a
genuinely necessary side-condition for the Fourier inversion to converge at
all, distinct from finite-horizon per se, and is checked in
`test_moment_strip_report_shrinks_with_maturity` (monotonicity) but should
additionally be checked at the exact `(α,τ)` pairs used by
`fourier_call_price` in the Task 4 verification sweep — **not yet done
there**, noted as a follow-up.

## 5. Lévy integrability

**PROVEN.** Every jump law in this construction (the current `p_n`, the
dominating envelope `ν̄`, every box member) is Gaussian, hence has finite
moments of every order and trivially satisfies the small-jump integrability
condition `∫_{|z|≤1}z²ν(dz)<∞` and the compensation condition
`∫_{|z|>1}(e^z-1)ν(dz)<∞` required for a well-posed compensated Poisson
integral. No numerical truncation of the jump measure's support is used
anywhere (`gaussian_bates_jump_exponent` is closed-form). The dominating
property itself (`ν(dz)≤ν̄(dz)` pointwise, box-uniformly) is checked
directly as a density-ratio bound
(`verify_envelope_dominates`, checked against both a dense numerical grid
and the closed-form bound) — this is an ordinary calculus inequality on a
ratio of two explicit densities, not a characteristic-function positivity
claim, so it carries none of the caveat in §7 below.

## 6. Jump-compensation convention

**PROVEN + VERIFIED.** `j_p(u)=λ_J[e^{ium_J-δ_J²u²/2}-1-iuκ_J]`,
`κ_J=e^{m_J+δ_J²/2}-1`, matches the manuscript's own
`eq:bcrc-lk-martingale` normalisation exactly (the `-iuκ_J` term is exactly
the "`-iu(e^z-1)`" compensator, specialised to the Gaussian jump-size law).
Cross-checked in `test_gaussian_jump_exponent_matches_direct_expectation_integral`
against an **independent** 80-node Gauss-Hermite quadrature of
`λ(E[e^{iuZ}]-1-iuE[e^Z-1])` computed directly from the Gaussian density —
i.e. not merely re-deriving the same closed-form algebraically, but
recomputing the defining expectation integral by an unrelated numerical
method. Agreement to `<1e-9`.

## 7. Heston branching argument (`lem:bcrc-branching`)

**PROVEN analytically, with an explicit non-numerical argument — per the
task's instruction not to use a frequency sweep as proof.** Two
complementary classical results are used, both cited (not re-derived) and
both stated precisely in `reserve_crc.py`'s module docstring:

1. **Fixed-`r` positivity of `D(u,r)` as a genuine characteristic exponent.**
   For `θ=0` (zero immigration), `E[exp(iuZ_r^{(v,0)})]=exp(vD(u,r))` for
   the *actual* random variable `Z_r^{(v,0)}` (log-return of a
   zero-immigration Heston process from initial variance `v`) — standard
   affine-model theory (Duffie–Pan–Singleton 2000), following from
   well-posedness of the CIR SDE. So for each fixed `r`, `v↦exp(vD(u,r))`
   is a genuine characteristic function at every `v≥0`; the family is a
   continuous convolution semigroup in `v`, which by Sato (*Lévy Processes
   and Infinitely Divisible Distributions*, Thm 9.8, applied to the
   `v`-indexed semigroup) has a Lévy–Khintchine representation with a
   **nonnegative** triplet.

2. **Validity of the immigration/reserve terms `κθ∫D dr` and
   `κb_k∫D(·,s-t_k)ds`.** This audit adds a second, independent argument not
   previously spelled out in the module docstring: both terms are
   "immigration-rate × time-integral-of-D" objects of *exactly* the same
   algebraic form as each other and as the ordinary Heston `A(u,τ)` term.
   The classical affine-process admissibility criterion (Duffie–Filipovic–
   Schachermayer 2003, "Affine Processes and Applications in Finance",
   Thm 2.7) requires precisely that such an immigration/killing rate be
   **nonnegative** for the resulting exponential-affine object to be a
   bona fide characteristic exponent. `κθ≥0` is enforced by
   `HestonParams.__post_init__` (`theta<0` raises); `κb_used[k]≥0` is
   enforced by `apply_reset` (raises if `b_used[n]<-1e-12`, clamps the
   `[-1e-12,0]` slack to exactly `0`). So the reserve term's admissibility
   reduces to a nonnegativity constraint that is already enforced as a hard
   invariant in code, not merely hoped for — this is the cleanest and most
   decisive part of the branching-lemma audit.

**Supplementary numerical check, explicitly NOT treated as proof:**
`branching_lemma_sanity_check` checks the *necessary* (not sufficient)
condition `|exp(vD(u,r))|≤1+ε` at a finite grid of `(u,r,v)`. Its docstring
and its returned `note` field both state in plain language that this is not
a proof of positivity. This directly follows the task's explicit
instruction: *"Do not use a numerical frequency sweep as proof of
Lévy-measure positivity."*

## 8. A non-obvious structural finding: `p_n` cancels out of the priced dynamics

Not requested explicitly by the task's audit checklist, but discovered while
assembling `bates_transform.py` and important enough to report prominently.
`continuation_transform` is

```
Φ_t(u,T) = exp{ V_t D(u,τ) + A(u,τ) + τ·j_{p_n}(u) + ∫_t^T ℓ_s(u) ds }
```

with `ℓ_s(u) = j̄(u) - j_{p_n}(u) + κΣ_k b_used[k]1{s≥t_k}D(u,s-t_k)`. The
`τ·j_{p_n}(u)` term and the `-∫_t^Tj_{p_n}(u)ds=-τ·j_{p_n}(u)` term inside
the reserve integral **cancel exactly**, for *every* `t,T`, not only before
the first reset (both terms use the same fixed snapshot
`p_n=state.jump_params_path[-1]` throughout a single evaluation, regardless
of how many future resets fall inside `[t,T]`). Verified as a regression:

- `test_continuation_transform_matches_hand_derived_closed_form`: an
  independently-written closed form `exp{V_tD+A+τj̄(u)+κΣb_used[k]∫D ds}`
  (no `p_n` term at all) matches `continuation_transform` to `<1e-9`.
- `test_jump_parameters_cancel_exactly_from_the_transform`: replacing the
  state's current jump parameters with an arbitrary *other* admissible
  (box-contained) jump law, holding `b_used`/`theta_path` fixed, changes
  `Φ_t(u,T)` by `<1e-9` — i.e. genuinely unchanged.

**Interpretation:** `p_n` never enters the priced Q-dynamics directly. It
only matters through the *feasibility* constraints it must have satisfied
at past resets (the admissible-`Δ` interval and the jump-box membership
check in `apply_reset`). The actually-priced process is always driven by
the fixed dominating envelope `ν̄` plus the scheduled, sign-constrained
reserve corrections `κb_used[k]D(u,s-t_k)`. This is precisely the point of
a *consistent recalibration* construction — the locally-calibrated
practitioner parameters `p_n` are bookkeeping for what a recalibration
*could* look like, while genuine Q-consistency is carried entirely by the
envelope + reserve terms. It is a clean, structurally reassuring finding,
not a defect — but it means any future report language must not claim that
`p_n` "drives" the priced dynamics; only the envelope and reserves do.

## 9. Independent pricing check (Task 1's requirement)

**VERIFIED.** Before the first reset (`T≤t_1`), §8's cancellation means the
reserve sum contributes nothing (no `t_k≤s` for `s<t_1`), so the priced
dynamics reduce exactly to Heston diffusion + Gaussian jumps under the
dominating envelope `ν̄` alone — a plain Bates model with fully explicit,
closed-form parameters. A from-scratch Monte Carlo simulation
(`gauss_hermite_mc_cross_check`, QE-free Euler variance stepping + exact
compound-Poisson-Gaussian jump sum, sharing no code with the Fourier
pricer) agrees with `fourier_call_price` to within 6 Monte Carlo standard
errors at two independent strikes and seeds
(`test_fourier_call_price_matches_independent_monte_carlo_before_first_reset`,
`..._with_a_second_strike_and_seed`; `n_paths=400,000`,
`n_var_substeps=100`). This is Task 1's "independent pricing check";
**it covers only the pre-first-reset window** — extending it past a reset
would additionally require simulating the branching-type reserve jump
term directly (a genuine Lévy process in its own right, not an i.i.d.
compound-Poisson sum), which is out of scope for this audit and flagged as
a follow-up for Task 4's verification battery (composition/consistency
checks there do not require simulating that term, since they compare two
*analytic* transforms against each other rather than against simulation).

## 10. Exact reset-preservation for protected options (added after Task 4's verification battery)

**PROVEN + VERIFIED, decisively.** `thm:bcrc-reserve`'s reset-preservation
claim is precise about *which* options are protected: options with
maturity `T≤t_n` (settling before or exactly at reset `n`) must be
perfectly unaffected by reset `n`, while options with `T>t_n` are *allowed*
to move (that is what recalibrating to new information means — it is not,
and should not be, a global price-invariance claim). Both halves were
checked directly:

- **Protected (`T<t_n`):** `verification.protected_option_sweep`, across
  all three scheduled resets, four maturity fractions each, and an
  independent four-point `u`-mesh (covering the whole protected-option
  book per reset, not a single probe), found `|Φ_before−Φ_after|` between
  `2e-20` and `4e-19` — i.e. **exactly zero up to floating-point noise**,
  for every one of the 3×4×4=48 combinations tested
  (`test_protected_option_sweep_all_resets_all_strikes_exact_invariance`).
- **Unprotected (`T>t_n`):** a direct check (not yet promoted to a formal
  test, run interactively during this audit) found `|Φ_before−Φ_after|`
  of order `1e-7` to `9e-6` at `T` up to `4×t_n` past the reset — small in
  absolute price terms but **not** zero, exactly as expected: these
  maturities are *supposed* to respond to the recalibration.
- **Negative control:** `verification.uncompensated_transform` (prices
  directly off `p_n` with the reserve/envelope machinery removed) VIOLATES
  protected-option invariance by `~8e-5` at the same test points where the
  real construction gives `<1e-8` — three orders of magnitude larger,
  confirming the test has power and the reserve mechanism is load-bearing,
  not a vacuous formality (`test_negative_control_uncompensated_transform_violates_invariance`).

This is the single most decisive confirmation obtained in this task that
the reserve-CRC construction, as implemented, actually delivers the
"consistent recalibration" property it is meant to: pre-existing,
about-to-settle option prices are provably untouched by a recalibration
event, to machine precision, while the reserve bookkeeping correctly lets
longer-dated prices move.

## 11. Off-grid generator residual

**VERIFIED.** `verification.y_factor_generator_residual` checks that
`d/dT log Φ_t(u,T)`, computed two independent ways (central finite
difference in `T` vs. the analytic RHS assembled from the Riccati
equation's own defining ODE plus the current jump exponent plus
`ℓ_T(u)`), agree at maturities deliberately chosen to coincide with
neither a reset time nor a scheduled model expiry (`3.3/252`, `9.1/252`,
`17.7/252`), at every one of the four states (initial + after each reset).
Residuals `<5e-4` (finite-difference-floor limited, not a real defect) in
`test_off_grid_generator_residual_vanishes_pre_and_post_reset`. This is
the closest analogue available, for a transform-based (rather than
decoder-network-based) construction, to `cor:bates-offgrid-drift`'s
backward-generator identity — it confirms `Φ` genuinely solves its own
defining equation pointwise, not just at the handful of times the
construction was assembled from.

## 12. Chain transition composition

**VERIFIED.** `verification.chain_composition_residual` checks
`P(τ₁)P(τ₂)=P(τ₁+τ₂)` (the semigroup/Chapman–Kolmogorov property every
genuine continuous-time Markov transition family must satisfy) for the
X-chain's `expm`-based transition matrix, at two independent `(τ₁,τ₂)`
pairs; residual `<1e-8` in both cases
(`test_chain_transition_matrix_satisfies_chapman_kolmogorov`).

## Summary table

| Item | Status |
|---|---|
| `D(-i,τ)=A(-i,τ)=0` for all τ | PROVEN + VERIFIED |
| `j_p(0)=j_p(-i)=0` | PROVEN + VERIFIED (incl. independent quadrature) |
| Full transform `Φ(0)=Φ(-i)=1`, pre/post-reset, all states/maturities | VERIFIED (<1e-8) |
| Manuscript's exact synthetic reset sequence reproduced | VERIFIED (<1e-12, exact table match) |
| Envelope dominates every box member | VERIFIED (grid + closed-form bound) |
| Branching lemma `D(u,r)` positivity | PROVEN analytically (Sato) — NOT by frequency sweep |
| Reserve-term admissibility (`κb_used[k]≥0`) | PROVEN (DFS 2003 criterion) + enforced as a code invariant |
| `p_n` cancels out of priced dynamics | VERIFIED, non-obvious, documented |
| Independent MC pricing check | VERIFIED, pre-first-reset window only |
| X/Y independence assumption | not verifiable from code alone (no data); algebraic consequences (tower/forward identity) VERIFIED exactly |
| Protected-option (`T<t_n`) exact reset-invariance | VERIFIED (<1e-8, 48 combinations, all 3 resets) |
| Unprotected (`T>t_n`) prices move under recalibration, as intended | VERIFIED (nonzero, ~1e-7 to 9e-6, expected) |
| Negative control (uncompensated transform) violates protected invariance | VERIFIED (~8e-5, confirms test power) |
| Off-grid generator residual | VERIFIED (<5e-4, finite-difference floor) |
| Chain Chapman-Kolmogorov composition | VERIFIED (<1e-8) |
| Moment-strip check at the exact `(α,τ)` used by the Fourier pricer | not yet cross-referenced — follow-up |
