# Selection-date diagnostics that shaped `config.py`, and what they already show

Per `DATA_PROVENANCE.md` Section 3, every hyperparameter in `config.py` was
chosen using ONLY the two selection dates (2005-11-21, 2010-07-07), before
any held-out-date calibration was run. This file records that process and
its already-informative results, so the choices are auditable rather than
opaque.

## 1. The original frozen anchor half-width was wrong

The first design froze the coherent-chain's anchor grid at a single
constant geometric half-width (anchors `exp(linspace(-w,+w,n))`, `w=0.45`).
A smoke test on 2005-11-21 (48 real SPX evaluation contracts, tenors
0.068-0.56y) showed: **even at the chain's lowest permitted transition
rate, the resulting spread overshot the market's shortest-dated (T=0.068y)
at-the-money implied vol (~9.5%) by a factor of ~2.8** (26.8% vs 9.5%), while
a MUCH narrower grid (`w=0.05`) could not reach the far-OTM strikes quoted
at longer tenors (`K/F` up to 1.14). A single frozen width cannot serve
both. **Fix**: `w` (anchor half-width) is now a FITTED parameter, bounded
by `ANCHOR_HALF_WIDTH_BOUNDS=(0.03,0.60)`, chosen per day/model like any
other parameter -- see `chain_black_calibration.py` / `chain_bates_calibration.py`
docstrings. `CHAIN_KAPPA_BOUNDS` was also widened to `(0.05, 300.0)` (from
`(0.5,150.0)`) since the optimal Black-model kappa on the selection date sat
at the old lower bound.

## 2. Even after that fix, the coherent-chain x Black model fits real SPX poorly

Re-running the (now free-half-width) DLV x Black calibration on 2005-11-21:
**10.4% of contracts land inside bid-ask, median implied-vol error 187.6
bps (1.9 vol points)** -- a real, structural limitation, not a bug: a
SINGLE scalar transition rate (`kappa`) cannot simultaneously match a
short-dated tenor's tight smile and a longer-dated tenor's wider one in an
absorbing birth-death chain (short tenors want slow transitions to avoid
over-spreading; long tenors want fast ones to reach realistic wing vol; a
single global rate is a compromise that fits neither well). This is exactly
the kind of finding this task asks for: **the DLV x Black family, even
optimally calibrated, does not fit real single-day SPX books well.**

## 3. The DLV x ordinary-Bates model fits essentially exactly -- but by making the chain nearly trivial

Re-running the (also now free-half-width, 9-free-parameter) DLV x ordinary
Bates calibration on the SAME 2005-11-21 evaluation set: **100% of
contracts land inside bid-ask, median implied-vol error 14.3 bps (0.14 vol
points), least-squares cost ~4e-13 (numerically exact).** But the fitted
anchor half-width landed AT its lower bound (`0.03`, giving anchors from
`0.970` to `1.030` -- a narrow +/-3% grid) -- meaning **the calibrated chain
component X is nearly deterministic; essentially all of the fit's
explanatory power comes from the Bates factor Y, not from X's cross-
sectional mixture.** This is itself an informative, reportable structural
finding, not a success to overstate: for this book, the coherent-chain's
extra degrees of freedom over a plain (X fixed at 1) Bates model are doing
very little useful work. Whether this holds on every date, and whether
widening the reserve/CRC extension changes it, is checked directly in the
main comparison and the reserve sweep (Sections A/C of the report) --
this file records the DESIGN-TIME finding, not the final held-out numbers.

## 4. The reserve sweep's headline result: turning on a reserve is expensive, recalibration recovers most (not all) of it, and one date's recalibration genuinely fails

Run on three representative SPX dates (2005-11-21 selection, 2011-07-07 and
2024-03-05 held-out), reserve budgets `r_1 in {0.01, 0.03, 0.06}`, reset one
month out, jump box widened `1.5x` from the calibrated jump parameters
(`config.JUMP_BOX_WIDEN_FACTOR`, frozen before any held-out date was run).

- **Turning on ANY reserve budget, for the fixed calibrated background,
  causes a large price shift and near-total fit collapse**: mean relative
  price shift 195-276% across dates/budgets; bid-ask coverage drops from
  the ordinary-Bates baseline (94-100% on the two working dates) to
  **0%** immediately after turning the reserve on. This is not a small
  correction -- the dominating envelope construction (Sec.~12,
  `dominating_envelope_params`) is, by design, a materially WIDER jump
  measure than the locally-fitted one everywhere in the box, and its
  amplification factor `R_*` can itself be large. **This is exactly the
  case the task warns against conflating**: a nonzero admissible reserve
  is not "a small amount of extra scenario diversity" -- turning it on is a
  large, measured regime change, reported as such.
- **Recalibration recovers MOST, but not all, of the cost**: after
  refitting the whole admissible model (chain + Bates background) under
  the reserve regime, bid-ask coverage recovers to 46-78% (vs.\ the
  ordinary-Bates baseline's 94-100%) and mean fit error drops from ~40 to
  0.3-1.05 (vs.\ baseline ~0.0-0.01). A real, quantified, partial recovery
  -- not "back to normal."
- **On 2024-03-05 (the date where the ordinary-Bates baseline itself was
  already the weakest, 20% inside bid-ask), recalibration under the
  reserve regime FAILS TO CONVERGE at every one of the three budgets
  tested** (`ChainExtendedBatesCalibrationError`, both multi-start
  attempts unsuccessful) -- recorded as a genuine solver failure, not
  hidden or silently retried away.
- **Once an admissible reset is actually applied, its OWN incremental
  effect is comparatively small**: mean absolute price change for
  post-reset-horizon contracts under the most extreme admissible upward or
  downward theta move is order `0.0005-0.005` -- one to two orders of
  magnitude smaller than the initial cost of establishing the reserve
  regime itself (order `0.03-0.07`). Establishing a reserve regime is
  expensive; subsequent resets WITHIN an established regime are cheap by
  comparison.

## 4b. The held-out generalization gap this whole discipline exists to catch

Running the (frozen-config) DLV x ordinary-Bates calibration across all six
SPX dates (Table below) shows the design-time finding from Sec. 3 does
**not** fully generalize:

| Date | dy2 | SANOS snap. | DLV x Black | DLV x Bates |
|---|---|---|---|---|
| 2005-11-21 (sel.) | 96% | 96% | 10% | **100%** |
| 2010-07-07 (sel.) | 100% | 100% | 5% | **98%** |
| 2011-07-07 (held-out) | 98% | 100% | 12% | **94%** |
| 2024-03-05 (held-out) | 100% | 100% | 0% | **20%** |
| 2024-12-26 (held-out) | 91% | 100% | 0% | **27%** |
| 2025-01-03 (held-out) | 87% | 100% | 4% | **13%** |

(percent of common evaluation contracts inside bid-ask; dy2/SANOS stay
90-100% throughout, as expected for unconstrained arbitrage-free fits.)

**DLV x ordinary-Bates fits near-exactly on the three older/sparser dates,
then collapses to 13-27% on the three dense, modern (2024-2025) held-out
dates.** This was discovered only because the held-out dates were never
touched while choosing bounds/settings -- exactly the failure mode the
selection/held-out split exists to expose, not something visible from the
selection-date diagnostics alone. Plausible drivers (not further
investigated here, out of scope): far denser modern strike/expiry grids
leave much less room for the same fit-quality margin per contract; modern
SPX skew/term-structure shape may differ from 2005-2011 in ways a
single-Bates-factor + a nearly-degenerate chain cannot track as well.

## 4c. Node-count sensitivity: negligible, as the chain-redundancy finding predicts

Run on 2011-07-07 (chosen as a representative held-out date), N=5/7/9
anchors, DLV x ordinary-Bates, otherwise identical settings:

| N | \% inside bid-ask | median abs IV err (bps) | final cost |
|---|---|---|---|
| 5 | 92.2% | 18.8 | 1.1e-14 |
| 7 | 94.1% | 18.7 | 1.3e-14 |
| 9 | 92.2% | 19.2 | 4.8e-17 |

Statistically indistinguishable across N. This is exactly consistent with
finding 3 above (the chain component collapses to near-determinism for the
Bates variant): if the DLV mixture isn't doing meaningful work, adding more
nodes to it cannot help. This is a genuine (if modest-scope: one
representative date) empirical check, not an assumption.

## 5. Runtime implications for the production run

Full-precision (`n_quad=200,u_max=250`) Bates calibration was abandoned
after >10 minutes without finishing on this 48-contract set; a two-tier
approach (`n_quad=80,u_max=150` during optimisation, matching the pattern
already used for the ordinary-Bates calibrator's own internal default) took
~30 seconds and reproduced the numerically-exact fit above. The production
run (`run_full_study.py`) uses this reduced-precision tier for all
optimisation loops, with a final high-precision re-price for reported
numbers -- documented explicitly there, not left implicit.
