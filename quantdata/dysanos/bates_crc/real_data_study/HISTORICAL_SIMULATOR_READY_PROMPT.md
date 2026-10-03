# Ready-to-resume prompt: "Historical Simulator"

This file exists so the deferred "Historical Simulator" task can be
resumed by pasting the prompt below verbatim, with no re-derivation of
scope needed. Full background, rationale, and what has already been
checked is in project memory (`project_historical_simulator_task.md`) and
in `REAL_DATA_FINDINGS.md` / `report/sections/sec_crc_real_sequence.tex`.

## Before pasting the prompt below: confirm the blocker is actually resolved

This task was deferred because the only real option-quote data in this
environment was 6 non-adjacent WIDB fixture dates (2005-11-21, 2010-07-07,
2011-07-07, 2024-03-05, 2024-12-26, 2025-01-03) — enough for a
cross-sectional comparison and a mechanical CRC-routine check (both
already done, see below), but not enough to fit or validate a historical
transition law. Resume only once ONE of the following is true:

- Live WRDS access is configured: `WRDS_USERNAME` set, a working
  `~/.pgpass` entry, Option Metrics entitlement on the account (see
  `DATA_STATUS.md` for the exact setup this codebase expects).
- A denser pre-populated `QUANTDATA_ROOT` is mounted, containing a real
  SPX option-quote series with materially more than 6 dates, ideally
  weekly-or-denser and spanning at least several months (AGENTS.md hints
  at a possibly already-configured OneDrive-hosted working tree on another
  machine — check there first).

If resuming with a WRDS pull: start small (a few months weekly) and
re-run the mechanical check in `crc_real_sequence.py` first — it's cheap,
needs no fitting, and will immediately show whether the denser series
actually brings transitions inside a realistic drift region before
committing to the full historical-dynamics estimation below.

## What is already built and verified (do not rebuild)

- `real_data_study/day_loader.py`, `config.py`, `metrics.py` — shared
  data-loading and evaluation-contract infrastructure.
- `real_data_study/chain_black_calibration.py`,
  `chain_bates_calibration.py`, `extended_bates_reserve_study.py` — the
  per-day cross-sectional calibrators this task extends into a time
  series. 68/68 tests passing as of the cross-sectional task.
- `real_data_study/crc_real_sequence.py` — the mechanical (non-statistical)
  CRC-routine check: chains already-calibrated per-day backgrounds
  chronologically, checks admissibility of the realised theta/jump moves,
  and verifies `apply_reset` preserves protected-option prices exactly on
  real numbers (confirmed: 4/5 transitions exact to 0.0, 1/5 hits a
  reported — not silent — numerical limit of the Carr-Madan pricer for
  extremely wide dominating envelopes). 5/5 tests passing. Findings:
  `REAL_DATA_FINDINGS.md` and `report/sections/sec_crc_real_sequence.tex`.
  **This already tells you which failure mode denser data would fix**
  (theta-band size: likely yes; jump-parameter identifiability on modern
  dense books: likely no, without also changing the calibration approach)
  — read it before re-deriving the same conclusion.

## The prompt to paste (verbatim, as given by the user)

```
Proceed after the exact-construction audit and snapshot-capacity study.

The objective is to estimate a historically useful simulator within the
admissible DLV × extended-Bates family, and compare it fairly with dy2.

1. Preserve the distinction between evolution and recalibration

Market observations change through time.
A CRC reset preserves the model's current evolved prices.

Do not overwrite evolved prices with the next observed surface and call this a
price-preserving reset. Infer latent states and fit transition laws so that model
evolution explains observations, allowing an explicit observation-error model.

2. Estimate an admissible physical law

Specify the physical law P and common pricing law Q explicitly.

For a diffusion component, enforce the appropriate relation

bP − bQ = Σ λ,

with the stated sign convention and a valid equivalent-measure density.
Treat jumps and discrete reset choices through their corresponding likelihood
ratios and support conditions.

Both P and Q reset laws must remain inside the price-preserving admissible region.
Their probabilities may differ subject to equivalence and integrability.

Do not assume an arbitrary historical AR(1) can be converted into this Q model.
The dy2 AR model is a comparator and possible statistical initialisation.

Do not infer continuous-time Brownian rank from a weekly innovation covariance
without specifying and validating an embedding.

3. Historical fitting

Fit:
- latent initial states and coherent DLV dynamics;
- fixed background parameters;
- admissible θ-reset proposal dynamics;
- reserve budgets/schedule;
- physical risk premia and reset likelihoods;
- observation noise where needed.

Respect the information restriction used to prove independence under Q.
If a proposed joint calibration policy violates it, implement the joint
conditional-law formulation or reject that policy; do not keep the separated
pricing formula.

Use chronological training, validation and held-out testing.
Distinguish conditional scenario tests from genuine forecasting.
Apply the same distinction to dy2.

4. Compare dynamics

Evaluate:
- fixed-contract option changes;
- spot–surface dependence;
- surface principal components and their dynamics;
- volatility clustering and term-structure/skew changes;
- short-horizon and multiweek tails;
- front-expiry behaviour;
- runtime and failure rates.

For arbitrage diagnostics:
- first check structural identities and discretisation convergence under Q;
- then conditional martingale moments over the full protected book;
- then held-out OCE/hedging challenger tests.

A positive OCE under P can reflect risk premia.
A small effective sample size diagnoses reweighting difficulty, not by itself
nonexistence of a martingale measure.

5. Scope and reporting

The initial experiment may use a finite horizon with no new expiries.
If so, compare dy2 over the same horizon and do not claim an indefinite rolling
market simulator.

Report whether any underperformance comes from:
- snapshot representation;
- coherent latent dynamics;
- independence;
- restricted Bates recalibration;
- reserve cost;
- numerical error;
- or data limitations.

Deliver a PDF report with LaTeX source, actual fitted parameter sequences,
reproducible experiment configurations, machine-readable metrics and figures.

End with a supported conclusion:
Does the model combine useful empirical performance with its stated guarantees?
Which restriction is the next justified target for extension?

Do not claim universal superiority, publication priority or successful empirical
validation from synthetic recovery or in-sample fits.
```
