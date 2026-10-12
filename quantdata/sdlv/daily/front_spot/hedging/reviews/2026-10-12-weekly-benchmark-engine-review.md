# Weekly benchmark engine review

## Verdict

**PASS for the Phase-8 timing, expiry, frozen-universe, and accounting gate.**
The latest implementation correctly constructs five retained-session episodes,
freezes inception contracts, rolls information sets without look-ahead, settles
and liquidates positions in origin-PV units, and enumerates overlapping episode
origins. No implementation blocker remains in the reviewed engine.

This is not yet a complete production weekly benchmark. The target-basis
convention and common prices for path-dependent claims must be recorded and
completed as described under remaining requirements.

## Findings

### Timing and information sets

- `build_weekly_episode_path` follows five successive one-day transitions, so
  weekends and holidays do not shorten or lengthen the episode
  (`benchmark/weekly.py:382`). Decision dates are the five transition origins;
  terminal settlement is the fifth endpoint.
- Frozen weekly exposes only the inception refit. Restarted weekly exposes all
  five decision dates through `EpisodeSpec.refit_dates`; both retain the same
  fixed contract book.
- `rolling_training_dates` obtains the trailing origins strictly before each
  refit date (`benchmark/weekly.py:580`). On the public 2023-01-17 episode, each
  252-origin window's final transition ended exactly on its decision date. The
  five restarted refits therefore used information through, but never beyond,
  2023-01-17, 18, 19, 20, and 23 respectively.
- `overlapping_weekly_origins` now enumerates every retained eligible start
  rather than stepping weekly (`benchmark/weekly.py:595`), and the committed
  test verifies two adjacent origins share four decision dates
  (`test_weekly_benchmark.py:379`).

### Contracts, marks, and normalization

- Eligibility is determined once at inception. Contract identity contains the
  fixed cash strike and expiry date; later marks reprice only those contracts
  (`benchmark/weekly.py:421`, `benchmark/weekly.py:470`). No newly listed option
  can enter an episode.
- Expiries through the actual fifth retained session are included. If that
  terminal expiry is unavailable, only the first later eligible expiry is
  added and it is liquidated at the fifth close (`benchmark/weekly.py:540`).
- Later `K/F` is recovered from each fixed cash strike, the realized
  spot-over-episode-origin path, and the current forward curve inside
  `PublicMarketDay.reprice_contracts`. The public replay reproduced this
  identity to `5.0e-11` maximum absolute error.
- Spot, option marks, option half-spreads, settlements, and later liquidation
  costs share the episode-origin spot unit. Later option values and costs are
  additionally multiplied by the cumulative discount-to-origin path
  (`benchmark/weekly.py:442`, `benchmark/weekly.py:490`).

### Rebalancing, settlement, and costs

- Five predictable position rows are applied to the following five return
  intervals. Costs use absolute position changes, so unchanged positions do not
  pay repeated spread (`benchmark/weekly.py:625`).
- Spot costs are one basis point of traded discounted spot notional at each
  rebalance and at terminal liquidation (`benchmark/weekly.py:666`,
  `benchmark/weekly.py:695`).
- Options settling at the next close earn their cash payoff less the prior
  origin-PV mark, are reset to zero, and incur no closing spread
  (`benchmark/weekly.py:503`, `benchmark/weekly.py:691`). An expired contract
  cannot be reopened.
- Options surviving the fifth close receive the last mark-to-market gain and
  then pay the terminal half-spread on liquidation (`benchmark/weekly.py:702`).
- Hedge gains, transaction-cost components, and the pricing-versus-pure-hedge
  decomposition reconcile exactly in both the unit test and a nontrivial public
  replay.

### Targets and schedule semantics

- The target catalogue contains the ten agreed five-day digitals, realized
  variance, 95% down-and-out ATM put, and 105% up-and-out ATM call.
- Following the latest remediation, every target settles on the same compounded
  forward-adjusted path. ATM digital strike is exactly one; non-ATM digital
  strikes are the inception NMS-derived `K/F`; barriers are observed only at
  the five future closes (`benchmark/weekly.py:310`). The actual spot path is
  retained separately for fixed-cash-strike hedge replay and transaction costs.
- Frozen versus restarted semantics are correctly represented by refit dates
  and rolling windows. Enforcement that a frozen model's parameters and policy
  weights are reused belongs in each model adapter; the accounting engine
  intentionally accepts already-computed position arrays.

## Public replay evidence

The 2023-01-17 restarted episode produced decision dates 2023-01-17, 18, 19,
20, and 23, with terminal date 2023-01-24. It froze 207 options across those
five expiries. Alive counts were `[207, 178, 138, 96, 45, 0]` and settlement
counts were `[29, 40, 42, 51, 45]`. All inception options were OTM-side books,
strictly inside the requested NMS screen, non-AM, and non-exterior. Both ATM
digitals had strike exactly `1.0`, and all 13 claims used the forward-adjusted
path.

## Remaining Phase-8 and production requirements

1. **Document the path convention in `PLAN.md`.** The code now makes a clear
   choice, but the plan's symbol `S[n]` does not yet state that every weekly
   target uses the compounded rolling one-day forward-adjusted path while hedge
   contracts use actual spot-over-episode-origin. This should be explicit to
   prevent later model adapters from using the wrong path.
2. **Resolve digital price/basis consistency.** The common digital mark is read
   from the inception SANOS terminal distribution, while realized settlement is
   based on the product of rolling one-day forward-adjusted returns. These
   coincide when the sequence of one-day carry factors compounds to the
   inception five-day forward. Measure and document the residual on the public
   panel, or define the target normalization so the equality is exact.
3. **Supply common prices for path-dependent targets.** Realized variance and
   barrier definitions deliberately return `market_price=None`. Before weekly
   economic P&L can be compared, the common SANOS-LV reference must price these
   targets, and callers should not be able to silently substitute inconsistent
   prices.
4. **Model-adapter gates remain.** Frozen adapters must demonstrate unchanged
   fit/policy identity across all five decisions; restarted adapters must show
   five dated refits with no future data. The engine exposes the correct dates
   but cannot infer how an externally supplied action matrix was produced.
5. **Statistical overlap handling remains.** Episode enumeration is correct;
   HAC or block-bootstrap inference for overlapping outcomes is still a later
   results-layer requirement.
6. `overlapping_weekly_origins` checks history and five transition endpoints,
   but not whether a non-empty eligible option book can actually be built. A
   production runner should either filter by successful episode construction or
   strengthen this helper's definition of “eligible.”

## Verification

- Focused CPU suite: `8 passed` in `3.79s` after the target-basis remediation.
- Public-candidate replay: frozen and restarted 2023-01-17 episodes passed all
  inspected timing, contract, fixed-strike, settlement, and PV invariants.
- Public overlap replay returned consecutive origins beginning 2023-01-17,
  2023-01-18, and 2023-01-19.
- `gpuctl resources --json` could not reach its daemon and used the local
  fallback; all RAM and disk checks reported `OK`. No GPU was used.
- No implementation file was modified by this review.
