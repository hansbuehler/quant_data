# Model conventions for the DLV x extended-Bates CRC comparison

One contract specification and one set of unit conventions, reconciling
`dy2.pdf`'s statistical simulator, the `dy2.pdf`-companion "DYSANOS II"
manuscript's zero-carry theory, and `quant_data-main`'s actual code
(`quantdata.sanos.arbfree`, `quantdata.dysanos.flex`). Every convention below
is traced to a specific file/field, not assumed from the papers alone.

## 1. Contract specification

| Field | Convention used | Source |
|---|---|---|
| Cash strike $K$ | Absolute dollar strike, as quoted | `quantdata.widb.day` `strike` column |
| Calendar expiry $T$ | Absolute date; remaining maturity `tte_y` in **years**, `DAYS_PER_YEAR`-scaled | `quantdata.biz_calendar.DAYS_PER_YEAR`, `quantdata.widb.day` `tte_y` |
| Valuation timestamp $t$ | Trading date `d` (daily granularity; FLEX further buckets to a fixed 5-business-day weekly stride) | `smooth_interpolated_market(ticker, d, ...)`, `WeeklyDynamicsConfig.weekly_session_stride=5` |
| Spot/forward | Both carried: `spot_ref`/`spot_ret` (raw spot) and `fwd` (forward to $T$) are kept as **separate** fields; discounted forward `df*fwd` is the normalisation denominator | `dysanos/market.py::smooth_interpolated_market` |
| Discount factor | `df`, per (day, expiry), from the raw WIDB zero curve | `quantdata.widb.day`, `raw/widb/zero_curve` |
| Dividends/carry | Implicit in `fwd` (forward already reflects the dividend-adjusted carry to $T$); **not** separately exposed as a continuous dividend yield anywhere in the traced pipeline | `dysanos/market.py` (uses `fwd` directly, never re-derives it from a yield) |
| Settlement | European cash-settled index option (SPX-style); `flex/weekly_simulation.py`'s front-expiry payoff is the ordinary `relu(gross_return - k)` intrinsic value, **not** a grid-pinned substitute | `hjm_roll_residuals` docstring: `"For the front expiry it uses relu(R-k) - c_current[1](k)"` |
| Quote units | **"Pure" units**: `C/(DF*F)`, i.e. call price divided by discounted forward — a dimensionless, unit-forward-normalised price. Strikes are correspondingly `pure_strike = K/F`. | `dysanos/market.py`: *"the WIDB data at this point has prices and forwards divided by spot, so dividing by df*fwd is exactly the dimensionless cash-price transformation C/(DF*F)"*; `quantdata/widb/day.py` `pure_strike`/`pure_bid_price`/`pure_ask_price` fields |

## 2. Carry normalisation: how zero-carry theory meets nonzero-carry quotes

Both `dy2.pdf`'s companion theory manuscript and this repository's product
theorems (`Theorem thm:product`, `Theorem thm:dlv-bates`) are stated **at
zero interest and dividend rates**, with an explicit remark that "an
appropriate deterministic carry normalisation may first be applied." The
actual code already performs exactly this normalisation, independently and
consistently, via the "pure" unit convention above: dividing by the
discounted forward `DF*F` converts a general-carry call price into the
zero-carry price of a claim on the **normalised forward price**
$Z_T = S_T/F_t(T)$ — precisely the paper's own opening identity
(`dy2.pdf`, Section 2: *"$Z_T = S_T/F_t(T)$ ... a $\Q$-martingale with unit
mean"*).

**Practical consequence for the first experiment**: no further carry
modelling is required to align the two. The product/Bates constructions can
be implemented directly in "pure" units with deterministic carry, matching
what `quantdata.dysanos.market`/`flex` already do, **provided**:
- the discount factor and forward used for normalisation are the ones
  already stored per (day, expiry) (`df`, `fwd`) — not re-derived from a
  separately assumed constant rate, and
- settlement/payoff calculations use the *same* normalised units throughout
  (already true of `_option_values`/`hjm_roll_residuals`, confirmed in the
  audit).

If a genuinely stochastic-carry extension is wanted later (e.g. stochastic
dividend yield), that is an explicit, separate enlargement, not something
either the current code or the zero-carry theory already covers — flagged
here as future scope, not silently assumed solved.

## 3. DLV normalisation, strike support, smoothing parameter, state dimension

**(new, resolving a real unit mismatch flagged in `REPOSITORY_MAP.md`)**:

| Quantity | `quantdata.sanos.arbfree`/`SanosFitter` (production static fit) | `quantdata.dysanos.flex` (torch joint decoder) |
|---|---|---|
| DLV coordinate | `dlv`, bounded `[0.01, 50]` — **unsquared** local-volatility scale | `dlvsq`, bounded `[0.01^2, 3.0^2]` — **squared** |
| Smoothing parameter | `smoothness` (a single scalar per fit, semantics local to `MkArbFree`) | `eta`, a **multiplier on `forward_vol`** (not itself a std, not itself a variance) — kernel total std is `forward_vol*eta*sqrt(dt)`, traced directly into `_black_call`'s `sqrt_variance` argument |
| Strike support | Actual observed strikes, per day, variable count | Fixed `num_inner_strikes=200` on a **relative** (normalised-moneyness) grid, `nms_range=(-3.0, 2.0)` |
| State dimension | Not a fixed-size "state" — a per-day optimisation output | Native affine map $u_t = b + Bh_t$, $h_t\in\R^{200}$ (`dy2.pdf` Definition 3.1); PCA-reduced weekly-dynamics state is **smaller**: `STRONG_STATE_SIZES` sums to 135 across 14 expiry blocks (not the same "200" as the native map — these are two different numbers referring to two different objects, confirmed by direct inspection, not assumed identical) |
| Brownian/jump factor structure | None (static, cross-sectional only) | Weekly dynamics: one Gaussian innovation per reduced-state coordinate plus one spot innovation (jointly correlated, `weekly_joint_innovation_covariance`), **no jumps** anywhere in `flex/` (confirmed: no jump/Poisson-measure code in `weekly_dynamics.py` or `weekly_simulation.py`) |
| Transition time step | N/A (static) | Fixed `weekly_session_stride=5` business days |

**A conversion is required, not assumed, before comparing `arbfree`-fitted
and `flex`-decoded surfaces directly**: `dlv <-> dlvsq` is a square/square-root
relation on the same underlying discrete local-variance quantity (subject to
confirming both use the same butterfly-mass normalisation — not yet
independently re-derived here, flagged as a check for the next task, not
claimed equal); `smoothness` and `eta` are **not** the same parameter and
must not be substituted for one another without a derived mapping (none
exists in the code today).

## 4. Moving relative coordinates vs. replacement of cash support

Two genuinely different operations, kept distinct throughout `dy2.pdf`'s
companion theory (Section "Regridding, rolling contracts and price-preserving
changes") and confirmed to be handled correctly, not conflated, in the code:

1. **Moving relative coordinates** (a pure re-labelling): FLEX's strike grid
   is stored in **normalised moneyness** (`nms_range`), so as spot moves the
   *cash* strikes represented by a fixed relative-grid index change, but
   every already-existing cash payoff and price is unchanged pathwise (the
   theory manuscript's own identity: $(sx_i)x_j' = sx_j$ for
   $x_j'=x_j/x_i$). This is what `flex/`'s "moving relative-grid geometry"
   (per `dy2.pdf` Section 4.6) actually implements.
2. **Replacement of cash support** (a genuine model change): swapping the
   actual protected cash-strike book requires the exact feasibility
   conditions in the theory manuscript's eq. `eq:regrid-lp`
   (nonnegative weights, mass, mean, and every protected payoff preserved)
   — **no code in this repository currently implements this as a checked,
   admissible operation**; `flex/`'s weekly roll only ever settles expired
   contracts and introduces new back-expiries via the fixed relative grid,
   which is case (1), not case (2). Any future cash-support change (e.g. to
   track a real, evolving liquid strike ladder) needs new, explicitly
   feasibility-checked code — not present today, flagged as a gap for the
   next task rather than silently assumed handled by the existing regridding.

## 5. What "zero drift" is checked against, precisely

Per the audit and the prior CRC-extension task: the operative discrete-time
identity actually computed by this codebase is `hjm_roll_residuals`
(Section 9 of `REPOSITORY_MAP.md`), which is the exact finite-market analogue
of `dy2.pdf`'s companion theory manuscript's `Corollary cor:bates-offgrid-drift`
/ eq. `eq:bates-offgrid-drift` — confirmed by direct derivation in the prior
task (`flex/crc_extension/section2_pricing_identity.py`'s module docstring),
not merely asserted. Any new Bates/CRC construction should be tested against
this **same** residual object for comparability with the already-measured
baseline numbers (t-stat 16.9 for raw P, t-stat 8.9 for the existing trained
target-measure kernel), rather than a newly invented diagnostic.
