# VolGAN and diffusion daily-baseline integration review

## Verdicts

- **VolGAN: FAIL.** Training/live timing, generated vanilla repricing, common
  result semantics and the committed CPU smoke operate as designed, but two
  shared financial-accounting blockers remain.
- **Diffusion: FAIL.** The same two shared blockers apply. Its live timing,
  conditional sampling interface, common result and committed CPU smoke also
  operate as designed.

## Blockers affecting both models

### 1. Remaining-maturity digital values are not smile consistent

For a three-day digital after one simulated day,
`build_generated_daily_scenarios` obtains the generated point implied
volatility and uses the constant-volatility Black digital `N(d2)`
(`models/generated_scenarios.py:195`, `models/generated_scenarios.py:203`,
`models/generated_scenarios.py:373`). A generated implied-volatility *surface*
instead defines calls

`C(k)=BlackCall(k, sigma(k))`,

so its internally consistent cash-digital probability is `-dC/dk`. The point
Black digital omits the smile-slope term and is not the derivative of the
generated call surface used to value the vanilla hedge instruments.

Independent real-date replay demonstrates that this is material. Feeding the
2023-01-18 reconstructed SANOS grid and the realized normalized spot into the
2023-01-17 three-day ATM call-digital transition gave:

- generated point-IV digital origin PV: `0.1172230093`;
- realized SANOS digital origin PV: `0.1049093770`; and
- absolute difference: `0.0123136323` (11.7% of the realized mark).

A finite-difference derivative of the generated grid's own call interpolant was
`0.1223983545`, confirming that the current point-IV value is not even the
digital implied by its own scenario call geometry. The remaining discrepancy
to SANOS reflects the coarse 9-by-11 representation and should be measured and
disclosed separately.

Derive the digital from the same interpolated call surface, or retain a
generated density/digital representation, and add a three-day smile regression.

### 2. Float32-rounded one-day options are charged a closing spread

The cost-lasso proxy distinguishes cash-settled one-day options with the exact
comparison `quote.business_dte == 1.0`
(`models/generated_scenarios.py:233`). The released geometry represents the
real 2023-01-17 one-day DTE as `1.0000000577419996`. Consequently all 29
one-day options are treated as longer-dated and charged two half-spreads even
though expiry settlement has zero closing cost. For one unit in every such
contract, the excess aggregate penalty is `0.0004760743` origin-spot units.

Use `_integral_business_dte(quote.business_dte) == 1` or the fixed expiry date,
and add a regression using the observed float32 value. The actual realized
accounting already charges no closing cost for settled options
(`benchmark/digital_daily.py:218`, `benchmark/digital_daily.py:288`).

## Shared checks that pass

- **Fixed contracts and carry.** Generated spot is
  `Z=S(t+1)/F_t(t+1)`. One-day vanilla settlement correctly uses
  `DF_t(t+1) F_t(t+1) (Z-k)^+`; longer vanillas correctly map the fixed strike
  to `k/Z` and use `Z DF_t(T) F_t(T)` times the next-state normalized Black
  price (`models/generated_scenarios.py:142`). In the real replay, every
  one-day generated vanilla return matched the realized fixed-contract return
  within `6.3e-12`.
- **Discounting.** One-day target settlement is discounted to origin, while a
  remaining digital is valued in origin-PV cash units. The generated baselines
  consistently freeze deterministic carry/discount curves at inception; this
  convention should be stated explicitly in the READMEs.
- **Longer vanilla repricing.** Log-strike interpolation and total-variance
  maturity interpolation are internally consistent. Replaying the actual
  endpoint grid produced mean absolute vanilla-price deviations around
  `1.2e-4` for two- to four-day options, primarily a 9-by-11 grid approximation
  rather than a carry error.
- **Training/live timing.** Both fits use histories ending at the origin date;
  live conditions end exactly at that date, require a gap-free chain and target
  the next retained session (`models/volgan/data.py:85`,
  `models/diffusion/data.py:84`). Both adapters assert that forecast and hedge
  endpoints agree (`models/volgan/adapter.py:117`,
  `models/diffusion/adapter.py:121`).
- **Common result semantics.** Both models are correctly labelled
  `historical-P`. Because neither supplies a risk-neutral price, `model_price`
  is the common released SANOS market mark and pricing P&L is zero; generated
  scenarios determine only the hedge. Realized payoff, fixed-contract gains,
  opening/closing execution costs and hedge P&L are then computed through the
  common benchmark accounting (`models/volgan/adapter.py:144`,
  `models/diffusion/adapter.py:148`).

## Model-specific observations

### VolGAN

- The target correctly contains the annualized normalized forward log return
  and the one-day log-IV increment; sampling divides the return by `sqrt(252)`
  and adds the generated surface increment to the live surface
  (`models/volgan/data.py:171`, `models/volgan/hedger.py:68`).
- The public 2023-01-17 smoke passed its committed fixture: 146 examples, 40
  epochs, 128 generated paths, 163 instruments and validation loss
  `1.1131898007`.

### Diffusion

- The live 21-day context and generated day-22 output are dated correctly. The
  sampler extracts the broadcast return channel by cross-sectional averaging
  and the log-IV channel directly (`models/diffusion/hedger.py:141`).
- The public 2023-01-17 smoke passed its committed fixture: 146 sequences,
  three epochs, 64 generated paths, 163 instruments and validation loss
  `1.1463043690`.

## Non-blocking findings

1. Both reduced smokes generate zero positive ATM one-day digital outcomes,
   hence zero actions, zero costs and the unhedged P&L. Their fixtures therefore
   prove deterministic wiring but do not exercise option actions, cost
   accounting or remaining-maturity repricing. Add a small non-degenerate
   financial fixture, preferably a three-day target, after fixing the blockers.
2. The READMEs call the hedge “full-cost”, while training uses predictable
   proxies: two spot basis points, one or two inception option half-spreads,
   rather than pathwise endpoint costs. Realized scoring does use actual
   endpoint spread and terminal spot costs. Describe the training penalty as a
   predictable round-trip proxy, or incorporate expected generated closing
   costs.
3. Diffusion averages a generated return channel that is only softly encouraged
   to be constant over the grid. Report its within-path cross-sectional
   dispersion so averaging cannot conceal a failed broadcast constraint.

## Verification

- Focused CPU tests: `22 passed` in `13.74s`.
- Fresh VolGAN CPU smoke with committed fixture: passed.
- Fresh diffusion CPU smoke and committed-fixture just-load replay: passed.
- Real fixed-contract, carry, cost and three-day-digital replay performed on
  the public 2023-01-17 to 2023-01-18 transition.
- `gpuctl resources --json` could not reach the daemon and used its local
  fallback; all reported RAM/disk checks were `OK`. No GPU was used.
- No implementation file was modified by this review.
