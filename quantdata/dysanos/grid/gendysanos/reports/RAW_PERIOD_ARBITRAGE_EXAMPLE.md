# A DYSANOS Raw Period-Arbitrage Candidate

This is the sparsest of the ten strongest weekly raw-return candidates at
10 basis points in the one-million-path DYSANOS experiment. It is a numerical
arbitrage candidate, not a proof of non-negative P&L over the complete
conditional support.

The origin state was selected from the component simulated directly under the
fitted physical measure. The portfolio is entered at trading time \(t_0\),
when the normalized spot is \(S_{t_0}=0.707497\). It contains no stock position
and nine non-zero call positions out of the 140 calls available. The notionals
have \(\lVert a\rVert_1=1\).

| Expiry (business days) | Relative strike | Cash strike | Notional | Entry price |
|---:|---:|---:|---:|---:|
| 2   | 0.864071 | 0.611328 | +0.059932 | 0.098684 |
| 2   | 1.012249 | 0.716164 | -0.084826 | 0.021338 |
| 5   | 0.872593 | 0.617357 | +0.009901 | 0.098071 |
| 5   | 1.000000 | 0.707497 | -0.148248 | 0.033225 |
| 10  | 1.083465 | 0.766548 | +0.267278 | 0.013130 |
| 20  | 0.749078 | 0.529970 | +0.188546 | 0.182275 |
| 40  | 0.862162 | 0.609977 | -0.164707 | 0.128614 |
| 126 | 0.768016 | 0.543369 | -0.069545 | 0.206716 |
| 126 | 0.802555 | 0.567806 | -0.007016 | 0.187030 |

Positive notionals are long calls and negative notionals are short calls. The
total long and short notionals are 0.525657 and 0.474343. The net entry premium
paid is 0.0011543. The gross premium notional is 0.0883697, so the 10 basis
point transaction cost is 0.00008837. No long/short or option/spot costs are
netted.

Over the weekly holding interval, the 2-day and 5-day options settle at their
exact expiries. The remaining fixed-cash-strike options are repriced after five
business days. Because the return is raw, there is no delta hedge. Its P&L is

\[
 X=\sum_\ell a^\ell\bigl(V_{t_0+5}^\ell-C_{t_0}^\ell\bigr)
   -10^{-3}\sum_\ell |a^\ell|C_{t_0}^\ell,
\]

where an option expiring inside the interval uses its settlement payoff in
place of \(V_{t_0+5}^\ell\).

The local validation region contained 994 paths, including 101 paths sampled
directly under the fitted physical measure. Its minimum validation P&L was
0.0003961. The direct-physical and importance-weighted mean P&Ls were
0.0051223 and 0.0052672, respectively.

Starting one million new independent DYSANOS continuations from the exact same
state produced no negative P&L. The minimum was 0.00039997 and the mean was
0.00464983 with Monte Carlo standard error 0.00000103. With zero observed
losses, the one-sided 95% binomial upper bound on the conditional loss
probability is 0.0002996%.

The complete 20-dimensional surface state, full 141-instrument action, and
replay quantiles are stored in
`incremental-candidate-replay/dysanos-weekly-1000000-incremental-raw-1e-03-top10-n1000000.json`
as rank 1, source query 686.
