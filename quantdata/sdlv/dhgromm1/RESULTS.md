# GROMM v1 deep-hedging results

Snapshot recorded on 5 November 2026. The P row is the saved epoch-89
snapshot, not a completed P result or a live reading of its advancing checkpoint.

## Reporting convention

Always report OCE, with higher being better, and uncertainty as **±3 Monte
Carlo standard errors**. For long and short, subtract the OCE of the separately
learned empty-claim strategy, not the utility of literal zero trading:

\[
\Delta U_{\mathrm{long}}=U_{\mathrm{long}}-U_0,\qquad
\Delta U_{\mathrm{short}}=U_{\mathrm{short}}-U_0.
\]

The Empty column reports unadjusted \(U_0\). The claim is
\(H=|S_{20}/S_{10}-1|\), with unchanged notional; the claim supplied to each
policy is \(s(H-m_Q)\), where \(m_Q\) is the frozen Q training mean.
Power-exponential risk aversion is 100 and both trade-cost coefficients are zero.
These runs use one simulation step per business day and daily trading for 20 days.

## Best training OCEs

Each policy is selected independently by its best training OCE. Every table
entry is an estimate ±3 MC standard errors, in the original normalized payoff units.

| Run | Long − Empty | Short − Empty | Empty |
|---|---:|---:|---:|
| Q with \(a_{10}\), completed | −0.000591 ± 0.000117 | −0.000393 ± 0.000142 | +0.000303 ± 0.000039 |
| Q without \(a_{10}\), completed | −0.000618 ± 0.000127 | −0.000425 ± 0.000166 | +0.000360 ± 0.000064 |
| P with \(a_{10}\), epoch 89 | −0.000493 ± 0.000303 | +0.000697 ± 0.000302 | +0.009446 ± 0.001128 |

The two completed Q runs reached their original 500-total-epoch targets.
P's much larger Empty OCE shows why subtracting the investment-only baseline
matters. Q's Empty is also positive in-sample, not zero; these results are not
a certificate of no arbitrage. The P result is interim and is not a converged
bid/ask quotation.

The entries above are OCE differences, not ask prices. For the uncentered claim,
the corresponding bid is \(m_Q+\Delta U_{\mathrm{long}}\) and the ask is
\(m_Q-\Delta U_{\mathrm{short}}\).

## Monte Carlo uncertainty

All calculations use the 100,000 training paths, arranged as 50,000 independent
interleaved antithetic pairs. Standard errors use the sample standard deviation
of pair averages divided by the square root of the number of pairs. Long/short
differences are paired with Empty on the same paths; their errors must not be
combined as if the two utilities were independent.

The quoted errors include first-order uncertainty from estimated Q gain/payoff
centering, holding selected policies and quote scaling fixed. P uses unchanged
Q pricing and the same frozen Q corrections; P return means are not subtracted.
The calculation retains covariance from the matched Q/P random innovations.
Network fitting and checkpoint-selection uncertainty are excluded. These are
training statistics, not independent validation results.

## Snapshot provenance

All external run paths below are relative to
`/home/hans/quantdata_data/experiments/quant_data/sdlv/`.

- Q with \(a_{10}\): `dhgromm1-floating-zero-cost-20261105-KMBK8u`.
  The zero-cost phase finishes at epoch 362 after continuation from 138 complete
  earlier epochs plus part of the next epoch. Selected zero-cost phase epochs:
  Empty 362, Long 357, Short 355. Checkpoint SHA256:
  `4b9d88e4b2359993091ecb5e84cde02baeb23c236607080485cd5cdf8576ab1e`.
  Compact completed reports are retained in
  [results/20261105-floating-zero-cost-a10](results/20261105-floating-zero-cost-a10/).
- Q without \(a_{10}\): `dhgromm1-q1-noa10-20261105-w4aKQh`.
  Completed epoch 500; selected epochs: Empty 492, Long 498, Short 498.
  Checkpoint SHA256:
  `fd8926b92ce2e3ab888d1bb2109526df134aad4f3b2ba82c0ef45156a0d5a0d7`.
- P with \(a_{10}\): `dhgromm1-P1-sequence-20261105-3Cq7c9/p-with-a10`.
  Checkpoint through epoch 89; selected epochs: Empty 89, Long 89, Short 88.
  Snapshot checkpoint SHA256:
  `2a70755a556e59cd3aefcc894f725f38f27a3c5e75a6009aff24d877ed1adc22`.
  The live checkpoint continues advancing; this table remains the epoch-89 record.
