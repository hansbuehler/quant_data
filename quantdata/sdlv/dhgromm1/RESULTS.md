# GROMM v1 deep-hedging results

The original training table is a 5 November 2026 snapshot. The P row remains
the saved epoch-89 snapshot, not a completed P result or a live checkpoint.
Independent validation and the conditional-MC pilot were added on 6 November;
the premium-based transaction-cost experiment was stopped on 6 November.

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
Power-exponential risk aversion is 100. The historical results below have zero
transaction costs; the stopped cost experiment is described separately below.
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

## Independent validation, 6 November 2026

Frozen training-selected policies, 100,000 separate validation paths arranged
as 50,000 antithetic pairs. These are full-distribution ordinary OCEs, not
bottom-90% scores. Cash is profiled globally to evaluate each fixed hedge;
no policy, quote scaling or return/payoff centering is fitted to validation.
The historical L1/L2 row reports raw OCE, without subtracting its training
regularizer and without inserting cash costs that were not present in that run.

| Frozen policy | Long − Empty | Short − Empty | Empty |
|---|---:|---:|---:|
| Ordinary DH, Q+a10 | +0.006458 ± 0.007256 | −0.001981 ± 0.005326 | −0.009574 ± 0.008200 |
| Historical DH+L1/L2, Q+a10 | +0.019585 ± 0.021417 | −0.024007 ± 0.033670 | −0.031638 ± 0.026101 |

Errors are ±3 antithetic-pair MC standard errors, adding the independent
validation and original-training centering contributions. Fitting/selection
uncertainty is excluded. The large errors and negative Empty OCE contrast
with the in-sample results: these estimates do not establish converged prices
or an arbitrage-free model. A few absorbing-boundary paths dominate the
validation utility variance; small action regularizers did not solve it.

Full frozen-cash and profiled-cash statistics and checkpoint hashes are in
[the retained validation report](results/20261106-independent-validation/report.json).
The report also preserves the discontinued robust experiment as historical
evidence; that model is not part of the current comparison.
External source: `dhgromm1-heldout-20261106-vBdW4J`.

## Conditional absorbing-boundary MC pilot

This is a CPU evaluation of the same two frozen policies, not new training.
Before each one-day transition, both absorbing endpoint continuations are
included with weights `W*p_endpoint/M`, with M=2 background continuations.
The surviving spine is sampled conditional on remaining interior and receives
weight `W*p_interior`; its terminal weight is retained. Background Black risk
and subsequent adaptive hedge decisions remain stochastic after absorption.
Utility is averaged across branches, not evaluated at their mean wealth.

There are 1,024 outer paths (512 independent antithetic pairs), with 82,944
weighted terminal branches. Cash is one global scalar per policy, never a
separate scalar per branch. Branch weights for each outer path sum to one.

| Frozen policy | Long − Empty | Short − Empty | Empty |
|---|---:|---:|---:|
| Ordinary DH | +0.005405 ± 0.007960 | +0.001152 ± 0.003631 | −0.007659 ± 0.008957 |
| Historical DH+L1/L2 | +0.016283 ± 0.024494 | −0.007036 ± 0.028016 | −0.027271 ± 0.028623 |

Errors are ±3 MC standard errors clustered by OUTER antithetic pair, keeping
all branch covariance. Original training constants and policies are fixed;
these pilot errors exclude training-centering uncertainty, unlike the larger
validation table above. All six paired conditional-minus-plain OCE differences
at fixed cash include zero within ±3 errors. Variance reduction is promising,
not evidence that the learned tail hedge has been repaired. Reconstruction
checks matched saved prices, gains, payoff and causal policy outputs before
running this pilot.

Full fixed-cash/profiled-cash and matched plain-MC results are in
[the retained pilot report](results/20261106-conditional-boundary-pilot/report.json).
External source: `dhgromm1-conditional-boundary-20261106-aSKrb1`.

## Stopped experiment: 1 bp cash-value costs inside utility

The comparison was ordinary DH against DH with actual transaction costs.
The latter replaces the L1/L2 treatment; bottom-90% training is discontinued.
Its pathwise cost is `C=0.0001*sum(abs(a[t,i])*P[t,i]/S[t])`, where P is the
current traded cash price. Options therefore pay 1 bp of premium. Fees are
deducted before applying utility and reduce subsequent causal running P&L.
There is no L2 term, outside-utility penalty or mean subtraction from fees.

The run retains Q one-step simulation, daily hedging, a0/a10, the same original
100,000 training paths and frozen centering. It warm-starts the selected
historical L1/L2 weights with those penalties removed, then runs 500 additional
epochs with fresh Adam at 1e-5. GPU compiled/eager value and gradient checks
passed. Training stopped on request after 128 complete epochs and 69,632 paths
of epoch 129; the checkpoint was saved and the GPU released. Training-best
epochs were Empty 124, Long 128 and Short 128. The stopped checkpoint has not
been independently scored with ±3 MC errors or validated, and is not a
completed 500-epoch result. The saved per-epoch training losses are in the
external run's `triple-losses.csv`.

External run: `dhgromm1-cost1bp-20261106-hcgjex`. Source checkpoint SHA256:
`db14199246734963e788ab79541478954f20794e001040751d87c4028d48d955`.
Stopped checkpoint SHA256:
`5f35ee8ca2b6969f4ae77c9a220ab1fbf1baf912b9f10992cc2ad0a87b7ad2c3`.
See [the model description](README.md#cash-value-transaction-costs-6-november-2026)
for accounting, initialization and the reproduction command.
