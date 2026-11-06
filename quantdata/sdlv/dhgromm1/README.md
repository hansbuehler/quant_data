# GROMM v1 floating-contract deep hedging

Reference: 29 August 2025, v1 grid (−4,3), eight PCA factors, diffusion fitted
on the preceding 252 observed dates. Claim `H=abs(S_20/S_10-1)`, unchanged
notional. This is separate from GROMM2. Checkpoint selection is training-only;
independent validation is reported separately in [RESULTS.md](RESULTS.md).

Current experiment (6 November 2026): ordinary Q one-step DH with a0/a10,
warm-started for 500 additional epochs with **1 bp of cash value traded inside
utility**. Option costs are proportional to option premium, not underlying
notional. L1/L2 regularizers are zero. The no-cost baseline remains the control;
bottom-90% training is discontinued. The earlier plans below are historical,
not additional active jobs.

## Contracts, utility and causality

An option opened at t pays `(S_T/S_t-K)^+`; price `C_t/S_t`, gain
`R=(closing_cash_mark-C_t)/S_t`, **not a premium percentage return**.
It settles intrinsically at its actual expiry, or is marked at day 20 if
expiry is later. A floating quantity a adds `a/S_t` actual cash-contract
units to the cohort book. Spot/zero-strike calls use the same opening-spot unit.

There are 21 instruments: spot, current-ATM fixed-day-20 call, five ITM calls
at standardized strikes `(-2,-1.5,-1,-.5,-.25)` for each of 1/5/min(10,25-t)
days, and zero-strike calls for day 20 and those three floating tenors.
Cash rates are zero. `contracts.py` generates v4 scenarios and converts
identified old v2/v3 observations once in memory; ambiguous units are rejected.
Retained old-schema source paths are not retained old hedging results.

For three independent policies s=0,+1,-1, minimize the SUM of

```
loss_s = -c_s - mean[u_100(G_s+s*(H-m_Q)-C_s-c_s)]
G_s = sum_t,i a_s[t,i]*(R[t,i]-mean_Q_training_R[t,i])
C_s = 0.0001*sum_t,i abs(a_s[t,i])*P[t,i]/S[t].
```

Here c=-y, and u(x)=(1-exp(-100*x))/100 for x>=0, otherwise x-50*x*x.
Here P is the contemporaneous cash price of the traded contract. Costs belong
INSIDE utility; they are not centered or subtracted after expected utility.
The no-cost control sets C=0. L1/L2 regularizers and Adam weight decay are zero
in the current comparison. Gradient-norm clipping at 10 is an optimization
safeguard, not a cost.
Q means are fit once on the complete training sample, including spot, then
frozen for minibatches, evaluation AND P. Subtracting P means would remove
the risk premia being studied. Marginal centering is not a measure change
and does not prove conditional drift zero. The accepted one-step Q warning
remains: unconditional max |Z| 4.84890 (59/570 above 2.576); conditional
squared/entropy Z −0.36545/−0.36848.

Raw quotes are `bid=m_Q+U_long-U_empty`, `ask=m_Q+U_empty-U_short`.
Positive empty utility or failure to bracket EQ must be reported, not forced
away. In-sample MC errors conditional on fitted means/policies omit selection
and optimization uncertainty and are not a no-arbitrage proof.

Features: current time/spot/floating prices, strike ratios/DTEs, executed
inventory, previous trades, current marked hedge P&L net of already paid fees. The LSTM and
uncapped affine-plus-deep decider see current inputs; the decider also sees
updated LSTM memory. No full simulator state, future payoff/return/fixing,
or explicit S10 fixing enters the shared policy. Maps start at live W minus
identical frozen W, not zero hidden-head weights. Production uses compiled
differentiable scans and GPU-resident data, with accurate compiled division
(`eager_numerics.division_rounding=True`). GPU values and every gradient are
checked against GPU/CPU eager references, plus the analytic cash derivative.

The causal market cache has 3,990 float32 values/path (1.596 GB per 100k),
not full model states. Complete caches are loaded read-only with source hashes.
New P marks are generated alongside paths, avoiding a second simulation.

## Completed corrected Q baseline

External run `sdlv/dhgromm1-floating-zero-cost-20261105-KMBK8u` below
`/home/hans/quantdata_data/experiments/quant_data/` finished the original
500-total target. It resumed corrected penalized run
`dhgromm1-floating-centered500-20261105-n0QhMI` at 138 complete epochs plus
94,720 paths, preserving current weights/Adam/RNG/minibatch. The zero-cost
phase ran 362 epochs with new-objective history and independent policy-best
selection: empty epoch 362, long 357, short 355.

Centered OCEs: empty +0.000303270, long −0.000287635, short −0.000090033.
The positive empty result is not suppressed. Small final reports, scalar
benchmarks, GPU verification, loss CSV and PNG are intentionally retained in
`results/20261105-floating-zero-cost-a10/`; raw paths, weights, logs, caches
and generated PDFs remain outside Git. `floating.py` implements generic
LSTM actions plus deterministic a0/a10 vectors. At day 10,
`call=-2*s*alpha, terminal_forward=s*alpha` contains the exact scalar hedge.

## Historical requested sequence, 5 November 2026

1. **Q one-step without a10:** external run
   `sdlv/dhgromm1-q1-noa10-20261105-w4aKQh`. Fresh initialization, no injected
   day-10 hedge, same original 100k training paths and completed cache.
   `time_decider.py` adds ten sine/cosine clock frequencies plus raw/quadratic
   time, a two-layer width-64 deep branch and two-layer width-32 time gate.
   The shared clock spans every deterministic daily action schedule. No a10
   parameter or fixing indicator exists; a0 stays explicit.
2. **P one-step with and without a10:** external sequence
   `sdlv/dhgromm1-P1-sequence-20261105-3Cq7c9`. CPU path preparation overlaps
   Q training; GPU training waits for Q completion. Both P runs use the same
   larger clock-aware architecture, differing only by a10, and share P paths,
   marks and frozen Q centering. Each runs 500 epochs, batch 512, Adam 1e-5.
3. **Q eight-step without a10**: eight simulation substeps, DAILY hedging
   (confirmed). The current sequence worker does not launch Q8. The later
   action-regularization control takes priority over this new-data experiment.

P reuses `experiments/quant_data/vix_2w/20261027-market-pq-v1/physical_fit.npz`:
249 strictly prior historical session pairs, 20 August 2024–28 August 2025,
whose indices match the frozen Q diffusion. Convention
`dW_Q=dW_P+lambda*dt`, `m_P=m_Q+xi*lambda`; the independent 5% Black factor
has zero risk premium. Q pricing and initial calibration are unchanged. This
existing constant P fit captures about 45.77% of weighted historical drift
variation; it is not an unrestricted P drift. P utility need not bracket EQ.

## Reproduction and maintenance

Interpreter `/home/hans/vqdata/bin/python`, cwd `/home/hans/python/quant_data`,
QUANTDATA_ROOT=/home/hans/quantdata_data, PYTHONPATH equal to cwd,
QUANTDATA_TEST_MODE=0. Detached jobs explicitly set PATH, scratch/cache roots,
MPLCONFIGDIR and JAX_PLATFORMS=cpu; Torch claims its GPU through gpuctl before
CUDA initialization. Logs emit progress/resource heartbeats every minute.

Use `floating_run --without-a10` or its matched `--clock-with-a10` control.
P requires `--centering-from <Q-run>/training-centering.npz`. `sequence`
prepares P on CPU, waits for Q completion, then trains both P fits sequentially.
Each allocation is released only after its owner exits. Failure stops the chain.

`load_floating_model` is read-only. Versioned checkpoints retain current and
independent-best policies, cash, Adam, RNG, pair order, offsets and losses.
Changed geometry is rejected. `--continue-from` allows only a cost-coefficient
change in a new directory, with current optimizer retained and new scoring.
`oce_history_plot --snapshot <checkpoint-MC.json> --output <new-external-directory>`
renders OCE histories, Empty-adjusted long/short curves, and last-100-epoch
objective diagnostics. Higher is better. Checkpoint markers carry paired ±3 MC
standard errors; historical epoch errors were not logged and are not invented.
Curves use same-epoch policies, while markers may use independently selected
best policies. The legacy `triple_plot` renders raw training-loss diagnostics;
`retain_result` selects compact completed results.

Obsolete premium-control results/plots/diagnoses and retired training wrappers
are removed. Original one-step path observations and causal cache remain at
their existing paths. Shared geometry/checkpoint components are retained where
the current architecture depends on them, not as old experiment endorsements.

## Historical action-regularization control (5 November 2026)

This replaces the planned unregularized additional 500 epochs. Finish the
active P-with-a10 and queued P-without-a10 baselines, then run 500 additional
epochs for each of Q-with-a10, Q-without-a10, P-with-a10, P-without-a10. All
three claim networks continue learning independently in every run. The old
baseline results are immutable; this is a new objective phase, not a claim
that an unchanged objective has converged at epoch 1,000.

The optimized objective for sign s is
`U_100(G_s + s*(H-m_Q)) - 1e-4*E[sum(abs(a_s))] - 1e-8*E[sum(a_s**2)]`.
Here a is the executed floating-contract quantity, summed over 20 daily
decisions and 21 instruments, averaged over training paths. These are action
regularizers OUTSIDE utility, not parameter weight decay, cash flows, or
transaction costs. Wealth/returns are unchanged; transaction costs remain zero.
The coefficients are deliberately not divided by the number of dates/slots.
L1 discourages gross trading; L2 discourages concentration but is weak at
these units. Neither an immediate plateau nor unchanged raw OCE is assumed.

`regularized_run --source-run <completed-baseline> --output <new-external-run>`
loads each policy's independently selected best weights and OCE cash, and
starts fresh Adam at 1e-5: selected epochs do not share a matching saved joint
optimizer. Frozen Q centering, features, paths, marks, architecture and trading
frequency are unchanged. Geometry/version/objective mismatches are rejected.
The original checkpoint remains untouched; regularized checkpoint restarts
preserve their own Adam, RNG, pair order/offset, best scores and history.

`regularized_sequence` waits without claiming a GPU until the original P
sequence completes, then runs the four controls in order. The persistent plan
and logs live under external `sdlv/dhgromm1-regularized500-20261105-HGtddI/`.
Failures stop the sequence rather than silently skipping a model.

Epoch CSVs distinguish raw OCE, the regularized objective, mean summed
absolute/squared trades, and maximum absolute trade. Selection uses the
regularized objective, never old unregularized losses. Initial/final reports
show raw Empty and Empty-adjusted long/short OCE, plus the penalized objective;
paired +/-3 MC standard errors include Q gain/payoff-centering uncertainty,
hold quote scaling/policies fixed, and exclude fitting/selection uncertainty.
Before/after changes use the same original antithetic pairs. Full-sample
evaluation can transfer inference outputs for diagnostics; production batches
remain GPU-resident and the recurrent policies use compiled differentiable scan.

`regularization_plot --sequence <regularized-sequence> --output <new-external-directory>`
compares raw OCE histories before/after the objective switch, with a separate
optimized-objective figure. Purple is the unpenalized baseline phase (negative
phase epochs); orange is the subsequent penalized phase. The switch loads
selected source weights and resets Adam, so these are not parallel controls.
Available selected-policy markers show +/-3 MC SE; epoch bands are not invented.
P penalty runs are labelled queued until their histories exist. Each render
freezes CSV rows/hashes in `snapshot.json`; use `--snapshot` to reproduce it
without rereading advancing jobs. Plotting needs no GPU and never locks training.

## Discontinued bottom-90% control (6 November 2026)

This experiment was checkpoint-stopped and is no longer an active comparison.
Its implementation and frozen diagnostics are retained for reproducibility;
the current transaction-cost run does not optimize a tail fraction or hinge.

Work is now restricted to Q, one simulation step/day, daily hedging, with a0
and a10. The old no-a10 penalty run was checkpoint-stopped at epoch 384 plus
45,056 paths of the next epoch; its GPU was released. Further P penalty runs
and Q8 work are not launched. Existing completed baselines remain available.

The new experiment has zero transaction costs, L1, L2 and weight decay. It
warm-starts the selected neural weights of the completed penalized Q+a10 run
(without retaining its penalties), then starts fresh Adam, 500 epochs at
1e-5. Paths, frozen Q controls/quote scales, causal compiled LSTM, actual
floating-contract accounting and payoff are unchanged. Reusable market marks
are loaded directly; neither simulator states nor paths are reconstructed.

For each sign s independently, write `X_s=G_s+s*(H-m_Q)`. Maximize
`c + eta - E[(eta-u_100(X_s-c))_+]/0.9` over the policy, cash c and a single
shared scalar eta. Supremizing eta equals averaging the lowest 90% of utility
outcomes over the FULL population. Thus minibatches give additive estimates
of one global objective: they do not sort, compute local quantiles or discard
their own best 10%. Each epoch's score sorts all 100,000 individual paths and
averages the worst 90,000. Antithetic pair averages are used only for MC errors,
not for tail membership. Global threshold/cash are training parameters, never
inputs to a decision or its recurrent state. A q subgradient at an exact hinge
tie is value-preserving and makes the constant zero hedge stationary.

Initial cash/cutoffs are profiled globally for the fixed source-best policies.
Subsequent full-population scoring logs exact tail utility at trained cash,
the jointly optimized variational bound, their gap, and threshold coverage.
Selection uses the exact global tail score, not an average of batch tails.
`robust_run`, `robust_training`, `robust_evaluation` and `robust` implement this
experiment. Restartable versioned checkpoints include current/best policies,
cash/thresholds, Adam, RNG, complete pair order and partial-epoch offset.

The external run is `sdlv/dhgromm1-robust90-q1-a10-20261106-tGNgnK/`.
Initial/final reports separately profile ordinary OCE and bottom-90% OCE for
each fixed learned policy, subtract Empty for long/short, and report +/-3 MC
errors from the 50,000 antithetic pairs. Winsorized tail influence includes
cutoff estimation; Q-control/payoff-centering influence is included, while
quote scaling, policy fitting and checkpoint selection are held fixed.
This training experiment is not a proof that the underlying Q model is arb-free.

## Cash-value transaction costs (6 November 2026)

The current comparison is ordinary DH versus DH with actual proportional
transaction costs. Bottom-90% training is no longer considered; the earlier
L1/L2 experiments above are retained as historical controls, not the active
objective. Conditional boundary sampling remains a separate evaluation method.

For a floating-contract trade quantity `a[t,i]`, the actual contract quantity
is `a[t,i]/S[t]`. Charge 1 bp of cash value traded:
`C = 0.0001 * sum(abs(a[t,i])*P[t,i]/S[t])`.
Thus options pay 1 bp of contemporaneous option premium, spot pays 1 bp of
underlying value, and zero-strike calls pay on their cash value. This is NOT
the old uniform L1 basis, and there is no L2 term or outside-utility penalty.
Opening trades incur fees once; natural cash settlement is free, and the
existing terminal marking convention is unchanged. Already paid fees reduce
the causal running-P&L feature from the next decision onward.

For each separate Empty/Long/Short policy, maximize ordinary power-exponential
OCE `sup_c { c + E[u_100(G_s + s*(H-m_Q) - C_s - c)] }`.
Fees enter pathwise wealth BEFORE utility. Actions, floating expiry gains,
one-step Q paths, daily trading, a0/a10 vectors, and original training-only
centering/scaling are unchanged. Small premium-based fees do not guarantee
bounded positions or eliminate the observed tail generalization problem.

`transaction_cost_run --source-run <completed-Q1-a10-run> --output <new-external-run>`
warm-starts selected learned weights with fresh Adam and runs 500 epochs by
default. Three separate policies train jointly, using GPU-resident data and
compiled differentiable scans. Checkpoints retain optimizer/RNG/best weights
and partial-epoch restart state. `load_transaction_cost_model` is a read-only
loader for the new versioned cost checkpoints. Reports distinguish net OCE,
unpenalized economic OCE, and cash fees; long/short are Empty-adjusted, with
paired +/-3 MC errors conditional on frozen centering and selected policies.

The active run is
`/home/hans/quantdata_data/experiments/quant_data/sdlv/dhgromm1-cost1bp-20261106-hcgjex`.
It reuses the original 100,000 Q training paths (50,000 antithetic pairs),
one simulation step/day and daily hedging; no paths or causal marks are rebuilt.
Initialization uses the independently selected Empty/Long/Short epochs
497/496/500 of the completed Q+a10 L1/L2 control, whose coefficients are now
removed. Adam starts fresh at 1e-5, batch 512. The immutable source checkpoint
SHA256 is `db14199246734963e788ab79541478954f20794e001040751d87c4028d48d955`.

Reproduce with the explicit project interpreter/environment described above:

```bash
/home/hans/vqdata/bin/python -m quantdata.sdlv.dhgromm1.transaction_cost_run \
  --source-run /home/hans/quantdata_data/experiments/quant_data/sdlv/dhgromm1-regularized500-20261105-HGtddI/01-dhgromm1-floating-zero-cost-20261105-KMBK8u \
  --output <unique-directory-below-QUANTDATA_ROOT> \
  --epochs 500 --batch-size 512 --learning-rate 1e-5
```

The runner claims a GPU through `gpuctl` after CPU loading. Detached services
must use the existing `allocation --release-after-exit <run>` post-exit hook
to release the allocation; do not release it while the trainer is alive.
`identity.json` records units, cost timing, source/data hashes and code versions;
`job.log`, `progress.json`, `triple-losses.csv` and the atomic checkpoint track
the active run outside Git. Its result is not yet final. The cost-model checks
include fee timing/gradients, zero-fee equivalence, paid-fee causal feedback,
read-only checkpoint/version roundtrips, and GPU compiled/eager gradients.
