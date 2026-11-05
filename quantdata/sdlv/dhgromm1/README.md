# GROMM v1 floating-contract deep hedging

Reference: 29 August 2025, v1 grid (−4,3), eight PCA factors, diffusion fitted
on the preceding 252 observed dates. Claim `H=abs(S_20/S_10-1)`, unchanged
notional. This is separate from GROMM2. Selection and reports are training-only.

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
C_s = l1*sum(abs(a_s)) + l2*sum(a_s**2).
```

Here c=-y, and u(x)=(1-exp(-100*x))/100 for x>=0, otherwise x-50*x*x.
Costs belong INSIDE utility. Current runs have l1=l2=0, Adam weight decay
zero. Gradient-norm clipping at 10 is an optimization safeguard, not a cost.
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
inventory, previous trades, current raw marked hedge P&L. The LSTM and
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

## Requested sequence, 5 November 2026

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
3. **Q eight-step without a10**, after P. Deferred: distinguish eight
   simulation substeps/daily hedging from eight actual hedge decisions/day.
   The current sequence worker does not launch Q8.

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
`triple_plot` renders losses; `retain_result` selects compact completed results.

Obsolete premium-control results/plots/diagnoses and retired training wrappers
are removed. Original one-step path observations and causal cache remain at
their existing paths. Shared geometry/checkpoint components are retained where
the current architecture depends on them, not as old experiment endorsements.
