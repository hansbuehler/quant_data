# Deep hedging with GROMM v1

This is a new GROMM v1 experiment, separate from the earlier GROMM2 deep-hedging
experiment. The initial target is the 29 August 2025 `v1-(4,3)` market fit,
with the eight-factor diffusion fitted to the preceding 252 observed dates.

The cash payoff at business day 20 is `abs(S_20 / S_10 - 1)`, in initial-spot
units. Simulate two independently seeded samples of 100,000 paths under the
original Q law, each with 50,000 complete antithetic pairs. The horizon is 20
business days with daily trading and **one simulation step per day**. One
sample trains the hedge and the other is kept independent for evaluation.

The current experiment uses 21 daily return controls: spot, a day-20 expiry
call whose ATM strike is reset at each trading date, and a floating menu of 15
ITM calls arranged as five standardized log-moneyness levels (-2, -1.5, -1, -0.5, -0.25)
using a 20% annual volatility scale, by 1-, 5-, and 10-business-day expiries.
Four co-terminal zero-strike calls supply terminal-day-20, 1-day, 5-day and
longest-tenor forward trades. These pay spot at expiry, or are marked at day 20
if they mature later. Their opening and day-20 marks use the density tower's
spot means. Spot remains separately available even though some zero-strike
claims have the same economics at zero rates.
The last expiry is shortened to `min(10, 25 - t)` near the end so it stays
inside the saved day-25 model. Each daily action opens its own trade cohort,
held until the selected expiry or day 20, whichever comes first. Calls expiring
after day 20 are liquidated at their day-20 model marks; calls expiring earlier
settle intrinsically at their expiry. The separate terminal ATM calls and spot
trades are valued at day 20. Cash has zero interest in these normalized units.
Thus a new cash-notional trade `a[t,i]` earns
`a[t,i] * (closing_value[t,i]/opening_value[t,i] - 1)` once. The closing value
belongs to that trade's actual strike and expiry. A future realized return is
used only to compute terminal P&L, never as a network input.
After the direct deep hedge, compare a Greek-guided selection using the same
paths. Earlier one-day-return scenarios are incompatible with this objective
and are rejected by the training entry point.

No claim of complete markets follows from the number of controls alone; in
particular the independent Black background and discrete daily trading must
be considered. Record instrument normalization because the numerical action
penalties depend on their units.

The current feedback policy (`feedback.py`) gives both the LSTM and action
network current time, all 21 current prices, the full causal GROMM simulator
state, the previous executed trade, and the live contract-unit inventory.
Inventory is a 20-by-21 cohort book: floating replacement contracts are never
mistaken for contracts bought earlier, and settled contracts leave the book
before the next decision. The action also reads the updated LSTM memory.
Its head is an affine map plus a two-hidden-layer
Tanh network. Both heads have ordinary random initial weights, and subtract
frozen copies of their entire initial maps. The initial action is exactly
zero while trainable deep layers have nonzero initial gradients. Frozen
reference parameters never update. Derivatives through the shared action
state are retained; the LSTM's initial gradient consequently cancels until
the heads diverge. The day-10 spot fixing must not be exposed before it
occurs; the recurrent state must be able to retain it after it occurs. Fit
separate policies for empty, long and short claim positions. There are no
economic transaction costs. The training regularizer is `1e-4 * sum(abs(a))`
plus `1e-8 * dot(a, a)` on the daily new-trade cash-notional vector, aggregated consistently
across paths and time. Report its aggregation and units and report both the
penalized training objective and the unpenalized economic OCE. The L1 term
encourages smaller/sparser trades; the small quadratic term provides some
coercivity, but neither represents a bid-ask spread or turnover cost. Their
numerical size is meaningful only relative to instrument and basket scaling.

Use the power-exponential utility with risk aversion 100 and a learned OCE
translation, via the shared `quantdata.torch.utility.torch_utility("powexp")`.
This preserves the correct derivative `u'(0)=1`; the earlier custom positive
clamp had zero derivative at that boundary in this Torch environment.
If `U_0`, `U_long` and `U_short` are the optimized economic OCEs
of trading gains with respectively no payoff, `+H`, and `-H`, report
`bid = U_long - U_0` and `ask = U_0 - U_short`, the Q expectation, and
pair-based Monte Carlo uncertainty. Compare the empty policy with the exact
zero-trading baseline. An approximate numerical Q model, finite Monte Carlo
samples, and incomplete optimization can produce nonzero empty utility or
quotes that fail to bracket the sample expectation; do not silently enforce
the desired answer or describe it as a guarantee.

Every report also checks the day-10 synthetic-straddle benchmark using two
terminal ATM calls and a short co-terminal zero-strike call. Benchmark cash
translations are fitted only on training paths. A neural policy below that
penalized training objective is explicitly flagged as a failed optimization.

The original completed ITM/forward run used eight simulation substeps per day
but only daily actions; it is not an intraday-hedging experiment. Its interrupted
500-epoch extension is stopped. The requested replacement uses **one simulation
step per day**, still over all 20 days, not a single transition in total.
The matching retained 200,000-path Q audit has unconditional maximum absolute
Z 4.84890 (59/570 coordinates above 2.576) and conditional out-of-fold Z
-0.36545 (squared) and -0.36848 (entropy). The user accepted proceeding with
that diagnostic limitation; do not label this an unconditional no-arbitrage pass.

For a new one-step dataset, use `run --substeps 1 --epochs 500 --patience 0
--policies long short --warm-start-from <original-training-directory>
--warm-start-long <latest-long-checkpoint> --audit-report <reviewed-q-audit.json>`
along with the required calibration, output and device options. Warm starts
restore the selected best policy, its normalization, cash translation and
frozen reference heads. Adam and epoch numbering restart because the
simulation data changed. The old best loss is not reused: the inherited
policy is re-evaluated on the new training sample as epoch zero. Empty remains
frozen, and loss plots identify this warm-start origin separately from no trading.

Execution now uses a native fused `torch.nn.LSTM` sequence call, with both
action heads evaluated as batches across all dates. The old Python cell loop
is retained only as `forward_eager` for value/gradient regression checks.
An execution-only LSTM view aliases the existing Cell parameters, so canonical
checkpoint names, frozen reference maps, optimizer identities and RNG state
are preserved; no weights are initialized anew. cuDNN reduced-precision TF32
is disabled for both the native recurrence's forward and backward operations.
The full training sample is resident on the selected device (about 329 MiB);
only one small shuffled pair-index vector is uploaded per epoch. Loss and
gradient-finiteness summaries are accumulated on device and read at epoch
boundaries, never once per minibatch. Nonfinite epochs cannot replace the
previous atomic checkpoint. An optional `TrainConfig.compile_model` allows
additional compilation, while the default recurrence is already native,
without a Python loop over time.

An interrupted matching run can continue without discarding completed work:
`continue_training --resume-run <run> --output <new-external-run>
--resume-incomplete --target-epochs 500 --device cuda` (as a module, with the
project environment and `gpuctl`). Each side stops at 500 total epochs; a side
already there is loaded, not retrained. No simulation is repeated.

Status: expiry settlement, day-20 marking, causal return accounting and frozen
reference gradients have focused tests. Every run uses a unique external run
directory, with code and scenario identities recorded in its reports.

Continue only the bid/ask fits on their existing paths with
`python -m quantdata.sdlv.dhgromm1.continue_training --resume-run <completed-run>
--output <new-external-run> --additional-epochs 500 --device cuda`.
Launch through `gpuctl` in the explicit project environment and send stdout
and stderr to `<new-external-run>/job.log`. This resumes the last epoch's
weights with their matching Adam and random-generator states, while retaining
the previous best checkpoint as a selection candidate. The frozen initial
action maps are restored, not regenerated. Older checkpoints' missing shuffled
pair order is reconstructed and checked against their saved generator state.
The empty policy is loaded without training; early stopping is disabled for
both continued fits. From a 100-epoch source this runs epochs 101--600 for
each fit, with no new simulation. Independent-sample evaluation and joined
loss-history plots run automatically afterwards. Selection still uses only
the training sample.

## Full-state feedback rerun

The previous completed native-LSTM policies above are retained as warm-start
sources and comparison results. They did **not** observe simulator latent state
or inventory; LSTM memory alone was not a substitute for those requested inputs.
The new, separately versioned feedback architecture explicitly migrates their
selected weights, checks compatibility, and initializes the new input columns
to zero. This preserves the learned initial policy but immediately permits
nonzero learning gradients on the added state/feedback connections. Adam restarts
for the changed architecture; matching new checkpoints support exact continuation.

The GROMM input consists of all 202 front-density coordinates, every still-live
future squared-DLV coordinate, and the background level. DLV uses invertible
`log1p` coordinates (including exact simulated zeros); background uses its log.
Expired fixed-expiry rows are masked out, not
replaced by forecasts. The padded dimension is 5,003; ragged packing stores
62,060 coordinates across the 20 opening states per path, with no PCA projection.
Training-only population normalization has a 0.001 scale floor; this rescales
features, not the simulated dynamics. Independent evaluation never fits scalers.

A small 1-input, 8-hidden-unit time gate multiplies the non-time action features
by `1 + tanh(G(t) - G_initial(t))`. It is exactly the identity on migration;
raw time still enters directly. It can learn date-dependent feature relevance
without imposing any cap on the affine-plus-deep trade output. Both gate and
action references are frozen parameter copies, with input derivatives retained.

Because actions affect the subsequent inventory, a plain precomputed LSTM
sequence is no longer valid. Production execution uses a differentiable
`torch` higher-order `scan`, compiled as a full graph, jointly updating LSTM,
actions and inventory. The Python loop exists only for output/gradient tests.
The packed training states and existing path data are uploaded once and then
gathered on GPU; there is no path-data transfer or scalar synchronization per
minibatch. Deterministic replay reconstructs missing opening states in RAM and
checks all prices, returns and payoffs against the original saved scenarios.
Its separate CPU replay process exits before Torch training. GPU replay is no
longer used: its round-off differences exceeded the consistency gate in a tail
state. CPU replay retains the same tolerance; no check is weakened.
No additional raw state dataset is written to disk.

Run `python -m quantdata.sdlv.dhgromm1.feedback_run --scenarios <one-step-scenarios>
--warm-start <completed-one-step-training> --output <unique-external-run>
--epochs 500 --batch-size 256 --learning-rate 1e-5 --claim-gpu-after-replay`
in the explicit project environment. The trainer asks `gpuctl` for a GPU only
after CPU preparation, binds itself to the manager-assigned UUID, and records
the allocation and actual owning PID. Launch with a systemd `ExecStopPost` hook
running `python -m quantdata.sdlv.dhgromm1.allocation --release-after-exit <run>`;
the hook releases that allocation only after the trainer has exited. First
run the same command with `--verify-only`
to check compiled GPU actions and gradients against the reference calculation.
The empty portfolio remains the frozen zero-trading baseline. Train **500 joint
epochs**, each visiting the full sample for both networks. The minimized loss
is the **sum** of long-claim and short-claim penalized negative OCE losses,
not their mean and not a sum of bid/ask prices. Each minibatch uses one summed
backward pass and one Adam step with separate parameter groups. Network weights,
LSTM/inventory states, time gates and OCE cash translations are independent.
Gradient clipping is applied separately per network, so one gradient norm
cannot rescale the other. Checkpoint selection minimizes the training sum and
selects a common epoch; independent evaluation is never used for selection.
Each recurrent policy is compiled separately; assembling the two losses still
uses device tensors and one summed backward pass. A monolithic compiled graph
containing both scans failed the GPU gradient regression and is not used.
The smaller batch accommodates both networks' simultaneous gradient graphs.
`joint-losses.csv` reports long, short and sum loss at every epoch, starting
with the migrated warm start at epoch zero. `joint.checkpoint.pt` includes both
networks, Adam and RNG state for continuation. The per-network `.selected.pt`
exports are valuation-only snapshots. `joint.load_joint_model` loads the
selected pair without starting training.

The failed GPU-replay attempt on 4 November 2026 stopped after 68,000 verified
paths, before any new training epochs. CPU/GPU replay discrepancies exceeded
the stated tolerance; the GPU was released. CPU replay of the failing
4,000-path chunk passed the unchanged tolerance (maximum return error 7.28e-9).
The failed run directory and its obsolete preflight outputs have been deleted;
partial reconstructed states existed only in RAM, not as a disk dataset.
The warm-start policies are the selected one-step models (long epoch 489,
short epoch 466) in `dhgromm1-1step-fused-20261103-OZWNEm/training`.
Original one-step scenarios remain in
`dhgromm1-1step-warm500-20261103-vi9yMf/scenarios`.
CPU reconstruction is checked against the original CPU scenarios at explicit
FP32-learning tolerances (`atol=5e-8`, `rtol=2e-7`); original saved prices,
payoffs and returns remain the actual training/evaluation inputs.

Cleanup removed superseded eight-substep raw scenarios, disposable smoke
outputs, the superseded 1,000-epoch CPU-preparation run (stopped at 40,000 paths,
zero training epochs), and failed/duplicate preflight folders. Calibration,
scenario identities, completed results and active scratch files are retained.
The unused one-day-return helper has been removed: the current simulator and
expiry-return/cohort-accounting tests establish the actual contract cashflows.
The old stand-alone fused-LSTM benchmark has been superseded by the current
compiled feedback-scan output/gradient verification. Native-LSTM warm-start
code, Python gradient references and the planned option-Greek helper remain in
use or have a defined next-stage purpose; they are not discarded experiments.
The retained production-batch verification is in the active run's
`verified/gpu-verification.json`.

Cleanup manifest and concise failure diagnosis:
`/home/hans/quantdata_data/experiments/quant_data/sdlv/cleanup-dhgromm1-20261104-bhOZkz/manifest.json`.
The earlier raw-data cleanup record was moved into that same directory as
`initial-cleanup.json`, preserving the regeneration/ownership history.

Current joint-500 run:
`/home/hans/quantdata_data/experiments/quant_data/sdlv/dhgromm1-joint500-20261104-Ll97mr`.
CPU preparation must complete before joint GPU training begins; the GPU remains
available to others until then. The prior best one-step networks initialize both
sides, including their independent learned translations and frozen references.

## Exact-date action heads

`events.py` adds separate day-0 and day-10 affine-plus-deep networks to each
long/short policy. The executed action is `shared(state) + 1[t=0]*head0(state)
+ 1[t=10]*head10(state)`, without an action cap. The exact date indicators are
fixed, not learned through the smooth time gate. All heads read the same
causal current/LSTM/inventory action state. There are four independent new
heads in total; neither dates nor long/short sides share their parameters.

The background/all-time strategy **continues to learn**: shared LSTM, action
heads, time gate and OCE translations remain trainable. Only initial reference
copies are frozen. Each new head subtracts its frozen random initial map, so
it initially adds zero without zero-initializing its trainable deep layers.
The combined executed trade enters inventory, P&L and penalties once, with
gradients through its effect on later decisions preserved inside the scan.

This is a new, versioned architecture, explicitly initialized from the base
joint training checkpoint's selected epoch (386), not its last epoch (500).
Matching weights, translations, price scales and training-only latent scalers
are preserved; Adam restarts for the architecture change. Old base checkpoint
loading is unchanged; new event checkpoints have their own version identity
and serialize both networks, optimizer, RNG and selection history.

Train with `python -m quantdata.sdlv.dhgromm1.event_run --warm-start
<base-run>/joint.checkpoint.pt --output <unique-external-run> --epochs 500
--claim-gpu-after-replay`. Use the explicit project environment and the same
allocation release hook described above. `--verify-only` checks GPU forward
values and gradients, including nonzero dated actions. CPU reconstruction
uses the original 100,000 training paths only; the GPU is claimed afterwards.
No evaluation sample is loaded, reconstructed or used to select checkpoints.
The report and loss CSV are training statistics, not validated bid/ask quotes.

Compact completed-run reports, training losses, training-path attribution and
the event-head GPU gradient check are deliberately retained in `results/`;
see `results/README.md` for their limitations and provenance. Raw datasets,
model weights, logs and active compiler caches remain outside Git.
