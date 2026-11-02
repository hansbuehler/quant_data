# GROMM2 no-arbitrage audit correctness review

Date: 1 November 2026. Requested independent reviewer: Astra, high reasoning.
This is a deliberately retained project review, not a generated training dataset.
Production sources, fitted models, retained populations and paper text were not
changed during the review. No GPU workload was launched.

## Scope and retained evidence

The primary case is the repaired GROMM2 (-4,3) model, 16 joint factors, eight
substeps per day, 100,000 paths and 125 days. Its unconditional daily maximum
absolute statistic is 16.0496. Both learned diagnostics selected the explicit
zero-action baseline in both folds. That is an inconclusive conditional
diagnostic, not a successful martingale test.

Canonical files reviewed include `conditional_audit.py`,
`local_audit_training.py`, `market_training_runtime.py`, `path_audit.py`,
`saved_model_audit.py`, `simulation.py`, `dlv.py`, and their focused tests.
Line numbers below identify sources at review time.

Full independent report and executable CPU reproducers:
`/tmp/quant_data/gromm2-astra-noarb-review-20261101/`.
Retained production reports, checkpoints, histories and timing log:
`/home/hans/quantdata_data/experiments/gromm2/noarb-pca16-20261031-v1/`.

## Confirmed findings

### 1. Resume silently ignores requested training controls

At `local_audit_training.py:300`, an already stopped checkpoint is returned
before considering changed learning-rate and batch controls. If continuation
is forced, lines 321-322 restore the old optimizer and scheduler configuration
over newly requested settings. The report at line 350 can nevertheless advertise
the new settings.

The CPU reproducer requested learning rate 0.02 and scheduler patience 2; the
resumed checkpoint actually used 0.001 and 50. This must be fixed before using
resume to perform a controlled batch/learning-rate experiment. It does not
explain the original fresh-run failure.

### 2. Selection, learning-rate reduction and stopping use different baselines

Checkpoint selection starts with the zero policy's loss of zero at
`local_audit_training.py:315`. The scheduler created at line 311 instead starts
with an infinite best loss. Improvements between positive losses can therefore
delay learning-rate reduction despite never beating the zero policy. Stopping
at lines 433-435 counts lack of improvement over the selected zero baseline.

The repaired eight-step histories reproduce first learning-rate reductions at
epochs 140/160 in fold 0 and 102 in fold 1. Both full-network phases began at
the original 0.00025 learning rate, not an inherited tiny rate. At epoch 300,
rates remained 0.000015625 or 0.00003125, above the configured 0.000001 floor.
This was validation early stopping, not certified optimization convergence.

The histories show substantial overfitting. Fold-0 entropy training/validation
losses changed from -0.01276/+0.03571 at epoch 100 to -18.84/+742.90 at epoch
300; fold-1 final values were -15.17/+1267.19. These are the implemented
positively scaled losses, not unscaled economic returns. The optimizer was
not simply disconnected. Better scheduling alone is not guaranteed to solve
the generalization problem.

### 3. Zero-score/zero-error cases look like valid test statistics

`conditional_audit.py:157` maps zero mean and zero standard error to `T=0`.
`local_audit_training.py:578` propagates this to combined `T=0` and `p=1`.
For an identically zero action these are conventions, not standardized test
statistics. Machine-readable output must explicitly mark the diagnostic as
degenerate/uninformative, rather than permitting it to be read as a pass.

### 4. Drift taming changes the prescribed Q generator

`simulation.py:227` applies the nonlinear drift compression defined at line 42,
while leaving the diffusion unchanged. This is not a martingale-neutral
stability safeguard. Independent differentiation of the saved origin decoder
reproduces 7.6477% initially tamed interior nodes and an audited-call generator
perturbation as large as 0.03812594 spot units per year.

This does not isolate the cause of the full-day rejection: the initial
perturbation for the 20-day NMS 1.5 call is positive, whereas its observed
full-day mean is negative. Finite-step integration and taming contributions
must be distinguished, not inferred from taming frequency alone.

### 5. The named Q-cancellation regression does not test cancellation

`tests/quantdata/gromm2/test_dlv.py:147` checks shapes and finiteness, despite
being named as a first-order drift-cancellation test. Astra's independent
autodiff fixture includes nonzero front Brownian loading and obtains maximum
generator residual 6.70e-17 and loading error 1.78e-15. That fixture supports
the continuous-time recursion; it does not validate finite-step martingality.
The actual cancellation calculation should become a permanent regression.

## What the review did not find

No confirmed objective-scaling, gradient-disconnection, normalization-leakage,
eligibility, settlement, or continuous-time Q-recursion error explains the
original zero-action selection. Objective values and gradients match the
specified local entropy and squared losses. Inner and outer splits preserve
antithetic pairs; normalization is training-only. These checks do not establish
the population optimum or absence of predictable drift.

## Historical GPU evidence and its limits

The repaired eight-step checkpoints retain original CUDA storage tags, and
the code keeps the training fold and minibatch gathers on the GPU. The full
raw population is about 41.44 GiB; one outer training fold is about 20.72 GiB.
The fold fits in VRAM, not the entire raw population.

All 600 repaired eight-step epoch timing records were inspected:

| Phase | Median epoch wall | Training wall | Evaluation wall | Checkpoint wall |
|---|---:|---:|---:|---:|
| Affine | 0.690 s | 0.469 s | 0.214 s | 0.006 s |
| Full network | 1.036 s | 0.707 s | 0.319 s | 0.010 s |

There is a real per-batch host synchronization at
`market_training_runtime.py:312`, before optimizer replay. Startup includes
normalization, I/O/upload and compilation; those stages can leave the GPU idle.
These particular epoch records exclude a multi-minute stall inside the
recorded training intervals. They contain no GPU-utilization or CUDA-event
timeline, however, so they cannot reconstruct or dismiss an earlier sustained
0% utilization observation, especially for the separate deep-hedging run.
Update counts and allocated VRAM are not utilization measurements.

## Validation and next acceptance criteria

The user subsequently required no undocumented deviations from the paper
methodology. Before rerunning production, reconcile the simulator against the
paper specification. Existing drift-tamed populations must be labelled as a
modified numerical scheme, not as faithful validation of the prescribed Q SDE.
Do not silently remove taming, alter coefficients or change the discretization
and reuse the old simulation identity; a changed scheme needs its own explicit
provenance and stability checks.

Astra ran 48 CPU tests successfully, with seven GPU-only skips, plus independent
generator and resume-control reproducers. The primary agent separately ran
28 focused tests successfully, with one allocated-GPU skip. These test counts
overlap and must not be added together. No GPU parity rerun was performed.

Before another production conditional fit:

1. Make resume overrides explicit and serialize actual effective settings.
2. Align scheduling with the zero baseline; distinguish validation stopping
   from optimization convergence and zero-action non-detection.
3. Add genuine generator regressions and simple training-only positive controls.
4. Select batch/learning-rate controls using inner training/validation only;
   record actual batch size, effective rates, optimizer steps, and GPU timing.
5. Freeze the revised protocol before fresh outer evaluation. Existing exposed
   holdouts must not be described as fresh after method development. Do not
   center returns, reweight paths, or alter the simulator to manufacture a pass.
