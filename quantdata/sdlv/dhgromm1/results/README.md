# Deliberately retained experiment records

These compact reports and loss histories are intentional research artifacts,
not raw training data or model checkpoints. Large paths, weights, logs and
scratch files remain under the external `QUANTDATA_ROOT`.

## Completed base joint-500 run, 4 November 2026

`20261104-joint500/` records the completed run
`dhgromm1-joint500-20261104-Ll97mr`. Both networks trained for 500 epochs on
the original 100,000 one-step/day training paths. Epoch 386 was selected by
minimum summed training loss (-0.0145382682, versus +0.0043002508 at warm
start); late training became unstable. The independent evaluation **failed**:
the reported long/short frozen-translation OCEs were approximately -931.62
and -524.60. These are unusable hedge outcomes, not credible price quotes.
The report's `status: complete` denotes successful execution, not convergence
or a passed financial test. Its full identities, configurations and source
hashes are retained verbatim; the loss CSV is an epoch-end training history.

## Training-path diagnosis

`20261104-training-paths/` uses only the first 4,000 original training paths,
not validation. Both the selected epoch-386 and final epoch-500 weights were
examined. The detailed case is the selected short policy on training path
2412, particularly day 10 to 11. Aggregate marked P&L was -0.2713413. The
shared first factor's linear contribution (-0.2734567) combines front/spot
and correlated surface shocks: the separate contributions were -0.2047756
and -0.0686811. It must not be described as pure spot loss. The other seven
surface factors contribute -0.0064757, background Y -0.0010272, and the
drift/nonlinear remainder +0.0096184.

The local exposure matrices span nine directions at relative threshold
1e-6 but can be poorly conditioned (approximately 90,000--200,000 near this
loss). This is a statement about local Greeks, **not** a proof of exact
finite-day market completeness or identification of the optimizer's failure.
Reconstructed opening marks agree with the original marks to about 6.5e-8.
Detailed cohort CSVs, replayed training-case states, and the analysis scripts
are retained externally in
`experiments/quant_data/sdlv/dhgromm1-training-diagnosis-20261104-j3gmUa/`.

## Exact-date-head verification

`20261104-event-heads/gpu-verification.json` records the preflight comparison
of compiled CUDA values and every trainable parameter gradient against the
reference calculation at batch size 256. The date-0 and date-10 additions
were nonzero during this check. This is an implementation regression check,
not evidence of training convergence or successful hedging.

The event run starts from the selected epoch-386 base weights. It trains
the background/all-time strategy and both dated heads together, on training
data only. No result of that still-running experiment is asserted here.

## Regeneration

The recorded report gives the model calibration and scenario identity,
seeds, original chunk/batch layout, utility, regularizer and training setup.
Use `rolling_simulation` to regenerate paths, `feedback_run` for the base
joint experiment, `event_run` for the date-head extension, and `plot_losses`
to render the loss CSV. Preserve complete antithetic pairs. Restore external
calibrations and source checkpoints before rerunning; their locations and
hashes are in the reports. Do not cold-start and label that an exact restart.
