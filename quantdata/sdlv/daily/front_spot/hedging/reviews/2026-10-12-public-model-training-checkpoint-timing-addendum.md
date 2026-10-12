# Public model training: timing and checkpoint addendum

## Gate recommendation

**FAIL overall.** Blocker 1 (live as-of timing) is **PASS**. Blocker 3
(checkpoint identity and exact restart) remains **FAIL** because a completed
checkpoint is still reused after outcome-relevant training settings change.

## Blocker 1: live as-of timing — PASS

- The VolGAN live builder constructs a 22-date history ending exactly on the
  decision date, rejects an incorrectly declared as-of date and retained-date
  gaps, and assigns the next retained one-session date as its forecast endpoint
  (`models/volgan/data.py:85`, `models/volgan/data.py:106`).
- Diffusion applies the same dated contract to its 21-observation live context
  (`models/diffusion/data.py:84`, `models/diffusion/data.py:105`).
- Both one-day adapters build these live states at the hedge origin and require
  the forecast endpoint to equal the realized benchmark endpoint before
  sampling (`models/volgan/adapter.py:115`, `models/diffusion/adapter.py:119`).
  The panel helper itself rejects a final observation or a retained-date gap
  (`data/market_panel.py:618`).
- On the public 2023-01-17 diagnostic, both live states ended on 2023-01-17 and
  forecast 2023-01-18. Their values differed from the obsolete final training
  inputs, which ended on 2023-01-13 and targeted 2023-01-17.
- The focused tests independently reconstruct the last historical training
  condition from a history ending at its condition date and reject a false
  as-of declaration (`test_volgan.py:76`, `test_diffusion.py:93`).

## Blocker 3: checkpoint identity and restart — FAIL

The main restart mechanics are substantially repaired:

- exact array/date fingerprints now enter checkpoint names and compatibility
  checks; diffusion also hashes its fine-tuning base checkpoint
  (`models/checkpointing.py:21`, `models/volgan/training.py:68`,
  `models/diffusion/training.py:78`, `models/gromm/training.py:185`);
- model, optimizer, random-generator and best-state data are saved, and the
  GROMM final checkpoint is written before the returned policy is replaced by
  the best validation state (`models/gromm/training.py:469`); and
- same-configuration forced-interruption tests reproduce uninterrupted VolGAN
  and diffusion state dictionaries exactly, while the GROMM budget-extension
  test reproduces the uninterrupted four-epoch result exactly
  (`test_volgan.py:116`, `test_diffusion.py:205`,
  `test_gromm_training.py:203`).

However, training configuration is only warned about, not made part of hard
compatibility (`models/checkpointing.py:57`). VolGAN, diffusion and GROMM all
load the checkpoint after such a warning (`models/volgan/training.py:181`,
`models/diffusion/training.py:194`, `models/gromm/training.py:278`). VolGAN and
diffusion then return immediately when the saved epoch already meets the new
budget (`models/volgan/training.py:232`, `models/diffusion/training.py:264`),
and a converged GROMM checkpoint likewise returns immediately
(`models/gromm/training.py:326`).

Independent reproduction: a completed one-epoch VolGAN fit with seed 13 and a
20% validation block was requested again at the same checkpoint directory with
seed 99 and a 30% validation block. It emitted
`Warm-starting VolGAN with changed training fields: seed, validation_fraction`,
used the same checkpoint path, ran no additional epoch, and returned
`completed_epochs=1, converged=True`. The reported split-dependent diagnostics
therefore describe a different requested split than the fitted checkpoint.

**Required fix:** hard-reject outcome-relevant configuration changes by
default, or include them in checkpoint identity. Only explicitly safe
operational fields and a deliberate training-budget extension should be
compatible. Any separate warm-start mode should require explicit opt-in and
must not be reported as an exact completed resume under a changed split.

## Numerical DTE finding

The current full diffusion hedge smoke reaches a quote with
`business_dte=1.0000000577419996` and is rejected by the `1e-8` integer check
in `models/generated_scenarios.py:331`. This value is the expected error from
converting the released float32 expiry `1/252` back to business days, not a
genuinely fractional expiry. Rounding within a tight absolute tolerance is the
correct fix. The tolerance should be `2e-6`, matching the public-dataset
geometry validation in `quantdata/hedging/data/public_dataset.py:379`; a test
should accept the observed value while still rejecting, for example, `1.0001`
and `1.5`. This issue is separate from the now-correct live timing, but it
currently blocks the end-to-end diffusion smoke.

## Verification

- Focused CPU suite: `15 passed` in `9.96s` for GROMM, VolGAN and diffusion.
- Public 2023-01-17 live-date/value audit: passed for VolGAN and diffusion.
- Completed-checkpoint configuration-drift reproduction: failed safely only by
  warning; stale checkpoint reuse was confirmed.
- `gpuctl resources --json` could not reach the daemon and used the local
  fallback; all reported RAM and disk checks were `OK`. No GPU was used.
- No implementation file was modified by this review.
