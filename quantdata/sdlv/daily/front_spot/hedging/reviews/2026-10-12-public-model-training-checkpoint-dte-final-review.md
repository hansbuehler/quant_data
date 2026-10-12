# Public model training: checkpoint and DTE final review

## Verdict

**PASS.** The checkpoint-identity/restart blocker and the generated-scenario
business-DTE failure are remediated. No blocking finding remains in this review
scope.

## Checkpoint configuration policy — PASS

- `validate_train_config_change` now rejects changed controls by default,
  rejects `seed` and `validation_fraction` unconditionally, and permits other
  changed controls only with the explicit
  `allow_changed_config_warm_start=True` opt-in
  (`models/checkpointing.py:57`).
- VolGAN, diffusion and GROMM all apply that policy before loading checkpoint
  state, with seed and split declared immutable
  (`models/volgan/training.py:189`, `models/diffusion/training.py:205`,
  `models/gromm/training.py:289`).
- An opted-in changed configuration must also request additional epochs;
  otherwise all three models reject it rather than relabel a completed fit
  (`models/volgan/training.py:226`, `models/diffusion/training.py:261`,
  `models/gromm/training.py:324`). GROMM additionally clears the prior
  convergence/staleness state before continuing.
- The three seed/split rejection tests pass
  (`test_volgan.py:176`, `test_diffusion.py:273`,
  `test_gromm_training.py:267`). The VolGAN regression also proves default
  rejection of a changed learning rate and successful explicit warm-start
  continuation (`test_volgan.py:213`). Independent CPU checks confirmed the
  same default-reject/explicit-opt-in behavior for diffusion and GROMM: each
  trained one additional epoch after the opt-in and emitted the expected
  changed-learning-rate warning.

## Exact restart — PASS

- Forced interruption followed by resume reproduces uninterrupted VolGAN
  generator/discriminator parameters exactly (`test_volgan.py:116`).
- Forced interruption followed by resume reproduces uninterrupted diffusion
  model and EMA parameters exactly (`test_diffusion.py:205`).
- A two-to-four-epoch GROMM budget extension reproduces the uninterrupted
  four-epoch best state and validation result bit-for-bit
  (`test_gromm_training.py:203`).
- On the public 2023-01-17 diffusion smoke, a fresh three-epoch fit and a
  subsequent `--just-load` replay produced byte-identical result JSON files
  (SHA-256
  `ff80574e68714340575028db8cf2cf23ed08f8967204dd936dfb682fa2713439`).

## Business-DTE handling — PASS

- `_integral_business_dte` now rounds only within `2e-6`, matching the released
  public-geometry validation tolerance, and still rejects genuinely fractional
  expiries (`models/generated_scenarios.py:331`).
- The regression accepts the observed float32 round-trip values for 1, 5 and 21
  business days, including `1.0000000577419996`, while rejecting `0`, `1.0001`
  and `1.5` (`test_generated_scenarios.py:99`).
- A fresh public 2023-01-17 diffusion hedge smoke completed end to end with 146
  training sequences, three epochs, 64 generated paths and 163 hedge
  instruments. The former DTE exception did not recur.

## Verification

- Focused CPU suite: `27 passed` in `7.14s`.
- Independent mutable-control warm starts: diffusion and GROMM passed.
- Fresh and just-loaded public diffusion smokes: both passed and were identical.
- `gpuctl resources --json` could not reach the daemon and used the local
  fallback; every reported RAM and disk check was `OK`. No GPU was used.
- No implementation file was modified by this review.

## Non-blocking test coverage

The implementation uses the same shared fail-closed policy for every model,
and the independent checks covered all three. For symmetry, retain dedicated
diffusion and GROMM regression tests for mutable-control default rejection and
explicit warm-start continuation, analogous to the committed VolGAN test.
