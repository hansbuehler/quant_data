# Generated Initializer Label-Lineage Review

## Verdict

**PASS for commit `d22ec9a0893313363fb503f8f53f5528159b1549`.** The
rolling VolGAN and diffusion runners now reconstruct the generated-model
configuration with the authenticated initializer label while retaining the
production case label as the evaluation label. This restores warm-start
compatibility without weakening checkpoint, geometry, dataset, or temporal
lineage checks. The reviewed source is byte-identical to the original
`71778b8a7724d298d22e90862c72dba97f0382c0` review target; the only tree
difference is the unrelated `quant_data_push_package.tar.gz` archive.

## Lineage and Geometry

- Generated evidence authenticates the initializer manifest and checkpoint by
  SHA-256, validates the model variant, dataset manifest, completed training
  budget, and pre-2023 training dates, and requires a non-empty
  `model_config.label` from the initializer manifest.
- The initializer-manifest SHA-256 already enters the generated production-gate
  scope binding, so the recovered model label is part of the pre-evaluation
  evidence rather than a runtime-selected parameter.
- Daily and weekly rolling matrix commands pass that value separately as
  `--model-label`. The existing `--label` remains the production-case/evaluation
  label and therefore does not alter model geometry.
- VolGAN and diffusion reconstruct their model configuration from
  `--model-label`. Their warm-start loaders still require the pinned checkpoint
  SHA-256, checkpoint format, code and user versions, dataset identity,
  precision, completed epoch state, variant where applicable, and the exact
  full `model_config_identity`. Fine-tuned diffusion retains its independently
  authenticated base-checkpoint lineage.
- The three production initializer manifests for VolGAN, AD-Seq-Vol, and
  AD-Seq-Vol-FT were checked directly. Reconstructing each current model config
  with its manifest label reproduced the checkpoint's stored
  `model_config_identity` exactly.

## Verification

Focused CPU verification completed successfully:

- 65 generated-evidence, production-matrix, daily CLI, and weekly CLI tests;
- 2 VolGAN and diffusion historical warm-start training tests;
- Ruff checks on all changed source and test files; and
- `git diff --check`.

Result: **67 passed; Ruff and diff checks passed.**

## Production Requirement

Production must execute from a clean worktree pinned to commit `d22ec9a0` or a
descendant containing the same reviewed fix, with production gates and artifact
namespaces bound to `d22ec9a0`. The prior `71778b8a` namespace reviews another
commit, while the `65fbf1d` production worktree does not contain `--model-label`
support; neither may be used for the rolling generated-model population.
