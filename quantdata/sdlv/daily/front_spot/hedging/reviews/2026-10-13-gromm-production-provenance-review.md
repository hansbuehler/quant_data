# GROMM Production Provenance Review

**Date:** 2026-10-13  
**Verdict:** **PASS**  
**Reviewed commit:** `dbf8bdea021e0e1e1f81463a682a93817c2645a1`

## Scope

Final independent review of the rolling GROMM2 production range and the queued
generated-model training pipeline. The review covered the committed source,
the clean pinned worktree, stale-output invalidation, dated and range-level
source authentication, concurrent-writer protection, the active v5 production
wrapper, and the generated v5 wrapper. No implementation file was modified.

## Verified invariants

- The repository and `/tmp/quant_data/hedging/worktrees/native-dbf8bde` resolve
  to the reviewed commit. The pinned worktree has no tracked modifications.
- Every dated rolling artifact uses schema v4 and records a source identity
  covering the numerical GROMM2, rolling-factor, calibration, public-data,
  pricing, and calendar implementations. Loading rejects another source
  identity. The range identity composes this dated-artifact identity with the
  range runner implementation.
- Re-preparation cannot relabel stale fitted output. After replacing source
  history and prepared books, `prepare_rolling_gromm2()` removes affine history,
  its checkpoint, dynamics, and the dynamics report before publishing the new
  prepared manifest. The committed regression verifies all four removals.
- Per-origin file locks serialize overlapping range writers. The production
  wrapper adds a pipeline-level lock, runs three disjoint preparation shards,
  then three fit shards through one managed GPU allocation, and finally creates
  canonical summaries. The completed fit summary is authenticated against the
  dataset manifest, artifact root, source identity, and every dated-manifest
  hash.
- The generated wrapper pins the same commit, waits for the v5 GROMM service,
  authenticates its canonical completed range before training, and holds a
  single-writer lock. Fine-tuned diffusion initializer validation reconciles
  the expected base manifest path, checkpoint hash, and checkpoint-internal
  base fingerprint for both selection and production initializers.
- Resource checks are enforced before material stages. The active preparation
  log reports elapsed time and completed counts for all three shards every 240
  seconds, satisfying the host progress-reporting requirement.

## Evidence

The following focused checks were performed:

```text
git rev-parse HEAD
git -C /tmp/quant_data/hedging/worktrees/native-dbf8bde rev-parse HEAD
git -C /tmp/quant_data/hedging/worktrees/native-dbf8bde status --short --untracked-files=no
bash -n /tmp/quant_data/hedging/gromm_production_pipeline_dbf8bde.sh
bash -n /tmp/quant_data/hedging/generated_training_pipeline_v1.sh
python -c "import inspect; ...; print(inspect.signature(load_generated_initializer_evidence))"
gpuctl resources --json
tail /home/hans/quantdata_data/training/hedging/gromm2-rolling-v3/operations/dbf8bde/pipeline.log
```

The pinned API exposes `expected_base_initializer_manifest`; both wrappers pass
shell syntax validation; all configured resource checks were `OK`; and the
active log showed all three preparation shards advancing at each heartbeat.
The committed focused remediation gate reports 90 relevant tests passing, with
Ruff, compilation, diff, and shell-syntax checks also passing.

## Conclusion

The previous overlapping-writer, unauthenticated-source, stale-fit reuse,
fine-tuned base-lineage, and weak-sentinel blockers are resolved in the pinned
deployment. No concrete production blocker remains for the rolling GROMM2
range or the subsequent generated-model training pipeline.
