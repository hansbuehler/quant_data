# Production handoff remediation

## Scope

This record covers the first live handoff from the completed legacy rolling
GROMM range into current-code migration and generated-model GPU readiness. It
records operational defects discovered by the production run, the narrow source
changes used to fix them, and the evidence required before either successor can
feed the final benchmark matrices.

## Completed legacy range

The legacy affine range completed all 611 eligible origins. The canonical fit
summary is
`/home/hans/quantdata_data/training/hedging/gromm2-rolling-v3/range-fit-2023-2025-v5.json`
with SHA-256
`6cbdcb13913d322d95e1ff644950f638aaf090ea36278553d6eb898b2bce3205`.
It reports 611 selected and completed origins and passed its canonical range
loader before the GPU allocation was released.

The launcher's exit trap initially waited on every shell job, including the
`tee` process substitution used for logging. That left the already-complete
service active even after its final success record and GPU release. Cleanup now
waits only for explicitly tracked model-worker PIDs. The hardened production
launcher SHA-256 is
`3fc43ad8a848261c51fa9b897554c7a7da794913fb64f43c3e6fa20c863013c1`.
The same correction is present in the migration launcher.

## Generated checkpoint resume

The first generated-readiness attempt failed while resuming VolGAN on CUDA:
`torch.set_rng_state` received a checkpoint tensor mapped onto the CUDA device
rather than a CPU byte tensor. VolGAN and GROMM policy checkpoints now load on
CPU before model, optimizer, and RNG restoration, matching the existing
diffusion implementation. This does not change the logical model or checkpoint
format.

Commit `b4348520c362fbf19e9fe1451f6546022f9f5241` contains the source fix.
The exact-restart regressions assert the CPU map location and reproduce the
uninterrupted weights. The focused VolGAN and GROMM training modules pass 17
tests. The immutable readiness launcher SHA-256 is
`5e81b05b0df7ef38aa6a3dbc2e65e915ccc2dd2ac82acf400a1150af1add60de`;
its preflight authenticated the generated training completion, imports, and
process-group shutdown path before service
`quantdata-hedging-generated-readiness-b434852.service` claimed the managed
GPU.

## Legacy migration identity

The first current-code migration attempt authenticated the range summary but
then passed its range-runner source identity as the expected source identity of
each dated market-model manifest. Those are deliberately different identities:

- range source:
  `80c4222fdbd27919d63fadfe8a1b48d7a99cd0a167c6b1bdd078765fb4e2bd26`;
- dated-manifest source:
  `7d5b32349ab807d63911adcffbefbd58ea212e0be3d7cf18d4182c33d59af7d1`.

Commit `661d5752c7813de3d7ab2cbdb8998a0b0ccc47ee` makes the distinction explicit.
The legacy range loader now authenticates every manifest hash, requires one
common valid dated-manifest identity across the cohort, preserves the separate
range identity, and passes the dated identity into affine migration. The two
focused range modules pass 21 tests, and the real preflight authenticates all
611 manifests with the identities above before any GPU claim. The immutable
migration launcher SHA-256 is
`11fcbcad32d5c1a6a7b39b08b41b0fd1df2d3d49f74060bc9b4f408d9da5573c`.

## Downstream handoff

The full-production coordinator remains pinned to its reviewed source commit,
but now waits on the corrected migration and generated-readiness services and
reads `range-fit-2023-2025-v7.json` plus the
`generated-readiness-b434852` evidence root. Its persisted launcher SHA-256 is
`d3787f21c7a8b3f61b2fc84d41e4c8c2d1a51fa7d0607d7973acb652aaf05d06`.
No daily or weekly benchmark output was relabelled or reused to conceal either
failure.
