# Agent baseline

Read and follow `STYLE_GUIDE.md` before making code changes in this repository. More specific instructions in subdirectories supplement or override the repository baseline.

Host-wide guardrails that apply to every project on this machine — GPU
allocation via `gpuctl`, resource-headroom handling, GPU progress-logging
cadence, and launching detached jobs — are defined once in
`~/.codex/AGENTS.md` and apply here too; this file only adds what's specific
to quant_data.

## Shared directories

When using any common directory, such as `/tmp` or a shared storage root,
place every file this project generates under a `quant_data/` subdirectory
of that root (e.g. `/tmp/quant_data/`). Do not scatter project outputs,
logs, checkpoints, caches, or intermediate files directly across a common
directory; keep them together so ownership and cleanup remain unambiguous.

## Project environment

Detached jobs and user services for this project must be launched with the
explicit project interpreter and working directory, and must set
`QUANTDATA_ROOT` and `PYTHONPATH` in the job definition. Before starting
material work, validate imports and data-root discovery in the same
detached environment that will run the job.

## Git working tree hygiene

Treat Git working trees under `~/python`, including this repository, as
source repositories rather than general-purpose data or artifact storage.
Regularly inspect repository status, total footprint, untracked files, and
unusually large files. Keep only relevant source, configuration,
specifications, tests, and deliberately retained project artifacts in Git
roots. Never remove user files merely because they are large without first
identifying ownership and obtaining any required approval.
