# Agent baseline

Read and follow `STYLE_GUIDE.md` before making code changes in this repository. More specific instructions in subdirectories supplement or override the repository baseline.

## Host Resource Guardrails

- Check RAM and filesystem headroom before launching material work and during
  every regular background-process inspection. Resource-triggered termination
  applies only to jobs owned by the current project; never terminate unrelated
  system or user activity.
- Measure usable RAM with Linux ``MemAvailable`` from ``/proc/meminfo``
  (equivalently the ``available`` column from ``free``), not raw ``MemFree``.
  Define memory usage as ``1 - MemAvailable / MemTotal``. At 80% usage, warn the
  user promptly and identify the largest owned processes and whether usage is
  stable or increasing. At 90% usage, do not launch more work; gracefully
  terminate active jobs owned by the current project after preserving their
  checkpoints and status, and immediately raise an alert in the active user
  conversation. Do not restart until the cause has been assessed and safe
  headroom restored.
- Monitor the filesystem device that hosts ``/media/storage`` using its capacity
  and available bytes. If free space falls below 20%, warn the user promptly and
  report current usage and trend. If free space is at or below 10%, do not
  launch more work; gracefully terminate active project jobs after preserving
  checkpoints and status, and immediately raise an alert in the active user
  conversation. Do not restart until the space problem is resolved.
- Monitor usage of the filesystem hosting ``/tmp``. At 70% usage, initiate a
  safe cleanup and re-measure. Cleanup must be limited to identified stale or
  reproducible temporary files owned by the current project; never
  blanket-delete ``/tmp``, remove open or shared files, or delete reusable
  caches. At 80% usage, warn the user promptly and report cleanup actions and
  the usage trend. At 90% usage, cease project operation gracefully, preserve
  checkpoints and status, and immediately raise an alert in the active user
  conversation. Do not restart until safe headroom is restored.
- Treat Git working trees under ``~/python``, including this repository, as
  source repositories rather than general-purpose data or artifact storage.
  Regularly inspect repository status, total footprint, untracked files, and
  unusually large files. Keep only relevant source, configuration,
  specifications, tests, and deliberately retained project artifacts in Git
  roots. Never remove user files merely because they are large without first
  identifying ownership and obtaining any required approval.
