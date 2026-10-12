# Generated-model managed-GPU readiness

## Scope

This gate used only the distributable hedging dataset and the managed `gpuctl`
allocation path. A CPU preflight first authenticated the fixed 2022 target-size
artifact, so the GPU allocation covered only the rolling diffusion fit and
scenario evaluation.

## Result

- Model: full-geometry AD-Seq-Vol, 8 transformer layers, width 256, 200
  diffusion steps.
- Rolling origin: 2023-01-17, using 252 historical observations and the pinned
  pre-evaluation initializer SHA-256
  `944ab2a8764f573be49e36168fc2706b25fb22c9d55db0ee44124a6349ec7235`.
- Work: one update epoch, 4,096 generated scenarios, and all 20 daily digital
  targets.
- Outcome: 20 results, zero cap hits, zero nonconverged results, and a reported
  elapsed time of 158.3 seconds.
- Resource observation: 2,636 MiB used VRAM during the run; host RAM and all
  monitored disks remained `OK` under `gpuctl resources --json`.
- Allocation lifecycle: `gpuctl run` acquired the GPU and released it when the
  process exited.

The first attempt was interrupted after its epoch checkpoint. Resuming that
checkpoint exposed that loading an RNG tensor directly onto CUDA is invalid for
`torch.set_rng_state`. Restart payloads are now loaded on CPU before model and
optimizer state restoration. The exact-restart CPU regression test passes, and
the same interrupted managed-GPU checkpoint then resumed and completed the
origin successfully.

## Evidence

- CPU preflight manifest:
  `/tmp/quant_data/hedging-gpu-gate/preflight/manifest.json`
- Persistent GPU log:
  `/tmp/quant_data/hedging-gpu-gate/diffusion-origin/run.log`
- Evaluation manifest:
  `/tmp/quant_data/hedging-gpu-gate/diffusion-origin/results/run.json`
- Origin record:
  `/tmp/quant_data/hedging-gpu-gate/diffusion-origin/results/records/2023-01-17.json`

The scratch evidence is not a production benchmark result. It establishes that
the full architecture, authenticated input handoff, managed allocation,
checkpoint resume, sampling, and common daily accounting execute together.
