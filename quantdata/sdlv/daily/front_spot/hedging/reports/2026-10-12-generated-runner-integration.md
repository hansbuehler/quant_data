# Generated daily-runner integration

## Scope

Both generated baselines were exercised through the resumable daily runner on
the public candidate at origin 2023-01-17. The runner first calibrated the 20
fixed target quantities on the strictly-2022 sample and pinned them in its run
identity.

These deliberately abbreviated fits validate orchestration, checkpoint paths,
scenario-to-hedge conversion, common accounting, and result gates. They are not
hedging-performance estimates.

## Results

| Model | Device | Training | Scenarios | Results | Nonconverged | Cap hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VolGAN | CPU | 2 epochs | 16 | 20 | 0 | 0 |
| AD-Seq-Vol | managed GPU | 1 epoch | 8 | 20 | 0 | 0 |

Every result reported the exact target quantity pinned by its
`hedging-daily-evaluation-v2` run specification. The diffusion run used a
one-off `gpuctl` allocation, which was released on completion and subsequently
verified clean. Production runs must use the published training schedules and
material scenario counts rather than these integration settings.
