# Weekly production runner smoke evidence

Date: 2026-10-13

## Scope

The unified public-data weekly runner was exercised for SANOS-LV on the first
eligible origin, 2023-01-17, using both frozen-weekly and restarted-weekly
schedules. Each run used the complete ordered 13-target book and the fixed
strictly-2022 target quantities.

- Public dataset manifest SHA-256:
  `58741552e0f6eea82378a5f26a61e1df04b9d14f3ab166883089812cf903007a`
- Target-notional artifact SHA-256:
  `9494bbe23699126b5137b5ab3b6d53070002e2ba8ee60d92b0327ff747fcf9e1`
- Device: CPU, with CUDA hidden from both services.

## Results

| Schedule | Run identity | Results | Cap hits | Initial elapsed | Restart elapsed |
|---|---|---:|---:|---:|---:|
| Frozen weekly | `8a554763a68225baff5ec34cd3bec5fa709ce5d6328db19e85cb7085b16b777a` | 13 | 0 | 150.9 s | 8.5 s |
| Restarted weekly | `86c46880e0ca4600a6df0e33e2a9fbc89253889903c97fea2f2e3a55b29aa3e9` | 13 | 0 | 250.5 s | 8.5 s |

Both restart runs reused the 2023-01-17 atomic record without fitting or
rewriting it. The restarted reader revalidated the exact five chronological
fit dates; the frozen reader revalidated five repetitions of the origin fit.

## Integrity gates

- Full public-member checksum validation is mandatory before execution.
- The preflight target-notional file hash is mandatory and is part of the run
  identity.
- A non-waiting whole-run file lock enforces one writer per output directory.
- Every record contains five decisions, one common terminal date, reconciled
  step gains and costs, and per-fit artifact fingerprints.
- Numerical precision, device, and debug mode are included in the immutable
  configuration snapshot.
- Generated-model checkpoints and initializers carry the public dataset
  manifest identity.

The independent production re-review in
`reviews/2026-10-13-weekly-production-quant-review.md` passes all five
previously blocking integrity findings. The complete common hedging CPU suite
passes 277 tests.
