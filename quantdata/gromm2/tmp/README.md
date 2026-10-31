# GROMM2 saved-model review snapshot

These five artifacts are deliberately retained in Git at the user's request
on 31 October 2026. This is a temporary review bundle, not a location for
ongoing run outputs or checkpoints. The external originals remain unchanged.

The bundle contains the legacy **GROMM2 (-3,2), affine-200, 16-factor** model
at **29 August 2025**, including its saved factor-law estimates. It is not
the newer (-4,3) rolling model. Historical calibration used an independent
5% Black background; the simulator-only zero-background hedging control does
not modify these files.

## Provenance

The following three files are copied byte-for-byte from
`/home/hans/quantdata_data/training/gromm2/inputs/2025-08-29-affine200/`:

- `affine-200-history.h5`
- `affine-200-models-rank16-final.npz`
- `affine-200-models-rank16-final.json`

These two are copied byte-for-byte from
`/home/hans/quantdata_data/training/gromm2/factor-laws/2025-08-29-affine200-v1/`:

- `factor-laws.npz`
- `factor-laws.json`

`SHA256SUMS` records the original file digests. Run `sha256sum -c SHA256SUMS`
from this directory to verify the snapshot. Absolute paths embedded in the
saved metadata describe its original creation; they have intentionally not
been rewritten. Use explicit local paths when loading the copied files.

See [the saved-model loader](../legacy_forward_model.py) for provenance and
geometry validation, and [DH_TEST.md](../DH_TEST.md) for the associated
simulation and deep-hedging experiment. Simulated path populations are not
part of this bundle.
