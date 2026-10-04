# Daily stochastic DLV

This directory contains the retained daily 25-expiry by 200-strike stochastic
DLV calibration workspace.

## Contents

- `train.py`: end-to-end JAX dataset preparation, fitting, and audit runner.
- `data/`: immutable 891-date market calibration tensor.
- `artifacts/`: initial 300-epoch and continued 1000-epoch fits for the full,
  affine-100, affine-200, and affine-400 state representations.
- `artifacts.py`: stable local paths and reconstruction helpers.

The historical reports retain the absolute paths used when the runs were
created. Use `quantdata.sdlv.daily.DATASET_PATH` and
`quantdata.sdlv.daily.FITS` for the relocated paths.

Run the relocated CLI with `python -m quantdata.sdlv.daily.train`. The former
`python -m quantdata.dysanos.flex.run_daily_25_jax_end_to_end` command remains
available as a compatibility entry point.

```python
from quantdata.sdlv.daily import load_raw_grid

raw_grid = load_raw_grid("affine-200", date_index=0)
assert raw_grid.shape == (25, 200)
```

The continued fits contain 891 date codes and use their corresponding final
1000-epoch checkpoints. The `full` fit stores all raw grids directly; affine
fits reconstruct them as `mean + codes @ loadings`.
