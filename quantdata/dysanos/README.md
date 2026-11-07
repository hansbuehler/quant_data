# DYSANOS

Dynamic Smooth Arbitrage-Free Non-Parametric Option Surfaces.

The current calibration is a rolling affine-200 surface with a directly
estimated **daily Gaussian AR(1)**. Its sweep is running; older results must
not be presented as completed results for this calibration.

## Documentation

| Document | Purpose |
| --- | --- |
| [MODELS.md](MODELS.md) | Current specification, reporting conventions and historical model disposition. |
| [REVIEW.md](REVIEW.md) | Code/quant review: equations, implementation map, checks and open issues. |
| [MARKET.md](MARKET.md) | Historical smooth-market cache schema; the current licensed reader is identified in the review guide. |
| [v11/README.md](v11/README.md) | Post-paper code entry points. |
| [tex/README.md](tex/README.md) | Manuscript sources and build instructions. |
| [flex/FLEX.md](flex/FLEX.md) | Historical Torch/FLEX specification and research reports. |

Source code belongs here. New datasets, checkpoints, logs and disposable
reports belong in uniquely named run directories under the configured
external data root, resolved through the project environment. Preserve older
70-day models and paper artifacts.
