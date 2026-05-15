"""
SANOS Smooth Arbitrage-free Non-parametruc Option Surfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[SANOS](https://arxiv.org/abs/2601.11209) is a rather generic no-arbitrage framework which can accomodate any number of model strikes and expires,
potentially depending on market data for example when used in "interpolation mode" as in https://arxiv.org/pdf/2601.11209.

Covered here are:

* :mod:`open`: SANOS adaptively applied to the market strikes. This mode fits well, but the number of model parameters is not constant.
* :mod:`grid`: SANOS applied to a fixed grid of strikes and expires. By definition this migtht fit less well, but it has a fixed number of model parameters
  which is more useful for machine learning applications.

Example:

```python

    geometry = GridGeometry.create( ... )  

    fitter   = Fitter( geometry, ... )
    sanos    = fitter( market_data )
    prices   = sanos.price( strikes, expiries )

````
"""

from dataclasses import dataclass
from .grid import GridGeometry, GridSanos
from .open import OpenGeometry, OpenSanos
from .fitter import Fitter
