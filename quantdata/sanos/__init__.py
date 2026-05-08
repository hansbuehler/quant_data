"""
SANOS Smooth Arbitrage-free Non-parametruc Option Surfaces

SANOS itself is a rather generic framework which can accomodate any number of model strikes and expires, potentially depending on market data for example when used in "interpolatnion mode" as in https://arxiv.org/pdf/2601.11209.
Grid SANOS, on the other hand, specifies SANOS on a fixed strike and expiry grid for a machine learned representation of an option surface. 

The framework here represents a fixed grid approach. The core class is :class:`dys.grid_sanos.GridGeometry`, which represents the geometry of the grid.

**Main components**

* :class:`GridGeometry`: represents the expiries and strikes of the grid, but no market data.
* :class:`GridSanos`: represents the SANOS framework on the grid, including market data and transition operators.
* :class:`Fitter`: fits the SANOS model to market data.

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
