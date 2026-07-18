# [Yahoo Finance](https://ranaroussi.github.io/yfinance/)

This directory provides access to Yahoo Finance via https://ranaroussi.github.io/yfinance/.
So far coverage is minimal.

## Data Access

This data is freely available and does not require configuration
beyon what is required by **openqai.core**:

  1. Make sure you configure the python environment for **openqai.core** following [the introduction](../../README.md) which means to set at 
  least the ``QUANTDATA_ROOT`` environment variable. 
  [See the discussion in data locations and how to set them](../../
  README.md#data-locations).

  2. Run [``notebooks/viz_history.ipynb``](notebooks/viz_history.ipynb) to validate everything works.
