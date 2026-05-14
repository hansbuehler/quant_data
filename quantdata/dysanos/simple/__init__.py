"""
Simple Dysaons: Thin Benchmarking for Implied Volatility Surface Modelling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **dysanos.simple** benchmark is a simple benchmark framework built around the following
problem geometry:

* We look at short time $t_0,\ldots,t_N$ series of data with the purpose of optimizing objectives at $t_N$.

* Each sample contains the entire time series; losses are not computed for intermediate points.
  Samples are drawn across time from multiple tickers. Samples are split into a training and validation set,
  where the former is strictly ahead of the latter in time. However, both contain all tickers (to the extent data
  is available).

* Samples sourced from the original time series may overlap. The :class:`quantdata.dysanos.simple.SimpleSampler`
  class provides a convenient way to sample from the original time series in a way that allows for overlapping samples
  using its :attr:`SamplingConfig.num_slide` parameter.

Main Classes
============

* :class:`quantdata.dysanos.simple.SimpleSampler` for sampling from the original time series
  in a way that allows for overlapping samples.

* :class:`quantdata.dysanos.simple.SimpleDataset` for wrapping the samples into a PyTorch dataset.  

"""

from quantdata.dysanos import _locations_dysanos, env, SubDir, version

_locations_simple = _locations_dysanos("simple", fmt_training=SubDir.PYTREE_HDF5, fmt_experiments=SubDir.PYTREE_HDF5)

