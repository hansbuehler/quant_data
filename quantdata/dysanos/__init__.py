"""
Dysanos Benchmarks for Implied Volatility Surface Modelling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

from quantdata import env, SubDir, version

_locations_dysanos = env.locations("dysanos", fmt_training=SubDir.PYTREE_HDF5, fmt_experiments=SubDir.PYTREE_HDF5)
