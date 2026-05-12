"""
Simple dysanos model
"""

from quantdata.dysanos import locations_dysanos, env, SubDir, version

locations_simple = locations_dysanos("simple", fmt_training=SubDir.PYTREE_HDF5, fmt_experiments=SubDir.PYTREE_HDF5)

