"""
Dysanos 
"""

from quantdata import env, SubDir, version

locations_dysanos = env.locations("dysanos", fmt_training=SubDir.PYTREE_HDF5, fmt_experiments=SubDir.PYTREE_HDF5)
