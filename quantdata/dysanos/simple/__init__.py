"""
Simple dysanos model
"""

from quantdata.dysanos import training_root_dysanos, env, SubDir, version

training_root_simple = training_root_dysanos("simple", fmt=SubDir.PYTREE_HDF5)
