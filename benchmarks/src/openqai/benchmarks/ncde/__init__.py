"""
NCDE - Neural Controlled Differential Equations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Project directory for NCDE based benchmarks.

"""

from openqai.core import env, SubDir, version
_locations_ncde = env.locations("ncde", fmt_training=SubDir.PYTREE_HDF5, fmt_experiments=SubDir.PYTREE_HDF5)
