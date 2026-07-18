from __future__ import annotations

"""
Pool configuration
^^^^^^^^^^^^^^^^^^

Simple utilities to manage multi-processing workflows. Uses :class:`cdxcore.jcpool.JCPool` under the hood.
Use the ``PoolConfig`` class to allow users to configure their pools. The ``pool()`` member function generates
a unique pool per process.

The main purpose is to support functions which may use pools as different levels::

    def f( ..., pool_config : PoolConfig = PoolConfig.off(), ...):
        ...        
        pool = pool_config.pool()
        ...

    def g( ..., pool_config : PoolConfig = PoolConfig.off(), ...):
        ...
        results = f( ..., pool_config=pool_config, ... )
        ...

        pool = pool_config.pool()
        ..

** :mod:joblib: memory leak detection **

In this project we turned on the rather aggressive memory leak detection of :mod:joblib: 
which is described `here <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`__.
"""

from cdxcore.jcpool import JCPool, JCPoolConfig as PoolConfig
JCPool.DEFAULT_MEM_LEAK_ENFORCE = False
