"""
openqai.core
============

Commons. Shared interfaces that two or more consumers must agree on:
environment and cache locations, the surface state, risk measures, P&L
accounting, and the numerical backbones.

This layer imports nothing from ``openqai.data`` or ``openqai.benchmarks``.
"""

from openqai.core.env import (  # noqa: F401
    Environment,
    Locations,
    env,
    SubDir,
    Context,
    version,
    __version__,
)

__all__ = [
    "Environment",
    "Locations",
    "env",
    "SubDir",
    "Context",
    "version",
    "__version__",
]
