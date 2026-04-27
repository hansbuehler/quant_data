"""
Compute defaults
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PoolConfig:
    """ Default pool configuration for parallel processing. """
    num_workers: int = 4
    threading: bool = False
    
