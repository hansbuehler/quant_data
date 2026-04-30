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

    def __str__(self):
        return f"workers {self.num_workers}{'t' if self.threading else 'p'}"
    
    @staticmethod
    def off():
        """ Pool configuration for no parallel processing. """
        return PoolConfig(num_workers=0, threading=True)
    
    