"""
Basic shared Utilities
======================
"""

from dataclasses import dataclass
from cdxcore.err import verify_inp

@dataclass
class Eps:
    """
    Detailed error tolerance for debugging.
    """
    prices    : float = 1E-6     #: Tolerance for option price errors.
    theta     : float = 1E-6     #: Tolerance for theta errors during DLV computation.
    gamma     : float = 1E-6     #: Tolerance for gamma errors during DLV computation.
    densities : float = 1E-6     #: Tolerance for density errors.
    dlvsq     : float = 1E-6     #: Tolerance for DLV squared errors being positive.

    def __post_init__(self):
        verify_inp( self.prices >= 0., f"Tolerance for prices must be non-negative, found {self.prices}" )
        verify_inp( self.theta >= 0., f"Tolerance for theta must be non-negative, found {self.theta}" )
        verify_inp( self.gamma >= 0., f"Tolerance for gamma must be non-negative, found {self.gamma}" )
        verify_inp( self.densities >= 0., f"Tolerance for densities must be non-negative, found {self.densities}" )
        verify_inp( self.dlvsq >= 0., f"Tolerance for dlvsq must be non-negative, found {self.dlvsq}" )

    def set_all(self, eps : float ):
        verify_inp( eps >= 0., f"Tolerance must be non-negative, found {eps}" )
        self.prices = eps
        self.theta = eps
        self.gamma = eps
        self.densities = eps
        self.dlvsq = eps
        
    @staticmethod
    def create( eps : float ) -> 'Eps':
        """ Create an Eps object with all tolerances set to the same value. """
        verify_inp( eps >= 0., f"Tolerance must be non-negative, found {eps}" )
        return Eps( prices=eps, theta=eps, gamma=eps, densities=eps, dlvsq=eps )

