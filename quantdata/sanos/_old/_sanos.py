"""
Grid Sanos

Implementation of the SANOS model from https://arxiv.org/pdf/2601.11209
for an input grid of model strikes and expories.
The model can then produce call prices as function of forward volatilities and discrete local volatilities. This works for ``numpy``
and ``torch`` so far, with ``jax`` support coming soon.

The model also provides as ``fit_dlvs`` function which can be used to
fit DLVs given option prices and forward volatities.
"""

from __future__ import annotations

from wandb.util import np

from quantdata.sanos._old._geometry import Geometry
from quantdata.utils.backbone import Backbone, BackboneNumpy
from quantdata.sanos._old._bbcalc import SanosBackbone
from cdxcore.verbose import Context
from cdxcore.pretty import PrettyObject
from cdxcore.err  import verify_inp
import cdxcore.bs as bs
from dataclasses import dataclass
import warnings
import numpy as nnp

class Sanos(object):
    """
    Standard Grid Sanos with learnable foward volatilities and dynamic option request grid:
    the model can price any call option based just on its pure strikes and time to expory.

    Parameters
    ----------
    geometry : :class:`dys.grid_sanos.GridGeometry`
        Geometry of the model, which includes the grid of strikes and expiries, and the transition operators for the interpolation step.

    backbone : :class:`dys.grid_sanos.Backbone`, default ``BackboneNumpy()``
        Backbone to use for the model; this controls the array library used for the computations, and also provides
        utility functions for debugging and verification. The default is a numpy backbone, which does not require any additional dependencies.

    debug : bool, default ``False``
        Whether to enable debug mode. If True, the model will perform additional checks and print debug information.
        This is heavy; avoid in a production setting.

    cvxpy_solver : str, default ``CLARABEL``
        Solver to use for the convex optimization problem in :meth:`fit_dlvsqs`, c.f. `list of cvxpy solvers <https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver>`_.

    cvxpy_solver_options : dict | None, default ``None``
        Options for the convex optimization solver; this can be used to pass additional parameters to the solver.

    """
    def __init__(self, geometry : Geometry,
                       fwd_vols : Backbone.Array|float|None,*,
                       densities: list[Backbone.Array]|None = None,
                       dlvsqs   : Backbone.Array|list[Backbone.Array]|None = None,
                       debug    : bool = False,
                       bb       : Backbone|None = None
                       ):
        """ Initialize """

        super().__init__()
        self.bb_calc   = SanosBackbone( geometry=geometry, backbone=bb ) #: :class:`dys.grid_sanos.bbcalc.SanosBackbone` Backbone calculator for the model, which provides the core computations for pricing and fitting.
        self.fwd_vols  = None
        self.densities = None
        self._dlvsqs   = None
        self.debug     = debug

        verify_inp( (not dlvsqs is None) and (densities is None), "At least one of 'dlvsqs' or 'densities' must be provided" )

        if not fwd_vols is None:
            if hasattr(fwd_vols, "shape"):
                fwd_vols = bb.to_array( fwd_vols ) 
                assert fwd_vols.shape[-1] == self.T, f"Expected 'fwd_vols' to have shape ending with {self.T}, found {fwd_vols.shape}"
                self.verify( lambda : bb.min( fwd_vols ) >= 0., lambda : f"Forward volatilities must be positive, found min {self.bb.min(fwd_vols)}" )
            else:
                fwd_vols = bb.full( ( self.T, ), fwd_vols, dtype=bb.float32 )

        if densities is None:
            self._dlvsqs   = bb.to_array( dlvsqs ) if hasattr(dlvsqs,"shape") else [ bb.to_array( dlvsq ) for dlvsq in dlvsqs ]
            self.densities = self.bb_calc.densities( dlvsqs=self._dlvsqs ) # returns a list

        else:
            assert not hasattr( densities, "shape" ), f"Expected 'densities' to be a list of Backbone.Array, found type {type(densities)} with shape {getattr(densities,'shape',None)}" 
            self.densities = [ bb.to_array( density ) for density in densities ]

    @property
    def geometry(self) -> Geometry:
        return self.bb_calc.geometry
    
    @property
    def bb(self) -> Backbone:
        return self.bb_calc.bb

    @property
    def grid_mode(self) -> str:
        """ Grid mode, which is either 'static' or 'dynamic'. In static mode, the model can only price options on the grid defined by the geometry. In dynamic mode, the model can price any option based on its strikes and time to expiry. """
        return self.geometry.grid_mode  
    
    @property
    def T(self)-> int:
        """ Number of expiries """
        return self.geometry.T

    @property
    def N(self) -> int|None:
        """ Number of strikes if in grid mode, None otherwise """
        return self.geometry.N    

    @property
    def dlvsqs(self) -> Backbone.Array|list[Backbone.Array]:
        if self._dlvsqs is None:
            self._dlvsqs = self.bb_calc.dlvsqs( densities=self.densities ) # returns a list natrix for grid_mode, otherwise a list
        return self._dlvsqs

    def __call__( self, opt_strikes      : Backbone.Array,
                        opt_tte          : Backbone.Array, *,
                        opt_is_calls     : Backbone.Array | None = None, 
                        opt_mask         : Backbone.Array | None = None,
                        return_implieds  : bool = False ) -> Backbone.Array|tuple[Backbone.Array, Backbone.Array]:
        """
        Compute the grid of option prices given forward volatilities and discrete local volatilities using https://arxiv.org/pdf/2601.11209.

        Parameters
        ----------
        opt_strikes : array of shape (...,M)
            Strikes of the options to price.
        
        opt_tte : array of shape (...,M)    
            Time to expiry of the options to price.
        
        opt_mask : array of shape (...,M) | None, default ``None``
            Mask for which options to price; if False, the model will skip the computations for that option and return 0. This can be used to speed up computations when only a subset of options is needed.

        opt_is_calls : array of shape (...,M)
            Boolean array indicating whether each option is a call (True) or put (False).

        return_implieds : bool, default ``False``
            Whether to return the implied volatilities as well.
            This currently only works in numpy mode.

        Returns
        -------
        prices : array of shape (...,M)
            Prices of the options with given strikes and times to expiry

        implieds : array of shape (...,M)
            Implied volatilities of the options; only returned if ``return_implieds`` is True.
        """ 
        prices   = self.bb_calc.prices( fwd_vols=self.fwd_vols,#
                                        dlvsqs=self.dlvsqs, 
                                        opt_strikes=opt_strikes, 
                                        opt_tte=opt_tte, 
                                        opt_is_calls=opt_is_calls, 
                                        opt_mask=opt_mask )
        self.verify( lambda : self.bb.all( self.bb.isfinite(prices) ), lambda : f"Computed prices must be finite")
        if not return_implieds:
            return prices

        assert isinstance( self.bb, BackboneNumpy ), f"Implied volatilities can only be returned if the model is initialized with a Backbone, found type {type(self.bb)}"
        implieds = bs.impliedS( k=opt_strikes, prices=prices, is_call=opt_is_calls, sqrtT=bb.sqrt(opt_tte) )
        return prices, implieds

    def price(self, opt_strikes  : Backbone.Array,
                    opt_tte      : Backbone.Array, *,
                    opt_is_calls : Backbone.Array | None, 
                    opt_mask     : Backbone.Array | None,
                    opt_lix      : Backbone.Array, 
                    opt_dweight  : Backbone.Array, 
                    opt_vweight  : Backbone.Array,
                    opt_vextra   : Backbone.Array ) -> Backbone.Array:
        """
        Price with pre-computed weighting scheme
        """
        prices   = self.bb_calc.prices( fwd_vols=self.fwd_vols,#
                                        dlvsqs=self.dlvsqs, 
                                        opt_strikes=opt_strikes, 
                                        opt_tte=opt_tte, 
                                        opt_is_calls=opt_is_calls, 
                                        opt_mask=opt_mask,
                                        opt_lix=opt_lix,
                                        opt_dweight=opt_dweight,
                                        opt_vweight=opt_vweight,
                                        opt_vextra=opt_vextra )
        self.verify( lambda : self.bb.all( self.bb.isfinite(prices) ), lambda : f"Computed prices must be finite")
        return prices
